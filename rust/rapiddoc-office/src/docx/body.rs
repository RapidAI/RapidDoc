//! DocVortex DOCX body orchestration: the original paragraph/list/index/section
//! rules, sharing native table/formula resources. Not the public Office API.
use super::{
    embedded::Shared,
    formulas::{self, Token, Tokens},
    spans::{self, Span},
    *,
};
use num_bigint::BigInt;
use serde::{Serialize, Serializer};
use std::{collections::BTreeSet, sync::OnceLock};

#[derive(Debug, Clone)]
pub struct Integer(pub BigInt);
impl Serialize for Integer {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        serde_json::value::RawValue::from_string(self.0.to_string())
            .map_err(serde::ser::Error::custom)?
            .serialize(s)
    }
}
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Text,
    DocTitle,
    ParagraphTitle,
    Caption,
    Code,
    Textbox,
    Equation,
    List,
    Index,
    Table,
    Header,
    Footer,
    PageFootnote,
    Image,
    Chart,
}
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Content {
    Spans(Vec<Span>),
    Blocks(Vec<Block>),
    Text(String),
    None,
}
#[derive(Debug, Serialize)]
pub struct Block {
    #[serde(rename = "type")]
    pub kind: Kind,
    #[serde(skip_serializing_if = "content_none")]
    pub content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_base64: Option<Arc<str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_numbered_style: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ilevel: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribute: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<Integer>,
}
impl Block {
    pub(crate) fn new(kind: Kind, content: Content) -> Self {
        Self {
            kind,
            content,
            image_base64: None,
            level: None,
            is_numbered_style: None,
            anchor: None,
            list_label: None,
            ilevel: None,
            attribute: None,
            start: None,
        }
    }
}
fn content_none(content: &Content) -> bool {
    matches!(content, Content::None)
}
#[derive(Debug, Serialize)]
pub struct Body {
    pub pages: Vec<Vec<Block>>,
    pub diagnostics: Vec<String>,
    pub normalization: normalize::Report,
    pub vector_reports: Vec<table_images::VectorReport>,
    pub image_fallbacks: Vec<table_images::ImageFallback>,
}
#[derive(Clone)]
struct Path {
    page: usize,
    indices: Vec<usize>,
}
struct Assemble<'c, 'd> {
    context: &'c Context<'d>,
    pub output: Body,
    numbering: Numbering,
    list: Vec<Path>,
    index: Vec<Path>,
    pre_num: i64,
    pre_level: i64,
    pre_index: i64,
    plain_base: Option<i64>,
    toc_anchors: BTreeSet<String>,
    aliases: BTreeMap<String, String>,
    heading_ids: BTreeSet<i64>,
}
fn bookmarks(p: Node<'_, '_>) -> Vec<String> {
    p.descendants()
        .filter(|n| is_word(*n, "bookmarkStart"))
        .filter_map(|n| attr(n, "name"))
        .map(|s| s.trim_matches(space))
        .filter(|s| !s.is_empty() && !s.starts_with("_GoBack"))
        .map(str::to_owned)
        .collect()
}
fn paragraph_text_direct(p: Node<'_, '_>) -> String {
    let mut s = String::new();
    for n in p.children() {
        if is_word(n, "r") {
            s.push_str(&fields::run_text(n));
        } else if is_word(n, "hyperlink") {
            for r in n.children().filter(|n| is_word(*n, "r")) {
                s.push_str(&fields::run_text(r));
            }
        }
    }
    s
}
fn targets(p: Node<'_, '_>) -> Vec<String> {
    let mut out = p
        .descendants()
        .filter(|n| is_word(*n, "hyperlink"))
        .filter_map(|n| attr(n, "anchor"))
        .map(|a| a.trim_matches(space))
        .filter(|a| !a.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if out.is_empty() {
        for instruction in fields::instructions(p) {
            if let (Some(s), true) = fields::hyperlink_instruction(&instruction) {
                if let Some(s) = s.strip_prefix('#').filter(|s| !s.is_empty()) {
                    out.push(s.into());
                }
            }
        }
    }
    out
}
fn corrected(level: i64, text: &str) -> i64 {
    if level == 0 {
        return 0;
    }
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r"^\d+(?:\.\d+)+").unwrap());
    if let Some(m) = re.find(text.trim_matches(space)) {
        let tail = &text.trim_matches(space)[m.end()..];
        if tail
            .chars()
            .next()
            .is_none_or(|c| c != '.' && super::decimal(c).is_none())
        {
            return level.min(m.as_str().split('.').count() as i64 - 1);
        }
    }
    level
}
fn drawing(node: Node<'_, '_>, name: &str) -> bool {
    node.is_element()
        && node.tag_name().name() == name
        && crate::package::namespace_family(node.tag_name().namespace(), "drawingml/main")
}
fn textbox_paragraphs<'a, 'd>(node: Node<'a, 'd>) -> (bool, Vec<Node<'a, 'd>>) {
    // XPath union returns document order. Containers preserve first encounter
    // order; paragraph position normally is its index among sibling w:p.
    let mut elements = node
        .descendants()
        .skip(1)
        .filter(|n| {
            is_word(*n, "txbxContent")
                || is_word(*n, "p")
                    && n.ancestors().skip(1).any(|a| {
                        a.tag_name().namespace() == Some("urn:schemas-microsoft-com:vml")
                            && a.tag_name().name() == "textbox"
                    })
        })
        .collect::<Vec<_>>();
    if elements.is_empty() && matches!(node.tag_name().name(), "drawing" | "pict") {
        elements = node
            .descendants()
            .skip(1)
            .filter(|n| {
                is_word(*n, "p")
                    && n.ancestors().skip(1).any(|a| {
                        a.tag_name().name() == "txbx"
                        && a.tag_name().namespace()
                            == Some(
                                "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
                            )
                        || a.tag_name().name() == "wrap"
                            && a.tag_name().namespace()
                                == Some("urn:schemas-microsoft-com:office:word")
                    })
                    || drawing(*n, "t") && n.ancestors().skip(1).any(|a| drawing(a, "p"))
            })
            .collect();
    }
    let found = !elements.is_empty();
    let mut processed = std::collections::HashSet::new();
    let mut groups = Vec::<(Option<roxmltree::NodeId>, Vec<Node<'a, 'd>>)>::new();
    for element in elements {
        if !processed.insert(element.id()) {
            continue;
        }
        let direct = is_word(element, "p");
        let container = if direct {
            element
                .ancestors()
                .skip(1)
                .find(|a| {
                    a.is_element()
                        && ["textbox", "shape", "txbx"]
                            .iter()
                            .any(|s| a.tag_name().name().contains(s))
                })
                .map(|n| n.id())
        } else {
            Some(element.id())
        };
        let index = groups
            .iter()
            .position(|(id, _)| *id == container)
            .unwrap_or_else(|| {
                groups.push((container, vec![]));
                groups.len() - 1
            });
        if direct {
            groups[index].1.push(element);
        } else {
            for p in element.descendants().skip(1).filter(|n| is_word(*n, "p")) {
                if processed.insert(p.id()) {
                    groups[index].1.push(p);
                }
            }
        }
    }
    let position = |p: Node<'a, 'd>| {
        p.parent().and_then(|parent| {
            parent
                .children()
                .filter(|n| n.is_element() && n.tag_name().name() == "p")
                .position(|n| n == p)
        })
    };
    let mut out = vec![];
    let mut seen = BTreeSet::new();
    for (_, mut group) in groups {
        group.sort_by_key(|p| position(*p));
        for p in group {
            let text = fields::plain_text(&fields::contents(p));
            if seen.insert((text, position(p))) {
                out.push(p);
            }
        }
    }
    (found, out)
}
impl<'c, 'd> Assemble<'c, 'd> {
    fn new(context: &'c Context<'d>, body: Node<'_, 'd>) -> Result<Self> {
        let mut anchors = BTreeSet::new();
        for n in body.descendants().filter(|n| is_word(*n, "hyperlink")) {
            if let Some(a) = attr(n, "anchor")
                .map(|s| s.trim_matches(space))
                .filter(|a| a.starts_with("_Toc"))
            {
                anchors.insert(a.into());
            }
        }
        for p in body.descendants().filter(|n| is_word(*n, "p")) {
            for instruction in fields::instructions(p) {
                if let (Some(target), true) = fields::hyperlink_instruction(&instruction) {
                    if let Some(a) = target.strip_prefix('#').filter(|a| a.starts_with("_Toc")) {
                        anchors.insert(a.into());
                    }
                }
            }
        }
        let mut aliases = BTreeMap::new();
        for p in body.descendants().filter(|n| is_word(*n, "p")) {
            let names = bookmarks(p)
                .into_iter()
                .filter(|s| s.starts_with("_Toc"))
                .collect::<Vec<_>>();
            if let Some(canonical) = names
                .iter()
                .find(|s| anchors.contains(*s))
                .or(names.first())
            {
                for name in &names {
                    aliases
                        .entry(name.clone())
                        .or_insert_with(|| canonical.clone());
                }
            }
        }
        let mut seen = BTreeMap::<i64, bool>::new();
        let mut interrupted = BTreeSet::new();
        let mut levels = BTreeMap::<i64, BTreeSet<i64>>::new();
        for n in body.children().filter(|n| n.is_element()) {
            let mut content = is_word(n, "tbl");
            if is_word(n, "p") {
                if let Ok(f) = context.paragraph_facts(n) {
                    if !matches!(f.label.as_str(), "Title" | "Heading")
                        && !f.text.trim_matches(space).is_empty()
                    {
                        if let Some((id, level)) = f.num_id.filter(|id| *id != 0).zip(f.level) {
                            if seen.get(&id) == Some(&true) {
                                interrupted.insert(id);
                            }
                            seen.insert(id, false);
                            levels.entry(id).or_default().insert(level);
                        } else {
                            content = true;
                        }
                    }
                }
            }
            if content {
                for v in seen.values_mut() {
                    *v = true;
                }
            }
        }
        interrupted.retain(|id| levels.get(id).is_some_and(|levels| levels.len() > 1));
        Ok(Self {
            context,
            output: Body {
                pages: vec![vec![]],
                diagnostics: vec![],
                normalization: normalize::Report::default(),
                vector_reports: vec![],
                image_fallbacks: vec![],
            },
            numbering: context.numbering.clone(),
            list: vec![],
            index: vec![],
            pre_num: -1,
            pre_level: -1,
            pre_index: -1,
            plain_base: None,
            toc_anchors: anchors,
            aliases,
            heading_ids: interrupted,
        })
    }
    fn push(&mut self, block: Block) -> Path {
        let page = self.output.pages.len() - 1;
        let i = self.output.pages[page].len();
        self.output.pages[page].push(block);
        Path {
            page,
            indices: vec![i],
        }
    }
    fn at(&mut self, path: &Path) -> &mut Block {
        let mut b = &mut self.output.pages[path.page][path.indices[0]];
        for &i in &path.indices[1..] {
            let Content::Blocks(children) = &mut b.content else {
                unreachable!()
            };
            b = &mut children[i];
        }
        b
    }
    fn child(&mut self, parent: Option<&Path>, block: Block) -> Path {
        if let Some(parent) = parent {
            let Content::Blocks(children) = &mut self.at(parent).content else {
                unreachable!()
            };
            let mut indices = parent.indices.clone();
            indices.push(children.len());
            children.push(block);
            Path {
                page: parent.page,
                indices,
            }
        } else {
            self.push(block)
        }
    }
    fn canonical(&self, a: &str) -> String {
        self.aliases.get(a).cloned().unwrap_or_else(|| a.into())
    }
    fn bookmark(&self, p: Node<'_, '_>) -> Option<String> {
        let names = bookmarks(p);
        let toc = names
            .iter()
            .filter(|s| s.starts_with("_Toc"))
            .collect::<Vec<_>>();
        if !toc.is_empty() {
            Some(
                self.canonical(
                    toc.iter()
                        .find(|s| self.toc_anchors.contains(**s))
                        .unwrap_or(&toc[0]),
                ),
            )
        } else {
            names.into_iter().next()
        }
    }
    fn target(&self, p: Node<'_, '_>) -> Option<String> {
        let targets = targets(p);
        targets
            .iter()
            .find(|s| s.starts_with("_Toc"))
            .or(targets.first())
            .map(|s| {
                if s.starts_with("_Toc") {
                    self.canonical(s)
                } else {
                    s.clone()
                }
            })
    }
    fn close_list(&mut self) {
        self.pre_num = -1;
        self.pre_level = -1;
        self.list.clear();
    }
    fn reset_index(&mut self) {
        self.index.clear();
        self.pre_index = -1;
        self.plain_base = None;
    }
    fn item(
        &mut self,
        id: i64,
        level: i64,
        numbered: bool,
        elements: bool,
        content: Vec<Span>,
    ) -> Result<()> {
        let content = spans::strip(content);
        if !elements || content.is_empty() {
            return Ok(());
        }
        let start = if numbered {
            self.numbering.advance(id, level)?.map(Integer)
        } else {
            None
        };
        let make = || {
            let mut b = Block::new(Kind::List, Content::Blocks(vec![]));
            b.ilevel = Some(level);
            b.attribute = Some(if numbered { "ordered" } else { "unordered" });
            b.start = start.clone();
            b
        };
        let path = if self.pre_num == -1 || self.pre_num != id {
            self.close_list();
            let p = self.push(make());
            self.list.push(p.clone());
            self.pre_num = id;
            self.pre_level = level;
            p
        } else if self.pre_level != -1 && self.pre_level < level {
            let parent = self.list.last().cloned();
            let p = self.child(parent.as_ref(), make());
            self.list.push(p.clone());
            self.pre_level = level;
            p
        } else if self.pre_level != -1 && level < self.pre_level {
            while let Some(top) = self.list.last().cloned() {
                if self.at(&top).ilevel == Some(level) {
                    break;
                }
                self.list.pop();
            }
            let p = if let Some(p) = self.list.last() {
                p.clone()
            } else {
                let p = self.push(make());
                self.list.push(p.clone());
                p
            };
            self.pre_level = level;
            p
        } else if self.pre_level == level {
            if let Some(p) = self.list.last() {
                p.clone()
            } else {
                let p = self.push(make());
                self.list.push(p.clone());
                p
            }
        } else {
            self.output
                .diagnostics
                .push("unexpected DOCX list state".into());
            return Ok(());
        };
        self.child(Some(&path), Block::new(Kind::Text, Content::Spans(content)));
        Ok(())
    }
    fn index_item(
        &mut self,
        level: i64,
        elements: bool,
        content: Vec<Span>,
        anchor: Option<String>,
    ) {
        let content = spans::strip(content);
        if !elements || content.is_empty() {
            return;
        }
        let make = || {
            let mut b = Block::new(Kind::Index, Content::Blocks(vec![]));
            b.ilevel = Some(level);
            b
        };
        let path = if self.pre_index == -1 {
            let p = self.push(make());
            self.index.push(p.clone());
            self.pre_index = level;
            Some(p)
        } else if self.pre_index < level {
            if self.index.is_empty() {
                self.pre_index = -1;
                self.index_item(level, elements, content, anchor);
                return;
            }
            let parent = self.index.last().cloned();
            let p = self.child(parent.as_ref(), make());
            self.index.push(p.clone());
            self.pre_index = level;
            Some(p)
        } else if level < self.pre_index {
            while let Some(p) = self.index.last().cloned() {
                if self.at(&p).ilevel == Some(level) {
                    break;
                }
                self.index.pop();
            }
            self.pre_index = level;
            self.index.last().cloned()
        } else {
            self.index.last().cloned()
        };
        // Preserve the source behavior for a missing lower TOC parent.
        if let Some(path) = path {
            let mut item = Block::new(Kind::Text, Content::Spans(content));
            item.anchor = anchor;
            self.child(Some(&path), item);
        }
    }
    fn layout_section(&self, p: Node<'_, '_>, f: &ParagraphFacts) -> bool {
        let Some(sect) = child(p, "pPr").and_then(|n| child(n, "sectPr")) else {
            return false;
        };
        if !f.text.trim_matches(space).is_empty()
            || p.descendants()
                .any(|n| matches!(n.tag_name().name(), "blip" | "imagedata"))
        {
            return false;
        }
        if child(sect, "type")
            .and_then(|n| attr(n, "val"))
            .unwrap_or("continuous")
            != "continuous"
        {
            return false;
        }
        let Some(margin) = child(sect, "pgMar") else {
            return false;
        };
        ["header", "footer", "top", "bottom", "left", "right"]
            .into_iter()
            .all(|key| attr(margin, key).unwrap_or("0") == "0")
    }
    fn paragraph(
        &mut self,
        p: Node<'_, 'd>,
        tokens: &mut Tokens<impl formulas::Embedded>,
    ) -> Result<()> {
        let f = self.context.paragraph_facts(p)?;
        let section = p.descendants().any(|n| is_word(n, "sectPr")) && !self.layout_section(p, &f);
        let end = section && p.text() != Some("");
        if section && !end {
            self.output.pages.push(vec![]);
        }
        let formula_tokens = tokens.extract(p)?;
        let (content, has_formulas) = spans::with_formulas(&f.elements, &f.text, &formula_tokens);
        let raw = f.text.trim_matches(space);
        let elements = !f.elements.is_empty();
        let anchor = self.bookmark(p);
        if let Some(toc) = f.toc_level {
            let target = self.target(p);
            if raw.is_empty() || target.is_some() || self.pre_index != -1 {
                if !raw.is_empty() {
                    if let Some(a) = target.as_ref().filter(|s| s.starts_with("_Toc")) {
                        self.toc_anchors.insert(a.clone());
                    }
                    let base = *self.plain_base.get_or_insert(toc);
                    self.index_item(
                        corrected((toc - base).max(0), raw),
                        elements,
                        content,
                        target,
                    );
                }
                if self.pre_num != -1 {
                    self.close_list();
                }
                if end {
                    self.output.pages.push(vec![]);
                }
                return Ok(());
            }
        }
        self.reset_index();
        let id = f.num_id.filter(|id| *id != 0);
        if let Some((id, level)) = id
            .zip(f.level)
            .filter(|_| !matches!(f.label.as_str(), "Title" | "Heading"))
        {
            if self.heading_ids.contains(&id) {
                if self.pre_num != -1 {
                    self.close_list();
                }
                if !content.is_empty() {
                    let mut b = Block::new(Kind::ParagraphTitle, Content::Spans(content));
                    b.level = Some(level + 2);
                    b.is_numbered_style = Some(f.numbered);
                    b.anchor = anchor;
                    self.push(b);
                }
            } else {
                self.item(id, level, f.numbered, elements, content)?;
            }
            return Ok(());
        } else if id.is_none()
            && self.pre_num != -1
            && !matches!(f.label.as_str(), "Title" | "Heading")
        {
            self.close_list();
        }
        let mut block = if f.label == "Title" {
            let mut b = Block::new(Kind::DocTitle, Content::Spans(content));
            b.level = Some(1);
            b
        } else if f.label.contains("Heading") {
            let mut b = Block::new(Kind::ParagraphTitle, Content::Spans(content));
            b.level = Some(f.heading_level.map_or(2, |v| (v + 1).max(2)));
            b.is_numbered_style = Some(id.zip(f.level).is_some() && f.numbered);
            b
        } else if has_formulas && raw.is_empty() {
            let values = formula_tokens
                .iter()
                .filter_map(|t| {
                    if let Token::Formula(_, s) = t {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            Block::new(Kind::Equation, Content::Text(values.join("\n")))
        } else {
            let known = matches!(
                f.label.as_str(),
                "Paragraph"
                    | "Normal"
                    | "Subtitle"
                    | "Author"
                    | "DefaultText"
                    | "ListParagraph"
                    | "ListBullet"
                    | "Quote"
            );
            let caption = !has_formulas
                && !known
                && p.descendants()
                    .filter(|n| is_word(*n, "instrText"))
                    .any(|n| n.text().is_some_and(|s| s.contains("SEQ")));
            Block::new(
                if caption { Kind::Caption } else { Kind::Text },
                Content::Spans(if caption {
                    content
                } else {
                    spans::strip(content)
                }),
            )
        };
        if !matches!(block.kind, Kind::Equation) {
            block.anchor = anchor;
        }
        if match &block.content {
            Content::Spans(s) => !s.is_empty(),
            Content::Text(s) => !s.is_empty(),
            _ => true,
        } {
            self.push(block);
        }
        if end {
            self.output.pages.push(vec![]);
        }
        Ok(())
    }
    fn toc_sdt(&self, node: Node<'_, 'd>) -> bool {
        if let Some(pr) = child(node, "sdtPr") {
            if pr
                .descendants()
                .find(|n| is_word(*n, "docPartGallery"))
                .and_then(|n| attr(n, "val"))
                .is_some_and(|v| {
                    v.contains("Table of Contents") || v.to_lowercase().contains("toc")
                })
            {
                return true;
            }
            if child(pr, "tag")
                .and_then(|n| attr(n, "val"))
                .is_some_and(|s| {
                    let s = s.to_lowercase().replace(' ', "");
                    s.contains("toc") || s.contains("contents")
                })
            {
                return true;
            }
        }
        child(node, "sdtContent").is_some_and(|c| {
            c.children().filter(|n| is_word(*n, "p")).take(5).any(|p| {
                self.context
                    .paragraph_facts(p)
                    .is_ok_and(|f| f.toc_level.is_some())
            })
        })
    }
    fn sdt_index(
        &mut self,
        content: Node<'_, 'd>,
        tokens: &mut Tokens<impl formulas::Embedded>,
    ) -> Result<()> {
        let mut items = vec![];
        for p in content.descendants().skip(1).filter(|n| is_word(*n, "p")) {
            let Ok(f) = self.context.paragraph_facts(p) else {
                continue;
            };
            let raw = paragraph_text_direct(p);
            if raw.trim_matches(space).is_empty() {
                continue;
            }
            let target = self.target(p);
            if let Some(a) = target.as_ref().filter(|s| s.starts_with("_Toc")) {
                self.toc_anchors.insert(a.clone());
            }
            let ts = tokens.extract(p)?;
            let (spans, _) = spans::with_formulas(&f.elements, &raw, &ts);
            items.push((
                f.toc_level.unwrap_or(0),
                raw,
                !f.elements.is_empty(),
                spans,
                target,
            ));
        }
        static FIG: OnceLock<regex::Regex> = OnceLock::new();
        let re = FIG.get_or_init(|| {
            regex::Regex::new(r"^(?:[图表][\d\s.]|(?i:Figure|Table)\s+\d)").unwrap()
        });
        let flat = items
            .iter()
            .filter(|(_, s, _, _, _)| re.is_match(s.trim_matches(space)))
            .count()
            * 2
            > items.len();
        self.reset_index();
        for (level, text, elements, spans, target) in items {
            self.index_item(
                if flat { 0 } else { corrected(level, &text) },
                elements,
                spans,
                target,
            );
        }
        self.reset_index();
        Ok(())
    }
}
/// Body without section headers/footers. This is not the complete public
/// DOCX converter: unified export and public Office adapters are separate.
pub fn convert_body(source: Source) -> Result<Body> {
    convert_impl(source, false)
}
pub fn convert(source: Source) -> Result<Body> {
    convert_impl(source, true)
}
fn convert_impl(source: Source, margins: bool) -> Result<Body> {
    let Source {
        document,
        styles,
        numbering,
        relationships,
        mut package,
        normalization,
        ..
    } = source;
    let context = Context::from_parts(
        &document,
        styles.as_deref(),
        numbering.as_deref(),
        &relationships,
    )?;
    let body = context
        .document
        .root_element()
        .children()
        .find(|n| is_word(*n, "body"))
        .ok_or_else(|| Error::missing("w:body"))?;
    let shared = Shared::new(&mut package, &relationships)?;
    let mut tokens = Tokens::new(shared.embedded());
    let mut a = Assemble::new(&context, body)?;
    a.output.normalization = normalization;
    let mut tables = super::table_resources::Tables::prepare(
        &context,
        body,
        styles.as_deref(),
        numbering.as_deref(),
        &relationships,
        &shared,
        &mut tokens,
        &mut a.output.diagnostics,
    )
    .unwrap_or_else(|error| {
        a.output
            .diagnostics
            .push(format!("DOCX table resource setup failed: {error}"));
        super::table_resources::Tables::failed(error)
    });
    let mut body_images = BTreeMap::<String, Option<Arc<str>>>::new();
    let mut chart_cache = BTreeMap::<String, String>::new();
    for n in body.children().filter(|n| n.is_element()) {
        // Original _handle_drawingml runs before textbox/table/picture/text
        // handling, and preserves an empty chart block on relation failures.
        for drawing in n.descendants().skip(1).filter(|n| is_word(*n, "drawing")) {
            let Some(chart) = drawing.descendants().skip(1).find(|n| {
                n.tag_name().name() == "chart"
                    && crate::package::namespace_family(n.tag_name().namespace(), "drawingml/chart")
            }) else {
                continue;
            };
            let mut block = Block::new(Kind::Chart, Content::Text(String::new()));
            if let Some(rel) = relationship_id(chart, "id").and_then(|id| {
                relationships
                    .iter()
                    .find(|r| r.id == id && !r.external && office_relation(&r.kind, "chart"))
            }) {
                if let Some(path) = rel.resolved.as_deref() {
                    let html = if let Some(html) = chart_cache.get(path) {
                        html.clone()
                    } else {
                        let result = shared.read_part(path).and_then(|data| {
                            let data = data.ok_or_else(|| Error::missing(path))?;
                            let wb = shared
                                .part_relationships(path)
                                .ok()
                                .and_then(|rels| {
                                    rels.into_iter().find(|r| {
                                        !r.external && office_relation(&r.kind, "package")
                                    })
                                })
                                .and_then(|r| r.resolved)
                                .and_then(|p| shared.read_part(&p).ok().flatten());
                            crate::chart::extract(&data, wb)
                        });
                        let html = match result {
                            Ok(html) => html,
                            Err(e) => {
                                a.output
                                    .diagnostics
                                    .push(format!("chart XML cannot be loaded: {path}: {e}"));
                                String::new()
                            }
                        };
                        chart_cache.insert(path.into(), html.clone());
                        html
                    };
                    block.content = Content::Text(html);
                }
            }
            a.push(block);
        }
        let (textboxes, paragraphs) = textbox_paragraphs(n);
        if !textboxes && matches!(n.tag_name().name(), "drawing" | "pict") {
            // Original XPath's ancestor axis for bodyPr can reach the complete
            // document. Preserve its selection rather than invent a new order.
            let root = n.document().root_element();
            let scope = if n.descendants().skip(1).any(|n| drawing(n, "bodyPr")) {
                root
            } else {
                n
            };
            let texts = scope
                .descendants()
                .skip(1)
                .filter(|t| {
                    drawing(*t, "t")
                        && (scope == root || t.ancestors().skip(1).any(|a| drawing(a, "txBody")))
                })
                .filter_map(|t| t.text())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>();
            let content = spans::strip(vec![Span::Text {
                content: texts.join(" "),
                styles: vec![],
            }]);
            if !content.is_empty() {
                a.push(Block::new(Kind::Text, Content::Spans(content)));
            }
        }
        for paragraph in paragraphs {
            a.paragraph(paragraph, &mut tokens)?;
        }
        if is_word(n, "tbl") {
            if a.pre_num != -1 {
                a.close_list();
            }
            let converted = tables.convert(
                &context,
                n,
                styles.as_deref(),
                numbering.as_deref(),
                &relationships,
                &shared,
                &mut tokens,
            );
            match converted {
                Ok(html) => {
                    a.push(Block::new(Kind::Table, Content::Text(html)));
                }
                Err(error) => a.output.diagnostics.push(format!(
                    "Could not parse DOCX table at XML byte {}: {error}",
                    n.range().start
                )),
            }
        } else if n.descendants().skip(1).any(picture) {
            let anchored = is_word(n, "p")
                && n.descendants().any(|n| {
                    n.tag_name().name() == "anchor"
                        && crate::package::namespace_family(
                            n.tag_name().namespace(),
                            "drawingml/wordprocessingDrawing",
                        )
                });
            if anchored {
                a.paragraph(n, &mut tokens)?;
            }
            let mut seen = BTreeSet::new();
            for image in n.descendants().skip(1).filter(|n| picture(*n)) {
                if shared.is_equation_preview(image)? {
                    continue;
                }
                let id = relationship_id(image, "embed")
                    .filter(|s| !s.is_empty())
                    .or_else(|| relationship_id(image, "id"))
                    .filter(|s| !s.is_empty());
                if let Some(id) = id {
                    if !seen.insert(id) {
                        continue;
                    }
                }
                let Some(id) = id else {
                    a.output.diagnostics.push("image cannot be found".into());
                    continue;
                };
                let Some((path, mime, bytes)) = shared.image_payload(id)? else {
                    a.output
                        .diagnostics
                        .push(format!("image cannot be found: {id}"));
                    continue;
                };
                let source = if let Some(source) = body_images.get(&path) {
                    source.clone()
                } else {
                    let source = shared.serialize_body_image(&path, mime.as_deref(), &bytes)?;
                    if source.is_none() {
                        a.output
                            .diagnostics
                            .push(format!("image cannot be decoded: {path}"));
                    }
                    body_images.insert(path, source.clone());
                    source
                };
                if let Some(source) = source {
                    let mut block = Block::new(Kind::Image, Content::None);
                    let alt = picture_alt(image);
                    if !alt.is_empty() {
                        block.content = Content::Text(alt);
                    }
                    block.image_base64 = Some(source);
                    a.push(block);
                }
            }
            if is_word(n, "p") && !anchored {
                a.paragraph(n, &mut tokens)?;
            }
        } else if is_word(n, "p") || is_word(n, "sdt") {
            if is_word(n, "p") {
                a.paragraph(n, &mut tokens)?;
            } else if let Some(content) = n.descendants().find(|n| is_word(*n, "sdtContent")) {
                if a.toc_sdt(n) {
                    a.sdt_index(content, &mut tokens)?;
                } else {
                    for p in content.descendants().skip(1).filter(|n| is_word(*n, "p")) {
                        a.paragraph(p, &mut tokens)?;
                    }
                }
            }
        }
    }
    a.output.diagnostics.extend(tokens.warnings);
    if margins {
        add_margins(
            &mut a.output,
            body,
            &shared,
            &relationships,
            styles.as_deref(),
            numbering.as_deref(),
        )?;
    }
    a.output.diagnostics.extend(shared.warnings());
    a.output.vector_reports = shared.vector_reports();
    a.output.image_fallbacks = shared.image_fallbacks();
    Ok(a.output)
}
fn picture(n: Node<'_, '_>) -> bool {
    n.is_element()
        && (n.tag_name().name() == "blip"
            && crate::package::namespace_family(n.tag_name().namespace(), "drawingml/main")
            || n.tag_name().name() == "imagedata"
                && n.tag_name().namespace() == Some("urn:schemas-microsoft-com:vml"))
}
fn office_relation(kind: &str, tail: &str) -> bool {
    kind == format!("http://schemas.openxmlformats.org/officeDocument/2006/relationships/{tail}")
        || kind == format!("http://purl.oclc.org/ooxml/officeDocument/relationships/{tail}")
}
fn relationship_id<'a>(n: Node<'a, '_>, name: &str) -> Option<&'a str> {
    n.attributes()
        .find(|a| {
            a.name() == name
                && a.namespace()
                    .is_some_and(|ns| ns.ends_with("/relationships"))
        })
        .map(|a| a.value())
}
fn picture_alt(n: Node<'_, '_>) -> String {
    for n in n.ancestors().filter(|n| n.is_element()) {
        if matches!(n.tag_name().name(), "inline" | "anchor") {
            return n
                .children()
                .find(|n| n.is_element() && n.tag_name().name() == "docPr")
                .and_then(|n| n.attribute("descr"))
                .unwrap_or("")
                .trim_matches(space)
                .into();
        }
        if n.tag_name().name() == "shape"
            && n.tag_name().namespace() == Some("urn:schemas-microsoft-com:vml")
        {
            return n.attribute("alt").unwrap_or("").trim_matches(space).into();
        }
    }
    String::new()
}
fn on_off(node: Option<Node<'_, '_>>) -> Result<bool> {
    let Some(n) = node else {
        return Ok(false);
    };
    match attr(n, "val").unwrap_or("true") {
        "true" | "on" | "1" => Ok(true),
        "false" | "off" | "0" => Ok(false),
        _ => Err(Error::malformed("invalid Word on/off value")),
    }
}
fn is_digit(text: &str) -> bool {
    !text.is_empty()
        && text.chars().all(|c| {
            super::decimal(c).is_some()
                || super::decimal_data::ISDIGIT_EXTRA
                    .iter()
                    .any(|&(lo, hi)| (lo..=hi).contains(&(c as u32)))
        })
}
fn add_margins(
    output: &mut Body,
    body: Node<'_, '_>,
    shared: &Shared<'_>,
    main_relationships: &[Relationship],
    styles: Option<&str>,
    numbering: Option<&str>,
) -> Result<()> {
    let settings = main_relationships
        .iter()
        .find(|r| !r.external && r.kind_tail() == "settings")
        .and_then(|r| r.resolved.as_deref());
    let odd_even = if let Some(path) = settings {
        if let Some(xml) = shared.read_xml(path)? {
            let tree = crate::xml::parse(&xml)?;
            on_off(child(tree.root_element(), "evenAndOddHeaders"))?
        } else {
            false
        }
    } else {
        false
    };
    let sections = body
        .children()
        .filter_map(|n| {
            if is_word(n, "sectPr") {
                Some(n)
            } else if is_word(n, "p") {
                child(n, "pPr").and_then(|n| child(n, "sectPr"))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut cached = BTreeMap::<String, Vec<Span>>::new();
    for (section, sect) in sections.iter().enumerate() {
        let first = on_off(child(*sect, "titlePg"))?;
        let mut variants = vec!["default"];
        if odd_even {
            variants.push("even");
        }
        if first {
            variants.push("first");
        }
        for (reference, kind) in [
            ("headerReference", Kind::Header),
            ("footerReference", Kind::Footer),
        ] {
            let mut seen = BTreeSet::new();
            for variant in &variants {
                let reference = sections[..=section].iter().rev().find_map(|s| {
                    s.children()
                        .find(|n| is_word(*n, reference) && attr(*n, "type") == Some(*variant))
                });
                let Some(reference) = reference else {
                    continue;
                };
                let id = reference
                    .attributes()
                    .find(|a| {
                        a.name() == "id"
                            && a.namespace().is_some_and(|s| s.ends_with("/relationships"))
                    })
                    .map(|a| a.value())
                    .ok_or_else(|| Error::missing("header/footer relationship ID"))?;
                let rel = main_relationships
                    .iter()
                    .find(|r| r.id == id && !r.external)
                    .ok_or_else(|| Error::missing(id))?;
                let path = rel.resolved.as_deref().ok_or_else(|| Error::missing(id))?;
                let spans = if let Some(spans) = cached.get(path) {
                    spans.clone()
                } else {
                    let text = shared.read_xml(path)?.ok_or_else(|| Error::missing(path))?;
                    let rels = shared.part_relationships(path)?;
                    shared.switch_relationships(&rels);
                    let context = Context::from_parts(&text, styles, numbering, &rels)?;
                    let mut tokens = Tokens::new(shared.embedded());
                    let mut result = vec![];
                    for p in context
                        .document
                        .root_element()
                        .children()
                        .filter(|n| is_word(*n, "p"))
                    {
                        let contents = fields::contents(p);
                        let raw = fields::plain_text(&contents);
                        let elements = fields::elements(
                            &contents,
                            p,
                            &context.styles,
                            &context.relationships,
                            context.styles.toc_level(p)?.is_some(),
                        )?;
                        let ts = tokens.extract(p)?;
                        let (spans, has) = spans::with_formulas(&elements, &raw, &ts);
                        if raw.trim_matches(space).is_empty() && !has {
                            continue;
                        }
                        if !spans.is_empty() {
                            if !result.is_empty() {
                                spans::text(&mut result, " ".into());
                            }
                            for s in spans {
                                spans::append(&mut result, s);
                            }
                        }
                    }
                    output.diagnostics.extend(tokens.warnings);
                    cached.insert(path.into(), result.clone());
                    result
                };
                let visible = spans::visible(&spans);
                if !spans.is_empty() && !is_digit(&visible) && seen.insert(visible) {
                    if let Some(page) = output.pages.get_mut(section) {
                        page.push(Block::new(kind, Content::Spans(spans)));
                    } else {
                        output
                            .diagnostics
                            .push("section index out of range when adding header/footer".into());
                    }
                }
            }
        }
    }
    shared.switch_relationships(main_relationships);
    Ok(())
}
