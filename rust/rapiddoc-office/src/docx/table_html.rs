//! Mammoth's table HTML path. Kept separate from Word paragraph formatting:
//! Mammoth uses direct run properties and HTML path collapsing, not inherited
//! python-docx formatting. Pending resource handlers must fail, never erase data.
use super::*;
use crate::ErrorKind;
use std::collections::{BTreeMap, HashSet};

use super::table_markup::{collapse, strip_empty, write};
pub use super::table_markup::{normalize_colspans, serialize, Html};

/// The caller supplies image/equation/note handlers once those paths are ready.
/// Ordinary Word run and table semantics never pass through a Python callback.
pub trait Resources {
    fn render(&mut self, node: Node<'_, '_>) -> Result<Vec<Html>>;
}
pub struct UnavailableResources;
impl Resources for UnavailableResources {
    fn render(&mut self, node: Node<'_, '_>) -> Result<Vec<Html>> {
        Err(Error::new(
            ErrorKind::Unsupported,
            format!(
                "DOCX resource renderer not connected: {}",
                node.tag_name().name()
            ),
        ))
    }
}

#[derive(Default)]
struct Read {
    body: Vec<Html>,
    extra: Vec<Html>,
}
impl Read {
    fn nodes(body: Vec<Html>) -> Self {
        Self {
            body,
            extra: Vec::new(),
        }
    }
    fn one(node: Html) -> Self {
        Self::nodes(vec![node])
    }
}
#[derive(Clone)]
struct Field {
    beginning: bool,
    node_checked: bool,
    href: Option<String>,
    checkbox: Option<bool>,
}
#[derive(Clone, Debug)]
struct ListLevel {
    index: String,
    ordered: bool,
}
struct Abstract {
    levels: BTreeMap<String, ListLevel>,
    link: Option<String>,
}

/// One reader per document/selected table batch, mirroring Mammoth's field and
/// deleted-paragraph state. Input trees stay borrowed; no JSON intermediate.
pub struct Reader<R> {
    style_map: super::style_map::Rules,
    pub(super) note_references: Vec<(String, String)>,
    pub(super) comments: Arc<BTreeMap<String, String>>,
    comment_count: usize,
    pub(super) comment_references: Vec<String>,
    style_names: BTreeMap<(String, String), Option<String>>,
    numbering_styles: BTreeMap<String, Option<String>>,
    abstract_nums: BTreeMap<String, Abstract>,
    nums: BTreeMap<String, String>,
    style_levels: BTreeMap<String, ListLevel>,
    relationships: BTreeMap<String, String>,
    fields: Vec<Field>,
    instruction: String,
    deleted: Vec<String>,
    pub resources: R,
}
impl<R: Resources> Reader<R> {
    pub fn new(
        styles: Option<&str>,
        numbering: Option<&str>,
        relationships: &[Relationship],
        resources: R,
    ) -> Result<Self> {
        let styles = styles.map(xml::parse).transpose()?;
        let mut result = Self {
            style_map: Default::default(),
            note_references: vec![],
            comments: Arc::new(BTreeMap::new()),
            comment_count: 0,
            comment_references: vec![],
            style_names: BTreeMap::new(),
            numbering_styles: BTreeMap::new(),
            abstract_nums: BTreeMap::new(),
            nums: BTreeMap::new(),
            style_levels: BTreeMap::new(),
            relationships: relationships
                .iter()
                .map(|r| (r.id.clone(), r.target.clone()))
                .collect(),
            fields: Vec::new(),
            instruction: String::new(),
            deleted: Vec::new(),
            resources,
        };
        if let Some(document) = &styles {
            for style in document
                .root_element()
                .children()
                .filter(|n| is_word(*n, "style"))
            {
                let kind =
                    attr(style, "type").ok_or_else(|| Error::malformed("style without type"))?;
                let id = attr(style, "styleId")
                    .ok_or_else(|| Error::malformed("style without styleId"))?;
                if kind == "numbering" {
                    result.numbering_styles.entry(id.into()).or_insert_with(|| {
                        child(style, "pPr")
                            .and_then(|n| child(n, "numPr"))
                            .and_then(|n| child(n, "numId"))
                            .and_then(|n| attr(n, "val"))
                            .map(str::to_owned)
                    });
                } else {
                    result
                        .style_names
                        .entry((kind.into(), id.into()))
                        .or_insert_with(|| {
                            child(style, "name")
                                .and_then(|n| attr(n, "val"))
                                .map(str::to_owned)
                        });
                }
            }
        }
        if let Some(input) = numbering {
            let document = xml::parse(input)?;
            // Build raw maps first: duplicate abstract IDs replace their earlier
            // definition while preserving Python dict insertion order for pStyle.
            let mut ordered = Vec::<(String, Node<'_, '_>)>::new();
            for n in document
                .root_element()
                .children()
                .filter(|n| is_word(*n, "abstractNum"))
            {
                let id = attr(n, "abstractNumId").unwrap_or("").to_owned();
                if let Some((_, old)) = ordered.iter_mut().find(|(key, _)| key == &id) {
                    *old = n;
                } else {
                    ordered.push((id, n));
                }
            }
            for (id, n) in ordered {
                let mut levels = BTreeMap::new();
                let mut ordered_levels = Vec::<(String, Node<'_, '_>)>::new();
                let mut missing = None;
                for level in n.children().filter(|n| is_word(*n, "lvl")) {
                    if let Some(index) = attr(level, "ilvl") {
                        if let Some((_, old)) =
                            ordered_levels.iter_mut().find(|(key, _)| key == index)
                        {
                            *old = level;
                        } else {
                            ordered_levels.push((index.to_owned(), level));
                        }
                    } else {
                        missing = Some(level);
                    }
                }
                if !ordered_levels.iter().any(|(id, _)| id == "0") {
                    if let Some(level) = missing {
                        ordered_levels.push(("0".into(), level));
                    }
                }
                for (index, level) in ordered_levels {
                    let value = ListLevel {
                        index: index.clone(),
                        ordered: child(level, "numFmt").and_then(|n| attr(n, "val"))
                            != Some("bullet"),
                    };
                    if let Some(style) = child(level, "pStyle").and_then(|n| attr(n, "val")) {
                        result.style_levels.insert(style.into(), value.clone());
                    }
                    levels.insert(index, value);
                }
                result.abstract_nums.insert(
                    id,
                    Abstract {
                        levels,
                        link: child(n, "numStyleLink")
                            .and_then(|n| attr(n, "val"))
                            .map(str::to_owned),
                    },
                );
            }
            for n in document
                .root_element()
                .children()
                .filter(|n| is_word(*n, "num"))
            {
                let abstract_id = child(n, "abstractNumId")
                    .and_then(|n| attr(n, "val"))
                    .ok_or_else(|| Error::malformed("num without abstractNumId"))?;
                result
                    .nums
                    .insert(attr(n, "numId").unwrap_or("").into(), abstract_id.into());
            }
        }
        Ok(result)
    }
    fn list(&self, id: &str, level: &str) -> Result<Option<ListLevel>> {
        let mut id = id;
        let mut seen = HashSet::new();
        loop {
            if !seen.insert(id) {
                return Err(Error::malformed("cyclic numbering style link"));
            }
            let Some(def) = self.nums.get(id).and_then(|id| self.abstract_nums.get(id)) else {
                return Ok(None);
            };
            if let Some(link) = &def.link {
                id = self
                    .numbering_styles
                    .get(link)
                    .and_then(|id| id.as_deref())
                    .ok_or_else(|| Error::malformed("missing numbering style link"))?;
            } else {
                return Ok(def.levels.get(level).cloned());
            }
        }
    }
    fn paragraph_path(&self, node: Node<'_, '_>, nodes: Vec<Html>) -> Result<Vec<Html>> {
        let props = child(node, "pPr");
        let id = props
            .and_then(|n| child(n, "pStyle"))
            .and_then(|n| attr(n, "val"));
        let upper_name = id
            .and_then(|id| self.style_names.get(&("paragraph".into(), id.into())))
            .and_then(|s| s.as_deref())
            .map(str::to_uppercase);
        let name = upper_name.as_deref();
        let list = self.paragraph_list(props, id)?;
        if let Some(path) = self.style_map.find(
            "p",
            id,
            self.style_name("paragraph", id),
            list.as_ref().map(|v| (v.index.as_str(), v.ordered)),
            None,
        ) {
            return Ok(path.wrap(nodes));
        }
        // Default style-map order matters: named headings precede list rules.
        for level in 1..=6 {
            if id == Some(format!("Heading{level}").as_str())
                || name == Some(format!("HEADING {level}").as_str())
            {
                return Ok(vec![Html::element(&format!("h{level}"), false, nodes)]);
            }
        }
        if id == Some("Heading") || name == Some("HEADING") {
            return Ok(vec![Html::element("h1", false, nodes)]);
        }
        if matches!(
            name,
            Some("FOOTNOTE TEXT" | "ENDNOTE TEXT" | "ANNOTATION TEXT" | "FOOTNOTE" | "ENDNOTE")
        ) {
            return Ok(vec![Html::element("p", false, nodes)]);
        }
        if let Some(list) = list {
            if let Some(index) = ["0", "1", "2", "3", "4"]
                .iter()
                .position(|s| *s == list.index)
            {
                let list_tag = if list.ordered { "ol" } else { "ul" };
                let mut output = vec![Html::element("li", false, nodes)];
                output = vec![Html::element(list_tag, true, output)];
                for _ in 0..index {
                    output = vec![Html::element("li", true, output)];
                    output = vec![Html::element("ul", true, output).alternatives(&["ul", "ol"])];
                }
                return Ok(output);
            }
        }
        Ok(vec![Html::element("p", false, nodes)])
    }
    fn paragraph_list(
        &self,
        props: Option<Node<'_, '_>>,
        id: Option<&str>,
    ) -> Result<Option<ListLevel>> {
        let num = props.and_then(|n| child(n, "numPr"));
        let num_id = num
            .and_then(|n| child(n, "numId"))
            .and_then(|n| attr(n, "val"));
        let level = num
            .and_then(|n| child(n, "ilvl"))
            .and_then(|n| attr(n, "val"));
        let list = if let (Some(id), Some(level)) = (num_id, level) {
            self.list(id, level)?
        } else if let Some(value) = id.and_then(|id| self.style_levels.get(id)) {
            Some(value.clone())
        } else if let Some(id) = num_id {
            self.list(id, "0")?
        } else {
            None
        };
        Ok(list)
    }
    pub fn set_style_map(&mut self, text: &str) -> Vec<String> {
        self.style_map = super::style_map::Rules::parse(text);
        self.style_map.warnings.clone()
    }
    pub(super) fn inherit_references<S>(&mut self, other: &mut Reader<S>) {
        self.note_references = std::mem::take(&mut other.note_references);
        self.comment_count = other.comment_count;
        self.comment_references = std::mem::take(&mut other.comment_references);
    }
    fn style_name(&self, kind: &str, id: Option<&str>) -> Option<&str> {
        id.and_then(|id| self.style_names.get(&(kind.into(), id.into())))
            .and_then(|s| s.as_deref())
    }
    pub fn read(&mut self, node: Node<'_, '_>) -> Result<Vec<Html>> {
        let mut nodes = self.node(node)?.body;
        self.resolve_references(&mut nodes)?;
        Ok(nodes)
    }
    fn resolve_references(&mut self, nodes: &mut Vec<Html>) -> Result<()> {
        let mut output = Vec::with_capacity(nodes.len());
        for mut node in std::mem::take(nodes) {
            match &mut node {
                Html::ResourceFragment(text) => {
                    let tree = xml::parse(text)?;
                    let source = tree
                        .root_element()
                        .children()
                        .find(|n| n.is_element())
                        .unwrap();
                    let mut rendered = self.resources.render(source)?;
                    self.resolve_references(&mut rendered)?;
                    output.extend(rendered);
                    continue;
                }
                Html::NoteReference { kind, id } => {
                    let is_comment = kind == "comment";
                    let label = if is_comment {
                        let initials = self
                            .comments
                            .get(id)
                            .ok_or_else(|| Error::missing(&format!("comment {id}")))?;
                        self.comment_count += 1;
                        self.comment_references.push(id.clone());
                        format!("[{initials}{}]", self.comment_count)
                    } else {
                        self.note_references.push((kind.clone(), id.clone()));
                        format!("[{}]", self.note_references.len())
                    };
                    let link = Html::element("a", false, vec![Html::Text(label)])
                        .attribute("href", format!("#{kind}-{id}"))
                        .attribute("id", format!("{kind}-ref-{id}"));
                    node = if is_comment {
                        link
                    } else {
                        Html::element("sup", false, vec![link])
                    };
                }
                Html::Element { children, .. } => self.resolve_references(children)?,
                _ => (),
            }
            output.push(node);
        }
        *nodes = output;
        Ok(())
    }
    pub fn table_html(&mut self, table: Node<'_, '_>) -> Result<String> {
        Ok(serialize(self.read(table)?))
    }
    pub(super) fn prepared(&mut self, table: Node<'_, '_>) -> Result<Vec<Html>> {
        Ok(collapse(strip_empty(self.read(table)?)))
    }
    /// Structural correction is applied after Mammoth's empty stripping and
    /// collapse, because hidden/empty HTML branches must not enter row counts.
    /// This is still the pre-equation, pre-link-validation stage.
    pub fn normalized_table_html(&mut self, table: Node<'_, '_>) -> Result<String> {
        let mut nodes = collapse(strip_empty(self.read(table)?));
        normalize_colspans(&mut nodes);
        let mut output = String::new();
        write(&nodes, &mut output);
        Ok(output)
    }
    fn children(&mut self, node: Node<'_, '_>) -> Result<Read> {
        let mut result = Read::default();
        for child in node.children().filter(|n| n.is_element()) {
            let mut next = self.node(child)?;
            result.body.append(&mut next.body);
            result.extra.append(&mut next.extra);
        }
        Ok(result)
    }
    fn node(&mut self, node: Node<'_, '_>) -> Result<Read> {
        let ns = node.tag_name().namespace().unwrap_or("");
        let tag = node.tag_name().name();
        if ns.contains("/math")
            || crate::package::namespace_family(Some(ns), "drawingml/wordprocessingDrawing")
                && matches!(tag, "inline" | "anchor")
            || ns == "urn:schemas-microsoft-com:vml" && tag == "imagedata"
            || ns == "urn:schemas-microsoft-com:office:office" && tag == "OLEObject"
        {
            // HtmlPath.ignore never calls its child generator in Mammoth.
            // Defer media rendering until all enclosing style paths are known.
            return Ok(Read::one(Html::ResourceFragment(fragment(node))));
        }
        if ns == "http://schemas.openxmlformats.org/markup-compatibility/2006"
            && tag == "AlternateContent"
        {
            return match node.children().find(|n| super::tables::mc(*n, "Fallback")) {
                Some(n) => self.children(n),
                None => Ok(Read::default()),
            };
        }
        if ns == "urn:schemas-microsoft-com:vml"
            && matches!(tag, "group" | "rect" | "roundrect" | "shape" | "textbox")
        {
            return self.children(node);
        }
        if !crate::package::word_namespace(Some(ns)) {
            return Ok(Read::default());
        }
        match tag {
            "t" => Ok(Read::one(Html::Text(
                node.descendants()
                    .filter(|n| n.is_text())
                    .filter_map(|n| n.text())
                    .collect(),
            ))),
            "tab" => Ok(Read::one(Html::Text("\t".into()))),
            "noBreakHyphen" | "softHyphen" | "sym" => {
                if tag == "sym"
                    && attr(node, "char")
                        .and_then(super::tables::symbol_number)
                        .is_none()
                {
                    return Err(Error::malformed("invalid symbol character code"));
                }
                Ok(Read::one(Html::Text(super::tables::char_fragment(node))))
            }
            "ins" | "object" | "smartTag" | "drawing" | "txbxContent" | "footnote" | "endnote"
            | "comment" => self.children(node),
            "pict" => {
                let mut read = self.children(node)?;
                read.extra.append(&mut read.body);
                Ok(read)
            }
            "r" => {
                let props = child(node, "rPr");
                let mut read = self.children(node)?;
                if let Some(href) = self.fields.iter().rev().find_map(|f| f.href.as_ref()) {
                    read.body = vec![Html::element("a", true, read.body).attribute("href", href)];
                }
                let mut paths = vec![];
                if let Some(color) = props
                    .and_then(|n| child(n, "highlight"))
                    .and_then(|n| attr(n, "val"))
                    .filter(|s| !s.is_empty() && *s != "none")
                {
                    if let Some(path) =
                        self.style_map
                            .find("highlight", None, None, None, Some(color))
                    {
                        paths.push(path.clone());
                    }
                }
                for (property, matcher, default) in [
                    ("smallCaps", "small-caps", None),
                    ("caps", "all-caps", None),
                    ("strike", "strike", Some("s")),
                    ("u", "u", None),
                ] {
                    let enabled = if property == "u" {
                        props.and_then(|n| child(n, "u")).is_some_and(|n| {
                            !matches!(attr(n, "val"), None | Some("none" | "false" | "0"))
                        })
                    } else {
                        onoff(props, property)
                    };
                    if enabled {
                        if let Some(path) = self.style_map.find(matcher, None, None, None, None) {
                            paths.push(path.clone());
                        } else if let Some(tag) = default {
                            paths.push(super::style_map::Path {
                                ignore: false,
                                elements: vec![Html::element(tag, true, vec![])],
                            });
                        }
                    }
                }
                let vertical = props
                    .and_then(|n| child(n, "vertAlign"))
                    .and_then(|n| attr(n, "val"));
                if let Some(tag) = match vertical {
                    Some("subscript") => Some("sub"),
                    Some("superscript") => Some("sup"),
                    _ => None,
                } {
                    paths.push(super::style_map::Path {
                        ignore: false,
                        elements: vec![Html::element(tag, true, vec![])],
                    });
                }
                for (property, matcher, tag) in [("i", "i", "em"), ("b", "b", "strong")] {
                    if onoff(props, property) {
                        paths.push(
                            self.style_map
                                .find(matcher, None, None, None, None)
                                .cloned()
                                .unwrap_or_else(|| super::style_map::Path {
                                    ignore: false,
                                    elements: vec![Html::element(tag, true, vec![])],
                                }),
                        );
                    }
                }
                let id = props
                    .and_then(|n| child(n, "rStyle"))
                    .and_then(|n| attr(n, "val"));
                let name = self.style_name("character", id);
                if let Some(path) = self.style_map.find("r", id, name, None, None) {
                    paths.push(path.clone());
                } else if name.is_some_and(|s| s.to_uppercase() == "STRONG") {
                    paths.push(super::style_map::Path {
                        ignore: false,
                        elements: vec![Html::element("strong", true, vec![])],
                    });
                }
                for path in paths {
                    read.body = path.wrap(read.body);
                }
                Ok(read)
            }
            "p" => {
                let deleted = child(node, "pPr")
                    .and_then(|n| child(n, "rPr"))
                    .is_some_and(|n| child(n, "del").is_some());
                // Deleted paragraph XML is read only when the next paragraph is
                // encountered, so fields are evaluated in their original order.
                if deleted {
                    self.deleted.push(fragment(node));
                    return Ok(Read::default());
                }
                let mut read = Read::default();
                for input in std::mem::take(&mut self.deleted) {
                    let tree = xml::parse(&input)?;
                    let paragraph = tree
                        .root_element()
                        .children()
                        .find(|n| n.is_element())
                        .unwrap();
                    let mut previous = self.children(paragraph)?;
                    read.body.append(&mut previous.body);
                    read.extra.append(&mut previous.extra);
                }
                let mut current = self.children(node)?;
                read.body.append(&mut current.body);
                read.extra.append(&mut current.extra);
                read.body = self.paragraph_path(node, read.body)?;
                read.body.append(&mut read.extra);
                Ok(read)
            }
            "tbl" => self.table(node),
            "hyperlink" => {
                let mut read = self.children(node)?;
                let id = node
                    .attributes()
                    .find(|a| {
                        a.name() == "id"
                            && a.namespace()
                                .is_some_and(|ns| ns.ends_with("/relationships"))
                    })
                    .map(|a| a.value());
                let anchor = attr(node, "anchor");
                let href = if let Some(id) = id {
                    let href = self
                        .relationships
                        .get(id)
                        .ok_or_else(|| Error::missing(id))?;
                    Some(if let Some(anchor) = anchor {
                        format!("{}#{anchor}", href.split('#').next().unwrap_or(""))
                    } else {
                        href.clone()
                    })
                } else {
                    anchor.map(|a| format!("#{a}"))
                };
                if let Some(href) = href {
                    let mut link = Html::element("a", true, read.body).attribute("href", href);
                    if let Some(frame) = attr(node, "tgtFrame").filter(|v| !v.is_empty()) {
                        link = link.attribute("target", frame);
                    }
                    read.body = vec![link];
                }
                Ok(read)
            }
            "bookmarkStart" => match attr(node, "name") {
                Some("_GoBack") => Ok(Read::default()),
                Some(name) => Ok(Read::one(
                    Html::element("a", true, vec![Html::ForceWrite]).attribute("id", name),
                )),
                None => Err(Error::malformed("bookmark without name")),
            },
            "br" => {
                let kind = match attr(node, "type") {
                    None | Some("" | "textWrapping") => "line",
                    Some("page") => "page",
                    Some("column") => "column",
                    _ => return Err(Error::malformed("unsupported break type")),
                };
                let nodes =
                    if let Some(path) = self.style_map.find("br", None, None, None, Some(kind)) {
                        path.wrap(vec![])
                    } else if kind == "line" {
                        vec![Html::element("br", false, vec![])]
                    } else {
                        vec![]
                    };
                Ok(Read::nodes(nodes))
            }
            "sdt" => {
                let mut read = match child(node, "sdtContent") {
                    Some(n) => self.children(n)?,
                    None => Read::default(),
                };
                let checkbox = child(node, "sdtPr").and_then(|n| {
                    n.children().find(|n| {
                        n.tag_name().namespace()
                            == Some("http://schemas.microsoft.com/office/word/2010/wordml")
                            && n.tag_name().name() == "checkbox"
                    })
                });
                if let Some(checkbox) = checkbox {
                    let checked = checkbox
                        .children()
                        .find(|n| n.tag_name().name() == "checked")
                        .is_some_and(|n| {
                            !matches!(
                                n.attributes()
                                    .find(|a| a.name() == "val")
                                    .map(|a| a.value()),
                                Some("false" | "0")
                            )
                        });
                    let replacement = checkbox_html(checked);
                    if !replace_first_text(&mut read.body, &replacement) {
                        read.body = vec![replacement];
                    }
                }
                Ok(read)
            }
            "instrText" => {
                self.instruction.push_str(&text(node));
                Ok(Read::default())
            }
            "fldChar" => self.field(node),
            "footnoteReference" | "endnoteReference" => Ok(Read::one(Html::NoteReference {
                kind: if tag == "footnoteReference" {
                    "footnote"
                } else {
                    "endnote"
                }
                .into(),
                id: attr(node, "id")
                    .ok_or_else(|| Error::malformed("note reference without id"))?
                    .into(),
            })),
            "commentReference" => {
                let id = attr(node, "id")
                    .ok_or_else(|| Error::malformed("comment reference without id"))?;
                let Some(path) = self
                    .style_map
                    .find("comment-reference", None, None, None, None)
                else {
                    return Ok(Read::default());
                };
                Ok(Read::nodes(path.wrap(vec![Html::NoteReference {
                    kind: "comment".into(),
                    id: id.into(),
                }])))
            }
            // Mammoth's default map suppresses comment references and unknown
            // elements (including fldSimple and customXml); do not unwrap them.
            _ => Ok(Read::default()),
        }
    }
    fn field(&mut self, node: Node<'_, '_>) -> Result<Read> {
        match attr(node, "fldCharType") {
            Some("begin") => {
                let checkbox = child(node, "ffData").and_then(|n| child(n, "checkBox"));
                let node_checked = checkbox.is_some_and(|n| {
                    let value = child(n, "checked").or_else(|| child(n, "default"));
                    value.is_some_and(|n| !matches!(attr(n, "val"), Some("false" | "0")))
                });
                self.fields.push(Field {
                    beginning: true,
                    node_checked,
                    href: None,
                    checkbox: None,
                });
                self.instruction.clear();
            }
            Some("separate" | "end") => {
                let mut field = self
                    .fields
                    .pop()
                    .ok_or_else(|| Error::malformed("orphan complex field boundary"))?;
                if field.beginning {
                    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
                    let regex = RE.get_or_init(|| regex::Regex::new(r#"^[\s\x1c-\x1f]*HYPERLINK[\s\x1c-\x1f]+(\\l[\s\x1c-\x1f]+)?(?:"(.*)"|([^\\][^\s\x1c-\x1f]*))"#).unwrap());
                    if let Some(capture) = regex.captures(&self.instruction) {
                        let target = capture.get(2).or_else(|| capture.get(3)).unwrap().as_str();
                        field.href = Some(if capture.get(1).is_some() {
                            format!("#{target}")
                        } else {
                            target.into()
                        });
                    } else if self
                        .instruction
                        .trim_start_matches(space)
                        .starts_with("FORMCHECKBOX")
                    {
                        field.checkbox = Some(field.node_checked);
                    }
                    field.beginning = false;
                }
                if attr(node, "fldCharType") == Some("separate") {
                    self.fields.push(field);
                } else if let Some(checked) = field.checkbox {
                    return Ok(Read::one(checkbox_html(checked)));
                }
            }
            _ => (),
        }
        Ok(Read::default())
    }
    fn table(&mut self, table: Node<'_, '_>) -> Result<Read> {
        let layout = super::tables::layout(table)?;
        let rows: BTreeMap<_, _> = layout
            .rows
            .into_iter()
            .map(|r| (r.source.start, r))
            .collect();
        let mut in_head = true;
        let mut head = Vec::new();
        let mut body = Vec::new();
        let mut extra = Vec::new();
        for source_row in super::tables::structural_children(table) {
            if !is_word(source_row, "tr") {
                let mut read = self.node(source_row)?;
                if !read.body.is_empty() {
                    in_head = false;
                }
                body.append(&mut read.body);
                extra.append(&mut read.extra);
                continue;
            }
            let Some(row) = rows.get(&source_row.range().start) else {
                continue;
            }; // Deleted row.
            in_head &= row.header;
            let kept: BTreeMap<_, _> = row.cells.iter().map(|c| (c.source.start, c)).collect();
            let mut cells = vec![Html::ForceWrite];
            for source_cell in super::tables::structural_children(source_row) {
                if !is_word(source_cell, "tc") {
                    let mut read = self.node(source_cell)?;
                    cells.append(&mut read.body);
                    extra.append(&mut read.extra);
                    continue;
                }
                // Parse continuation cells too: field/deletion state and pict
                // extras affect later cells even when this cell is merged out.
                let mut read = self.children(source_cell)?;
                extra.append(&mut read.extra);
                let Some(cell) = kept.get(&source_cell.range().start) else {
                    continue;
                };
                let mut contents = vec![Html::ForceWrite];
                contents.append(&mut read.body);
                let mut node = Html::element(if in_head { "th" } else { "td" }, false, contents);
                if cell.colspan != 1 {
                    node = node.attribute("colspan", cell.colspan.to_string());
                }
                if cell.rowspan != 1 {
                    node = node.attribute("rowspan", cell.rowspan.to_string());
                }
                cells.push(node);
            }
            let node = Html::element("tr", false, cells);
            if in_head {
                head.push(node);
            } else {
                body.push(node);
            }
        }
        let mut children = vec![Html::ForceWrite];
        if !head.is_empty() {
            children.push(Html::element("thead", false, head));
            children.push(Html::element("tbody", false, body));
        } else {
            children.extend(body);
        }
        Ok(Read {
            body: {
                let id = child(table, "tblPr")
                    .and_then(|n| child(n, "tblStyle"))
                    .and_then(|n| attr(n, "val"));
                if let Some(path) =
                    self.style_map
                        .find("table", id, self.style_name("table", id), None, None)
                {
                    path.wrap(children)
                } else {
                    vec![Html::element("table", false, children)]
                }
            },
            extra,
        })
    }
}
fn onoff(properties: Option<Node<'_, '_>>, name: &str) -> bool {
    properties
        .and_then(|n| child(n, name))
        .is_some_and(|n| !matches!(attr(n, "val"), Some("false" | "0")))
}
fn checkbox_html(checked: bool) -> Html {
    let node = Html::element("input", false, vec![]).attribute("type", "checkbox");
    if checked {
        node.attribute("checked", "checked")
    } else {
        node
    }
}
fn replace_first_text(nodes: &mut [Html], replacement: &Html) -> bool {
    for node in nodes {
        match node {
            Html::Text(text) if !text.is_empty() => {
                *node = replacement.clone();
                return true;
            }
            Html::Element { children, .. } => {
                if replace_first_text(children, replacement) {
                    return true;
                }
            }
            _ => (),
        }
    }
    false
}
pub(super) fn fragment(node: Node<'_, '_>) -> String {
    // Owned deferred/referent fragments retain their in-scope namespaces.
    let mut result = String::from("<deferred");
    for ns in node.namespaces() {
        result.push_str(" xmlns");
        if let Some(prefix) = ns.name() {
            result.push(':');
            result.push_str(prefix);
        }
        result.push_str("=\"");
        result.push_str(
            &ns.uri()
                .replace('&', "&amp;")
                .replace('"', "&quot;")
                .replace('<', "&lt;")
                .replace('>', "&gt;"),
        );
        result.push('"');
    }
    result.push('>');
    result.push_str(&node.document().input_text()[node.range()]);
    result.push_str("</deferred>");
    result
}
