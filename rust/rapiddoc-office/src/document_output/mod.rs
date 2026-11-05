//! Native Office model -> middle contract. JSON is only a serialization boundary.
use crate::{
    docx::{
        body::{self, Content, Integer, Kind},
        spans,
    },
    Result,
};
use num_bigint::BigInt;
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
pub mod html;
pub mod inline;
pub mod render;

#[derive(Debug, Default, Clone, Serialize)]
pub struct Span {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub style: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<Span>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub html: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_base64: Option<Arc<str>>,
}
#[derive(Debug, Clone, Serialize)]
pub struct Line {
    pub spans: Vec<Span>,
}
#[derive(Debug, Default, Clone, Serialize)]
pub struct Block {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lines: Option<Vec<Line>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocks: Option<Vec<Block>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_numbered_style: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section_number: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribute: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ilevel: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<Integer>,
    #[serde(skip)]
    depth: i64,
}
#[derive(Debug, Serialize)]
pub struct Page {
    pub para_blocks: Vec<Block>,
    pub discarded_blocks: Vec<Block>,
    pub page_idx: usize,
}
#[derive(Debug, Serialize)]
pub struct Middle {
    pub pdf_info: Vec<Page>,
    #[serde(rename = "_backend")]
    pub backend: &'static str,
    #[serde(rename = "_version_name")]
    pub version: String,
}
pub struct Output {
    pub middle: Middle,
    pub assets: BTreeMap<String, Arc<[u8]>>,
}
impl Span {
    fn text(kind: &str, content: String) -> Self {
        Self {
            kind: kind.into(),
            content: Some(content),
            ..Self::default()
        }
    }
}
impl Block {
    fn lines(kind: &str, spans: Vec<Span>) -> Self {
        Self {
            kind: kind.into(),
            lines: Some(vec![Line { spans }]),
            ..Self::default()
        }
    }
    pub fn spans(&self) -> impl Iterator<Item = &Span> {
        self.lines.iter().flatten().flat_map(|l| &l.spans)
    }
    pub fn children(&self) -> &[Block] {
        self.blocks.as_deref().unwrap_or(&[])
    }
}
fn inline(s: Vec<spans::Span>) -> Vec<Span> {
    let mut out = vec![];
    for s in s {
        match s {
            spans::Span::Text { content, styles } => out.push(Span {
                style: styles.into_iter().map(str::to_owned).collect(),
                ..Span::text("text", content)
            }),
            spans::Span::Equation { content } => out.push(Span::text("inline_equation", content)),
            spans::Span::Link { url, content } => {
                let children = inline(content);
                if let Some(url) = crate::links::sanitize(Some(&url)) {
                    let content = children
                        .iter()
                        .filter_map(|s| s.content.as_deref())
                        .collect::<String>();
                    out.push(Span {
                        url: Some(url.into()),
                        children: Some(children),
                        ..Span::text("hyperlink", content)
                    });
                } else {
                    out.extend(children);
                }
            }
        }
    }
    out
}
fn plain(content: &Content) -> String {
    match content {
        Content::Spans(s) => s
            .iter()
            .map(|s| match s {
                spans::Span::Text { content, .. } | spans::Span::Equation { content } => {
                    content.clone()
                }
                spans::Span::Link { content, .. } => plain(&Content::Spans(content.clone())),
            })
            .collect(),
        _ => String::new(),
    }
}
fn kind(k: Kind) -> &'static str {
    match k {
        Kind::Text => "text",
        Kind::DocTitle | Kind::ParagraphTitle => "title",
        Kind::Caption => "caption",
        Kind::Code => "code",
        Kind::Textbox => "textbox",
        Kind::Equation => "interline_equation",
        Kind::List => "list",
        Kind::Index => "index",
        Kind::Table => "table",
        Kind::Header => "header",
        Kind::Footer => "footer",
        Kind::PageFootnote => "page_footnote",
        Kind::Image => "image",
        Kind::Chart => "chart",
    }
}
fn types(source: &[body::Block]) -> Vec<String> {
    let mut kinds = source
        .iter()
        .map(|b| kind(b.kind).to_owned())
        .collect::<Vec<_>>();
    for i in 0..source.len() {
        if matches!(kinds[i].as_str(), "image" | "table" | "chart")
            && i + 1 < source.len()
            && kinds[i + 1] == "text"
        {
            let text = plain(&source[i + 1].content).trim().to_lowercase();
            let prefixes: &[&str] = match kinds[i].as_str() {
                "table" => &["表", "table"],
                "image" => &["图", "fig"],
                _ => &["图", "fig", "chart"],
            };
            if prefixes.iter().any(|s| text.starts_with(s)) {
                kinds[i + 1] = "caption".into();
            }
        }
    }
    let original = kinds.clone();
    for i in 0..source.len() {
        if original[i] != "caption" {
            continue;
        }
        let parent = (0..i)
            .rev()
            .find(|j| original[*j] != "caption")
            .filter(|j| matches!(original[*j].as_str(), "image" | "table" | "chart"))
            .or_else(|| {
                (i + 1..source.len())
                    .find(|j| original[*j] != "caption")
                    .filter(|j| matches!(original[*j].as_str(), "image" | "table" | "chart"))
            });
        kinds[i] = parent
            .map(|j| format!("{}_caption", original[j]))
            .unwrap_or_else(|| "text".into());
    }
    kinds
}
fn convert(b: body::Block, k: &str) -> Result<Option<Block>> {
    let empty = match &b.content {
        Content::Spans(v) => v.is_empty(),
        Content::Blocks(v) => v.is_empty(),
        Content::Text(s) => s.is_empty(),
        Content::None => true,
    };
    if empty && !matches!(k, "image" | "chart") {
        return Ok(None);
    }
    let mut block = match b.content {
        Content::Spans(s) => Block::lines(k, inline(s)),
        Content::Text(s) => match k {
            "code" => Block::lines("code", vec![Span::text("text", s)]),
            "table" => Block::lines(
                "table_body",
                vec![Span {
                    html: Some(html::clean(&s)?),
                    kind: "table".into(),
                    ..Default::default()
                }],
            ),
            "chart" => Block::lines(
                "chart_body",
                vec![Span {
                    image_base64: b.image_base64.clone(),
                    ..Span::text("chart", s)
                }],
            ),
            _ => Block::lines(k, vec![Span::text(k, s)]),
        },
        Content::Blocks(children) => {
            let mut converted = vec![];
            for child in children {
                let ck = kind(child.kind);
                if (k == "list" && matches!(ck, "text" | "list"))
                    || (k == "index" && matches!(ck, "text" | "index"))
                {
                    let prefix = child.list_label.clone().filter(|s| !s.is_empty());
                    if let Some(mut child) = convert(child, ck)? {
                        child.index = None;
                        if k == "list" {
                            child.anchor = None;
                            child.prefix = prefix;
                        }
                        converted.push(child);
                    }
                }
            }
            Block {
                kind: k.into(),
                blocks: Some(converted),
                ilevel: Some(b.ilevel.unwrap_or(0)),
                attribute: if k == "list" {
                    Some(b.attribute.unwrap_or("unordered").into())
                } else {
                    None
                },
                start: b.start,
                ..Default::default()
            }
        }
        Content::None => Block::default(),
    };
    if k == "image" {
        block = Block::lines(
            "image_body",
            vec![Span {
                kind: "image".into(),
                image_base64: b.image_base64,
                ..Default::default()
            }],
        );
    }
    if matches!(k, "title" | "text" | "interline_equation") {
        block.anchor = b
            .anchor
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty());
    }
    if k == "title" {
        block.level = Some(b.level.unwrap_or(1));
        block.is_numbered_style = Some(b.is_numbered_style.unwrap_or(false));
        block.depth = if matches!(b.kind, Kind::DocTitle) {
            0
        } else {
            (block.level.unwrap() - 1).max(1)
        };
    }
    Ok(Some(block))
}
fn group(mut blocks: Vec<Block>, media: &str) -> (Vec<Block>, Vec<Block>) {
    let body = format!("{media}_body");
    let caption = format!("{media}_caption");
    let indices = blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| b.kind == body)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let caps = blocks
        .iter()
        .filter(|b| b.kind == caption)
        .map(|b| b.index.unwrap())
        .collect::<BTreeSet<_>>();
    let mut assigned = BTreeMap::<usize, Vec<usize>>::new();
    for (i, b) in blocks.iter().enumerate().filter(|(_, b)| b.kind == caption) {
        let at = b.index.unwrap();
        let best = indices.iter().min_by_key(|i| {
            let other = blocks[**i].index.unwrap();
            let (a, z) = (at.min(other), at.max(other));
            (z - a).saturating_sub(caps.range(a + 1..z).count())
        });
        if let Some(best) = best {
            assigned.entry(*best).or_default().push(i);
        }
    }
    let mut used = BTreeSet::new();
    let mut out = vec![];
    for i in indices {
        let bi = blocks[i].index.unwrap();
        let mut selected = assigned.remove(&i).unwrap_or_default();
        selected.sort_by_key(|j| std::cmp::Reverse(blocks[*j].index.unwrap()));
        let mut keep: Vec<usize> = vec![];
        for j in selected {
            if let Some(prev) = keep.last().copied() {
                let prev = blocks[prev].index.unwrap();
                let now = blocks[j].index.unwrap();
                if prev != now + 1 && !(prev == now + 2 && now + 1 == bi) {
                    break;
                }
            }
            keep.push(j);
        }
        keep.push(i);
        keep.sort_by_key(|j| blocks[*j].index);
        let mut children = vec![];
        for j in keep {
            used.insert(j);
            children.push(std::mem::take(&mut blocks[j]));
        }
        out.push(Block {
            kind: media.into(),
            index: Some(bi),
            blocks: Some(children),
            ..Default::default()
        });
    }
    let mut rest = vec![];
    for (i, mut b) in blocks.into_iter().enumerate() {
        if !used.contains(&i) {
            b.kind = "text".into();
            rest.push(b);
        }
    }
    (out, rest)
}
pub fn middle(pages: Vec<Vec<body::Block>>, version: &str) -> Result<Output> {
    let mut output = Output {
        middle: Middle {
            pdf_info: vec![],
            backend: "office",
            version: version.into(),
        },
        assets: BTreeMap::new(),
    };
    for (page_idx, source) in pages.into_iter().enumerate() {
        let kinds = types(&source);
        let mut regular = vec![];
        let mut discarded = vec![];
        let mut media = BTreeMap::<&str, Vec<Block>>::new();
        for (index, (source, k)) in source.into_iter().zip(kinds).enumerate() {
            if let Some(mut b) = convert(source, &k)? {
                b.index = Some(index);
                match b.kind.as_str() {
                    "image_body" | "image_caption" => media.entry("image").or_default().push(b),
                    "table_body" | "table_caption" => media.entry("table").or_default().push(b),
                    "chart_body" | "chart_caption" => media.entry("chart").or_default().push(b),
                    "header" | "footer" | "page_footnote" => discarded.push(b),
                    _ => regular.push(b),
                }
            }
        }
        for k in ["image", "chart", "table"] {
            let (mut grouped, mut rest) = group(media.remove(k).unwrap_or_default(), k);
            regular.append(&mut grouped);
            regular.append(&mut rest);
        }
        regular.sort_by_key(|b| b.index);
        for b in &mut regular {
            html::save_images(b, &mut output.assets)?;
        }
        output.middle.pdf_info.push(Page {
            para_blocks: regular,
            discarded_blocks: discarded,
            page_idx,
        });
    }
    let mut counters = BTreeMap::<i64, BigInt>::new();
    for page in &mut output.middle.pdf_info {
        for b in &mut page.para_blocks {
            if b.kind != "title" || b.depth == 0 {
                continue;
            }
            let mut parts = None;
            if b.is_numbered_style == Some(true) {
                for ancestor in 1..b.depth {
                    let n = counters.entry(ancestor).or_default();
                    if *n == BigInt::from(0) {
                        *n = BigInt::from(1);
                    }
                }
                *counters.entry(b.depth).or_default() += 1;
                for (k, v) in &mut counters {
                    if *k > b.depth {
                        *v = BigInt::from(0);
                    }
                }
                b.section_number = Some(
                    (1..=b.depth)
                        .map(|k| counters[&k].to_string())
                        .collect::<Vec<_>>()
                        .join("."),
                );
            } else if let Some(s) = b.spans().next().and_then(|s| s.content.as_deref()) {
                let s = s.trim();
                let prefix = s
                    .chars()
                    .take_while(|c| *c == '.' || crate::docx::decimal(*c).is_some())
                    .collect::<String>();
                let pieces = prefix
                    .trim_end_matches('.')
                    .split('.')
                    .map(|s| {
                        crate::docx::integer_text(Some(s)).and_then(|s| s.parse::<BigInt>().ok())
                    })
                    .collect::<Option<Vec<_>>>();
                if pieces.as_ref().is_some_and(|p| p.len() == b.depth as usize) {
                    parts = pieces;
                }
            }
            if let Some(parts) = parts {
                let depth = parts.len() as i64;
                for (i, v) in parts.into_iter().enumerate() {
                    counters.insert(i as i64 + 1, v);
                }
                for (k, v) in &mut counters {
                    if *k > depth {
                        *v = BigInt::from(0);
                    }
                }
            }
        }
    }
    let anchors = output
        .middle
        .pdf_info
        .iter()
        .flat_map(|p| &p.para_blocks)
        .filter_map(|b| b.anchor.clone())
        .collect::<BTreeSet<_>>();
    fn link(b: &mut Block, anchors: &BTreeSet<String>) {
        for c in b.blocks.iter_mut().flatten() {
            if c.kind == "index" {
                link(c, anchors);
            } else if c.kind == "text" && c.anchor.as_ref().is_none_or(|s| !anchors.contains(s)) {
                c.anchor = None;
            }
        }
    }
    if !anchors.is_empty() {
        for page in &mut output.middle.pdf_info {
            for b in &mut page.para_blocks {
                if b.kind == "index" {
                    link(b, &anchors);
                }
            }
        }
    }
    Ok(output)
}
