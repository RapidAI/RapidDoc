//! Effective run/font and list marker inheritance in the existing PPTX path.
//! Master markers intentionally override layout markers, as in Python.
use super::{
    context::{self, Inheritance},
    geometry::{child, A, P},
};
use crate::docx::body::Integer;
use num_bigint::BigInt;
use roxmltree::Node;
use serde::Serialize;

pub fn level(p: Node<'_, '_>) -> BigInt {
    child(p, A, "pPr")
        .and_then(|n| n.attribute("lvl"))
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or_default()
}
fn level_properties<'a, 'i>(bucket: Node<'a, 'i>, level: &BigInt) -> Option<Node<'a, 'i>> {
    child(bucket, A, &format!("lvl{}pPr", level + 1))
}
fn text_body<'a, 'i>(shape: Node<'a, 'i>) -> Option<Node<'a, 'i>> {
    shape
        .descendants()
        .skip(1)
        .find(|n| n.has_tag_name((P, "txBody")))
}
fn list_style<'a, 'i>(shape: Node<'a, 'i>) -> Option<Node<'a, 'i>> {
    child(text_body(shape)?, A, "lstStyle")
}
pub fn master_bucket<'a, 'i>(
    master: Option<Node<'a, 'i>>,
    shape: Node<'a, 'i>,
) -> Option<Node<'a, 'i>> {
    let styles = master?
        .descendants()
        .skip(1)
        .find(|n| n.has_tag_name((P, "txStyles")))?;
    let bucket = match context::ph_type(shape)? {
        "body" | "obj" => "bodyStyle",
        "title" | "ctrTitle" | "subTitle" => "titleStyle",
        _ => "otherStyle",
    };
    child(styles, P, bucket)
}
pub fn font_sources<'a, 'i>(
    shape: Node<'a, 'i>,
    p: Node<'a, 'i>,
    inheritance: Inheritance<'a, 'i>,
) -> Vec<Node<'a, 'i>> {
    let mut sources = vec![];
    if let Some(n) = child(p, A, "pPr").and_then(|n| child(n, A, "defRPr")) {
        sources.push(n);
    }
    if let Some(n) = child(p, A, "endParaRPr") {
        sources.push(n);
    }
    let level = level(p);
    let mut extend = |bucket: Option<Node<'a, 'i>>| {
        if let Some(bucket) = bucket {
            if let Some(n) = level_properties(bucket, &level).and_then(|n| child(n, A, "defRPr")) {
                sources.push(n);
            }
            if let Some(n) = child(bucket, A, "defPPr").and_then(|n| child(n, A, "defRPr")) {
                sources.push(n);
            }
        }
    };
    extend(list_style(shape));
    if context::placeholder(shape).is_some() {
        extend(
            inheritance
                .layout_placeholder(shape, true)
                .and_then(list_style),
        );
        extend(master_bucket(inheritance.master, shape));
    }
    sources
}
pub fn toggle(value: &str) -> Option<bool> {
    match value.trim().to_lowercase().as_str() {
        "1" | "true" | "t" | "on" => Some(true),
        "0" | "false" | "f" | "off" | "none" => Some(false),
        _ => None,
    }
}
fn bool_attr(n: Node<'_, '_>, attr: &str) -> Option<bool> {
    let value = n.attribute(attr)?;
    if attr == "u" || attr == "strike" {
        let value = value.trim().to_lowercase();
        Some(
            !matches!(value.as_str(), "0" | "false" | "f" | "off" | "none")
                && !(attr == "strike" && value == "nostrike"),
        )
    } else {
        toggle(value)
    }
}
fn font_size(n: Node<'_, '_>) -> Option<f64> {
    let size = n.attribute("sz")?.trim().parse::<i64>().ok()?;
    // Python round(x, 1) rounds the actual binary value. Multiplying by ten
    // before rounding loses the side of a halfway value (e.g. 10.65).
    format!("{:.1}", size as f64 / 100.).parse().ok()
}
#[derive(Debug, Serialize)]
pub struct RunStyle {
    pub styles: Vec<&'static str>,
    pub font_size_pt: Option<f64>,
}
pub fn run_style(run: Node<'_, '_>, sources: &[Node<'_, '_>]) -> RunStyle {
    let direct = child(run, A, "rPr");
    let all = || direct.into_iter().chain(sources.iter().copied());
    let styles = [
        ("b", "bold"),
        ("i", "italic"),
        ("u", "underline"),
        ("strike", "strikethrough"),
    ]
    .into_iter()
    .filter_map(|(attr, name)| {
        all()
            .find_map(|n| bool_attr(n, attr))
            .unwrap_or(false)
            .then_some(name)
    })
    .collect();
    RunStyle {
        styles,
        font_size_pt: all().find_map(font_size),
    }
}
#[derive(Debug, Serialize)]
pub struct ListInfo {
    pub is_list: bool,
    pub attribute: &'static str,
    pub level: Integer,
    pub kind: Option<&'static str>,
    pub start: Option<Integer>,
    pub start_is_explicit_restart: bool,
}
struct Marker {
    list: bool,
    kind: &'static str,
    start: Option<Integer>,
}
fn positive(value: Option<&str>) -> Option<Integer> {
    let n: BigInt = value?.trim().parse().ok()?;
    (n > BigInt::from(0)).then_some(Integer(n))
}
fn marker(properties: Option<Node<'_, '_>>) -> Option<Marker> {
    let p = properties?;
    for (tag, list) in [
        ("buNone", false),
        ("buChar", true),
        ("buAutoNum", true),
        ("buBlip", true),
    ] {
        if let Some(n) = child(p, A, tag) {
            return Some(Marker {
                list,
                kind: tag,
                start: if tag == "buAutoNum" {
                    positive(n.attribute("startAt"))
                } else {
                    None
                },
            });
        }
    }
    None
}
pub fn list_info(
    shape: Node<'_, '_>,
    p: Node<'_, '_>,
    inheritance: Inheritance<'_, '_>,
) -> ListInfo {
    let lvl = level(p);
    let direct = marker(child(p, A, "pPr"));
    let explicit = direct.is_some();
    let resolved = direct
        .or_else(|| marker(list_style(shape).and_then(|n| level_properties(n, &lvl))))
        .or_else(|| {
            let layout = inheritance.layout_placeholder(shape, true)?;
            let layout_result = marker(list_style(layout).and_then(|n| level_properties(n, &lvl)));
            let tag = format!("lvl{}pPr", &lvl + 1);
            let master = master_bucket(inheritance.master, shape).and_then(|n| {
                n.descendants()
                    .skip(1)
                    .find(|n| n.has_tag_name((A, tag.as_str())))
            });
            marker(master).or(layout_result)
        });
    let fallback = resolved.is_none();
    let resolved = resolved.or_else(|| {
        if let Some(n) = p
            .descendants()
            .skip(1)
            .find(|n| n.has_tag_name((A, "buAutoNum")))
        {
            Some(Marker {
                list: true,
                kind: "buAutoNum",
                start: positive(n.attribute("startAt")),
            })
        } else if p
            .descendants()
            .skip(1)
            .any(|n| n.has_tag_name((A, "buChar")))
        {
            Some(Marker {
                list: true,
                kind: "buChar",
                start: None,
            })
        } else {
            None
        }
    });
    match resolved {
        Some(m) => ListInfo {
            is_list: m.list,
            attribute: if m.list && m.kind == "buAutoNum" {
                "ordered"
            } else {
                "unordered"
            },
            level: Integer(lvl),
            kind: Some(m.kind),
            start_is_explicit_restart: m.kind == "buAutoNum"
                && m.start.is_some()
                && (explicit || fallback),
            start: m.start,
        },
        None => ListInfo {
            is_list: lvl > BigInt::from(0),
            attribute: "unordered",
            level: Integer(lvl),
            kind: None,
            start: None,
            start_is_explicit_restart: false,
        },
    }
}
#[derive(Debug, Serialize)]
pub struct ParagraphInfo {
    pub font_size_pt: Option<f64>,
    pub all_bold: bool,
    pub runs: Vec<RunStyle>,
    pub list: ListInfo,
}
pub fn paragraph_info(
    shape: Node<'_, '_>,
    p: Node<'_, '_>,
    inheritance: Inheritance<'_, '_>,
) -> ParagraphInfo {
    let sources = font_sources(shape, p, inheritance);
    let mut runs = vec![];
    let mut size = None::<f64>;
    let mut all_bold = true;
    let mut nonblank = false;
    for run in p.children().filter(|n| {
        (n.tag_name().namespace() == Some(A) && matches!(n.tag_name().name(), "r" | "br" | "fld"))
            || n.has_tag_name(("http://schemas.microsoft.com/office/drawing/2010/main", "m"))
    }) {
        let style = run_style(run, &sources);
        let text = run
            .descendants()
            .filter(|n| {
                n.has_tag_name((A, "t"))
                    || n.has_tag_name((
                        "http://schemas.openxmlformats.org/officeDocument/2006/math",
                        "t",
                    ))
            })
            .filter_map(|n| n.text())
            .collect::<String>();
        if !text
            .trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
            .is_empty()
        {
            nonblank = true;
            all_bold &= style.styles.contains(&"bold");
            if let Some(s) = style.font_size_pt {
                size = Some(size.map_or(s, |old| old.max(s)));
            }
        }
        runs.push(style);
    }
    ParagraphInfo {
        font_size_pt: size,
        all_bold: nonblank && all_bold,
        runs,
        list: list_info(shape, p, inheritance),
    }
}
