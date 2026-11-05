//! Original RapidDoc layer selection, ordering, colors and cross-page policies.
use super::*;

pub(super) struct Draw<'a> {
    pub item: &'a Value,
    pub color: [u8; 3],
    pub fill: bool,
    pub number: Option<usize>,
    pub shape: bool,
}

fn layer<'a>(out: &mut Vec<Draw<'a>>, items: &[&'a Value], color: [u8; 3], fill: bool) {
    out.extend(items.iter().map(|item| Draw {
        item,
        color,
        fill,
        number: None,
        shape: true,
    }));
}
pub(super) fn list<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}
fn kind(v: &Value) -> &str {
    v["type"].as_str().unwrap_or("")
}
fn cross(v: &Value) -> bool {
    v["cross_page"].as_bool().unwrap_or(false)
}

pub(super) fn plan(page: &Value, mode: Kind) -> Vec<Draw<'_>> {
    match mode {
        Kind::Layout => layout(page),
        Kind::Span => spans(page),
        Kind::LineSort => lines(page),
    }
}

fn layout(page: &Value) -> Vec<Draw<'_>> {
    // Layer order is visible where translucent areas overlap.
    let mut groups: [Vec<&Value>; 15] = std::array::from_fn(|_| vec![]);
    groups[2].extend(list(page, "discarded_blocks"));
    let mut numbered = vec![];
    let mut inner = vec![];
    for block in list(page, "para_blocks") {
        match kind(block) {
            "table" | "image" | "code" => {
                for child in list(block, "blocks") {
                    let group = match (kind(block), kind(child)) {
                        ("code", "code_body") => Some(0),
                        ("code", "code_caption") => Some(1),
                        ("table", "table_body") => Some(3),
                        ("table", "table_caption") => Some(4),
                        ("table", "table_footnote") if !cross(child) => Some(5),
                        ("image", "image_body") => Some(6),
                        ("image", "image_caption") => Some(7),
                        ("image", "image_footnote") => Some(8),
                        _ => None,
                    };
                    if let Some(index) = group {
                        groups[index].push(child);
                    }
                }
                let mut children: Vec<_> = list(block, "blocks").iter().collect();
                if kind(block) == "table" {
                    children.sort_by_key(|b| match kind(b) {
                        "table_caption" => 1,
                        "table_body" => 2,
                        "table_footnote" => 3,
                        _ => 0,
                    });
                }
                for child in children {
                    if kind(block) == "table" && cross(child) {
                        continue;
                    }
                    numbered.push(child);
                    if kind(block) == "table" {
                        for line in list(child, "lines") {
                            for span in list(line, "spans") {
                                inner.extend(list(span, "img_boxes"));
                                inner.extend(list(span, "latex_boxes"));
                            }
                        }
                    }
                }
            }
            "title" | "text" | "ref_text" | "interline_equation" | "list" | "index" => {
                let index = match kind(block) {
                    "title" => 9,
                    "text" | "ref_text" => 10,
                    "interline_equation" => 11,
                    "list" => 12,
                    _ => 14,
                };
                groups[index].push(block);
                numbered.push(block);
                if kind(block) == "list" {
                    groups[13].extend(list(block, "blocks"));
                }
            }
            _ => {}
        }
    }
    let colors = [
        [102, 0, 204],
        [204, 153, 255],
        [158, 158, 158],
        [204, 204, 0],
        [255, 255, 102],
        [229, 255, 204],
        [153, 255, 51],
        [102, 178, 255],
        [255, 178, 102],
        [102, 102, 255],
        [153, 0, 76],
        [0, 255, 0],
        [40, 169, 92],
        [40, 169, 92],
        [40, 169, 92],
    ];
    let mut out = vec![];
    for (index, items) in groups.iter().enumerate() {
        layer(&mut out, items, colors[index], index != 13);
    }
    out.extend(numbered.into_iter().enumerate().map(|(i, item)| Draw {
        item,
        color: [255, 0, 0],
        fill: false,
        number: Some(i + 1),
        shape: false,
    }));
    layer(&mut out, &inner, [0, 255, 0], false);
    out
}

fn spans(page: &Value) -> Vec<Draw<'_>> {
    let mut groups: [Vec<&Value>; 6] = std::array::from_fn(|_| vec![]);
    for block in list(page, "discarded_blocks") {
        if kind(block) == "discarded" {
            for line in list(block, "lines") {
                groups[5].extend(list(line, "spans"));
            }
        }
    }
    for block in list(page, "preproc_blocks") {
        let blocks: Vec<_> = match kind(block) {
            "text" | "title" | "interline_equation" | "list" | "index" => vec![block],
            "image" | "table" => list(block, "blocks").iter().collect(),
            _ => vec![],
        };
        for block in blocks {
            for line in list(block, "lines") {
                for span in list(line, "spans") {
                    let index = match kind(span) {
                        "text" => 0,
                        "inline_equation" | "checkbox" => 1,
                        "interline_equation" => 2,
                        "image" => 3,
                        "table" => 4,
                        _ => continue,
                    };
                    groups[index].push(span);
                }
            }
        }
    }
    let mut out = vec![];
    for (items, color) in groups.iter().zip([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 204, 0],
        [204, 0, 255],
        [158, 158, 158],
    ]) {
        layer(&mut out, items, color, false);
    }
    out
}

fn lines(page: &Value) -> Vec<Draw<'_>> {
    let mut lines = vec![];
    for block in list(page, "preproc_blocks") {
        match kind(block) {
            "text" => lines.extend(list(block, "lines")),
            "title" | "interline_equation" => {
                if block.get("virtual_lines").is_some() {
                    let virtuals = list(block, "virtual_lines");
                    if virtuals.first().is_some_and(|v| !v["index"].is_null()) {
                        lines.extend(virtuals);
                    }
                } else {
                    lines.extend(list(block, "lines"));
                }
            }
            "image" | "table" => {
                for child in list(block, "blocks") {
                    match kind(child) {
                        "image_body" | "table_body" => {
                            let virtuals = list(child, "virtual_lines");
                            lines.extend(
                                if virtuals.first().is_some_and(|v| !v["index"].is_null()) {
                                    virtuals
                                } else {
                                    list(child, "lines")
                                },
                            );
                        }
                        "image_caption" | "table_caption" | "image_footnote" | "table_footnote" => {
                            lines.extend(list(child, "lines"))
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    lines.sort_by(|a, b| {
        a["index"]
            .as_f64()
            .unwrap_or(0.)
            .total_cmp(&b["index"].as_f64().unwrap_or(0.))
    });
    lines
        .into_iter()
        .enumerate()
        .map(|(i, line)| Draw {
            item: &line["bbox"],
            color: [255, 0, 0],
            fill: false,
            number: Some(i + 1),
            shape: true,
        })
        .collect()
}
