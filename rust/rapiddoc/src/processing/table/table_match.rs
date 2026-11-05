//! 旧SLANet TableMatch的IoU/distance、原顺序HTML和逻辑点；不替代预览安全输出器。
// Adapted from PaddlePaddle TableMatch (Copyright 2022 PaddlePaddle Authors), Apache-2.0;
// see rapiddoc-inference/licenses/PADDLE-TABLE.txt and the wheel's PADDLE-TABLE notice.
use super::coordinates;
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};

fn cells(v: &Value) -> Result<Vec<[f64; 4]>, Error> {
    let a = v
        .as_array()
        .filter(|a| !a.is_empty() && a.len() <= rapiddoc_core::resources::limit(10_000))
        .ok_or_else(invalid)?;
    a.iter()
        .map(|b| {
            let b = b
                .as_array()
                .filter(|b| matches!(b.len(), 4 | 8))
                .ok_or_else(invalid)?;
            if b.len() == 4 {
                coordinates(&json!(b))
            } else {
                let nums = b
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .filter(|n| n.is_finite() && n.abs() <= 1e9)
                            .ok_or_else(invalid)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok([
                    nums.iter()
                        .step_by(2)
                        .copied()
                        .fold(f64::INFINITY, f64::min),
                    nums.iter()
                        .skip(1)
                        .step_by(2)
                        .copied()
                        .fold(f64::INFINITY, f64::min),
                    nums.iter()
                        .step_by(2)
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max),
                    nums.iter()
                        .skip(1)
                        .step_by(2)
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max),
                ])
            }
        })
        .collect()
}
fn metric(a: [f64; 4], b: [f64; 4]) -> (f64, f64) {
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let (left, right, top, bottom) = (
        a[1].max(b[1]),
        a[3].min(b[3]),
        a[0].max(b[0]),
        a[2].min(b[2]),
    );
    let intersection = (right - left) * (bottom - top);
    let union = (area_a + area_b) - intersection;
    let iou = if left < right && top < bottom && union != 0. {
        intersection / union
    } else {
        0.
    };
    let d = [
        (b[0] - a[0]).abs(),
        (b[1] - a[1]).abs(),
        (b[2] - a[2]).abs(),
        (b[3] - a[3]).abs(),
    ];
    (
        1. - iou,
        ((d[0] + d[1]) + d[2]) + d[3] + (d[0] + d[1]).min(d[2] + d[3]),
    )
}
fn matches(
    cells: &[[f64; 4]],
    dt: &[[f64; 4]],
    work: &mut Work,
) -> Result<BTreeMap<usize, Vec<usize>>, Error> {
    let mut out: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &a) in dt.iter().enumerate() {
        let mut best = 0;
        let mut score = (f64::INFINITY, f64::INFINITY);
        for (j, &b) in cells.iter().enumerate() {
            work.charge(1)?;
            let candidate = metric(a, b);
            if candidate.0 < score.0 || (candidate.0 == score.0 && candidate.1 < score.1) {
                best = j;
                score = candidate;
            }
        }
        // 保留比较1-IoU的舍入和阈值语义，不改成直接比较IoU。
        if score.0 < 1. - 0.1f64.powi(8) {
            out.entry(best).or_default().push(i);
        }
    }
    Ok(out)
}
fn tokens(v: &Value) -> Result<Vec<&str>, Error> {
    v.as_array()
        .filter(|a| a.len() <= rapiddoc_core::resources::limit(20_000))
        .ok_or_else(invalid)?
        .iter()
        .map(|v| v.as_str().ok_or_else(invalid))
        .collect()
}
fn html(
    tags: &[&str],
    matched: &BTreeMap<usize, Vec<usize>>,
    texts: &[String],
) -> Result<(String, Vec<String>), Error> {
    let mut budget = tags.iter().map(|s| s.len()).sum::<usize>();
    for indices in matched.values() {
        budget = budget.checked_add(16).ok_or_else(limit)?;
        for &i in indices {
            budget = budget
                .checked_add(texts.get(i).ok_or_else(invalid)?.len() + 1)
                .filter(|n| *n <= rapiddoc_core::resources::limit(4_000_000))
                .ok_or_else(limit)?;
        }
    }
    if budget > rapiddoc_core::resources::limit(4_000_000) {
        return Err(limit());
    }
    let mut chunks = Vec::new();
    let mut index = 0;
    for tag in tags {
        if !tag.contains("</td>") {
            chunks.push((*tag).to_owned());
            continue;
        }
        if *tag == "<td></td>" {
            chunks.push("<td>".into());
        }
        if let Some(indices) = matched.get(&index) {
            let bold = indices.len() > 1 && texts[indices[0]].contains("<b>");
            if bold {
                chunks.push("<b>".into());
            }
            let mut content = Vec::new();
            for &i in indices {
                let mut text = texts.get(i).ok_or_else(invalid)?.clone();
                if indices.len() > 1 {
                    if text.is_empty() {
                        continue;
                    }
                    if text.starts_with(' ') {
                        text.remove(0);
                    }
                    text = text.replace("<b>", "").replace("</b>", "");
                    text = text.trim_matches(python_space).to_owned();
                    if text.is_empty() {
                        continue;
                    }
                }
                content.push(text);
            }
            chunks.push(content.join(" "));
            if bold {
                chunks.push("</b>".into());
            }
        }
        chunks.push(if *tag == "<td></td>" {
            "</td>".into()
        } else {
            (*tag).to_owned()
        });
        index += 1;
    }
    // 旧逻辑过滤所有完整chunk，包含恰好等于这些标签的OCR内容。
    chunks.retain(|s| !matches!(s.as_str(), "<thead>" | "</thead>" | "<tbody>" | "</tbody>"));
    let size = chunks.iter().map(String::len).sum::<usize>();
    if size > rapiddoc_core::resources::limit(4_000_000) {
        return Err(limit());
    }
    Ok((chunks.concat(), chunks))
}
fn logic(tags: &[&str], work: &mut Work) -> Result<Vec<[i64; 4]>, Error> {
    let mut occupied = HashSet::new();
    let (mut row, mut col, mut i) = (0i64, 0i64, 0usize);
    let mut out = Vec::new();
    while i < tags.len() {
        let tag = tags[i];
        if tag == "<tr>" {
            col = 0;
        } else if tag == "</tr>" {
            row += 1;
        } else if tag.starts_with("<td") {
            let (mut rs, mut cs, mut j) = (1i64, 1i64, i);
            if tag != "<td></td>" {
                j += 1;
                while j < tags.len() && !tags[j].starts_with('>') {
                    let t = tags[j];
                    let slot = if t.contains("colspan=") {
                        Some(&mut cs)
                    } else if t.contains("rowspan=") {
                        Some(&mut rs)
                    } else {
                        None
                    };
                    if let Some(slot) = slot {
                        let value = t
                            .split('=')
                            .nth(1)
                            .ok_or_else(invalid)?
                            .trim_matches(['\'', '"'])
                            .trim_matches(python_space);
                        *slot = value.parse::<i64>().map_err(|_| invalid())?;
                        if slot.unsigned_abs() > rapiddoc_core::resources::limit(1000) {
                            return Err(limit());
                        }
                    }
                    j += 1;
                }
            }
            i = j;
            while occupied.contains(&(row, col)) {
                work.charge(1)?;
                col += 1;
            }
            if row.abs() > rapiddoc_core::resources::limit(1_000_000)
                || col.abs() > rapiddoc_core::resources::limit(1_000_000)
            {
                return Err(limit());
            }
            out.push([row, row + rs - 1, col, col + cs - 1]);
            if rs > 0 && cs > 0 {
                let count = (rs as usize) * (cs as usize);
                work.charge(count)?;
                if occupied.len() + count > rapiddoc_core::resources::limit(100_000) {
                    return Err(limit());
                }
                for r in row..row + rs {
                    for c in col..col + cs {
                        occupied.insert((r, c));
                    }
                }
            }
            col += cs;
        }
        i += 1;
    }
    Ok(out)
}
pub fn geometry(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    if request["dt_boxes"].as_array().is_some_and(|a| a.is_empty())
        || request["cell_bboxes"].is_null()
        || request["cell_bboxes"]
            .as_array()
            .is_some_and(|a| a.is_empty())
    {
        return Ok(json!({}));
    }
    let c = cells(&request["cell_bboxes"])?;
    let dt = request["dt_boxes"]
        .as_array()
        .filter(|a| a.len() <= rapiddoc_core::resources::limit(1000))
        .ok_or_else(invalid)?
        .iter()
        .map(coordinates)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!(matches(&c, &dt, &mut Work(0))?))
}
pub fn assemble(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let tags = tokens(&request["tokens"])?;
    let matched: BTreeMap<usize, Vec<usize>> =
        serde_json::from_value(request["matched"].clone()).map_err(|_| invalid())?;
    let texts = request["texts"]
        .as_array()
        .filter(|a| a.len() <= rapiddoc_core::resources::limit(1000))
        .ok_or_else(invalid)?
        .iter()
        .map(|v| v.as_str().map(str::to_owned).ok_or_else(invalid))
        .collect::<Result<Vec<_>, _>>()?;
    if matched
        .values()
        .any(|a| a.iter().any(|i| *i >= texts.len()))
    {
        return Err(invalid());
    }
    let (raw_html, pred) = html(&tags, &matched, &texts)?;
    Ok(json!({"raw_html":raw_html,"pred":pred,"logic_points":logic(&tags,&mut Work(0))?}))
}
pub fn run(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let tags = tokens(&request["structure"]["tokens"])?;
    let c = cells(&request["structure"]["cell_bboxes"])?;
    let width = request["width"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or_else(invalid)?;
    let height = request["height"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or_else(invalid)?;
    width
        .checked_mul(height)
        .filter(|n| *n <= rapiddoc_core::resources::limit(4_000_000))
        .ok_or_else(limit)?;
    let columns = request["ocr_result"]
        .as_array()
        .filter(|a| a.len() == 3)
        .ok_or_else(invalid)?;
    let qs = columns[0]
        .as_array()
        .filter(|a| !a.is_empty() && a.len() <= rapiddoc_core::resources::limit(1000))
        .ok_or_else(invalid)?;
    if columns[1].as_array().is_none_or(|a| a.len() != qs.len())
        || columns[2].as_array().is_none_or(|a| a.len() != qs.len())
    {
        return Err(invalid());
    }
    let y1 = c
        .iter()
        .map(|b| b[1].min(b[3]))
        .fold(f64::INFINITY, f64::min);
    let integer_coordinates = qs.iter().all(|q| {
        q.as_array().is_some_and(|q| {
            q.iter().all(|p| {
                p.as_array()
                    .is_some_and(|p| p.iter().all(|n| n.is_i64() || n.is_u64()))
            })
        })
    });
    let (mut dt, mut texts) = (Vec::new(), Vec::new());
    for (i, q) in qs.iter().enumerate() {
        let q = q.as_array().filter(|q| q.len() == 4).ok_or_else(invalid)?;
        let (mut x0, mut y0, mut x1, mut y2) = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for p in q {
            let p = p.as_array().filter(|p| p.len() == 2).ok_or_else(invalid)?;
            let b = coordinates(&json!([p[0], p[1], p[0], p[1]]))?;
            x0 = x0.min(b[0]);
            y0 = y0.min(b[1]);
            x1 = x1.max(b[0]);
            y2 = y2.max(b[1]);
        }
        let b = [
            x0.max(0.),
            y0.max(0.),
            x1.min(width as f64),
            y2.min(height as f64),
        ];
        let text = columns[1][i].as_str().ok_or_else(invalid)?;
        columns[2][i]
            .as_f64()
            .filter(|n| n.is_finite())
            .ok_or_else(invalid)?;
        if b[1].max(b[3]) < y1 {
            continue;
        }
        dt.push(b);
        texts.push(text.to_owned());
    }
    let mut work = Work(0);
    let matched = matches(&c, &dt, &mut work)?;
    let (raw_html, pred) = html(&tags, &matched, &texts)?;
    let formatted_dt = if integer_coordinates {
        json!(dt.iter().map(|b| b.map(|n| n as i64)).collect::<Vec<_>>())
    } else {
        json!(dt)
    };
    Ok(
        json!({"raw_html":raw_html,"pred":pred,"matched":matched,"dt_boxes":formatted_dt,"logic_points":logic(&tags,&mut work)?}),
    )
}
