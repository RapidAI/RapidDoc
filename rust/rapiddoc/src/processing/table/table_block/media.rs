//! 旧表内媒体几何、原语遮罩和唯一物理单元格归属；PNG编码另行处理。
use super::*;
use crate::legacy_model::{table_native_text as text, table_native_vector};
use crate::legacy_page::truth;
fn area(b: Box4) -> f64 {
    (b[2] - b[0]).max(0.) * (b[3] - b[1]).max(0.)
}
pub(super) fn overlap_first(a: Box4, b: Box4) -> f64 {
    let denominator = area(a);
    if denominator <= 0. {
        0.
    } else {
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.)
            / denominator
    }
}
fn expanded(b: Box4) -> Box4 {
    [b[0] - 0.5, b[1] - 0.5, b[2] + 0.5, b[3] + 0.5]
}
fn boxes<'a>(source: &'a Value, k: &str) -> Result<&'a [Value], Error> {
    match source.get(k) {
        None | Some(Value::Null) => Ok(&[]),
        Some(v) => Ok(array(v, 5000)?.as_slice()),
    }
}
pub(super) fn select(source: &Value, table: Box4, work: &mut Work) -> Result<Vec<Value>, Error> {
    let mut values = Vec::new();
    for b in boxes(source, "form_bboxes")? {
        let bbox = rect(b)?;
        let mut count = 0;
        let mut complex = false;
        for p in boxes(source, "path_infos")? {
            work.charge(1)?;
            if (truth(&p["fill_visible"]) || truth(&p["stroke_visible"]))
                && p["segment_count"].as_i64().unwrap_or(0) >= 4
                && overlap_first(rect(&p["bbox"])?, bbox) >= 0.99
            {
                count += 1;
                complex |= p["segment_count"].as_i64().unwrap_or(0) >= 6;
            }
        }
        if count >= 2 || complex {
            values.push(b.clone());
        }
    }
    values.extend(boxes(source, "image_bboxes")?.iter().cloned());
    let mut filtered = Vec::new();
    for b in values {
        work.charge(1)?;
        let q = rect(&b)?;
        if area(q) > 0. && area(q) <= 0.25 * area(table) && overlap_first(q, table) >= 0.99 {
            filtered.push(b);
        }
    }
    filtered.sort_by(|a, b| {
        area(rect(b).unwrap())
            .partial_cmp(&area(rect(a).unwrap()))
            .unwrap()
    });
    let mut selected: Vec<Value> = Vec::new();
    for b in filtered {
        let bb = rect(&b)?;
        let mut inside = false;
        for p in &selected {
            work.charge(1)?;
            if overlap_first(bb, rect(p)?) >= 0.99 {
                inside = true;
                break;
            }
        }
        if !inside {
            selected.push(b);
        }
    }
    selected.sort_by(|a, b| {
        a[1].as_f64()
            .partial_cmp(&b[1].as_f64())
            .unwrap()
            .then(a[0].as_f64().partial_cmp(&b[0].as_f64()).unwrap())
    });
    Ok(selected)
}
pub(super) fn mask(input: &Value, boxes: &[Value], work: &mut Work) -> Result<Value, Error> {
    let masks = boxes
        .iter()
        .map(|b| rect(b).map(expanded))
        .collect::<Result<Vec<_>, _>>()?;
    let mut output = input.clone();
    for k in ["drawing_lines", "rectangles", "chars"] {
        let mut retained = Vec::new();
        for record in array(&input[k], 20000)? {
            let b = if k == "chars" && truth(&record["tight_bbox"]) {
                &record["tight_bbox"]
            } else {
                &record["bbox"]
            };
            let b = text::normalize_bbox(b)?;
            let mut inside = false;
            if let Some(b) = b {
                for m in &masks {
                    work.charge(1)?;
                    if overlap_first(b, *m) >= 0.99 {
                        inside = true;
                        break;
                    }
                }
            }
            if !inside {
                retained.push(record.clone());
            }
        }
        output[k] = json!(retained);
    }
    Ok(output)
}
pub(super) fn attach(
    content: &str,
    result: &Value,
    input: &Value,
    boxes: &[Value],
    work: &mut Work,
) -> Result<Option<(String, Vec<Value>)>, Error> {
    if result["source"] != "vector_grid" {
        return Ok(None);
    }
    let mut cells = array(&result["cells"], 2000)?.iter().collect::<Vec<_>>();
    for c in &cells {
        integer(&c["row"])?;
        integer(&c["col"])?;
        rect(&c["bbox"])?;
    }
    cells.sort_by_key(|c| (c["row"].as_i64().unwrap(), c["col"].as_i64().unwrap()));
    let mut images = Vec::new();
    let mut by_cell = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    let angle = text::angle(&input["angle"])?;
    let Some(table) = text::normalize_bbox(&input["table_bbox"])? else {
        return Ok(None);
    };
    for box_value in boxes {
        let bbox = rect(box_value)?;
        let local = table_native_vector::local(bbox, table, i64::from(angle), 0.);
        let mut matches = Vec::new();
        for (i, c) in cells.iter().enumerate() {
            work.charge(1)?;
            if local.is_some_and(|b| overlap_first(b, expanded(rect(&c["bbox"]).unwrap())) >= 0.99)
            {
                matches.push(i);
            }
        }
        if matches.len() != 1 {
            return Ok(None);
        }
        let mut fallback = String::new();
        for c in array(&input["chars"], 20000)? {
            work.charge(1)?;
            let b = if truth(&c["tight_bbox"]) {
                &c["tight_bbox"]
            } else {
                &c["bbox"]
            };
            if text::normalize_bbox(b)?.is_some_and(|b| overlap_first(b, expanded(bbox)) >= 0.99) {
                fallback.push_str(
                    &crate::legacy_model::table_native_multiline::spatial::raw_string(
                        c.get("char"),
                        false,
                    ),
                );
            }
        }
        let index = images.len();
        images.push(json!({"bbox":box_value,"text":fallback}));
        by_cell.entry(matches[0]).or_default().push(index);
    }
    // 复刻旧(<td\b[^>]*>)(.*?)</td>的顺序插入；不解析或重写已恢复HTML。
    let mut output = String::new();
    let mut offset = 0;
    let mut cell_index = 0;
    while let Some(start) = content[offset..].find("<td").map(|i| i + offset) {
        let before = output.len();
        let after = start + 3;
        if content[after..]
            .chars()
            .next()
            .is_some_and(|c| python_alnum(c) || c == '_')
        {
            output.push_str(&content[offset..after]);
            offset = after;
            continue;
        }
        let Some(open_end) = content[after..].find('>').map(|i| i + after + 1) else {
            break;
        };
        let Some(close) = content[open_end..].find("</td>").map(|i| i + open_end) else {
            break;
        };
        output.push_str(&content[offset..open_end]);
        if let Some(indices) = by_cell.get(&cell_index) {
            for index in indices {
                output.push_str(&format!(
                    "<img src=\"docvortex-table-image:{index}\" alt=\"Image\" />"
                ));
            }
        }
        output.push_str(&content[open_end..close + 5]);
        offset = close + 5;
        cell_index += 1;
        work.charge(output.len() - before)?;
        if output.len() > rapiddoc_core::resources::limit(4_000_000) {
            return Err(limit());
        }
    }
    output.push_str(&content[offset..]);
    if output.len() > rapiddoc_core::resources::limit(4_000_000) {
        return Err(limit());
    }
    Ok((cell_index == cells.len()).then_some((output, images)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_media_selection_mask_and_attachment() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-table-media.json");
        let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let mut work = Work(0);
            let selected =
                select(&c["source"], rect(&c["table_bbox"]).unwrap(), &mut work).unwrap();
            assert_eq!(json!(selected), c["selected"], "select {}", c["label"]);
            assert_eq!(
                mask(&c["table_input"], &selected, &mut work).unwrap(),
                c["masked"],
                "mask {}",
                c["label"]
            );
            let attached = attach(
                c["content"].as_str().unwrap(),
                &c["result"],
                &c["table_input"],
                &selected,
                &mut work,
            )
            .unwrap()
            .map(|(content, images)| json!({"content":content,"images":images}));
            assert_eq!(json!(attached), c["attached"], "attach {}", c["label"]);
        }
    }
}
