use super::*;
use rapiddoc_native::{python_printable, text_boundary::python_space, visual_text};
fn local(b: [f64; 4], p: [f64; 2], a: i64) -> [f64; 4] {
    match a {
        90 => [b[1], p[0] - b[2], b[3], p[0] - b[0]],
        180 => [p[0] - b[2], p[1] - b[3], p[0] - b[0], p[1] - b[1]],
        270 => [p[1] - b[3], b[0], p[1] - b[1], b[2]],
        _ => b,
    }
}
fn world(b: [f64; 4], p: [f64; 2], a: i64) -> [f64; 4] {
    match a {
        90 => [p[0] - b[3], b[0], p[0] - b[1], b[2]],
        180 => [p[0] - b[2], p[1] - b[3], p[0] - b[0], p[1] - b[1]],
        270 => [b[1], p[1] - b[2], b[3], p[1] - b[0]],
        _ => b,
    }
}
fn text(raw: &str) -> String {
    let raw = super::super::code_containers::text::sanitize(raw).replace('\n', "");
    let mut out = String::new();
    let mut space = false;
    for c in raw.chars() {
        if c == ' ' {
            if !space {
                out.push(c)
            }
            space = true
        } else {
            out.push(c);
            space = false
        }
    }
    out.trim_matches(python_space).into()
}
fn split(
    line: &mut Value,
    p: [f64; 2],
    visual: &BTreeMap<i64, Value>,
    w: &mut Work,
) -> Result<Vec<(Value, Vec<usize>)>, Error> {
    let chars = optional(line, "chars", 20000)?;
    let char_count = chars.len();
    let a = integer(line, "angle", 0)?;
    let angle = i32::try_from(a).map_err(|_| invalid())?;
    let mut tokens = Vec::new();
    let mut raw = Vec::new();
    let mut overrides = Vec::new();
    let mut flags = Vec::new();
    for (i, c) in chars.iter().enumerate() {
        let text = super::super::table_native_multiline::spatial::raw_string(c.get("char"), true);
        w.charge(text.len() + 1)?;
        if text == "\r" || text == "\n" {
            continue;
        }
        let visible = !text.is_empty()
            && text.chars().all(python_printable)
            && !text.chars().all(python_space);
        let space = !text.is_empty() && text.chars().all(python_space);
        flags.push(u8::from(visible) | (u8::from(space) << 1));
        raw.push(coerce(&c["bbox"])?.map(|b| b.n));
        overrides.push(if visible {
            c["char_idx"]
                .as_i64()
                .or_else(|| c["char_idx"].as_bool().map(i64::from))
                .and_then(|i| visual.get(&i))
                .map(coerce)
                .transpose()?
                .flatten()
                .map(|b| b.n)
        } else {
            None
        });
        tokens.push((i, c.clone(), text));
    }
    w.charge(tokens.len().saturating_mul(20))?;
    let ranges = visual_text::split_runs(raw, overrides, flags, p, angle).ok_or_else(limit)?;
    if ranges.is_empty() {
        let t = text(line["text"].as_str().ok_or_else(invalid)?);
        if t.is_empty() {
            return Ok(Vec::new());
        }
        line["text"] = json!(t);
        let b = local(bbox(&line["bbox"])?.n, p, a);
        line["effective_height"] = json!((b[3] - b[1]).max(0.1));
        return Ok(vec![(line.clone(), (0..char_count).collect())]);
    }
    let count = ranges.len();
    let vertical = local(bbox(&line["bbox"])?.n, p, a);
    let mut out = Vec::new();
    for (run, (start, end, b)) in ranges.into_iter().enumerate() {
        let raw = tokens[start..end]
            .iter()
            .map(|(_, _, t)| t.as_str())
            .collect::<String>();
        let t = text(&raw);
        let Some(b) = b.filter(|_| !t.is_empty()) else {
            continue;
        };
        let b = local(b, p, a);
        let b = world([b[0], vertical[1], b[2], vertical[3]], p, a);
        let mut item = native_lines::defaults();
        item["text"] = json!(t);
        item["bbox"] = json!(b);
        item["source_bbox"] = json!(b);
        item["angle"] = json!(a);
        item["source_index"] = json!(-1);
        item["chars"] = json!(tokens[start..end]
            .iter()
            .map(|(_, c, _)| c)
            .collect::<Vec<_>>());
        item["visual_row_id"] = line["visual_row_id"].clone();
        item["run_index"] = json!(run);
        item["split_from_row"] = json!(count > 1);
        item["formula_candidate_only"] =
            json!(line["formula_candidate_only"].as_bool().unwrap_or(false));
        native_lines::markers(&mut item);
        let metrics = preparation::metrics(&item, p)?;
        for (k, v) in metrics.as_object().unwrap() {
            item[k] = v.clone()
        }
        out.push((
            item,
            tokens[start..end].iter().map(|(i, _, _)| *i).collect(),
        ));
    }
    Ok(out)
}
pub(super) fn resplit(
    v: &Value,
    w: &mut Work,
) -> Result<(Vec<Value>, BTreeMap<i64, Resplit>, Vec<Value>), Error> {
    let lines = array(&v["lines"], 4096)?;
    let p = size(&v["page_size"])?;
    let mut visual = BTreeMap::new();
    for r in optional(v, "visual_bboxes", 20000)? {
        visual.insert(
            r["char_idx"].as_i64().ok_or_else(invalid)?,
            r["bbox"].clone(),
        );
    }
    if lines.is_empty() || visual.is_empty() {
        return Ok((lines.to_vec(), BTreeMap::new(), lines.to_vec()));
    }
    let mut next = if !v["source_index_start"].is_null() {
        v["source_index_start"].as_i64().ok_or_else(invalid)?
    } else {
        lines
            .iter()
            .map(|l| integer(l, "source_index", 0))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(-1)
            .checked_add(1)
            .ok_or_else(limit)?
    };
    let mut out = Vec::new();
    let mut resplits = BTreeMap::new();
    let mut source_lines = Vec::new();
    for line in lines {
        let mut line = line.clone();
        let a = integer(&line, "angle", 0)?;
        let b = local(bbox(&line["bbox"])?.n, p, a);
        let (width, height) = if [90, 270].contains(&a) {
            (p[1], p[0])
        } else {
            (p[0], p[1])
        };
        let cy = (b[1] + b[3]) * 0.5;
        if !line["semantic_type"].is_null()
            || b[2] - b[0] < 0.45 * width
            || cy < 0.07 * height
            || cy > 0.93 * height
        {
            source_lines.push(line.clone());
            out.push(line);
            continue;
        }
        let mut members = split(&mut line, p, &visual, w)?;
        source_lines.push(line.clone());
        if members.len() <= 1 {
            out.push(line);
            continue;
        }
        let mut boxes = members
            .iter()
            .map(|(l, _)| Ok(local(bbox(&l["bbox"])?.n, p, a)))
            .collect::<Result<Vec<_>, Error>>()?;
        boxes.sort_by(|a, b| a[0].total_cmp(&b[0]));
        if boxes.len() != 2
            || boxes.iter().any(|b| b[2] - b[0] < 0.15 * width)
            || boxes[1][0] - boxes[0][2] < 0.02 * width
            || boxes[0][2] > 0.52 * width
            || boxes[1][0] < 0.48 * width
        {
            out.push(line);
            continue;
        }
        let id = integer(&line, "source_index", 0)?;
        for (i, (member, _)) in members.iter_mut().enumerate() {
            member["source_index"] = if i == 0 {
                json!(id)
            } else {
                let id = next;
                next = next.checked_add(1).ok_or_else(limit)?;
                json!(id)
            };
            for key in [
                "visual_row_id",
                "baseline",
                "geometry_state",
                "geometry_confidence",
                "split_y_candidate",
                "preserve_split_boundary",
                "semantic_type",
                "formula_candidate_only",
                "style_scale_repaired",
            ] {
                if let Some(value) = line.get(key) {
                    member[key] = value.clone()
                }
            }
            member["run_index"] = json!(integer(&line, "run_index", 0)?
                .checked_add(i as i64)
                .ok_or_else(limit)?);
            member["source_bbox"] = member["bbox"].clone();
            member["em_height"] = line
                .get("em_height")
                .filter(|h| h.as_f64().is_some_and(|h| h != 0.))
                .cloned()
                .unwrap_or(member["effective_height"].clone());
            member["split_from_row"] = json!(true);
            out.push(member.clone());
        }
        resplits.insert(
            id,
            Resplit {
                source: line,
                members: members.iter().map(|(l, _)| l.clone()).collect(),
                positions: members.into_iter().map(|(_, p)| p).collect(),
            },
        );
    }
    for l in &out {
        bbox(&l["bbox"])?;
        integer(l, "source_index", 0)?;
        integer(l, "run_index", 0)?;
        if !l["visual_row_id"].is_null() {
            l["visual_row_id"].as_i64().ok_or_else(invalid)?;
        }
    }
    out.sort_by(|a, b| {
        let row = |l: &Value| l["visual_row_id"].as_i64();
        match (row(a), row(b)) {
            (Some(a), Some(b)) => a.cmp(&b),
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (Some(_), None) => std::cmp::Ordering::Less,
            _ => std::cmp::Ordering::Equal,
        }
        .then(
            integer(a, "run_index", 0)
                .unwrap()
                .cmp(&integer(b, "run_index", 0).unwrap()),
        )
        .then(
            integer(a, "source_index", 0)
                .unwrap()
                .cmp(&integer(b, "source_index", 0).unwrap()),
        )
    });
    Ok((out, resplits, source_lines))
}
