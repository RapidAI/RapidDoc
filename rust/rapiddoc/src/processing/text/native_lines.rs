//! 旧native_text字符→视觉行：保留来源字符、排版、跨粗行上下标及稳定索引。
use super::table_text_styles::{numeric, preparation};
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{
    python_decimal,
    text_boundary::{python_digit, python_space},
    visual_text::{self, TextChar},
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, sync::OnceLock};
type Box4 = [f64; 4];
fn number(v: &Value) -> Result<f64, Error> {
    let n = v.as_f64().filter(|n| n.is_finite()).ok_or_else(invalid)?;
    if n.abs() > 1e9 {
        return Err(limit());
    }
    Ok(n)
}
fn rect(v: &Value) -> Result<Box4, Error> {
    let a = v.as_array().filter(|a| a.len() == 4).ok_or_else(invalid)?;
    Ok([
        number(&a[0])?,
        number(&a[1])?,
        number(&a[2])?,
        number(&a[3])?,
    ])
}
fn local(b: Box4, p: [f64; 2], angle: i64) -> Box4 {
    match angle {
        90 => [b[1], p[0] - b[2], b[3], p[0] - b[0]],
        180 => [p[0] - b[2], p[1] - b[3], p[0] - b[0], p[1] - b[1]],
        270 => [p[1] - b[3], b[0], p[1] - b[1], b[2]],
        _ => b,
    }
}
fn union(a: Box4, b: Box4) -> Box4 {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}
pub(super) fn defaults() -> Value {
    static D: OnceLock<Value> = OnceLock::new();
    D.get_or_init(|| serde_json::from_str(include_str!("native_line_defaults.json")).unwrap())
        .clone()
}
pub(super) fn markers(line: &mut Value) {
    let text = line["text"].as_str().unwrap().trim_matches(python_space);
    let note = (!text.is_empty() && text.chars().count() <= 3 && text.chars().all(python_digit))
        .then(|| text.to_owned());
    // ^\d+(?:\.\d+)*\.?(?:\s|$)\S ; after strip the terminal alternative cannot match.
    let cs: Vec<_> = text.chars().collect();
    let mut i = 0;
    while i < cs.len() && python_decimal(cs[i]).is_some() {
        i += 1
    }
    let has_number = i > 0;
    while i < cs.len() && cs[i] == '.' {
        i += 1;
        let start = i;
        while i < cs.len() && python_decimal(cs[i]).is_some() {
            i += 1
        }
        if i == start {
            break;
        }
    }
    let heading = has_number && i + 1 < cs.len() && python_space(cs[i]) && !python_space(cs[i + 1]);
    line["note_marker_value"] = json!(note);
    line["numbered_heading_start"] = json!(heading);
}
fn measure(line: &mut Value, p: [f64; 2], work: &mut Work) -> Result<(), Error> {
    let n = line["chars"].as_array().unwrap().len();
    work.charge(n.saturating_mul(5))?;
    let metrics = preparation::metrics(line, p)?;
    for (k, v) in metrics.as_object().unwrap() {
        line[k] = v.clone()
    }
    Ok(())
}
fn scale(line: &Value) -> f64 {
    let mut sizes: Vec<_> = line["chars"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| numeric(c["font"].get("size")))
        .filter(|n| *n > 0.)
        .collect();
    if sizes.is_empty() {
        return line["em_height"]
            .as_f64()
            .unwrap()
            .max(line["effective_height"].as_f64().unwrap())
            .max(0.1);
    }
    sizes.sort_by(f64::total_cmp);
    let n = sizes.len();
    (if n % 2 == 1 {
        sizes[n / 2]
    } else {
        (sizes[n / 2 - 1] + sizes[n / 2]) / 2.
    })
    .max(0.1)
}
fn reference(s: &str) -> bool {
    let cs: Vec<_> = s.chars().collect();
    cs.len() >= 3
        && cs.len() <= 6
        && "[（(［".contains(cs[0])
        && "]）)］".contains(*cs.last().unwrap())
        && cs[1..cs.len() - 1]
            .iter()
            .all(|c| python_decimal(*c).is_some())
}
fn merge(lines: Vec<Value>, p: [f64; 2], work: &mut Work) -> Result<Vec<Value>, Error> {
    if lines.is_empty() {
        return Ok(lines);
    }
    work.charge(lines.len().saturating_mul(lines.len()))?;
    let mut angles = BTreeMap::new();
    let mut records = Vec::new();
    for l in &lines {
        let angle = l["angle"].as_i64().unwrap();
        let n = angles.len();
        let aid = *angles.entry(angle).or_insert(n);
        let compact: String = l["text"]
            .as_str()
            .unwrap()
            .chars()
            .filter(|c| !python_space(*c))
            .collect();
        records.push((
            local(rect(&l["bbox"])?, p, angle),
            l["effective_height"].as_f64().unwrap(),
            scale(l),
            compact.chars().count(),
            reference(&compact),
            l["visual_row_id"].as_u64().unwrap() as usize,
            aid,
        ));
    }
    let pairs = visual_text::matches(records).ok_or_else(invalid)?;
    let mut positions = vec![[None; 2]; lines.len()];
    let mut detached = vec![[false; 2]; lines.len()];
    let mut consumed = vec![false; lines.len()];
    let mut bases = Vec::new();
    for (small, base, prefix, outside) in pairs {
        let i = usize::from(!prefix);
        if !bases.contains(&base) {
            bases.push(base)
        }
        positions[base][i] = Some(small);
        detached[base][i] = outside;
        consumed[small] = true;
    }
    let mut lines = lines;
    let mut state = vec![0u8; lines.len()];
    // Iterative postorder avoids stack growth on small→medium→large chains.
    for base in bases.iter().filter(|&&i| !consumed[i]).chain(bases.iter()) {
        let mut stack = vec![(*base, false)];
        while let Some((i, finish)) = stack.pop() {
            if state[i] == 2 {
                continue;
            }
            if !finish {
                if state[i] == 1 {
                    continue;
                }
                state[i] = 1;
                stack.push((i, true));
                for j in positions[i].iter().flatten().rev() {
                    if state[*j] == 0 {
                        stack.push((*j, false))
                    }
                }
                continue;
            }
            let mut b = rect(&lines[i]["bbox"])?;
            let mut text = lines[i]["text"].as_str().unwrap().to_owned();
            let mut chars = lines[i]["chars"].as_array().unwrap().clone();
            let mut split = lines[i]["split_from_row"].as_bool().unwrap();
            let mut formula = lines[i]["formula_candidate_only"].as_bool().unwrap();
            let mut restored = lines[i]["restored_inline_cluster"].as_bool().unwrap();
            for (position, j) in positions[i]
                .iter()
                .enumerate()
                .filter_map(|(p, i)| i.map(|i| (p, i)))
            {
                let child = &lines[j];
                let child_text = child["text"].as_str().unwrap().trim_matches(python_space);
                if position == 0 {
                    text = format!("{} {}", child_text, text.trim_start_matches(python_space));
                    let mut selected = child["chars"].as_array().unwrap().clone();
                    selected.extend(chars);
                    chars = selected;
                } else {
                    text = format!("{}{}", text.trim_end_matches(python_space), child_text);
                    chars.extend(child["chars"].as_array().unwrap().iter().cloned());
                }
                b = union(b, rect(&child["bbox"])?);
                split |= child["split_from_row"].as_bool().unwrap();
                formula &= child["formula_candidate_only"].as_bool().unwrap();
                restored |=
                    child["restored_inline_cluster"].as_bool().unwrap() || detached[i][position];
            }
            let l = &mut lines[i];
            l["text"] = json!(text);
            l["bbox"] = json!(b);
            l["chars"] = Value::Array(chars);
            l["split_from_row"] = json!(split);
            l["formula_candidate_only"] = json!(formula);
            l["restored_inline_cluster"] = json!(restored);
            measure(l, p, work)?;
            state[i] = 2;
        }
    }
    let mut out: Vec<_> = lines
        .into_iter()
        .enumerate()
        .filter_map(|(i, l)| (!consumed[i]).then_some(l))
        .collect();
    out.sort_by_key(|l| {
        (
            l["visual_row_id"].as_u64().unwrap(),
            l["run_index"].as_u64().unwrap(),
        )
    });
    Ok(out)
}
pub fn build(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    build_chars_impl(&v["chars"], v)
}
pub(super) fn build_chars(raw_chars: &Value, v: &Value) -> Result<Value, Error> {
    let limits = crate::legacy_page::budget::current();
    if limits.nodes != usize::MAX
        || limits.bytes != usize::MAX
        || rapiddoc_core::resources::limit(64usize) != usize::MAX
    {
        // Retain exact outer-tree budget accounting when explicitly configured.
        // Production's unlimited host scope can borrow characters directly.
        let mut request = v.clone();
        request
            .as_object_mut()
            .ok_or_else(invalid)?
            .insert("chars".into(), raw_chars.clone());
        return build(&request);
    }
    build_chars_impl(raw_chars, v)
}
fn build_chars_impl(raw_chars: &Value, v: &Value) -> Result<Value, Error> {
    let chars = raw_chars.as_array().ok_or_else(invalid)?;
    if chars.len() > crate::legacy_page::budget::records(4096) {
        return Err(limit());
    }
    let a = v["page_size"]
        .as_array()
        .filter(|a| a.len() == 2)
        .ok_or_else(invalid)?;
    let p = [number(&a[0])?, number(&a[1])?];
    if p.iter().any(|n| *n <= 0.) {
        return Err(invalid());
    }
    let rotation = v.get("page_rotation").map_or(Ok(0), |v| {
        v.as_i64()
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(invalid)
    })?;
    let angles = if let Some(a) = v.get("supported_angles") {
        let a = a
            .as_array()
            .filter(|a| !a.is_empty() && a.len() <= 16)
            .ok_or_else(invalid)?;
        a.iter().map(number).collect::<Result<Vec<_>, _>>()?
    } else {
        vec![0., 90., 270.]
    };
    let mut work = Work(0);
    work.charge(chars.len().saturating_mul(chars.len() + 1) / 2)?;
    let mut input = Vec::new();
    let mut font_id = 0;
    for (i, c) in chars.iter().enumerate() {
        let text = c["char"]
            .as_str()
            .filter(|s| s.len() <= rapiddoc_core::resources::limit(256))
            .ok_or_else(invalid)?;
        if !c["font"].is_object() {
            return Err(invalid());
        }
        if i > 0 && !eq(&chars[i - 1]["font"], &c["font"]) {
            font_id += 1
        }
        input.push(TextChar {
            bbox: rect(&c["bbox"])?,
            text: text.to_owned(),
            rotation: number(&c["rotation"])?,
            font_id,
            font_size: numeric(c["font"].get("size")),
            signature_id: None,
            font_weight: None,
        });
    }
    let runs = visual_text::prepare(&input, p, rotation, &angles);
    let mut normal = Vec::new();
    let mut formula = Vec::new();
    for run in runs {
        let mut l = defaults();
        l["text"] = json!(run.text);
        l["bbox"] = json!(run.bbox);
        l["source_bbox"] = json!(run.bbox);
        l["angle"] = json!(run.angle);
        l["source_index"] = json!(-1);
        l["chars"] = Value::Array(
            run.indices
                .iter()
                .map(|i| chars[*i].clone())
                .collect::<Vec<_>>(),
        );
        l["visual_row_id"] = json!(run.row_id);
        l["run_index"] = json!(run.run_index);
        l["split_from_row"] = json!(run.split);
        l["formula_candidate_only"] = json!(run.formula);
        markers(&mut l);
        if run.coarse_fallback {
            let b = local(run.bbox, p, run.angle.into());
            l["effective_height"] = json!((b[3] - b[1]).max(0.1));
        } else {
            measure(&mut l, p, &mut work)?
        }
        if run.formula {
            formula.push(l)
        } else {
            normal.push(l)
        }
    }
    let mut normal = merge(normal, p, &mut work)?;
    let formula = merge(formula, p, &mut work)?;
    normal.extend(formula);
    for (i, l) in normal.iter_mut().enumerate() {
        l["source_index"] = json!(i)
    }
    normal.sort_by_key(|l| {
        (
            l["visual_row_id"].as_u64().unwrap(),
            l["run_index"].as_u64().unwrap(),
        )
    });
    Ok(Value::Array(normal))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn same(a: &Value, b: &Value, path: &str) {
        match (a, b) {
            (Value::Array(a), Value::Array(b)) => {
                assert_eq!(a.len(), b.len(), "{path}/length");
                for (i, (a, b)) in a.iter().zip(b).enumerate() {
                    same(a, b, &format!("{path}/{i}"))
                }
            }
            (Value::Object(a), Value::Object(b)) => {
                assert_eq!(
                    a.keys().collect::<Vec<_>>(),
                    b.keys().collect::<Vec<_>>(),
                    "{path}/keys"
                );
                for (k, a) in a {
                    same(a, &b[k], &format!("{path}/{k}"))
                }
            }
            _ => assert_eq!(a, b, "{path}"),
        }
    }
    #[test]
    fn old_complete_visual_lines() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-visual-lines.json");
        let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-visual-lines-extra.json");
        let g: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in f["cases"]
            .as_array()
            .unwrap()
            .iter()
            .chain(g["cases"].as_array().unwrap())
        {
            let result = build(&c["request"]).unwrap_or_else(|e| panic!("{}: {e:?}", c["label"]));
            same(&result, &c["expected"], c["label"].as_str().unwrap());
        }
    }
}
