//! 旧 PDF 表格块准备：字符框修复、嵌入脚注分离和投影成员扩展。
//! 不提交输出块或认领；failed_annotations 是后续注释物化阶段的显式输入。
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::{python_alnum, python_space};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
type Box4 = [f64; 4];
pub mod commit;
pub mod images;
pub mod media;
pub mod output;

fn array(v: &Value, maximum: usize) -> Result<&Vec<Value>, Error> {
    let a = v.as_array().ok_or_else(invalid)?;
    let maximum = if maximum > 8 {
        crate::legacy_page::budget::records(maximum)
    } else {
        maximum
    };
    if a.len() > maximum {
        return Err(limit());
    }
    Ok(a)
}
fn number(v: &Value) -> Result<f64, Error> {
    v.as_f64()
        .filter(|n| n.is_finite() && n.abs() <= 1e9)
        .ok_or_else(invalid)
}
fn rect(v: &Value) -> Result<Box4, Error> {
    let b = candidate_rect(v)?;
    if b[2] < b[0] || b[3] < b[1] {
        return Err(invalid());
    }
    Ok(b)
}
fn candidate_rect(v: &Value) -> Result<Box4, Error> {
    let a = array(v, 4)?;
    if a.len() != 4 {
        return Err(invalid());
    }
    let b = [
        number(&a[0])?,
        number(&a[1])?,
        number(&a[2])?,
        number(&a[3])?,
    ];
    Ok(b)
}
fn candidate_box(v: &Value) -> Result<Option<Box4>, Error> {
    if v.is_null() {
        Ok(None)
    } else {
        candidate_rect(v).map(Some)
    }
}
fn integer(v: &Value) -> Result<i64, Error> {
    v.as_i64()
        .or_else(|| v.as_bool().map(i64::from))
        .filter(|n| n.unsigned_abs() <= rapiddoc_core::resources::limit(1_000_000_000_000))
        .ok_or_else(invalid)
}
fn ids(v: &Value) -> Result<BTreeSet<i64>, Error> {
    array(v, 20000)?.iter().map(integer).collect()
}
fn character_key(v: &Value) -> Result<String, Error> {
    if let Ok(id) = integer(v) {
        return Ok(id.to_string());
    }
    let n = v
        .as_f64()
        .filter(|n| n.is_finite() && n.abs() <= 1_000_000_000_000.)
        .ok_or_else(invalid)?;
    Ok(if n.fract() == 0. {
        (n as i64).to_string()
    } else {
        v.to_string()
    })
}
fn optional_box(v: &Value) -> Result<Option<Box4>, Error> {
    if v.is_null() {
        Ok(None)
    } else {
        rect(v).map(Some)
    }
}
fn union_value(a: &Value, b: &Value) -> Value {
    json!((0..4)
        .map(|i| {
            let x = a[i].as_f64().unwrap();
            let y = b[i].as_f64().unwrap();
            if (i < 2 && y < x) || (i >= 2 && y > x) {
                b[i].clone()
            } else {
                a[i].clone()
            }
        })
        .collect::<Vec<_>>())
}
fn center(b: Box4) -> (f64, f64) {
    ((b[0] + b[2]) / 2., (b[1] + b[3]) / 2.)
}
fn inside(p: (f64, f64), b: Box4) -> bool {
    b[0] <= p.0 && p.0 <= b[2] && b[1] <= p.1 && p.1 <= b[3]
}
fn rotate(b: Box4, p: [f64; 2], a: i64) -> Box4 {
    match a {
        90 => [b[1], p[0] - b[2], b[3], p[0] - b[0]],
        180 => [p[0] - b[2], p[1] - b[3], p[0] - b[0], p[1] - b[1]],
        270 => [p[1] - b[3], b[0], p[1] - b[1], b[2]],
        _ => b,
    }
}
fn overlap(a: Box4, b: Box4) -> f64 {
    let shorter = (a[2] - a[0]).min(b[2] - b[0]);
    if shorter <= 0. {
        0.
    } else {
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.) / shorter
    }
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let i = values.len() / 2;
    if values.len() % 2 == 1 {
        values[i]
    } else {
        (values[i - 1] + values[i]) / 2.
    }
}
fn flag(v: &Value, key: &str) -> Result<bool, Error> {
    v.get(key)
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))
}
struct Line<'a> {
    value: &'a Value,
    bbox: Box4,
    source: i64,
    angle: i64,
}
fn height(l: &Line<'_>, b: Box4) -> Result<f64, Error> {
    let h = l.value.get("effective_height").map_or(Ok(0.), number)?;
    let em = l.value.get("em_height").map_or(Ok(0.), number)?;
    Ok(
        0.1f64.max(if flag(l.value, "style_scale_repaired")? && em > 0. {
            em
        } else if h != 0. {
            h
        } else {
            b[3] - b[1]
        }),
    )
}
fn same_baseline(a: Box4, ah: f64, b: Box4, bh: f64) -> bool {
    let h = ah.max(bh);
    if ah.min(bh) <= 0. || h / ah.min(bh) > 1.35 {
        return false;
    }
    let y = (a[3].min(b[3]) - a[1].max(b[1])).max(0.);
    if y / (a[3] - a[1]).min(b[3] - b[1]).max(0.1) < 0.7 && (a[3] - b[3]).abs() > 0.25 * h {
        return false;
    }
    let gap = if a[0] <= b[0] {
        b[0] - a[2]
    } else {
        a[0] - b[2]
    };
    gap >= -0.25 * h && gap <= 3.0f64.max(0.75 * h)
}
pub(super) fn note(text: &str) -> bool {
    let text = text.trim_matches(python_space);
    // Python re.IGNORECASE 额外将这四种字符视作 ASCII I/S/K。
    let lower: String = text
        .chars()
        .flat_map(|c| match c {
            'İ' | 'ı' => 'i'.to_lowercase(),
            'ſ' => 's'.to_lowercase(),
            'K' => 'k'.to_lowercase(),
            _ => c.to_lowercase(),
        })
        .collect();
    if ["注", "说明"].iter().any(|prefix| text.starts_with(prefix)) {
        return true;
    }
    if ["note", "notes", "source", "sources"].iter().any(|p| {
        lower.strip_prefix(p).is_some_and(|s| {
            s.chars()
                .next()
                .is_none_or(|c| !python_alnum(c) && c != '_')
        })
    }) {
        return true;
    }
    let rest = if let Some(rest) = lower.strip_prefix("for") {
        let stripped = rest.trim_start_matches(python_space);
        if stripped.len() == rest.len() {
            return false;
        }
        stripped
    } else {
        text
    };
    if let Some(rest) = rest.strip_prefix(['*', '†', '‡']) {
        // `for\s+[*†‡]` 不要求标记之后还有正文。
        if lower.starts_with("for") {
            return true;
        }
        return !rest.trim_start_matches(python_space).is_empty();
    }
    false
}
fn annotations(candidate: &Value) -> Result<&Vec<Value>, Error> {
    array(&candidate["annotations"], 2000)
}
fn validate_annotation(v: &Value) -> Result<(), Error> {
    if !matches!(v["kind"].as_str(), Some("caption" | "footnote")) {
        return Err(invalid());
    }
    rect(&v["bbox"])?;
    ids(&v["line_indices"])?;
    let m = v["line_bboxes"].as_object().ok_or_else(invalid)?;
    if m.len() > rapiddoc_core::resources::limit(20000) {
        return Err(limit());
    }
    for (k, b) in m {
        k.parse::<i64>().map_err(|_| invalid())?;
        rect(b)?;
    }
    Ok(())
}
fn separate(
    lines: &[Line<'_>],
    rules: &[Value],
    candidate: &mut Value,
    work: &mut Work,
) -> Result<(), Error> {
    if integer(&candidate["angle"])? != 0
        || annotations(candidate)?
            .iter()
            .any(|a| a["kind"] == "footnote")
    {
        return Ok(());
    }
    let raw_core = if candidate["core_bbox"].is_null() {
        candidate["bbox"].clone()
    } else {
        candidate["core_bbox"].clone()
    };
    let core = candidate_rect(&raw_core)?;
    let mut members = Vec::new();
    for l in lines {
        work.charge(1)?;
        if core[0] - 2. <= l.bbox[0]
            && l.bbox[2] <= core[2] + 2.
            && core[1] <= center(l.bbox).1
            && center(l.bbox).1 <= core[3]
        {
            members.push(l);
        }
    }
    let mut ys = Vec::new();
    for r in rules {
        work.charge(1)?;
        let b = candidate_rect(&r["bbox"])?;
        if r["orientation"] == "horizontal"
            && overlap(b, core) >= 0.8
            && core[1] <= b[1]
            && b[1] <= core[3]
        {
            ys.push(r["bbox"][1].clone());
        }
    }
    ys.sort_by(|a, b| {
        a.as_f64()
            .unwrap()
            .partial_cmp(&b.as_f64().unwrap())
            .unwrap()
    });
    let mut ordered = members.clone();
    ordered.sort_by(|a, b| a.bbox[1].partial_cmp(&b.bbox[1]).unwrap());
    for first in ordered {
        work.charge(1)?;
        if !note(first.value["text"].as_str().ok_or_else(invalid)?)
            || first.bbox[2] - first.bbox[0] < 0.6 * (core[2] - core[0])
        {
            continue;
        }
        let em = height(first, first.bbox)?;
        work.charge(ys.len() + members.len())?;
        let above: Vec<_> = ys
            .iter()
            .filter(|y| y.as_f64().unwrap() <= first.bbox[1] + 0.1 * em)
            .collect();
        if above.len() < 3 || first.bbox[1] - above[above.len() - 1].as_f64().unwrap() > 1.5 * em {
            continue;
        }
        let notes: Vec<_> = members
            .iter()
            .filter(|l| l.bbox[1] >= first.bbox[1] - 0.2 * em)
            .collect();
        if notes.len() < 2
            || ys
                .iter()
                .any(|y| y.as_f64().unwrap() > first.bbox[3] && y.as_f64().unwrap() < core[3] - em)
            || notes.iter().any(|l| (l.bbox[0] - first.bbox[0]).abs() > em)
        {
            continue;
        }
        let boxes: Map<_, _> = notes
            .iter()
            .map(|l| (l.source.to_string(), l.value["bbox"].clone()))
            .collect();
        let selected: BTreeSet<_> = notes.iter().map(|l| l.source).collect();
        let bbox = notes
            .iter()
            .map(|l| l.value["bbox"].clone())
            .reduce(|a, b| union_value(&a, &b))
            .unwrap();
        candidate["annotations"].as_array_mut().unwrap().push(
            json!({"kind":"footnote","bbox":bbox,"line_indices":selected,"line_bboxes":boxes}),
        );
        candidate["core_bbox"] = json!([
            raw_core[0],
            raw_core[1],
            raw_core[2],
            above[above.len() - 1]
        ]);
        let mut members = ids(&candidate["line_indices"])?;
        members.retain(|id| !selected.contains(id));
        candidate["line_indices"] = json!(members);
        return Ok(());
    }
    Ok(())
}
fn projection(
    lines: &[Line<'_>],
    candidate: &Value,
    page: [f64; 2],
    work: &mut Work,
) -> Result<BTreeSet<i64>, Error> {
    let mut selected = ids(&candidate["line_indices"])?;
    let core = candidate_box(&candidate["core_bbox"])?;
    let outer = candidate_rect(&candidate["bbox"])?;
    let angle = integer(&candidate["angle"])?;
    for l in lines {
        work.charge(1)?;
        if core.is_some_and(|c| inside(center(l.bbox), c)) {
            selected.insert(l.source);
        }
    }
    if angle != 0 {
        if let Some(core) = core {
            let outer = rotate(outer, page, angle);
            let core = rotate(core, page, angle);
            for l in lines {
                work.charge(1)?;
                if l.angle != angle || !inside(center(l.bbox), candidate_rect(&candidate["bbox"])?)
                {
                    continue;
                }
                let b = rotate(l.bbox, page, angle);
                let y = center(b).1;
                if outer[1] <= y && y <= outer[3] && overlap(b, core) >= 0.05 {
                    selected.insert(l.source);
                }
            }
        }
        return Ok(selected);
    }
    let mut changed = true;
    while changed {
        changed = false;
        work.charge(lines.len())?;
        // 本轮不能用新加入的成员提前延长链；与旧 selected_lines 快照一致。
        let previous: Vec<_> = lines
            .iter()
            .filter(|l| l.angle == angle && selected.contains(&l.source))
            .collect();
        for l in lines {
            work.charge(1)?;
            if l.angle != angle || selected.contains(&l.source) || !inside(center(l.bbox), outer) {
                continue;
            }
            let b = rotate(l.bbox, page, angle);
            let h = height(l, b)?;
            for p in &previous {
                work.charge(1)?;
                let a = &l.value["font_signature"];
                let z = &p.value["font_signature"];
                if !a.is_null() && !z.is_null() && !eq(a, z) {
                    continue;
                }
                let pb = rotate(p.bbox, page, angle);
                if same_baseline(b, h, pb, height(p, pb)?) {
                    selected.insert(l.source);
                    changed = true;
                    break;
                }
            }
        }
    }
    Ok(selected)
}
fn repairs(chars: &[Value], work: &mut Work) -> Result<Map<String, Value>, Error> {
    let mut groups: Vec<([Value; 3], Vec<&Value>)> = Vec::new();
    for c in chars {
        let font = &c["font"];
        let key = [
            font["name"].clone(),
            font["weight"].clone(),
            font["flags"].clone(),
        ];
        work.charge(groups.len() + 1)?;
        if let Some((_, group)) = groups
            .iter_mut()
            .find(|(k, _)| k.iter().zip(&key).all(|(a, b)| eq(a, b)))
        {
            group.push(c);
        } else {
            groups.push((key, vec![c]));
        }
    }
    let mut output = Map::new();
    for (_, group) in groups {
        work.charge(group.len())?;
        let anchors: Vec<_> = group
            .iter()
            .filter(|c| {
                c["char"].as_str().is_some_and(|s| {
                    let mut it = s.chars();
                    it.next()
                        .is_some_and(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
                        && it.next().is_none()
                }) && !c["tight_bbox"].is_null()
                    && !c["origin"].is_null()
            })
            .collect();
        if anchors.len() < 20 {
            continue;
        }
        let bs = anchors
            .iter()
            .map(|c| rect(&c["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let tight = anchors
            .iter()
            .map(|c| rect(&c["tight_bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let em = median(bs.iter().map(|b| b[2] - b[0]).collect());
        let loose = median(bs.iter().map(|b| b[3] - b[1]).collect());
        let ink = median(tight.iter().map(|b| b[3] - b[1]).collect());
        if em <= 0. || loose < 2.2 * em || ink < 0.5 * em || ink > 1.4 * em {
            continue;
        }
        for c in group {
            work.charge(1)?;
            if c["bbox"].is_null() || c["origin"].is_null() || c["char_idx"].is_null() {
                continue;
            }
            let b = rect(&c["bbox"])?;
            let o = array(&c["origin"], 2)?;
            if b[3] - b[1] >= 2. * em {
                output.insert(
                    character_key(&c["char_idx"])?,
                    json!([
                        c["bbox"][0],
                        number(&o[1])? - 0.85 * em,
                        c["bbox"][2],
                        number(&o[1])? + 0.15 * em
                    ]),
                );
            }
        }
    }
    Ok(output)
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_with_work(v, &mut Work(0), false)
}
pub fn prepare_annotations(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_with_work(v, &mut Work(0), true)
}
fn parse_lines(source: &Value) -> Result<Vec<Line<'_>>, Error> {
    let mut lines = Vec::new();
    let mut unique = BTreeSet::new();
    for l in array(&source["lines"], 4096)? {
        let source = integer(&l["source_index"])?;
        if !unique.insert(source) || !l["text"].is_string() {
            return Err(invalid());
        }
        let line = Line {
            value: l,
            bbox: rect(&l["bbox"])?,
            source,
            angle: integer(&l["angle"])?,
        };
        height(&line, line.bbox)?;
        let sig = &l["font_signature"];
        if !sig.is_null() {
            let a = array(sig, 2)?;
            if a.len() != 2 || !a[0].is_string() {
                return Err(invalid());
            }
            integer(&a[1])?;
        }
        lines.push(line);
    }
    Ok(lines)
}
fn prepare_with_work(v: &Value, work: &mut Work, automatic: bool) -> Result<Value, Error> {
    prepare_cached(v, work, automatic, None, false)
}
fn prepare_cached(
    v: &Value,
    work: &mut Work,
    automatic: bool,
    cached: Option<(&[Value], &Map<String, Value>)>,
    separated: bool,
) -> Result<Value, Error> {
    let source = &v["source"];
    let p = array(&source["page_size"], 2)?;
    if p.len() != 2 {
        return Err(invalid());
    }
    let page = [number(&p[0])?, number(&p[1])?];
    if page.iter().any(|n| *n <= 0.) {
        return Err(invalid());
    }
    let lines = parse_lines(source)?;
    let rules = array(&source["drawing_lines"], 5000)?;
    for r in rules {
        candidate_rect(&r["bbox"])?;
        if !matches!(r["orientation"].as_str(), Some("horizontal" | "vertical")) {
            return Err(invalid());
        }
    }
    let chars = array(&source["chars"], 20000)?;
    for c in chars {
        if !c.is_object() || !c["char"].is_string() {
            return Err(invalid());
        }
        for k in ["bbox", "tight_bbox"] {
            optional_box(&c[k])?;
        }
        if !c["origin"].is_null() {
            let a = array(&c["origin"], 2)?;
            if a.len() != 2 {
                return Err(invalid());
            }
            number(&a[0])?;
            number(&a[1])?;
        }
        if !c["char_idx"].is_null() {
            character_key(&c["char_idx"])?;
        }
        let font = &c["font"];
        if !font.is_null() && !font.is_object() {
            return Err(invalid());
        }
        for k in ["name", "weight", "flags"] {
            if font[k].is_array() || font[k].is_object() {
                return Err(invalid());
            }
        }
    }
    let mut calibrated = if let Some((_, calibrated)) = cached {
        calibrated.clone()
    } else {
        repairs(chars, work)?
    };
    if let Some(m) = v.get("calibrated_char_bboxes").filter(|v| !v.is_null()) {
        let m = m.as_object().ok_or_else(invalid)?;
        if m.len() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        for (id, b) in m {
            let id = integer(&json!(id.parse::<i64>().map_err(|_| invalid())?))?;
            rect(b)?;
            calibrated.insert(id.to_string(), b.clone());
        }
    }
    let native_chars: Vec<_> = if let Some((chars, _)) = cached {
        chars.to_vec()
    } else {
        chars
            .iter()
            .map(|c| {
                let mut c = c.clone();
                if let Ok(id) = character_key(&c["char_idx"]) {
                    if let Some(b) = calibrated.get(&id) {
                        c["bbox"] = b.clone();
                    }
                }
                c
            })
            .collect()
    };
    let mut candidate = v["candidate"].clone();
    if !candidate.is_object() {
        return Err(invalid());
    }
    candidate_rect(&candidate["bbox"])?;
    candidate_box(&candidate["core_bbox"])?;
    integer(&candidate["angle"])?;
    candidate["line_indices"] = json!(ids(&candidate["line_indices"])?);
    for a in annotations(&candidate)? {
        validate_annotation(a)?;
    }
    for a in candidate["annotations"].as_array_mut().unwrap() {
        a["line_indices"] = json!(ids(&a["line_indices"])?);
    }
    if !separated {
        separate(&lines, rules, &mut candidate, work)?;
    }
    let materialized = if automatic {
        output::annotations(source, &lines, &candidate, work)?
    } else {
        Value::Null
    };
    let failed = if automatic {
        materialized["failed_annotations"].clone()
    } else {
        v.get("failed_annotations").cloned().unwrap_or(json!([]))
    };
    let failed = array(&failed, 2000)?;
    for a in failed {
        validate_annotation(a)?;
    }
    let mut body = candidate["bbox"].clone();
    if !annotations(&candidate)?.is_empty() && failed.len() != annotations(&candidate)?.len() {
        if !candidate["core_bbox"].is_null() {
            body = candidate["core_bbox"].clone();
        }
        for a in failed {
            work.charge(1)?;
            body = union_value(&body, &a["bbox"]);
        }
    }
    let mut selected = projection(&lines, &candidate, page, work)?;
    if automatic {
        for a in annotations(&candidate)? {
            selected.extend(ids(&a["line_indices"])?);
        }
        for id in ids(&materialized["externalized_line_indices"])? {
            selected.remove(&id);
        }
    }
    let mut result = json!({"candidate":candidate,"projection_line_indices":selected,"body_bbox":body,"calibrated_char_bboxes":calibrated,"native_chars":native_chars});
    if automatic {
        result
            .as_object_mut()
            .unwrap()
            .extend(materialized.as_object().unwrap().clone());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_native_table_block_preparation() {
        for name in ["native-table-block", "native-table-block-types"] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                assert_eq!(
                    prepare(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"])),
                    c["expected"],
                    "{}",
                    c["label"]
                );
            }
        }
    }
}
