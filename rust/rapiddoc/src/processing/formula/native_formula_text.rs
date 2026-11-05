//! 旧公式文字恢复；行首分式复用完整 inline 物化和视觉行连接。
use super::{
    native_lines,
    page_containers::{coerce, height, media, size},
    path_graphics::{array, bbox, integer, Box4},
    table_text_styles::clusters,
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_decimal, text_boundary::python_space};
use serde_json::{json, Value};
use std::borrow::Cow;
fn greek(c: char) -> bool {
    ('α'..='ω').contains(&c) || ('Α'..='Ω').contains(&c)
}
fn axis(a: &Box4, b: &Box4, k: usize) -> f64 {
    let d = (a.n[k + 2] - a.n[k]).min(b.n[k + 2] - b.n[k]);
    if d > 0. {
        (a.n[k + 2].min(b.n[k + 2]) - a.n[k].max(b.n[k])).max(0.) / d
    } else {
        0.
    }
}
pub fn restore_leading(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    restore_with_work(v, &mut Work(0))
}
pub(super) fn restore_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    restore_with_sources(v, w).map(|(out, _)| out)
}
pub(super) fn restore_with_sources(v: &Value, w: &mut Work) -> Result<(Value, Vec<Value>), Error> {
    restore_with_identities(v, w).map(|(out, source, _)| (out, source))
}
pub(super) fn restore_with_identities(
    v: &Value,
    w: &mut Work,
) -> Result<(Value, Vec<Value>, std::collections::BTreeSet<i64>), Error> {
    let sz = size(&v["page_size"])?;
    let original = array(&v["lines"], 4096)?;
    let rules = array(&v["drawing_lines"], 5000)?;
    let (remaining, source, detached) = restore_state(original, rules, sz, w)?;
    Ok((
        Value::Array(remaining.into_iter().map(|(_, l)| l.into_owned()).collect()),
        source.into_iter().map(Cow::into_owned).collect(),
        detached,
    ))
}

/// 正式页面链只需要来源号/字宽更新；未变化的行直接移交，不复制字符树。
pub(super) fn restore_owned_with_identities(
    lines: Value,
    page_size: &Value,
    drawings: &Value,
    w: &mut Work,
) -> Result<(Value, Vec<Value>, std::collections::BTreeSet<i64>), Error> {
    let sz = size(page_size)?;
    let original = array(&lines, 4096)?;
    let rules = array(drawings, 5000)?;
    let (remaining, source, detached) = restore_state(original, rules, sz, w)?;
    let updates = source
        .iter()
        .map(|l| {
            value_object!({"source_index":l["source_index"],
        "median_glyph_width":l["median_glyph_width"]})
        })
        .collect();
    let moves: Vec<_> = remaining
        .into_iter()
        .map(|(id, line)| match line {
            Cow::Borrowed(_) => (id, None),
            Cow::Owned(line) => (id, Some(line)),
        })
        .collect();
    drop(source);
    let Value::Array(lines) = lines else {
        return Err(invalid());
    };
    let mut originals: Vec<_> = lines.into_iter().map(Some).collect();
    let output = moves
        .into_iter()
        .map(|(id, line)| {
            line.or_else(|| originals.get_mut(id).and_then(Option::take))
                .ok_or_else(invalid)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((Value::Array(output), updates, detached))
}
type RestoreState<'a> = (
    Vec<(usize, Cow<'a, Value>)>,
    Vec<Cow<'a, Value>>,
    std::collections::BTreeSet<i64>,
);
fn restore_state<'a>(
    original: &'a [Value],
    rules: &[Value],
    sz: [f64; 2],
    w: &mut Work,
) -> Result<RestoreState<'a>, Error> {
    for line in original {
        bbox(&line["bbox"])?;
        integer(line, "source_index", 0)?;
        integer(line, "angle", 0)?;
        w.charge(line["text"].as_str().ok_or_else(invalid)?.len() + 1)?;
    }
    // Identity follows original object identity, including equal-valued rows.
    let mut remaining = original
        .iter()
        .map(Cow::Borrowed)
        .enumerate()
        .collect::<Vec<_>>();
    let mut source_lines = original.iter().map(Cow::Borrowed).collect::<Vec<_>>();
    let mut next = original.len();
    for (identity, numerator) in original.iter().enumerate() {
        if !remaining.iter().any(|(i, _)| *i == identity)
            || integer(numerator, "angle", 0)? != 0
            || !numerator["semantic_type"].is_null()
        {
            continue;
        }
        let raw = numerator["text"].as_str().ok_or_else(invalid)?;
        let t = raw.trim_matches(python_space);
        let count = t.chars().count();
        if !(1..=8).contains(&count)
            || !t
                .chars()
                .all(|c| python_decimal(c).is_some() || greek(c) || "+−-".contains(c))
            || !raw.chars().any(greek)
        {
            continue;
        }
        let nb = bbox(&numerator["bbox"])?;
        let em = height(numerator, &nb)?;
        let mut replacement = None;
        for (den_id, denominator) in remaining.clone() {
            w.charge(1)?;
            let text = denominator["text"].as_str().ok_or_else(invalid)?;
            if den_id == identity
                || integer(&denominator, "angle", 0)? != 0
                || !denominator["semantic_type"].is_null()
                || text
                    .chars()
                    .filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c))
                    .count()
                    < 4
                || !text
                    .trim_start_matches(python_space)
                    .chars()
                    .next()
                    .is_some_and(greek)
            {
                continue;
            }
            let Some(chars) = denominator["chars"].as_array().filter(|a| !a.is_empty()) else {
                continue;
            };
            let first = chars.iter().find(|c| {
                !super::table_native_multiline::spatial::raw_string(c.get("char"), false)
                    .trim_matches(python_space)
                    .is_empty()
            });
            let ink = if let Some(c) = first {
                coerce(
                    c.get("tight_bbox")
                        .filter(|b| !b.is_null())
                        .unwrap_or(&c["bbox"]),
                )?
            } else {
                None
            };
            let Some(ink) = ink else { continue };
            let mut has_rule = false;
            for r in rules {
                w.charge(1)?;
                let b = bbox(&r["bbox"])?;
                let width = b.n[2] - b.n[0];
                let cx = (ink.n[0] + ink.n[2]) * 0.5;
                if r["orientation"] == "horizontal"
                    && width >= 0.5 * (nb.n[2] - nb.n[0])
                    && width <= 1.5 * em
                    && b.n[0] <= cx
                    && cx <= b.n[2]
                    && nb.n[1] < b.n[1]
                    && b.n[1] < ink.n[3]
                    && axis(&b, &nb, 0) >= 0.7
                    && ink.n[3] - nb.n[1] > 0.
                    && ink.n[3] - nb.n[1] <= 2.5 * em
                {
                    has_rule = true
                }
            }
            if !has_rule {
                continue;
            }
            let db = bbox(&denominator["bbox"])?;
            let mut preceding = false;
            for (i, l) in &remaining {
                w.charge(1)?;
                if *i == identity || *i == den_id {
                    continue;
                }
                let b = bbox(&l["bbox"])?;
                if l["text"].as_str().ok_or_else(invalid)?.chars().count() >= 20
                    && b.n[3] <= nb.n[1] + 0.2 * em
                    && nb.n[1] - b.n[3] <= em
                    && (b.n[0] - db.n[0]).abs() <= em
                {
                    preceding = true
                }
            }
            if !preceding {
                continue;
            }
            let mut members = vec![numerator.clone(), denominator.as_ref().clone()];
            let mut order = vec![0usize, 1];
            order.sort_by(|&i, &j| {
                members[i]["bbox"][1]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&members[j]["bbox"][1].as_f64().unwrap())
                    .then(
                        members[i]["bbox"][0]
                            .as_f64()
                            .unwrap()
                            .total_cmp(&members[j]["bbox"][0].as_f64().unwrap()),
                    )
            });
            let mut rows: Vec<Vec<usize>> = Vec::new();
            for i in order {
                let b = bbox(&members[i]["bbox"])?;
                let mut found = None;
                for (j, row) in rows.iter().enumerate() {
                    for &k in row {
                        w.charge(1)?;
                        if axis(&b, &bbox(&members[k]["bbox"])?, 1) >= 0.55 {
                            found = Some(j);
                            break;
                        }
                    }
                    if found.is_some() {
                        break;
                    }
                }
                if let Some(j) = found {
                    rows[j].push(i)
                } else {
                    rows.push(vec![i])
                }
            }
            let merged = clusters::materialize_members(&members, sz, em, false, w)?;
            let mut out = native_lines::defaults();
            for (k, v) in merged.as_object().ok_or_else(invalid)? {
                out[k] = v.clone()
            }
            // Constructor marker features precede both subsequent text assignments in Python.
            native_lines::markers(&mut out);
            let mut content = Vec::new();
            for row in rows {
                content.push(media::row(&row, &mut members, sz, w)?)
            }
            source_lines[identity] = Cow::Owned(members[0].clone());
            if den_id < source_lines.len() {
                source_lines[den_id] = Cow::Owned(members[1].clone());
            }
            out["text"] = json!(content.join(" "));
            out["paragraph_formula_context"] = json!(true);
            out["formula_candidate_only"] = json!(false);
            out["text"] = json!(format!(
                "{}/{}",
                raw.trim_matches(python_space),
                text.trim_start_matches(python_space)
            ));
            out["semantic_type"] = json!("text");
            out["title_suppressed"] = json!(true);
            replacement = Some((den_id, out));
            break;
        }
        if let Some((den_id, out)) = replacement {
            remaining.retain(|(i, _)| *i != identity && *i != den_id);
            remaining.push((next, Cow::Owned(out)));
            next = next.checked_add(1).ok_or_else(limit)?;
        }
    }
    remaining.sort_by(|(_, a), (_, b)| {
        a["bbox"][1]
            .as_f64()
            .unwrap()
            .total_cmp(&b["bbox"][1].as_f64().unwrap())
            .then(
                a["bbox"][0]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&b["bbox"][0].as_f64().unwrap()),
            )
            .then(
                integer(a, "source_index", 0)
                    .unwrap_or(0)
                    .cmp(&integer(b, "source_index", 0).unwrap_or(0)),
            )
    });
    let detached = remaining
        .iter()
        .filter(|(i, _)| *i >= original.len())
        .map(|(_, l)| integer(l, "source_index", 0))
        .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
    Ok((remaining, source_lines, detached))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owned_leading_restore_keeps_python_output_updates_and_work() {
        let fixture: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!("../../../../../tests/fixtures/native_leading_owned_reference.json.gz")
                [..],
        ))
        .unwrap();
        for host in [false, true] {
            let _scope = host.then(crate::legacy_page::budget::host_page);
            for case in fixture["cases"].as_array().unwrap() {
                let q = &case["request"];
                let (mut a, mut b) = (Work(0), Work(0));
                let (old, sources, ids) = restore_with_identities(q, &mut a).unwrap();
                let (actual, updates, new_ids) = restore_owned_with_identities(
                    q["lines"].clone(),
                    &q["page_size"],
                    &q["drawing_lines"],
                    &mut b,
                )
                .unwrap();
                assert_eq!(old, case["expected"], "{}", case["label"]);
                assert_eq!(actual, old, "{}", case["label"]);
                assert_eq!(a.0, b.0, "{}", case["label"]);
                assert_eq!(ids, new_ids);
                let expected: Vec<_> = sources
                    .iter()
                    .map(|l| {
                        value_object!({
                    "source_index":l["source_index"],"median_glyph_width":l["median_glyph_width"]})
                    })
                    .collect();
                assert_eq!(updates, expected, "{}", case["label"]);
            }
        }
        for lines in [json!(null), json!([{"text":"bad","bbox":[0,1,2]}])] {
            let q = json!({"lines":lines,"page_size":[400,600],"drawing_lines":[]});
            let a = restore_with_identities(&q, &mut Work(0)).unwrap_err();
            let b = restore_owned_with_identities(
                lines,
                &q["page_size"],
                &q["drawing_lines"],
                &mut Work(0),
            )
            .unwrap_err();
            assert_eq!(a.code, b.code);
            assert_eq!(a.message, b.message);
        }
    }
    #[test]
    fn old_leading_formula_text() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-leading-fractions.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                restore_leading(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"])),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
}
