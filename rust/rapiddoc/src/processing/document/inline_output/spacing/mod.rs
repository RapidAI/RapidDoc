//! Exact inline.spacing preparation and projected insertion/removal, before inline spans.
use super::*;
mod characters;
use crate::legacy_model::{page_containers::optional, path_graphics::integer};
fn size(v: &Value) -> Result<[f64; 2], Error> {
    crate::legacy_model::page_containers::size(v)
}
fn finished(v: Value) -> Result<Value, Error> {
    if crate::legacy_page::budget::output_exceeds_limit(
        &v,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(v)
    }
}
pub(super) fn prepare(v: &Value, w: &mut Work) -> Result<Value, Error> {
    prepare_lines(&v["lines"], v, w)
}
pub(super) fn prepare_lines(lines: &Value, v: &Value, w: &mut Work) -> Result<Value, Error> {
    let ls = array(lines, 4096)?;
    let space = v
        .get("space_before")
        .filter(|v| !v.is_null())
        .map(|v| array(v, 20000))
        .transpose()?
        .map(|a| a.iter().map(space_key).collect::<Result<BTreeSet<_>, _>>())
        .transpose()?;
    let mut out: Vec<Value> = vec![];
    let mut members = 0usize;
    for l in ls {
        if crate::legacy_model::native_text_assembly::python_truth(&l["formula_candidate_only"]) {
            continue;
        }
        let cs = optional(l, "chars", 20000)?;
        members += cs.len();
        if members > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        w.charge(cs.len().saturating_mul(16))?;
        let mut positions = BTreeSet::new();
        let mut suppressed = BTreeSet::new();
        for i in 1..cs.len() {
            let yes = if let Some(space) = space.as_ref() {
                space.contains(&space_key(&cs[i]["char_idx"])?)
                    && override_consecutive(&cs[i - 1], &cs[i])?
            } else {
                crate::legacy_model::table_native_multiline::spatial::tight(&cs[i - 1], &cs[i])
            };
            if yes {
                positions.insert(i);
            }
        }
        let source = integer(l, "source_index", 0)?;
        let continuation = out.last().is_some_and(|p| {
            p["source_index"].as_i64().and_then(|v| v.checked_add(1)) == Some(source)
                && characters::hyphen(p["text"].as_str().unwrap())
        });
        for i in 1..cs.len().saturating_sub(1) {
            if characters::cjk(&cs[i - 1], &cs[i], &cs[i + 1]) || characters::ligature(cs, i) {
                suppressed.insert(i);
            }
        }
        if positions.is_empty() && !continuation && suppressed.is_empty() {
            continue;
        }
        let (mut content, mut boundaries, mut removed, mut cursor) =
            (String::new(), vec![], vec![], 0usize);
        for (i, c) in cs.iter().enumerate() {
            if positions.contains(&i) {
                boundaries.push(cursor)
            }
            if suppressed.contains(&i) {
                removed.push(cursor)
            }
            let fragment = crate::legacy_model::inline_evidence::compact(
                &crate::legacy_model::table_native_multiline::spatial::raw_string(
                    Some(&c["char"]),
                    true,
                ),
            );
            cursor += fragment.chars().count();
            content.push_str(&fragment);
        }
        if !boundaries.is_empty() || continuation || !removed.is_empty() {
            out.push(json!({"bbox":l["bbox"],"text":content,"source_index":source,"boundaries":boundaries,"suppressed_boundaries":removed}))
        }
    }
    finished(json!(out))
}
fn space_key(v: &Value) -> Result<String, Error> {
    if let Some(n) = v.as_i64() {
        Ok(format!("i{n}"))
    } else if let Some(n) = v.as_u64() {
        Ok(format!("i{n}"))
    } else if let Some(n) = v.as_bool() {
        Ok(format!("i{}", u8::from(n)))
    } else if let Some(n) = v.as_f64() {
        if n.fract() == 0. && n.abs() < 1e30 {
            Ok(format!("i{}", n as i128))
        } else {
            Ok(format!("f{}", n.to_bits()))
        }
    } else if let Some(s) = v.as_str() {
        Ok(format!("s{s}"))
    } else if v.is_null() {
        Ok("none".into())
    } else {
        Err(invalid())
    }
}
fn override_consecutive(a: &Value, b: &Value) -> Result<bool, Error> {
    let a = a.get("char_idx").cloned().unwrap_or(json!(-2));
    let next = if let Some(n) = a.as_i64() {
        serde_json::to_value(i128::from(n) + 1).map_err(|_| limit())?
    } else if let Some(n) = a.as_u64() {
        serde_json::to_value(i128::from(n) + 1).map_err(|_| limit())?
    } else if let Some(n) = a.as_bool() {
        json!(i64::from(n) + 1)
    } else if let Some(n) = a.as_f64() {
        json!(n + 1.)
    } else {
        return Err(invalid());
    };
    Ok(space_key(&next)? == space_key(&b["char_idx"])?)
}
pub(super) fn apply(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let mut bs = array(&v["blocks"], 4000)?.to_vec();
    let raw = optional(v, "spacing_lines", 4096)?;
    if raw.is_empty() {
        return finished(json!(bs));
    }
    let p = size(&v["page_size"])?;
    let mut ls = vec![];
    for l in raw {
        let mut edits = vec![];
        for (key, flag) in [("boundaries", 0), ("suppressed_boundaries", 1)] {
            for b in optional(l, key, 20000)? {
                let n = b
                    .as_i64()
                    .or_else(|| b.as_bool().map(i64::from))
                    .ok_or_else(invalid)?;
                edits.push(Range {
                    start: n.unsigned_abs() as usize,
                    end: usize::from(n < 0),
                    styles: flag,
                    target: None,
                    source: 0,
                });
            }
        }
        ls.push(Line {
            bbox: bbox(&l["bbox"])?.ok_or_else(invalid)?,
            text: chars(l["text"].as_str().ok_or_else(invalid)?),
            source: integer(l, "source_index", 0)?,
            ranges: edits,
        });
    }
    for (i, assigned) in assignments(&bs, &ls, p, w)? {
        let content = chars(bs[i]["content"].as_str().unwrap());
        let projected = project(&content, w)?;
        let normalized = projected.iter().map(|t| t.value).collect::<Vec<_>>();
        let mut edits = BTreeSet::new();
        let mut cursor = 0;
        for (j, l) in assigned.iter().enumerate() {
            let start = if let Some(start) = matching::find(&normalized, &l.text, cursor, w)? {
                cursor = start + l.text.len();
                start
            } else {
                let Some((end, mapping)) =
                    matching::dehyphen(&normalized, l, assigned.get(j + 1), cursor, false, w)?
                else {
                    continue;
                };
                cursor = end;
                let Some(start) = mapping.first().copied().flatten() else {
                    continue;
                };
                start
            };
            for b in &l.ranges {
                let Some(k) = (if b.end == 1 {
                    start.checked_sub(b.start)
                } else {
                    start.checked_add(b.start)
                })
                .filter(|&k| k > 0 && k < projected.len()) else {
                    continue;
                };
                let left = &projected[k - 1];
                let right = &projected[k];
                if right.gap {
                    continue;
                }
                if b.styles == 0 && left.end == right.start {
                    edits.insert((right.start, right.start, " "));
                } else if b.styles == 1 && content.get(left.end..right.start) == Some(&[' '][..]) {
                    edits.insert((left.end, right.start, ""));
                }
            }
        }
        if !edits.is_empty() {
            let (mut out, mut cursor) = (String::new(), 0);
            for (start, end, replacement) in edits {
                w.charge(start.saturating_sub(cursor) + 1)?;
                if start < cursor {
                    return Err(invalid());
                }
                out.extend(content[cursor..start].iter());
                out.push_str(replacement);
                cursor = end;
            }
            out.extend(content[cursor..].iter());
            bs[i]["content"] = json!(out);
        }
    }
    finished(json!(bs))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_python_spacing() {
        for name in ["native-spacing", "native-spacing-contracts"] {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                let got = match c["kind"].as_str().unwrap() {
                    "prepare" => prepare(&c["request"], &mut Work(0)),
                    "apply" => apply(&c["request"], &mut Work(0)),
                    _ => unreachable!(),
                };
                if c.get("error").is_some() {
                    assert!(got.is_err(), "{}", c["label"])
                } else {
                    let got = got.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                    if let Some(d) =
                        crate::legacy_model::native_document::tests::diff(&got, &c["expected"], "")
                    {
                        panic!("{}: {d}", c["label"])
                    }
                }
            }
        }
    }
}
