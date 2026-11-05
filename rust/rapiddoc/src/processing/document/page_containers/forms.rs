//! 旧 _select_form_image_bboxes：实际字符归属、边缘 ink 补回、矢量收紧与映射别名。
use super::*;
use crate::legacy_model::path_graphics::{area, integer, overlap_first};
use rapiddoc_native::text_boundary::python_space;
use std::collections::BTreeSet;
pub(super) fn maps(s: &Value, k: &str, w: &mut Work) -> Result<Vec<(Box4, BTreeSet<i64>)>, Error> {
    let mut result: Vec<(Box4, BTreeSet<i64>)> = Vec::new();
    for v in optional(s, k, 5000)? {
        let b = bbox(&v["bbox"])?;
        w.charge(array(&v["source_indices"], 20000)?.len() * 16 + 1)?;
        let set = array(&v["source_indices"], 20000)?
            .iter()
            .map(|i| i.as_i64().ok_or_else(invalid))
            .collect::<Result<_, _>>()?;
        w.charge(result.len() + 1)?;
        if let Some((_, ids)) = result.iter_mut().find(|(old, _)| old.n == b.n) {
            *ids = set;
        } else {
            result.push((b, set));
        }
    }
    Ok(result)
}
fn owns(ch: &Value, ids: &BTreeSet<i64>, w: &mut Work) -> Result<bool, Error> {
    if let Some(s) = ch.get("source_indices") {
        w.charge(array(s, 20000)?.len() * 16 + 1)?;
        Ok(array(s, 20000)?
            .iter()
            .any(|i| i.as_i64().is_some_and(|i| ids.contains(&i))))
    } else {
        Ok(ch["char_idx"].as_i64().is_some_and(|i| ids.contains(&i)))
    }
}
fn visible(ch: &Value) -> bool {
    ch.get("char")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.trim_matches(python_space).is_empty())
}
pub(super) fn member(
    l: &Value,
    form: &Box4,
    maps: &[(Box4, BTreeSet<i64>)],
    w: &mut Work,
) -> Result<Option<Box4>, Error> {
    w.charge(maps.len() + 1)?;
    if let Some((_, members)) = maps.iter().find(|(b, _)| b.n == form.n) {
        let mut any = false;
        for c in optional(l, "chars", 20000)? {
            w.charge(1)?;
            if visible(c) {
                any = true;
                if !owns(c, members, w)? {
                    return Ok(None);
                }
            }
        }
        if !any {
            return Ok(None);
        }
    }
    let b = bbox(&l["bbox"])?;
    if overlap_first(&b, form) >= 0.9 {
        return Ok(Some(b));
    }
    let ink = if !l["ink_bbox"].is_null() {
        Some(bbox(&l["ink_bbox"])?)
    } else {
        let mut boxes = Vec::new();
        for ch in optional(l, "chars", 20000)? {
            w.charge(1)?;
            if !visible(ch) {
                continue;
            }
            let Some(b) = coerce(&ch["tight_bbox"])? else {
                return Ok(None);
            };
            boxes.push(b);
        }
        if boxes.is_empty() {
            None
        } else {
            Some(combined(&boxes)?)
        }
    };
    Ok(ink.filter(|b| overlap_first(b, form) >= 0.99))
}
fn tighten(
    s: &Value,
    form: &Box4,
    members: &[(Box4, BTreeSet<i64>)],
    paths: &[(Box4, BTreeSet<i64>)],
    sz: [f64; 2],
    w: &mut Work,
) -> Result<Box4, Error> {
    let mut evidence = Vec::new();
    let path_ids = paths
        .iter()
        .find(|(b, _)| b.n == form.n)
        .map(|(_, ids)| ids);
    for p in optional(s, "path_infos", 5000)? {
        w.charge(1)?;
        let b = bbox(&p["bbox"])?;
        if integer(p, "form_depth", 0)? > 0
            && overlap_first(&b, form) >= 0.9
            && path_ids
                .is_none_or(|ids| p["source_index"].as_i64().is_some_and(|i| ids.contains(&i)))
        {
            evidence.push(b);
        }
    }
    if evidence.len() < 2 {
        return Ok(form.clone());
    }
    for r in optional(s, "drawing_lines", 5000)? {
        w.charge(1)?;
        let b = bbox(&r["bbox"])?;
        if overlap_first(&b, form) >= 0.9 {
            evidence.push(b);
        }
    }
    if evidence.len() < 4 {
        return Ok(form.clone());
    }
    for l in optional(s, "lines", 4096)? {
        if let Some(b) = member(l, form, members, w)? {
            evidence.push(b);
        }
    }
    let ids = members
        .iter()
        .find(|(b, _)| b.n == form.n)
        .map(|(_, ids)| ids);
    for ch in optional(s, "chars", 20000)? {
        w.charge(1)?;
        if visible(ch) {
            if let Some(b) = coerce(&ch["tight_bbox"])? {
                if overlap_first(&b, form) >= 0.99
                    && match ids {
                        Some(ids) => owns(ch, ids, w)?,
                        None => true,
                    }
                {
                    evidence.push(b);
                }
            }
        }
    }
    let Some(b) = clip(&combined(&evidence)?, sz)? else {
        return Ok(form.clone());
    };
    if b.n[2] - b.n[0] < 0.5 * (form.n[2] - form.n[0]).max(0.1)
        || b.n[3] - b.n[1] < 0.5 * (form.n[3] - form.n[1]).max(0.1)
        || area(&b) < 0.25 * area(form)
    {
        Ok(form.clone())
    } else {
        Ok(b)
    }
}
pub(super) fn select(raw: &Value, w: &mut Work) -> Result<Value, Error> {
    select_owned(raw.clone(), w)
}
pub(super) fn select_owned(mut raw: Value, w: &mut Work) -> Result<Value, Error> {
    let (updates, forms) = select_fields(&raw, w)?;
    for (key, value) in updates.as_object().unwrap() {
        raw[key] = value.clone();
    }
    Ok(crate::processing::with_source(
        raw,
        value_object!({"form_bboxes":forms}),
    ))
}
fn select_fields(raw: &Value, w: &mut Work) -> Result<(Value, Value), Error> {
    raw.as_object().ok_or_else(invalid)?;
    let sz = size(&raw["page_size"])?;
    // Only the two alias maps change. Keep the character/line trees borrowed.
    let mut source = json!({});
    let page_area = sz[0].max(0.) * sz[1].max(0.);
    let forms = optional(raw, "form_bboxes", 5000)?;
    let lines = optional(raw, "lines", 4096)?;
    for (key, max) in [
        ("path_infos", 5000),
        ("drawing_lines", 5000),
        ("form_infos", 5000),
        ("chars", 20000),
    ] {
        optional(raw, key, max)?;
    }
    let mut unique = BTreeSet::new();
    let mut chars = 0;
    let mut hs = Vec::new();
    for l in lines {
        w.charge(1)?;
        let text = l["text"].as_str().ok_or_else(invalid)?;
        if text.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        w.charge(text.len() + 1)?;
        if !unique.insert(integer(l, "source_index", 0)?) {
            return Err(invalid());
        }
        chars += optional(l, "chars", 20000)?.len();
        if chars > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        if integer(l, "angle", 0)? == 0 {
            hs.push(height(l, &bbox(&l["bbox"])?)?);
        }
    }
    let mut members = maps(raw, "form_member_sources", w)?;
    let mut paths = maps(raw, "form_path_sources", w)?;
    for (key, map) in [
        ("form_member_sources", &members),
        ("form_path_sources", &paths),
    ] {
        if raw.get(key).is_some() {
            source[key] = json!(map
                .iter()
                .map(|(b, ids)| json!({"bbox":b.raw,"source_indices":ids}))
                .collect::<Vec<_>>());
        }
    }
    let em = if hs.is_empty() { 1. } else { median(hs) };
    let mut selected = Vec::new();
    if page_area > 0. {
        for raw_form in forms {
            w.charge(1)?;
            let Some(b) = coerce(raw_form)? else { continue };
            let Some(b) = clip(&b, sz)? else { continue };
            let ratio = area(&b) / page_area;
            let retained = optional(raw, "form_infos", 5000)?.iter().any(|f| {
                bbox(&f["bbox"])
                    .is_ok_and(|x| x.n == bbox(raw_form).map(|x| x.n).unwrap_or([f64::NAN; 4]))
            });
            w.charge(optional(raw, "form_infos", 5000)?.len())?;
            if ratio < 0.01
                || (ratio > 0.8 && !retained)
                || b.n[2] - b.n[0] < 4. * em
                || b.n[3] - b.n[1] < 4. * em
            {
                continue;
            }
            let mut rows = BTreeSet::new();
            for l in lines {
                if member(l, &b, &members, w)?.is_some() {
                    rows.insert(
                        l.get("visual_row_id")
                            .filter(|v| !v.is_null())
                            .unwrap_or(&l["source_index"])
                            .as_i64()
                            .ok_or_else(invalid)?,
                    );
                }
            }
            let mut drawings = 0;
            for r in optional(raw, "drawing_lines", 5000)? {
                w.charge(1)?;
                if overlap_first(&bbox(&r["bbox"])?, &b) >= 0.9 {
                    drawings += 1;
                }
            }
            if rows.len() < 2 && drawings < 4 {
                continue;
            }
            let tightened = tighten(raw, &b, &members, &paths, sz, w)?;
            for (key, map) in [
                ("form_member_sources", &mut members),
                ("form_path_sources", &mut paths),
            ] {
                let original = bbox(raw_form)?;
                if let Some((_, ids)) = map.iter().find(|(b, _)| b.n == original.n) {
                    let ids = ids.clone();
                    if let Some((_, old)) = map.iter_mut().find(|(b, _)| b.n == tightened.n) {
                        *old = ids.clone();
                    } else {
                        map.push((tightened.clone(), ids.clone()));
                    }
                    let mut values = optional(&source, key, 5000)?.to_vec();
                    let alias = json!({"bbox":tightened.raw,"source_indices":ids});
                    if let Some(entry) = values
                        .iter_mut()
                        .find(|v| bbox(&v["bbox"]).is_ok_and(|b| b.n == tightened.n))
                    {
                        entry["source_indices"] = json!(ids);
                    } else {
                        values.push(alias);
                    }
                    if values.len() > rapiddoc_core::resources::limit(5000) {
                        return Err(limit());
                    }
                    source[key] = json!(values);
                }
            }
            selected.push(tightened);
        }
    }
    selected.sort_by(|a, b| {
        [1, 0, 3, 2]
            .into_iter()
            .map(|i| a.n[i].partial_cmp(&b.n[i]).unwrap())
            .find(|o| !o.is_eq())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok((
        source,
        Value::Array(selected.into_iter().map(|b| Value::from(b.raw)).collect()),
    ))
}
pub(crate) fn supersedes(form: &Box4, nested: &Box4) -> bool {
    area(form) > 0. && area(nested) < 0.5 * area(form) && overlap_first(nested, form) >= 0.9
}
