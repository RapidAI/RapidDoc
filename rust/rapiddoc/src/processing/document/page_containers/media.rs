//! 旧 Form 图像认领及内部文字：真实视觉行、旋转数值标签、字宽空格与缓存副作用。
use super::*;
use crate::legacy_model::path_graphics::integer;
use rapiddoc_native::{python_decimal, python_printable, text_boundary::python_space};
use std::collections::{BTreeMap, BTreeSet};
pub(in crate::processing) fn glyph_width(
    l: &mut Value,
    sz: [f64; 2],
    w: &mut Work,
) -> Result<Option<f64>, Error> {
    if !l["median_glyph_width"].is_null() {
        return Ok(Some(num(&l["median_glyph_width"])?));
    }
    let mut widths = Vec::new();
    let a = integer(l, "angle", 0)?;
    for ch in optional(l, "chars", 20000)? {
        w.charge(1)?;
        let raw = ch["char"].as_str().unwrap_or("");
        if raw.is_empty() || !raw.chars().all(python_printable) || raw.chars().all(python_space) {
            continue;
        }
        if let Some(b) = coerce(&ch["bbox"])? {
            if let Some(b) = clip(&b, sz)? {
                let b = super::index::local(&b, sz, a);
                widths.push((b.n[2] - b.n[0]).max(0.1));
            }
        }
    }
    if widths.is_empty() {
        Ok(None)
    } else {
        let n = median(widths);
        l["median_glyph_width"] = json!(n);
        Ok(Some(n))
    }
}
pub(in crate::processing) fn row(
    members: &[usize],
    ls: &mut [Value],
    sz: [f64; 2],
    w: &mut Work,
) -> Result<String, Error> {
    let mut pairs = members
        .iter()
        .map(|&i| {
            Ok((
                i,
                super::index::local(&bbox(&ls[i]["bbox"])?, sz, integer(&ls[i], "angle", 0)?),
            ))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    pairs.sort_by(|(i, a), (j, b)| {
        a.n[0]
            .partial_cmp(&b.n[0])
            .unwrap()
            .then(a.n[1].partial_cmp(&b.n[1]).unwrap())
            .then(
                ls[*i]["source_index"]
                    .as_i64()
                    .cmp(&ls[*j]["source_index"].as_i64()),
            )
    });
    let Some(&(first, _)) = pairs.first() else {
        return Ok(String::new());
    };
    let mut out = ls[first]["text"]
        .as_str()
        .ok_or_else(invalid)?
        .trim_matches(python_space)
        .to_owned();
    for pair in pairs.windows(2) {
        w.charge(1)?;
        let (i, a) = &pair[0];
        let (j, b) = &pair[1];
        let gap = b.n[0] - a.n[2];
        let h = height(&ls[*i], a)?.max(height(&ls[*j], b)?);
        let mut widths = Vec::new();
        for i in [*i, *j] {
            if let Some(n) = glyph_width(&mut ls[i], sz, w)? {
                widths.push(n)
            }
        }
        let width = if widths.is_empty() {
            (0.5 * h).max(1.)
        } else {
            median(widths)
        };
        if gap > (0.2 * h).max(0.5) {
            if width == 0. {
                return Err(invalid());
            }
            let count = (gap / width).round_ties_even().clamp(1., 8.) as usize;
            out.push_str(&" ".repeat(count));
        }
        out.push_str(
            ls[*j]["text"]
                .as_str()
                .ok_or_else(invalid)?
                .trim_matches(python_space),
        );
        if out.len() > rapiddoc_core::resources::limit(8_000_000) {
            return Err(limit());
        }
    }
    Ok(out.trim_matches(python_space).to_owned())
}
pub(super) fn content(
    members: &[usize],
    ls: &mut [Value],
    sz: [f64; 2],
    w: &mut Work,
) -> Result<String, Error> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut keys = BTreeMap::new();
    let mut numeric = 0;
    for &i in members {
        let l = &ls[i];
        let text = l["text"]
            .as_str()
            .ok_or_else(invalid)?
            .trim_matches(python_space);
        w.charge(text.len() + 1)?;
        if !text.is_empty()
            && text
                .chars()
                .all(|c| python_decimal(c).is_some() || ".,%+/−-".contains(c))
        {
            numeric += 1
        }
        let id = if let Some(row) = l.get("visual_row_id").filter(|v| !v.is_null()) {
            (
                integer(l, "angle", 0)?,
                0,
                row.as_i64().ok_or_else(invalid)?,
            )
        } else {
            (integer(l, "angle", 0)?, 1, integer(l, "source_index", 0)?)
        };
        let next = groups.len();
        let group = *keys.entry(id).or_insert(next);
        if group == next {
            groups.push(Vec::new());
        }
        groups[group].push(i);
    }
    let chart = numeric >= 6 && numeric as f64 >= 0.45 * members.len() as f64;
    let mut rows: Vec<(Box4, String)> = Vec::new();
    let mut rotated = Vec::new();
    for group in groups {
        let bs = group
            .iter()
            .map(|&i| bbox(&ls[i]["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let b = combined(&bs)?;
        let angle = integer(&ls[group[0]], "angle", 0)?;
        let text = row(&group, ls, sz, w)?;
        if !text.is_empty() {
            if chart && [90, 270].contains(&angle) {
                rotated.push((
                    rows.len(),
                    b.clone(),
                    angle,
                    ls[group[0]].get("effective_height").map_or(Ok(0.), num)?,
                ));
            }
            rows.push((b, text));
        }
    }
    for (_, bounds, a, em) in &rotated {
        w.charge(rotated.len() + 1)?;
        let peers: Vec<_> = rotated
            .iter()
            .filter(|(_, b, angle, h)| {
                angle == a
                    && (b.n[1] - bounds.n[1])
                        .abs()
                        .min((b.n[3] - bounds.n[3]).abs())
                        <= 0.5 * em.min(*h)
            })
            .collect();
        if peers.len() >= 3 {
            let top = peers
                .iter()
                .map(|(_, b, _, _)| b.n[1])
                .fold(f64::INFINITY, f64::min);
            for (i, _, _, _) in peers {
                rows[*i].0.n[1] = top;
                rows[*i].0.raw[1] = json!(top);
            }
        }
    }
    rows.sort_by(|(a, _), (b, _)| {
        a.n[1]
            .partial_cmp(&b.n[1])
            .unwrap()
            .then(a.n[0].partial_cmp(&b.n[0]).unwrap())
    });
    let text = rows
        .iter()
        .map(|(_, s)| s.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    Ok(crate::legacy_model::code_containers::text::sanitize(&text))
}
#[cfg(test)]
pub(super) fn forms(v: &Value, w: &mut Work) -> Result<Value, Error> {
    forms_owned(v.clone(), w)
}
pub(super) fn forms_owned(mut v: Value, w: &mut Work) -> Result<Value, Error> {
    let mut source = v["source"].take();
    let sz = size(&source["page_size"])?;
    let mut ls = optional(&source, "lines", 4096)?.to_vec();
    let forms = array(&v["form_bboxes"], 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let prior = array(&v["claimed_line_indices"], 20000)?
        .iter()
        .map(|i| i.as_i64().ok_or_else(invalid))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let maps = super::forms::maps(&source, "form_member_sources", w)?;
    let mut groups = vec![Vec::new(); forms.len()];
    let mut claimed = BTreeSet::new();
    for (i, l) in ls.iter().enumerate() {
        let id = integer(l, "source_index", 0)?;
        if prior.contains(&id) {
            continue;
        }
        let mut best: Option<usize> = None;
        for (j, b) in forms.iter().enumerate() {
            if super::forms::member(l, b, &maps, w)?.is_some()
                && best.is_none_or(|old| {
                    crate::legacy_model::graphic_arbitration::area_key(b)
                        < crate::legacy_model::graphic_arbitration::area_key(&forms[old])
                })
            {
                best = Some(j)
            }
        }
        if let Some(j) = best {
            groups[j].push(i);
            claimed.insert(id);
        }
    }
    let mut blocks = Vec::new();
    for (b, group) in forms.iter().zip(groups) {
        blocks.push(
            json!({"type":"image","bbox":b.raw,"angle":0,"content":content(&group,&mut ls,sz,w)?}),
        );
    }
    blocks.sort_by(|a, b| {
        a["bbox"][1]
            .as_f64()
            .partial_cmp(&b["bbox"][1].as_f64())
            .unwrap()
            .then(
                a["bbox"][0]
                    .as_f64()
                    .partial_cmp(&b["bbox"][0].as_f64())
                    .unwrap(),
            )
    });
    source["lines"] = Value::Array(ls);
    Ok(crate::processing::with_source(
        source,
        json!({"form_image_blocks":blocks,"claimed_line_indices":claimed}),
    ))
}
