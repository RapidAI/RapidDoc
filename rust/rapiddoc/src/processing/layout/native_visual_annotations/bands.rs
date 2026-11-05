//! Entire original annotation_bands pass, including separated captions and multi-column symbol legends.
use super::*;
fn bb(b: &Value) -> Result<Box4, Error> {
    super::super::path_graphics::bbox(&b["bbox"])
}
fn em(b: &Value) -> Result<f64, Error> {
    let hs = optional(b, "_line_heights", 4096)?
        .iter()
        .map(|v| {
            if let Some(b) = v.as_bool() {
                Ok(f64::from(b))
            } else {
                num(v)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if hs.is_empty() {
        let b = bb(b)?;
        Ok(b.n[3] - b.n[1])
    } else {
        Ok(median(hs))
    }
}
fn has_lines(b: &Value) -> bool {
    b["_text_lines"].as_array().is_some_and(|ls| !ls.is_empty())
}
fn merge(bs: &[Value], ids: &[usize], spaces: bool) -> Result<Value, Error> {
    super::super::native_text_assembly::merge_internal_blocks_mode(bs, ids, spaces)
}
fn alpha(t: &str) -> bool {
    !t.is_empty() && t.chars().all(rapiddoc_native::text_boundary::python_alpha)
}
pub(super) fn recover(bs: &mut Vec<Value>, rules: &[Value], w: &mut Work) -> Result<(), Error> {
    let images = bs
        .iter()
        .enumerate()
        .filter(|(_, b)| b["type"] == "image")
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let mut removed = BTreeSet::new();
    for image in images {
        let bounds = bb(&bs[image])?;
        let mut seeds = vec![];
        for (i, b) in bs.iter().enumerate() {
            w.charge(1)?;
            if !strong_caption(text(b))
                || !has_lines(b)
                || bb(b)?.n[0] >= bounds.n[2]
                || !regex(
                    r"(?i)^\s*(?:fig(?:ure)?\.?|diagram|图)\s*(?:\d+[A-Za-z]?|[IVX]+)[.:：]?\s*$",
                )
                .is_match(&fold(text(b)))
            {
                continue;
            }
            let h = em(b)?;
            let b = bb(b)?;
            if (b.n[3] - bounds.n[1]).abs() < 1.5 * h
                || (0.0..4. * h).contains(&(b.n[1] - bounds.n[3]))
            {
                seeds.push(i)
            }
        }
        let Some(seed) = seeds.into_iter().min_by(|&i, &j| {
            let a = bb(&bs[i]).unwrap();
            let b = bb(&bs[j]).unwrap();
            (a.n[3] - bounds.n[1])
                .abs()
                .min((a.n[1] - bounds.n[3]).abs())
                .total_cmp(
                    &(b.n[3] - bounds.n[1])
                        .abs()
                        .min((b.n[1] - bounds.n[3]).abs()),
                )
        }) else {
            continue;
        };
        let h = em(&bs[seed])?;
        let leading = bb(&bs[seed])?.n[3] <= bounds.n[1] + h;
        bs[seed]["type"] = json!("caption");
        bs[seed]["_annotation_band_parent_index"] = json!(image);
        let mut ids = vec![];
        for (i, b) in bs.iter().enumerate() {
            if i == image
                || removed.contains(&i)
                || !has_lines(b)
                || !["text", "paragraph_title", "caption", "footnote"]
                    .iter()
                    .any(|t| b["type"] == *t)
            {
                continue;
            }
            let b = bb(b)?;
            if b.n[1] >= bounds.n[3] - 0.25 * h && axis(&b, &bounds, 0) >= 0.5 {
                ids.push(i)
            }
        }
        ids.sort_by(|&i, &j| {
            let a = bb(&bs[i]).unwrap();
            let b = bb(&bs[j]).unwrap();
            a.n[1].total_cmp(&b.n[1]).then(a.n[0].total_cmp(&b.n[0]))
        });
        if ids.is_empty() {
            continue;
        }
        let first = if leading { ids[0] } else { seed };
        let fb = bb(&bs[first])?;
        if fb.n[1] - bounds.n[3] > 4. * h {
            continue;
        }
        let left = fb.n[0];
        let mut right = fb.n[2].max(bounds.n[2]);
        if !leading {
            for &i in &ids {
                let b = bb(&bs[i])?;
                if b.n[1] <= fb.n[3] + h {
                    right = right.max(b.n[2])
                }
            }
        }
        let mut group = vec![];
        let mut bottom = fb.n[1];
        for i in ids {
            let b = bb(&bs[i])?;
            if b.n[0] < left - h || b.n[2] > right + h {
                continue;
            }
            if b.n[1] - bottom > 1.4 * h
                || leading && b.n[2] - b.n[0] < 0.65 * (right - left)
                || i != seed && strong_caption(text(&bs[i]))
                || candidates::footnote(text(&bs[i]))
            {
                break;
            }
            group.push(i);
            bottom = bottom.max(b.n[3]);
        }
        if group.is_empty() {
            continue;
        }
        let mut merged = merge(bs, &group, false)?;
        merged["content"] = json!(super::super::native_text_assembly::reassemble_members(
            optional(&merged, "_text_lines", 4096)?,
            text(&merged)
        )?);
        merged["type"] = json!("caption");
        merged["_annotation_band_parent_index"] = json!(image);
        removed.extend(&group);
        bs.push(merged);
        if !leading {
            continue;
        }
        let mut separators = vec![];
        for rule in rules {
            let b = super::super::path_graphics::bbox(&rule["bbox"])?;
            if rule["orientation"] == "horizontal"
                && b.n[1] > bottom
                && b.n[2] - b.n[0] >= 0.8 * (right - left)
                && (b.n[0] - left).abs() < 2. * h
            {
                separators.push(b.n[1])
            }
        }
        let Some(end) = separators.into_iter().min_by(f64::total_cmp) else {
            continue;
        };
        let mut legend = vec![];
        for (i, b) in bs.iter().enumerate() {
            if removed.contains(&i)
                || !has_lines(b)
                || !["text", "paragraph_title"].iter().any(|t| b["type"] == *t)
            {
                continue;
            }
            let b = bb(b)?;
            if (bottom..end).contains(&b.n[1]) && (left - h..right).contains(&b.n[0]) {
                legend.push(i)
            }
        }
        let symbols = legend
            .iter()
            .filter(|&&i| {
                let t = text(&bs[i]).trim_matches(python_space);
                t.chars().count() == 1 && alpha(t)
            })
            .count();
        if symbols < 3 {
            continue;
        }
        legend.sort_by(|&i, &j| {
            let a = bb(&bs[i]).unwrap();
            let b = bb(&bs[j]).unwrap();
            cy(&a).total_cmp(&cy(&b)).then(a.n[0].total_cmp(&b.n[0]))
        });
        let mut physical: Vec<Vec<usize>> = vec![];
        for &i in &legend {
            let y = cy(&bb(&bs[i])?);
            let target = physical
                .iter()
                .rposition(|r| (cy(&bb(&bs[r[0]]).unwrap()) - y).abs() < 0.65 * h);
            if let Some(k) = target {
                physical[k].push(i)
            } else {
                physical.push(vec![i])
            }
        }
        let mut rows = vec![];
        for mut physical in physical {
            physical
                .sort_by(|&i, &j| bb(&bs[i]).unwrap().n[0].total_cmp(&bb(&bs[j]).unwrap().n[0]));
            let mut group: Vec<usize> = vec![];
            for i in physical {
                if let Some(&j) = group.last() {
                    if bb(&bs[i])?.n[0] - bb(&bs[j])?.n[2] > 3. * h {
                        rows.push(std::mem::take(&mut group))
                    }
                }
                group.push(i)
            }
            if !group.is_empty() {
                rows.push(group)
            }
        }
        let heading_tops = legend
            .iter()
            .filter(|&&i| bs[i]["type"] == "paragraph_title")
            .map(|&i| bb(&bs[i]).map(|b| b.n[1]))
            .collect::<Result<Vec<_>, _>>()?;
        let mut starts = rows
            .iter()
            .filter(|r| text(&bs[r[0]]).chars().count() > 2)
            .map(|r| bb(&bs[r[0]]).map(|b| b.n[0]))
            .collect::<Result<Vec<_>, _>>()?;
        starts.sort_by(f64::total_cmp);
        let mut cols: Vec<Vec<f64>> = vec![];
        for start in starts {
            if let Some(c) = cols.last() {
                if start - median(c.clone()) <= 1.5 * h {
                    cols.last_mut().unwrap().push(start);
                    continue;
                }
            }
            cols.push(vec![start])
        }
        let stable = cols
            .into_iter()
            .filter(|c| c.len() >= 3)
            .map(median)
            .collect::<Vec<_>>();
        for row in rows {
            let mut note = merge(bs, &row, true)?;
            note["type"] = json!("footnote");
            note["_annotation_band_parent_index"] = json!(image);
            let b = bb(&note)?;
            if stable.len() >= 2 && b.n[2] - b.n[0] < 0.65 * (right - left) {
                let section = heading_tops
                    .iter()
                    .copied()
                    .filter(|&top| top <= b.n[1] + 0.25 * h)
                    .fold(bottom, f64::max);
                let mut lane = *stable
                    .iter()
                    .min_by(|&&a, &&c| (a - b.n[0]).abs().total_cmp(&(c - b.n[0]).abs()))
                    .unwrap();
                if row.iter().any(|&i| bs[i]["type"] == "paragraph_title") {
                    lane = left - h
                }
                note["_legend_sort_band"] = json!([section, lane]);
            }
            bs.push(note);
            removed.extend(row)
        }
    }
    let mut remap = BTreeMap::new();
    let mut out = vec![];
    for (i, b) in bs.drain(..).enumerate() {
        if !removed.contains(&i) {
            remap.insert(i, out.len());
            out.push(b)
        }
    }
    for b in &mut out {
        if let Some(i) = b.get("_annotation_band_parent_index") {
            let i = i.as_u64().ok_or_else(invalid)? as usize;
            b["_annotation_band_parent_index"] =
                json!(remap.get(&i).copied().ok_or_else(invalid)?);
        }
    }
    *bs = out;
    Ok(())
}
