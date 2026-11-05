//! Original hard-split visual-row restoration before formula claiming.
use super::super::{native_page_preparation as prep, page_containers::media::glyph_width};
use super::*;
use std::collections::BTreeMap;
pub(super) fn restore(p: &mut Page, w: &mut Work) -> Result<Vec<Value>, Error> {
    if p.lines.len() < 3 {
        return Ok(p.lines.clone());
    }
    let mut keys = BTreeMap::new();
    let angles = p
        .lines
        .iter()
        .map(|l| integer(l, "angle", 0))
        .collect::<Result<BTreeSet<_>, _>>()?;
    for a in angles {
        let rows = p
            .lines
            .iter()
            .enumerate()
            .filter(|(_, l)| integer(l, "angle", 0).ok() == Some(a))
            .map(|(index, l)| {
                Ok(Row {
                    index,
                    b: local(&bbox(&l["bbox"])?, p.size, a),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        if rows.is_empty() {
            continue;
        }
        let h = median(rows.iter().map(|r| p.h(r)).collect::<Result<Vec<_>, _>>()?);
        let raw = super::super::table_text_styles::lanes::infer_formula_layout(
            &json!({"line_geometry":rows.iter().map(|r|json!([super::super::native_titles::reference_lane_line(p.l(r)),r.b.raw])).collect::<Vec<_>>(),"local_page_width":if a==90||a==270{p.size[1]}else{p.size[0]},"median_height":h}),
            w,
        )?;
        for (j, lane) in array(&raw, 4096)?.iter().enumerate() {
            if lane["is_span"] == true {
                continue;
            }
            for i in array(&lane["line_indices"], 4096)? {
                let r = rows
                    .get(i.as_u64().ok_or_else(invalid)? as usize)
                    .ok_or_else(invalid)?;
                keys.insert(p.id(r)?, (a, j));
            }
        }
    }
    let mut groups: Vec<((i64, i64), Vec<usize>)> = Vec::new();
    for (i, l) in p.lines.iter().enumerate() {
        if l["visual_row_id"].is_null() {
            continue;
        }
        let key = (integer(l, "angle", 0)?, integer(l, "visual_row_id", 0)?);
        if let Some((_, members)) = groups.iter_mut().find(|(k, _)| *k == key) {
            members.push(i)
        } else {
            groups.push((key, vec![i]))
        }
    }
    let mut restored = Vec::new();
    let mut consumed = BTreeSet::new();
    for (_, members) in groups {
        if !can(p, &members, &keys, w)? {
            continue;
        }
        let mut ordered = members
            .iter()
            .map(|&i| {
                Ok(Row {
                    index: i,
                    b: local(
                        &bbox(&p.lines[i]["bbox"])?,
                        p.size,
                        integer(&p.lines[i], "angle", 0)?,
                    ),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        ordered.sort_by(|a, b| {
            a.b.n[0]
                .total_cmp(&b.b.n[0])
                .then(a.b.n[1].total_cmp(&b.b.n[1]))
                .then(p.id(a).unwrap().cmp(&p.id(b).unwrap()))
        });
        let bounds = p
            .lines
            .iter()
            .map(|l| Ok(local(&bbox(&l["bbox"])?, p.size, integer(l, "angle", 0)?)))
            .collect::<Result<Vec<_>, Error>>()?;
        restored.push(prep::formula_dense_materialize(
            &ordered.iter().map(|r| r.index).collect::<Vec<_>>(),
            &mut p.lines,
            &bounds,
            p.size,
            w,
        )?);
        for &i in &members {
            consumed.insert(integer(&p.lines[i], "source_index", 0)?);
        }
    }
    let mut out = p
        .lines
        .iter()
        .filter(|l| !consumed.contains(&integer(l, "source_index", 0).unwrap()))
        .cloned()
        .collect::<Vec<_>>();
    out.extend(restored);
    out.sort_by(|a, b| {
        let aa = integer(a, "angle", 0).unwrap();
        let ba = integer(b, "angle", 0).unwrap();
        let ab = local(&bbox(&a["bbox"]).unwrap(), p.size, aa);
        let bb = local(&bbox(&b["bbox"]).unwrap(), p.size, ba);
        aa.cmp(&ba)
            .then(ab.n[1].total_cmp(&bb.n[1]))
            .then(ab.n[0].total_cmp(&bb.n[0]))
            .then(
                integer(a, "source_index", 0)
                    .unwrap()
                    .cmp(&integer(b, "source_index", 0).unwrap()),
            )
    });
    Ok(out)
}
fn can(
    p: &mut Page,
    members: &[usize],
    keys: &BTreeMap<i64, (i64, usize)>,
    w: &mut Work,
) -> Result<bool, Error> {
    if members.len() < 2
        || members.iter().any(|&i| {
            p.lines[i]["split_from_row"] != true || p.lines[i]["preserve_split_boundary"] == true
        })
    {
        return Ok(false);
    }
    let first = &p.lines[members[0]];
    if members.iter().any(|&i| {
        !eq(&p.lines[i]["semantic_type"], &first["semantic_type"])
            || p.lines[i]["formula_candidate_only"] != first["formula_candidate_only"]
            || !eq(&p.lines[i]["font_signature"], &first["font_signature"])
    }) {
        return Ok(false);
    }
    let mut ordered = members.to_vec();
    ordered.sort_by_key(|&i| integer(&p.lines[i], "run_index", 0).unwrap());
    if ordered
        .iter()
        .enumerate()
        .any(|(j, &i)| integer(&p.lines[i], "run_index", 0).ok() != Some(j as i64))
    {
        return Ok(false);
    }
    let key = keys.get(&integer(&p.lines[ordered[0]], "source_index", 0)?);
    let same = key.is_some()
        && ordered
            .iter()
            .all(|&i| keys.get(&integer(&p.lines[i], "source_index", 0).unwrap()) == key);
    let mut rows = Vec::new();
    for &i in &ordered {
        let l = &p.lines[i];
        let raw = bbox(&l["bbox"])?;
        w.charge(p.tables.len() + 1)?;
        if p.tables.iter().any(|t| intersects(&raw, t)) {
            return Ok(false);
        }
        rows.push(Row {
            index: i,
            b: local(&raw, p.size, integer(l, "angle", 0)?),
        })
    }
    rows.sort_by(|a, b| {
        a.b.n[0]
            .total_cmp(&b.b.n[0])
            .then(a.b.n[1].total_cmp(&b.b.n[1]))
            .then(p.id(a).unwrap().cmp(&p.id(b).unwrap()))
    });
    if !same {
        let widths = rows.iter().map(|r| r.b.n[2] - r.b.n[0]).collect::<Vec<_>>();
        if rows.len() == 2
            && widths.iter().copied().reduce(f64::min).unwrap()
                > 0.35 * widths.iter().copied().reduce(f64::max).unwrap()
            || (3..=6).contains(&rows.len())
        {
            return Ok(false);
        }
    }
    let h = median(rows.iter().map(|r| p.h(r)).collect::<Result<Vec<_>, _>>()?);
    let mut widths = Vec::new();
    for r in &rows {
        if let Some(width) = glyph_width(&mut p.lines[r.index], p.size, w)? {
            widths.push(width)
        }
    }
    let width = if widths.is_empty() {
        0.
    } else {
        median(widths)
    };
    let gap = if same {
        12f64.max(1.75 * h).max(3. * width)
    } else {
        8f64.max(2. * h).max(2.5 * width)
    };
    for pair in rows.windows(2) {
        if !prep::formula_same_baseline(&pair[0].b, p.h(&pair[0])?, &pair[1].b, p.h(&pair[1])?, gap)
        {
            return Ok(false);
        }
    }
    let b = p.bounds(&rows)?;
    let occupied = rows.iter().map(|r| r.b.n[2] - r.b.n[0]).sum::<f64>();
    let threshold = if rows.len() == 2 {
        if same {
            0.8
        } else {
            0.85
        }
    } else {
        0.65
    };
    Ok(occupied / (b.n[2] - b.n[0]).max(0.1) >= threshold)
}
