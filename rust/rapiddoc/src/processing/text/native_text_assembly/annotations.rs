//! Original multi-line title, fragmented marginal, front-matter and compact-title block rules.
use super::*;
use super::{block_common as c, member_blocks as m};
fn height(b: &Value, minimum: bool) -> Result<f64, Error> {
    let h = c::heights(b, false)?;
    if !h.is_empty() {
        median(h)
    } else {
        let b = bbox(&b["bbox"])?;
        let h = b.n[3] - b.n[1];
        Ok(if minimum { h.max(0.1) } else { h })
    }
}
pub(super) fn titles(bs: Vec<Value>) -> Result<Vec<Value>, Error> {
    let (mut used, mut replacements) = (BTreeSet::new(), BTreeMap::new());
    for ty in ["doc_title", "paragraph_title"] {
        let mut ids = bs
            .iter()
            .enumerate()
            .filter(|(_, b)| b["type"] == ty && b["bbox"].is_array())
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        ids.sort_by(|&i, &j| {
            let a = bbox(&bs[i]["bbox"]).unwrap();
            let b = bbox(&bs[j]["bbox"]).unwrap();
            a.n[1].total_cmp(&b.n[1]).then(a.n[0].total_cmp(&b.n[0]))
        });
        let mut groups: Vec<Vec<usize>> = vec![];
        for i in ids {
            let Some(group) = groups.last() else {
                groups.push(vec![i]);
                continue;
            };
            let j = *group.last().unwrap();
            let (a, b) = (&bs[j], &bs[i]);
            let al = m::lines(a)?;
            let bl = m::lines(b)?;
            if ty == "paragraph_title"
                && !bl.is_empty()
                && m::truth(&bl[0]["explicit_section_title"])
                && !bl[0]["title_band_id"].is_null()
                && (al.is_empty()
                    || !eq(
                        &al.last().unwrap()["title_band_id"],
                        &bl[0]["title_band_id"],
                    ))
            {
                groups.push(vec![i]);
                continue;
            }
            let ab = bbox(&a["bbox"])?;
            let bb = bbox(&b["bbox"])?;
            let h = height(a, false)?.max(height(b, false)?);
            let mut gap = bb.n[1] - ab.n[3];
            if gap < -0.2 * h
                && m::truth(&a["_output_bbox_repaired"])
                && m::truth(&b["_output_bbox_repaired"])
            {
                let ar = optional(a, "_local_output_line_bboxes", 4096)?;
                let br = optional(b, "_local_output_line_bboxes", 4096)?;
                if !ar.is_empty() && !br.is_empty() {
                    gap = bbox(&br[0])?.n[1] - bbox(ar.last().unwrap())?.n[3]
                }
            }
            let shared = !al.is_empty()
                && !bl.is_empty()
                && m::truth(&al.last().unwrap()["structural_title"])
                && m::truth(&bl[0]["structural_title"])
                && !bl[0]["title_band_id"].is_null()
                && eq(
                    &al.last().unwrap()["title_band_id"],
                    &bl[0]["title_band_id"],
                );
            if (-0.2 * h..=(if shared { 1. } else { 0.4 }) * h).contains(&gap)
                && axis(&ab, &bb, 0) >= 0.2
                && (!c::conflict(a, b) || shared)
            {
                groups.last_mut().unwrap().push(i)
            } else {
                groups.push(vec![i])
            }
        }
        for group in groups {
            if group.len() < 2 {
                continue;
            }
            replacements.insert(group[0], c::merge(&bs, &group, false)?);
            used.extend(group.into_iter().skip(1));
        }
    }
    Ok(c::replace(bs, replacements, used))
}
pub(super) fn headers(bs: Vec<Value>) -> Result<Vec<Value>, Error> {
    let mut groups: Vec<((i64, i64), Vec<usize>)> = vec![];
    for (i, b) in bs.iter().enumerate() {
        if !["header", "footer"].iter().any(|s| b["type"] == *s)
            || !(b["_single_run_row_id"].is_i64() || b["_single_run_row_id"].is_boolean())
        {
            continue;
        }
        let id = b["_single_run_row_id"]
            .as_i64()
            .or_else(|| b["_single_run_row_id"].as_bool().map(i64::from))
            .unwrap();
        let key = (c::angle(b)?, id);
        if let Some((_, g)) = groups.iter_mut().find(|(k, _)| *k == key) {
            g.push(i)
        } else {
            groups.push((key, vec![i]))
        }
    }
    let (mut used, mut replacements) = (BTreeSet::new(), BTreeMap::new());
    for (_, mut ids) in groups {
        ids.sort_by(|&i, &j| {
            num(&bs[i]["bbox"][0])
                .unwrap()
                .total_cmp(&num(&bs[j]["bbox"][0]).unwrap())
        });
        let mut components: Vec<Vec<usize>> = vec![];
        for i in ids {
            let b = bbox(&bs[i]["bbox"])?;
            let h = height(&bs[i], true)?;
            if bs[i]["type"] == "header" && b.n[2] - b.n[0] > 1.25 * h {
                continue;
            }
            if let Some(g) = components.last() {
                let j = *g.last().unwrap();
                let a = bbox(&bs[j]["bbox"])?;
                if b.n[0] - a.n[2] <= 4. * h.max(height(&bs[j], true)?) && axis(&a, &b, 1) >= 0.5 {
                    components.last_mut().unwrap().push(i);
                    continue;
                }
            }
            components.push(vec![i])
        }
        for g in components {
            if g.len() < 2 {
                continue;
            }
            let mut b = c::merge(&bs, &g, true)?;
            b["_single_run_row_id"] = Value::Null;
            replacements.insert(g[0], b);
            used.extend(g.into_iter().skip(1));
        }
    }
    Ok(c::replace(bs, replacements, used))
}
pub(super) fn front(bs: Vec<Value>, p: [f64; 2], page: Option<i64>) -> Result<Vec<Value>, Error> {
    if page != Some(0) || p[0] <= 0. || p[1] <= 0. {
        return Ok(bs);
    }
    let titles = bs
        .iter()
        .filter(|b| b["type"] == "doc_title" && b["bbox"].is_array())
        .map(|b| num(&b["bbox"][3]))
        .collect::<Result<Vec<_>, _>>()?;
    if titles.is_empty() {
        return Ok(bs);
    }
    let bottom = titles.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let mut ids = vec![];
    for (i, b) in bs.iter().enumerate() {
        if b["type"] != "text" || !b["bbox"].is_array() {
            continue;
        }
        let b = bbox(&b["bbox"])?;
        if bottom < b.n[1]
            && b.n[3] <= (0.4 * p[1]).min(bottom + 0.22 * p[1])
            && b.n[2] - b.n[0] <= 0.32 * p[0]
            && b.n[3] - b.n[1] <= 0.035 * p[1]
        {
            ids.push(i)
        }
    }
    if ids.len() < 9 {
        return Ok(bs);
    }
    let h = median(
        ids.iter()
            .map(|&i| {
                let b = bbox(&bs[i]["bbox"]).unwrap();
                b.n[3] - b.n[1]
            })
            .collect(),
    )?;
    let mut ordered = ids.clone();
    ordered.sort_by(|&i, &j| {
        let a = bbox(&bs[i]["bbox"]).unwrap();
        let b = bbox(&bs[j]["bbox"]).unwrap();
        cy(&a).total_cmp(&cy(&b)).then(a.n[0].total_cmp(&b.n[0]))
    });
    let mut rows: Vec<Vec<usize>> = vec![];
    for i in ordered {
        let y = cy(&bbox(&bs[i]["bbox"])?);
        let mut target = None;
        for (k, row) in rows.iter().enumerate() {
            if (y - median(
                row.iter()
                    .map(|&i| cy(&bbox(&bs[i]["bbox"]).unwrap()))
                    .collect(),
            )?)
            .abs()
                <= 0.6 * h
            {
                target = Some(k);
                break;
            }
        }
        if let Some(k) = target {
            rows[k].push(i)
        } else {
            rows.push(vec![i])
        }
    }
    let dense = rows
        .into_iter()
        .filter(|r| {
            if !(3..=6).contains(&r.len()) {
                return false;
            }
            let left = r
                .iter()
                .map(|&i| num(&bs[i]["bbox"][0]).unwrap())
                .fold(f64::INFINITY, f64::min);
            let right = r
                .iter()
                .map(|&i| num(&bs[i]["bbox"][2]).unwrap())
                .fold(f64::NEG_INFINITY, f64::max);
            right - left >= 0.55 * p[0]
        })
        .collect::<Vec<_>>();
    if dense.len() < 2 {
        return Ok(bs);
    }
    let anchor = dense
        .iter()
        .min_by(|a, b| {
            b.len().cmp(&a.len()).then(
                median(
                    a.iter()
                        .map(|&i| cy(&bbox(&bs[i]["bbox"]).unwrap()))
                        .collect(),
                )
                .unwrap()
                .total_cmp(
                    &median(
                        b.iter()
                            .map(|&i| cy(&bbox(&bs[i]["bbox"]).unwrap()))
                            .collect(),
                    )
                    .unwrap(),
                ),
            )
        })
        .unwrap();
    if anchor.len() != 4 {
        return Ok(bs);
    }
    let mut centers = anchor
        .iter()
        .map(|&i| cy_x(&bs[i]))
        .collect::<Result<Vec<_>, _>>()?;
    centers.sort_by(f64::total_cmp);
    let boundaries = centers
        .windows(2)
        .map(|p| 0.5 * (p[0] + p[1]))
        .collect::<Vec<_>>();
    let top = dense
        .iter()
        .flatten()
        .map(|&i| num(&bs[i]["bbox"][1]).unwrap())
        .fold(f64::INFINITY, f64::min)
        - h;
    let bottom = dense
        .iter()
        .flatten()
        .map(|&i| num(&bs[i]["bbox"][3]).unwrap())
        .fold(f64::NEG_INFINITY, f64::max)
        + h;
    let mut groups = vec![vec![]; 4];
    for i in ids {
        let b = bbox(&bs[i]["bbox"])?;
        if !(top..=bottom).contains(&cy(&b)) {
            continue;
        }
        let k = boundaries.iter().filter(|&&x| cx(&b) > x).count();
        if k < groups.len() {
            groups[k].push(i)
        }
    }
    if groups.iter().any(|g| g.len() < 3) {
        return Ok(bs);
    }
    let (mut used, mut replacements) = (BTreeSet::new(), BTreeMap::new());
    for mut group in groups {
        group.sort_by(|&i, &j| {
            let a = bbox(&bs[i]["bbox"]).unwrap();
            let b = bbox(&bs[j]["bbox"]).unwrap();
            a.n[1].total_cmp(&b.n[1]).then(a.n[0].total_cmp(&b.n[0]))
        });
        replacements.insert(group[0], c::merge(&bs, &group, false)?);
        used.extend(group.into_iter().skip(1));
    }
    Ok(c::replace(bs, replacements, used))
}
fn cy_x(b: &Value) -> Result<f64, Error> {
    Ok(cx(&bbox(&b["bbox"])?))
}
pub(super) fn compact(bs: Vec<Value>, p: [f64; 2]) -> Result<Vec<Value>, Error> {
    let mut pairs = vec![];
    for (i, a) in bs.iter().enumerate() {
        if a["type"] != "paragraph_title"
            || !a["bbox"].is_array()
            || c::rows(a)?.len() < 2
            || optional(a, "_font_signatures", 4096)?.is_empty()
        {
            continue;
        }
        let angle = c::angle(a)?;
        let page = if angle == 90 || angle == 270 {
            p[1]
        } else {
            p[0]
        };
        let hs = c::heights(a, true)?;
        let h = if hs.is_empty() { 0. } else { median(hs)? };
        let ab = bbox(&a["bbox"])?;
        if h <= 0. || ab.n[2] - ab.n[0] > 0.55 * page {
            continue;
        }
        let mut matches = vec![];
        for (j, b) in bs.iter().enumerate() {
            if b["type"] != "text"
                || c::angle(b)? != angle
                || !b["bbox"].is_array()
                || !c::conflict(a, b)
            {
                continue;
            }
            let bb = bbox(&b["bbox"])?;
            if bb.n[2] - bb.n[0] > 0.6 * page {
                continue;
            }
            let gap = bb.n[1] - ab.n[3];
            if (-0.25 * h..=0.6 * h).contains(&gap) && (bb.n[0] - ab.n[0]).abs() <= 0.75 * h {
                matches.push((gap.max(0.), j))
            }
        }
        if let Some((_, j)) = matches.into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            pairs.push((i, j, h))
        }
    }
    let (mut used, mut replacements) = (BTreeSet::new(), BTreeMap::new());
    for &(i, j, h) in &pairs {
        let left = num(&bs[i]["bbox"][0])?;
        let support = pairs
            .iter()
            .filter(|&&(other, _, oh)| {
                (num(&bs[other]["bbox"][0]).unwrap() - left).abs() <= h.max(oh)
                    && (0.75..=1.25).contains(&(oh / h))
            })
            .count();
        if support < 2 || used.contains(&i) || used.contains(&j) {
            continue;
        }
        let mut merged = c::merge(&bs, &[i, j], false)?;
        merged["type"] = json!("text");
        replacements.insert(i.min(j), merged);
        used.insert(i);
        used.insert(j);
    }
    Ok(c::replace(bs, replacements, used))
}
