//! 旧栅格图片完整认领：照片框去重、纵向切片、同行间隔符、签名及原生背景排除。
use super::*;
use crate::legacy_model::path_graphics::{area, integer, overlap_first, overlap_smaller};
use std::collections::{BTreeMap, BTreeSet};
fn center(b: &Box4) -> [f64; 2] {
    [(b.n[0] + b.n[2]) / 2., (b.n[1] + b.n[3]) / 2.]
}
fn inside(p: [f64; 2], b: &Box4) -> bool {
    b.n[0] <= p[0] && p[0] <= b.n[2] && b.n[1] <= p[1] && p[1] <= b.n[3]
}
fn ordered(a: &Box4, b: &Box4) -> std::cmp::Ordering {
    [1, 0, 3, 2]
        .into_iter()
        .map(|i| a.n[i].partial_cmp(&b.n[i]).unwrap())
        .find(|o| !o.is_eq())
        .unwrap_or(std::cmp::Ordering::Equal)
}
fn near(a: &Box4, b: &Box4) -> bool {
    a.n.iter().zip(b.n).all(|(a, b)| (a - b).abs() <= 0.5)
}
fn photos(bs: Vec<Box4>, ls: &[Value], w: &mut Work) -> Result<Vec<Box4>, Error> {
    let mut consumed = BTreeSet::new();
    for (i, a) in bs.iter().enumerate() {
        let (width, height) = (a.n[2] - a.n[0], a.n[3] - a.n[1]);
        if width.min(height) <= 0. {
            continue;
        }
        for (j, b) in bs.iter().enumerate() {
            w.charge(1)?;
            if i == j
                || area(b) >= area(a)
                || overlap_first(b, a) < 0.999
                || area(b) / area(a) < 0.85
                || !(0.0..=0.08 * width).contains(&(b.n[0] - a.n[0]))
                || !(0.0..=0.08 * width).contains(&(a.n[2] - b.n[2]))
                || !(0.0..=0.08 * height).contains(&(b.n[1] - a.n[1]))
                || !(0.0..=0.08 * height).contains(&(a.n[3] - b.n[3]))
            {
                continue;
            }
            w.charge(ls.len() + 1)?;
            let mut boundary = false;
            for l in ls {
                let p = center(&bbox(&l["bbox"])?);
                if inside(p, a) && !inside(p, b) {
                    boundary = true;
                    break;
                }
            }
            if !boundary {
                consumed.insert(j);
            }
        }
    }
    Ok(bs
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !consumed.contains(i))
        .map(|(_, b)| b)
        .collect())
}
fn tiles(bs: Vec<Box4>, sz: [f64; 2], w: &mut Work) -> Result<Vec<Box4>, Error> {
    let pa = sz[0].max(0.) * sz[1].max(0.);
    if bs.len() < 3 || pa <= 0. {
        return Ok(bs);
    }
    let tol = 0.75f64.max(0.002 * sz[0]);
    let mut indices: Vec<_> = (0..bs.len()).collect();
    indices.sort_by(|&i, &j| {
        [0, 2, 1]
            .into_iter()
            .map(|k| bs[i].n[k].partial_cmp(&bs[j].n[k]).unwrap())
            .find(|o| !o.is_eq())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in indices {
        let mut target = None;
        for (j, g) in groups.iter().enumerate() {
            w.charge(g.len() * 16 + 1)?;
            if (bs[i].n[0] - median(g.iter().map(|&i| bs[i].n[0]).collect())).abs() <= tol
                && (bs[i].n[2] - median(g.iter().map(|&i| bs[i].n[2]).collect())).abs() <= tol
            {
                target = Some(j);
                break;
            }
        }
        if let Some(j) = target {
            groups[j].push(i)
        } else {
            groups.push(vec![i]);
        }
    }
    let mut merged = Vec::new();
    let mut consumed = BTreeSet::new();
    for mut g in groups {
        g.sort_by(|&i, &j| {
            bs[i].n[1]
                .partial_cmp(&bs[j].n[1])
                .unwrap()
                .then(bs[i].n[3].partial_cmp(&bs[j].n[3]).unwrap())
        });
        let mut segments: Vec<Vec<usize>> = Vec::new();
        for i in g {
            if let Some(seg) = segments.last_mut() {
                let p = &bs[*seg.last().unwrap()];
                let b = &bs[i];
                let gap =
                    1f64.max(0.5 * (p.n[3] - p.n[1]).max(0.1).max((b.n[3] - b.n[1]).max(0.1)));
                if b.n[1] - p.n[3] <= gap {
                    seg.push(i);
                    continue;
                }
            }
            segments.push(vec![i]);
        }
        for seg in segments {
            if seg.len() < 3 {
                continue;
            }
            let b = combined(seg.iter().map(|&i| &bs[i]))?;
            let coverage_heights = seg
                .iter()
                .map(|&i| (bs[i].n[3] - bs[i].n[1]).max(0.))
                .collect::<Vec<_>>();
            let coverage = crate::legacy_model::table_native_candidate::sum(&coverage_heights)
                / (b.n[3] - b.n[1]).max(0.1);
            if area(&b) / pa < 0.0038
                || b.n[2] - b.n[0] < 0.12 * sz[0]
                || b.n[2] - b.n[0] > 0.9 * sz[0]
                || coverage < 0.9
            {
                continue;
            }
            merged.push(b);
            consumed.extend(seg);
        }
    }
    merged.extend(
        bs.into_iter()
            .enumerate()
            .filter(|(i, _)| !consumed.contains(i))
            .map(|(_, b)| b),
    );
    merged.sort_by(ordered);
    Ok(merged)
}
fn gap(
    ls: &[Value],
    a: &Box4,
    b: &Box4,
    prior: &BTreeSet<i64>,
    em: f64,
    w: &mut Work,
) -> Result<Option<usize>, Error> {
    let ah = (a.n[3] - a.n[1]).max(0.1);
    let bh = (b.n[3] - b.n[1]).max(0.1);
    if ah.max(bh) / ah.min(bh) > 1.25
        || (a.n[3].min(b.n[3]) - a.n[1].max(b.n[1])).max(0.) / ah.min(bh) < 0.8
        || !(0.0..=1.5 * em).contains(&(b.n[0] - a.n[2]))
    {
        return Ok(None);
    }
    let t = 0.5f64.max(0.25 * em);
    let top = a.n[1].max(b.n[1]);
    let bottom = a.n[3].min(b.n[3]);
    let mut out = None;
    for (i, l) in ls.iter().enumerate() {
        w.charge(1)?;
        if prior.contains(&integer(l, "source_index", 0)?)
            || integer(l, "angle", 0)? != 0
            || l["split_from_row"] != true
            || l["visual_row_id"].is_null()
        {
            continue;
        }
        let line = bbox(&l["bbox"])?;
        let p = center(&line);
        if (a.n[2] - t..=b.n[0] + t).contains(&p[0]) && (top - t..=bottom + t).contains(&p[1]) {
            if out.is_some() {
                return Ok(None);
            }
            out = Some(i);
        }
    }
    Ok(out.filter(|&i| {
        let line = bbox(&ls[i]["bbox"]).unwrap();
        (line.n[0] - a.n[2]).abs() <= t && (line.n[2] - b.n[0]).abs() <= t
    }))
}
fn inline(
    bs: Vec<Box4>,
    ls: &[Value],
    containers: &[Box4],
    prior: &BTreeSet<i64>,
    w: &mut Work,
) -> Result<Vec<(Box4, Option<i64>)>, Error> {
    if bs.len() < 3 {
        return Ok(bs.into_iter().map(|b| (b, None)).collect());
    }
    let mut hs = Vec::new();
    for l in ls {
        if integer(l, "angle", 0)? == 0 && !prior.contains(&integer(l, "source_index", 0)?) {
            hs.push(height(l, &bbox(&l["bbox"])?)?)
        }
    }
    let em = if hs.is_empty() { 1. } else { median(hs) };
    let mut adj = vec![BTreeSet::new(); bs.len()];
    let mut gaps = BTreeMap::new();
    for (i, a) in bs.iter().enumerate() {
        for (j, b) in bs.iter().enumerate() {
            w.charge(1)?;
            if i == j || a.n[0] >= b.n[0] {
                continue;
            }
            if let Some(k) = gap(ls, a, b, prior, em, w)? {
                adj[i].insert(j);
                adj[j].insert(i);
                gaps.insert((i, j), k);
            }
        }
    }
    let mut visited = BTreeSet::new();
    let mut consumed = BTreeSet::new();
    let mut merged = Vec::new();
    for start in 0..bs.len() {
        if visited.contains(&start) {
            continue;
        }
        let mut component = Vec::new();
        let mut pending = vec![start];
        while let Some(i) = pending.pop() {
            if !visited.insert(i) {
                continue;
            }
            component.push(i);
            pending.extend(adj[i].difference(&visited).copied());
        }
        if component.len() < 3 {
            continue;
        }
        component.sort_by(|&i, &j| bs[i].n[0].partial_cmp(&bs[j].n[0]).unwrap());
        let members: Option<Vec<usize>> = component
            .windows(2)
            .map(|p| gaps.get(&(p[0], p[1])).copied())
            .collect();
        let Some(members) = members else { continue };
        let rows: BTreeSet<_> = members
            .iter()
            .map(|&i| ls[i]["visual_row_id"].as_i64())
            .collect();
        if rows.len() != 1 || rows.contains(&None) {
            continue;
        }
        let mut boxes = component.iter().map(|&i| bs[i].clone()).collect::<Vec<_>>();
        for &i in &members {
            boxes.push(bbox(&ls[i]["bbox"])?)
        }
        let group = combined(&boxes)?;
        w.charge(containers.len() + ls.len() + 1)?;
        if containers.iter().any(|b| overlap_smaller(&group, b) >= 0.5) {
            continue;
        }
        let ids: BTreeSet<_> = members
            .iter()
            .map(|&i| integer(&ls[i], "source_index", 0).unwrap())
            .collect();
        let mut unexpected = false;
        for l in ls {
            let id = integer(l, "source_index", 0)?;
            if prior.contains(&id) || ids.contains(&id) {
                continue;
            }
            let p = center(&bbox(&l["bbox"])?);
            if inside(p, &group) && !component.iter().any(|&i| inside(p, &bs[i])) {
                unexpected = true;
                break;
            }
        }
        if unexpected {
            continue;
        }
        merged.push((group, *rows.first().unwrap()));
        consumed.extend(component);
    }
    merged.extend(
        bs.into_iter()
            .enumerate()
            .filter(|(i, _)| !consumed.contains(i))
            .map(|(_, b)| (b, None)),
    );
    merged.sort_by(|(a, _), (b, _)| ordered(a, b));
    Ok(merged)
}
#[cfg(test)]
pub(super) fn build(v: &Value, w: &mut Work) -> Result<Value, Error> {
    build_owned(v.clone(), w)
}
pub(super) fn build_owned(mut v: Value, w: &mut Work) -> Result<Value, Error> {
    let mut source = v["source"].take();
    let sz = size(&source["page_size"])?;
    let pa = sz[0].max(0.) * sz[1].max(0.);
    let mut ls = optional(&source, "lines", 4096)?.to_vec();
    let prior = array(&v["claimed_line_indices"], 20000)?
        .iter()
        .map(|i| i.as_i64().ok_or_else(invalid))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut containers = Vec::new();
    for b in array(&v["container_blocks"], 4000)? {
        if let Some(b) = coerce(&b["bbox"])? {
            containers.push(b)
        }
    }
    if pa <= 0. {
        return Ok(crate::processing::with_source(
            source,
            json!({"raster_image_blocks":[],"claimed_line_indices":[]}),
        ));
    }
    let mut signatures: Vec<Box4> = Vec::new();
    for raw in optional(&source, "signature_bboxes", 5000)? {
        if let Some(b) = coerce(raw)? {
            if let Some(b) = clip(&b, sz)? {
                w.charge(containers.len() + signatures.len() + 1)?;
                if !containers.iter().any(|c| overlap_smaller(&b, c) >= 0.5)
                    && !signatures.iter().any(|s| near(&b, s))
                {
                    signatures.push(b)
                }
            }
        }
    }
    let mut rasters = Vec::new();
    for raw in optional(&source, "image_bboxes", 5000)? {
        if let Some(b) = coerce(raw)? {
            if let Some(b) = clip(&b, sz)? {
                rasters.push(b)
            }
        }
    }
    let mut hs = Vec::new();
    for l in &ls {
        if integer(l, "angle", 0)? == 0 {
            hs.push(height(l, &bbox(&l["bbox"])?)?)
        }
    }
    let has_height = !hs.is_empty();
    let em = if has_height { median(hs) } else { 10. };
    let mut admitted = Vec::new();
    for b in tiles(photos(rasters, &ls, w)?, sz, w)? {
        w.charge(ls.len() + containers.len() + signatures.len() + 1)?;
        let mut count = 0;
        let mut chars = 0;
        let mut long = 0;
        let mut any_overlap = false;
        for l in &ls {
            let lb = bbox(&l["bbox"])?;
            if overlap_first(&lb, &b) > 0.01 {
                any_overlap = true
            }
            if integer(l, "angle", 0)? == 0 && overlap_first(&lb, &b) >= 0.9 {
                count += 1;
                let n = l["text"].as_str().ok_or_else(invalid)?.chars().count();
                chars += n;
                if n >= 12 {
                    long += 1
                }
            }
        }
        let _ = count;
        if area(&b) >= 0.6 * pa && long >= 8 && chars >= 300 {
            continue;
        }
        let meaningful = has_height
            && (b.n[2] - b.n[0]).min(b.n[3] - b.n[1]) >= 3. * em
            && area(&b) >= 0.0005 * pa
            && !any_overlap;
        if (area(&b) / pa < 0.0038 && !meaningful)
            || containers.iter().any(|c| overlap_smaller(&b, c) >= 0.5)
            || signatures.iter().any(|s| near(&b, s))
        {
            continue;
        }
        admitted.push(b);
    }
    let mut specs = inline(admitted, &ls, &containers, &prior, w)?;
    specs.extend(signatures.into_iter().map(|b| (b, None)));
    specs.sort_by(|(a, _), (b, _)| ordered(a, b));
    let mut groups = vec![Vec::new(); specs.len()];
    let mut claimed = BTreeSet::new();
    for (i, l) in ls.iter().enumerate() {
        let id = integer(l, "source_index", 0)?;
        if prior.contains(&id) {
            continue;
        }
        w.charge(specs.len() + 1)?;
        let p = center(&bbox(&l["bbox"])?);
        let mut best = None;
        for (j, (b, _)) in specs.iter().enumerate() {
            if inside(p, b)
                && area(b) < 0.6 * pa
                && best.is_none_or(|k: usize| {
                    crate::legacy_model::graphic_arbitration::area_key(b)
                        < crate::legacy_model::graphic_arbitration::area_key(&specs[k].0)
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
    for ((b, row), members) in specs.iter().zip(groups) {
        let mut block = json!({"type":"image","bbox":b.raw,"angle":0,"content":super::media::content(&members,&mut ls,sz,w)?});
        if let Some(row) = row {
            block["_inline_visual_row_id"] = json!(row);
        }
        blocks.push(block);
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
        json!({"raster_image_blocks":blocks,"claimed_line_indices":claimed}),
    ))
}
