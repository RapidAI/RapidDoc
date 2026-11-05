//! 一般矢量图完整候选/行归属/图题和正文保护/最终物化，复用已迁移的检测内核。
use super::*;
use crate::legacy_model::{
    drawing_graphics, graphic_captions, graphic_grid, isolated_graphics, path_graphics,
    path_graphics::{area, integer, overlap_first, overlap_smaller, union},
    raster_charts, raster_diagrams, stacked_graphics,
    table_text_styles::lanes,
};
use crate::legacy_page::eq;
use rapiddoc_native::{python_decimal, text_boundary::python_space};
use std::collections::{BTreeMap, BTreeSet};
struct Candidate {
    b: Box4,
    lane: i64,
    margin: f64,
    strict: bool,
    ids: BTreeSet<i64>,
}
fn cx(b: &Box4) -> f64 {
    (b.n[0] + b.n[2]) / 2.
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) / 2.
}
fn inside(x: f64, y: f64, b: &Box4) -> bool {
    b.n[0] <= x && x <= b.n[2] && b.n[1] <= y && y <= b.n[3]
}
pub(super) fn distance(a: &Box4, b: &Box4) -> f64 {
    rapiddoc_pdf::legacy_vectors::hypot(
        (a.n[0] - b.n[2]).max(b.n[0] - a.n[2]).max(0.),
        (a.n[1] - b.n[3]).max(b.n[1] - a.n[3]).max(0.),
    )
}
fn prefix_section(s: &str) -> bool {
    let c: Vec<_> = s.trim_start_matches(python_space).chars().collect();
    let mut i = 0;
    while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
        i += 1
    }
    if i > 0 && c.get(i).is_some_and(|c| ".、".contains(*c)) {
        i += 1;
        while c.get(i).is_some_and(|c| python_space(*c)) {
            i += 1
        }
        if c.get(i)
            .is_some_and(|c| ('\u{4e00}'..='\u{9fff}').contains(c))
        {
            return true;
        }
    }
    let mut i = usize::from(c.first().is_some_and(|c| "（(".contains(*c)));
    let start = i;
    while c
        .get(i)
        .is_some_and(|c| "一二三四五六七八九十".contains(*c))
    {
        i += 1
    }
    i > start
        && c.get(i).is_some_and(|c| {
            if start == 0 {
                *c == '、'
            } else {
                "）)".contains(*c)
            }
        })
}
fn dates(s: &str) -> bool {
    let c: Vec<_> = s.trim_matches(python_space).chars().collect();
    let mut i = 0;
    let mut count = 0;
    while i < c.len() {
        for _ in 0..4 {
            if !c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
                return false;
            }
            i += 1
        }
        while c.get(i).is_some_and(|c| python_space(*c)) {
            i += 1
        }
        if c.get(i) != Some(&'(') {
            return false;
        }
        i += 1;
        let start = i;
        while c.get(i).is_some_and(|c| c.is_ascii_alphabetic()) {
            i += 1
        }
        if i == start || c.get(i) != Some(&')') {
            return false;
        }
        i += 1;
        while c.get(i).is_some_and(|c| python_space(*c)) {
            i += 1
        }
        count += 1;
    }
    count >= 2
}
fn label(l: &Value, b: &Box4, em: f64, margin: f64) -> Result<bool, Error> {
    let text = l["text"].as_str().ok_or_else(invalid)?;
    if graphic_captions::source_note(text, true) {
        return Ok(false);
    }
    let line = bbox(&l["bbox"])?;
    let ink = bbox(
        l.get("ink_bbox")
            .filter(|v| !v.is_null())
            .unwrap_or(&l["bbox"]),
    )?;
    if overlap_first(&ink, b) < 0.5 && prefix_section(text) {
        return Ok(false);
    }
    if inside(cx(&line), cy(&line), b) {
        return Ok(true);
    }
    let h = l.get("effective_height").map_or(Ok(0.), num)?;
    let h = (if h != 0. {
        h
    } else {
        (line.n[2] - line.n[0])
            .max(0.1)
            .min((line.n[3] - line.n[1]).max(0.1))
    })
    .max(0.1);
    let angle = integer(l, "angle", 0)?;
    let (length, core) = if [90, 270].contains(&angle) {
        (line.n[3] - line.n[1], b.n[3] - b.n[1])
    } else {
        (line.n[2] - line.n[0], b.n[2] - b.n[0])
    };
    let hg = (b.n[0] - line.n[2]).max(line.n[0] - b.n[2]).max(0.);
    let vg = (b.n[1] - line.n[3]).max(line.n[1] - b.n[3]).max(0.);
    if [0, 180].contains(&angle)
        && ((length <= 8. * h && length <= 0.45 * core) || (dates(text) && length <= 1.15 * core))
        && graphic_captions::axis(&line, b, 0) >= 0.15
        && vg <= 2.5 * em
    {
        return Ok(true);
    }
    if length > (5. * h).min(0.5 * core) {
        return Ok(false);
    }
    if graphic_captions::axis(&line, b, 0) >= 0.15 {
        return Ok(vg <= margin * em);
    }
    if graphic_captions::axis(&line, b, 1) >= 0.15 {
        return Ok(hg <= (margin * em).max(0.2 * (b.n[2] - b.n[0])));
    }
    let corner = margin.min(1.5) * em;
    Ok(hg <= corner && vg <= corner && rapiddoc_pdf::legacy_vectors::hypot(hg, vg) <= corner)
}
fn lane(b: &Box4, ls: &[(f64, f64)]) -> usize {
    let mut best = 0;
    let mut score = (-1, -1., f64::NEG_INFINITY);
    for (i, &(l, r)) in ls.iter().enumerate() {
        let inside = l <= cx(b) && cx(b) <= r;
        let next = (
            i32::from(inside),
            (b.n[2].min(r) - b.n[0].max(l)).max(0.),
            if inside {
                0.
            } else {
                -(cx(b) - l).abs().min((cx(b) - r).abs())
            },
        );
        if next > score {
            score = next;
            best = i
        }
    }
    best
}
fn strong_lane(b: &Box4, ls: &[(f64, f64)], em: f64) -> i64 {
    let mut matches = Vec::new();
    for (i, &(l, r)) in ls.iter().enumerate() {
        if (b.n[2].min(r) - b.n[0].max(l)).max(0.) / (b.n[2] - b.n[0]).max(0.1) >= 0.9
            && b.n[0] >= l - em.max(1.)
            && b.n[2] <= r + em.max(1.)
        {
            matches.push(i)
        }
    }
    if matches.len() == 1 {
        matches[0] as i64
    } else {
        -1
    }
}
fn text_lanes(ls: &[Value], sz: [f64; 2], em: f64, w: &mut Work) -> Result<Vec<(f64, f64)>, Error> {
    let mut geometry = Vec::new();
    let mut hs = Vec::new();
    for l in ls {
        if integer(l, "angle", 0)? == 0 {
            let b = bbox(&l["bbox"])?;
            hs.push(height(l, &b)?);
            geometry.push(json!([l, b.raw]));
        }
    }
    if geometry.is_empty() {
        return Ok(vec![(0., sz[0])]);
    }
    let h = if hs.is_empty() { em } else { median(hs) };
    let raw = lanes::infer_layout(
        &json!({"line_geometry":geometry,"local_page_width":sz[0],"median_height":h}),
        w,
    )?;
    let mut lanes = Vec::new();
    for l in array(&raw, 4096)? {
        if l["is_span"] == false {
            lanes.push((num(&l["left"])?, num(&l["right"])?));
        }
    }
    if lanes.is_empty() {
        lanes.push((0., sz[0]));
    }
    Ok(lanes)
}
fn body_tails(
    ls: &[Value],
    cs: &[Candidate],
    lanes: &[(f64, f64)],
    em: f64,
    w: &mut Work,
) -> Result<BTreeSet<i64>, Error> {
    let mut out = BTreeSet::new();
    for tail in ls {
        if integer(tail, "angle", 0)? != 0 {
            continue;
        }
        let tb = bbox(&tail["bbox"])?;
        let k = lane(&tb, lanes);
        let width = (lanes[k].1 - lanes[k].0).max(0.1);
        let tw = tb.n[2] - tb.n[0];
        if tw > 0.5 * width {
            continue;
        }
        let mut matching = Vec::new();
        for c in cs {
            w.charge(1)?;
            if (c.lane < 0 || c.lane == k as i64)
                && tb.n[3] <= c.b.n[1] + 0.25 * em
                && label(tail, &c.b, em, c.margin)?
            {
                matching.push(c)
            }
        }
        if matching.is_empty() {
            continue;
        }
        let th = height(tail, &tb)?;
        for p in ls {
            w.charge(1)?;
            if integer(p, "angle", 0)? != 0 || p["source_index"] == tail["source_index"] {
                continue;
            }
            let pb = bbox(&p["bbox"])?;
            if lane(&pb, lanes) != k
                || !(-0.25 * em..=0.75 * em).contains(&(tb.n[1] - pb.n[3]))
                || (pb.n[0] - tb.n[0]).abs() > 0.75 * em
            {
                continue;
            }
            let pw = pb.n[2] - pb.n[0];
            if pw < 0.75 * width || lanes[k].1 - pb.n[2] > em || pw < 1.5 * tw {
                continue;
            }
            let ph = height(p, &pb)?;
            if ph.max(th) > 1.25 * ph.min(th)
                || (!p["font_signature"].is_null()
                    && !tail["font_signature"].is_null()
                    && !eq(&p["font_signature"], &tail["font_signature"]))
            {
                continue;
            }
            let mut member = false;
            for c in &matching {
                if label(p, &c.b, em, c.margin)? {
                    member = true;
                    break;
                }
            }
            if !member {
                out.insert(integer(tail, "source_index", 0)?);
                break;
            }
        }
    }
    Ok(out)
}
#[cfg(test)]
pub(super) fn build(v: &Value, w: &mut Work) -> Result<Value, Error> {
    build_owned(v.clone(), w)
}
pub(super) fn build_owned(mut v: Value, w: &mut Work) -> Result<Value, Error> {
    let mut source = v["source"].take();
    let sz = size(&source["page_size"])?;
    let prior = array(&v["claimed_line_indices"], 20000)?
        .iter()
        .map(|i| i.as_i64().ok_or_else(invalid))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let barriers = array(&v["excluded_bboxes"], 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let isolated = isolated_graphics::detect_source(&source, &json!({}), w)?;
    let mut boxes = array(&isolated["vector_bboxes"], 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let (fallback, _) = stacked_graphics::measure_with_work(&source, 10., w)?;
    let raster = raster_diagrams::detect_source(&source, &json!({"em":fallback}), w)?;
    for b in array(&raster["raster_diagram_bboxes"], 5000)? {
        boxes.push(bbox(b)?);
    }
    let mut global = optional(&source, "lines", 4096)?.to_vec();
    let positions: Vec<_> = global
        .iter()
        .enumerate()
        .filter(|(_, l)| !prior.contains(&integer(l, "source_index", 0).unwrap_or(0)))
        .map(|(i, _)| i)
        .collect();
    let mut ls: Vec<_> = positions.iter().map(|&i| global[i].clone()).collect();
    let strong = array(&v["strong_core_bboxes"], 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    if ls.len() < 2 {
        let blocks: Vec<_> = boxes
            .iter()
            .filter(|b| !barriers.iter().any(|t| overlap_smaller(b, t) >= 0.5))
            .map(|b| json!({"type":"image","content":"","bbox":b.raw,"angle":0}))
            .collect();
        return Ok(crate::processing::with_source(
            source,
            json!({"graphic_blocks":blocks,"claimed_line_indices":[]}),
        ));
    }
    if optional(&source, "drawing_lines", 5000)?.len() < 4 && strong.is_empty() {
        return Ok(crate::processing::with_source(
            source,
            json!({"graphic_blocks":[],"claimed_line_indices":[]}),
        ));
    }
    let mut hs = Vec::new();
    for l in &ls {
        let b = bbox(&l["bbox"])?;
        let h = l.get("effective_height").map_or(Ok(0.), num)?;
        hs.push(
            (if h != 0. {
                h
            } else {
                (b.n[2] - b.n[0]).max(0.1).min((b.n[3] - b.n[1]).max(0.1))
            })
            .max(0.1),
        );
    }
    let (em, _) = stacked_graphics::measure_with_work(&source, median(hs), w)?;
    let paths = path_graphics::detect_with_work(
        &json!({"page_size":source["page_size"],"path_infos":source["path_infos"],"median_height":em}),
        w,
    )?;
    for raw in array(&paths["container_bboxes"], 5000)? {
        let b = bbox(raw)?;
        w.charge(global.len() + optional(&source, "image_bboxes", 5000)?.len() + 1)?;
        let mut crossed = false;
        for l in &global {
            if overlap_first(&bbox(&l["bbox"])?, &b) > 0.05 {
                crossed = true;
                break;
            }
        }
        let mut raster = false;
        for raw in optional(&source, "image_bboxes", 5000)? {
            if overlap_smaller(&b, &bbox(raw)?) > 0.2 {
                raster = true;
                break;
            }
        }
        if !crossed && !raster {
            boxes.push(b);
        }
    }
    let lanes = text_lanes(&ls, sz, em, w)?;
    let summary = drawing_graphics::detect_with_work(
        &json!({"path_infos":[],"drawing_lines":source["drawing_lines"],"page_size":sz,"median_height":em}),
        w,
    )?;
    let mut lines = Vec::new();
    for c in array(&summary["components"], 5000)? {
        let b = bbox(&c["bbox"])?;
        if array(&c["line_indices"], 5000)?.len() < 4
            || integer(c, "horizontal_count", 0)? < 2
            || integer(c, "vertical_count", 0)? < 2
            || b.n[2] - b.n[0] < 4. * em
            || b.n[3] - b.n[1] < 3. * em
            || b.n[2] - b.n[0] > 0.5 * sz[0]
            || b.n[3] - b.n[1] > 0.5 * sz[1]
            || barriers.iter().any(|t| overlap_smaller(&b, t) >= 0.5)
        {
            continue;
        }
        lines.push(b);
    }
    let flags = raster_diagrams::detect_source(
        &source,
        &json!({"em":em,"frame_bboxes":lines.iter().map(|b|b.raw.clone()).collect::<Vec<_>>()}),
        w,
    )?;
    lines = lines
        .into_iter()
        .zip(array(&flags["plain_text_frame_flags"], 5000)?)
        .filter(|(_, plain)| **plain != true)
        .map(|(b, _)| b)
        .collect();
    let axes = raster_charts::cores_source(&source, w)?;
    let axes = array(&axes["graphic_bboxes"], 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let mut candidates: Vec<_> = lines
        .iter()
        .filter(|b| !strong.iter().any(|c| overlap_smaller(b, c) >= 0.5))
        .map(|b| Candidate {
            b: b.clone(),
            lane: lane(b, &lanes) as i64,
            margin: 2.5,
            strict: false,
            ids: BTreeSet::new(),
        })
        .collect();
    for b in strong {
        if barriers.iter().any(|t| overlap_smaller(&b, t) >= 0.5) {
            continue;
        }
        let margin = if lines
            .iter()
            .any(|l| overlap_smaller(l, &b) >= 0.5 && area(l) >= 0.8 * area(&b))
        {
            2.5
        } else {
            1.
        };
        let strict = axes
            .iter()
            .any(|a| overlap_first(&b, a) >= 0.98 && overlap_first(a, &b) >= 0.98);
        candidates.push(Candidate {
            lane: strong_lane(&b, &lanes, em),
            b,
            margin,
            strict,
            ids: BTreeSet::new(),
        });
    }
    if candidates.is_empty() {
        return Ok(crate::processing::with_source(
            source,
            json!({"graphic_blocks":[],"claimed_line_indices":[]}),
        ));
    }
    if candidates.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut keys = BTreeMap::new();
    for (i, l) in ls.iter().enumerate() {
        let b = bbox(&l["bbox"])?;
        let a = integer(l, "angle", 0)?;
        let k = lane(&b, &lanes);
        let key = if let Some(id) = l.get("visual_row_id").filter(|v| !v.is_null()) {
            (a, 0, id.as_i64().ok_or_else(invalid)?, k)
        } else {
            (a, 1, integer(l, "source_index", 0)?, k)
        };
        let next = groups.len();
        let j = *keys.entry(key).or_insert(next);
        if j == next {
            groups.push(Vec::new());
        }
        groups[j].push(i);
    }
    let cs: Vec<_> = candidates.iter().map(|c| c.b.clone()).collect();
    let mut protected = graphic_captions::protect(&mut ls, &cs, em, w)?;
    protected.extend(body_tails(&ls, &candidates, &lanes, em, w)?);
    for group in groups {
        if group
            .iter()
            .any(|&i| protected.contains(&integer(&ls[i], "source_index", 0).unwrap()))
        {
            continue;
        }
        let row_lane = lane(&bbox(&ls[group[0]]["bbox"])?, &lanes) as i64;
        let mut best: Option<(i64, f64, usize)> = None;
        for (i, c) in candidates.iter().enumerate() {
            w.charge(group.len() + 1)?;
            if c.lane >= 0 && c.lane != row_lane {
                continue;
            }
            let mut all = true;
            let mut inside_count = 0;
            let mut distances = Vec::new();
            for &j in &group {
                let b = bbox(&ls[j]["bbox"])?;
                if !(if c.strict {
                    overlap_first(&b, &c.b) >= 0.95
                } else {
                    label(&ls[j], &c.b, em, c.margin)?
                }) {
                    all = false;
                    break;
                }
                inside_count += i64::from(inside(cx(&b), cy(&b), &c.b));
                distances.push(distance(&b, &c.b));
            }
            if all {
                let next = (-inside_count, graphic_grid::mean(&distances, true, w)?, i);
                if best.is_none_or(|p| next < p) {
                    best = Some(next);
                }
            }
        }
        if let Some((_, _, i)) = best {
            for j in group {
                candidates[i]
                    .ids
                    .insert(integer(&ls[j], "source_index", 0)?);
            }
        }
    }
    let by_id: BTreeMap<_, _> = ls
        .iter()
        .enumerate()
        .map(|(i, l)| (integer(l, "source_index", 0).unwrap(), i))
        .collect();
    let mut claimed = BTreeSet::new();
    let mut blocks = Vec::new();
    for c in candidates {
        w.charge(boxes.len() + 1)?;
        if boxes
            .iter()
            .any(|b| overlap_first(&c.b, b) >= 0.95 && overlap_first(b, &c.b) >= 0.95)
        {
            blocks.push(json!({"type":"image","content":"","bbox":c.b.raw,"angle":0}));
            continue;
        }
        let members: Vec<_> = c
            .ids
            .iter()
            .filter_map(|id| by_id.get(id).copied())
            .collect();
        if members.len() < 2 {
            continue;
        }
        let text = super::media::content(&members, &mut ls, sz, w)?;
        if text.is_empty() {
            continue;
        }
        let bs = members
            .iter()
            .map(|&i| bbox(&ls[i]["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let b = union(c.b, &combined(&bs)?);
        let member_values = members.iter().map(|&i| ls[i].clone()).collect::<Vec<_>>();
        let numeric = crate::legacy_model::table_page::numeric_grid(&member_values, &b, w)?;
        blocks.push(json!({"type":"image","bbox":b.raw,"angle":0,"content":text,"_native_numeric_grid":numeric}));
        claimed.extend(c.ids);
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
    for (&i, l) in positions.iter().zip(ls) {
        global[i] = l;
    }
    source["lines"] = Value::Array(global);
    Ok(crate::processing::with_source(
        source,
        json!({"graphic_blocks":blocks,"claimed_line_indices":claimed}),
    ))
}
