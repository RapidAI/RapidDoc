//! 旧全文 Form 角色判定：绘图遮罩、稳定正文/图题、跨页框架继承与真实来源归属。
use super::*;
use crate::legacy_model::{
    closed_rule_pipeline,
    path_graphics::{area, integer, overlap_first},
    strong_graphics,
    table_text_styles::lanes,
};
use rapiddoc_native::{python_decimal, text_boundary::python_space};
use rapiddoc_pdf::form_structure::FormInfo;
use std::collections::{BTreeMap, BTreeSet};
use unicode_normalization::UnicodeNormalization;
fn round(v: f64, n: usize) -> f64 {
    format!("{v:.n$}").parse().unwrap()
}
fn fold(c: char) -> char {
    match c {
        'İ' | 'ı' => 'i',
        'ſ' => 's',
        'K' => 'k',
        _ => c.to_ascii_lowercase(),
    }
}
fn alpha(c: char) -> bool {
    fold(c).is_ascii_alphabetic()
}
fn prefix(c: &[char], s: &str) -> bool {
    s.chars()
        .enumerate()
        .all(|(i, s)| c.get(i).is_some_and(|c| fold(*c) == s))
}
pub(super) fn caption(s: &str) -> bool {
    let s: String = s.nfkc().collect();
    let c: Vec<_> = s.trim_matches(python_space).chars().collect();
    let mut i = 0;
    let mut matched = false;
    for p in [
        "figure",
        "fig",
        "table",
        "tab",
        "algorithm",
        "alg",
        "listing",
        "chart",
        "scheme",
        "diagram",
    ] {
        if prefix(&c, p) {
            i = p.len();
            if c.get(i)
                .is_some_and(|c| *c == '.' || python_space(*c) || c.is_ascii_digit())
            {
                matched = true;
                break;
            }
        }
    }
    if matched {
        if c.get(i) == Some(&'.') {
            i += 1
        }
    } else {
        for p in ["程序清单", "图表", "表格", "算法", "图", "表"] {
            let v: Vec<_> = p.chars().collect();
            if c.starts_with(&v) {
                i = v.len();
                matched = true;
                break;
            }
        }
    }
    if !matched {
        return false;
    }
    while c.get(i).is_some_and(|c| python_space(*c)) {
        i += 1
    }
    let start = i;
    if c.get(i).is_some_and(|c| alpha(*c)) {
        i += 1;
        while c.get(i).is_some_and(|c| python_space(*c)) {
            i += 1
        }
        if c.get(i).is_some_and(|c| ".-".contains(*c)) {
            i += 1
        }
        while c.get(i).is_some_and(|c| python_space(*c)) {
            i += 1
        }
    }
    let digits = i;
    while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
        i += 1
    }
    if i > digits {
        loop {
            let saved = i;
            let mut j = i;
            while c.get(j).is_some_and(|c| python_space(*c)) {
                j += 1
            }
            if !c.get(j).is_some_and(|c| "-./–—".contains(*c)) {
                break;
            }
            j += 1;
            while c.get(j).is_some_and(|c| python_space(*c)) {
                j += 1
            }
            if c.get(j).is_some_and(|c| alpha(*c)) {
                j += 1
            }
            let n = j;
            while c.get(j).is_some_and(|c| python_decimal(*c).is_some()) {
                j += 1
            }
            if j == n {
                i = saved;
                break;
            }
            i = j;
        }
        if c.get(i).is_some_and(|c| alpha(*c)) {
            i += 1
        }
    } else {
        i = start;
        while c.get(i).is_some_and(|c| "ivxlcdm".contains(fold(*c))) {
            i += 1
        }
        if i == start {
            while c
                .get(i)
                .is_some_and(|c| "零〇一二三四五六七八九十百千两".contains(*c))
            {
                i += 1
            }
        }
    }
    i > start && !crate::legacy_model::graphic_captions::tail_reference(&c[i..])
}
fn form_infos(s: &Value) -> Result<Vec<FormInfo>, Error> {
    optional(s, "form_infos", 5000)?
        .iter()
        .map(|v| serde_json::from_value(v.clone()).map_err(|_| invalid()))
        .collect()
}
fn rect(b: [f64; 4]) -> Result<Box4, Error> {
    bbox(&json!(b))
}
fn frame(s: &Value, f: &FormInfo) -> Result<bool, Error> {
    if !f.structure_valid {
        return Ok(false);
    }
    let Some(b) = f.declared_bbox else {
        return Ok(false);
    };
    let sz = size(&s["page_size"])?;
    Ok([[0., 0., sz[0], sz[1]], f.media_bbox].iter().any(|a| {
        a.iter()
            .zip(b)
            .enumerate()
            .all(|(i, (a, b))| (a - b).abs() <= 0.02 * sz[i % 2])
    }))
}
fn owns(l: &Value, ids: &BTreeSet<usize>, w: &mut Work) -> Result<bool, Error> {
    let mut any = false;
    for c in optional(l, "chars", 20000)? {
        w.charge(1)?;
        if c["char"]
            .as_str()
            .is_some_and(|s| !s.trim_matches(python_space).is_empty())
        {
            any = true;
            let owned = if let Some(v) = c.get("source_indices") {
                let a = array(v, 20000)?;
                w.charge(a.len() + 1)?;
                a.iter()
                    .any(|i| i.as_u64().is_some_and(|i| ids.contains(&(i as usize))))
            } else {
                c["char_idx"]
                    .as_u64()
                    .is_some_and(|i| ids.contains(&(i as usize)))
            };
            if !owned {
                return Ok(false);
            }
        }
    }
    Ok(any)
}
fn scale(s: &Value) -> Result<f64, Error> {
    let mut hs = Vec::new();
    for l in optional(s, "lines", 4096)? {
        hs.push(height(l, &bbox(&l["bbox"])?)?);
    }
    Ok(if hs.is_empty() { 10. } else { median(hs) })
}
fn paths(s: &Value, f: &FormInfo, w: &mut Work) -> Result<Vec<Value>, Error> {
    let mut out = Vec::new();
    for p in optional(s, "path_infos", 5000)? {
        w.charge(1)?;
        if p["source_index"]
            .as_u64()
            .is_some_and(|i| f.path_indices.contains(&(i as usize)))
        {
            let mut p = p.clone();
            p["form_depth"] =
                json!((integer(&p, "form_depth", 0)? - f.occurrence.len() as i64).max(0));
            out.push(p);
        }
    }
    Ok(out)
}
fn masks(
    s: &mut Value,
    f: &FormInfo,
    infos: &[FormInfo],
    w: &mut Work,
) -> Result<Vec<Box4>, Error> {
    let sz = size(&s["page_size"])?;
    let mut out = f
        .image_bboxes
        .iter()
        .map(|b| rect(*b))
        .collect::<Result<Vec<_>, _>>()?;
    for child in infos {
        w.charge(1)?;
        if child.parent_id == Some(f.instance_id)
            && !frame(s, child)?
            && area(&rect(child.bbox)?) >= 0.01 * sz[0] * sz[1]
        {
            out.push(rect(child.bbox)?);
        }
    }
    let relative = paths(s, f, w)?;
    let em = scale(s)?;
    let mut graphic = relative.clone();
    let mut connectors = Vec::new();
    for (p, g) in relative.iter().zip(&mut graphic) {
        let b = bbox(&p["bbox"])?;
        let span = (b.n[2] - b.n[0]).min(b.n[3] - b.n[1]);
        if integer(p, "segment_count", 0)? == 2 && p["stroke_visible"] == true {
            if span >= 1.5 * em {
                g["segment_count"] = json!(6);
            }
            if span >= 0.5 * em {
                connectors.push(b);
            }
        }
    }
    for p in &relative {
        let n = integer(p, "segment_count", 0)?;
        if ![4, 5].contains(&n) {
            continue;
        }
        let b = bbox(&p["bbox"])?;
        w.charge(connectors.len() + relative.len() + 1)?;
        if !connectors.iter().any(|c| overlap_first(c, &b) >= 0.9) {
            continue;
        }
        let mut count = 0;
        for other in &relative {
            if other["source_index"] != p["source_index"]
                && overlap_first(&bbox(&other["bbox"])?, &b) >= 0.9
            {
                count += 1;
            }
        }
        if count >= 4 {
            out.push(b);
        }
    }
    let mut local = s.clone();
    local["path_infos"] = json!(graphic);
    local["page_size"] = json!([2. * sz[0], 2. * sz[1]]);
    local["drawing_component_cache"] = json!([]);
    let graphics = strong_graphics::detect_source(&local, w)?;
    s["lines"] = graphics["source"]["lines"].clone();
    for b in array(&graphics["graphic_bboxes"], 5000)? {
        out.push(bbox(b)?);
    }
    if out.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    Ok(out)
}
fn masked(b: &Box4, masks: &[Box4], w: &mut Work) -> Result<bool, Error> {
    w.charge(masks.len() + 1)?;
    let rs: Vec<_> = masks
        .iter()
        .map(|m| {
            [
                b.n[0].max(m.n[0]),
                b.n[1].max(m.n[1]),
                b.n[2].min(m.n[2]),
                b.n[3].min(m.n[3]),
            ]
        })
        .filter(|r| r[2] > r[0] && r[3] > r[1])
        .collect();
    let mut edges: Vec<_> = rs.iter().flat_map(|r| [r[0], r[2]]).collect();
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
    edges.dedup();
    let mut area = 0.;
    for pair in edges.windows(2) {
        w.charge(rs.len() + 1)?;
        let mut spans: Vec<_> = rs
            .iter()
            .filter(|r| r[0] <= pair[0] && r[2] >= pair[1])
            .map(|r| [r[1], r[3]])
            .collect();
        spans.sort_by(|a, b| {
            a[0].partial_cmp(&b[0])
                .unwrap()
                .then(a[1].partial_cmp(&b[1]).unwrap())
        });
        let (mut top, mut bottom, mut covered) = (0., 0., 0.);
        for [start, end] in spans {
            if start > bottom {
                covered += bottom - top;
                top = start;
                bottom = end;
            } else {
                bottom = bottom.max(end);
            }
        }
        area += (pair[1] - pair[0]) * (covered + bottom - top);
    }
    Ok(area >= 0.8 * crate::legacy_model::path_graphics::area(b).max(0.1))
}
pub(super) struct Layout {
    ls: Vec<Value>,
    bs: Vec<Box4>,
    pub(super) lanes: Vec<Value>,
}
pub(super) fn layout(ls: Vec<Value>, sz: [f64; 2], a: i64, w: &mut Work) -> Result<Layout, Error> {
    let mut lines = Vec::new();
    let mut bs = Vec::new();
    let mut hs = Vec::new();
    for l in ls {
        if integer(&l, "angle", 0)? != a {
            continue;
        }
        let raw = l
            .get("ink_bbox")
            .filter(|v| !v.is_null())
            .unwrap_or(&l["bbox"]);
        let b = super::index::local(&bbox(raw)?, sz, a);
        hs.push(height(&l, &b)?);
        lines.push(l);
        bs.push(b);
    }
    if lines.is_empty() {
        return Ok(Layout {
            ls: lines,
            bs,
            lanes: Vec::new(),
        });
    }
    let h = median(hs);
    let width = if [90, 270].contains(&a) { sz[1] } else { sz[0] };
    let geometry: Vec<_> = lines
        .iter()
        .zip(&bs)
        .map(|(l, b)| json!([l, b.raw]))
        .collect();
    let raw = lanes::infer_layout(
        &json!({"line_geometry":geometry,"local_page_width":width,"median_height":h,"recalculate_intervals":false}),
        w,
    )?;
    let mut stable: Vec<Value> = array(&raw, 4096)?
        .iter()
        .filter(|l| l["is_span"] == false && l["line_indices"].as_array().unwrap().len() >= 3)
        .cloned()
        .collect();
    stable.sort_by(|a, b| a["left"].as_f64().partial_cmp(&b["left"].as_f64()).unwrap());
    let mut distinct: Vec<Value> = Vec::new();
    for l in stable {
        if distinct
            .last()
            .is_some_and(|p| l["left"].as_f64().unwrap() < p["right"].as_f64().unwrap())
        {
            if l["line_indices"].as_array().unwrap().len()
                > distinct.last().unwrap()["line_indices"]
                    .as_array()
                    .unwrap()
                    .len()
            {
                *distinct.last_mut().unwrap() = l;
            }
        } else {
            distinct.push(l)
        }
    }
    Ok(Layout {
        ls: lines,
        bs,
        lanes: distinct,
    })
}
fn body(s: &Value, f: &FormInfo, ms: &[Box4], minimum: usize, w: &mut Work) -> Result<bool, Error> {
    let mut source = optional(s, "lines", 4096)?.to_vec();
    if source.is_empty() && !optional(s, "chars", 20000)?.is_empty() {
        let n = optional(s, "chars", 20000)?.len();
        w.charge(n * (n + 1) / 2 + n * 64)?;
        source=crate::legacy_model::native_lines::build(&json!({"chars":s["chars"],"page_size":s["page_size"],"supported_angles":[0.,90.,180.,270.]}))?.as_array().unwrap().clone();
    }
    let mut lines = Vec::new();
    let mut angles = BTreeSet::new();
    for l in source {
        w.charge(l["text"].as_str().ok_or_else(invalid)?.len() + 1)?;
        if l["semantic_type"].is_null()
            && owns(&l, &f.text_indices, w)?
            && !caption(l["text"].as_str().unwrap())
            && !masked(
                &bbox(
                    l.get("ink_bbox")
                        .filter(|v| !v.is_null())
                        .unwrap_or(&l["bbox"]),
                )?,
                ms,
                w,
            )?
        {
            angles.insert(integer(&l, "angle", 0)?);
            lines.push(l);
        }
    }
    let sz = size(&s["page_size"])?;
    for a in angles {
        let local = layout(lines.clone(), sz, a, w)?;
        for lane in local.lanes {
            let ids = array(&lane["line_indices"], 4096)?;
            if ids.len() < minimum {
                continue;
            }
            let width = num(&lane["right"])? - num(&lane["left"])?;
            let hs = ids
                .iter()
                .map(|i| {
                    let i = i.as_u64().unwrap() as usize;
                    height(&local.ls[i], &local.bs[i])
                })
                .collect::<Result<Vec<_>, _>>()?;
            let em = median(hs);
            let mut long: Vec<_> = ids
                .iter()
                .map(|i| &local.bs[i.as_u64().unwrap() as usize])
                .filter(|b| {
                    b.n[2] - b.n[0] >= (10. * em).max(0.6 * width)
                        && ((b.n[0] - num(&lane["left"]).unwrap()).abs() <= em
                            || (b.n[2] - num(&lane["right"]).unwrap()).abs() <= em)
                })
                .collect();
            long.sort_by(|a, b| a.n[1].partial_cmp(&b.n[1]).unwrap());
            let mut count = 0;
            let mut previous: Option<&Box4> = None;
            for b in long {
                count = if previous
                    .is_some_and(|p| (-0.2 * em..=2. * em).contains(&(b.n[1] - p.n[3])))
                {
                    count + 1
                } else {
                    1
                };
                if count >= minimum {
                    return Ok(true);
                }
                previous = Some(b);
            }
        }
    }
    Ok(false)
}
fn figures(s: &Value, f: &FormInfo, ms: &[Box4], w: &mut Work) -> Result<bool, Error> {
    let mut captions = Vec::new();
    for l in optional(s, "lines", 4096)? {
        w.charge(l["text"].as_str().ok_or_else(invalid)?.len() + 1)?;
        if owns(l, &f.text_indices, w)?
            && caption(l["text"].as_str().unwrap())
            && !masked(
                &bbox(
                    l.get("ink_bbox")
                        .filter(|v| !v.is_null())
                        .unwrap_or(&l["bbox"]),
                )?,
                ms,
                w,
            )?
        {
            captions.push((integer(l, "source_index", 0)?, bbox(&l["bbox"])?));
        }
    }
    let em = scale(s)?;
    let mut paired = BTreeSet::new();
    for m in ms {
        w.charge(captions.len() + 1)?;
        let distance = |b: &Box4| (m.n[1] - b.n[3]).abs().min((b.n[1] - m.n[3]).abs());
        if let Some((id, _)) = captions
            .iter()
            .filter(|(_, b)| {
                m.n[2].min(b.n[2]) > m.n[0].max(b.n[0])
                    && distance(b) <= 4. * em
                    && overlap_first(b, m) < 0.5
            })
            .min_by(|(_, a), (_, b)| distance(a).partial_cmp(&distance(b)).unwrap())
        {
            paired.insert(*id);
        }
    }
    Ok(paired.len() >= 2)
}
fn underlay(s: &Value, f: &FormInfo, w: &mut Work) -> Result<bool, Error> {
    if f.paint_order != 0 || !f.text_indices.is_empty() || !f.image_bboxes.is_empty() {
        return Ok(false);
    }
    let sz = size(&s["page_size"])?;
    let ps = paths(s, f, w)?;
    if ps.is_empty() {
        return Ok(false);
    }
    for p in ps {
        if ![4, 5].contains(&integer(&p, "segment_count", 0)?)
            || area(&bbox(&p["bbox"])?) < 0.7 * sz[0] * sz[1]
        {
            return Ok(false);
        }
    }
    for l in optional(s, "lines", 4096)? {
        if l["semantic_type"].is_null() && !owns(l, &f.text_indices, w)? {
            return Ok(true);
        }
    }
    Ok(false)
}
fn tables(s: &Value, f: &FormInfo, ms: &[Box4], w: &mut Work) -> Result<bool, Error> {
    let sz = size(&s["page_size"])?;
    if ms.iter().any(|m| area(m) >= 0.05 * sz[0] * sz[1]) {
        return Ok(false);
    }
    let mut lines = Vec::new();
    for l in optional(s, "lines", 4096)? {
        if l["semantic_type"].is_null() && owns(l, &f.text_indices, w)? {
            lines.push(l.clone())
        }
    }
    if lines.len() < 6 {
        return Ok(false);
    }
    let n = lines.len();
    let mut local = s.clone();
    local["lines"] = json!(lines);
    local["form_bboxes"] = json!([]);
    local["path_infos"] = json!(paths(s, f, w)?);
    local["drawing_component_cache"] = json!([]);
    let found = closed_rule_pipeline::detect_complete_with_work(
        &json!({"source":local,"excluded_bboxes":ms.iter().map(|b|b.raw.clone()).collect::<Vec<_>>()}),
        w,
    )?;
    let mut ids = BTreeSet::new();
    for c in array(&found["candidates"], 5000)? {
        for id in array(&c["line_indices"], 20000)? {
            ids.insert(id.as_i64().ok_or_else(invalid)?);
        }
    }
    Ok(ids.len() as f64 >= 6f64.max(0.5 * n as f64))
}
fn framework(
    s: &Value,
    f: &FormInfo,
    ms: &[Box4],
    w: &mut Work,
) -> Result<Vec<(i64, f64, f64)>, Error> {
    let sz = size(&s["page_size"])?;
    let mut lines = Vec::new();
    let mut angles = BTreeSet::new();
    for l in optional(s, "lines", 4096)? {
        if owns(l, &f.text_indices, w)?
            && !masked(
                &bbox(
                    l.get("ink_bbox")
                        .filter(|v| !v.is_null())
                        .unwrap_or(&l["bbox"]),
                )?,
                ms,
                w,
            )?
        {
            angles.insert(integer(l, "angle", 0)?);
            lines.push(l.clone());
        }
    }
    let mut out = Vec::new();
    for a in angles {
        let width = if [90, 270].contains(&a) { sz[1] } else { sz[0] };
        let local = layout(lines.clone(), sz, a, w)?;
        for lane in local.lanes {
            out.push((a, num(&lane["left"])? / width, num(&lane["right"])? / width));
        }
    }
    let mut rules: Vec<((f64, f64), usize)> = Vec::new();
    for p in optional(s, "path_infos", 5000)? {
        w.charge(1)?;
        if p["source_index"]
            .as_u64()
            .is_some_and(|id| f.path_indices.contains(&(id as usize)))
            && p["stroke_visible"] == true
        {
            let b = bbox(&p["bbox"])?;
            if b.n[3] - b.n[1] <= 2. && b.n[2] - b.n[0] >= 0.2 * sz[0] {
                let edge = (round(b.n[0] / sz[0], 3), round(b.n[2] / sz[0], 3));
                if let Some((_, n)) = rules.iter_mut().find(|(e, _)| *e == edge) {
                    *n += 1
                } else {
                    rules.push((edge, 1));
                }
            }
        }
    }
    for ((l, r), n) in rules {
        if n >= 3 {
            out.push((0, l, r));
        }
    }
    Ok(out)
}
fn family(s: &Value, f: &FormInfo) -> Result<[f64; 4], Error> {
    let sz = size(&s["page_size"])?;
    Ok(std::array::from_fn(|i| {
        round(f.declared_bbox.unwrap()[i] / sz[i % 2], 2)
    }))
}
pub(super) fn normalize(raw: &[Value], w: &mut Work) -> Result<Vec<Value>, Error> {
    normalize_owned(raw.to_vec(), w)
}
pub(super) fn normalize_owned(mut sources: Vec<Value>, w: &mut Work) -> Result<Vec<Value>, Error> {
    let mut decisions = Vec::new();
    let mut families: Vec<([f64; 4], Vec<(i64, f64, f64)>)> = Vec::new();
    for s in &mut sources {
        let fs = form_infos(s)?;
        let (mut expanded, mut deco) = (BTreeSet::new(), BTreeSet::new());
        for f in &fs {
            w.charge(1)?;
            if !frame(s, f)? {
                continue;
            }
            let ms = masks(s, f, &fs, w)?;
            if underlay(s, f, w)? {
                deco.insert(f.instance_id);
            } else if body(s, f, &ms, 6, w)? || figures(s, f, &ms, w)? {
                expanded.insert(f.instance_id);
                let key = family(s, f)?;
                let layout = framework(s, f, &ms, w)?;
                if let Some((_, existing)) = families.iter_mut().find(|(k, _)| *k == key) {
                    existing.extend(layout);
                } else {
                    families.push((key, layout));
                }
            }
        }
        decisions.push((fs, expanded, deco));
    }
    for (s, (fs, mut expanded, mut deco)) in sources.iter_mut().zip(decisions) {
        if fs.is_empty() {
            continue;
        }
        for f in &fs {
            if expanded.contains(&f.instance_id) || deco.contains(&f.instance_id) || !frame(s, f)? {
                continue;
            }
            let key = family(s, f)?;
            if let Some((_, other)) = families.iter().find(|(k, _)| *k == key) {
                let ms = masks(s, f, &fs, w)?;
                let local = framework(s, f, &ms, w)?;
                w.charge(local.len() * other.len() + 1)?;
                let compatible = local.iter().any(|(a, l, r)| {
                    other
                        .iter()
                        .any(|(b, x, y)| a == b && ((l - x).abs() <= 0.03 || (r - y).abs() <= 0.03))
                });
                if compatible && (body(s, f, &ms, 3, w)? || tables(s, f, &ms, w)?) {
                    expanded.insert(f.instance_id);
                }
            }
        }
        for f in &fs {
            if f.parent_id
                .is_some_and(|p| !expanded.contains(&p) && !deco.contains(&p))
            {
                expanded.remove(&f.instance_id);
                deco.remove(&f.instance_id);
            }
        }
        let removed: BTreeSet<_> = expanded.union(&deco).copied().collect();
        let active: Vec<_> = fs
            .iter()
            .filter(|f| {
                !removed.contains(&f.instance_id)
                    && (f.parent_id.is_none() || removed.contains(&f.parent_id.unwrap()))
            })
            .filter(|f| !removed.is_empty() || f.parent_id.is_none())
            .collect();
        s["form_bboxes"] = json!(active
            .iter()
            .filter(|f| visible(f.bbox))
            .map(|f| f.bbox)
            .collect::<Vec<_>>());
        let mut retained = Vec::new();
        for f in &active {
            if frame(s, f)? && !retained.contains(&f.bbox) {
                retained.push(f.bbox);
            }
        }
        retained.sort_by(|a, b| {
            a.iter()
                .zip(b)
                .map(|(a, b)| a.partial_cmp(b).unwrap())
                .find(|o| !o.is_eq())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        s["retained_page_forms"] = json!(retained);
        let mut maps: Vec<([f64; 4], BTreeSet<usize>, BTreeSet<usize>)> = Vec::new();
        for f in &active {
            if !f.structure_valid
                || active
                    .iter()
                    .any(|other| other.bbox == f.bbox && !other.structure_valid)
            {
                continue;
            }
            if let Some((_, text, paths)) = maps.iter_mut().find(|(b, _, _)| *b == f.bbox) {
                text.extend(&f.text_indices);
                paths.extend(&f.path_indices);
            } else {
                maps.push((f.bbox, f.text_indices.clone(), f.path_indices.clone()));
            }
        }
        maps.sort_by(|(a, _, _), (b, _, _)| {
            a.iter()
                .zip(b)
                .map(|(a, b)| a.partial_cmp(b).unwrap())
                .find(|o| !o.is_eq())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        s["form_member_sources"] = json!(maps
            .iter()
            .map(|(b, ids, _)| json!({"bbox":b,"source_indices":ids}))
            .collect::<Vec<_>>());
        s["form_path_sources"] = json!(maps
            .iter()
            .map(|(b, _, ids)| json!({"bbox":b,"source_indices":ids}))
            .collect::<Vec<_>>());
        let by_id: BTreeMap<_, _> = fs.iter().map(|f| (f.instance_id, f)).collect();
        let mut ps = optional(s, "path_infos", 5000)?.to_vec();
        for p in &mut ps {
            let id = integer(p, "source_index", 0)?;
            let delta = expanded
                .iter()
                .map(|i| by_id.get(i).ok_or_else(invalid))
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .filter(|f| f.path_indices.contains(&(id as usize)))
                .count();
            p["form_depth"] = json!((integer(p, "form_depth", 0)? - delta as i64).max(0));
        }
        s["path_infos"] = json!(ps);
        if s.get("drawing_component_cache").is_some() {
            s["drawing_component_cache"] = json!([]);
        }
    }
    Ok(sources)
}
fn visible(b: [f64; 4]) -> bool {
    b[2] > b[0] && b[3] > b[1]
}
