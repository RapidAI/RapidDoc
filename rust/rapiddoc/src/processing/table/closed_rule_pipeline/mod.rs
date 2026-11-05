//! 闭合物理网格完整候选链：自动局部行/轨道/表题/续表、正式组装及连通扩框。
mod page_policies;
mod rule_intervals;
mod supplemental;
use super::{
    graphic_grid,
    path_graphics::{array, bbox, integer, num, overlap_first, overlap_smaller, union, Box4},
    rule_annotation_preparation as rowprep,
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{
    python_decimal,
    text_boundary::{python_alnum, python_lower, python_space},
};
use serde_json::{json, Value};
use std::{borrow::Cow, collections::BTreeSet};
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k)
        .filter(|v| !v.is_null())
        .map_or(Ok(&[]), |v| array(v, max))
}
fn local(b: &Box4, p: &[Value], angle: i64) -> Result<Box4, Error> {
    let swapped = [p[1].clone(), p[0].clone()];
    let v = match angle {
        90 => super::rule_candidate_expansion::rotate(&b.raw, &swapped, 270),
        270 => super::rule_candidate_expansion::rotate(&b.raw, &swapped, 90),
        _ => super::rule_candidate_expansion::rotate(&b.raw, p, angle),
    };
    bbox(&v)
}
fn center(b: &Box4, i: usize) -> f64 {
    (b.n[i] + b.n[i + 2]) / 2.
}
fn inside(x: f64, y: f64, b: &Box4) -> bool {
    b.n[0] <= x && x <= b.n[2] && b.n[1] <= y && y <= b.n[3]
}
struct Page<'a> {
    size: Vec<Value>,
    angle: i64,
    em: f64,
    lines: Vec<Cow<'a, Value>>,
    rows: Vec<rowprep::Row>,
    raw_rows: Vec<Value>,
    p: graphic_grid::Prepared,
}
impl Page<'_> {
    fn line_values(&self) -> impl Iterator<Item = &Value> {
        self.lines.iter().map(Cow::as_ref)
    }
    fn into_owned(self) -> Page<'static> {
        Page {
            size: self.size,
            angle: self.angle,
            em: self.em,
            lines: self
                .lines
                .into_iter()
                .map(|l| Cow::Owned(l.into_owned()))
                .collect(),
            rows: self.rows,
            raw_rows: self.raw_rows,
            p: self.p,
        }
    }
}
fn prepare<'a>(v: &'a Value, work: &mut Work) -> Result<Page<'a>, Error> {
    prepare_source(&v["source"], v, work)
}
fn prepare_source<'a>(source: &'a Value, v: &Value, work: &mut Work) -> Result<Page<'a>, Error> {
    source.as_object().ok_or_else(invalid)?;
    let size = array(&source["page_size"], 2)?;
    if size.len() != 2 {
        return Err(invalid());
    }
    for n in size {
        if num(n)? <= 0. {
            return Err(invalid());
        }
    }
    let angle = integer(v, "angle", 0)?;
    let mut lines = Vec::new();
    let mut fragments = Vec::new();
    let mut synthetic = Vec::new();
    let mut heights = Vec::new();
    for l in optional(source, "lines", 4096)? {
        work.charge(1)?;
        if l.get("angle").map_or(Ok(0.), num)? != angle as f64 {
            continue;
        }
        let text = l["text"].as_str().ok_or_else(invalid)?;
        if text.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        work.charge(text.len() + 1)?;
        let b = local(&bbox(&l["bbox"])?, size, angle)?;
        let h = b.n[3] - b.n[1];
        if h > 0. {
            heights.push(h);
        }
        let visual = l.get("visual_row_id").cloned().unwrap_or(Value::Null);
        let id = integer(l, "source_index", 0)?;
        fragments.push(value_object!({"text":text,"bbox":l["bbox"],"local_bbox":b.raw,"line_index":id,"visual_row_id":visual}));
        synthetic.push(value_object!({"bbox":b.raw,"angle":0,"visual_row_id":visual}));
        lines.push(Cow::Borrowed(l));
    }
    work.charge(heights.len() * 16 + 1)?;
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = heights.len();
    let fallback = if n == 0 {
        1.
    } else if n % 2 == 1 {
        heights[n / 2]
    } else {
        (heights[n / 2 - 1] + heights[n / 2]) / 2.
    };
    let em = v
        .get("median_height")
        .filter(|v| !v.is_null())
        .map_or(Ok(fallback.max(0.1)), num)?;
    if em <= 0. {
        return Err(invalid());
    }
    let mut axis = Vec::new();
    for r in optional(source, "drawing_lines", 5000)? {
        let b = local(&bbox(&r["bbox"])?, size, angle)?;
        axis.push(value_object!({"bbox":b.raw}));
    }
    let p = graphic_grid::prepare(
        &json!({"page_size":size,"lines":synthetic,"drawing_lines":axis,"form_bboxes":[]}),
        em,
        work,
    )?;
    let mut raw_rows = Vec::new();
    for r in &p.rows {
        work.charge(r.ids.len() + 1)?;
        let mut b = p.fragments[r.ids[0]].clone();
        let mut visual = BTreeSet::new();
        let fs: Vec<_> = r.ids.iter().map(|i| fragments[*i].clone()).collect();
        for i in &r.ids {
            b = union(b, &p.fragments[*i]);
            if !fragments[*i]["visual_row_id"].is_null() {
                visual.insert(
                    fragments[*i]["visual_row_id"]
                        .as_i64()
                        .ok_or_else(invalid)?,
                );
            }
        }
        raw_rows.push(json!({"fragments":fs,"bbox":b.raw,"center_y":r.cy,"visual_row_id":if visual.len()==1{json!(visual.iter().next().unwrap())}else{Value::Null}}));
    }
    let mut total = 0;
    let rows = raw_rows
        .iter()
        .map(|r| rowprep::row(r, &mut total))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Page {
        size: size.to_vec(),
        angle,
        em,
        lines,
        rows,
        raw_rows,
        p,
    })
}
fn component_boxes(p: &Page, work: &mut Work) -> Result<Vec<Box4>, Error> {
    let mut out = Vec::new();
    for ids in &p.p.components {
        work.charge(ids.len() + 1)?;
        let mut b = p.p.rules[ids[0]].b.clone();
        for i in &ids[1..] {
            b = union(b, &p.p.rules[*i].b);
        }
        out.push(b);
    }
    Ok(out)
}
fn local_boxes(v: &Value, p: &Page, work: &mut Work) -> Result<Vec<Box4>, Error> {
    let a = array(v, 5000)?;
    work.charge(a.len() + 1)?;
    a.iter()
        .map(|v| local(&bbox(v)?, &p.size, p.angle))
        .collect()
}
fn exclusions(v: &Value, p: &Page, work: &mut Work) -> Result<Vec<Box4>, Error> {
    local_boxes(v.get("excluded_bboxes").unwrap_or(&json!([])), p, work)
}
fn word(c: char) -> bool {
    c == '_' || python_alnum(c)
}
fn roman(c: char) -> bool {
    matches!(
        c,
        'i' | 'I'
            | 'İ'
            | 'ı'
            | 'v'
            | 'V'
            | 'x'
            | 'X'
            | 'l'
            | 'L'
            | 'c'
            | 'C'
            | 'd'
            | 'D'
            | 'm'
            | 'M'
    )
}
fn chinese(c: char) -> bool {
    "一二三四五六七八九十".contains(c)
}
fn prefix(c: &[char], text: &str) -> bool {
    text.chars()
        .enumerate()
        .all(|(i, x)| c.get(i).is_some_and(|c| c.to_ascii_lowercase() == x))
}
fn caption_text(s: &str) -> bool {
    let c: Vec<_> = s.chars().collect();
    let mut i = if prefix(&c, "table") {
        5
    } else if prefix(&c, "tab") {
        3
    } else if c.first() == Some(&'表') {
        if c.get(1) == Some(&'格') {
            2
        } else {
            1
        }
    } else {
        return false;
    };
    while c
        .get(i)
        .is_some_and(|c| python_space(*c) || " :.–—-".contains(*c))
    {
        i += 1;
    }
    let start = i;
    let mode = if c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
        0
    } else if c.get(i).is_some_and(|c| roman(*c)) {
        1
    } else {
        2
    };
    while c.get(i).is_some_and(|c| match mode {
        0 => python_decimal(*c).is_some(),
        1 => roman(*c),
        _ => chinese(*c),
    }) {
        i += 1;
    }
    if i == start || c.get(i).is_some_and(|c| word(*c)) {
        return false;
    }
    if c[i..].contains(&'\n') {
        return false;
    }
    let suffix: String = c[i..].iter().collect();
    !suffix
        .trim_matches(|c| " .:–—-".contains(c))
        .chars()
        .next()
        .is_some_and(python_lower)
}
fn split_number(s: &str) -> bool {
    let c: Vec<_> = s.trim_matches(python_space).chars().collect();
    let mut i = 0;
    let mode = if c.first().is_some_and(|c| python_decimal(*c).is_some()) {
        0
    } else if c.first().is_some_and(|c| roman(*c)) {
        1
    } else {
        2
    };
    while c.get(i).is_some_and(|c| match mode {
        0 => python_decimal(*c).is_some(),
        1 => roman(*c),
        _ => chinese(*c),
    }) {
        i += 1;
    }
    if i == 0 {
        return false;
    }
    if c.get(i).is_some_and(|c| ".:：".contains(*c)) {
        i += 1;
    }
    i == c.len()
}
fn captions(p: &Page, work: &mut Work) -> Result<Vec<(i64, Box4)>, Error> {
    let mut out = Vec::new();
    for l in p.line_values() {
        let text = l["text"].as_str().unwrap().trim_matches(python_space);
        work.charge(text.len() + 1)?;
        let split = matches!(
            text.to_ascii_lowercase().trim_end_matches('.'),
            "table" | "tab" | "表" | "表格"
        );
        let full = caption_text(text);
        if !full && !split {
            continue;
        }
        let b = local(&bbox(&l["bbox"])?, &p.size, p.angle)?;
        let id = integer(l, "source_index", 0)?;
        if split {
            let mut found = false;
            for peer in &p.lines {
                let text = peer["text"].as_str().unwrap();
                work.charge(text.len() + 1)?;
                if integer(peer, "source_index", 0)? == id || !split_number(text) {
                    continue;
                }
                let pb = local(&bbox(&peer["bbox"])?, &p.size, p.angle)?;
                let gap = pb.n[0] - b.n[2];
                if super::graphic_captions::axis(&b, &pb, 1) >= 0.5 && 0. <= gap && gap <= 4. * p.em
                {
                    found = true;
                    break;
                }
            }
            if !found {
                continue;
            }
        }
        out.push((id, b));
    }
    Ok(out)
}
fn find_caption(cs: &[(i64, Box4)], grid: &Box4, em: f64, work: &mut Work) -> Result<Value, Error> {
    work.charge(cs.len() + 1)?;
    let mut winner = None;
    for (id, b) in cs {
        let gap = grid.n[1] - b.n[3];
        if super::graphic_captions::axis(b, grid, 0) < 0.05 || gap < -em || gap > 12. * em {
            continue;
        }
        if winner.is_none_or(|(_, best)| gap.abs() < best) {
            winner = Some((*id, gap.abs()));
        }
    }
    Ok(winner.map_or(Value::Null, |(id, _)| json!(id)))
}
fn corridor(p: &Page, b: &Box4, excluded: &[Box4], work: &mut Work) -> Result<Vec<Value>, Error> {
    let mut out = Vec::new();
    for row in &p.rows {
        work.charge(1)?;
        if row.b.n[1] > b.n[3] || row.b.n[3] < b.n[1] {
            continue;
        }
        let Some(mut r) = rowprep::clipped(row, b, 0., work)? else {
            continue;
        };
        if r.center < b.n[1] || r.center > b.n[3] {
            continue;
        }
        work.charge(r.fragments.len() * (excluded.len() + 1))?;
        r.fragments.retain(|f| {
            !excluded
                .iter()
                .any(|e| inside(center(&f.local, 0), center(&f.local, 1), e))
        });
        if r.fragments.is_empty() {
            continue;
        }
        r.b = r.fragments[0].local.clone();
        for f in &r.fragments[1..] {
            r.b = union(r.b, &f.local);
        }
        r.center = super::table_native_candidate::sum(
            &r.fragments
                .iter()
                .map(|f| center(&f.local, 1))
                .collect::<Vec<_>>(),
        ) / r.fragments.len() as f64;
        out.push(rowprep::output(&r));
    }
    Ok(out)
}
fn previous_matches(
    source: &Value,
    p: &Page,
    b: &Box4,
    positions: &[f64],
    obstacles: &[Box4],
    work: &mut Work,
) -> Result<bool, Error> {
    let Some(prev) = source.get("previous_table_grid").filter(|v| !v.is_null()) else {
        return Ok(false);
    };
    let Some(page) = source.get("page_index").and_then(Value::as_i64) else {
        return Ok(false);
    };
    if integer(prev, "page_index", -2)?.checked_add(1) != Some(page)
        || integer(prev, "angle", 0)? != p.angle
    {
        return Ok(false);
    }
    let prior = array(&prev["positions"], 5000)?;
    let tol = num(&prev["tolerance"])?;
    let width = num(&p.size[if matches!(p.angle, 0 | 180) { 0 } else { 1 }])?;
    let tolerance = tol.max(2f64.max(0.25 * p.em) / width);
    work.charge(prior.len() + p.rows.len() + obstacles.len() + 1)?;
    if prior.len() != positions.len() {
        return Ok(false);
    }
    for (x, old) in positions.iter().zip(prior) {
        if (x / width - num(old)?).abs() > tolerance {
            return Ok(false);
        }
    }
    let edge = 2f64.max(0.25 * p.em);
    Ok(!p.rows.iter().any(|r| r.b.n[3] < b.n[1] - edge)
        && !obstacles.iter().any(|r| r.n[3] < b.n[1] - edge))
}
fn expand_candidates(
    candidates: &[Value],
    p: &Page,
    grids: &[Box4],
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    expand_candidates_cached(candidates, p, grids, None, work)
}
fn expand_candidates_cached(
    candidates: &[Value],
    p: &Page,
    grids: &[Box4],
    mut members: Option<&mut std::collections::HashMap<[u64; 4], BTreeSet<i64>>>,
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    let tolerance = 2f64.max(p.em);
    let mut output = Vec::new();
    let mut total = 0;
    for original in candidates {
        let raw_ids = optional(original, "line_indices", 20000)?;
        work.charge(raw_ids.len() * 16 + 1)?;
        total += raw_ids.len();
        for a in optional(original, "annotations", 2000)? {
            let ids = optional(a, "line_indices", 20000)?;
            let boxes = optional(a, "line_bboxes", 20000)?;
            total += ids.len() + boxes.len();
            work.charge((ids.len() + boxes.len()) * 16 + 1)?;
        }
        if total > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        let mut c = original.clone();
        if c["core_bbox"].is_null() {
            output.push(c);
            continue;
        }
        let core = local(&bbox(&c["core_bbox"])?, &p.size, p.angle)?;
        work.charge(grids.len() + 1)?;
        let mut best: Option<usize> = None;
        for (i, b) in grids.iter().enumerate() {
            if super::graphic_captions::axis(&core, b, 0) < 0.9
                || core.n[3] < b.n[1] - tolerance
                || core.n[1] > b.n[3] + tolerance
            {
                continue;
            }
            let score = core.n[3].min(b.n[3]) - core.n[1].max(b.n[1]);
            if best.is_none_or(|j| {
                let old = &grids[j];
                let prior = core.n[3].min(old.n[3]) - core.n[1].max(old.n[1]);
                score > prior
                    || (score == prior
                        && super::graphic_arbitration::area_key(b)
                            > super::graphic_arbitration::area_key(old))
            }) {
                best = Some(i);
            }
        }
        if let Some(i) = best {
            let expanded = union(core, &grids[i]);
            let outer = union(bbox(&c["local_bbox"])?, &grids[i]);
            c["local_bbox"] = json!(outer.raw);
            c["core_bbox"] =
                super::rule_candidate_expansion::rotate(&expanded.raw, &p.size, p.angle);
            c["bbox"] = super::rule_candidate_expansion::rotate(&outer.raw, &p.size, p.angle);
            let raw = array(&c["line_indices"], 20000)?;
            let mut ids = raw
                .iter()
                .map(|v| v.as_i64().ok_or_else(invalid))
                .collect::<Result<BTreeSet<_>, _>>()?;
            if let Some(cache) = members.as_mut() {
                let key = expanded.n.map(f64::to_bits);
                if !cache.contains_key(&key) {
                    let members = grid_members(p, &expanded, work)?;
                    cache.insert(key, members);
                }
                ids.extend(cache[&key].iter().copied());
            } else {
                ids.extend(grid_members(p, &expanded, work)?);
            }
            for a in optional(&c, "annotations", 2000)? {
                let raw = array(&a["line_indices"], 20000)?;
                work.charge(raw.len() * 16 + 1)?;
                for id in raw {
                    ids.remove(&id.as_i64().ok_or_else(invalid)?);
                }
            }
            total += ids.len().saturating_sub(raw.len());
            if total > rapiddoc_core::resources::limit(20000) {
                return Err(limit());
            }
            c["line_indices"] = json!(ids);
        }
        output.push(c);
    }
    Ok(output)
}
fn grid_members(p: &Page, expanded: &Box4, work: &mut Work) -> Result<BTreeSet<i64>, Error> {
    let mut members = BTreeSet::new();
    for row in &p.rows {
        work.charge(row.fragments.len() * 16 + 1)?;
        if row.center < expanded.n[1] || row.center > expanded.n[3] {
            continue;
        }
        for f in &row.fragments {
            if inside(center(&f.local, 0), center(&f.local, 1), &expanded) {
                members.insert(f.id);
            }
        }
    }
    Ok(members)
}
pub fn inspect(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut work = Work(0);
    let p = prepare(v, &mut work)?;
    let boxes = component_boxes(&p, &mut work)?;
    let mut tracks = Vec::new();
    let mut count = 0;
    for ids in &p.p.components {
        let t = graphic_grid::tracks(&p.p, ids, &mut work)?;
        count += t.len();
        if count > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        tracks.push(t);
    }
    let cs = captions(&p, &mut work)?;
    let out = json!({"median_height":p.em,"rows":p.raw_rows,"components":p.p.components.iter().zip(boxes).zip(tracks).map(|((ids,b),tracks)|json!({"axis_line_indices":ids,"bbox":b.raw,"vertical_positions":tracks})).collect::<Vec<_>>(),"caption_candidates":cs.into_iter().map(|(id,b)|json!({"source_index":id,"local_bbox":b.raw})).collect::<Vec<_>>()});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
pub(super) fn connected_horizontal_rules(
    source: &Value,
    work: &mut Work,
) -> Result<Vec<Box4>, Error> {
    let p = prepare_source(source, &value_object!({"angle":0}), work)?;
    if p.lines.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for ids in &p.p.components {
        for &id in ids {
            work.charge(out.len() + 1)?;
            let rule = &p.p.rules[id];
            if rule.h && !out.iter().any(|b: &Box4| b.n == rule.b.n) {
                out.push(rule.b.clone())
            }
        }
    }
    Ok(out)
}
pub fn expand(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut work = Work(0);
    let p = prepare(v, &mut work)?;
    let excluded = exclusions(v, &p, &mut work)?;
    let grids = component_boxes(&p, &mut work)?
        .into_iter()
        .filter(|b| !excluded.iter().any(|e| overlap_smaller(b, e) >= 0.5))
        .collect::<Vec<_>>();
    work.charge(grids.len() * (excluded.len() + 1))?;
    let candidates = array(&v["candidates"], 5000)?;
    let out = value_object!({"candidates":expand_candidates(candidates,&p,&grids,&mut work)?});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
pub fn detect(v: &Value) -> Result<Value, Error> {
    detect_with_work(v, &mut Work(0))
}
pub fn detect_rules(v: &Value) -> Result<Value, Error> {
    detect_rules_with_work(v, &mut Work(0))
}
pub fn detect_supplemental(v: &Value) -> Result<Value, Error> {
    supplemental::detect(v, &mut Work(0))
}
pub(crate) fn detect_rules_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    let p = prepare(v, work)?;
    let excluded = exclusions(v, &p, work)?;
    let drafts = rule_intervals::drafts(&v["source"], &p, &excluded, work)?;
    let mut candidates = Vec::new();
    for draft in &drafts {
        candidates.push(rule_intervals::materialize(draft, &p, &excluded, work)?);
    }
    let out = value_object!({"candidates":candidates});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
pub(crate) fn detect_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    let p = prepare(v, work)?;
    let excluded = exclusions(v, &p, work)?;
    generate(&v["source"], &p, excluded.clone(), excluded, work)
}
fn generate(
    source: &Value,
    p: &Page,
    excluded: Vec<Box4>,
    leading: Vec<Box4>,
    work: &mut Work,
) -> Result<Value, Error> {
    let forms = local_boxes(source.get("form_bboxes").unwrap_or(&json!([])), &p, work)?;
    let mut obstacles = leading;
    for key in ["image_bboxes", "signature_bboxes", "form_bboxes"] {
        obstacles.extend(local_boxes(
            source.get(key).unwrap_or(&json!([])),
            &p,
            work,
        )?);
    }
    let boxes = component_boxes(&p, work)?;
    let cs = captions(&p, work)?;
    let mut candidates = Vec::new();
    for (ids, b) in p.p.components.iter().zip(&boxes) {
        work.charge(excluded.len() + forms.len() + 1)?;
        if excluded.iter().any(|e| overlap_smaller(b, e) >= 0.5)
            || forms.iter().any(|f| overlap_first(b, f) >= 0.9)
        {
            continue;
        }
        let core = corridor(&p, b, &excluded, work)?;
        if core.is_empty() {
            continue;
        }
        let positions = graphic_grid::tracks(&p.p, ids, work)?;
        if positions.len() < 2 {
            continue;
        }
        let edge = 2f64.max(0.25 * p.em);
        if (positions[0] - b.n[0]).abs() > edge
            || (positions[positions.len() - 1] - b.n[2]).abs() > edge
        {
            continue;
        }
        if ids.len() == 2 {
            if positions.len() < 3 {
                continue;
            }
            let mut occupied = BTreeSet::new();
            for r in &core {
                for f in r["fragments"].as_array().unwrap() {
                    let x = center(&bbox(&f["local_bbox"])?, 0);
                    work.charge(positions.len() + 1)?;
                    let hits: Vec<_> = positions
                        .windows(2)
                        .enumerate()
                        .filter(|(_, p)| p[0] < x && x < p[1])
                        .map(|(i, _)| i)
                        .collect();
                    if hits.len() == 1 {
                        occupied.insert(hits[0]);
                    }
                }
            }
            if occupied.len() < 2 && !previous_matches(source, &p, b, &positions, &obstacles, work)?
            {
                continue;
            }
        }
        let caption = find_caption(&cs, b, p.em, work)?;
        let boundaries = vec![
            p.p.rules[ids[0]].b.raw.clone(),
            p.p.rules[*ids.last().unwrap()].b.raw.clone(),
        ];
        let q = json!({"rows":p.raw_rows,"lines":p.lines,"rule_bboxes":boundaries,"core_row_indices":[],"median_height":p.em,"page_size":p.size,"angle":p.angle,"caption_source_index":caption,"has_possible_table_notes":true});
        let mut c = super::rule_candidate_expansion::expand_with_core(&q, Some(&core), work)?
            ["candidate"]
            .clone();
        c["score"] = json!((100 + ids.len() + positions.len()) as f64);
        candidates.push(c);
    }
    let out = value_object!({"median_height":p.em,"candidates":candidates});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}

fn candidate_pairs(mut c: Value) -> Result<Value, Error> {
    for a in c["annotations"].as_array_mut().ok_or_else(invalid)? {
        if let Some(map) = a["line_bboxes"].as_object() {
            let pairs = map
                .iter()
                .map(|(id, b)| Ok(json!([id.parse::<i64>().map_err(|_| invalid())?, b])))
                .collect::<Result<Vec<_>, Error>>()?;
            a["line_bboxes"] = json!(pairs);
        }
    }
    Ok(c)
}
pub fn detect_page(v: &Value) -> Result<Value, Error> {
    detect_page_with_work(v, &mut Work(0))
}
pub(crate) fn detect_page_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    page_with_rules(v, false, false, work)
}
pub fn detect_ruled_page(v: &Value) -> Result<Value, Error> {
    page_with_rules(v, true, false, &mut Work(0))
}
pub fn detect_structural_page(v: &Value) -> Result<Value, Error> {
    page_with_rules(v, true, true, &mut Work(0))
}
pub fn apply_page_policies(v: &Value) -> Result<Value, Error> {
    page_policies::apply(v)
}
pub fn trailing_table_grid(v: &Value) -> Result<Value, Error> {
    page_policies::trailing(v)
}
pub(crate) fn trailing_table_grid_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    page_policies::trailing_with_work(v, work)
}
pub(super) fn trailing_table_grid_sources(
    source: &Value,
    prepared: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    page_policies::trailing_sources(source, prepared, work)
}
pub(crate) fn detect_supplemental_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    supplemental::detect(v, work)
}
pub(crate) fn detect_supplemental_source(
    source: &Value,
    options: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    supplemental::detect_source(source, options, work)
}
pub fn detect_complete_page(v: &Value) -> Result<Value, Error> {
    detect_complete_with_work(v, &mut Work(0))
}
pub(crate) fn detect_complete_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    page_impl(v, true, true, true, work)
}
fn page_with_rules(
    v: &Value,
    ordinary: bool,
    supplemental: bool,
    work: &mut Work,
) -> Result<Value, Error> {
    page_impl(v, ordinary, supplemental, false, work)
}
fn page_impl(
    v: &Value,
    ordinary: bool,
    supplemental: bool,
    finish: bool,
    work: &mut Work,
) -> Result<Value, Error> {
    validate_tree(v)?;
    page_sources(&v["source"], v, ordinary, supplemental, finish, work)
}
pub(crate) fn detect_complete_source(
    source: &Value,
    options: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    crate::legacy_page::validate_source_request(source, options)?;
    page_sources(source, options, true, true, true, work)
}
fn page_sources(
    source: &Value,
    v: &Value,
    ordinary: bool,
    supplemental: bool,
    finish: bool,
    work: &mut Work,
) -> Result<Value, Error> {
    let mut profile = crate::processing::NativeStageProfile::new();
    source.as_object().ok_or_else(invalid)?;
    let empty = json!([]);
    let paths = source.get("path_infos").unwrap_or(&empty);
    let excluded = v.get("excluded_bboxes").unwrap_or(&empty);
    array(excluded, 5000)?;
    let filled = super::filled_grid::detect_with_work(
        &value_object!({"path_infos":paths,"page_size":source["page_size"],"excluded_bboxes":excluded}),
        work,
    )?;
    profile.mark("table_candidates_filled");
    let tiled = super::tiled_tables::detect_with_work(
        &value_object!({"source":source,"excluded_bboxes":excluded}),
        work,
    )?;
    profile.mark("table_candidates_tiled");
    let tiles = tiled["candidates"].as_array().unwrap();
    let tile_boxes = tiles
        .iter()
        .map(|c| bbox(&c["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut base = Vec::new();
    for c in filled["candidates"].as_array().unwrap() {
        let b = bbox(&c["bbox"])?;
        work.charge(tile_boxes.len() + 1)?;
        if !tile_boxes.iter().any(|t| overlap_smaller(&b, t) >= 0.9) {
            base.push(candidate_pairs(c.clone())?);
        }
    }
    for c in tiles {
        base.push(candidate_pairs(c.clone())?);
    }
    let filled_boxes = base
        .iter()
        .map(|c| bbox(&c["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut angles = BTreeSet::new();
    for l in optional(source, "lines", 4096)? {
        let angle = l.get("angle").map_or(Ok(0.), num)?;
        if angle.fract() != 0. {
            return Err(invalid());
        }
        angles.insert(angle as i64);
    }
    let streaming = crate::legacy_page::budget::current().host
        && crate::legacy_page::budget::current().work == usize::MAX;
    let mut ruled = Vec::new();
    let mut contexts = Vec::new();
    let mut pending: Vec<(Option<(usize, usize)>, Value)> = Vec::new();
    for angle in angles {
        let q =
            value_object!({"angle":angle,"median_height":Value::Null,"excluded_bboxes":excluded});
        let p = prepare_source(source, &q, work)?;
        profile.mark("table_candidates_prepare");
        let original = exclusions(&q, &p, work)?;
        let mut local_excluded = Vec::new();
        let raw_excluded = array(excluded, 5000)?
            .iter()
            .map(bbox)
            .collect::<Result<Vec<_>, _>>()?;
        for b in raw_excluded.iter().chain(&filled_boxes) {
            work.charge(tile_boxes.len() + 1)?;
            let margin = if tile_boxes.iter().any(|t| t.n == b.n) {
                (2.5 * p.em).min(0.01 * num(&p.size[0])?)
            } else {
                2.5 * p.em
            };
            let b = local(b, &p.size, angle)?;
            local_excluded.push(bbox(&json!([
                b.n[0] - margin,
                b.n[1] - margin,
                b.n[2] + margin,
                b.n[3] + margin
            ]))?);
        }
        let drafts = if ordinary {
            rule_intervals::drafts(source, &p, &local_excluded, work)?
        } else {
            Vec::new()
        };
        profile.mark("table_candidates_drafts");
        for (i, draft) in drafts.iter().enumerate() {
            pending.push((
                Some((contexts.len(), i)),
                if streaming {
                    rule_intervals::projection_geometry(draft, &p)
                } else {
                    rule_intervals::projection(draft, &p)
                },
            ));
        }
        let generated = generate(source, &p, local_excluded.clone(), original, work)?;
        profile.mark("table_candidates_generate");
        for c in generated["candidates"].as_array().unwrap() {
            if ordinary {
                let mut projection = c.clone();
                projection["kind"] = json!("table");
                pending.push((None, projection));
            } else {
                ruled.push(c.clone());
            }
        }
        if ordinary {
            contexts.push((
                p,
                drafts,
                local_excluded,
                rule_intervals::MaterializationCache::default(),
            ));
        }
        if ruled.len() + pending.len() > rapiddoc_core::resources::limit(5000) {
            return Err(limit());
        }
    }
    let mut ordered_merger = super::table_candidate_merge::OrderedMerger::new();
    if ordinary {
        let result = if streaming {
            let projected: Vec<_> = pending.iter().map(|(_, c)| c).collect();
            super::prose_rule_conflicts::exclude_lazy_members(
                source,
                &projected,
                &mut |index| {
                    let (context, draft) = pending[index].0.unwrap();
                    let (page, drafts, _, cache) = &mut contexts[context];
                    rule_intervals::draft_members(&drafts[draft], page, cache)
                },
                work,
            )?
        } else {
            let projected: Vec<_> = pending.iter().map(|(_, c)| c.clone()).collect();
            super::prose_rule_conflicts::exclude_with_work(
                &value_object!({"source":source,"candidates":projected}),
                work,
            )?
        };
        profile.mark("table_candidates_prose_conflicts");
        let rejected: BTreeSet<_> = result["rejected_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i.as_u64().unwrap() as usize)
            .collect();
        drop(result);
        let mut retained: Vec<_> = pending
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !rejected.contains(i))
            .map(|(_, (draft, c))| {
                (
                    draft,
                    if streaming && draft.is_some() {
                        Value::Null
                    } else {
                        c
                    },
                )
            })
            .collect();
        if streaming {
            retained.sort_by(|(a, c), (b, d)| {
                let score = |draft: &Option<(usize, usize)>, value: &Value| {
                    draft.map_or_else(
                        || value["score"].as_f64().unwrap(),
                        |(context, index)| contexts[context].1[index].score,
                    )
                };
                score(b, d).partial_cmp(&score(a, c)).unwrap()
            });
        }
        for (draft, mut c) in retained {
            if let Some((context, index)) = draft {
                let (p, drafts, excluded, notes) = &mut contexts[context];
                if streaming {
                    let candidate = rule_intervals::materialize_typed(
                        &drafts[index],
                        p,
                        excluded,
                        notes,
                        work,
                    )?;
                    ordered_merger.add_candidate(candidate, work)?;
                    continue;
                }
                c = rule_intervals::materialize_cached(&drafts[index], p, excluded, notes, work)?;
            } else {
                c.as_object_mut().unwrap().remove("kind");
            }
            if streaming {
                ordered_merger.add(&c, work)?;
            } else {
                ruled.push(c);
            }
        }
    }
    profile.mark("table_candidates_materialize");
    let merged = if streaming && ordinary {
        ordered_merger.finish(work)?
    } else {
        super::table_candidate_merge::merge_with_work(&value_object!({"candidates":ruled}), work)?
    };
    profile.mark("table_candidates_merge");
    for c in merged["candidates"].as_array().unwrap() {
        let b = bbox(&c["bbox"])?;
        work.charge(filled_boxes.len() + 1)?;
        if !filled_boxes.iter().any(|f| overlap_smaller(&b, f) >= 0.2) {
            base.push(c.clone());
        }
    }
    if supplemental {
        let mut detector = supplemental::Detector::new(source);
        let shaded = detector.detect("shaded_header", &excluded, work)?;
        base.extend(shaded["candidates"].as_array().unwrap().iter().cloned());
        let open = detector.detect("open_vertical", &excluded, work)?;
        let partial = detector.detect("partial_grid", &excluded, work)?;
        let mut recovered_candidates = partial["candidates"].as_array().unwrap().clone();
        recovered_candidates.extend(open["candidates"].as_array().unwrap().iter().cloned());
        let mut recovered = Vec::new();
        for c in &recovered_candidates {
            let core = bbox(&c["core_bbox"])?;
            work.charge(base.len() + 1)?;
            if !base.iter().any(|b| {
                overlap_first(
                    &core,
                    &bbox(
                        b.get("core_bbox")
                            .filter(|v| !v.is_null())
                            .unwrap_or(&b["bbox"]),
                    )
                    .unwrap(),
                ) >= 0.98
            }) {
                recovered.push(c.clone());
            }
        }
        let bounds = recovered
            .iter()
            .map(|c| bbox(&c["core_bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        work.charge(base.len() * (bounds.len() + 1))?;
        base.retain(|c| {
            let b = bbox(
                c.get("core_bbox")
                    .filter(|v| !v.is_null())
                    .unwrap_or(&c["bbox"]),
            )
            .unwrap();
            !bounds.iter().any(|r| overlap_first(&b, r) >= 0.9)
        });
        let years = detector.detect("year_header", &excluded, work)?;
        let numeric = detector.detect("aligned_numeric", &excluded, work)?;
        let mut numeric_candidates = Vec::new();
        for c in numeric["candidates"].as_array().unwrap() {
            let b = bbox(&c["bbox"])?;
            work.charge(base.len() + recovered.len() + 1)?;
            let mut covered = false;
            for old in base.iter().chain(&recovered) {
                let core = bbox(
                    old.get("core_bbox")
                        .filter(|v| !v.is_null())
                        .unwrap_or(&old["bbox"]),
                )?;
                let outer = bbox(&old["bbox"])?;
                if overlap_smaller(&core, &b) >= 0.9
                    && outer.n[2] - outer.n[0] >= 0.95 * (b.n[2] - b.n[0])
                    && outer.n[3] - outer.n[1] >= 0.85 * (b.n[3] - b.n[1])
                {
                    covered = true;
                    break;
                }
            }
            if !covered {
                numeric_candidates.push(c.clone());
            }
        }
        let metrics = detector.detect("metric_pair", &excluded, work)?;
        let mut replacements = years["candidates"].as_array().unwrap().clone();
        replacements.extend(metrics["candidates"].as_array().unwrap().iter().cloned());
        replacements.extend(numeric_candidates);
        let preferred = super::table_replacements::prefer_with_work(
            &value_object!({"base":base,"recovered":recovered,"replacements":replacements}),
            work,
        )?;
        base = preferred["base"].as_array().unwrap().clone();
        base.extend(
            preferred["replacements"]
                .as_array()
                .unwrap()
                .iter()
                .cloned(),
        );
        base.extend(preferred["recovered"].as_array().unwrap().iter().cloned());
        let mut clip_excluded = array(excluded, 5000)?.to_vec();
        clip_excluded.extend(base.iter().map(|c| {
            c.get("core_bbox")
                .filter(|v| !v.is_null())
                .unwrap_or(&c["bbox"])
                .clone()
        }));
        let short = detector.detect("short_caption", &excluded, work)?;
        base.extend(short["candidates"].as_array().unwrap().iter().cloned());
        let blank = detector.detect("blank_answer", &excluded, work)?;
        base.extend(blank["candidates"].as_array().unwrap().iter().cloned());
        for kind in ["unruled_numeric", "captioned_mixed", "captioned_pair"] {
            let result = detector.detect(kind, &clip_excluded, work)?;
            base.extend(result["candidates"].as_array().unwrap().iter().cloned());
        }
        let clipped = detector.detect("page_clipped", &clip_excluded, work)?;
        base.extend(clipped["candidates"].as_array().unwrap().iter().cloned());
    }
    profile.mark("table_candidates_supplemental");
    if base.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let mut member_count = 0;
    let mut grid_count = 0;
    for c in &base {
        member_count += c["line_indices"].as_array().ok_or_else(invalid)?.len();
        grid_count += c["inferred_grid"].as_array().ok_or_else(invalid)?.len();
        for a in c["annotations"].as_array().ok_or_else(invalid)? {
            member_count += a["line_indices"].as_array().ok_or_else(invalid)?.len();
        }
    }
    work.charge(member_count + grid_count + 1)?;
    if member_count > rapiddoc_core::resources::limit(20000)
        || grid_count > rapiddoc_core::resources::limit(20000)
    {
        return Err(limit());
    }
    if finish {
        let result = page_policies::finish(source, base, "all", work);
        profile.mark("table_candidates_policies");
        drop(contexts);
        profile.mark("table_candidates_context_release");
        return result;
    }
    work.charge(base.len() * 16 + 1)?;
    base.sort_by(|a, b| {
        let a = &a["bbox"];
        let b = &b["bbox"];
        a[1].as_f64()
            .unwrap()
            .partial_cmp(&b[1].as_f64().unwrap())
            .unwrap()
            .then(
                a[0].as_f64()
                    .unwrap()
                    .partial_cmp(&b[0].as_f64().unwrap())
                    .unwrap(),
            )
    });
    let out = value_object!({"candidates":base});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "development-only external input; set RAPIDDOC_RULE_PDF or RAPIDDOC_RULE_REPLAY"]
    fn external_rule_page_profile() {
        if let Ok(path) = std::env::var("RAPIDDOC_RULE_REPLAY") {
            let q: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            let _scope = crate::legacy_page::budget::host_page();
            let mut work = Work(0);
            let p = prepare_source(&q["source"], &q["options"], &mut work).unwrap();
            let excluded = q["excluded"]
                .as_array()
                .unwrap()
                .iter()
                .map(bbox)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let start = std::time::Instant::now();
            let result = rule_intervals::drafts(&q["source"], &p, &excluded, &mut work).unwrap();
            eprintln!("drafts={} elapsed={:?}", result.len(), start.elapsed());
            if std::env::var_os("RAPIDDOC_RULE_MATERIALIZE").is_some() {
                let mut cache = rule_intervals::MaterializationCache::default();
                let mut drafts = result;
                drafts.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                for d in drafts.iter().take(20) {
                    let start = std::time::Instant::now();
                    let c = super::super::table_candidate_merge::into_value(
                        rule_intervals::materialize_typed(d, &p, &excluded, &mut cache, &mut work)
                            .unwrap(),
                    );
                    eprintln!(
                        "materialized {:?} members={}",
                        start.elapsed(),
                        c["line_indices"].as_array().unwrap().len()
                    );
                }
            }
            return;
        }
        let path = std::env::var("RAPIDDOC_RULE_PDF").expect("external PDF path");
        let engine = crate::RapidDoc::new(
            crate::RapidDocConfig {
                layout_config: json!({"model_type":"docvortex_rule"}),
                formula_enable: false,
                table_enable: false,
                ..Default::default()
            },
            crate::RuntimeResources::default(),
        );
        let result = engine.parse(
            path,
            &crate::ParseOptions {
                end_page: Some(1),
                ..Default::default()
            },
        );
        if std::env::var_os("RAPIDDOC_RULE_CAPTURE").is_some() {
            assert!(result.is_err());
        } else {
            eprintln!("parsed={:?}", result.map(|d| d.markdown));
        }
    }
    #[test]
    fn shared_supplemental_preparation_matches_installed_real_sources() {
        let _scope = crate::legacy_page::budget::host_page();
        let fixture: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(
                "../../../../../../tests/fixtures/upstream_068/shared_supplemental_tables.json.gz"
            )[..],
        ))
        .unwrap();
        for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
            let actual = detect_complete_with_work(&case["request"], &mut Work(0)).unwrap();
            assert_eq!(actual, case["expected"], "real table source {i}");
        }
    }
    #[test]
    fn supplemental_cache_preserves_finite_work_counts() {
        let fixture: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(
                "../../../../../../tests/fixtures/upstream_068/shared_supplemental_tables.json.gz"
            )[..],
        ))
        .unwrap();
        let source = &fixture["cases"][0]["request"]["source"];
        crate::legacy_page::budget::with_resources(
            crate::legacy_page::budget::HostResources {
                max_work: Some(usize::MAX - 1),
                ..Default::default()
            },
            || {
                let _scope = crate::legacy_page::budget::host_page();
                let mut detector = supplemental::Detector::new(source);
                let mut old_work = Work(0);
                let mut new_work = Work(0);
                for kind in [
                    "shaded_header",
                    "open_vertical",
                    "partial_grid",
                    "year_header",
                    "aligned_numeric",
                    "metric_pair",
                    "short_caption",
                    "blank_answer",
                    "unruled_numeric",
                    "captioned_mixed",
                    "captioned_pair",
                    "page_clipped",
                ] {
                    let request = json!({"source":source,"kind":kind,"excluded_bboxes":[]});
                    let old = supplemental::detect(&request, &mut old_work).unwrap();
                    let mut source_work = Work(new_work.0);
                    let direct =
                        detect_supplemental_source(source, &request, &mut source_work).unwrap();
                    let actual = detector.detect(kind, &json!([]), &mut new_work).unwrap();
                    assert_eq!(actual, old, "{kind}");
                    assert_eq!(direct, old, "borrowed source {kind}");
                    assert_eq!(source_work.0, old_work.0, "borrowed source {kind}");
                    assert_eq!(new_work.0, old_work.0, "{kind}");
                }
            },
        );
    }
    #[test]
    fn original_page_policies() {
        facts("native-table-page-policies", apply_page_policies);
        facts("native-table-page-policy-boundaries", apply_page_policies);
    }
    #[test]
    fn original_complete_candidate_detection() {
        facts("native-complete-table-candidates", detect_complete_page);
    }
    #[test]
    fn original_trailing_table_grid() {
        facts("native-trailing-table-grid", trailing_table_grid);
    }
    #[test]
    fn page_policies_share_complete_detector_budget() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build/migration");
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("native-table-page-policies.json")).unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let q = &c["request"];
            let mut w = Work(0);
            let cs = q["candidates"].as_array().unwrap().clone();
            let op = q["operation"].as_str().unwrap();
            assert!(page_policies::finish(&q["source"], cs.clone(), op, &mut w).is_ok());
            assert!(
                page_policies::finish(&q["source"], cs, op, &mut Work(10_000_000 - w.0 + 1))
                    .is_err()
            );
            assert!(apply_page_policies(q).is_ok());
        }
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("native-complete-table-candidates.json")).unwrap(),
        )
        .unwrap();
        let q = &f["cases"][0]["request"];
        let mut w = Work(0);
        assert!(page_impl(q, true, true, true, &mut w).is_ok());
        assert!(page_impl(q, true, true, true, &mut Work(10_000_000 - w.0 + 1)).is_err());
        assert!(detect_complete_page(q).is_ok());
    }
    fn facts(name: &str, f: fn(&Value) -> Result<Value, Error>) {
        let corpus: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join(format!("../../build/migration/{name}.json")),
            )
            .unwrap(),
        )
        .unwrap();
        for c in corpus["cases"].as_array().unwrap() {
            assert_eq!(
                f(&c["request"]).unwrap(),
                c["expected"],
                "{name} {}",
                c["label"]
            );
        }
    }
    #[test]
    fn original_page_rows_caption_and_tracks() {
        facts("native-rule-page", inspect);
    }
    #[test]
    fn original_full_rule_interval_generator() {
        facts("native-rule-interval-pipeline", detect_rules);
    }
    #[test]
    fn original_combined_ruled_page() {
        facts("native-ruled-table-page", detect_ruled_page);
    }
    #[test]
    fn original_supplemental_rule_tables() {
        facts("native-supplemental-rule-tables", detect_supplemental);
    }
    #[test]
    fn original_structural_page() {
        facts("native-structural-table-page", detect_structural_page);
        facts(
            "native-structural-table-expanded-page",
            detect_structural_page,
        );
    }
    #[test]
    fn original_shaded_table_pipeline() {
        facts("native-shaded-table-pipeline", detect_supplemental);
    }
    #[test]
    fn original_form_table_pipeline() {
        facts("native-form-table-pipeline", detect_supplemental);
        facts("native-blank-answer-positive", detect_supplemental);
    }
    #[test]
    fn original_numeric_candidate_segments() {
        facts("native-numeric-candidate-segments", detect_supplemental);
        facts("native-numeric-candidate-page", detect_structural_page);
    }
    #[test]
    fn original_text_numeric_candidates() {
        facts("native-text-numeric-candidates", detect_supplemental);
        facts("native-text-numeric-page", detect_structural_page);
    }
    #[test]
    fn original_final_candidate_producers() {
        facts("native-final-candidate-producers", detect_supplemental);
        facts("native-final-candidate-specials", detect_supplemental);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-final-candidate-producers.json");
        let corpus: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in corpus["errors"].as_array().unwrap() {
            assert_eq!(
                detect_supplemental(&c["request"]).unwrap_err().code,
                rapiddoc_core::ErrorCode::InvalidInput
            );
        }
    }
    #[test]
    fn original_numeric_region_bounds() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-final-candidate-specials.json");
        let corpus: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in corpus["regions"].as_array().unwrap() {
            let text = serde_json::from_value(c["text"].clone()).unwrap();
            assert_eq!(
                json!(super::super::table_native_aligned_numeric::regions(
                    &c["table_input"],
                    &text,
                    &mut Work(0)
                )
                .unwrap()),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn original_all_candidate_producer_branches() {
        facts("native-all-candidate-producers", detect_structural_page);
    }
    #[test]
    fn final_producer_shared_budget_and_recovery() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-final-candidate-producers.json");
        let corpus: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for kind in ["aligned_numeric", "captioned_mixed"] {
            let q = &corpus["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| {
                    c["request"]["kind"] == kind
                        && !c["expected"]["candidates"].as_array().unwrap().is_empty()
                })
                .unwrap()["request"];
            let mut work = Work(0);
            assert!(supplemental::detect(q, &mut work).is_ok());
            assert!(supplemental::detect(q, &mut Work(10_000_000 - work.0 + 1)).is_err());
            assert!(detect_supplemental(q).is_ok());
        }
    }
    #[test]
    fn rule_interval_shared_budget() {
        let q = json!({"source":{"page_size":[400,600],"lines":[],"drawing_lines":[]}});
        assert!(detect_rules_with_work(&q, &mut Work(10_000_000)).is_err());
        assert!(detect_rules(&q).is_ok());
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-rule-interval-pipeline.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        let q = &f["cases"][0]["request"];
        let mut work = Work(0);
        assert!(detect_rules_with_work(q, &mut work).is_ok());
        assert!(detect_rules_with_work(q, &mut Work(10_000_000 - work.0 + 1)).is_err());
    }
    #[test]
    fn structural_and_supplemental_shared_budget() {
        let source =
            json!({"page_size":[400,600],"lines":[],"drawing_lines":[],"path_infos":[],"chars":[]});
        let q = json!({"source":source,"excluded_bboxes":[]});
        assert!(page_with_rules(&q, true, true, &mut Work(10_000_000)).is_err());
        assert!(detect_structural_page(&q).is_ok());
        for kind in [
            "open_vertical",
            "short_caption",
            "shaded_header",
            "blank_answer",
            "page_clipped",
            "partial_grid",
            "year_header",
            "metric_pair",
            "unruled_numeric",
            "captioned_pair",
            "aligned_numeric",
            "captioned_mixed",
        ] {
            let q = json!({"source":source,"kind":kind,"excluded_bboxes":[]});
            assert!(supplemental::detect(&q, &mut Work(10_000_000)).is_err());
            assert!(detect_supplemental(&q).is_ok());
        }
    }
    #[test]
    fn original_closed_rule_generator() {
        facts("native-closed-rule-tables", detect);
    }
    #[test]
    fn original_connected_grid_expansion() {
        facts("native-connected-table-grids", expand);
    }
    #[test]
    fn original_physical_page_orchestration() {
        facts("native-physical-table-page", detect_page);
    }
    #[test]
    fn physical_page_shared_budget() {
        let q = json!({"source":{"page_size":[400,600],"path_infos":[],"lines":[],"chars":[]},"excluded_bboxes":[]});
        assert!(detect_page_with_work(&q, &mut Work(10_000_000)).is_err());
        assert!(detect_page(&q).is_ok());
    }
    #[test]
    fn shared_budget() {
        let corpus: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-closed-rule-tables.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let q = &corpus["cases"][0]["request"];
        let mut work = Work(0);
        assert!(detect_with_work(q, &mut work).is_ok());
        assert!(detect_with_work(q, &mut Work(10_000_000 - work.0 + 1)).is_err());
    }
}
