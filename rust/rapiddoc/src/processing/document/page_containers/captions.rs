//! 表格之前的完整图题支持容器：真正表格/代码屏障、照片外正文、图题卡与两轮扩框。
use super::*;
use crate::legacy_model::{
    closed_grid, closed_rule_pipeline, code_containers, graphic_captions,
    path_graphics::{area, integer, overlap_first, overlap_smaller},
};
use crate::legacy_page::eq;
use rapiddoc_native::{
    python_decimal,
    text_boundary::{python_alnum, python_space},
};
use std::collections::BTreeSet;
fn cx(b: &Box4) -> f64 {
    (b.n[0] + b.n[2]) / 2.
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) / 2.
}
fn fold(c: char) -> char {
    match c {
        'İ' | 'ı' => 'i',
        'ſ' => 's',
        'K' => 'k',
        _ => c.to_ascii_lowercase(),
    }
}
fn starts(c: &[char], s: &str) -> bool {
    s.chars()
        .enumerate()
        .all(|(i, x)| c.get(i).is_some_and(|c| fold(*c) == x))
}
fn figure(s: &str, bare: bool, explicit: bool) -> bool {
    let c: Vec<_> = s.trim_matches(python_space).chars().collect();
    let mut i = if starts(&c, "figure") {
        6
    } else if starts(&c, "fig.") {
        4
    } else if !explicit && starts(&c, "fig") {
        3
    } else {
        return false;
    };
    if explicit && !bare {
        return true;
    }
    if c.get(i) == Some(&'.') {
        i += 1
    }
    while c.get(i).is_some_and(|c| {
        if bare {
            python_space(*c)
        } else {
            matches!(c, ' ' | '\t')
        }
    }) {
        i += 1
    }
    let n = i;
    while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
        i += 1
    }
    i > n && (!bare || i == c.len())
}
fn words(s: &str, min: usize, boundary: bool) -> usize {
    let c: Vec<_> = s.chars().collect();
    let mut n = 0;
    let mut i = 0;
    while i < c.len() {
        if !c[i].is_ascii_alphabetic() {
            i += 1;
            continue;
        }
        let start = i;
        while c.get(i).is_some_and(|c| c.is_ascii_alphabetic()) {
            i += 1
        }
        if i - start >= min
            && (!boundary
                || (start == 0 || !(python_alnum(c[start - 1]) || c[start - 1] == '_'))
                    && (i == c.len() || !(python_alnum(c[i]) || c[i] == '_')))
        {
            n += 1;
        }
    }
    n
}
fn language(s: &str) -> usize {
    words(s, 2, false)
        + s.chars()
            .filter(|c| ('\u{3400}'..='\u{9fff}').contains(c))
            .count()
}
fn ending(s: &str) -> bool {
    s.trim_end_matches(python_space)
        .chars()
        .last()
        .is_some_and(|c| ".!?。！？".contains(c))
}
fn numbered_code(s: &str) -> usize {
    s.split('\n')
        .filter(|s| {
            let mut c: Vec<_> = s.trim_start_matches(python_space).chars().collect();
            let n = c
                .iter()
                .take_while(|&&c| python_decimal(c).is_some())
                .count();
            if n > 0 {
                let mut i = n;
                while c.get(i).is_some_and(|c| python_space(*c)) {
                    i += 1
                }
                if c.get(i).is_some_and(|c| ":.".contains(*c)) {
                    i += 1
                }
                while c.get(i).is_some_and(|c| python_space(*c)) {
                    i += 1
                }
                c = c[i..].to_vec();
            }
            [
                "int", "float", "void", "class", "def", "for", "while", "if", "return", "else",
            ]
            .iter()
            .any(|s| {
                starts(&c, s)
                    && c.get(s.len())
                        .is_none_or(|c| !(python_alnum(*c) || *c == '_'))
            })
        })
        .count()
}
fn group(ls: &[Value]) -> Result<i64, Error> {
    ls.iter()
        .filter_map(|l| l.get("paragraph_group").filter(|v| !v.is_null()))
        .map(|v| v.as_i64().ok_or_else(invalid))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(-1)
        .checked_add(1)
        .ok_or_else(limit)
}
fn ink(l: &Value) -> Result<Box4, Error> {
    bbox(
        l.get("ink_bbox")
            .filter(|v| !v.is_null())
            .unwrap_or(&l["bbox"]),
    )
}
pub(super) fn allow(
    l: &Value,
    b: &Box4,
    maps: &[(Box4, BTreeSet<i64>)],
    w: &mut Work,
) -> Result<bool, Error> {
    let mut any = false;
    for (form, _) in maps {
        w.charge(1)?;
        if overlap_first(form, b) >= 0.75 && super::forms::member(l, form, &[], w)?.is_some() {
            any = true;
            if super::forms::member(l, form, maps, w)?.is_some() {
                return Ok(true);
            }
        }
    }
    Ok(!any)
}
pub(super) fn corridor(b: &Box4, lanes: &[Value], width: f64) -> Option<(f64, f64)> {
    if lanes.is_empty() {
        return None;
    }
    let count = lanes
        .iter()
        .filter(|l| {
            let left = l["left"].as_f64().unwrap();
            let right = l["right"].as_f64().unwrap();
            (b.n[2].min(right) - b.n[0].max(left)).max(0.)
                >= 0.25 * (b.n[2] - b.n[0]).min(right - left)
        })
        .count();
    if count > 1 {
        return Some((0., width));
    }
    let mut edges = vec![0.];
    for pair in lanes.windows(2) {
        edges.push((pair[0]["right"].as_f64().unwrap() + pair[1]["left"].as_f64().unwrap()) / 2.);
    }
    edges.push(width);
    edges
        .windows(2)
        .find(|p| p[0] <= cx(b) && cx(b) <= p[1])
        .map(|p| (p[0], p[1]))
}
fn prose(
    ls: &mut [Value],
    images: &[Box4],
    caps: &BTreeSet<i64>,
    em: f64,
    w: &mut Work,
) -> Result<BTreeSet<i64>, Error> {
    let mut free = Vec::new();
    for (i, l) in ls.iter().enumerate() {
        w.charge(images.len() + l["text"].as_str().ok_or_else(invalid)?.len() + 1)?;
        let b = bbox(&l["bbox"])?;
        if integer(l, "angle", 0)? == 0
            && l["semantic_type"].is_null()
            && !caps.contains(&integer(l, "source_index", 0)?)
            && language(l["text"].as_str().unwrap()) >= 1
            && (0.7 * em..=1.3 * em).contains(&height(l, &b)?)
            && images
                .iter()
                .all(|image| overlap_first(&ink(l).unwrap(), image) <= 0.1)
        {
            free.push(i)
        }
    }
    free.sort_by(|&i, &j| {
        ls[i]["bbox"][1]
            .as_f64()
            .partial_cmp(&ls[j]["bbox"][1].as_f64())
            .unwrap()
            .then(
                ls[i]["bbox"][0]
                    .as_f64()
                    .partial_cmp(&ls[j]["bbox"][0].as_f64())
                    .unwrap(),
            )
    });
    let mut rows: Vec<Vec<usize>> = Vec::new();
    for i in free {
        let b = bbox(&ls[i]["bbox"])?;
        let mut peer = None;
        for (j, row) in rows.iter().enumerate() {
            w.charge(row.len() + 1)?;
            let first = bbox(&ls[row[0]]["bbox"])?;
            let right = row
                .iter()
                .map(|&i| ls[i]["bbox"][2].as_f64().unwrap())
                .fold(f64::NEG_INFINITY, f64::max);
            if graphic_captions::axis(&first, &b, 1) >= 0.65
                && (-0.1 * em..=3. * em).contains(&(b.n[0] - right))
            {
                peer = Some(j)
            }
        }
        if let Some(j) = peer {
            rows[j].push(i)
        } else {
            rows.push(vec![i]);
        }
    }
    let bounds = |row: &[usize], ls: &[Value]| -> Result<Box4, Error> {
        let bs = row
            .iter()
            .map(|&i| bbox(&ls[i]["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        combined(&bs)
    };
    let main = |row: &[usize], ls: &[Value]| -> usize {
        let mut best = row[0];
        for &i in &row[1..] {
            if ls[i]["text"].as_str().unwrap().chars().count()
                > ls[best]["text"].as_str().unwrap().chars().count()
            {
                best = i;
            }
        }
        best
    };
    let mut pending = rows;
    let mut protected = BTreeSet::new();
    while !pending.is_empty() {
        let mut run = vec![pending.remove(0)];
        loop {
            let p = bounds(run.last().unwrap(), ls)?;
            let m = main(run.last().unwrap(), ls);
            let mut follower = None;
            for (j, row) in pending.iter().enumerate() {
                w.charge(row.len() + 1)?;
                let b = bounds(row, ls)?;
                if eq(
                    &ls[main(row, ls)]["font_signature"],
                    &ls[m]["font_signature"],
                ) && (b.n[0] - p.n[0]).abs() <= 0.4 * em
                    && (-0.25 * em..=0.85 * em).contains(&(b.n[1] - p.n[3]))
                    && cy(&b) > cy(&p) + 0.5 * em
                    && follower
                        .is_none_or(|k: usize| b.n[1] < bounds(&pending[k], ls).unwrap().n[1])
                {
                    follower = Some(j)
                }
            }
            let Some(j) = follower else { break };
            let b = bounds(&pending[j], ls)?;
            let last = run.last().unwrap();
            let text = last
                .iter()
                .map(|&i| ls[i]["text"].as_str().unwrap())
                .collect::<Vec<_>>()
                .join(" ");
            if run.len() >= 2
                && last.iter().any(|&i| ls[i]["paragraph_terminal"] == true)
                && ending(&text)
                && p.n[2] - p.n[0] < 0.65 * (b.n[2] - b.n[0])
            {
                break;
            }
            run.push(pending.remove(j));
        }
        let text = run
            .iter()
            .flatten()
            .map(|&i| ls[i]["text"].as_str().unwrap())
            .collect::<Vec<_>>()
            .join(" ");
        let trailing = text
            .trim_matches(python_space)
            .trim_end_matches(|c| python_decimal(c).is_some() || python_space(c));
        if run.len() >= 3 && language(&text) >= 15 && ending(trailing) {
            for &i in run.iter().flatten() {
                protected.insert(integer(&ls[i], "source_index", 0)?);
            }
            let all = run.iter().flatten().copied().collect::<Vec<_>>();
            let b = bounds(&all, ls)?;
            let tail = bounds(run.last().unwrap(), ls)?;
            if images.iter().any(|im| {
                graphic_captions::axis(&b, im, 0) <= 0.1
                    && graphic_captions::axis(&b, im, 1) >= 0.5
                    && b.n[2] - b.n[0] <= 1.5 * (im.n[2] - im.n[0])
                    && (-0.3 * em..=2. * em).contains(&(b.n[1] - im.n[1]))
                    && (0.0..=5. * em).contains(&(b.n[3] - im.n[3]))
                    && ls[run.last().unwrap()[0]]["paragraph_terminal"] == true
                    && tail.n[2] - b.n[0] < 0.7 * (b.n[2] - b.n[0])
            }) {
                let g = group(ls)?;
                for &i in &all {
                    ls[i]["paragraph_group"] = json!(g);
                }
            }
        }
    }
    Ok(protected)
}
fn side(
    ls: &[Value],
    images: &[Box4],
    rules: &[Value],
    i: usize,
    w: &mut Work,
) -> Result<Vec<usize>, Error> {
    let l = &ls[i];
    if !figure(l["text"].as_str().unwrap(), true, true) || l["font_signature"].is_null() {
        return Ok(Vec::new());
    }
    let b = bbox(&l["bbox"])?;
    let em = height(l, &b)?;
    if images
        .iter()
        .filter(|im| {
            im.n[2] - im.n[0] >= 8. * em
                && im.n[3] - im.n[1] >= 8. * em
                && (0.0..=6. * em).contains(&(b.n[0] - im.n[2]))
                && im.n[1] <= cy(&b)
                && cy(&b) <= im.n[3]
        })
        .count()
        != 1
    {
        return Ok(Vec::new());
    }
    let mut desc = Vec::new();
    for (j, other) in ls.iter().enumerate() {
        w.charge(1)?;
        if j == i
            || integer(other, "angle", 0)? != integer(l, "angle", 0)?
            || !other["semantic_type"].is_null()
            || !eq(&other["font_signature"], &l["font_signature"])
        {
            continue;
        }
        let ob = bbox(&other["bbox"])?;
        if (0.85..=1.15).contains(&(height(other, &ob)? / em))
            && (0.5 * em..=6. * em).contains(&(ob.n[0] - b.n[2]))
            && (cy(&ob) - cy(&b)).abs() <= 3. * em
        {
            desc.push(j);
        }
    }
    desc.sort_by(|&i, &j| {
        ls[i]["bbox"][1]
            .as_f64()
            .partial_cmp(&ls[j]["bbox"][1].as_f64())
            .unwrap()
            .then(
                ls[i]["bbox"][0]
                    .as_f64()
                    .partial_cmp(&ls[j]["bbox"][0].as_f64())
                    .unwrap(),
            )
    });
    if desc.len() < 3 {
        return Ok(Vec::new());
    }
    let text = desc
        .iter()
        .map(|&i| ls[i]["text"].as_str().unwrap())
        .collect::<Vec<_>>()
        .join(" ");
    let lefts = desc
        .iter()
        .map(|&i| ls[i]["bbox"][0].as_f64().unwrap())
        .collect::<Vec<_>>();
    if words(&text, 2, false) < 8
        || lefts.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - lefts.iter().copied().fold(f64::INFINITY, f64::min)
            > 0.5 * em
        || desc.windows(2).any(|p| {
            !(0.7 * em..=1.75 * em).contains(
                &(cy(&bbox(&ls[p[1]]["bbox"]).unwrap()) - cy(&bbox(&ls[p[0]]["bbox"]).unwrap())),
            )
        })
    {
        return Ok(Vec::new());
    }
    let members = std::iter::once(i).chain(desc).collect::<Vec<_>>();
    let bs = members
        .iter()
        .map(|&i| bbox(&ls[i]["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let bound = combined(&bs)?;
    let mut top = false;
    let mut bottom = false;
    let mut left = false;
    let mut right = false;
    for rule in rules {
        w.charge(1)?;
        if rule.get("width").map_or(Ok(0.), num)? > 0.15 * em {
            continue;
        }
        let r = bbox(&rule["bbox"])?;
        if rule["orientation"] == "horizontal"
            && r.n[0] <= bound.n[0] + 0.25 * em
            && r.n[2] >= bound.n[2] - 0.25 * em
        {
            top |= (-0.1 * em..=0.5 * em).contains(&(bound.n[1] - r.n[3]));
            bottom |= (-0.1 * em..=0.5 * em).contains(&(r.n[1] - bound.n[3]));
        }
        if rule["orientation"] == "vertical"
            && r.n[1] <= bound.n[1] + 0.25 * em
            && r.n[3] >= bound.n[3] - 0.25 * em
        {
            left |= (0.0..=em).contains(&(bound.n[0] - r.n[2]));
            right |= (0.0..=1.5 * em).contains(&(r.n[0] - bound.n[2]));
        }
    }
    Ok(if top && bottom && left && right {
        members
    } else {
        Vec::new()
    })
}
fn heading_floor(
    ls: &[Value],
    images: &[Box4],
    cb: &Box4,
    em: f64,
    w: &mut Work,
) -> Result<f64, Error> {
    let mut floor = 0f64;
    let mut order = (0..ls.len()).collect::<Vec<_>>();
    order.sort_by(|&i, &j| {
        ls[i]["bbox"][1]
            .as_f64()
            .partial_cmp(&ls[j]["bbox"][1].as_f64())
            .unwrap()
    });
    for im in images {
        if im.n[3] > cb.n[1] + 0.2 * em
            || !(0.0..=8. * em).contains(&(cb.n[1] - im.n[3]))
            || graphic_captions::axis(im, cb, 0) < 0.35
        {
            continue;
        }
        let width = im.n[2] - im.n[0];
        for h in ls {
            w.charge(1)?;
            let b = bbox(&h["bbox"])?;
            let hh = height(h, &b)?;
            if integer(h, "angle", 0)? != 0
                || h["caption_start"] == true
                || hh < 1.3 * em
                || b.n[2] - b.n[0] < 0.5 * width
                || (cx(&b) - cx(im)).abs() > 0.08 * width
                || !(0.5 * em..=4. * em).contains(&(im.n[1] - b.n[3]))
            {
                continue;
            }
            let mut end = b.n[3];
            for &i in &order {
                w.charge(1)?;
                let t = &ls[i];
                let tb = bbox(&t["bbox"])?;
                if integer(t, "angle", 0)? == 0
                    && eq(&t["font_signature"], &h["font_signature"])
                    && (0.0..=0.75 * em).contains(&(tb.n[1] - end))
                    && tb.n[3] < im.n[1]
                    && (cx(&tb) - cx(im)).abs() <= 0.08 * width
                    && (0.85..=1.15).contains(&(height(t, &tb)? / hh))
                {
                    end = tb.n[3];
                }
            }
            floor = floor.max(end);
        }
    }
    Ok(floor)
}
pub(super) fn build(raw: &Value, w: &mut Work) -> Result<Value, Error> {
    build_owned(raw.clone(), w)
}
pub(super) fn build_owned(mut source: Value, w: &mut Work) -> Result<Value, Error> {
    let sz = size(&source["page_size"])?;
    let mut ls = optional(&source, "lines", 4096)?.to_vec();
    let mut caps = BTreeSet::new();
    for l in &ls {
        let t = l["text"].as_str().ok_or_else(invalid)?;
        w.charge(t.len() + 1)?;
        if super::roles::caption(t) {
            caps.insert(integer(l, "source_index", 0)?);
        }
    }
    if !ls.iter().any(|l| {
        caps.contains(&integer(l, "source_index", 0).unwrap())
            && figure(l["text"].as_str().unwrap(), false, true)
    }) {
        return Ok(crate::processing::with_source(
            source,
            json!({"caption_graphics":[],"claimed_line_indices":[]}),
        ));
    }
    let table_caption = ls.iter().any(|l| {
        let c: Vec<_> = l["text"]
            .as_str()
            .unwrap()
            .trim_start_matches(python_space)
            .chars()
            .collect();
        let mut i = if starts(&c, "table") {
            5
        } else if starts(&c, "tab.") {
            4
        } else if starts(&c, "tab") {
            3
        } else if c.first() == Some(&'表') {
            1
        } else {
            return false;
        };
        while c.get(i).is_some_and(|c| python_space(*c)) {
            i += 1
        }
        c.get(i).is_some_and(|c| python_decimal(*c).is_some())
    });
    let found =
        closed_rule_pipeline::detect_complete_source(&source, &json!({"excluded_bboxes":[]}), w)?;
    let flags = closed_grid::inspect_with_work(
        &json!({"source":source,"candidates":found["candidates"]}),
        w,
    )?;
    let mut tables = Vec::new();
    for (c, closed) in array(&found["candidates"], 5000)?
        .iter()
        .zip(array(&flags["closed_grid_flags"], 5000)?)
    {
        if (table_caption && !optional(c, "annotations", 5000)?.is_empty()) || *closed == true {
            tables.push(bbox(
                c.get("core_bbox")
                    .filter(|v| !v.is_null())
                    .unwrap_or(&c["bbox"]),
            )?);
        }
    }
    let filled = code_containers::build_source(
        &source,
        &json!({"kind":"filled","excluded_bboxes":[],"claimed_line_indices":[]}),
        w,
    )?;
    let ruled = code_containers::build_source(
        &source,
        &json!({"kind":"ruled","excluded_bboxes":[],"claimed_line_indices":[]}),
        w,
    )?;
    let mut codes = array(&filled["code_blocks"], 4000)?
        .iter()
        .map(|b| bbox(&b["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    for b in array(&ruled["code_blocks"], 4000)? {
        if numbered_code(b["content"].as_str().ok_or_else(invalid)?) >= 2 {
            codes.push(bbox(&b["bbox"])?);
        }
    }
    let images = optional(&source, "image_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let paths = optional(&source, "path_infos", 5000)?.to_vec();
    let rules = optional(&source, "drawing_lines", 5000)?.to_vec();
    let retained = optional(&source, "retained_page_forms", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let forms = optional(&source, "form_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let maps = super::forms::maps(&source, "form_member_sources", w)?;
    let mut hs = Vec::new();
    for l in &ls {
        if integer(l, "angle", 0)? == 0 {
            hs.push(height(l, &bbox(&l["bbox"])?)?)
        }
    }
    let em = if hs.is_empty() { 10. } else { median(hs) };
    let mut captions = Vec::new();
    for (i, l) in ls.iter().enumerate() {
        if integer(l, "angle", 0)? != 0
            || !caps.contains(&integer(l, "source_index", 0)?)
            || !figure(l["text"].as_str().unwrap(), false, false)
        {
            continue;
        }
        let mut owned = false;
        for f in &retained {
            if super::forms::member(l, f, &maps, w)?.is_some() {
                owned = true;
                break;
            }
        }
        if !owned {
            captions.push(i);
        }
    }
    let raster_prose = prose(
        &mut ls,
        &images
            .iter()
            .filter(|b| b.n[2] - b.n[0] >= 5. * em && b.n[3] - b.n[1] >= 5. * em)
            .cloned()
            .collect::<Vec<_>>(),
        &caps,
        em,
        w,
    )?;
    let mut primitives = images.clone();
    for p in &paths {
        if p["fill_visible"] != true && p["stroke_visible"] != true {
            continue;
        }
        let b = bbox(&p["bbox"])?;
        let color = optional(p, "fill_rgba", 4)?;
        if color.len() >= 3
            && color[..3]
                .iter()
                .all(|c| c.as_i64().is_some_and(|c| c >= 250))
            && p["stroke_visible"] != true
            && area(&b) > 0.02 * sz[0] * sz[1]
        {
            continue;
        }
        primitives.push(b);
    }
    let mut geometry = Vec::new();
    for l in &ls {
        let b = bbox(&l["bbox"])?;
        if !tables
            .iter()
            .chain(&codes)
            .any(|t| overlap_first(&b, t) >= 0.8)
        {
            geometry.push(l.clone());
        }
    }
    let layout = super::roles::layout(geometry, sz, 0, w)?;
    let mut code_members = BTreeSet::new();
    for l in &ls {
        let b = bbox(&l["bbox"])?;
        if codes.iter().any(|c| overlap_first(&b, c) >= 0.8) {
            code_members.insert(integer(l, "source_index", 0)?);
        }
    }
    captions.sort_by(|&i, &j| {
        ls[i]["bbox"][1]
            .as_f64()
            .partial_cmp(&ls[j]["bbox"][1].as_f64())
            .unwrap()
            .then(
                ls[i]["bbox"][0]
                    .as_f64()
                    .partial_cmp(&ls[j]["bbox"][0].as_f64())
                    .unwrap(),
            )
    });
    let caption_set: BTreeSet<_> = captions.iter().copied().collect();
    let mut claimed = BTreeSet::new();
    let mut blocks = Vec::new();
    for &i in &captions {
        let side = side(&ls, &images, &rules, i, w)?;
        if !side.is_empty() {
            let bs = side
                .iter()
                .map(|&i| bbox(&ls[i]["bbox"]))
                .collect::<Result<Vec<_>, _>>()?;
            let hs = side
                .iter()
                .zip(&bs)
                .map(|(&i, b)| height(&ls[i], b))
                .collect::<Result<Vec<_>, _>>()?;
            let mut fonts: Vec<Value> = Vec::new();
            for &j in &side {
                let f = &ls[j]["font_signature"];
                if !f.is_null() && !fonts.iter().any(|a| eq(a, f)) {
                    fonts.push(f.clone());
                }
            }
            fonts.sort_by_key(Value::to_string);
            blocks.push(json!({"type":"caption","bbox":combined(&bs)?.raw,"angle":0,"content":side.iter().map(|&j|ls[j]["text"].as_str().unwrap().trim_matches(python_space)).collect::<Vec<_>>().join(" "),"_line_heights":hs,"_font_signatures":fonts}));
            for j in side {
                claimed.insert(integer(&ls[j], "source_index", 0)?);
            }
            continue;
        }
        let rawcb = bbox(&ls[i]["bbox"])?;
        let mut companions = vec![rawcb.clone()];
        for (j, l) in ls.iter().enumerate() {
            w.charge(1)?;
            if j != i && integer(l, "angle", 0)? == 0 && !caption_set.contains(&j) {
                let b = bbox(&l["bbox"])?;
                if graphic_captions::axis(&rawcb, &b, 1) >= 0.5
                    && (b.n[0] - rawcb.n[2]).abs().min((rawcb.n[0] - b.n[2]).abs()) <= em
                {
                    companions.push(b);
                }
            }
        }
        let cb = combined(&companions)?;
        let Some((mut left, mut right)) = corridor(&cb, &layout.lanes, sz[0]) else {
            continue;
        };
        for &j in &captions {
            if j == i {
                continue;
            }
            let p = bbox(&ls[j]["bbox"])?;
            if (cy(&p) - cy(&cb)).abs() <= 1.5 * em {
                if p.n[2] <= cb.n[0] - 0.5 * em {
                    left = left.max((p.n[2] + cb.n[0]) / 2.);
                } else if p.n[0] >= cb.n[2] + 0.5 * em {
                    right = right.min((cb.n[2] + p.n[0]) / 2.);
                }
            }
        }
        let mut barriers = Vec::new();
        for (j, l) in ls.iter().enumerate() {
            w.charge(paths.len() + forms.len() + 1)?;
            if j == i || integer(l, "angle", 0)? != 0 {
                continue;
            }
            let b = bbox(&l["bbox"])?;
            if b.n[3] > cb.n[1] - 0.3 * em || cx(&b) < left || cx(&b) > right {
                continue;
            }
            let text = l["text"].as_str().unwrap();
            let filled = paths.iter().any(|p| {
                let pb = bbox(&p["bbox"]).unwrap();
                p["fill_visible"] == true
                    && integer(p, "segment_count", 0).unwrap() >= 4
                    && area(&b) < area(&pb)
                    && area(&pb) < 0.4 * sz[0] * sz[1]
                    && overlap_first(&b, &pb) >= 0.95
            });
            if caption_set.contains(&j)
                || ((words(text, 3, false) >= 6
                    || text
                        .chars()
                        .filter(|c| ('\u{3400}'..='\u{9fff}').contains(c))
                        .count()
                        >= 10)
                    && b.n[2] - b.n[0] > 0.25 * (right - left)
                    && !caps.contains(&integer(l, "source_index", 0)?)
                    && !forms.iter().any(|f| overlap_first(&b, f) >= 0.9)
                    && !filled)
            {
                barriers.push(b.n[3]);
            }
        }
        let mut floor = if barriers.is_empty() {
            0.06 * sz[1]
        } else {
            barriers.into_iter().fold(f64::NEG_INFINITY, f64::max)
        };
        floor = floor.max(heading_floor(&ls, &images, &cb, em, w)?);
        for b in tables.iter().chain(&codes) {
            if b.n[3] <= cb.n[1] && b.n[2] > left && b.n[0] < right {
                floor = floor.max(b.n[3]);
            }
        }
        let mut selected = Vec::new();
        for b in &primitives {
            w.charge(tables.len() + codes.len() + 1)?;
            if b.n[1] >= floor - 0.3 * em
                && b.n[3] <= cb.n[1] + 0.2 * em
                && (left..=right).contains(&cx(b))
                && b.n[0] >= left - em
                && b.n[2] <= right + em
                && !tables.iter().any(|t| overlap_first(b, t) >= 0.25)
                && !codes.iter().any(|c| overlap_first(b, c) >= 0.25)
            {
                selected.push(b.clone());
            }
        }
        if selected.is_empty() {
            continue;
        }
        let bottom = selected
            .iter()
            .map(|b| b.n[3])
            .fold(f64::NEG_INFINITY, f64::max);
        if cb.n[1] - bottom > 8. * em {
            continue;
        }
        let mut b = combined(&selected)?;
        let raster = images
            .iter()
            .any(|im| overlap_first(im, &b) >= 0.95 && area(im) >= 0.85 * area(&b));
        if codes.iter().any(|c| overlap_smaller(&b, c) >= 0.8)
            || b.n[2] - b.n[0] < 4. * em
            || b.n[3] - b.n[1] < 3. * em
            || (selected.len() < 4 && !images.iter().any(|im| overlap_first(im, &b) >= 0.95))
        {
            continue;
        }
        let mut members = Vec::new();
        for _ in 0..2 {
            members.clear();
            for (j, l) in ls.iter().enumerate() {
                w.charge(forms.len() + maps.len() + 1)?;
                let id = integer(l, "source_index", 0)?;
                if claimed.contains(&id)
                    || (raster && raster_prose.contains(&id))
                    || code_members.contains(&id)
                    || caption_set.contains(&j)
                    || caps.contains(&id)
                    || (!l["semantic_type"].is_null()
                        && ![90, 270].contains(&integer(l, "angle", 0)?))
                {
                    continue;
                }
                let lb = bbox(&l["bbox"])?;
                if lb.n[1] < floor
                    || lb.n[3] > cb.n[1]
                    || cx(&lb) < left
                    || cx(&lb) > right
                    || super::graphics::distance(&ink(l)?, &b) > 4. * em
                    || !allow(l, &b, &maps, w)?
                {
                    continue;
                }
                let text = l["text"].as_str().unwrap();
                if text.split(python_space).filter(|s| !s.is_empty()).count() >= 8
                    && overlap_first(&lb, &b) < 0.9
                    && !forms
                        .iter()
                        .any(|f| overlap_first(&lb, f) >= 0.95 && overlap_first(f, &b) >= 0.75)
                {
                    continue;
                }
                let terminal = text.trim_end_matches(python_space).chars().last();
                if overlap_first(&lb, &b) < 0.5
                    && (terminal.is_some_and(|c| "。！？".contains(c))
                        || (terminal.is_some_and(|c| ".!?".contains(c))
                            && words(text, 2, false) >= 2))
                {
                    continue;
                }
                members.push(j);
            }
            if !members.is_empty() {
                let mut bs = vec![b];
                for &j in &members {
                    bs.push(ink(&ls[j])?);
                }
                b = combined(&bs)?;
            }
        }
        if blocks.iter().any(|existing| {
            bbox(&existing["bbox"])
                .is_ok_and(|e| e.n.iter().zip(b.n).all(|(x, y)| (x - y).abs() <= 0.5))
        }) {
            continue;
        }
        let text = super::media::content(&members, &mut ls, sz, w)?;
        let values = members.iter().map(|&j| ls[j].clone()).collect::<Vec<_>>();
        let numeric_bounds = values
            .iter()
            .map(|l| bbox(&l["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let numeric_bounds = if numeric_bounds.is_empty() {
            b.clone()
        } else {
            combined(&numeric_bounds)?
        };
        let numeric = crate::legacy_model::table_page::numeric_grid(&values, &numeric_bounds, w)?;
        blocks.push(json!({"type":"image","bbox":b.raw,"angle":0,"content":text,"_caption_graphic":true,"_native_numeric_grid":numeric}));
        for j in members {
            claimed.insert(integer(&ls[j], "source_index", 0)?);
        }
    }
    source["lines"] = Value::Array(ls);
    Ok(crate::processing::with_source(
        source,
        json!({"caption_graphics":blocks,"claimed_line_indices":claimed}),
    ))
}
