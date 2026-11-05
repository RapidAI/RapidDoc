//! 旧整页候选后处理；坐标原始数值类型、稳定顺序和注释字典顺序均保留。
use super::*;
use crate::legacy_model::{
    graphic_captions::{axis, words},
    tiled_tables::pending::Pending,
};
use std::collections::{BTreeMap, BTreeSet};

struct Line {
    raw: Value,
    b: Box4,
    id: i64,
    angle: i64,
    height: f64,
}
struct Rule {
    b: Box4,
    orientation: String,
}
struct Source {
    lines: Vec<Line>,
    rules: Vec<Rule>,
}
fn union(a: &Box4, b: &Box4) -> Box4 {
    super::union(a.clone(), b)
}
fn median(mut a: Vec<f64>) -> f64 {
    a.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = a.len();
    if n % 2 == 1 {
        a[n / 2]
    } else {
        (a[n / 2 - 1] + a[n / 2]) / 2.
    }
}
fn parse(s: &Value, w: &mut Work) -> Result<Source, Error> {
    s.as_object().ok_or_else(invalid)?;
    let mut lines = Vec::new();
    for l in optional(s, "lines", 4096)? {
        let text = l["text"].as_str().ok_or_else(invalid)?;
        if text.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        w.charge(text.len() + 1)?;
        lines.push(Line {
            raw: l.clone(),
            b: bbox(&l["bbox"])?,
            id: integer(l, "source_index", 0)?,
            angle: integer(l, "angle", 0)?,
            height: l.get("effective_height").map_or(Ok(0.), num)?,
        });
    }
    let mut rules = Vec::new();
    for r in optional(s, "drawing_lines", 5000)? {
        w.charge(1)?;
        rules.push(Rule {
            b: bbox(&r["bbox"])?,
            orientation: r["orientation"].as_str().ok_or_else(invalid)?.to_owned(),
        });
    }
    Ok(Source { lines, rules })
}
fn angle(c: &Value) -> Result<i64, Error> {
    integer(c, "angle", 0)
}
fn core(c: &Value, fallback: &str) -> Result<Box4, Error> {
    bbox(
        c.get("core_bbox")
            .filter(|v| !v.is_null())
            .unwrap_or(&c[fallback]),
    )
}
fn annotations(c: &Value) -> Result<&[Value], Error> {
    array(&c["annotations"], 20000)
}
fn ids(v: &Value) -> Result<BTreeSet<i64>, Error> {
    array(v, 20000)?
        .iter()
        .map(|x| x.as_i64().ok_or_else(invalid))
        .collect()
}
fn outer(b: &Box4, anns: &[Value]) -> Result<Box4, Error> {
    let mut out = b.clone();
    for a in anns {
        out = union(&out, &bbox(&a["bbox"])?);
    }
    Ok(out)
}
fn replace_body(
    c: &mut Value,
    b: &Box4,
    members: BTreeSet<i64>,
    anns: Vec<Value>,
    include_annotations: bool,
) -> Result<(), Error> {
    c["bbox"] = json!(if include_annotations {
        outer(b, &anns)?.raw
    } else {
        b.raw.clone()
    });
    c["local_bbox"] = json!(b.raw);
    c["core_bbox"] = json!(b.raw);
    c["line_indices"] = json!(members);
    c["annotations"] = json!(anns);
    Ok(())
}
fn caption(s: &str) -> bool {
    let c: Vec<char> = s.trim_start_matches(python_space).chars().collect();
    let mut i = if prefix(&c, "table") {
        5
    } else if prefix(&c, "tab") {
        if c.get(3) == Some(&'.') {
            4
        } else {
            3
        }
    } else if c.first() == Some(&'表') {
        1
    } else {
        return false;
    };
    while c.get(i).is_some_and(|c| python_space(*c)) {
        i += 1;
    }
    if c.get(i)
        .is_some_and(|c| c.is_ascii_alphabetic() || "İıſK".contains(*c))
    {
        i += 1;
        if c.get(i) == Some(&'.') {
            i += 1;
        }
    }
    let start = i;
    while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
        i += 1;
    }
    i > start && c.get(i).is_some_and(|c| ".:：".contains(*c))
}
fn prose_tokens(s: &str) -> usize {
    let mut n = 0;
    let mut run = 0;
    for c in s.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_alphabetic() {
            run += 1;
        } else {
            if run >= 3 {
                n += 1;
            }
            run = 0;
            if ('\u{3400}'..='\u{9fff}').contains(&c) {
                n += 1;
            }
        }
    }
    n
}
fn sparse(
    s: &Source,
    cs: Vec<Value>,
    caps: &[usize],
    em: f64,
    w: &mut Work,
) -> Result<Vec<Value>, Error> {
    let mut result: Vec<(usize, Value)> = cs.into_iter().enumerate().collect();
    let mut next = result.len();
    for &ci in caps {
        let cap = &s.lines[ci];
        for (_, seed) in result.clone() {
            w.charge(1)?;
            let b = bbox(&seed["bbox"])?;
            if angle(&seed)? != 0 || b.n[3] >= cap.b.n[1] || axis(&b, &cap.b, 0) < 0.8 {
                continue;
            }
            let floor = caps
                .iter()
                .filter(|&&j| {
                    j != ci && s.lines[j].b.n[3] < cap.b.n[1] && axis(&s.lines[j].b, &b, 0) >= 0.8
                })
                .map(|&j| s.lines[j].b.n[3])
                .reduce(f64::max)
                .unwrap_or(0.);
            w.charge(caps.len() + s.rules.len() + 1)?;
            let mut aligned: Vec<_> = s
                .rules
                .iter()
                .filter(|r| {
                    r.orientation == "horizontal"
                        && (r.b.n[0] - b.n[0]).abs() <= em
                        && (r.b.n[2] - b.n[2]).abs() <= em
                        && floor < r.b.n[1]
                        && r.b.n[1] < cap.b.n[1]
                })
                .collect();
            aligned.sort_by(|a, b| a.b.n[1].partial_cmp(&b.b.n[1]).unwrap());
            if aligned.len() < 6 || cap.b.n[1] - aligned.last().unwrap().b.n[3] > 2. * em {
                continue;
            }
            let mut chain = vec![aligned.pop().unwrap()];
            for r in aligned.into_iter().rev() {
                if chain.last().unwrap().b.n[1] - r.b.n[3] > 5. * em {
                    break;
                }
                chain.push(r);
            }
            if chain.len() < 6 {
                continue;
            }
            let mut bounds = chain[0].b.clone();
            for r in &chain[1..] {
                bounds = union(&bounds, &r.b);
            }
            let mut group = Vec::new();
            w.charge(result.len() + s.lines.len() + 1)?;
            for (id, c) in &result {
                let cb = bbox(&c["bbox"])?;
                if angle(c)? == 0
                    && (cb.n[0] - b.n[0]).abs() <= em
                    && (cb.n[2] - b.n[2]).abs() <= em
                    && overlap_first(&cb, &bounds) >= 0.9
                {
                    group.push((*id, c.clone()));
                }
            }
            if group.len() < 2 {
                continue;
            }
            let rows: Vec<_> = s
                .lines
                .iter()
                .filter(|l| l.angle == 0 && inside(center(&l.b, 0), center(&l.b, 1), &bounds))
                .collect();
            if rows.iter().any(|l| {
                !l.raw["semantic_type"].is_null()
                    && !matches!(l.raw["semantic_type"].as_str(), Some("header" | "footer"))
                    || l.b.n[0] < bounds.n[0] - 0.25 * em
                    || l.b.n[2] > bounds.n[2] + 0.25 * em
            }) {
                continue;
            }
            if rows.iter().any(|l| {
                prose_tokens(l.raw["text"].as_str().unwrap()) >= 8
                    && l.b.n[2] - l.b.n[0] >= 0.85 * (b.n[2] - b.n[0])
            }) {
                continue;
            }
            let mut anns = Vec::new();
            for (_, c) in &group {
                anns.extend_from_slice(annotations(c)?);
            }
            let mut merged = group[0].1.clone();
            replace_body(
                &mut merged,
                &bounds,
                rows.iter().map(|l| l.id).collect(),
                anns,
                false,
            )?;
            result.retain(|(id, _)| !group.iter().any(|(g, _)| g == id));
            result.push((next, merged));
            next += 1;
        }
    }
    Ok(result.into_iter().map(|(_, c)| c).collect())
}
fn join(s: &Source, cs: Vec<Value>, w: &mut Work) -> Result<Vec<Value>, Error> {
    let heights: Vec<_> = s
        .lines
        .iter()
        .filter(|l| l.angle == 0 && l.height > 0.)
        .map(|l| l.height)
        .collect();
    let em = if heights.is_empty() {
        10.
    } else {
        median(heights)
    };
    let caps: Vec<_> = s
        .lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.angle == 0 && caption(l.raw["text"].as_str().unwrap()))
        .map(|(i, _)| i)
        .collect();
    let cs = sparse(s, cs, &caps, em, w)?;
    let mut pending = Pending::new(cs.len());
    let mut out = Vec::new();
    while let Some(seed) = pending.pop() {
        let mut group = vec![seed];
        let mut frontier = vec![seed];
        while let Some(i) = frontier.pop() {
            let mut adjacent = Vec::new();
            for j in pending.ordered() {
                w.charge(caps.len() + 1)?;
                if angle(&cs[i])? != 0 || angle(&cs[j])? != 0 {
                    continue;
                }
                let a = core(&cs[i], "local_bbox")?;
                let b = core(&cs[j], "local_bbox")?;
                let gap = 0f64.max(a.n[1] - b.n[3]).max(b.n[1] - a.n[3]);
                let barrier = caps.iter().any(|&k| {
                    a.n[3].min(b.n[3]) <= s.lines[k].b.n[1]
                        && s.lines[k].b.n[1] <= a.n[1].max(b.n[1])
                        && axis(&s.lines[k].b, &a, 0) >= 0.5
                });
                if (a.n[0] - b.n[0]).abs() <= em
                    && (a.n[2] - b.n[2]).abs() <= em
                    && gap <= 1.5 * em
                    && !barrier
                {
                    adjacent.push(j);
                }
            }
            pending.difference(&adjacent);
            frontier.extend(&adjacent);
            group.extend(adjacent);
        }
        let mut b = core(&cs[group[0]], "local_bbox")?;
        for &j in &group[1..] {
            b = union(&b, &core(&cs[j], "local_bbox")?);
        }
        w.charge(caps.len() + 1)?;
        if angle(&cs[group[0]])? != 0
            || !caps.iter().any(|&k| {
                0. <= s.lines[k].b.n[1] - b.n[3]
                    && s.lines[k].b.n[1] - b.n[3] <= 4. * em
                    && axis(&s.lines[k].b, &b, 0) >= 0.8
            })
        {
            out.extend(group.iter().map(|&j| cs[j].clone()));
            continue;
        }
        let mut anns = Vec::new();
        let mut owned = BTreeSet::new();
        for &j in &group {
            for a in annotations(&cs[j])? {
                owned.extend(ids(&a["line_indices"])?);
                anns.push(a.clone());
            }
        }
        for _ in 0..3 {
            w.charge(s.lines.len() + 1)?;
            let header: Vec<_> = s
                .lines
                .iter()
                .enumerate()
                .filter(|(i, l)| {
                    l.angle == 0
                        && !owned.contains(&l.id)
                        && !caps.contains(i)
                        && b.n[0] - em <= l.b.n[0]
                        && l.b.n[2] <= b.n[2] + em
                        && 0. <= b.n[1] - l.b.n[3]
                        && b.n[1] - l.b.n[3] <= 1.5 * em
                })
                .map(|(_, l)| l)
                .collect();
            let Some(bottom) = header.iter().map(|l| l.b.n[3]).reduce(f64::max) else {
                break;
            };
            let row: Vec<_> = header
                .into_iter()
                .filter(|l| (l.b.n[3] - bottom).abs() <= 0.5 * em)
                .collect();
            let separated = row.len() >= 2
                && row.iter().map(|l| l.b.n[0]).reduce(f64::max).unwrap()
                    - row.iter().map(|l| l.b.n[0]).reduce(f64::min).unwrap()
                    > 2. * em;
            let italic = row.len() == 1
                && !row[0].raw["font_signature"].is_null()
                && row[0].raw["font_signature"][1]
                    .as_i64()
                    .ok_or_else(invalid)?
                    & 64
                    != 0;
            if !separated && !italic {
                break;
            }
            let top = row
                .iter()
                .min_by(|a, b| a.b.n[1].partial_cmp(&b.b.n[1]).unwrap())
                .unwrap();
            b.n[1] = top.b.n[1];
            b.raw[1] = top.b.raw[1].clone();
        }
        let matches: Vec<_> = s
            .rules
            .iter()
            .filter(|r| {
                r.orientation == "horizontal"
                    && (r.b.n[0] - b.n[0]).abs() <= em
                    && (r.b.n[2] - b.n[2]).abs() <= em
                    && b.n[1] - em <= r.b.n[1]
                    && r.b.n[1] <= b.n[3] + 0.25 * em
            })
            .collect();
        w.charge(s.rules.len() + s.lines.len() + 1)?;
        for r in matches {
            b = union(&b, &r.b);
        }
        let members = s
            .lines
            .iter()
            .filter(|l| {
                l.angle == 0
                    && b.n[0] - 0.25 * em <= l.b.n[0]
                    && l.b.n[2] <= b.n[2] + 0.25 * em
                    && b.n[1] <= center(&l.b, 1)
                    && center(&l.b, 1) <= b.n[3]
            })
            .map(|l| l.id)
            .collect();
        let mut c = cs[group[0]].clone();
        replace_body(&mut c, &b, members, anns, true)?;
        out.push(c);
    }
    Ok(out)
}
fn numeric(s: &str) -> bool {
    let s = s.trim_matches(python_space);
    !s.is_empty()
        && s.chars()
            .all(|c| python_decimal(c).is_some() || python_space(c) || "+-−$€£¥.,()%‰".contains(c))
}
fn trim(s: &Source, c: &mut Value, w: &mut Work) -> Result<(), Error> {
    if angle(c)? != 0 {
        return Ok(());
    }
    let b = core(c, "local_bbox")?;
    w.charge(s.lines.len() + 1)?;
    let members: Vec<_> = s
        .lines
        .iter()
        .filter(|l| {
            l.angle == 0
                && b.n[0] - 2. <= l.b.n[0]
                && l.b.n[2] <= b.n[2] + 2.
                && b.n[1] <= center(&l.b, 1)
                && center(&l.b, 1) <= b.n[3]
        })
        .collect();
    let hs: Vec<_> = members
        .iter()
        .filter(|l| l.height > 0.)
        .map(|l| l.height)
        .collect();
    if hs.is_empty() {
        return Ok(());
    }
    let em = median(hs);
    let width = b.n[2] - b.n[0];
    if width < 8. * em {
        return Ok(());
    }
    let mut sorted = members.clone();
    sorted.sort_by(|a, b| {
        center(&a.b, 1)
            .partial_cmp(&center(&b.b, 1))
            .unwrap()
            .then(a.b.n[0].partial_cmp(&b.b.n[0]).unwrap())
    });
    let mut rows: Vec<Vec<&Line>> = Vec::new();
    for l in sorted {
        w.charge(rows.last().map_or(1, |r| r.len() + 1))?;
        if rows.last().is_some_and(|r| {
            (center(&l.b, 1) - median(r.iter().map(|l| center(&l.b, 1)).collect())).abs()
                <= 0.45 * em
        }) {
            rows.last_mut().unwrap().push(l);
        } else {
            rows.push(vec![l]);
        }
    }
    let data: Vec<_> = rows
        .iter()
        .filter(|r| {
            r.len() >= 2
                && r.iter().map(|l| l.b.n[0]).reduce(f64::max).unwrap()
                    - r.iter().map(|l| l.b.n[0]).reduce(f64::min).unwrap()
                    >= 0.3 * width
                && r.iter()
                    .filter(|l| numeric(l.raw["text"].as_str().unwrap()))
                    .count()
                    >= 2
        })
        .collect();
    if data.len() < 3 {
        return Ok(());
    }
    let last = data
        .iter()
        .flat_map(|r| r.iter())
        .map(|l| l.b.n[3])
        .reduce(f64::max)
        .unwrap();
    let data_em = median(
        data.iter()
            .flat_map(|r| r.iter())
            .map(|l| l.height)
            .collect(),
    );
    let prose: Vec<_> = members
        .iter()
        .filter(|l| {
            l.b.n[1] - last >= 0.65 * em
                && l.b.n[2] - l.b.n[0] >= 0.8 * width
                && (l.b.n[0] - b.n[0]).abs() <= 0.5 * em
                && words(l.raw["text"].as_str().unwrap()) >= 6
                && l.height >= 1.08 * data_em
        })
        .collect();
    if prose.len() < 2 {
        return Ok(());
    }
    let first = prose
        .iter()
        .min_by(|a, b| a.b.n[1].partial_cmp(&b.b.n[1]).unwrap())
        .unwrap();
    if crate::legacy_model::table_block::note(first.raw["text"].as_str().unwrap()) {
        return Ok(());
    }
    let mut tail: Vec<_> = members
        .iter()
        .filter(|l| l.b.n[1] >= first.b.n[1])
        .collect();
    tail.sort_by(|a, b| a.b.n[1].partial_cmp(&b.b.n[1]).unwrap());
    if tail.len() < 3
        || !tail
            .windows(2)
            .any(|r| r[1].b.n[1] - r[0].b.n[3] >= 0.6 * em)
    {
        return Ok(());
    }
    let bottom = tail.iter().map(|l| l.b.n[3]).reduce(f64::max).unwrap();
    w.charge(s.rules.len() + 1)?;
    if s.rules.iter().any(|r| {
        r.orientation == "vertical"
            && b.n[0] - em <= center(&r.b, 0)
            && center(&r.b, 0) <= b.n[2] + em
            && r.b.n[1] <= first.b.n[3]
            && r.b.n[3] >= bottom - em
            || r.orientation == "horizontal"
                && first.b.n[1] <= r.b.n[1]
                && r.b.n[1] <= bottom
                && axis(&r.b, &b, 0) >= 0.8
    }) {
        return Ok(());
    }
    let trimmed = bbox(&json!([b.raw[0], b.raw[1], b.raw[2], last + 0.25 * em]))?;
    let mut members = ids(&c["line_indices"])?;
    for l in tail {
        members.remove(&l.id);
    }
    replace_body(c, &trimmed, members, annotations(c)?.to_vec(), true)
}
fn continuation(s: &str) -> bool {
    let c: Vec<_> = s.trim_matches(python_space).chars().collect();
    if c.first() == Some(&'续') {
        let mut i = 1;
        while c.get(i).is_some_and(|c| python_space(*c)) {
            i += 1;
        }
        if c.get(i) != Some(&'表') {
            return false;
        }
        i += 1;
        if i == c.len() {
            return true;
        }
        while c.get(i).is_some_and(|c| python_space(*c)) {
            i += 1;
        }
        let start = i;
        while c.get(i).is_some_and(|c| {
            c.is_ascii_digit() || ('０'..='９').contains(c) || roman(*c) || chinese(*c)
        }) {
            i += 1;
        }
        return i > start && i == c.len();
    }
    let mut i = if prefix(&c, "table") {
        5
    } else if prefix(&c, "tab") {
        if c.get(3) == Some(&'.') {
            4
        } else {
            3
        }
    } else {
        return false;
    };
    while c.get(i).is_some_and(|c| python_space(*c)) {
        i += 1;
    }
    let start = i;
    while c.get(i).is_some_and(|c| c.is_ascii_digit() || roman(*c)) {
        i += 1;
    }
    // The optional number must backtrack when its initial 'c' belongs to 'continued'.
    for at in (start..=i).rev() {
        let mut j = at;
        if at != start {
            while c.get(j).is_some_and(|c| python_space(*c)) {
                j += 1;
            }
        }
        let tail: String = c[j..]
            .iter()
            .map(|c| {
                if matches!(c, 'İ' | 'ı') {
                    'i'
                } else {
                    c.to_ascii_lowercase()
                }
            })
            .collect();
        if matches!(tail.as_str(), "continued" | "cont" | "cont.") {
            return true;
        }
    }
    false
}
fn caption_annotation(id: i64, b: &Box4) -> Value {
    json!({"kind":"caption","bbox":b.raw,"line_indices":[id],"line_bboxes":[[id,b.raw]]})
}
fn externalize(raw: &Value, s: &Source, cs: &mut [Value], w: &mut Work) -> Result<(), Error> {
    let mut prepared: BTreeMap<i64, Page> = BTreeMap::new();
    for c in cs {
        let a = angle(c)?;
        if !s.lines.iter().any(|l| l.angle == a) {
            continue;
        }
        if !prepared.contains_key(&a) {
            prepared.insert(a, prepare_source(raw, &value_object!({"angle":a}), w)?);
        }
        let p = &prepared[&a];
        if p.lines.is_empty() {
            continue;
        }
        let cb = bbox(&c["local_bbox"])?;
        let mut best: Option<Box4> = None;
        for grid in component_boxes(p, w)? {
            w.charge(1)?;
            if axis(&grid, &cb, 0) >= 0.9
                && overlap_smaller(&grid, &cb) >= 0.9
                && best.as_ref().is_none_or(|b| {
                    crate::legacy_model::graphic_arbitration::area_key(&grid)
                        > crate::legacy_model::graphic_arbitration::area_key(b)
                })
            {
                best = Some(grid);
            }
        }
        let Some(grid) = best else {
            continue;
        };
        let mut matched = Vec::new();
        for l in &s.lines {
            w.charge(1)?;
            if l.angle != a || !continuation(l.raw["text"].as_str().unwrap()) {
                continue;
            }
            let b = local(&l.b, &p.size, a)?;
            let gap = grid.n[1] - b.n[3];
            if b.n[3] <= grid.n[1] + 0.25 * p.em
                && -0.25 * p.em <= gap
                && gap <= 3. * p.em
                && axis(&b, &grid, 0) >= 0.8
            {
                matched.push((l, b));
            }
        }
        if matched.len() != 1 {
            continue;
        }
        let (line, lb) = &matched[0];
        let mut anns = annotations(c)?.to_vec();
        if let Some(an) = anns.iter_mut().find(|a| a["kind"] == "caption") {
            an["bbox"] = json!(union(&bbox(&an["bbox"])?, &line.b).raw);
            let mut members = ids(&an["line_indices"])?;
            members.insert(line.id);
            an["line_indices"] = json!(members);
            let mut pairs = array(&an["line_bboxes"], 20000)?.to_vec();
            if let Some(pair) = pairs.iter_mut().find(|v| v[0].as_i64() == Some(line.id)) {
                pair[1] = json!(line.b.raw);
            } else {
                pairs.push(json!([line.id, line.b.raw]));
            }
            an["line_bboxes"] = json!(pairs);
        } else {
            anns.push(caption_annotation(line.id, &line.b));
        }
        let mut members = ids(&c["line_indices"])?;
        members.remove(&line.id);
        c["line_indices"] = json!(members);
        c["annotations"] = json!(anns);
        let body = bbox(&crate::legacy_model::rule_candidate_expansion::rotate(
            &grid.raw, &p.size, a,
        ))?;
        c["core_bbox"] = json!(body.raw);
        c["local_bbox"] = json!(union(&grid, lb).raw);
        c["bbox"] = json!(union(&body, &line.b).raw);
    }
    Ok(())
}
fn cjk(s: &str) -> bool {
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
        && chars.next().is_none()
}
fn bind(raw: &Value, s: &Source, cs: &mut [Value], w: &mut Work) -> Result<(), Error> {
    let chars = optional(raw, "chars", 20000)?;
    let mut widths = Vec::new();
    let mut loose = Vec::new();
    let mut ink = Vec::new();
    let mut all = Vec::new();
    for ch in chars {
        w.charge(1)?;
        if !cjk(ch
            .get("char")
            .map_or(Ok(""), |v| v.as_str().ok_or_else(invalid))?)
        {
            continue;
        }
        let b = bbox(&ch["bbox"])?;
        all.push(b.n[2] - b.n[0]);
        if let Some(t) = ch
            .get("tight_bbox")
            .filter(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
        {
            let t = bbox(t)?;
            widths.push(b.n[2] - b.n[0]);
            loose.push(b.n[3] - b.n[1]);
            ink.push(t.n[3] - t.n[1]);
        }
    }
    if widths.len() < 20 {
        return Ok(());
    }
    let width = median(widths);
    let height = median(loose);
    let ink = median(ink);
    if !(width > 0. && height >= 2.2 * width && 0.5 * width <= ink && ink <= 1.4 * width) {
        return Ok(());
    }
    let em = median(all);
    let mut boxes = BTreeMap::new();
    for l in &s.lines {
        let mut b: Option<Box4> = None;
        for ch in optional(&l.raw, "chars", 20000)? {
            w.charge(1)?;
            if let Some(t) = ch
                .get("tight_bbox")
                .filter(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
            {
                let t = bbox(t)?;
                b = Some(b.map_or(t.clone(), |b| union(&b, &t)));
            }
        }
        boxes.insert(l.id, b.ok_or_else(invalid)?);
    }
    let mut owned = BTreeSet::new();
    for c in cs.iter() {
        for a in annotations(c)? {
            owned.extend(ids(&a["line_indices"])?);
        }
    }
    for c in cs {
        if angle(c)? != 0 {
            continue;
        }
        let b = core(c, "bbox")?;
        for (i, l) in s.lines.iter().enumerate() {
            w.charge(1)?;
            let lb = &boxes[&l.id];
            let text = l.raw["text"].as_str().unwrap().trim_matches(python_space);
            let count = text.chars().count();
            if owned.contains(&l.id)
                || l.angle != 0
                || !(8..=35).contains(&count)
                || !text.chars().all(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
                || !(0. < b.n[1] - lb.n[3] && b.n[1] - lb.n[3] <= 1.25 * em)
                || (lb.n[0] - b.n[0]).abs() > em
                || lb.n[2] - lb.n[0] >= 0.6 * (b.n[2] - b.n[0])
            {
                continue;
            }
            w.charge(s.lines.len() + 1)?;
            if s.lines.iter().enumerate().any(|(j, other)| {
                i != j
                    && (center(&boxes[&other.id], 1) - center(lb, 1)).abs() <= 0.5 * em
                    && b.n[0] <= center(&boxes[&other.id], 0)
                    && center(&boxes[&other.id], 0) <= b.n[2]
            }) {
                continue;
            }
            let mut anns = annotations(c)?.to_vec();
            anns.push(caption_annotation(l.id, lb));
            c["annotations"] = json!(anns);
            let mut members = ids(&c["line_indices"])?;
            members.insert(l.id);
            c["line_indices"] = json!(members);
            owned.insert(l.id);
        }
    }
    Ok(())
}
fn bounded(cs: &[Value], w: &mut Work) -> Result<(), Error> {
    if cs.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let (mut members, mut pairs, mut grids) = (0usize, 0usize, 0usize);
    for c in cs {
        members += array(&c["line_indices"], 20000)?.len();
        grids += optional(c, "inferred_grid", 20000)?.len();
        for a in annotations(c)? {
            members += array(&a["line_indices"], 20000)?.len();
            pairs += array(&a["line_bboxes"], 20000)?.len();
        }
        if members > rapiddoc_core::resources::limit(20000)
            || pairs > rapiddoc_core::resources::limit(20000)
            || grids > rapiddoc_core::resources::limit(20000)
        {
            return Err(limit());
        }
    }
    w.charge(members + pairs + grids + cs.len() * 16 + 1)
}
pub(super) fn finish(
    raw: &Value,
    mut cs: Vec<Value>,
    operation: &str,
    w: &mut Work,
) -> Result<Value, Error> {
    bounded(&cs, w)?;
    let s = parse(raw, w)?;
    if matches!(operation, "all" | "join") {
        cs = join(&s, cs, w)?;
    }
    if matches!(operation, "all" | "trim") {
        for c in &mut cs {
            trim(&s, c, w)?;
        }
    }
    if matches!(operation, "all" | "forms") {
        let forms = optional(raw, "form_bboxes", 5000)?
            .iter()
            .map(bbox)
            .collect::<Result<Vec<_>, _>>()?;
        let mut kept = Vec::new();
        for c in cs {
            w.charge(forms.len() + 1)?;
            let b = core(&c, "bbox")?;
            if !forms.iter().any(|f| overlap_first(&b, f) >= 0.9) {
                kept.push(c);
            }
        }
        cs = kept;
    }
    if matches!(operation, "all" | "externalize") {
        externalize(raw, &s, &mut cs, w)?;
    }
    if matches!(operation, "all" | "bind") {
        bind(raw, &s, &mut cs, w)?;
    }
    if operation == "all" {
        cs.sort_by(|a, b| {
            num(&a["bbox"][1])
                .unwrap()
                .partial_cmp(&num(&b["bbox"][1]).unwrap())
                .unwrap()
                .then(
                    num(&a["bbox"][0])
                        .unwrap()
                        .partial_cmp(&num(&b["bbox"][0]).unwrap())
                        .unwrap(),
                )
        });
    }
    bounded(&cs, w)?;
    let result = json!({"candidates":cs});
    if crate::legacy_page::budget::output_exceeds_limit(
        &result,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(result)
}
pub(super) fn apply(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let op = v.get("operation").and_then(Value::as_str).unwrap_or("all");
    if !matches!(
        op,
        "all" | "join" | "trim" | "forms" | "externalize" | "bind"
    ) {
        return Err(invalid());
    }
    let cs = array(&v["candidates"], 5000)?.to_vec();
    for c in &cs {
        bbox(&c["bbox"])?;
        bbox(&c["local_bbox"])?;
        core(c, "bbox")?;
        ids(&c["line_indices"])?;
        for a in annotations(c)? {
            bbox(&a["bbox"])?;
            ids(&a["line_indices"])?;
            for pair in array(&a["line_bboxes"], 20000)? {
                if array(pair, 2)?.len() != 2 || pair[0].as_i64().is_none() {
                    return Err(invalid());
                }
                bbox(&pair[1])?;
            }
        }
    }
    finish(&v["source"], cs, op, &mut Work(0))
}
pub(super) fn trailing(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    trailing_with_work(v, &mut Work(0))
}
pub(super) fn trailing_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    trailing_sources(&v["source"], &v["prepared"], w)
}
pub(super) fn trailing_sources(
    raw: &Value,
    prepared: &Value,
    w: &mut Work,
) -> Result<Value, Error> {
    let blocks = optional(prepared, "fixed_blocks", 5000)?;
    let mut chosen: Option<(usize, Box4)> = None;
    for (i, c) in blocks.iter().enumerate() {
        w.charge(1)?;
        if c["type"] == "table" && angle(c)? == 0 {
            let b = bbox(&c["bbox"])?;
            if chosen.as_ref().is_none_or(|(_, old)| b.n[3] > old.n[3]) {
                chosen = Some((i, b));
            }
        }
    }
    if raw["page_index"].is_null() || chosen.is_none() {
        return Ok(Value::Null);
    }
    let (index, b) = chosen.unwrap();
    let s = parse(raw, w)?;
    let heights: Vec<_> = s
        .lines
        .iter()
        .filter(|l| l.height > 0.)
        .map(|l| l.height)
        .collect();
    let em = if heights.is_empty() {
        1.
    } else {
        median(heights)
    };
    let tol = 2f64.max(0.25 * em);
    let marginal =
        |kind: &Value| matches!(kind.as_str(), Some("header" | "footer" | "page_number"));
    for (i, c) in blocks.iter().enumerate() {
        w.charge(1)?;
        if i != index && !marginal(&c["type"]) && bbox(&c["bbox"])?.n[3] > b.n[3] + tol {
            return Ok(Value::Null);
        }
    }
    for l in optional(prepared, "remaining_lines", 4096)? {
        w.charge(1)?;
        if !marginal(&l["semantic_type"]) && bbox(&l["bbox"])?.n[3] > b.n[3] + tol {
            return Ok(Value::Null);
        }
    }
    let mut p = prepare_source(raw, &value_object!({"angle":0,"median_height":em}), w)?;
    p.p.em = em;
    let mut horizontal: Vec<_> =
        p.p.rules
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                r.h && (r.b.n[0] - b.n[0]).abs() <= tol
                    && (r.b.n[2] - b.n[2]).abs() <= tol
                    && b.n[1] - tol <= center(&r.b, 1)
                    && center(&r.b, 1) <= b.n[3] + tol
            })
            .map(|(i, _)| i)
            .collect();
    w.charge(p.p.rules.len() + horizontal.len() * 16 + 1)?;
    horizontal.sort_by(|&a, &b| {
        center(&p.p.rules[a].b, 1)
            .partial_cmp(&center(&p.p.rules[b].b, 1))
            .unwrap()
    });
    if horizontal.len() < 2 {
        return Ok(Value::Null);
    }
    let bottom = center(&p.p.rules[*horizontal.last().unwrap()].b, 1);
    if (center(&p.p.rules[horizontal[0]].b, 1) - b.n[1])
        .abs()
        .max((bottom - b.n[3]).abs())
        > tol
    {
        return Ok(Value::Null);
    }
    let band = &horizontal[horizontal.len() - 2..];
    let top = center(&p.p.rules[band[0]].b, 1);
    if bottom - top < em.max(2. * tol) {
        return Ok(Value::Null);
    }
    let mut positions = Vec::new();
    for x in graphic_grid::tracks(&p.p, band, w)? {
        w.charge(p.p.rules.len() + 1)?;
        if p.p.rules.iter().any(|r| {
            !r.h && (center(&r.b, 0) - x).abs() <= 1f64.max(0.1 * em)
                && r.b.n[1] <= bottom + tol
                && r.b.n[3] >= bottom - tol
        }) {
            positions.push(x);
        }
    }
    if positions.len() < 3
        || (positions[0] - b.n[0])
            .abs()
            .max((positions.last().unwrap() - b.n[2]).abs())
            > tol
    {
        return Ok(Value::Null);
    }
    let width = num(&raw["page_size"][0])?;
    Ok(
        json!({"page_index":raw["page_index"],"angle":0,"positions":positions.iter().map(|x|x/width).collect::<Vec<_>>(),"tolerance":tol/width}),
    )
}
