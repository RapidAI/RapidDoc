//! 开边竖线表和表题短表的旧完整产生器；共用有界表题及同尺度续行收集。
mod aligned;
mod forms;
mod mixed;
mod numeric_segments;
mod shaded;
mod text_numeric;
use super::*;

/// All supplemental kinds inspect the same immutable page. Keep one request and
/// reuse its angle-zero preparation; candidate rules and exclusions still run
/// separately, in the original order. Finite work budgets retain exact charges.
// Borrow the immutable page independently of the short-lived per-kind options.
// Cached preparation can therefore borrow its lines without retaining a second
// page tree. The request is never serialized or returned to callers.
struct Request<'a, 'b> {
    source: &'a Value,
    options: &'b Value,
}
pub(super) struct Detector<'a> {
    source: &'a Value,
    options: Value,
    page: Option<Option<Page<'a>>>,
}
impl<'a> Detector<'a> {
    pub(super) fn new(source: &'a Value) -> Self {
        Self {
            source,
            options: json!({}),
            page: None,
        }
    }
    pub(super) fn detect(
        &mut self,
        kind: &str,
        excluded: &impl serde::Serialize,
        work: &mut Work,
    ) -> Result<Value, Error> {
        let mut profile = crate::processing::NativeStageProfile::new();
        self.options["kind"] = json!(kind);
        self.options["excluded_bboxes"] = json!(excluded);
        crate::legacy_page::validate_source_request(self.source, &self.options)?;
        let result = detect_cached(
            &Request {
                source: self.source,
                options: &self.options,
            },
            work,
            &mut self.page,
        );
        profile.mark(&format!("supplemental_{kind}"));
        result
    }
}
fn cmp(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap()
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| cmp(*a, *b));
    if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.
    }
}
fn flag(v: &Value, k: &str) -> Result<bool, Error> {
    v.get(k)
        .filter(|v| !v.is_null())
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))
}
fn effective(v: &Value) -> Result<f64, Error> {
    v.get("effective_height").map_or(Ok(0.), num)
}
fn words(v: &Value) -> usize {
    v["text"]
        .as_str()
        .unwrap()
        .split(python_space)
        .filter(|s| !s.is_empty())
        .count()
}
fn source<'a>(source: &'a Value, work: &mut Work) -> Result<Option<Page<'a>>, Error> {
    let mut heights = Vec::new();
    for l in optional(source, "lines", 4096)? {
        work.charge(1)?;
        if l.get("angle").map_or(Ok(0.), num)? == 0. && effective(l)? > 0. {
            heights.push(effective(l)?);
        }
    }
    if heights.is_empty() {
        return Ok(None);
    }
    let mut p = prepare_source(
        source,
        &value_object!({"angle":0,"median_height":median(heights)}),
        work,
    )?;
    let mut retained = Vec::new();
    for l in p.lines {
        if effective(&l)? > 0. {
            retained.push(l);
        }
    }
    p.lines = retained;
    Ok(Some(p))
}
fn bounded_marker(s: &str) -> bool {
    let c: Vec<_> = s.trim_start_matches(python_space).chars().collect();
    let mut i = if prefix(&c, "table") {
        5
    } else if prefix(&c, "tab") {
        if c.get(3) == Some(&'.') {
            4
        } else {
            3
        }
    } else if c.first() == Some(&'表') {
        if c.get(1) == Some(&'格') {
            2
        } else {
            1
        }
    } else {
        return false;
    };
    while c.get(i).is_some_and(|c| python_space(*c)) {
        i += 1;
    }
    if c.get(i).is_some_and(|c| ":：".contains(*c)) {
        return true;
    }
    if c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
        while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
            i += 1;
        }
        while c.get(i) == Some(&'.') && c.get(i + 1).is_some_and(|c| python_decimal(*c).is_some()) {
            i += 1;
            while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
                i += 1;
            }
        }
    } else {
        let start = i;
        while c
            .get(i)
            .is_some_and(|c| matches!(c, 'i' | 'I' | 'İ' | 'ı' | 'v' | 'V' | 'x' | 'X'))
        {
            i += 1;
        }
        if i == start {
            return false;
        }
    }
    c.get(i)
        .is_some_and(|c| python_space(*c) || ".:：".contains(*c))
}
fn bold(l: &Value) -> Result<bool, Error> {
    let sig = l.get("font_signature").filter(|v| !v.is_null());
    if let Some(sig) = sig {
        let sig = array(sig, 2)?;
        if sig.len() != 2 {
            return Err(invalid());
        }
        if sig[0].as_str().is_none()
            || sig[0].as_str().unwrap().len() > rapiddoc_core::resources::limit(4096)
            || (sig[1].as_i64().is_none() && sig[1].as_bool().is_none())
        {
            return Err(invalid());
        }
        Ok(super::super::table_text_styles::font_styles(
            &json!({"font":{"name":sig[0],"flags":sig[1],"weight":l["dominant_font_weight"]}}),
        )
        .0)
    } else {
        Ok(l.get("dominant_font_weight")
            .filter(|v| !v.is_null())
            .map_or(Ok(false), |v| num(v).map(|n| n >= 600.))?)
    }
}
fn numeric_title(s: &str) -> bool {
    let s = s.trim_matches(python_space);
    !s.is_empty()
        && s.chars()
            .all(|c| python_decimal(c).is_some() || ".,:() -".contains(c))
}
fn bounded(v: &Request, p: &Page, b: &Box4, work: &mut Work) -> Result<Vec<Value>, Error> {
    bounded_gap(v, p, b, 2., work)
}
fn bounded_gap(
    v: &Request,
    p: &Page,
    b: &Box4,
    below_gap: f64,
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    bounded_em(v, b, p.em, below_gap, work)
}
fn bounded_em(
    v: &Request,
    b: &Box4,
    em: f64,
    below_gap: f64,
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    // Caption fallback traverses ALL angle-zero lines, including zero-height lines.
    let lines = optional(v.source, "lines", 4096)?;
    for l in lines {
        bbox(&l["bbox"])?;
    }
    let mut candidates = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        work.charge(1)?;
        if l.get("angle").map_or(Ok(0.), num)? != 0. {
            continue;
        }
        let t = l["text"].as_str().ok_or_else(invalid)?;
        let box_ = bbox(&l["bbox"])?;
        if bounded_marker(t)
            && super::super::graphic_captions::axis(&box_, b, 0) >= 0.5
            && ((0. <= b.n[1] - box_.n[3] && b.n[1] - box_.n[3] <= 4. * em)
                || (0. <= box_.n[1] - b.n[3] && box_.n[1] - b.n[3] <= below_gap * em))
        {
            candidates.push(i);
        }
    }
    if candidates.is_empty() {
        for (i, l) in lines.iter().enumerate() {
            work.charge(1)?;
            if l.get("angle").map_or(Ok(0.), num)? != 0. {
                continue;
            }
            let box_ = bbox(&l["bbox"])?;
            let w = b.n[2] - b.n[0];
            if bold(l)?
                && 0. <= b.n[1] - box_.n[3]
                && b.n[1] - box_.n[3] <= 0.75 * em
                && 0.1 * w <= box_.n[2] - box_.n[0]
                && box_.n[2] - box_.n[0] <= 0.7 * w
                && (center(&box_, 0) - center(b, 0)).abs() <= 0.03 * w
                && !numeric_title(l["text"].as_str().ok_or_else(invalid)?)
            {
                candidates.push(i);
            }
        }
    }
    let Some(&seed) = candidates.iter().min_by(|a, c| {
        let a = bbox(&lines[**a]["bbox"]).unwrap();
        let c = bbox(&lines[**c]["bbox"]).unwrap();
        cmp(
            (a.n[3] - b.n[1]).abs().min((a.n[1] - b.n[3]).abs()),
            (c.n[3] - b.n[1]).abs().min((c.n[1] - b.n[3]).abs()),
        )
    }) else {
        return Ok(vec![]);
    };
    let sb = bbox(&lines[seed]["bbox"])?;
    let below = sb.n[1] >= b.n[3];
    let mut members = vec![seed];
    let mut ordered: Vec<_> = (0..lines.len()).collect();
    ordered.sort_by(|a, c| {
        let a = &lines[*a]["bbox"];
        let c = &lines[*c]["bbox"];
        cmp(a[1].as_f64().unwrap(), c[1].as_f64().unwrap())
            .then(cmp(a[0].as_f64().unwrap(), c[0].as_f64().unwrap()))
    });
    for i in ordered {
        work.charge(members.len() + 1)?;
        let l = &lines[i];
        if i == seed || l.get("angle").map_or(Ok(0.), num)? != 0. {
            continue;
        }
        let lb = bbox(&l["bbox"])?;
        let right = members
            .iter()
            .map(|i| lines[*i]["bbox"][2].as_f64().unwrap())
            .reduce(f64::max)
            .unwrap();
        if (center(&lb, 1) - center(&sb, 1)).abs() <= 0.3 * em
            && 0. <= lb.n[0] - right
            && lb.n[0] - right <= 4. * em
            && (effective(l)? - effective(&lines[seed])?).abs() <= 0.2 * em
            && lb.n[2] <= b.n[2] + em
        {
            members.push(i);
            continue;
        }
        if lb.n[1] < sb.n[1] + 0.5 * em {
            continue;
        }
        if !(if below {
            lb.n[1] >= b.n[3]
        } else {
            lb.n[3] <= b.n[1]
        }) {
            continue;
        }
        let bottom = members
            .iter()
            .map(|i| lines[*i]["bbox"][3].as_f64().unwrap())
            .reduce(f64::max)
            .unwrap();
        if lb.n[1] - bottom > 0.8 * em
            || (effective(l)? - effective(&lines[seed])?).abs() > 0.2 * em
            || super::super::graphic_captions::axis(&lb, &sb, 0) < 0.5
        {
            break;
        }
        members.push(i);
    }
    let mut boxes: Vec<(i64, Box4)> = Vec::new();
    for i in members {
        let id = integer(&lines[i], "source_index", 0)?;
        let b = bbox(&lines[i]["bbox"])?;
        if let Some((_, old)) = boxes.iter_mut().find(|(n, _)| *n == id) {
            *old = b;
        } else {
            boxes.push((id, b));
        }
    }
    let mut bound = boxes[0].1.clone();
    for (_, b) in &boxes[1..] {
        bound = union(bound, b);
    }
    let ids: BTreeSet<_> = boxes.iter().map(|(id, _)| *id).collect();
    Ok(vec![
        json!({"kind":"caption","bbox":bound.raw,"line_indices":ids,"line_bboxes":boxes.into_iter().map(|(i,b)|json!([i,b.raw])).collect::<Vec<_>>()}),
    ])
}
fn rows<'a>(members: &[&'a Value], em: f64, work: &mut Work) -> Result<Vec<Vec<&'a Value>>, Error> {
    let mut ordered = members.to_vec();
    ordered.sort_by(|a, b| {
        let a = bbox(&a["bbox"]).unwrap();
        let b = bbox(&b["bbox"]).unwrap();
        cmp(center(&a, 1), center(&b, 1)).then(cmp(a.n[0], b.n[0]))
    });
    let mut rows: Vec<Vec<&Value>> = Vec::new();
    for l in ordered {
        work.charge(rows.last().map_or(0, Vec::len) + 1)?;
        let y = center(&bbox(&l["bbox"])?, 1);
        if rows.last().is_some_and(|r| {
            (y - median(
                r.iter()
                    .map(|l| center(&bbox(&l["bbox"]).unwrap(), 1))
                    .collect(),
            ))
            .abs()
                <= 0.5 * em
        }) {
            rows.last_mut().unwrap().push(l);
        } else {
            rows.push(vec![l]);
        }
    }
    Ok(rows)
}
fn axis(b: &Box4, x: f64, horizontal: bool) -> Value {
    if horizontal {
        json!({"bbox":[b.raw[0],x-0.05,b.raw[2],x+0.05],"width":0.1,"orientation":"horizontal"})
    } else {
        json!({"bbox":[x-0.05,b.raw[1],x+0.05,b.raw[3]],"width":0.1,"orientation":"vertical"})
    }
}
fn candidate(
    b: Box4,
    members: &[&Value],
    annotations: Vec<Value>,
    grid: Vec<Value>,
    score: i64,
) -> Result<Value, Error> {
    let mut outer = b.clone();
    for a in &annotations {
        outer = union(outer, &bbox(&a["bbox"])?);
    }
    let ids = members
        .iter()
        .map(|l| integer(l, "source_index", 0))
        .collect::<Result<BTreeSet<_>, _>>()?;
    Ok(
        json!({"bbox":outer.raw,"local_bbox":b.raw,"core_bbox":b.raw,"angle":0,"score":score,"line_indices":ids,"annotations":annotations,"inferred_grid":grid,"preserve_inline_font_styles":false,"inferred_grid_authoritative":false}),
    )
}
fn short(v: &Request, p: &Page, excluded: &[Box4], work: &mut Work) -> Result<Vec<Value>, Error> {
    let cs = captions(p, work)?;
    let caps = p
        .line_values()
        .filter(|l| {
            cs.iter()
                .any(|(id, _)| Some(*id) == l["source_index"].as_i64())
        })
        .collect::<Vec<_>>();
    let mut raw_rules = Vec::new();
    for r in optional(v.source, "drawing_lines", 5000)? {
        if r["orientation"] == "horizontal" {
            let b = bbox(&r["bbox"])?;
            if b.n[2] - b.n[0] >= 6. * p.em {
                raw_rules.push(b);
            }
        }
    }
    let rules = raw_rules.iter().collect::<Vec<_>>();
    let mut out = Vec::new();
    for caption in &caps {
        let cb = bbox(&caption["bbox"])?;
        work.charge(rules.len() + caps.len() + 1)?;
        let ends = rules
            .iter()
            .copied()
            .filter(|b| {
                0. <= cb.n[1] - b.n[3]
                    && cb.n[1] - b.n[3] <= 2. * p.em
                    && super::super::graphic_captions::axis(b, &cb, 0) >= 0.8
            })
            .collect::<Vec<_>>();
        let Some(end) = ends.into_iter().max_by(|a, b| cmp(a.n[3], b.n[3])) else {
            continue;
        };
        // Python max retains the first equal item; stable endpoint choice is handled below.
        let end = rules
            .iter()
            .copied()
            .find(|b| {
                b.n[3] == end.n[3]
                    && 0. <= cb.n[1] - b.n[3]
                    && cb.n[1] - b.n[3] <= 2. * p.em
                    && super::super::graphic_captions::axis(b, &cb, 0) >= 0.8
            })
            .unwrap();
        let floor = caps
            .iter()
            .filter_map(|l| {
                let b = bbox(&l["bbox"]).unwrap();
                (b.n[3] < end.n[1] && super::super::graphic_captions::axis(&b, end, 0) >= 0.8)
                    .then_some(b.n[3])
            })
            .reduce(f64::max)
            .unwrap_or(0.);
        let mut aligned = rules
            .iter()
            .copied()
            .filter(|b| {
                floor < b.n[1]
                    && b.n[1] <= end.n[1]
                    && (b.n[0] - end.n[0]).abs() <= 0.3 * p.em
                    && (b.n[2] - end.n[2]).abs() <= 0.3 * p.em
            })
            .collect::<Vec<_>>();
        aligned.sort_by(|a, b| cmp(a.n[1], b.n[1]));
        if aligned.len() != 3 {
            continue;
        }
        let b = bbox(&json!([
            end.raw[0],
            aligned[0].raw[1],
            end.raw[2],
            end.raw[3]
        ]))?;
        work.charge(excluded.len() + p.lines.len() + 1)?;
        if excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5) {
            continue;
        }
        let members = p
            .line_values()
            .filter(|l| overlap_first(&bbox(&l["bbox"]).unwrap(), &b) >= 0.9)
            .collect::<Vec<_>>();
        let rows = rows(&members, p.em, work)?;
        let single = rows.len() >= 5 && rows.iter().all(|r| r.len() == 1 && words(r[0]) <= 5);
        let paired = rows.len() == 2
            && rows[1].len() >= 3
            && (rows[0].len() == rows[1].len()
                || (rows[0].len() == 1 && words(rows[0][0]) == rows[1].len()));
        if !single && !paired {
            continue;
        }
        if paired
            && rows[1]
                .iter()
                .filter(|l| {
                    l["text"]
                        .as_str()
                        .unwrap()
                        .chars()
                        .any(|c| python_decimal(c).is_some())
                })
                .count()
                < 2
        {
            continue;
        }
        let mut grid = Vec::new();
        if single {
            grid.push(axis(&b, b.n[0], false));
            grid.push(axis(&b, b.n[2], false));
            let mut mid = vec![b.n[1]];
            let mut nearest = vec![b.n[1]];
            for pair in rows.windows(2) {
                let a = bbox(&pair[0][0]["bbox"])?;
                let c = bbox(&pair[1][0]["bbox"])?;
                nearest.push((a.n[3].max(a.n[1]) + c.n[1]) / 2.);
                mid.push((a.n[3] + c.n[1]) / 2.);
            }
            mid.push(b.n[3]);
            nearest.push(b.n[3]);
            for (y, n) in mid.into_iter().zip(nearest) {
                let line = aligned
                    .iter()
                    .min_by(|a, b| cmp((center(a, 1) - n).abs(), (center(b, 1) - n).abs()))
                    .unwrap();
                let cy = center(line, 1);
                grid.push(axis(
                    &b,
                    if (cy - y).abs() <= 0.5 * p.em { cy } else { y },
                    true,
                ));
            }
        }
        out.push(candidate(
            b.clone(),
            &members,
            bounded(v, p, &b, work)?,
            grid,
            9,
        )?);
    }
    Ok(out)
}
fn header_marker(s: &str) -> bool {
    let c: Vec<_> = s.trim_start_matches(python_space).chars().collect();
    let mut i = if prefix(&c, "table") {
        5
    } else if prefix(&c, "figure") {
        6
    } else if c.first().is_some_and(|c| "表图".contains(*c)) {
        1
    } else {
        return false;
    };
    while c.get(i).is_some_and(|c| python_space(*c)) {
        i += 1;
    }
    c.get(i).is_some_and(|c| {
        python_decimal(*c).is_some() || matches!(c, 'I' | 'i' | 'İ' | 'ı' | 'V' | 'v' | 'X' | 'x')
    })
}
fn open_headers<'a>(
    p: &'a Page,
    b: &Box4,
    tracks: &[&Box4],
    work: &mut Work,
) -> Result<Vec<&'a Value>, Error> {
    let mut edges = tracks.iter().map(|t| center(t, 0)).collect::<Vec<_>>();
    edges.sort_by(|a, b| cmp(*a, *b));
    edges.insert(0, b.n[0]);
    edges.push(b.n[2]);
    let mut candidates = Vec::new();
    let mut cols = BTreeSet::new();
    let mut duplicate = false;
    for l in p.line_values() {
        work.charge(edges.len() + 1)?;
        let lb = bbox(&l["bbox"])?;
        let h = effective(l)?;
        if flag(l, "caption_start")?
            || b.n[1] - lb.n[3] < 0.
            || b.n[1] - lb.n[3] > 1.5 * p.em
            || h < 0.7 * p.em
            || h > 1.35 * p.em
            || lb.n[0] < b.n[0]
            || lb.n[2] > b.n[2]
            || words(l) < 1
            || words(l) > 5
            || flag(l, "paragraph_terminal")?
            || header_marker(l["text"].as_str().unwrap())
        {
            continue;
        }
        if let Some(i) = edges
            .windows(2)
            .position(|e| e[0] <= lb.n[0] && lb.n[2] <= e[1])
        {
            duplicate |= !cols.insert(i);
            candidates.push(l);
        }
    }
    if candidates.len() < 2 || duplicate {
        return Ok(vec![]);
    }
    let ys = candidates
        .iter()
        .map(|l| center(&bbox(&l["bbox"]).unwrap(), 1))
        .collect::<Vec<_>>();
    if ys.iter().copied().reduce(f64::max).unwrap() - ys.iter().copied().reduce(f64::min).unwrap()
        > 0.3 * p.em
    {
        return Ok(vec![]);
    }
    Ok(candidates)
}
fn open(v: &Request, p: &Page, excluded: &[Box4], work: &mut Work) -> Result<Vec<Value>, Error> {
    // Orientation here is the original source orientation, not inferred local aspect ratio.
    let mut horizontal = Vec::new();
    let mut vertical = Vec::new();
    for r in optional(v.source, "drawing_lines", 5000)? {
        let b = bbox(&r["bbox"])?;
        if r["orientation"] == "vertical" && b.n[3] - b.n[1] >= 5. * p.em {
            vertical.push(b);
        } else if r["orientation"] == "horizontal" && b.n[2] - b.n[0] >= 6. * p.em {
            horizontal.push(b);
        }
    }
    horizontal.sort_by(|a, b| cmp(b.n[2] - b.n[0], a.n[2] - a.n[0]));
    let mut out: Vec<Value> = Vec::new();
    for rule in &horizontal {
        work.charge(vertical.len() + horizontal.len() + excluded.len() + out.len() + 1)?;
        let tracks = vertical
            .iter()
            .filter(|b| {
                rule.n[0] + p.em < center(b, 0)
                    && center(b, 0) < rule.n[2] - p.em
                    && b.n[1] - 0.3 * p.em <= center(rule, 1)
                    && center(rule, 1) <= b.n[3] + 0.3 * p.em
            })
            .collect::<Vec<_>>();
        if tracks.len() < 2 {
            continue;
        }
        let top = tracks.iter().map(|b| b.n[1]).reduce(f64::min).unwrap();
        let bottom = tracks.iter().map(|b| b.n[3]).reduce(f64::max).unwrap();
        let spread = bottom - tracks.iter().map(|b| b.n[3]).reduce(f64::min).unwrap();
        let full = horizontal
            .iter()
            .filter(|b| {
                top <= center(b, 1)
                    && center(b, 1) <= bottom
                    && (b.n[0] - rule.n[0]).abs() <= 0.3 * p.em
                    && (b.n[2] - rule.n[2]).abs() <= 0.3 * p.em
            })
            .collect::<Vec<_>>();
        if tracks.iter().map(|b| b.n[1]).reduce(f64::max).unwrap() - top > 3. * p.em
            || spread > 1.5 * p.em
            || (spread > p.em && full.len() < 3)
        {
            continue;
        }
        let top_raw = &tracks
            .iter()
            .min_by(|a, b| cmp(a.n[1], b.n[1]))
            .unwrap()
            .raw[1];
        let bottom_raw = &tracks
            .iter()
            .rev()
            .max_by(|a, b| cmp(a.n[3], b.n[3]))
            .unwrap()
            .raw[3];
        let b = bbox(&json!([rule.raw[0], top_raw, rule.raw[2], bottom_raw]))?;
        if [b.n[0], b.n[2]].iter().all(|edge| {
            vertical.iter().any(|r| {
                (center(r, 0) - edge).abs() <= 0.3 * p.em
                    && r.n[1] <= top + 0.3 * p.em
                    && r.n[3] >= bottom - 0.3 * p.em
            })
        }) || excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5)
            || out
                .iter()
                .any(|c| overlap_first(&b, &bbox(&c["core_bbox"]).unwrap()) >= 0.9)
        {
            continue;
        }
        work.charge(p.lines.len() + 1)?;
        let mut members = p
            .line_values()
            .filter(|l| {
                let b = bbox(&l["bbox"]).unwrap();
                top - 0.25 * p.em <= center(&b, 1)
                    && center(&b, 1) <= bottom + 0.25 * p.em
                    && rule.n[0] - 0.25 * p.em <= b.n[0]
                    && b.n[2] <= rule.n[2] + 0.25 * p.em
            })
            .collect::<Vec<_>>();
        let headers = open_headers(p, &b, &tracks, work)?;
        members.extend(headers.iter().copied());
        let rows = rows(&members, p.em, work)?;
        let data = rows
            .iter()
            .filter(|r| {
                r.iter()
                    .map(|l| l["bbox"][1].as_f64().unwrap())
                    .reduce(f64::min)
                    .unwrap()
                    >= rule.n[3] - 0.25 * p.em
            })
            .collect::<Vec<_>>();
        let short = tracks.len() >= 3
            && full.len() >= 2
            && rows.iter().any(|r| {
                r.len() >= 3
                    && r.iter()
                        .map(|l| l["bbox"][3].as_f64().unwrap())
                        .reduce(f64::max)
                        .unwrap()
                        < rule.n[1]
            });
        let required = if short { 2 } else { 4 };
        if data.len() < required || data.iter().filter(|r| r.len() >= 2).count() < required {
            continue;
        }
        let mut crossing = false;
        for row in &data {
            for l in *row {
                work.charge(tracks.len() + 1)?;
                let lb = bbox(&l["bbox"])?;
                if (lb.n[2] - lb.n[0] > 0.75 * (rule.n[2] - rule.n[0]) && words(l) > 6)
                    || tracks.iter().any(|t| {
                        t.n[1] <= top + 0.5 * p.em
                            && lb.n[0] + 0.3 * p.em < center(t, 0)
                            && center(t, 0) < lb.n[2] - 0.3 * p.em
                    })
                {
                    crossing = true;
                }
            }
        }
        if crossing {
            continue;
        }
        let mut b = b;
        for l in &members {
            let lb = bbox(&l["bbox"])?;
            if lb.n[1] < b.n[1] {
                b.n[1] = lb.n[1];
                b.raw[1] = lb.raw[1].clone();
            }
            if lb.n[3] > b.n[3] {
                b.n[3] = lb.n[3];
                b.raw[3] = lb.raw[3].clone();
            }
        }
        let mut grid = vec![axis(&b, b.n[0], false), axis(&b, b.n[2], false)];
        if !headers.is_empty() {
            for t in &tracks {
                let x = center(t, 0);
                grid.push(json!({"bbox":[x-0.05,b.raw[1],x+0.05,top.max(t.n[1])],"width":0.1,"orientation":"vertical"}));
            }
        }
        let mut edges = vec![b.n[1], b.n[3]];
        for path in optional(v.source, "path_infos", 5000)? {
            work.charge(1)?;
            if !flag(path, "fill_visible")? || integer(path, "segment_count", 0)? != 5 {
                continue;
            }
            let rgba = optional(path, "fill_rgba", 4)?;
            if rgba.is_empty() {
                continue;
            }
            if rgba.len() < 3 {
                return Err(invalid());
            }
            let pb = bbox(&path["bbox"])?;
            if rgba[..3]
                .iter()
                .map(num)
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .all(|n| *n >= 180.)
                && (pb.n[0] - b.n[0]).abs() <= p.em
                && (pb.n[2] - b.n[2]).abs() <= p.em
                && b.n[1] <= pb.n[1]
                && pb.n[1] < pb.n[3]
                && pb.n[3] <= b.n[3] + 0.2 * p.em
            {
                edges.extend([pb.n[1], b.n[3].min(pb.n[3])]);
            }
        }
        let left_track = tracks
            .iter()
            .map(|t| center(t, 0))
            .reduce(f64::min)
            .unwrap();
        let mut keys = Vec::new();
        for l in &members {
            let lb = bbox(&l["bbox"])?;
            let text = l["text"].as_str().unwrap().trim_matches(python_space);
            if center(&lb, 0) < left_track
                && lb.n[1] >= rule.n[3] - 0.25 * p.em
                && (1..=3).contains(&text.chars().count())
                && text.chars().all(|c| python_decimal(c).is_some())
            {
                let n = text
                    .chars()
                    .fold(0u32, |n, c| n * 10 + python_decimal(c).unwrap() as u32);
                keys.push((*l, n));
            }
        }
        keys.sort_by(|a, b| {
            cmp(
                a.0["bbox"][1].as_f64().unwrap(),
                b.0["bbox"][1].as_f64().unwrap(),
            )
        });
        if keys.len() >= 4 && keys.windows(2).all(|k| k[1].1 == k[0].1 + 1) {
            for pair in keys.windows(2) {
                edges.push(
                    (center(&bbox(&pair[0].0["bbox"])?, 1) + center(&bbox(&pair[1].0["bbox"])?, 1))
                        / 2.,
                );
            }
        }
        edges.sort_by(|a, b| cmp(*a, *b));
        edges.dedup();
        for y in edges {
            work.charge(horizontal.len() + 1)?;
            if !horizontal.iter().any(|r| {
                (center(r, 1) - y).abs() <= 0.25 * p.em
                    && (r.n[0] - b.n[0]).abs() <= p.em
                    && (r.n[2] - b.n[2]).abs() <= p.em
            }) {
                grid.push(axis(&b, y, true));
            }
        }
        out.push(candidate(
            b.clone(),
            &members,
            bounded(v, p, &b, work)?,
            grid,
            12,
        )?);
    }
    Ok(out)
}
pub(super) fn detect(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    detect_cached(
        &Request {
            source: &v["source"],
            options: v,
        },
        work,
        &mut None,
    )
}
pub(super) fn detect_source(
    source: &Value,
    options: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    crate::legacy_page::validate_source_request(source, options)?;
    detect_cached(&Request { source, options }, work, &mut None)
}
fn detect_cached<'a>(
    v: &Request<'a, '_>,
    work: &mut Work,
    cached: &mut Option<Option<Page<'a>>>,
) -> Result<Value, Error> {
    work.charge(1)?;
    v.source.as_object().ok_or_else(invalid)?;
    let kind = v.options["kind"].as_str().ok_or_else(invalid)?;
    if !matches!(
        kind,
        "short_caption"
            | "open_vertical"
            | "shaded_header"
            | "blank_answer"
            | "page_clipped"
            | "partial_grid"
            | "year_header"
            | "metric_pair"
            | "unruled_numeric"
            | "captioned_pair"
            | "aligned_numeric"
            | "captioned_mixed"
    ) {
        return Err(invalid());
    }
    if kind == "shaded_header" {
        return shaded::detect(v, work);
    }
    if kind == "aligned_numeric" {
        return aligned::detect(v, work);
    }
    let uncached;
    let prepared = if crate::legacy_page::budget::current().work == usize::MAX {
        if cached.is_none() {
            *cached = Some(source(v.source, work)?);
        }
        cached.as_ref().unwrap()
    } else {
        uncached = source(v.source, work)?;
        &uncached
    };
    let Some(p) = prepared.as_ref() else {
        return Ok(json!({"candidates":[]}));
    };
    let excluded = exclusions(v.options, &p, work)?;
    let candidates = if kind == "captioned_mixed" {
        mixed::detect(v, &p, &excluded, work)?
    } else if matches!(kind, "unruled_numeric" | "captioned_pair") {
        text_numeric::detect(v, &p, &excluded, work)?
    } else if matches!(kind, "partial_grid" | "year_header" | "metric_pair") {
        numeric_segments::detect(v, &p, &excluded, work)?
    } else if kind == "blank_answer" || kind == "page_clipped" {
        forms::detect(v, &p, &excluded, work)?
    } else if kind == "short_caption" {
        short(v, &p, &excluded, work)?
    } else {
        open(v, &p, &excluded, work)?
    };
    if candidates.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let mut members = 0;
    let mut grid = 0;
    for c in &candidates {
        members += c["line_indices"].as_array().unwrap().len();
        grid += c["inferred_grid"].as_array().unwrap().len();
    }
    work.charge(members + grid + 1)?;
    if members > rapiddoc_core::resources::limit(20000)
        || grid > rapiddoc_core::resources::limit(20000)
    {
        return Err(limit());
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
