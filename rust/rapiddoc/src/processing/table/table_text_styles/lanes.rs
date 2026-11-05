//! 旧line_layout._infer_text_lanes：稳定栏带、局部双栏与跨栏尾行编排。
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::BTreeSet;
type Box4 = [f64; 4];
#[derive(Clone)]
struct Line {
    b: Box4,
    h: f64,
    source: i64,
    semantic: Option<String>,
    font: Value,
    coverage: f64,
    weight: Option<f64>,
    baseline: Option<f64>,
    restored: bool,
    row: Option<i64>,
}
#[derive(Clone, Debug)]
pub(super) struct Lane {
    pub left: f64,
    pub right: f64,
    pub members: Vec<usize>,
    pub span: bool,
}
pub(super) fn number(v: &Value) -> Result<f64, Error> {
    let n = v
        .as_f64()
        .or_else(|| v.as_bool().map(|b| u8::from(b) as f64))
        .filter(|n| n.is_finite())
        .ok_or_else(invalid)?;
    if n.abs() > 1e9 {
        return Err(limit());
    }
    Ok(n)
}
pub(super) fn num(v: &Value, k: &str, default: f64) -> Result<f64, Error> {
    v.get(k).map_or(Ok(default), number)
}
fn opt(v: &Value, k: &str) -> Result<Option<f64>, Error> {
    v.get(k)
        .filter(|x| !x.is_null())
        .map(|v| {
            if k == "dominant_font_weight" {
                v.as_f64()
                    .or_else(|| v.as_bool().map(|b| u8::from(b) as f64))
                    .filter(|n| n.is_finite())
                    .ok_or_else(invalid)
            } else {
                number(v)
            }
        })
        .transpose()
}
pub(super) fn flag(v: &Value, k: &str) -> Result<bool, Error> {
    v.get(k)
        .map_or(Ok(false), |x| x.as_bool().ok_or_else(invalid))
}
pub(super) fn rect(v: &Value) -> Result<Box4, Error> {
    let a = v.as_array().filter(|a| a.len() == 4).ok_or_else(invalid)?;
    let b = [
        number(&a[0])?,
        number(&a[1])?,
        number(&a[2])?,
        number(&a[3])?,
    ];
    if b[2] <= b[0] || b[3] <= b[1] {
        return Err(invalid());
    }
    Ok(b)
}
pub(super) fn median(mut a: Vec<f64>) -> f64 {
    a.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = a.len();
    if n % 2 == 1 {
        a[n / 2]
    } else {
        (a[n / 2 - 1] + a[n / 2]) / 2.
    }
}
fn center(b: Box4) -> f64 {
    (b[0] + b[2]) / 2.
}
fn overlap(a: Box4, b: Box4, axis: usize) -> f64 {
    let den = (a[axis + 2] - a[axis]).min(b[axis + 2] - b[axis]);
    if den <= 0. {
        0.
    } else {
        (a[axis + 2].min(b[axis + 2]) - a[axis].max(b[axis])).max(0.) / den
    }
}
fn sort_members(m: &mut [usize], lines: &[Line]) {
    m.sort_by(|&i, &j| {
        lines[i].b[1]
            .partial_cmp(&lines[j].b[1])
            .unwrap()
            .then_with(|| lines[i].b[0].partial_cmp(&lines[j].b[0]).unwrap())
            .then(lines[i].source.cmp(&lines[j].source))
    });
}
fn compatible(a: &Line, b: &Line) -> bool {
    !(!a.font.is_null()
        && !b.font.is_null()
        && a.coverage >= 0.75
        && b.coverage >= 0.75
        && !eq(&a.font, &b.font)
        || matches!((a.weight,b.weight),(Some(x),Some(y))if(x-y).abs()>=100.&&x.max(y)>=1.15*x.min(y)))
}
fn gap(a: &Line, b: &Line) -> f64 {
    if a.restored {
        (b.b[1] - a.b[3]).max(-0.25 * a.h)
    } else {
        b.b[1] - (a.b[1] + a.h)
    }
}
fn new_lane(left: f64, right: f64) -> Lane {
    Lane {
        left,
        right,
        members: Vec::new(),
        span: false,
    }
}
fn span(members: Vec<usize>, lines: &[Line]) -> Lane {
    Lane {
        left: members
            .iter()
            .map(|&i| lines[i].b[0])
            .reduce(f64::min)
            .unwrap(),
        right: members
            .iter()
            .map(|&i| lines[i].b[2])
            .reduce(f64::max)
            .unwrap(),
        members,
        span: true,
    }
}
fn expanded(lanes: &mut [Lane], lines: &[Line], t: f64, w: &mut Work) -> Result<(), Error> {
    for lane in lanes.iter_mut() {
        w.charge(lane.members.len())?;
        let boxes: Vec<_> = lane
            .members
            .iter()
            .map(|&i| &lines[i])
            .filter(|l| {
                l.semantic.is_none()
                    && ((l.b[0] - lane.left).abs() <= 1f64.max(0.5 * t)
                        || (l.b[2] - lane.right).abs() <= 1f64.max(0.5 * t))
            })
            .map(|l| l.b)
            .collect();
        for b in boxes {
            lane.left = lane.left.min(b[0]);
            lane.right = lane.right.max(b[2]);
        }
    }
    let mut order: Vec<_> = (0..lanes.len()).collect();
    order.sort_by(|&i, &j| lanes[i].left.partial_cmp(&lanes[j].left).unwrap());
    for pair in order.windows(2) {
        let (l, r) = (pair[0], pair[1]);
        if lanes[l].right < lanes[r].left - t {
            continue;
        }
        let mid = (lanes[l].right + lanes[r].left) / 2.;
        lanes[l].right = lanes[l].right.min(mid);
        lanes[r].left = lanes[r].left.max(mid);
    }
    Ok(())
}
fn nested(
    lines: &[Line],
    anchors: &[usize],
    width: f64,
    h: f64,
    outer: (f64, f64),
    enhanced: bool,
    w: &mut Work,
) -> Result<Option<(Vec<Lane>, f64, f64)>, Error> {
    let outer_width = (outer.1 - outer.0).max(0.1);
    let mut candidates: Vec<_> = anchors
        .iter()
        .copied()
        .filter(|&i| {
            let l = &lines[i];
            l.semantic.is_none()
                && l.b[2] - l.b[0] >= (4. * l.h).max(0.12 * width)
                && l.b[2] - l.b[0] <= 0.62 * outer_width
        })
        .collect();
    if candidates.len() < 6 {
        return Ok(None);
    }
    candidates.sort_by(|&i, &j| center(lines[i].b).partial_cmp(&center(lines[j].b)).unwrap());
    let mut clusters = Vec::<Vec<usize>>::new();
    let tolerance = (2. * h).max(0.06 * width);
    for i in candidates {
        let mut target = None;
        for (k, c) in clusters.iter().enumerate() {
            w.charge(c.len())?;
            if (center(lines[i].b) - median(c.iter().map(|&i| center(lines[i].b)).collect())).abs()
                <= tolerance
            {
                target = Some(k);
                break;
            }
        }
        if let Some(k) = target {
            clusters[k].push(i)
        } else {
            clusters.push(vec![i]);
        }
    }
    let mut supported: Vec<_> = clusters.into_iter().filter(|c| c.len() >= 3).collect();
    supported.sort_by(|a, b| {
        median(a.iter().map(|&i| center(lines[i].b)).collect())
            .partial_cmp(&median(b.iter().map(|&i| center(lines[i].b)).collect()))
            .unwrap()
    });
    type Choice = ((usize, f64, f64), (f64, f64), (f64, f64), Vec<usize>);
    let mut best: Option<Choice> = None;
    for i in 0..supported.len() {
        for j in i + 1..supported.len() {
            if !enhanced && j != i + 1 {
                continue;
            }
            let left = &supported[i];
            let right = &supported[j];
            w.charge(left.len() + right.len())?;
            let li = (
                median(left.iter().map(|&i| lines[i].b[0]).collect()),
                median(left.iter().map(|&i| lines[i].b[2]).collect()),
            );
            let ri = (
                median(right.iter().map(|&i| lines[i].b[0]).collect()),
                median(right.iter().map(|&i| lines[i].b[2]).collect()),
            );
            let gutter = ri.0 - li.1;
            let top = left
                .iter()
                .map(|&i| lines[i].b[1])
                .reduce(f64::min)
                .unwrap()
                .max(
                    right
                        .iter()
                        .map(|&i| lines[i].b[1])
                        .reduce(f64::min)
                        .unwrap(),
                );
            let bottom = left
                .iter()
                .map(|&i| lines[i].b[3])
                .reduce(f64::max)
                .unwrap()
                .min(
                    right
                        .iter()
                        .map(|&i| lines[i].b[3])
                        .reduce(f64::max)
                        .unwrap(),
                );
            if gutter < 6f64.max(0.75 * h)
                || bottom - top < 2. * h
                || ri.1 - li.0 < 0.55 * outer_width
            {
                continue;
            }
            let score = (left.len().min(right.len()), bottom - top, gutter);
            if best.as_ref().is_none_or(|v| score > v.0) {
                best = Some((score, li, ri, left.iter().chain(right).copied().collect()));
            }
        }
    }
    Ok(best.map(|(_, l, r, ids)| {
        let top = ids.iter().map(|&i| lines[i].b[1]).reduce(f64::min).unwrap() - 0.5 * h;
        let bottom = ids.iter().map(|&i| lines[i].b[3]).reduce(f64::max).unwrap() + 0.5 * h;
        (vec![new_lane(l.0, l.1), new_lane(r.0, r.1)], top, bottom)
    }))
}
fn parallel(
    candidate: usize,
    lanes: &[Lane],
    regular: &[usize],
    lines: &[Line],
    w: &mut Work,
) -> Result<bool, Error> {
    for &r in regular {
        for &other in &lanes[r].members {
            w.charge(1)?;
            if candidate != other && overlap(lines[candidate].b, lines[other].b, 1) >= 0.5 {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
fn move_row(
    lanes: &mut [Lane],
    source: usize,
    target: usize,
    id: usize,
    lines: &[Line],
    expand: bool,
) {
    let p = lanes[source].members.iter().position(|&i| i == id).unwrap();
    lanes[source].members.remove(p);
    lanes[target].members.push(id);
    if expand {
        lanes[target].left = lanes[target].left.min(lines[id].b[0]);
        lanes[target].right = lanes[target].right.max(lines[id].b[2]);
    }
    sort_members(&mut lanes[target].members, lines);
}
fn repeated_tails(
    s: usize,
    regular: &[usize],
    lanes: &mut [Lane],
    lines: &[Line],
    h: f64,
    w: &mut Work,
) -> Result<(), Error> {
    let mut span_rows = lanes[s].members.clone();
    sort_members(&mut span_rows, lines);
    let mut matches = Vec::new();
    for (pos, &id) in span_rows.iter().enumerate() {
        let l = &lines[id];
        let next = span_rows
            .get(pos + 1)
            .map_or(f64::INFINITY, |&i| lines[i].b[1]);
        for &r in regular {
            for &c in &lanes[r].members {
                w.charge(1)?;
                let b = &lines[c];
                if l.semantic != b.semantic {
                    continue;
                }
                let g = gap(l, b);
                let indent = b.b[0] - l.b[0];
                if b.b[1] <= l.b[1]
                    || b.b[1] >= next
                    || g < -0.25 * h
                    || g > 0.75 * h
                    || indent < 0.75 * h
                    || indent > 6. * h
                    || l.h.max(b.h) / l.h.min(b.h) > 1.35
                    || !compatible(l, b)
                {
                    continue;
                }
                if !parallel(c, lanes, regular, lines, w)? {
                    matches.push((r, c, indent));
                }
            }
        }
    }
    if matches.len() < 2 {
        return Ok(());
    }
    let indent = median(matches.iter().map(|v| v.2).collect());
    let supported: Vec<_> = matches
        .into_iter()
        .filter(|v| (v.2 - indent).abs() <= (0.75 * h).max(0.25 * indent))
        .collect();
    if supported.len() < 2 {
        return Ok(());
    }
    for (r, c, _) in supported {
        if lanes[r].members.contains(&c) {
            move_row(lanes, r, s, c, lines, true)
        }
    }
    sort_members(&mut lanes[s].members, lines);
    Ok(())
}
fn span_continuations(
    lanes: &mut [Lane],
    lines: &[Line],
    h: f64,
    w: &mut Work,
) -> Result<(), Error> {
    let regular: Vec<_> = lanes
        .iter()
        .enumerate()
        .filter_map(|(i, l)| (!l.span).then_some(i))
        .collect();
    let spans: Vec<_> = lanes
        .iter()
        .enumerate()
        .filter_map(|(i, l)| l.span.then_some(i))
        .collect();
    if regular.len() < 2 || spans.is_empty() {
        return Ok(());
    }
    for s in spans {
        repeated_tails(s, &regular, lanes, lines, h, w)?;
        loop {
            sort_members(&mut lanes[s].members, lines);
            let mut candidates = Vec::new();
            for &r in &regular {
                for &c in &lanes[r].members {
                    let b = &lines[c];
                    w.charge(lanes[s].members.len())?;
                    let preceding: Vec<_> = lanes[s]
                        .members
                        .iter()
                        .copied()
                        .filter(|&i| lines[i].semantic == b.semantic && lines[i].b[1] < b.b[1])
                        .collect();
                    if preceding.len() < 2 {
                        continue;
                    }
                    let a = &lines[preceding[preceding.len() - 2]];
                    let last = &lines[preceding[preceding.len() - 1]];
                    if (a.b[0] - last.b[0]).abs() > 0.75 * h
                        || a.h.max(last.h) / a.h.min(last.h) > 1.35
                        || !compatible(a, last)
                        || gap(a, last) < -0.25 * h
                        || gap(a, last) > 0.75 * h
                    {
                        continue;
                    }
                    let g = gap(last, b);
                    if g < -0.25 * h
                        || g > 0.75 * h
                        || (b.b[0] - last.b[0]).abs() > 0.75 * h
                        || last.h.max(b.h) / last.h.min(b.h) > 1.35
                        || !compatible(last, b)
                    {
                        continue;
                    }
                    if !parallel(c, lanes, &regular, lines, w)? {
                        candidates.push((b.b[1], g.max(0.), r, c));
                    }
                }
            }
            let choice = candidates.into_iter().min_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap()
                    .then_with(|| a.1.partial_cmp(&b.1).unwrap())
            });
            let Some((_, _, r, c)) = choice else { break };
            move_row(lanes, r, s, c, lines, true);
        }
    }
    Ok(())
}
fn cross_tails(
    lanes: &mut [Lane],
    lines: &[Line],
    h: f64,
    w: &mut Work,
    formula: bool,
) -> Result<(), Error> {
    if lanes.len() < 2 {
        return Ok(());
    }
    let mut signatures = Vec::<Value>::new();
    let mut records = Vec::new();
    let mut ids = Vec::new();
    let bounds: Vec<_> = lanes.iter().map(|l| (l.left, l.right)).collect();
    for (lane, l) in lanes.iter().enumerate() {
        for &id in &l.members {
            let v = &lines[id];
            if v.semantic.is_some() {
                continue;
            }
            let font = if v.font.is_null() {
                None
            } else {
                let mut found = None;
                for (i, s) in signatures.iter().enumerate() {
                    w.charge(1)?;
                    if eq(s, &v.font) {
                        found = Some(i);
                        break;
                    }
                }
                if found.is_none() {
                    found = Some(signatures.len());
                    signatures.push(v.font.clone());
                }
                found
            };
            ids.push((lane, id));
            records.push((
                v.b, v.h, font, v.coverage, v.weight, v.baseline, v.restored, v.row, v.source, lane,
            ));
        }
    }
    w.charge(records.len().saturating_mul(lanes.len()))?;
    let destinations = if formula {
        rapiddoc_native::text_formula_tail_destinations(&records, &bounds, h)?
    } else {
        rapiddoc_native::text_short_tail_destinations(&records, &bounds, h)?
    };
    for (i, target) in destinations {
        let (source, id) = ids[i];
        move_row(lanes, source, target, id, lines, false)
    }
    Ok(())
}
fn infer(
    lines: &[Line],
    width: f64,
    h: f64,
    recalculate: bool,
    w: &mut Work,
    formula: bool,
) -> Result<Vec<Lane>, Error> {
    let geometries: Vec<_> = lines
        .iter()
        .map(|l| {
            (
                l.b,
                l.h,
                !matches!(
                    l.semantic.as_deref(),
                    Some("header" | "footer" | "page_number" | "aside_text")
                ),
                matches!(
                    l.semantic.as_deref(),
                    Some("header" | "footer" | "page_number" | "page_footnote" | "aside_text")
                ),
            )
        })
        .collect();
    w.charge(lines.len().saturating_mul(lines.len()))?;
    let intervals = if formula {
        rapiddoc_native::text_formula_lane_intervals(&geometries, width, h)?
    } else {
        rapiddoc_native::text_supported_lane_intervals(&geometries, width, h)?
    };
    let anchors: Vec<_> = geometries
        .iter()
        .enumerate()
        .filter_map(|(i, l)| l.2.then_some(i))
        .collect();
    let tolerance = 3f64.max(0.75 * h);
    let band = if intervals.len() == 1 && !anchors.is_empty() {
        let outer = if recalculate {
            (
                anchors
                    .iter()
                    .map(|&i| lines[i].b[0])
                    .reduce(f64::min)
                    .unwrap(),
                anchors
                    .iter()
                    .map(|&i| lines[i].b[2])
                    .reduce(f64::max)
                    .unwrap(),
            )
        } else {
            (intervals[0].0, intervals[0].1)
        };
        nested(lines, &anchors, width, h, outer, recalculate, w)?
    } else {
        None
    };
    let (mut lanes, nested_band) = if let Some((lanes, top, bottom)) = band {
        (lanes, Some((top, bottom)))
    } else {
        (
            intervals.iter().map(|&(l, r, _)| new_lane(l, r)).collect(),
            None,
        )
    };
    w.charge(lines.len().saturating_mul(lanes.len()))?;
    let bounds: Vec<_> = lanes.iter().map(|l| (l.left, l.right)).collect();
    let assignments = if formula {
        rapiddoc_native::text_formula_lane_assignments(
            &geometries,
            &bounds,
            tolerance,
            nested_band,
        )?
    } else {
        rapiddoc_native::text_lane_assignments(&geometries, &bounds, tolerance, nested_band)?
    };
    let mut fallback = new_lane(intervals[0].0, intervals[0].1);
    let mut spans = Vec::new();
    for (i, d) in assignments.into_iter().enumerate() {
        match d {
            Some(k) if k == lanes.len() => fallback.members.push(i),
            Some(k) => lanes[k].members.push(i),
            None => spans.push(i),
        }
    }
    if nested_band.is_some() {
        lanes.retain(|l| !l.members.is_empty())
    }
    if recalculate {
        expanded(&mut lanes, lines, tolerance, w)?
    }
    if !fallback.members.is_empty() {
        lanes.push(fallback)
    }
    if !spans.is_empty() {
        lanes.push(span(spans, lines))
    }
    span_continuations(&mut lanes, lines, h, w)?;
    cross_tails(&mut lanes, lines, h, w, formula)?;
    Ok(lanes)
}
pub fn infer_json(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    infer_with_work(v, &mut Work(0))
}
pub(super) fn infer_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    infer_bounded(v, work, 512, false)
}
pub(in crate::processing) fn infer_layout(v: &Value, work: &mut Work) -> Result<Value, Error> {
    infer_bounded(v, work, 4096, false)
}
// Formula replay keeps original object identities even when source IDs coincide.
pub(in crate::processing) fn infer_formula_layout(
    v: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    infer_bounded(v, work, 4096, true)
}
fn infer_bounded(
    v: &Value,
    work: &mut Work,
    maximum: usize,
    duplicate_sources: bool,
) -> Result<Value, Error> {
    let rows = v["line_geometry"]
        .as_array()
        .filter(|a| !a.is_empty() && a.len() <= crate::legacy_page::budget::records(maximum))
        .ok_or_else(limit)?;
    infer_rows(
        rows.iter().map(|row| {
            let row = row
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(invalid)?;
            Ok((&row[0], &row[1]))
        }),
        v,
        work,
        duplicate_sources,
    )
}
/// Internal read-only entry: the geometry pairs borrow the original line trees.
pub(in crate::processing) fn infer_layout_rows(
    rows: &[(&Value, &Value)],
    v: &Value,
    work: &mut Work,
    full: bool,
) -> Result<Value, Error> {
    if rows.is_empty()
        || rows.len() > crate::legacy_page::budget::records(if full { 4096 } else { 512 })
    {
        return Err(limit());
    }
    infer_rows(rows.iter().copied().map(Ok), v, work, false)
}
fn infer_rows<'a>(
    rows: impl IntoIterator<Item = Result<(&'a Value, &'a Value), Error>>,
    v: &Value,
    work: &mut Work,
    duplicate_sources: bool,
) -> Result<Value, Error> {
    let width = number(&v["local_page_width"])?;
    let h = number(&v["median_height"])?;
    if width <= 0. || h <= 0. {
        return Err(invalid());
    }
    let recalculate = v
        .get("recalculate_intervals")
        .map_or(Ok(true), |v| v.as_bool().ok_or_else(invalid))?;
    let mut lines = Vec::new();
    let mut sources = BTreeSet::new();
    for row in rows {
        let (v, bounds) = row?;
        if !v.is_object() {
            return Err(invalid());
        }
        let b = if duplicate_sources {
            // Original formula replay accepts finite degenerate and inverted boxes.
            let raw = bounds
                .as_array()
                .filter(|a| a.len() == 4)
                .ok_or_else(invalid)?;
            [
                number(&raw[0])?,
                number(&raw[1])?,
                number(&raw[2])?,
                number(&raw[3])?,
            ]
        } else {
            rect(bounds)?
        };
        let source = v["source_index"].as_i64().ok_or_else(invalid)?;
        if !sources.insert(source) && !duplicate_sources {
            return Err(invalid());
        }
        let effective = num(v, "effective_height", 0.)?;
        let em = num(v, "em_height", 0.)?;
        let height = 0.1f64.max(if flag(v, "style_scale_repaired")? && em > 0. {
            em
        } else if effective != 0. {
            effective
        } else {
            b[3] - b[1]
        });
        let semantic = v
            .get("semantic_type")
            .filter(|v| !v.is_null())
            .map(|v| v.as_str().map(str::to_owned).ok_or_else(invalid))
            .transpose()?;
        let font = v.get("font_signature").cloned().unwrap_or(Value::Null);
        if !font.is_null() {
            let a = font
                .as_array()
                .filter(|a| {
                    a.len() == 2 && a[0].is_string() && (a[1].is_number() || a[1].is_boolean())
                })
                .ok_or_else(invalid)?;
            if a[0].as_str().unwrap().len() > rapiddoc_core::resources::limit(4096) {
                return Err(limit());
            }
        }
        let row = v
            .get("visual_row_id")
            .filter(|v| !v.is_null())
            .map(|v| v.as_i64().ok_or_else(invalid))
            .transpose()?;
        lines.push(Line {
            b,
            h: height,
            source,
            semantic,
            font,
            coverage: num(v, "font_coverage", 0.)?,
            weight: opt(v, "dominant_font_weight")?,
            baseline: opt(v, "baseline")?,
            restored: flag(v, "restored_inline_cluster")?,
            row,
        });
    }
    let lanes = infer(&lines, width, h, recalculate, work, duplicate_sources)?;
    Ok(json!(lanes
        .into_iter()
        .map(|l| json!({"left":l.left,"right":l.right,"is_span":l.span,"line_indices":l.members}))
        .collect::<Vec<_>>()))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn borrowed_lane_geometry_matches_json_and_work() {
        let f: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!("../../../../../../tests/fixtures/native_owned_line_reference.json.gz")
                [..],
        ))
        .unwrap();
        for stage in ["roles", "notes", "clusters"] {
            for c in f[stage].as_array().unwrap() {
                let lines = c["request"]["lines"].as_array().unwrap();
                let pairs: Vec<_> = lines.iter().map(|l| (l, &l["bbox"])).collect();
                for full in [false, true] {
                    for recalculate in [false, true] {
                        let v = json!({"line_geometry":lines.iter().map(|l|json!([l,l["bbox"]])).collect::<Vec<_>>(),"local_page_width":400.,"median_height":10.,"recalculate_intervals":recalculate});
                        let (mut a, mut b) = (Work(0), Work(0));
                        let old = infer_bounded(&v, &mut a, if full { 4096 } else { 512 }, false);
                        let new = infer_layout_rows(&pairs, &v, &mut b, full);
                        assert_eq!(a.0, b.0, "{stage} {}", c["label"]);
                        match (old, new) {
                            (Ok(a), Ok(b)) => assert_eq!(a, b, "{stage} {}", c["label"]),
                            (Err(a), Err(b)) => {
                                assert_eq!(a.code, b.code);
                                // Budget diagnostics include the call-site path/line.
                                assert_eq!(
                                    a.message.split('（').next(),
                                    b.message.split('（').next()
                                );
                            }
                            other => panic!("{stage} {}: {other:?}", c["label"]),
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn old_complete_text_lanes() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-text-lanes.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                infer_json(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
}
