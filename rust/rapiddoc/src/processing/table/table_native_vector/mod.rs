//! 旧vector.py物理证据、轨道硬门、拓扑与矢量候选；输入已有页面原语。
use super::table_native_candidate::sum;
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
mod pipeline;
pub(super) mod python_set;
mod topology;
pub(super) use pipeline::generate_with_work;
pub use pipeline::{generate, inspect_tracks};
pub use topology::build;
pub(super) fn validate_text(v: &Value) -> Result<super::table_native_text::Text, Error> {
    topology::validate_text(v)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Rule {
    orientation: String,
    coordinate: f64,
    start: f64,
    end: f64,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
struct Track {
    coordinate: f64,
    aliases: Vec<f64>,
}
fn combined(a: &Track, b: &Track) -> Track {
    let mut aliases = a
        .aliases
        .iter()
        .chain(&b.aliases)
        .copied()
        .collect::<Vec<_>>();
    aliases.sort_by(|a, b| a.partial_cmp(b).unwrap());
    aliases.dedup();
    let m = aliases.len() / 2;
    let coordinate = if aliases.len() % 2 == 1 {
        aliases[m]
    } else {
        (aliases[m - 1] + aliases[m]) / 2.
    };
    Track {
        coordinate,
        aliases,
    }
}
fn indices(
    track: &Track,
    rules: &[Rule],
    orientation: &str,
    tolerance: f64,
    work: &mut Work,
) -> Result<Vec<usize>, Error> {
    work.charge(rules.len() * track.aliases.len())?;
    Ok(rules
        .iter()
        .enumerate()
        .filter(|(_, r)| {
            r.orientation == orientation
                && track
                    .aliases
                    .iter()
                    .any(|a| (r.coordinate - a).abs() <= tolerance)
        })
        .map(|(i, _)| i)
        .collect())
}
fn track_coverage(
    track: &Track,
    rules: &[Rule],
    orientation: &str,
    extent: f64,
    tolerance: f64,
    work: &mut Work,
) -> Result<f64, Error> {
    let selected = indices(track, rules, orientation, tolerance, work)?;
    coverage(
        selected
            .iter()
            .map(|i| (rules[*i].start, rules[*i].end))
            .collect(),
        0.,
        extent,
    )
}
fn numbers(v: &Value) -> Result<Vec<f64>, Error> {
    v.as_array()
        .ok_or_else(invalid)?
        .iter()
        .map(number)
        .collect()
}
/// 外缘吸附/窄带折叠/alias冲突，保留旧物理双边界和最先命中顺序。
pub fn canonicalize(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    if v["positions"]
        .as_array()
        .is_some_and(|a| a.len() > rapiddoc_core::resources::limit(200))
    {
        return Err(limit());
    }
    canonicalize_with_work(v, &mut Work(0))
}
pub(super) fn canonicalize_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    let positions = numbers(&v["positions"])?;
    let centers = numbers(&v["glyph_centers"])?;
    if positions.len() > crate::legacy_page::budget::records(20000)
        || centers.len() > crate::legacy_page::budget::records(3000)
    {
        return Err(limit());
    }
    let threshold = number(&v["threshold"])?;
    let extent = number(&v["extent"])?;
    let halo = number(&v["evidence_halo"])?;
    let tolerance = number(&v["snap_tolerance"])?;
    let separator_extent = number(&v["separator_extent"])?;
    let minimum = v["minimum_track_count"]
        .as_u64()
        .filter(|n| *n <= 200)
        .ok_or_else(invalid)? as usize;
    if [threshold, extent, halo, tolerance, separator_extent]
        .iter()
        .any(|n| *n < 0.)
    {
        return Err(invalid());
    }
    let rules: Option<Vec<Rule>> = v
        .get("rules")
        .filter(|r| !r.is_null())
        .map(|r| serde::Deserialize::deserialize(r).map_err(|_| invalid()))
        .transpose()?;
    let oriented = v["orientation"]
        .as_str()
        .filter(|s| ["", "horizontal", "vertical"].contains(s))
        .ok_or_else(invalid)?;
    if let Some(rs) = &rules {
        if rs.len() > rapiddoc_core::resources::limit(5000) {
            return Err(limit());
        }
        if rs.iter().any(|r| {
            !["horizontal", "vertical"].contains(&r.orientation.as_str())
                || [r.coordinate, r.start, r.end]
                    .iter()
                    .any(|n| !n.is_finite() || n.abs() > 1e9)
        }) {
            return Err(invalid());
        }
    }
    let leading = flag(v, "collapse_leading_edge", true)?;
    let narrow = flag(v, "collapse_narrow_bands", true)?;
    let preserve = flag(v, "preserve_double_boundary", false)?;
    let mut tracks: Vec<_> = positions
        .into_iter()
        .map(|coordinate| Track {
            coordinate,
            aliases: vec![coordinate],
        })
        .collect();
    for edge in [0., extent] {
        let candidates: Vec<_> = tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| (t.coordinate - edge).abs() <= halo)
            .map(|(i, _)| i)
            .collect();
        if candidates.is_empty() {
            continue;
        }
        let closest = tracks[*candidates
            .iter()
            .min_by(|a, b| {
                (tracks[**a].coordinate - edge)
                    .abs()
                    .partial_cmp(&(tracks[**b].coordinate - edge).abs())
                    .unwrap()
            })
            .unwrap()]
        .coordinate;
        let matched: Vec<_> = candidates
            .into_iter()
            .filter(|i| (tracks[*i].coordinate - closest).abs() <= tolerance)
            .collect();
        let mut aliases = vec![edge];
        for i in &matched {
            aliases.extend(&tracks[*i].aliases)
        }
        aliases.sort_by(|a, b| a.partial_cmp(b).unwrap());
        aliases.dedup();
        tracks.splice(
            matched[0]..=*matched.last().unwrap(),
            [Track {
                coordinate: edge,
                aliases,
            }],
        );
    }
    let mut outer = 0;
    if let Some(rs) = rules.as_ref().filter(|_| !oriented.is_empty()) {
        while tracks.len() >= 2 {
            let mut candidates = vec![tracks.len() - 2];
            if leading {
                candidates.insert(0, 0)
            }
            let mut chosen = None;
            for i in candidates {
                let (left, right) = (&tracks[i], &tracks[i + 1]);
                work.charge(centers.len())?;
                if right.coordinate - left.coordinate > threshold
                    || centers
                        .iter()
                        .any(|c| left.coordinate < *c && *c < right.coordinate)
                {
                    continue;
                }
                if right.coordinate > threshold.max(2. * tolerance)
                    && extent - left.coordinate > threshold.max(2. * tolerance)
                {
                    continue;
                }
                let a = indices(left, rs, oriented, tolerance, work)?;
                let b = indices(right, rs, oriented, tolerance, work)?;
                if !a.is_empty() && !b.is_empty() {
                    work.charge(a.len() * 13)?;
                    if !a.iter().any(|i| b.binary_search(i).is_ok()) {
                        continue;
                    }
                    let near = |ids: &[usize], c: f64| {
                        python_set::ordered(ids)
                            .into_iter()
                            .min_by(|a, b| {
                                (rs[*a].coordinate - c)
                                    .abs()
                                    .partial_cmp(&(rs[*b].coordinate - c).abs())
                                    .unwrap()
                            })
                            .unwrap()
                    };
                    if rs[near(&a, left.coordinate)].coordinate
                        != rs[near(&b, right.coordinate)].coordinate
                    {
                        continue;
                    }
                }
                let merged = combined(left, right);
                if merged.aliases.last().unwrap() - merged.aliases[0] > threshold
                    || track_coverage(&merged, rs, oriented, separator_extent, tolerance, work)?
                        < 0.8
                {
                    continue;
                }
                chosen = Some(i);
                break;
            }
            let Some(i) = chosen else { break };
            let merged = combined(&tracks[i], &tracks[i + 1]);
            tracks.splice(i..=i + 1, [merged]);
            outer += 1;
        }
    }
    let mut conflict = false;
    while narrow && tracks.len() > minimum {
        let mut chosen = None;
        for i in 0..tracks.len().saturating_sub(1) {
            let (a, b) = (&tracks[i], &tracks[i + 1]);
            work.charge(centers.len())?;
            if b.coordinate - a.coordinate > threshold
                || centers
                    .iter()
                    .any(|c| a.coordinate < *c && *c < b.coordinate)
            {
                continue;
            }
            if preserve {
                if let Some(rs) = &rules {
                    if track_coverage(a, rs, oriented, separator_extent, tolerance, work)? > 0.
                        && track_coverage(b, rs, oriented, separator_extent, tolerance, work)? > 0.
                    {
                        continue;
                    }
                }
            }
            chosen = Some(i);
            break;
        }
        let Some(i) = chosen else { break };
        let merged = combined(&tracks[i], &tracks[i + 1]);
        if merged.aliases.last().unwrap() - merged.aliases[0] > threshold {
            conflict = true;
            break;
        }
        tracks.splice(i..=i + 1, [merged]);
    }
    let mut unique = true;
    if let Some(rs) = &rules {
        for r in rs.iter().filter(|r| r.orientation == oriented) {
            let mut matches = 0;
            for t in &tracks {
                work.charge(t.aliases.len())?;
                matches += usize::from(
                    t.aliases
                        .iter()
                        .any(|a| (r.coordinate - a).abs() <= tolerance),
                );
            }
            if matches > 1 {
                unique = false;
                break;
            }
        }
    }
    Ok(
        json!({"tracks":tracks,"alias_conflict":conflict,"outer_collapse_count":outer,"unique":unique}),
    )
}
#[derive(Deserialize)]
struct Drawing {
    bbox: Value,
}
#[derive(Deserialize)]
struct Rectangle {
    bbox: Value,
    segment_count: i64,
    fill_visible: bool,
    stroke_visible: bool,
}
fn flag(v: &Value, key: &str, default: bool) -> Result<bool, Error> {
    v.get(key)
        .map_or(Ok(default), |v| v.as_bool().ok_or_else(invalid))
}
pub(super) fn number(v: &Value) -> Result<f64, Error> {
    v.as_f64()
        .filter(|n| n.is_finite() && n.abs() <= 1e9)
        .ok_or_else(invalid)
}
// 非四坐标/退化框沿旧normalize_bbox跳过；非有限/过大坐标属于资源输入合同。
pub(super) fn normalized(v: &Value) -> Result<Option<[f64; 4]>, Error> {
    let Some(a) = v.as_array().filter(|a| a.len() == 4) else {
        return Ok(None);
    };
    let b = [
        number(&a[0])?,
        number(&a[1])?,
        number(&a[2])?,
        number(&a[3])?,
    ];
    let b = [
        b[0].min(b[2]),
        b[1].min(b[3]),
        b[0].max(b[2]),
        b[1].max(b[3]),
    ];
    Ok((b[2] > b[0] && b[3] > b[1]).then_some(b))
}
fn intersection(b: [f64; 4], t: [f64; 4]) -> Option<[f64; 4]> {
    let a = [
        b[0].max(t[0]),
        b[1].max(t[1]),
        b[2].min(t[2]),
        b[3].min(t[3]),
    ];
    (a[2] > a[0] && a[3] > a[1]).then_some(a)
}
fn rotate(b: [f64; 4], w: f64, h: f64, angle: i64) -> [f64; 4] {
    match angle {
        90 => [b[1], w - b[2], b[3], w - b[0]],
        180 => [w - b[2], h - b[3], w - b[0], h - b[1]],
        270 => [h - b[3], b[0], h - b[1], b[2]],
        _ => b,
    }
}
pub(super) fn local(b: [f64; 4], t: [f64; 4], angle: i64, halo: f64) -> Option<[f64; 4]> {
    // 只有完全不相交时才采用halo；部分相交仍按原表框裁剪。
    let clipped = intersection(b, t).or_else(|| {
        if halo > 0. {
            intersection(b, [t[0] - halo, t[1] - halo, t[2] + halo, t[3] + halo])
        } else {
            None
        }
    })?;
    Some(rotate(
        [
            clipped[0] - t[0],
            clipped[1] - t[1],
            clipped[2] - t[0],
            clipped[3] - t[1],
        ],
        t[2] - t[0],
        t[3] - t[1],
        angle,
    ))
}
fn rule(orientation: &str, coordinate: f64, start: f64, end: f64) -> Rule {
    Rule {
        orientation: orientation.into(),
        coordinate,
        start,
        end,
    }
}
pub(super) fn cluster(
    mut values: Vec<f64>,
    tolerance: f64,
    work: &mut Work,
) -> Result<Vec<f64>, Error> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut groups: Vec<Vec<f64>> = Vec::new();
    for value in values {
        let append = if let Some(g) = groups.last() {
            work.charge(g.len())?;
            (value - sum(g) / g.len() as f64).abs() <= tolerance
        } else {
            false
        };
        if append {
            groups.last_mut().unwrap().push(value)
        } else {
            groups.push(vec![value])
        }
    }
    Ok(groups.iter().map(|g| sum(g) / g.len() as f64).collect())
}
fn coverage(intervals: Vec<(f64, f64)>, start: f64, end: f64) -> Result<f64, Error> {
    if end <= start {
        return Ok(0.);
    }
    rapiddoc_native::interval_coverage(intervals, start, end)
}
fn endpoints(
    rules: &[&Rule],
    extent: f64,
    tolerance: f64,
    work: &mut Work,
) -> Result<Vec<f64>, Error> {
    if rules.is_empty() {
        return Ok(Vec::new());
    }
    let max = rules
        .iter()
        .map(|r| r.end - r.start)
        .fold(f64::NEG_INFINITY, f64::max);
    let long: Vec<_> = rules
        .iter()
        .filter(|r| r.end - r.start >= 0.8 * max)
        .collect();
    let required = 2usize.max((0.5 * long.len() as f64).ceil() as usize);
    let positions = cluster(
        long.iter().flat_map(|r| [r.start, r.end]).collect(),
        tolerance,
        work,
    )?;
    work.charge(positions.len() * long.len())?;
    Ok(positions
        .into_iter()
        .filter(|p| {
            long.iter()
                .filter(|r| (r.start - p).abs().min((r.end - p).abs()) <= tolerance)
                .count()
                >= required
                || *p <= 2. * tolerance
                || extent - p <= 2. * tolerance
        })
        .collect())
}
fn infer(
    rules: &[Rule],
    tolerance: f64,
    width: f64,
    height: f64,
    prune: bool,
    work: &mut Work,
) -> Result<Value, Error> {
    let horizontal: Vec<_> = rules
        .iter()
        .filter(|r| r.orientation == "horizontal")
        .collect();
    let vertical: Vec<_> = rules
        .iter()
        .filter(|r| r.orientation == "vertical")
        .collect();
    let mut xv: Vec<_> = vertical.iter().map(|r| r.coordinate).collect();
    xv.extend(endpoints(&horizontal, width, tolerance, work)?);
    let x = cluster(xv, tolerance, work)?;
    let mut removed = Vec::new();
    let supported = if prune && x.len() >= 2 {
        let mut coordinates = Vec::new();
        for coordinate in cluster(
            horizontal.iter().map(|r| r.coordinate).collect(),
            tolerance,
            work,
        )? {
            work.charge(horizontal.len())?;
            let rs: Vec<_> = horizontal
                .iter()
                .filter(|r| (r.coordinate - coordinate).abs() <= tolerance)
                .collect();
            let intervals: Vec<_> = rs.iter().map(|r| (r.start, r.end)).collect();
            let mut band = false;
            for pair in x.windows(2) {
                work.charge(rs.len() * 3)?;
                let (left, right) = (pair[0], pair[1]);
                if coverage(intervals.clone(), left, right)? < 0.8 {
                    continue;
                }
                let touches_left = rs
                    .iter()
                    .any(|r| r.start <= left + tolerance && r.end >= left - tolerance);
                let touches_right = rs
                    .iter()
                    .any(|r| r.start <= right + tolerance && r.end >= right - tolerance);
                if touches_left && touches_right {
                    band = true;
                    break;
                }
            }
            if coverage(intervals, 0., width)? >= 0.8 || band {
                coordinates.push(coordinate)
            } else {
                removed.push(coordinate)
            }
        }
        work.charge(horizontal.len() * coordinates.len())?;
        horizontal
            .iter()
            .copied()
            .filter(|r| {
                coordinates
                    .iter()
                    .any(|c| (r.coordinate - c).abs() <= tolerance)
            })
            .collect::<Vec<_>>()
    } else {
        horizontal
    };
    let mut yv: Vec<_> = supported.iter().map(|r| r.coordinate).collect();
    yv.extend(endpoints(&vertical, height, tolerance, work)?);
    Ok(json!([x, cluster(yv, tolerance, work)?, removed]))
}
/// 输出片段、合并线、raw/supported轨道；5000原语旧硬门仍由后续完整候选应用。
pub fn prepare(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_with_work(v, &mut Work(0))
}
pub(super) fn prepare_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    let tolerance = number(&v["snap_tolerance"])?;
    let join = number(&v["join_gap"])?;
    let halo = number(&v["evidence_halo"])?;
    if tolerance < 0. || join < 0. || halo < 0. {
        return Err(invalid());
    }
    let input = &v["table_input"];
    let drawings: Vec<Drawing> =
        serde_json::from_value(input.get("drawing_lines").cloned().unwrap_or(json!([])))
            .map_err(|_| invalid())?;
    let rectangles: Vec<Rectangle> =
        serde_json::from_value(input.get("rectangles").cloned().unwrap_or(json!([])))
            .map_err(|_| invalid())?;
    if drawings.len() + rectangles.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let Some(t) = normalized(&input["table_bbox"])? else {
        return Ok(json!({"fragments":[],"rules":[],"inferred":[[],[],[]],"supported":[[],[],[]]}));
    };
    let a = input.get("angle").map_or(Ok(0.), number)?;
    let n = (a.trunc() as i64).rem_euclid(360);
    let angle = if [0, 90, 180, 270].contains(&n) { n } else { 0 };
    let (width, height) = if angle == 90 || angle == 270 {
        (t[3] - t[1], t[2] - t[0])
    } else {
        (t[2] - t[0], t[3] - t[1])
    };
    let mut fragments = Vec::new();
    if flag(v, "include_drawing", true)? {
        for d in drawings {
            let Some(b) = normalized(&d.bbox)?.and_then(|b| local(b, t, angle, halo)) else {
                continue;
            };
            let (w, h) = (b[2] - b[0], b[3] - b[1]);
            if w >= h && w >= (2. * tolerance).max(1.) {
                fragments.push(rule("horizontal", (b[1] + b[3]) / 2., b[0], b[2]))
            } else if h > w && h >= (2. * tolerance).max(1.) {
                fragments.push(rule("vertical", (b[0] + b[2]) / 2., b[1], b[3]))
            }
        }
    }
    if flag(v, "include_rectangles", true)? {
        for r in rectangles {
            if r.segment_count != 5 || !(r.fill_visible || r.stroke_visible) {
                continue;
            }
            let Some(b) = normalized(&r.bbox)?.and_then(|b| local(b, t, angle, 0.)) else {
                continue;
            };
            let (w, h) = (b[2] - b[0], b[3] - b[1]);
            if w <= tolerance || h <= tolerance || w * h / (width * height) >= 0.85 {
                continue;
            }
            fragments.extend([
                rule("horizontal", b[1], b[0], b[2]),
                rule("horizontal", b[3], b[0], b[2]),
                rule("vertical", b[0], b[1], b[3]),
                rule("vertical", b[2], b[1], b[3]),
            ]);
        }
    }
    if halo == 0. && (fragments.is_empty() || fragments.len() > 5000) {
        return Ok(
            json!({"fragments":fragments,"rules":[],"inferred":[[],[],[]],"supported":[[],[],[]]}),
        );
    }
    let mut coordinates = Vec::new();
    for (n, o) in [(0, "horizontal"), (1, "vertical")] {
        let c = cluster(
            fragments
                .iter()
                .filter(|r| r.orientation == o)
                .map(|r| r.coordinate)
                .collect(),
            tolerance,
            work,
        )?;
        work.charge(c.len() * fragments.len())?;
        coordinates.push((n, c));
    }
    let raw = fragments
        .iter()
        .map(|r| {
            (
                u8::from(r.orientation == "vertical"),
                r.coordinate,
                r.start,
                r.end,
            )
        })
        .collect();
    let rules: Vec<_> = rapiddoc_native::table_merge_rules(raw, coordinates, tolerance, join)?
        .into_iter()
        .map(|(o, c, s, e)| rule(if o == 1 { "vertical" } else { "horizontal" }, c, s, e))
        .collect();
    let inferred = infer(&rules, tolerance, width, height, false, work)?;
    let supported = infer(&rules, tolerance, width, height, true, work)?;
    Ok(json!({"fragments":fragments,"rules":rules,"inferred":inferred,"supported":supported}))
}
