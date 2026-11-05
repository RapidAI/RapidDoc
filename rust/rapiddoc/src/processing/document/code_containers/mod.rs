//! 旧代码容器完整入口：填充背景、横线清单和算法标题；实际字形/布局证明。
pub(super) mod text;
use super::path_graphics::{
    area, array, bbox, integer, num, overlap_first, overlap_smaller, union, Box4,
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{
    python_decimal,
    text_boundary::{python_alnum, python_space},
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k)
        .filter(|v| !v.is_null())
        .map_or(Ok(&[]), |v| array(v, max))
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
fn cx(b: &Box4) -> f64 {
    (b.n[0] + b.n[2]) / 2.
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) / 2.
}
fn combined<'a>(items: impl IntoIterator<Item = &'a Box4>) -> Result<Box4, Error> {
    let mut it = items.into_iter();
    let mut b = it.next().ok_or_else(invalid)?.clone();
    for other in it {
        b = union(b, other);
    }
    Ok(b)
}
struct Line<'a> {
    v: &'a Value,
    b: Box4,
    id: i64,
    angle: i64,
    h: f64,
    row: Option<i64>,
    control: bool,
    heading: bool,
    section: bool,
}
struct Rule {
    b: Box4,
    h: bool,
    v: bool,
}
struct Source<'a> {
    raw: &'a Value,
    lines: Vec<Line<'a>>,
    rules: Vec<Rule>,
    size: [f64; 2],
}
struct Candidate {
    b: Box4,
    angle: i64,
    ids: BTreeSet<i64>,
}
fn parse<'a>(s: &'a Value, w: &mut Work) -> Result<Source<'a>, Error> {
    let size = array(&s["page_size"], 2)?;
    if size.len() != 2 {
        return Err(invalid());
    }
    let size = [num(&size[0])?, num(&size[1])?];
    let mut lines = Vec::new();
    let mut ids = BTreeSet::new();
    let mut chars = 0;
    for l in optional(s, "lines", 4096)? {
        let text = l["text"].as_str().ok_or_else(invalid)?;
        if text.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        w.charge(text.len() * 4 + 1)?;
        let id = integer(l, "source_index", 0)?;
        if !ids.insert(id) {
            return Err(invalid());
        }
        chars += optional(l, "chars", 20000)?.len();
        if chars > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        let row = l
            .get("visual_row_id")
            .filter(|v| !v.is_null())
            .map(|v| v.as_i64().ok_or_else(invalid))
            .transpose()?;
        lines.push(Line {
            v: l,
            b: bbox(&l["bbox"])?,
            id,
            angle: integer(l, "angle", 0)?,
            h: l.get("effective_height").map_or(Ok(0.), num)?,
            row,
            control: text::control(text),
            heading: text::heading(text),
            section: text::section(text),
        });
    }
    let mut rules = Vec::new();
    for r in optional(s, "drawing_lines", 5000)? {
        w.charge(1)?;
        let ori = r["orientation"].as_str().ok_or_else(invalid)?;
        rules.push(Rule {
            b: bbox(&r["bbox"])?,
            h: ori == "horizontal",
            v: ori == "vertical",
        });
    }
    Ok(Source {
        raw: s,
        lines,
        rules,
        size,
    })
}
fn available<'a, 'b>(s: &'a Source<'b>, claimed: &BTreeSet<i64>) -> Vec<&'a Line<'b>> {
    s.lines
        .iter()
        .filter(|l| !claimed.contains(&l.id))
        .collect()
}
fn accept(mut raw: Vec<Candidate>, threshold: f64, w: &mut Work) -> Result<Vec<Candidate>, Error> {
    w.charge(raw.len() * 16 + 1)?;
    raw.sort_by_cached_key(|c| super::graphic_arbitration::area_key(&c.b));
    let mut out: Vec<Candidate> = Vec::new();
    for c in raw {
        w.charge(out.len() + 1)?;
        if !out.iter().any(|a| overlap_smaller(&c.b, &a.b) >= threshold) {
            out.push(c);
        }
    }
    out.sort_by(|a, b| {
        a.b.n[1]
            .partial_cmp(&b.b.n[1])
            .unwrap()
            .then(a.b.n[0].partial_cmp(&b.b.n[0]).unwrap())
    });
    Ok(out)
}
fn filled(
    s: &Source,
    excluded: &[Box4],
    claimed: &BTreeSet<i64>,
    w: &mut Work,
) -> Result<Vec<Candidate>, Error> {
    let area_page = s.size[0].max(0.) * s.size[1].max(0.);
    if area_page <= 0. {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let free = available(s, claimed);
    for p in optional(s.raw, "path_infos", 5000)? {
        let b = bbox(&p["bbox"])?;
        let fill = p
            .get("fill_visible")
            .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
        let color = optional(p, "fill_rgba", 4)?;
        let visible = if color.is_empty() {
            false
        } else {
            if color.len() != 4 {
                return Err(invalid());
            }
            let c = color
                .iter()
                .map(|v| v.as_i64().map(i128::from).ok_or_else(invalid))
                .collect::<Result<Vec<_>, _>>()?;
            c[3] > 0 && (255 - c[0]).max(255 - c[1]).max(255 - c[2]) >= 5
        };
        w.charge(excluded.len() + 1)?;
        if integer(p, "form_depth", 0)? != 0
            || !fill
            || integer(p, "segment_count", 0)? < 4
            || !visible
            || b.n[2] - b.n[0] < 0.5 * s.size[0]
            || b.n[3] - b.n[1] < 0.008 * s.size[1]
            || area(&b) >= 0.8 * area_page
            || excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5)
        {
            continue;
        }
        w.charge(free.len() + 1)?;
        let members: Vec<_> = free
            .iter()
            .copied()
            .filter(|l| overlap_first(&l.b, &b) >= 0.8)
            .collect();
        if members.is_empty() {
            continue;
        }
        let mut support = BTreeMap::new();
        for l in &members {
            let n = text::estimated(l, w)?;
            *support.entry(l.angle).or_insert(0.) += n;
        }
        let angle = support
            .iter()
            .fold(None, |best: Option<(i64, f64)>, (&angle, &value)| {
                if best.is_none_or(|(_, v)| value > v) {
                    Some((angle, value))
                } else {
                    best
                }
            })
            .unwrap()
            .0;
        let selected: Vec<_> = members
            .iter()
            .copied()
            .filter(|l| l.angle == angle)
            .collect();
        let total = super::table_native_candidate::sum(
            &members
                .iter()
                .map(|l| text::estimated(l, w))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let angle_support = super::table_native_candidate::sum(
            &selected
                .iter()
                .map(|l| text::estimated(l, w))
                .collect::<Result<Vec<_>, _>>()?,
        );
        if total <= 0. || angle_support / total < 0.8 {
            continue;
        }
        let (ratio, widths) = text::monospace(&selected, w)?;
        if ratio < 0.8 || !text::stable(&widths) {
            continue;
        }
        let cell = median(widths.iter().flatten().copied().collect());
        if !spatial_rows(&selected, &b, cell, w)? {
            continue;
        }
        out.push(Candidate {
            b,
            angle,
            ids: selected.iter().map(|l| l.id).collect(),
        });
        if out.len() > rapiddoc_core::resources::limit(5000) {
            return Err(limit());
        }
    }
    accept(out, 0.9, w)
}
fn spatial_rows(lines: &[&Line], b: &Box4, cell: f64, w: &mut Work) -> Result<bool, Error> {
    let mut ordered = lines.to_vec();
    ordered.sort_by(|a, b| {
        cy(&a.b)
            .partial_cmp(&cy(&b.b))
            .unwrap()
            .then(a.b.n[0].partial_cmp(&b.b.n[0]).unwrap())
    });
    let mut rows: Vec<Vec<&Line>> = Vec::new();
    for l in ordered {
        let mut at = None;
        for (i, row) in rows.iter().enumerate() {
            w.charge(row.len() + 1)?;
            if row.iter().any(|m| {
                m.b.n[3].min(l.b.n[3]) - m.b.n[1].max(l.b.n[1])
                    >= 0.5 * (m.b.n[3] - m.b.n[1]).min(l.b.n[3] - l.b.n[1])
            }) {
                at = Some(i);
                break;
            }
        }
        if let Some(i) = at {
            rows[i].push(l);
        } else {
            rows.push(vec![l]);
        }
    }
    if rows.is_empty() {
        return Ok(false);
    }
    let valid = lines
        .iter()
        .filter(|l| {
            let slot = (l.b.n[0] - b.n[0]) / cell.max(0.1);
            (slot - slot.round_ties_even()).abs() <= 0.35
        })
        .count();
    if valid as f64 / (lines.len() as f64) < 0.8 {
        return Ok(false);
    }
    if rows.len() == 1 {
        return Ok(true);
    }
    let mut tops: Vec<_> = rows
        .iter()
        .map(|r| r.iter().map(|l| l.b.n[1]).reduce(f64::min).unwrap())
        .collect();
    tops.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let deltas: Vec<_> = tops
        .windows(2)
        .filter_map(|a| (a[1] > a[0]).then_some(a[1] - a[0]))
        .collect();
    if deltas.is_empty() {
        return Ok(false);
    }
    let mut sorted = deltas.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = (0.6 * sorted.len() as f64).ceil().max(1.) as usize;
    let pitch = median(sorted[..n].to_vec());
    Ok(pitch > 0.
        && deltas
            .iter()
            .all(|d| (d / pitch - (d / pitch).round_ties_even()).abs() <= 0.35))
}
fn structured(members: &[&Line], b: &Box4, em: f64, w: &mut Work) -> Result<bool, Error> {
    if members.len() < 5 {
        return Ok(false);
    }
    let mut rows: Vec<(i64, Vec<&Line>)> = Vec::new();
    let mut fallback = 1_000_000;
    for l in members {
        let key = if let Some(id) = l.row {
            id
        } else {
            let id = fallback;
            fallback += 1;
            id
        };
        if let Some((_, r)) = rows.iter_mut().find(|(id, _)| *id == key) {
            r.push(l);
        } else {
            rows.push((key, vec![l]));
        }
    }
    if rows.len() < 5 {
        return Ok(false);
    }
    rows.sort_by(|a, b| {
        a.1.iter()
            .map(|l| cy(&l.b))
            .reduce(f64::min)
            .unwrap()
            .partial_cmp(&b.1.iter().map(|l| cy(&l.b)).reduce(f64::min).unwrap())
            .unwrap()
    });
    let centers: Vec<_> = rows
        .iter()
        .map(|(_, r)| median(r.iter().map(|l| cy(&l.b)).collect()))
        .collect();
    let gaps: Vec<_> = centers
        .windows(2)
        .filter_map(|a| (a[1] > a[0]).then_some(a[1] - a[0]))
        .collect();
    if gaps.is_empty() {
        return Ok(false);
    }
    let pitch = median(gaps.clone());
    if pitch <= 0.
        || gaps
            .iter()
            .filter(|&&g| 0.65 * pitch <= g && g <= 1.8 * pitch)
            .count() as f64
            / (gaps.len() as f64)
            < 0.75
        || rows.iter().filter(|(_, r)| r.len() >= 3).count() as f64 / (rows.len() as f64) >= 0.25
    {
        return Ok(false);
    }
    let mut lefts: Vec<_> = members.iter().map(|l| l.b.n[0]).collect();
    lefts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut clusters: Vec<Vec<f64>> = Vec::new();
    for x in lefts {
        w.charge(clusters.last().map_or(1, |a| a.len() + 1))?;
        if clusters
            .last()
            .is_none_or(|a| x - median(a.clone()) > 0.75 * em)
        {
            clusters.push(vec![x]);
        } else {
            clusters.last_mut().unwrap().push(x);
        }
    }
    let mut gutter = 0;
    for (_, row) in &mut rows {
        row.sort_by(|a, b| a.b.n[0].partial_cmp(&b.b.n[0]).unwrap());
        if row.len() >= 2
            && row[0].b.n[2] - row[0].b.n[0] <= 2. * em
            && row[1].b.n[0] - row[0].b.n[2] >= 0.5 * em
        {
            gutter += 1;
        }
    }
    let gutter = gutter as f64 / rows.len() as f64 >= 0.35;
    let hierarchy = clusters.len() >= 3 && clusters.iter().filter(|a| a.len() >= 2).count() >= 2;
    if !gutter && !hierarchy {
        return Ok(false);
    }
    if gutter {
        return Ok(true);
    }
    w.charge(
        members
            .iter()
            .map(|l| l.v["text"].as_str().unwrap().len() * 12 + 1)
            .sum(),
    )?;
    let syntax = members
        .iter()
        .filter(|l| text::syntax(l.v["text"].as_str().unwrap()))
        .count();
    let (ratio, widths) = text::monospace(members, w)?;
    if syntax < 2 && !(ratio >= 0.8 && text::stable(&widths)) {
        return Ok(false);
    }
    let occupied = members.iter().map(|l| l.b.n[2]).reduce(f64::max).unwrap()
        - members.iter().map(|l| l.b.n[0]).reduce(f64::min).unwrap();
    Ok(occupied <= 0.95 * (b.n[2] - b.n[0]).max(0.1))
}
fn ruled(
    s: &Source,
    excluded: &[Box4],
    claimed: &BTreeSet<i64>,
    w: &mut Work,
) -> Result<Vec<Candidate>, Error> {
    if s.size[0] <= 0. || s.size[1] <= 0. {
        return Ok(Vec::new());
    }
    let free: Vec<_> = available(s, claimed)
        .into_iter()
        .filter(|l| l.angle == 0)
        .collect();
    if free.len() < 5 {
        return Ok(Vec::new());
    }
    let em = median(
        free.iter()
            .map(|l| (if l.h != 0. { l.h } else { l.b.n[3] - l.b.n[1] }).max(0.1))
            .collect(),
    );
    let mut horizontal: Vec<_> = s
        .rules
        .iter()
        .filter(|r| r.h && r.b.n[2] - r.b.n[0] >= 0.22 * s.size[0])
        .collect();
    horizontal.sort_by(|a, b| {
        a.b.n[1]
            .partial_cmp(&b.b.n[1])
            .unwrap()
            .then(a.b.n[0].partial_cmp(&b.b.n[0]).unwrap())
    });
    let vertical: Vec<_> = s.rules.iter().filter(|r| r.v).collect();
    let mut out = Vec::new();
    let tol = 2f64.max(0.75 * em);
    for (i, top) in horizontal.iter().enumerate() {
        for bottom in &horizontal[i + 1..] {
            w.charge(1)?;
            if (top.b.n[0] - bottom.b.n[0]).abs() > tol || (top.b.n[2] - bottom.b.n[2]).abs() > tol
            {
                continue;
            }
            let b = union(top.b.clone(), &bottom.b);
            let h = b.n[3] - b.n[1];
            w.charge(excluded.len() + 1)?;
            if !(6. * em <= h && h <= 0.5 * s.size[1])
                || excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5)
            {
                continue;
            }
            w.charge(horizontal.len() + vertical.len() + free.len() + 1)?;
            if horizontal.iter().any(|r| {
                top.b.n[3] + 0.5 * em < r.b.n[1]
                    && r.b.n[1] < bottom.b.n[1] - 0.5 * em
                    && overlap_first(&r.b, &b) >= 0.8
                    && r.b.n[2] - r.b.n[0] >= 0.6 * (b.n[2] - b.n[0])
            }) {
                continue;
            }
            if vertical.iter().any(|r| {
                b.n[0] + em < cx(&r.b)
                    && cx(&r.b) < b.n[2] - em
                    && (r.b.n[3].min(b.n[3]) - r.b.n[1].max(b.n[1])).max(0.) / h >= 0.6
            }) {
                continue;
            }
            let members: Vec<_> = free
                .iter()
                .copied()
                .filter(|l| {
                    b.n[0] - 0.5 * em <= cx(&l.b)
                        && cx(&l.b) <= b.n[2] + 0.5 * em
                        && top.b.n[3] <= cy(&l.b)
                        && cy(&l.b) <= bottom.b.n[1]
                })
                .collect();
            if structured(&members, &b, em, w)? {
                out.push(Candidate {
                    b,
                    angle: 0,
                    ids: members.iter().map(|l| l.id).collect(),
                });
                if out.len() > rapiddoc_core::resources::limit(5000) {
                    return Err(limit());
                }
            }
        }
    }
    accept(out, 0.85, w)
}
fn algorithms(
    s: &Source,
    excluded: &[Box4],
    claimed: &BTreeSet<i64>,
    w: &mut Work,
) -> Result<Vec<Candidate>, Error> {
    let headings: Vec<_> = s
        .lines
        .iter()
        .filter(|l| l.angle == 0 && l.heading)
        .collect();
    if headings.is_empty() {
        return Ok(Vec::new());
    }
    let mut geometry = Vec::new();
    let mut heights = Vec::new();
    for l in &s.lines {
        w.charge(excluded.len() + 1)?;
        if l.angle != 0 || excluded.iter().any(|b| overlap_first(&l.b, b) >= 0.8) {
            continue;
        }
        let raw =
            l.v.get("ink_bbox")
                .filter(|v| !v.is_null())
                .unwrap_or(&l.v["bbox"]);
        let b = bbox(raw)?;
        let em = l.v.get("em_height").map_or(Ok(0.), num)?;
        let repaired =
            l.v.get("style_scale_repaired")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let h = (if repaired && em > 0. {
            em
        } else if l.h != 0. {
            l.h
        } else {
            b.n[3] - b.n[1]
        })
        .max(0.1);
        heights.push(h);
        geometry.push(json!([l.v, b.raw]));
    }
    if geometry.is_empty() {
        return Ok(Vec::new());
    }
    let em = median(heights);
    let lanes = super::table_text_styles::lanes::infer_layout(
        &json!({"line_geometry":geometry,"local_page_width":s.size[0],"median_height":em,"recalculate_intervals":false}),
        w,
    )?;
    let mut stable: Vec<_> = lanes
        .as_array()
        .unwrap()
        .iter()
        .filter(|l| l["is_span"] == false && l["line_indices"].as_array().unwrap().len() >= 3)
        .collect();
    stable.sort_by(|a, b| {
        a["left"]
            .as_f64()
            .unwrap()
            .partial_cmp(&b["left"].as_f64().unwrap())
            .unwrap()
    });
    let mut distinct: Vec<&Value> = Vec::new();
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
            distinct.push(l);
        }
    }
    let mut output = Vec::new();
    for heading in headings {
        if distinct.is_empty() {
            continue;
        }
        let hb = &heading.b;
        let substantial = distinct
            .iter()
            .filter(|l| {
                let (left, right) = (l["left"].as_f64().unwrap(), l["right"].as_f64().unwrap());
                (hb.n[2].min(right) - hb.n[0].max(left)).max(0.)
                    >= 0.25 * (hb.n[2] - hb.n[0]).min(right - left)
            })
            .count();
        let mut corridor = None;
        if substantial > 1 {
            corridor = Some((0., s.size[0]));
        } else {
            let mut left = 0.;
            for i in 0..distinct.len() {
                let right = if i + 1 == distinct.len() {
                    s.size[0]
                } else {
                    (distinct[i]["right"].as_f64().unwrap()
                        + distinct[i + 1]["left"].as_f64().unwrap())
                        / 2.
                };
                if left <= cx(hb) && cx(hb) <= right {
                    corridor = Some((left, right));
                    break;
                }
                left = right;
            }
        }
        let Some((left, right)) = corridor else {
            continue;
        };
        let em = heading.h.max(1.);
        let mut free: Vec<_> = s
            .lines
            .iter()
            .filter(|l| {
                !std::ptr::eq(*l, heading)
                    && l.angle == 0
                    && !claimed.contains(&l.id)
                    && left <= cx(&l.b)
                    && cx(&l.b) < right
                    && l.b.n[1] >= hb.n[1]
            })
            .collect();
        free.sort_by(|a, b| {
            a.b.n[1]
                .partial_cmp(&b.b.n[1])
                .unwrap()
                .then(a.b.n[0].partial_cmp(&b.b.n[0]).unwrap())
        });
        let Some(start) = free
            .iter()
            .position(|l| l.control && l.b.n[1] - hb.n[3] < 5. * em)
        else {
            continue;
        };
        let mut members = Vec::new();
        let mut bottom = free[start].b.n[3];
        for &l in &free[start..] {
            w.charge(excluded.len() + 1)?;
            if l.b.n[1] - bottom > 1.5 * em
                || l.section && !l.control
                || excluded.iter().any(|e| overlap_smaller(&l.b, e) >= 0.5)
            {
                break;
            }
            members.push(l);
            bottom = bottom.max(l.b.n[3]);
        }
        if members.iter().filter(|l| l.control).count() < 3 {
            continue;
        }
        let mut b = combined(members.iter().map(|l| &l.b))?;
        if b.n[3] - b.n[1] < 5. * em {
            continue;
        }
        let mut borders = Vec::new();
        for r in &s.rules {
            w.charge(1)?;
            if r.h
                && (r.b.n[0] - hb.n[0]).abs() < em
                && r.b.n[2] >= b.n[2] - em
                && r.b.n[2] <= right + em
                && (hb.n[3] <= r.b.n[1] && r.b.n[1] <= b.n[1] && b.n[1] - r.b.n[3] <= 1.5 * em
                    || b.n[3] <= r.b.n[1] && r.b.n[1] <= b.n[3] + 1.5 * em)
            {
                borders.push(&r.b);
            }
        }
        for border in borders {
            b = union(b, border);
        }
        output.push(Candidate {
            b,
            angle: 0,
            ids: members.iter().map(|l| l.id).collect(),
        });
    }
    Ok(output)
}
pub fn build(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    build_with_work(v, &mut Work(0))
}
pub(crate) fn build_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    build_source(&v["source"], v, w)
}
pub(crate) fn build_source(source: &Value, v: &Value, w: &mut Work) -> Result<Value, Error> {
    let s = parse(source, w)?;
    let excluded = optional(v, "excluded_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let claimed = optional(v, "claimed_line_indices", 20000)?
        .iter()
        .map(|v| v.as_i64().ok_or_else(invalid))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let kind = v.get("kind").and_then(Value::as_str).unwrap_or("filled");
    let mut cs = match kind {
        "filled" => filled(&s, &excluded, &claimed, w)?,
        "ruled" => ruled(&s, &excluded, &claimed, w)?,
        _ => return Err(invalid()),
    };
    if kind == "ruled" {
        let mut barriers = excluded;
        barriers.extend(cs.iter().map(|c| c.b.clone()));
        if barriers.len() > rapiddoc_core::resources::limit(5000) {
            return Err(limit());
        }
        cs.extend(algorithms(&s, &barriers, &claimed, w)?);
    }
    if cs.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let by_id: BTreeMap<_, _> = s.lines.iter().map(|l| (l.id, l)).collect();
    let mut member_count = 0;
    let mut blocks = Vec::new();
    let mut owned = BTreeSet::new();
    for c in cs {
        member_count += c.ids.len();
        if member_count > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        w.charge(c.ids.len() * 12 + 1)?;
        let members: Vec<_> = c
            .ids
            .iter()
            .filter_map(|id| by_id.get(id).copied())
            .collect();
        if members.is_empty() {
            continue;
        }
        let content = text::materialize(&members, &c, w)?;
        if content.is_empty() {
            continue;
        }
        blocks.push(json!({"type":"code","bbox":c.b.raw,"angle":c.angle,"content":content}));
        owned.extend(c.ids);
    }
    blocks.sort_by(|a, b| {
        a["bbox"][1]
            .as_f64()
            .unwrap()
            .partial_cmp(&b["bbox"][1].as_f64().unwrap())
            .unwrap()
            .then(
                a["bbox"][0]
                    .as_f64()
                    .unwrap()
                    .partial_cmp(&b["bbox"][0].as_f64().unwrap())
                    .unwrap(),
            )
    });
    let out = json!({"code_blocks":blocks,"claimed_line_indices":owned});
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
    fn old_complete_code_builders() {
        let directory =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build/migration");
        for name in [
            "native-code-containers",
            "native-code-container-boundaries",
            "native-code-integer-boundaries",
        ] {
            let root = directory.join(format!("{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                assert_eq!(
                    build(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"])),
                    c["expected"],
                    "{}",
                    c["label"]
                );
            }
        }
    }
    #[test]
    fn shared_work_and_recovery() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-code-containers.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap().iter().step_by(13) {
            let q = &c["request"];
            let mut w = Work(0);
            assert!(build_with_work(q, &mut w).is_ok());
            assert!(build_with_work(q, &mut Work(10_000_000 - w.0 + 1)).is_err());
            assert!(build(q).is_ok());
        }
    }
}
