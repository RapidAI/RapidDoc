//! 旧多列闭合网格证明；图形跨有字物理列/带且覆盖充分时才排除。
use super::path_graphics::{area, array, bbox, integer, num, overlap_smaller, union, Box4};
use super::table_native_candidate::sum;
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
fn center(b: &Box4, axis: usize) -> f64 {
    (b.n[axis] + b.n[axis + 2]) / 2.
}
pub(super) fn mean(v: &[f64], fmean: bool, work: &mut Work) -> Result<f64, Error> {
    work.charge(v.len() * 4 + 1)?;
    let mut s = BigRational::zero();
    for x in v {
        s += BigRational::from_float(*x).ok_or_else(invalid)?;
    }
    if fmean {
        Ok(s.to_f64().ok_or_else(invalid)? / v.len() as f64)
    } else {
        (s / BigRational::from_integer(v.len().into()))
            .to_f64()
            .ok_or_else(invalid)
    }
}
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k).map_or(Ok(&[]), |v| array(v, max))
}
pub(super) struct Rule {
    pub(super) b: Box4,
    pub(super) h: bool,
}
pub(super) struct Row {
    pub(super) ids: Vec<usize>,
    pub(super) cy: f64,
}
pub(super) struct Prepared {
    pub(super) rules: Vec<Rule>,
    pub(super) fragments: Vec<Box4>,
    pub(super) rows: Vec<Row>,
    pub(super) forms: Vec<Box4>,
    pub(super) components: Vec<Vec<usize>>,
    pub(super) em: f64,
}
fn position_groups(values: &[f64], tol: f64, work: &mut Work) -> Result<Vec<f64>, Error> {
    let mut v = values.to_vec();
    work.charge(v.len() * (log(v.len()) + 1))?;
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut groups: Vec<Vec<f64>> = Vec::new();
    for x in v {
        if let Some(g) = groups.last_mut() {
            if (x - mean(g, false, work)?).abs() <= tol {
                g.push(x);
                continue;
            }
        }
        groups.push(vec![x]);
    }
    groups.iter().map(|g| mean(g, false, work)).collect()
}
fn shared(a: &Box4, b: &Box4, rules: &[Rule], em: f64, work: &mut Work) -> Result<bool, Error> {
    let left = a.n[0].max(b.n[0]);
    let right = a.n[2].min(b.n[2]);
    let ep = 4f64.max(2. * em);
    if (right - left).max(0.) / (a.n[2] - a.n[0]).max(0.1).max((b.n[2] - b.n[0]).max(0.1)) < 0.9
        || (a.n[0] - b.n[0]).abs() > ep
        || (a.n[2] - b.n[2]).abs() > ep
    {
        return Ok(false);
    }
    work.charge(rules.len())?;
    let tol = 1f64.max(0.25 * em);
    let top = center(a, 1);
    let bottom = center(b, 1);
    let raw: Vec<_> = rules
        .iter()
        .filter(|r| {
            !r.h && r.b.n[1] <= top + tol
                && r.b.n[3] >= bottom - tol
                && left - tol <= center(&r.b, 0)
                && center(&r.b, 0) <= right + tol
        })
        .map(|r| center(&r.b, 0))
        .collect();
    let p = position_groups(&raw, tol, work)?;
    Ok(
        (p.iter().any(|p| (*p - left).abs() <= ep) && p.iter().any(|p| (*p - right).abs() <= ep))
            || p.iter()
                .filter(|p| left + tol < **p && **p < right - tol)
                .count()
                >= 2,
    )
}
pub(super) fn prepare(source: &Value, em: f64, work: &mut Work) -> Result<Prepared, Error> {
    source.as_object().ok_or_else(invalid)?;
    let size = array(&source["page_size"], 2)?;
    if size.len() != 2 {
        return Err(invalid());
    }
    size.iter().map(num).collect::<Result<Vec<_>, _>>()?;
    let mut rules = Vec::new();
    for r in optional(source, "drawing_lines", 5000)? {
        let b = bbox(&r["bbox"])?;
        let h = b.n[2] - b.n[0] >= b.n[3] - b.n[1];
        rules.push(Rule { b, h });
    }
    let forms = optional(source, "form_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let mut fragments = Vec::new();
    let mut native: BTreeMap<i64, usize> = BTreeMap::new();
    let mut seeds: Vec<Vec<usize>> = Vec::new();
    let mut geometric = Vec::new();
    for l in optional(source, "lines", 4096)? {
        if l.get("angle").map_or(Ok(0.), num)? != 0. {
            continue;
        }
        let i = fragments.len();
        fragments.push(bbox(&l["bbox"])?);
        if l["visual_row_id"].is_null() {
            geometric.push(vec![i])
        } else {
            let id = integer(l, "visual_row_id", 0)?;
            let next = seeds.len();
            let j = *native.entry(id).or_insert(next);
            if j == seeds.len() {
                seeds.push(Vec::new())
            }
            seeds[j].push(i);
        }
    }
    seeds.extend(geometric);
    let mut prepared = Vec::new();
    for ids in seeds {
        let cy = mean(
            &ids.iter()
                .map(|i| center(&fragments[*i], 1))
                .collect::<Vec<_>>(),
            true,
            work,
        )?;
        let left = ids
            .iter()
            .map(|i| fragments[*i].n[0])
            .reduce(f64::min)
            .unwrap();
        prepared.push((ids, cy, left));
    }
    work.charge(prepared.len() * (log(prepared.len()) + 1))?;
    prepared.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap()
            .then(a.2.partial_cmp(&b.2).unwrap())
    });
    let mut groups: Vec<(Vec<usize>, f64)> = Vec::new();
    for (ids, cy, _) in prepared {
        work.charge(groups.len() + ids.len())?;
        if let Some(j) = groups
            .iter()
            .position(|(_, c)| (cy - *c).abs() <= 2f64.max(0.5 * em))
        {
            groups[j].0.extend(ids);
            groups[j].1 = mean(
                &groups[j]
                    .0
                    .iter()
                    .map(|i| center(&fragments[*i], 1))
                    .collect::<Vec<_>>(),
                true,
                work,
            )?;
        } else {
            groups.push((ids, cy));
        }
    }
    let mut rows = Vec::new();
    for (mut ids, _) in groups {
        work.charge(ids.len() * (log(ids.len()) + 1))?;
        ids.sort_by(|a, b| fragments[*a].n[0].partial_cmp(&fragments[*b].n[0]).unwrap());
        let cy = sum(&ids
            .iter()
            .map(|i| center(&fragments[*i], 1))
            .collect::<Vec<_>>())
            / ids.len() as f64;
        rows.push(Row { ids, cy });
    }
    rows.sort_by(|a, b| a.cy.partial_cmp(&b.cy).unwrap());
    let mut hs: Vec<_> = rules
        .iter()
        .enumerate()
        .filter(|(_, r)| r.h && r.b.n[2] - r.b.n[0] >= 40f64.max(10. * em))
        .map(|(i, _)| i)
        .collect();
    work.charge(hs.len() * (log(hs.len()) + 1))?;
    hs.sort_by(|a, b| {
        let (a, b) = (&rules[*a].b, &rules[*b].b);
        a.n[0]
            .partial_cmp(&b.n[0])
            .unwrap()
            .then(a.n[2].partial_cmp(&b.n[2]).unwrap())
            .then(a.n[1].partial_cmp(&b.n[1]).unwrap())
    });
    let ep = 4f64.max(2. * em);
    let mut spans: Vec<Vec<usize>> = Vec::new();
    for i in hs {
        work.charge(spans.len() + 1)?;
        if let Some(j) = spans.iter().position(|g| {
            (rules[i].b.n[0] - rules[g[0]].b.n[0]).abs() <= ep
                && (rules[i].b.n[2] - rules[g[0]].b.n[2]).abs() <= ep
        }) {
            spans[j].push(i)
        } else {
            spans.push(vec![i])
        }
    }
    let mut components = Vec::new();
    for mut span in spans {
        work.charge(span.len() * (log(span.len()) + 1))?;
        span.sort_by(|a, b| {
            center(&rules[*a].b, 1)
                .partial_cmp(&center(&rules[*b].b, 1))
                .unwrap()
        });
        let mut unique: Vec<usize> = Vec::new();
        for i in span {
            work.charge(unique.len() + 1)?;
            if !unique
                .iter()
                .any(|j| (center(&rules[i].b, 1) - center(&rules[*j].b, 1)).abs() <= 1.)
            {
                unique.push(i)
            }
        }
        if unique.len() < 2 {
            continue;
        }
        let mut c: Vec<usize> = Vec::new();
        for i in unique {
            if !c.is_empty()
                && !shared(&rules[*c.last().unwrap()].b, &rules[i].b, &rules, em, work)?
            {
                if c.len() >= 2 {
                    components.push(c)
                }
                c = Vec::new();
            }
            c.push(i);
        }
        if c.len() >= 2 {
            components.push(c)
        }
    }
    Ok(Prepared {
        rules,
        fragments,
        rows,
        forms,
        components,
        em,
    })
}
pub(super) fn tracks(p: &Prepared, ids: &[usize], work: &mut Work) -> Result<Vec<f64>, Error> {
    let top = center(&p.rules[ids[0]].b, 1);
    let bottom = center(&p.rules[*ids.last().unwrap()].b, 1);
    let height = (bottom - top).max(0.1);
    let left = ids
        .iter()
        .map(|i| p.rules[*i].b.n[0])
        .reduce(f64::min)
        .unwrap();
    let right = ids
        .iter()
        .map(|i| p.rules[*i].b.n[2])
        .reduce(f64::max)
        .unwrap();
    let edge = 2f64.max(0.25 * p.em);
    work.charge(p.rules.len() + ids.len())?;
    let mut ts: Vec<_> = p
        .rules
        .iter()
        .filter(|r| !r.h)
        .filter_map(|r| {
            let x = center(&r.b, 0);
            let overlap = (r.b.n[3].min(bottom) - r.b.n[1].max(top)).max(0.);
            (overlap > 0. && left - edge <= x && x <= right + edge).then_some((
                x,
                top.max(r.b.n[1]),
                bottom.min(r.b.n[3]),
            ))
        })
        .collect();
    work.charge(ts.len() * (log(ts.len()) + 1))?;
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tol = 1f64.max(0.1 * p.em);
    let mut groups: Vec<Vec<(f64, f64, f64)>> = Vec::new();
    for t in ts {
        if let Some(g) = groups.last_mut() {
            if (t.0 - mean(&g.iter().map(|t| t.0).collect::<Vec<_>>(), false, work)?).abs() <= tol {
                g.push(t);
                continue;
            }
        }
        groups.push(vec![t]);
    }
    let mut out = Vec::new();
    for g in groups {
        let covered = intervals(g.iter().map(|t| (t.1, t.2)).collect(), work)?;
        if covered >= 0.9 * height {
            out.push(mean(
                &g.iter().map(|t| t.0).collect::<Vec<_>>(),
                false,
                work,
            )?);
        }
    }
    Ok(out)
}
fn intervals(mut v: Vec<(f64, f64)>, work: &mut Work) -> Result<f64, Error> {
    work.charge(v.len() * (log(v.len()) + 1))?;
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut spans: Vec<(f64, f64)> = Vec::new();
    for (a, b) in v {
        if let Some(last) = spans.last_mut() {
            if a <= last.1 {
                last.1 = last.1.max(b);
                continue;
            }
        }
        spans.push((a, b));
    }
    Ok(sum(&spans.iter().map(|(a, b)| b - a).collect::<Vec<_>>()))
}
fn occupied(
    p: &Prepared,
    rows: &[usize],
    pos: &[f64],
    region: Option<&Box4>,
    work: &mut Work,
) -> Result<usize, Error> {
    let mut cols = BTreeSet::new();
    for &i in rows {
        for &j in &p.rows[i].ids {
            let x = center(&p.fragments[j], 0);
            if region.is_some_and(|r| x < r.n[0] || x > r.n[2]) {
                continue;
            }
            work.charge(pos.len() + 1)?;
            let hits: Vec<_> = pos
                .windows(2)
                .enumerate()
                .filter(|(_, v)| v[0] < x && x < v[1])
                .map(|(i, _)| i)
                .collect();
            if hits.len() == 1 {
                cols.insert(hits[0]);
            }
        }
    }
    Ok(cols.len())
}
pub(super) fn grids(p: &Prepared, region: &Box4, work: &mut Work) -> Result<Vec<Box4>, Error> {
    let tolerance = 2f64.max(0.25 * p.em);
    let mut output = Vec::new();
    for component in &p.components {
        if component.len() < 4 {
            continue;
        }
        work.charge(component.len() + p.forms.len())?;
        let b = component
            .iter()
            .map(|i| p.rules[*i].b.clone())
            .reduce(|a, b| union(a, &b))
            .unwrap();
        if overlap_smaller(region, &b) < 0.5
            || p.forms.iter().any(|f| overlap_smaller(&b, f) >= 0.5)
        {
            continue;
        }
        let outer = tracks(p, component, work)?;
        if outer.len() < 2
            || (outer[0] - b.n[0])
                .abs()
                .max((outer[outer.len() - 1] - b.n[2]).abs())
                > tolerance
        {
            continue;
        }
        let mut proven: Vec<Vec<f64>> = Vec::new();
        let mut region_bands = 0;
        for band in component.windows(2) {
            let top = center(&p.rules[band[0]].b, 1);
            let bottom = center(&p.rules[band[1]].b, 1);
            let positions = tracks(p, band, work)?;
            work.charge(p.rows.len() + positions.len())?;
            let rows: Vec<_> = p
                .rows
                .iter()
                .enumerate()
                .filter(|(_, r)| top < r.cy && r.cy < bottom)
                .map(|(i, _)| i)
                .collect();
            if occupied(p, &rows, &positions, None, work)? >= 2 {
                if region.n[1] < (top + bottom) / 2.
                    && (top + bottom) / 2. < region.n[3]
                    && positions
                        .windows(2)
                        .filter(|v| {
                            region.n[0] < (v[0] + v[1]) / 2. && (v[0] + v[1]) / 2. < region.n[2]
                        })
                        .count()
                        >= 2
                    && occupied(p, &rows, &positions, Some(region), work)? >= 2
                {
                    region_bands += 1
                }
                proven.push(positions);
            }
        }
        if region_bands == 0 || (region_bands < 2 && region.n[3] - region.n[1] < 3. * p.em) {
            continue;
        }
        let mut same = false;
        for i in 0..proven.len() {
            for j in i + 1..proven.len() {
                work.charge(proven[i].len() + 1)?;
                if proven[i].len() == proven[j].len()
                    && proven[i]
                        .iter()
                        .zip(&proven[j])
                        .all(|(a, b)| (*a - *b).abs() <= tolerance)
                {
                    same = true;
                    break;
                }
            }
            if same {
                break;
            }
        }
        if same {
            output.push(b);
        }
    }
    let boxes: Vec<_> = output
        .iter()
        .map(|b| {
            [
                b.n[0].max(region.n[0]),
                b.n[1].max(region.n[1]),
                b.n[2].min(region.n[2]),
                b.n[3].min(region.n[3]),
            ]
        })
        .collect();
    let mut edges: Vec<_> = boxes.iter().flat_map(|b| [b[0], b[2]]).collect();
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
    edges.dedup();
    let mut terms = Vec::new();
    for e in edges.windows(2) {
        work.charge(boxes.len() + 1)?;
        let spans = boxes
            .iter()
            .filter(|b| b[0] <= e[0] && b[2] >= e[1])
            .map(|b| (b[1], b[3]))
            .collect();
        terms.push((e[1] - e[0]) * intervals(spans, work)?);
    }
    if terms.into_iter().fold(0., |total, x| total + x) >= 0.7 * area(region) {
        Ok(output)
    } else {
        Ok(Vec::new())
    }
}
pub fn inspect(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut work = Work(0);
    let em = num(&v["median_height"])?;
    let p = prepare(&v["source"], em, &mut work)?;
    let mut output = Vec::new();
    let mut proof_count = 0usize;
    for b in array(&v["regions"], 5000)? {
        let proofs = grids(&p, &bbox(b)?, &mut work)?;
        proof_count += proofs.len();
        if proof_count > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        output.push(proofs.into_iter().map(|b| b.raw).collect::<Vec<_>>());
    }
    let out = json!({"grid_bboxes":output});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
fn log(n: usize) -> usize {
    if n < 2 {
        1
    } else {
        usize::BITS as usize - (n - 1).leading_zeros() as usize + 1
    }
}
