//! 旧原生矢量公式完整入口；正文栏、路径网格连通、编号归属及装饰图形判定。
mod text;
pub(super) fn publication(s: &str) -> bool {
    text::publication(s)
}
pub(super) fn english_words(s: &str) -> usize {
    text::english_words(s)
}
pub(super) fn formula_math(s: &str) -> bool {
    text::math(s)
}
pub(super) fn formula_number(s: &str) -> bool {
    text::number(s)
}
pub(super) fn formula_residue(s: &str) -> String {
    text::residue(s)
}
pub(super) fn formula_sentence(s: &str) -> bool {
    text::prose(s)
}
use super::{
    page_containers::{clip, coerce, combined, height, median, optional, size},
    path_graphics::{array, bbox, integer, num, overlap_first, overlap_smaller, union, Box4},
    table_text_styles::lanes,
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone)]
struct Path {
    b: Box4,
    source: i64,
    segments: i64,
    fill: bool,
    stroke: bool,
    color: Option<[f64; 3]>,
}
struct Lane {
    left: f64,
    right: f64,
    lines: Vec<usize>,
}
struct Component {
    lane: usize,
    paths: Vec<usize>,
    b: Box4,
}
#[derive(Clone)]
struct Candidate {
    lane: usize,
    b: Box4,
    sources: BTreeSet<i64>,
    number: bool,
}
fn width(b: &Box4) -> f64 {
    b.n[2] - b.n[0]
}
fn high(b: &Box4) -> f64 {
    b.n[3] - b.n[1]
}
fn center(b: &Box4, a: usize) -> f64 {
    (b.n[a] + b.n[a + 2]) * 0.5
}
fn axis(a: &Box4, b: &Box4, k: usize) -> f64 {
    let d = (a.n[k + 2] - a.n[k]).min(b.n[k + 2] - b.n[k]);
    if d > 0. {
        (a.n[k + 2].min(b.n[k + 2]) - a.n[k].max(b.n[k])).max(0.) / d
    } else {
        0.
    }
}
fn distance(a: &Box4, b: &Box4) -> f64 {
    (a.n[0] - b.n[2])
        .max(b.n[0] - a.n[2])
        .max(0.)
        .hypot((a.n[1] - b.n[3]).max(b.n[1] - a.n[3]).max(0.))
}
fn intersects(a: [f64; 4], b: [f64; 4]) -> bool {
    a[2].min(b[2]) > a[0].max(b[0]) && a[3].min(b[3]) > a[1].max(b[1])
}
fn root(parents: &mut [usize], mut i: usize) -> usize {
    while parents[i] != i {
        parents[i] = parents[parents[i]];
        i = parents[i]
    }
    i
}
fn connect(
    members: &[usize],
    paths: &[Path],
    lane: usize,
    em: f64,
    w: &mut Work,
) -> Result<Vec<Component>, Error> {
    let mut ordered = members.to_vec();
    ordered.sort_by_key(|&i| paths[i].source);
    let mut parents = (0..ordered.len()).collect::<Vec<_>>();
    let mut expanded = Vec::new();
    let mut grid = BTreeMap::<(i64, i64), Vec<usize>>::new();
    let mut seen = BTreeSet::new();
    let cell = em.max(1.);
    for (i, &p) in ordered.iter().enumerate() {
        let b = paths[p].b.n;
        let margin = 0.5 * em;
        let b = [b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin];
        expanded.push(b);
        let cells = [
            (b[0] / cell).floor() as i64,
            (b[1] / cell).floor() as i64,
            (b[2] / cell).floor() as i64,
            (b[3] / cell).floor() as i64,
        ];
        let count = (cells[2] - cells[0] + 1)
            .checked_mul(cells[3] - cells[1] + 1)
            .ok_or_else(limit)?;
        w.charge(usize::try_from(count).map_err(|_| limit())?)?;
        for x in cells[0]..=cells[2] {
            for y in cells[1]..=cells[3] {
                let members = grid.entry((x, y)).or_default();
                w.charge(members.len() + 1)?;
                for &j in members.iter() {
                    if seen.insert((j, i)) && intersects(b, expanded[j]) {
                        let a = root(&mut parents, i);
                        let b = root(&mut parents, j);
                        if a != b {
                            parents[b] = a
                        }
                    }
                }
                members.push(i)
            }
        }
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, &p) in ordered.iter().enumerate() {
        let r = root(&mut parents, i);
        if let Some((_, g)) = groups.iter_mut().find(|(k, _)| *k == r) {
            g.push(p)
        } else {
            groups.push((r, vec![p]))
        }
    }
    groups
        .into_iter()
        .map(|(_, ps)| {
            let b = combined(ps.iter().map(|&i| &paths[i].b))?;
            Ok(Component { lane, paths: ps, b })
        })
        .collect()
}
fn components(
    paths: &[Path],
    lanes: &[Lane],
    em: f64,
    w: &mut Work,
) -> Result<Vec<Component>, Error> {
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, p) in paths.iter().enumerate() {
        w.charge(lanes.len() + 1)?;
        let mut matches = Vec::new();
        for (j, l) in lanes.iter().enumerate() {
            let x = center(&p.b, 0);
            if x < l.left - 0.75 * em || x > l.right + 0.75 * em {
                continue;
            }
            let coverage =
                (p.b.n[2].min(l.right) - p.b.n[0].max(l.left)).max(0.) / width(&p.b).max(0.1);
            matches.push((-coverage, (x - (l.left + l.right) / 2.).abs(), j))
        }
        matches.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let Some(&(_, _, j)) = matches.first() else {
            continue;
        };
        let l = &lanes[j];
        if !(width(&p.b) <= 3. * em && high(&p.b) <= 3. * em
            || high(&p.b) <= 0.2 * em && width(&p.b) <= (l.right - l.left).max(0.1) + em)
        {
            continue;
        }
        if let Some((_, g)) = groups.iter_mut().find(|(k, _)| *k == j) {
            g.push(i)
        } else {
            groups.push((j, vec![i]))
        }
    }
    let mut out = Vec::new();
    for (lane, g) in groups {
        out.extend(connect(&g, paths, lane, em, w)?)
    }
    out.sort_by(|a, b| {
        a.b.n[1]
            .total_cmp(&b.b.n[1])
            .then(a.b.n[0].total_cmp(&b.b.n[0]))
            .then(paths[a.paths[0]].source.cmp(&paths[b.paths[0]].source))
    });
    Ok(out)
}
fn inline(
    c: &Component,
    paths: &[Path],
    lane: &Lane,
    lines: &[Value],
    em: f64,
    containers: &[Box4],
    w: &mut Work,
) -> Result<BTreeSet<i64>, Error> {
    let b = &c.b;
    w.charge(containers.len() + c.paths.len() + 1)?;
    if !(0.45 * em <= high(b) && high(b) <= 1.6 * em)
        || width(b) > 0.45 * (lane.right - lane.left)
        || containers.iter().any(|a| overlap_smaller(b, a) >= 0.1)
        || c.paths.iter().any(|&i| {
            paths[i].color.is_some_and(|c| {
                c.into_iter().fold(f64::NEG_INFINITY, f64::max)
                    - c.into_iter().fold(f64::INFINITY, f64::min)
                    > 20.
            })
        })
    {
        return Ok(BTreeSet::new());
    }
    let complex = c
        .paths
        .iter()
        .copied()
        .filter(|&i| paths[i].segments >= 20)
        .collect::<Vec<_>>();
    let mut scripted = false;
    for &i in &complex {
        for &j in &complex {
            w.charge(1)?;
            if i == j {
                continue;
            }
            let a = &paths[i].b;
            let b = &paths[j].b;
            let gap = b.n[0] - a.n[2];
            scripted |= high(a) >= 1.4 * high(b)
                && (center(a, 1) - center(b, 1)).abs() >= 0.1 * em
                && gap >= -0.2 * em
                && gap <= 0.5 * em
                || 0.3 * em <= high(a).min(high(b))
                    && high(a).max(high(b)) <= 0.6 * em
                    && (a.n[0] - b.n[0]).abs() <= 0.15 * em
                    && (center(a, 1) - center(b, 1)).abs() >= 0.5 * em
        }
    }
    if !(complex.len() >= 8 && width(b) >= 3. * em
        || complex.len() >= 2 && width(b) <= 2. * em && scripted)
    {
        return Ok(BTreeSet::new());
    }
    let mut neighbors = Vec::new();
    for &i in &lane.lines {
        w.charge(1)?;
        let l = &lines[i];
        let a = bbox(&l["bbox"])?;
        let ratio = height(l, &a)? / em;
        if integer(l, "angle", 0)? == 0
            && l["semantic_type"].is_null()
            && l["caption_start"] != true
            && !l["font_signature"].is_null()
            && (0.85..=1.15).contains(&ratio)
            && axis(&a, b, 1) >= 0.5
            && overlap_smaller(&a, b) < 0.1
        {
            neighbors.push((i, a))
        }
    }
    let left = neighbors
        .iter()
        .filter(|(_, a)| b.n[0] - a.n[2] >= 0. && b.n[0] - a.n[2] <= 0.8 * em)
        .collect::<Vec<_>>();
    let right = neighbors
        .iter()
        .filter(|(_, a)| a.n[0] - b.n[2] >= 0. && a.n[0] - b.n[2] <= 0.8 * em)
        .collect::<Vec<_>>();
    let hosts = left.iter().chain(right.iter()).copied().collect::<Vec<_>>();
    let text = hosts
        .iter()
        .map(|(i, _)| lines[*i]["text"].as_str().unwrap_or(""))
        .collect::<Vec<_>>()
        .join(" ");
    if hosts.is_empty() || left.len() > 1 || right.len() > 1 || text::english_words(&text) < 4 {
        return Ok(BTreeSet::new());
    }
    if left.is_empty()
        && (right.is_empty()
            || !lines[right[0].0]["text"]
                .as_str()
                .unwrap_or("")
                .trim_start_matches(rapiddoc_native::text_boundary::python_space)
                .starts_with([',', '.', ';', ':', ')'])
            || neighbors.iter().any(|(_, a)| a.n[2] < b.n[0]))
    {
        return Ok(BTreeSet::new());
    }
    if right.is_empty()
        && (left.is_empty()
            || width(&left[0].1) < 0.65 * (lane.right - lane.left)
            || b.n[2] < lane.right - em)
    {
        return Ok(BTreeSet::new());
    }
    hosts
        .into_iter()
        .map(|(i, _)| integer(&lines[*i], "source_index", 0))
        .collect()
}
fn math(paths: &[&Path], b: &Box4, lines: &[Value], em: f64, w: &mut Work) -> Result<bool, Error> {
    for l in lines {
        w.charge(1)?;
        let text = l["text"].as_str().ok_or_else(invalid)?;
        if (text::standalone(text) || text::math(text) && !text::prose(text))
            && distance(&bbox(&l["bbox"])?, b) <= 2. * em
        {
            return Ok(true);
        }
    }
    for p in paths {
        if high(&p.b) > 0.2 * em || width(&p.b) < 0.4 * em {
            continue;
        }
        let mut above = false;
        let mut below = false;
        for q in paths {
            w.charge(1)?;
            if (center(&q.b, 0) - center(&p.b, 0)).abs() <= 0.25 * width(&p.b) {
                let a = p.b.n[1] - q.b.n[3];
                let d = q.b.n[1] - p.b.n[3];
                above |= a >= 0. && a <= em;
                below |= d >= 0. && d <= em
            }
        }
        if above && below {
            return Ok(true);
        }
    }
    Ok(false)
}
fn core(
    c: &Component,
    paths: &[Path],
    lane: &Lane,
    lines: &[Value],
    em: f64,
    sz: [f64; 2],
    containers: &[Box4],
    w: &mut Work,
) -> Result<bool, Error> {
    let ps = c.paths.iter().map(|&i| &paths[i]).collect::<Vec<_>>();
    let complex = ps.iter().filter(|p| p.segments >= 8).count();
    let n = ps.len();
    if n < 5
        || complex < 5 && !(complex >= 4 && math(&ps, &c.b, &[], em, w)?)
        || complex as f64 / (n as f64) < 0.5
    {
        return Ok(false);
    }
    let b = &c.b;
    if !(width(b) >= 2.5 * em
        && high(b) >= 0.6 * em
        && high(b) <= 8. * em
        && width(b) >= 1.4 * high(b))
        || b.n[3] <= 0.05 * sz[1]
        || b.n[1] >= 0.95 * sz[1]
    {
        return Ok(false);
    }
    w.charge(containers.len() + 1)?;
    if containers.iter().any(|a| overlap_smaller(b, a) >= 0.5) {
        return Ok(false);
    }
    for &i in &lane.lines {
        w.charge(1)?;
        let l = &lines[i];
        if text::standalone(l["text"].as_str().ok_or_else(invalid)?) {
            continue;
        }
        let a = bbox(&l["bbox"])?;
        let gap = (b.n[0] - a.n[2]).max(a.n[0] - b.n[2]).max(0.);
        if overlap_first(b, &a) >= 0.2 || axis(b, &a, 1) >= 0.5 && gap <= em {
            return Ok(false);
        }
    }
    Ok(true)
}
fn matches(
    b: &Box4,
    lane: usize,
    candidates: &[Candidate],
    em: f64,
    left: bool,
    w: &mut Work,
) -> Result<Vec<(f64, f64, f64, usize)>, Error> {
    let mut out = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        w.charge(1)?;
        if c.number || c.lane != lane {
            continue;
        }
        let overlap = axis(&c.b, b, 1);
        let is_left = left && b.n[2] <= c.b.n[0] && c.b.n[0] - b.n[2] <= 3. * em;
        if b.n[0] < c.b.n[2] && !is_left || overlap < 0.6 {
            continue;
        }
        let gap = if is_left {
            c.b.n[0] - b.n[2]
        } else {
            (b.n[0] - c.b.n[2]).max(0.)
        };
        out.push((
            -overlap,
            (center(&c.b, 1) - center(b, 1)).abs(),
            gap / em.max(0.1),
            i,
        ))
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(out)
}
fn right_number(c: &Component, paths: &[Path], l: &Lane, em: f64) -> bool {
    let b = &c.b;
    (3..=6).contains(&c.paths.len())
        && c.paths.iter().all(|&i| paths[i].segments >= 8)
        && width(b) >= 0.5 * em
        && width(b) <= 2. * em
        && high(b) >= 0.6 * em
        && high(b) <= 1.4 * em
        && (l.right - b.n[2]).abs() <= 1.5 * em
}
fn local_number(c: &Component, paths: &[Path], em: f64) -> bool {
    let mut ps = c.paths.iter().map(|&i| &paths[i]).collect::<Vec<_>>();
    ps.sort_by(|a, b| a.b.n[0].total_cmp(&b.b.n[0]));
    if !(3..=6).contains(&ps.len()) || ps.iter().any(|p| p.segments < 8) {
        return false;
    }
    let a = &ps[0].b;
    let b = &ps[ps.len() - 1].b;
    let c = &c.b;
    width(c) >= 0.5 * em
        && width(c) <= 2. * em
        && high(c) >= 0.6 * em
        && high(c) <= 1.5 * em
        && [a, b].iter().all(|b| high(b) >= 2.5 * width(b))
        && (a.n[1] - b.n[1]).abs() <= 0.15 * em
        && (a.n[3] - b.n[3]).abs() <= 0.15 * em
        && (width(a) - width(b)).abs() <= 0.15 * em
}
fn attach(
    candidates: &mut Vec<Candidate>,
    components: &[Component],
    paths: &[Path],
    lanes: &[Lane],
    lines: &[Value],
    em: f64,
    prior: &BTreeSet<i64>,
    w: &mut Work,
) -> Result<BTreeSet<i64>, Error> {
    let mut used = candidates
        .iter()
        .flat_map(|c| c.sources.iter().copied())
        .collect::<BTreeSet<_>>();
    for c in components {
        let ids = c
            .paths
            .iter()
            .map(|&i| paths[i].source)
            .collect::<BTreeSet<_>>();
        if !ids.is_disjoint(&used) || c.paths.iter().any(|&i| high(&paths[i].b) > 0.2 * em) {
            continue;
        }
        let mut owners = Vec::new();
        for (i, a) in candidates.iter().enumerate() {
            w.charge(1)?;
            let d = distance(&a.b, &c.b);
            if a.lane == c.lane && d <= 0.5 * em && axis(&a.b, &c.b, 0) >= 0.5 {
                owners.push((d, (center(&a.b, 1) - center(&c.b, 1)).abs(), i))
            }
        }
        owners.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if let Some(&(_, _, i)) = owners.first() {
            candidates[i].b = union(candidates[i].b.clone(), &c.b);
            candidates[i].sources.extend(ids.iter());
            used.extend(ids)
        }
    }
    for c in components {
        let ids = c
            .paths
            .iter()
            .map(|&i| paths[i].source)
            .collect::<BTreeSet<_>>();
        let local = local_number(c, paths, em);
        let right = right_number(c, paths, &lanes[c.lane], em);
        if !ids.is_disjoint(&used) || !(local || right) {
            continue;
        }
        let mut owners = matches(&c.b, c.lane, candidates, em, false, w)?;
        if local && !right {
            owners.retain(|m| m.2 <= 4.)
        }
        if let Some(&(_, _, _, i)) = owners.first() {
            candidates[i].b = union(candidates[i].b.clone(), &c.b);
            candidates[i].sources.extend(ids.iter());
            candidates[i].number = true;
            used.extend(ids)
        }
    }
    let mut claimed = BTreeSet::new();
    for (k, lane) in lanes.iter().enumerate() {
        let mut ids = lane.lines.clone();
        ids.sort_by(|&i, &j| {
            let a = &lines[i]["bbox"];
            let b = &lines[j]["bbox"];
            a[1].as_f64()
                .unwrap()
                .total_cmp(&b[1].as_f64().unwrap())
                .then(a[0].as_f64().unwrap().total_cmp(&b[0].as_f64().unwrap()))
        });
        for i in ids {
            let l = &lines[i];
            let id = integer(l, "source_index", 0)?;
            if prior.contains(&id) || !text::number(l["text"].as_str().ok_or_else(invalid)?) {
                continue;
            }
            let b = bbox(&l["bbox"])?;
            let right = width(&b) >= 0.5 * em
                && width(&b) <= 2. * em
                && high(&b) >= 0.6 * em
                && high(&b) <= 1.4 * em
                && (lane.right - b.n[2]).abs() <= 1.5 * em;
            let left = width(&b) >= 0.5 * em
                && width(&b) <= 4. * em
                && high(&b) >= 0.6 * em
                && high(&b) <= 1.4 * em
                && (lane.left - b.n[0]).abs() <= 1.5 * em
                && l["semantic_type"].is_null();
            if !(right || left) {
                continue;
            }
            let mut owners = matches(&b, k, candidates, em, left, w)?;
            if left && !right {
                owners.retain(|m| candidates[m.3].b.n[0] >= b.n[2]);
                if owners.len() != 1 {
                    continue;
                }
            }
            if let Some(&(_, _, _, j)) = owners.first() {
                candidates[j].b = union(candidates[j].b.clone(), &b);
                candidates[j].number = true;
                claimed.insert(id);
            }
        }
    }
    Ok(claimed)
}
fn join(
    candidates: &mut Vec<Candidate>,
    components: &[Component],
    paths: &[Path],
    lines: &[Value],
    em: f64,
    w: &mut Work,
) -> Result<(), Error> {
    // Stable identity survives deletion of a following candidate, as in list(candidates).
    let mut active = vec![true; candidates.len()];
    for i in 0..candidates.len() {
        if !active[i] || candidates[i].number {
            continue;
        }
        let b = candidates[i].b.clone();
        if width(&b) < 8. * em || width(&b) > 15. * em {
            continue;
        }
        let mut glyphs = Vec::new();
        for c in components {
            for &p in &c.paths {
                w.charge(1)?;
                if candidates[i].sources.contains(&paths[p].source) {
                    glyphs.push(p)
                }
            }
        }
        let Some(&first) = glyphs.first() else {
            continue;
        };
        let mut last = first;
        for &p in &glyphs[1..] {
            if paths[p].b.n[2] > paths[last].b.n[2] {
                last = p;
            }
        }
        let last = &paths[last].b;
        if width(last) < 0.65 * em
            || width(last) > 1.2 * em
            || high(last) < 0.12 * em
            || high(last) > 0.4 * em
            || width(last) < 2.5 * high(last)
            || (center(last, 1) - center(&b, 1)).abs() > 0.2 * em
        {
            continue;
        }
        let mut owners = Vec::new();
        for (j, c) in candidates.iter().enumerate() {
            w.charge(1)?;
            if j == i
                || !active[j]
                || c.number
                || c.lane != candidates[i].lane
                || (c.b.n[0] - b.n[0]).abs() > 0.5 * em
                || c.b.n[1] - b.n[3] < 0.
                || c.b.n[1] - b.n[3] > 1.3 * em
                || width(&c.b) < 3. * width(&b)
            {
                continue;
            }
            let mut blocked = false;
            for l in lines {
                w.charge(1)?;
                let a = bbox(&l["bbox"])?;
                if b.n[3] < center(&a, 1) && center(&a, 1) < c.b.n[1] && axis(&a, &c.b, 0) >= 0.2 {
                    blocked = true;
                    break;
                }
            }
            if !blocked {
                owners.push(j)
            }
        }
        if owners.len() == 1 {
            let j = owners[0];
            candidates[i].b = union(b, &candidates[j].b);
            let ids = candidates[j].sources.clone();
            candidates[i].sources.extend(ids);
            active[j] = false
        }
    }
    let mut i = 0;
    candidates.retain(|_| {
        let keep = active[i];
        i += 1;
        keep
    });
    Ok(())
}
fn decoration(paths: &[&Path], w: &mut Work) -> Result<bool, Error> {
    let mut hs = paths
        .iter()
        .map(|p| high(&p.b))
        .filter(|&h| h > 0.)
        .collect::<Vec<_>>();
    if hs.len() < 5 {
        return Ok(false);
    }
    let typical = median(hs.clone());
    let mut ordered = paths.to_vec();
    ordered.sort_by(|a, b| center(&a.b, 1).total_cmp(&center(&b.b, 1)));
    let mut rows: Vec<Vec<&Path>> = Vec::new();
    for p in ordered {
        let same = if let Some(row) = rows.last() {
            w.charge(row.len())?;
            (center(&p.b, 1) - median(row.iter().map(|p| center(&p.b, 1)).collect())).abs()
                <= 0.2 * typical
        } else {
            false
        };
        if same {
            rows.last_mut().unwrap().push(p)
        } else {
            rows.push(vec![p])
        }
    }
    hs.sort_by(f64::total_cmp);
    let wordmark = (2..=3).contains(&rows.len())
        && rows.iter().all(|r| r.len() >= 4)
        && hs[hs.len() - 1] <= 1.5 * hs[0];
    let small = hs[hs.len() / 4];
    let large = hs[3 * hs.len() / 4];
    let mut overlaps = 0;
    for (i, a) in paths.iter().enumerate() {
        for b in &paths[i + 1..] {
            w.charge(1)?;
            if overlap_smaller(&a.b, &b.b) >= 0.25 {
                overlaps += 1
            }
        }
    }
    let large_paths = paths
        .iter()
        .filter(|p| high(&p.b) >= large)
        .collect::<Vec<_>>();
    let irregular = large_paths.len() >= 3
        && large_paths
            .iter()
            .map(|p| p.b.n[3])
            .fold(f64::NEG_INFINITY, f64::max)
            - large_paths
                .iter()
                .map(|p| p.b.n[3])
                .fold(f64::INFINITY, f64::min)
            >= 0.6 * large;
    Ok(wordmark || small > 0. && large >= 2. * small && (overlaps >= 2 || irregular))
}
fn signature(paths: &[&Path], b: &Box4) -> Value {
    let width = width(b).max(0.1);
    let height = high(b).max(0.1);
    let mut rows = paths
        .iter()
        .map(|p| {
            let mut coordinates = [0.; 4];
            for i in 0..4 {
                coordinates[i] = format!(
                    "{:.2}",
                    (p.b.n[i] - b.n[i % 2]) / if i % 2 == 0 { width } else { height }
                )
                .parse::<f64>()
                .unwrap()
            }
            (p.segments, p.fill, p.stroke, coordinates)
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
            .then(a.3.partial_cmp(&b.3).unwrap())
    });
    json!(rows
        .into_iter()
        .map(|(s, f, t, c)| json!([s, f, t, c[0], c[1], c[2], c[3]]))
        .collect::<Vec<_>>())
}
pub fn build(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    build_with_work(v, &mut Work(0))
}
pub(super) fn build_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    build_source(&v["source"], v, w)
}
pub(super) fn build_source(source: &Value, v: &Value, w: &mut Work) -> Result<Value, Error> {
    let sz = size(&source["page_size"])?;
    let prior = array(&v["claimed_line_indices"], 20000)?
        .iter()
        .map(|v| v.as_i64().ok_or_else(invalid))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut lines = Vec::new();
    for l in optional(source, "lines", 4096)? {
        if integer(l, "angle", 0)? == 0 && !prior.contains(&integer(l, "source_index", 0)?) {
            lines.push(l.clone())
        }
    }
    let raw_paths = optional(source, "path_infos", 5000)?;
    let empty = json!({"vector_formula_blocks":[],"claimed_line_indices":[]});
    if lines.len() < 3 || raw_paths.is_empty() {
        return Ok(empty);
    }
    let mut hs = Vec::new();
    for l in &lines {
        hs.push(height(l, &bbox(&l["bbox"])?)?)
    }
    let em = median(hs);
    if em <= 0. {
        return Ok(empty);
    }
    let geometry = lines
        .iter()
        .map(|l| json!([l, l["bbox"]]))
        .collect::<Vec<_>>();
    let inferred = lanes::infer_layout(
        &json!({"line_geometry":geometry,"local_page_width":sz[0],"median_height":em}),
        w,
    )?;
    let mut lanes = Vec::new();
    for l in array(&inferred, 4096)? {
        let ids = array(&l["line_indices"], 4096)?;
        if l["is_span"] == false && ids.len() >= 3 {
            lanes.push(Lane {
                left: num(&l["left"])?,
                right: num(&l["right"])?,
                lines: ids
                    .iter()
                    .map(|i| {
                        i.as_u64()
                            .and_then(|i| usize::try_from(i).ok())
                            .filter(|&i| i < lines.len())
                            .ok_or_else(invalid)
                    })
                    .collect::<Result<_, _>>()?,
            })
        }
    }
    if lanes.is_empty() {
        return Ok(empty);
    }
    let mut paths = Vec::new();
    for p in raw_paths {
        w.charge(1)?;
        if integer(p, "form_depth", 0)? != 0
            || p["fill_visible"] != true
            || p["stroke_visible"] == true
        {
            continue;
        }
        let color = if let Some(color) = p.get("fill_rgba").filter(|v| !v.is_null()) {
            let a = array(color, 4)?;
            if a.len() < 3 {
                return Err(invalid());
            }
            Some([num(&a[0])?, num(&a[1])?, num(&a[2])?])
        } else {
            None
        };
        paths.push(Path {
            b: bbox(&p["bbox"])?,
            source: integer(p, "source_index", 0)?,
            segments: integer(p, "segment_count", 0)?,
            fill: true,
            stroke: false,
            color,
        })
    }
    let components = components(&paths, &lanes, em, w)?;
    let mut containers = Vec::new();
    for b in array(&v["container_blocks"], 4000)? {
        if let Some(b) = coerce(&b["bbox"])? {
            containers.push(b)
        }
    }
    let mut inline_sources = Vec::new();
    let mut candidates = Vec::new();
    for c in &components {
        let hosts = inline(c, &paths, &lanes[c.lane], &lines, em, &containers, w)?;
        if !hosts.is_empty() || core(c, &paths, &lanes[c.lane], &lines, em, sz, &containers, w)? {
            candidates.push(Candidate {
                lane: c.lane,
                b: c.b.clone(),
                sources: c.paths.iter().map(|&i| paths[i].source).collect(),
                number: false,
            })
        }
        inline_sources.push(hosts)
    }
    if candidates.is_empty() {
        return Ok(empty);
    }
    let claimed = attach(
        &mut candidates,
        &components,
        &paths,
        &lanes,
        &lines,
        em,
        &prior,
        w,
    )?;
    join(&mut candidates, &components, &paths, &lines, em, w)?;
    candidates.sort_by(|a, b| {
        a.b.n[1]
            .total_cmp(&b.b.n[1])
            .then(a.b.n[0].total_cmp(&b.b.n[0]))
    });
    let mut publication = optional(source, "publication_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    for l in &lines {
        if text::publication(l["text"].as_str().ok_or_else(invalid)?) {
            publication.push(bbox(&l["bbox"])?)
        }
    }
    let padding = 1.5f64.min(0.1 * em);
    let mut blocks = Vec::new();
    for c in candidates {
        let b = c.b.n;
        let expanded = bbox(&json!([
            b[0] - padding,
            b[1] - padding,
            b[2] + padding,
            b[3] + padding
        ]))?;
        let Some(b) = clip(&expanded, sz)? else {
            continue;
        };
        // Fetch every original path for shape/evidence, including shared source identities.
        let mut all = Vec::new();
        for p in raw_paths {
            w.charge(1)?;
            if !c.sources.contains(&integer(p, "source_index", 0)?) {
                continue;
            }
            all.push(Path {
                b: bbox(&p["bbox"])?,
                source: integer(p, "source_index", 0)?,
                segments: integer(p, "segment_count", 0)?,
                fill: p["fill_visible"] == true,
                stroke: p["stroke_visible"] == true,
                color: None,
            })
        }
        let ps = all.iter().collect::<Vec<_>>();
        let mathematical = c.number || math(&ps, &b, &lines, em, w)?;
        let decorative = decoration(&ps, w)?;
        let edge = center(&b, 1) < 0.085 * sz[1] || center(&b, 1) > 0.90 * sz[1];
        w.charge(publication.len() + 1)?;
        let publication = publication
            .iter()
            .any(|a| distance(a, &b) <= 8. * em || axis(a, &b, 1) >= 0.5);
        let mut block = json!({"type":if edge&&decorative&&publication&&!mathematical{"image"}else{"equation"},"bbox":b.raw,"angle":0,"content":""});
        if edge && decorative && !mathematical {
            block["_vector_shape"] = signature(&ps, &b);
            block["_vector_edge_decoration"] = json!(true)
        }
        let mut hosts = BTreeSet::new();
        for (i, component) in components.iter().enumerate() {
            w.charge(component.paths.len() + 1)?;
            if component
                .paths
                .iter()
                .all(|&i| c.sources.contains(&paths[i].source))
            {
                hosts.extend(inline_sources[i].iter().copied())
            }
        }
        if !hosts.is_empty() {
            block["_inline_formula_prose_sources"] = json!(hosts)
        }
        blocks.push(block)
    }
    Ok(json!({"vector_formula_blocks":blocks,"claimed_line_indices":claimed}))
}
/// Whole-document repeat pass retains the original normalized geometry/signature key.
pub fn classify_repeated(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    classify_repeated_with_work(v, &mut Work(0))
}
pub(super) fn classify_repeated_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let mut pages = array(&v["pages"], 1000)?.to_vec();
    classify_repeated_in_place(&mut pages, w)?;
    let mut out = json!({});
    out["pages"] = Value::Array(pages);
    Ok(out)
}
pub(super) fn classify_repeated_in_place(pages: &mut [Value], w: &mut Work) -> Result<(), Error> {
    let mut groups = BTreeMap::<String, Vec<(usize, usize)>>::new();
    for (i, p) in pages.iter().enumerate() {
        let sz = size(p.get("page_size").unwrap_or(&p["source"]["page_size"]))?;
        for (j, b) in array(&p["fixed_blocks"], 4000)?.iter().enumerate() {
            w.charge(1)?;
            if b["_vector_edge_decoration"] != true
                || b["_vector_shape"].as_array().is_none_or(Vec::is_empty)
            {
                continue;
            }
            let b_box = bbox(&b["bbox"])?;
            let bounds = b_box
                .n
                .iter()
                .enumerate()
                .map(|(k, x)| format!("{:.2}", x / sz[k % 2]).parse::<f64>().unwrap())
                .collect::<Vec<_>>();
            let key = json!([b["_vector_shape"], bounds]).to_string();
            groups.entry(key).or_default().push((i, j));
        }
    }
    for members in groups.values() {
        if members
            .iter()
            .map(|(i, _)| i)
            .collect::<BTreeSet<_>>()
            .len()
            >= 2
        {
            for &(i, j) in members {
                pages[i]["fixed_blocks"][j]["type"] = json!("image")
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn borrowed_vector_formula_matches_python() {
        let f: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(
                "../../../../../../tests/fixtures/native_table_owned_reference.json.gz"
            )[..],
        ))
        .unwrap();
        for c in f["vectors"].as_array().unwrap() {
            let q = &c["request"];
            let (mut a, mut b) = (Work(0), Work(0));
            let old = build_with_work(q, &mut a).unwrap();
            let new = build_source(&q["source"], q, &mut b).unwrap();
            assert_eq!(new, c["expected"], "{}", c["label"]);
            assert_eq!(new, old);
            assert_eq!(a.0, b.0);
        }
    }
    #[test]
    fn old_vector_formula_document() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-vector-document.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let actual = crate::legacy_model::page_containers::prepare_document(&c["request"]);
            if c.get("error").is_some() {
                assert!(actual.is_err(), "{}", c["label"]);
                continue;
            }
            let actual = actual.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            assert_eq!(actual, c["expected"], "{}", c["label"]);
        }
    }
    #[test]
    fn old_vector_formula_text() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-vector-text.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let s = c["text"].as_str().unwrap();
            for (k, actual) in [
                ("number", text::number(s)),
                ("standalone", text::standalone(s)),
                ("math", text::math(s)),
                ("prose", text::prose(s)),
                ("publication", text::publication(s)),
            ] {
                assert_eq!(json!(actual), c[k], "{k}: {s:?}");
            }
        }
    }
    #[test]
    fn old_vector_formula_pipeline() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-vector-formulas.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let out = build(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            assert_eq!(out, c["expected"], "{}", c["label"]);
        }
        for c in f["repeated"].as_array().unwrap() {
            assert_eq!(
                classify_repeated(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
}
