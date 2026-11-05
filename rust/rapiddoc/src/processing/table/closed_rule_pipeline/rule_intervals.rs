//! 普通横线候选完整参考分支；缓存不参与判定，原序枚举且不提前删除子候选。
use super::*;
use rowprep::Row;
use super::super::table_members::{Domain, Lookup, Members};
use std::cmp::Ordering;
use std::{collections::HashMap, sync::Arc};

pub(super) struct Draft {
    pub(super) rows: Vec<Arc<Row>>,
    pub(super) boundaries: Vec<[Value; 4]>,
    pub(super) caption: Value,
    pub(super) score: f64,
}
fn cmp(a: f64, b: f64) -> Ordering {
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
fn stable(rows: &[&Row], em: f64, work: &mut Work) -> Result<(usize, f64), Error> {
    if rows.is_empty() {
        return Ok((0, 0.));
    }
    let mut best = (0, 0.);
    for axis in 0..3 {
        let mut clusters: Vec<(f64, Vec<f64>, BTreeSet<usize>)> = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            for f in &row.fragments {
                let x = match axis {
                    0 => f.local.n[0],
                    1 => center(&f.local, 0),
                    _ => f.local.n[2],
                };
                work.charge(clusters.len() + 1)?;
                if let Some(c) = clusters
                    .iter_mut()
                    .find(|c| (x - c.0).abs() <= 3f64.max(0.75 * em))
                {
                    c.1.push(x);
                    c.2.insert(i);
                    work.charge(c.1.len() + 1)?;
                    c.0 = super::super::table_native_candidate::sum(&c.1) / c.1.len() as f64;
                } else {
                    clusters.push((x, vec![x], BTreeSet::from([i])));
                }
            }
        }
        let covered: Vec<_> = clusters
            .iter()
            .map(|c| c.2.len() as f64 / rows.len() as f64)
            .filter(|c| *c >= 0.5)
            .collect();
        let result = (
            covered.len(),
            covered.into_iter().reduce(f64::min).unwrap_or(0.),
        );
        if result > best {
            best = result;
        }
    }
    Ok(best)
}
type ColumnClusters = Vec<(f64, Vec<f64>, BTreeSet<usize>)>;
#[derive(Default)]
struct StableCache {
    results: HashMap<Vec<usize>, (usize, f64)>,
    prefixes: HashMap<usize, (Vec<usize>, [ColumnClusters; 3])>,
}
impl StableCache {
    fn count(&mut self, rows: &[&Row], em: f64, work: &mut Work) -> Result<(usize, f64), Error> {
        if !crate::legacy_page::budget::current().host
            || crate::legacy_page::budget::current().work != usize::MAX
        {
            return stable(rows, em, work);
        }
        if rows.is_empty() {
            return Ok((0, 0.));
        }
        let ids: Vec<usize> = rows.iter().map(|r| *r as *const Row as usize).collect();
        if let Some(result) = self.results.get(&ids) {
            return Ok(*result);
        }
        let state = self
            .prefixes
            .entry(ids[0])
            .or_insert_with(|| (Vec::new(), std::array::from_fn(|_| Vec::new())));
        if !ids.starts_with(&state.0) {
            *state = (Vec::new(), std::array::from_fn(|_| Vec::new()));
        }
        let start = state.0.len();
        for (axis, clusters) in state.1.iter_mut().enumerate() {
            for (i, row) in rows.iter().enumerate().skip(start) {
                for f in &row.fragments {
                    let x = match axis {
                        0 => f.local.n[0],
                        1 => center(&f.local, 0),
                        _ => f.local.n[2],
                    };
                    if let Some(c) = clusters
                        .iter_mut()
                        .find(|c| (x - c.0).abs() <= 3f64.max(0.75 * em))
                    {
                        c.1.push(x);
                        c.2.insert(i);
                        c.0 = super::super::table_native_candidate::sum(&c.1) / c.1.len() as f64;
                    } else {
                        clusters.push((x, vec![x], BTreeSet::from([i])));
                    }
                }
            }
        }
        state.0 = ids.clone();
        let result = state
            .1
            .iter()
            .map(|clusters| {
                let covered: Vec<_> = clusters
                    .iter()
                    .map(|c| c.2.len() as f64 / rows.len() as f64)
                    .filter(|c| *c >= 0.5)
                    .collect();
                (
                    covered.len(),
                    covered.into_iter().reduce(f64::min).unwrap_or(0.),
                )
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        self.results.insert(ids, result);
        Ok(result)
    }
}
// Keep admission geometry from before exclusion, as in Python's _RuleCorridorRow.
struct CorridorRow {
    source: Box4,
    gate_center: f64,
    row: Row,
}
fn prepare_corridor(
    p: &Page,
    b: &Box4,
    excluded: &[Box4],
    work: &mut Work,
) -> Result<Vec<CorridorRow>, Error> {
    let mut out = Vec::new();
    for source in &p.rows {
        let Some(mut row) = rowprep::clipped(source, b, 0., work)? else {
            continue;
        };
        let gate_center = row.center;
        row.fragments.retain(|f| {
            !excluded
                .iter()
                .any(|e| inside(center(&f.local, 0), center(&f.local, 1), e))
        });
        if row.fragments.is_empty() {
            continue;
        }
        row.b = row.fragments[0].local.clone();
        for f in &row.fragments[1..] {
            row.b = union(row.b, &f.local);
        }
        row.center = super::super::table_native_candidate::sum(
            &row.fragments
                .iter()
                .map(|f| center(&f.local, 1))
                .collect::<Vec<_>>(),
        ) / row.fragments.len() as f64;
        out.push(CorridorRow {
            source: source.b.clone(),
            gate_center,
            row,
        });
    }
    Ok(out)
}
fn groups(p: &Page, work: &mut Work) -> Result<Vec<Vec<usize>>, Error> {
    let mut ids: Vec<_> =
        p.p.rules
            .iter()
            .enumerate()
            .filter(|(_, r)| r.h && r.b.n[2] - r.b.n[0] >= 40f64.max(10. * p.em))
            .map(|(i, _)| i)
            .collect();
    work.charge(ids.len() * 16 + 1)?;
    ids.sort_by(|a, b| {
        let a = &p.p.rules[*a].b.n;
        let b = &p.p.rules[*b].b.n;
        cmp(a[0], b[0]).then(cmp(a[2], b[2])).then(cmp(a[1], b[1]))
    });
    let mut spans: Vec<Vec<usize>> = Vec::new();
    for i in ids {
        let b = &p.p.rules[i].b;
        work.charge(spans.len() + 1)?;
        if let Some(g) = spans.iter_mut().find(|g| {
            let a = &p.p.rules[g[0]].b;
            (b.n[0] - a.n[0]).abs() <= 4f64.max(2. * p.em)
                && (b.n[2] - a.n[2]).abs() <= 4f64.max(2. * p.em)
        }) {
            g.push(i);
        } else {
            spans.push(vec![i]);
        }
    }
    let mut out = Vec::new();
    for mut g in spans {
        g.sort_by(|a, b| cmp(center(&p.p.rules[*a].b, 1), center(&p.p.rules[*b].b, 1)));
        let mut unique: Vec<usize> = Vec::new();
        for i in g {
            work.charge(unique.len() + 1)?;
            if !unique
                .iter()
                .any(|j| (center(&p.p.rules[i].b, 1) - center(&p.p.rules[*j].b, 1)).abs() <= 1.)
            {
                unique.push(i);
            }
        }
        if unique.len() >= 2 {
            out.push(unique);
        }
    }
    Ok(out)
}
fn partition<'a>(
    rows: &[&'a Row],
    rules: &[usize],
    p: &Page,
    work: &mut Work,
) -> Result<Vec<Vec<&'a Row>>, Error> {
    if crate::legacy_page::budget::current().host
        && crate::legacy_page::budget::current().work == usize::MAX
    {
        let centers: Vec<_> = rules.iter().map(|&i| center(&p.p.rules[i].b, 1)).collect();
        // Match Python's prepared-band assignment. A row exactly on an inner
        // rule belongs to BOTH adjacent bands, in original row order.
        if centers.windows(2).all(|w| w[0] < w[1]) {
            let mut out = vec![Vec::new(); centers.len().saturating_sub(1)];
            for &row in rows {
                let at = centers.partition_point(|&y| y < row.center);
                if at > 0 && at < centers.len() {
                    out[at - 1].push(row);
                }
                if at < out.len() && centers[at] == row.center {
                    out[at].push(row);
                }
            }
            return Ok(out);
        }
    }
    let mut out = Vec::new();
    for band in rules.windows(2) {
        let top = center(&p.p.rules[band[0]].b, 1);
        let bottom = center(&p.p.rules[band[1]].b, 1);
        work.charge(rows.len() + 1)?;
        out.push(
            rows.iter()
                .copied()
                .filter(|r| top <= r.center && r.center <= bottom)
                .collect(),
        );
    }
    Ok(out)
}
fn multiline(rows: &[&Row], em: f64, work: &mut Work) -> Result<bool, Error> {
    if rows.len() < 3 || rows[0].fragments.len() != 2 {
        return Ok(false);
    }
    let mut first = rows[0].fragments.iter().collect::<Vec<_>>();
    first.sort_by(|a, b| cmp(a.local.n[0], b.local.n[0]));
    let (left, right) = (&first[0].local.n, &first[1].local.n);
    if left[2] - left[0] > 0.45 * (right[2] - left[0]) || right[0] - left[2] < 0.5 * em {
        return Ok(false);
    }
    work.charge(rows.len() + 1)?;
    Ok(rows.windows(2).all(|pair| {
        let row = pair[1];
        row.center - pair[0].center <= 2. * em
            && row.fragments.len() == 1
            && (row.fragments[0].local.n[0] - right[0]).abs() <= 0.75 * em
            && row.fragments[0].local.n[2] <= right[2] + em
    }))
}
fn compatible(
    bands: &[Vec<&Row>],
    rules: &[usize],
    p: &Page,
    work: &mut Work,
    cache: &mut StableCache,
) -> Result<bool, Error> {
    let heights: Vec<_> = rules
        .windows(2)
        .map(|g| center(&p.p.rules[g[1]].b, 1) - center(&p.p.rules[g[0]].b, 1))
        .collect();
    if heights.iter().all(|h| *h <= 6. * p.em) {
        return Ok(true);
    }
    let mut profiles = Vec::new();
    for (rows, h) in bands.iter().zip(heights) {
        if multiline(rows, p.em, work)? {
            profiles.push((2, h, rows.len(), 1.));
        } else {
            let dense: Vec<_> = rows
                .iter()
                .copied()
                .filter(|r| r.fragments.len() >= 2)
                .collect();
            let (n, c) = cache.count(&dense, p.em, work)?;
            profiles.push((n, h, dense.len(), c));
        }
    }
    let maximum = profiles.iter().map(|p| p.0).max().unwrap_or(0);
    Ok(profiles.iter().enumerate().all(|(i, (n, h, count, c))| {
        (i == 0 && *h <= 2.5 * p.em)
            || *h <= 6. * p.em
            || (*count >= 2usize.max((*h / (8. * p.em).max(0.1)) as usize)
                && *c >= 0.5
                && *n >= 2usize.max((0.5 * maximum as f64) as usize))
    }))
}
fn full_tracks(p: &Page, b: &Box4, work: &mut Work) -> Result<Vec<f64>, Error> {
    let h = (b.n[3] - b.n[1]).max(0.1);
    work.charge(p.p.rules.len() + 1)?;
    let mut xs: Vec<_> =
        p.p.rules
            .iter()
            .filter(|r| {
                !r.h && (r.b.n[3].min(b.n[3]) - r.b.n[1].max(b.n[1])).max(0.) / h >= 0.8
                    && r.b.n[3] - r.b.n[1] >= 0.8 * h
                    && b.n[0] - p.em <= center(&r.b, 0)
                    && center(&r.b, 0) <= b.n[2] + p.em
            })
            .map(|r| center(&r.b, 0))
            .collect();
    xs.sort_by(|a, b| cmp(*a, *b));
    let mut groups: Vec<Vec<f64>> = Vec::new();
    for x in xs {
        if let Some(g) = groups.last_mut() {
            if (x - graphic_grid::mean(g, false, work)?).abs() <= 1f64.max(0.1 * p.em) {
                g.push(x);
                continue;
            }
        }
        groups.push(vec![x]);
    }
    groups
        .iter()
        .map(|g| graphic_grid::mean(g, false, work))
        .collect()
}
fn compact(
    rows: &[&Row],
    dense: &[&Row],
    rules: &[usize],
    p: &Page,
    b: &Box4,
    work: &mut Work,
) -> Result<usize, Error> {
    if rows.len() != 2
        || dense.len() != 2
        || rules.len() < 3
        || (b.n[3] - b.n[1]).max(0.1) > 6. * p.em
    {
        return Ok(0);
    }
    let tracks = full_tracks(p, b, work)?;
    if tracks.len() < 3 {
        return Ok(0);
    }
    let left = *tracks
        .iter()
        .min_by(|a, c| cmp((**a - b.n[0]).abs(), (**c - b.n[0]).abs()))
        .unwrap();
    let right = *tracks
        .iter()
        .min_by(|a, c| cmp((**a - b.n[2]).abs(), (**c - b.n[2]).abs()))
        .unwrap();
    let tol = 1.5f64.max(0.25 * p.em);
    if (left - b.n[0]).abs() > tol || (right - b.n[2]).abs() > tol || right <= left {
        return Ok(0);
    }
    let boundaries: Vec<_> = tracks
        .into_iter()
        .filter(|x| left <= *x && *x <= right)
        .collect();
    if boundaries.len() < 3 {
        return Ok(0);
    }
    let mut occupied = Vec::new();
    for row in dense {
        let mut cols = BTreeSet::new();
        for f in &row.fragments {
            let x = center(&f.local, 0);
            work.charge(boundaries.len() + 1)?;
            let matches: Vec<_> = boundaries
                .windows(2)
                .enumerate()
                .filter(|(_, b)| b[0] <= x && x <= b[1])
                .map(|(i, _)| i)
                .collect();
            if matches.len() != 1 || !cols.insert(matches[0]) {
                return Ok(0);
            }
        }
        if cols.len() < 2 {
            return Ok(0);
        }
        occupied.push(cols);
    }
    Ok(if occupied[0].intersection(&occupied[1]).count() >= 2 {
        boundaries.len() - 1
    } else {
        0
    })
}
fn fills(source: &Value, p: &Page, work: &mut Work) -> Result<Vec<Box4>, Error> {
    let mut out = Vec::new();
    for path in optional(source, "path_infos", 5000)? {
        work.charge(1)?;
        if integer(path, "form_depth", 0)? == 0
            && path
                .get("fill_visible")
                .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?
        {
            out.push(local(&bbox(&path["bbox"])?, &p.size, p.angle)?);
        }
    }
    Ok(out)
}
fn fill_count(paths: &[Box4], b: &Box4, em: f64, work: &mut Work) -> Result<usize, Error> {
    let mut candidates: Vec<&Box4> = Vec::new();
    for p in paths {
        work.charge(candidates.len() + 1)?;
        let (w, h, y) = (p.n[2] - p.n[0], p.n[3] - p.n[1], center(p, 1));
        if w < (8. * em).max(0.3 * (b.n[2] - b.n[0]))
            || h < 0.25 * em
            || h > 3. * em
            || y < b.n[1]
            || y > b.n[3]
            || super::super::graphic_captions::axis(p, b, 0) < 0.8
            || candidates.iter().any(|c| overlap_smaller(p, c) >= 0.9)
        {
            continue;
        }
        candidates.push(p);
    }
    let mut groups: Vec<(&Box4, usize)> = Vec::new();
    for p in candidates {
        work.charge(groups.len() + 1)?;
        if let Some((_, n)) = groups.iter_mut().find(|(a, _)| {
            (p.n[0] - a.n[0]).abs() <= 3f64.max(em)
                && (p.n[2] - a.n[2]).abs() <= 3f64.max(em)
                && ((p.n[3] - p.n[1]) - (a.n[3] - a.n[1])).abs() <= 3f64.max(em)
        }) {
            *n += 1;
        } else {
            groups.push((p, 1));
        }
    }
    Ok(groups.iter().map(|g| g.1).max().unwrap_or(0))
}
pub(super) fn drafts(
    source: &Value,
    p: &Page,
    excluded: &[Box4],
    work: &mut Work,
) -> Result<Vec<Draft>, Error> {
    #[cfg(test)]
    if let Ok(path) = std::env::var("RAPIDDOC_RULE_CAPTURE") {
        std::fs::write(
            path,
            serde_json::to_vec(&json!({"source":source,
            "options":{"angle":p.angle,"median_height":p.em},
            "excluded":excluded.iter().map(|b|b.raw.clone()).collect::<Vec<_>>()}))
            .unwrap(),
        )
        .unwrap();
        eprintln!(
            "captured rows={} lines={} rules={} groups={:?}",
            p.rows.len(),
            p.lines.len(),
            p.p.rules.len(),
            groups(p, work)?.iter().map(Vec::len).collect::<Vec<_>>()
        );
        return Err(invalid());
    }
    let captions = captions(p, work)?;
    let paths = fills(source, p, work)?;
    let mut out = Vec::new();
    let mut total_members = 0;
    // Intervals overlap extensively. Python drafts retain references to their
    // corridor rows; share identical immutable row payloads instead of copying
    // their complete character trees into every candidate.
    let mut shared_rows: HashMap<String, Arc<Row>> = HashMap::new();
    let mut corridors: HashMap<(u64, u64), Vec<CorridorRow>> = HashMap::new();
    let mut cached_rows: HashMap<usize, Arc<Row>> = HashMap::new();
    let mut stable_cache = StableCache::default();
    for group in groups(p, work)? {
        for first in 0..group.len() - 1 {
            stable_cache.prefixes.clear();
            for bottom in first + 1..group.len() {
                work.charge(1)?;
                let rules = &group[first..=bottom];
                let top = &p.p.rules[rules[0]].b;
                let low = &p.p.rules[*rules.last().unwrap()].b;
                let b = union(top.clone(), low);
                let cache_key = (b.n[0].to_bits(), b.n[2].to_bits());
                let direct_rows;
                let rows = if crate::legacy_page::budget::current().host
                    && crate::legacy_page::budget::current().work == usize::MAX
                {
                    // Python caches rows by horizontal corridor. Vertical
                    // intervals only select from those immutable clipped rows.
                    let cached = corridors.entry(cache_key).or_insert_with(Vec::new);
                    if cached.is_empty() {
                        let mut horizontal = b.clone();
                        horizontal.n[1] = -f64::MAX;
                        horizontal.n[3] = f64::MAX;
                        *cached = prepare_corridor(p, &horizontal, excluded, work)?;
                    }
                    cached
                        .iter()
                        .filter(|r| {
                            r.source.n[1] <= b.n[3]
                                && r.source.n[3] >= b.n[1]
                                && r.gate_center >= b.n[1]
                                && r.gate_center <= b.n[3]
                        })
                        .map(|r| &r.row)
                        .collect::<Vec<_>>()
                } else {
                    let raw = corridor(p, &b, excluded, work)?;
                    let mut total = 0;
                    direct_rows = raw
                        .iter()
                        .map(|r| rowprep::row(r, &mut total))
                        .collect::<Result<Vec<_>, _>>()?;
                    direct_rows.iter().collect::<Vec<_>>()
                };
                let caption = find_caption(&captions, &b, p.em, work)?;
                let anchored = !caption.is_null()
                    && rules.len() >= 3
                    && b.n[3] - b.n[1]
                        <= 0.15 * num(&p.size[if matches!(p.angle, 90 | 270) { 0 } else { 1 }])?;
                let bands = partition(&rows, rules, p, work)?;
                let every = bands.iter().enumerate().all(|(i, g)| {
                    g.iter().any(|r| r.fragments.len() >= 2)
                        || (i == 0
                            && !g.is_empty()
                            && center(&p.p.rules[rules[1]].b, 1) - center(top, 1) <= 2.5 * p.em)
                });
                if !every && !anchored {
                    continue;
                }
                if !compatible(&bands, rules, p, work, &mut stable_cache)? && !anchored {
                    continue;
                }
                let fill = fill_count(&paths, &b, p.em, work)?;
                work.charge(p.p.rules.len() + 1)?;
                let aligned =
                    p.p.rules
                        .iter()
                        .filter(|r| {
                            !r.h && b.n[0] - p.em <= center(&r.b, 0)
                                && center(&r.b, 0) <= b.n[2] + p.em
                                && r.b.n[3] - r.b.n[1] >= (4. * p.em).max(0.5 * (b.n[3] - b.n[1]))
                                && super::super::graphic_captions::axis(&r.b, &b, 1) >= 0.8
                        })
                        .count();
                let mut ordered: Vec<_> = rows.iter().copied().collect();
                ordered.sort_by(|a, b| cmp(a.center, b.center));
                let mut segments: Vec<Vec<&Row>> = Vec::new();
                for row in ordered {
                    work.charge(1)?;
                    if segments
                        .last()
                        .is_none_or(|g| (row.b.n[1] - g.last().unwrap().b.n[3]).max(0.) > 3. * p.em)
                    {
                        segments.push(vec![row]);
                    } else {
                        segments.last_mut().unwrap().push(row);
                    }
                }
                let mut accepted: Option<(Vec<&Row>, usize, usize, f64)> = None;
                for segment in segments {
                    let dense: Vec<_> = segment
                        .iter()
                        .copied()
                        .filter(|r| r.fragments.len() >= 2)
                        .collect();
                    let compact = if dense.len() == 2 {
                        compact(&segment, &dense, rules, p, &b, work)?
                    } else {
                        0
                    };
                    let (mut columns, coverage) = stable_cache.count(&dense, p.em, work)?;
                    if compact > 0 {
                        columns = compact;
                    }
                    if dense.len() < 3
                        && compact == 0
                        && !(anchored && dense.len() >= 2 && columns >= 3 && coverage >= 0.5)
                    {
                        continue;
                    }
                    if dense.len() as f64 / (segment.len() as f64) < 0.2
                        || columns < 2
                        || coverage < 0.5
                    {
                        continue;
                    }
                    if columns == 2
                        && fill < 2
                        && aligned == 0
                        && dense.len() as f64 / segment.len() as f64 >= 0.55
                    {
                        let mut ratios = Vec::new();
                        for r in &dense {
                            work.charge(r.fragments.len() + 1)?;
                            ratios.push(
                                super::super::table_native_candidate::sum(
                                    &r.fragments
                                        .iter()
                                        .map(|f| f.local.n[2] - f.local.n[0])
                                        .collect::<Vec<_>>(),
                                ) / (b.n[2] - b.n[0]).max(0.1),
                            );
                        }
                        if median(ratios) >= 0.75 {
                            continue;
                        }
                    }
                    if (segment[0].b.n[1] - b.n[1]).max(0.) > 2.5 * p.em
                        || (b.n[3] - segment.last().unwrap().b.n[3]).max(0.) > 2.5 * p.em
                    {
                        continue;
                    }
                    let mut rb = segment[0].b.clone();
                    for row in &segment[1..] {
                        rb = union(rb, &row.b);
                    }
                    let width = (b.n[2] - b.n[0]).max(0.1);
                    let rw = (rb.n[2] - rb.n[0]).max(0.1);
                    let overlap = (rb.n[2].min(b.n[2]) - rb.n[0].max(b.n[0])).max(0.);
                    if overlap / width.min(rw) < 0.9 || rw < (8. * p.em).max(0.25 * width) {
                        continue;
                    }
                    let key = (segment.len(), dense.len(), columns, coverage);
                    if accepted
                        .as_ref()
                        .is_none_or(|a| key > (a.0.len(), a.1, a.2, a.3))
                    {
                        accepted = Some((segment, dense.len(), columns, coverage));
                    }
                }
                if let Some((rows, dense, columns, _)) = accepted {
                    total_members += rows.iter().map(|r| r.fragments.len()).sum::<usize>();
                    if total_members > rapiddoc_core::resources::limit(20000)
                        || out.len() >= rapiddoc_core::resources::limit(5000)
                    {
                        return Err(limit());
                    }
                    out.push(Draft {
                        rows: rows
                            .into_iter()
                            .map(|r| {
                                if crate::legacy_page::budget::current().host
                                    && crate::legacy_page::budget::current().work == usize::MAX
                                {
                                    // Corridor rows stay alive and immutable throughout enumeration.
                                    return Arc::clone(
                                        cached_rows
                                            .entry(r as *const Row as usize)
                                            .or_insert_with(|| Arc::new(r.clone())),
                                    );
                                }
                                let value = rowprep::output(r);
                                let key = value.to_string();
                                Arc::clone(
                                    shared_rows
                                        .entry(key)
                                        .or_insert_with(|| Arc::new(r.clone())),
                                )
                            })
                            .collect(),
                        boundaries: vec![top.raw.clone(), low.raw.clone()],
                        caption,
                        score: (2 + dense + columns + fill.min(8)) as f64,
                    });
                }
            }
        }
    }
    Ok(out)
}
// Sparse row masks share the page's sorted source-ID domain. Combining an
// interval no longer sorts every fragment ID again. Only occupied words are
// stored for each immutable corridor row, so storage follows fragment count.
struct CoreMemberIndex {
    domain: Arc<Domain>,
    rows: HashMap<usize, Vec<(usize, u64)>>,
    scratch: Vec<u64>,
}
impl CoreMemberIndex {
    fn new(page: &Page) -> Self {
        let ids = page
            .rows
            .iter()
            .flat_map(|r| &r.fragments)
            .map(|f| f.id)
            .collect::<BTreeSet<_>>();
        let scratch = vec![0; ids.len().div_ceil(64)];
        let domain = Domain::new(ids);
        Self {
            domain,
            rows: HashMap::new(),
            scratch,
        }
    }
    fn members(&mut self, rows: &[Arc<Row>]) -> Option<Members> {
        self.scratch.fill(0);
        for row in rows {
            let key = Arc::as_ptr(row) as usize;
            if !self.rows.contains_key(&key) {
                let mut words = std::collections::BTreeMap::<usize, u64>::new();
                for fragment in &row.fragments {
                    let rank = *self.domain.ranks.get(&fragment.id)?;
                    *words.entry(rank / 64).or_default() |= 1u64 << (rank % 64);
                }
                self.rows.insert(key, words.into_iter().collect());
            }
            for &(word, bits) in &self.rows[&key] {
                self.scratch[word] |= bits;
            }
        }
        Some(Members::packed(
            Arc::clone(&self.domain),
            self.scratch.clone(),
        ))
    }
}
#[derive(Default)]
pub(super) struct MaterializationCache {
    timings: crate::processing::NativeStageTotals,
    notes: super::super::table_footnotes::FootnoteCache,
    core_members: Option<CoreMemberIndex>,
    grids: Option<Vec<Box4>>,
    grid_members: HashMap<[u64; 4], BTreeSet<i64>>,
    shared_grid_members: HashMap<[u64; 4], Members>,
}
pub(super) fn materialize(
    d: &Draft,
    p: &Page,
    excluded: &[Box4],
    work: &mut Work,
) -> Result<Value, Error> {
    materialize_cached(d, p, excluded, &mut MaterializationCache::default(), work)
}
pub(super) fn materialize_cached(
    d: &Draft,
    p: &Page,
    excluded: &[Box4],
    cache: &mut MaterializationCache,
    work: &mut Work,
) -> Result<Value, Error> {
    #[cfg(test)]
    let start = std::time::Instant::now();
    let production = crate::legacy_page::budget::current().host
        && crate::legacy_page::budget::current().work == usize::MAX;
    let mut c = if production {
        let core: Vec<_> = d.rows.iter().map(|r| r.as_ref()).collect();
        let q = value_object!({"rule_bboxes":d.boundaries,"core_row_indices":json!([]),"median_height":p.em,"page_size":p.size,"angle":p.angle,"caption_source_index":d.caption,"has_possible_table_notes":true});
        let lines: Vec<_> = p.lines.iter().map(|l| l.as_ref()).collect();
        super::super::rule_candidate_expansion::expand_with_inputs(
            &q,
            None,
            Some(&core),
            Some(&p.rows),
            Some(&lines),
            Some(&mut cache.notes),
            work,
        )?["candidate"]
            .clone()
    } else {
        let rows: Vec<_> = d.rows.iter().map(|r| rowprep::output(r)).collect();
        let q = json!({"rows":p.raw_rows,"lines":p.lines,"rule_bboxes":d.boundaries,"core_row_indices":[],"median_height":p.em,"page_size":p.size,"angle":p.angle,"caption_source_index":d.caption,"has_possible_table_notes":true});
        super::super::rule_candidate_expansion::expand_with_core(&q, Some(&rows), work)?
            ["candidate"]
            .clone()
    };
    #[cfg(test)]
    if std::env::var_os("RAPIDDOC_RULE_MATERIALIZE").is_some() {
        eprintln!("expand {:?}", start.elapsed());
    }
    c["score"] = json!(d.score);
    if production {
        if cache.grids.is_none() {
            let mut grids = component_boxes(p, work)?;
            grids.retain(|b| !excluded.iter().any(|e| overlap_smaller(b, e) >= 0.5));
            cache.grids = Some(grids);
        }
        Ok(expand_candidates_cached(
            &[c],
            p,
            cache.grids.as_ref().unwrap(),
            Some(&mut cache.grid_members),
            work,
        )?
        .pop()
        .unwrap())
    } else {
        let mut grids = component_boxes(p, work)?;
        work.charge(grids.len() * (excluded.len() + 1))?;
        grids.retain(|b| !excluded.iter().any(|e| overlap_smaller(b, e) >= 0.5));
        Ok(expand_candidates(&[c], p, &grids, work)?.pop().unwrap())
    }
}
pub(super) fn draft_members(
    draft: &Draft,
    page: &Page,
    cache: &mut MaterializationCache,
) -> Members {
    cache
        .core_members
        .get_or_insert_with(|| CoreMemberIndex::new(page))
        .members(&draft.rows)
        .unwrap_or_else(|| {
            Members::from(
                draft
                    .rows
                    .iter()
                    .flat_map(|row| &row.fragments)
                    .map(|fragment| fragment.id)
                    .collect::<BTreeSet<_>>(),
            )
        })
}
pub(super) fn projection_geometry(d: &Draft, p: &Page) -> Value {
    json!({"kind":"draft","angle":p.angle,"median_height":p.em,"boundaries":d.boundaries,"caption":d.caption})
}
// Production candidates keep their member sets typed until the final merge output.
pub(super) fn materialize_typed(
    d: &Draft,
    p: &Page,
    excluded: &[Box4],
    cache: &mut MaterializationCache,
    work: &mut Work,
) -> Result<super::super::table_candidate_merge::Candidate, Error> {
    cache.timings.begin();
    #[cfg(test)]
    let mut stage = std::time::Instant::now();
    let core_rows: Vec<_> = d.rows.iter().map(|r| r.as_ref()).collect();
    let request = value_object!({"rule_bboxes":d.boundaries,"core_row_indices":json!([]),"median_height":p.em,"caption_source_index":d.caption,"footnote_row_indices":json!([])});
    let members = cache
        .core_members
        .get_or_insert_with(|| CoreMemberIndex::new(p))
        .members(&d.rows);
    let prepared = rowprep::prepare_typed_members(
        &request,
        None,
        Some(&p.rows),
        Some(&core_rows),
        members,
        work,
    )?;
    cache.timings.mark("rule_materialize_core");
    #[cfg(test)]
    if std::env::var_os("RAPIDDOC_RULE_MATERIALIZE").is_some() {
        eprintln!("typed core {:?}", stage.elapsed());
        stage = std::time::Instant::now();
    }
    let request = value_object!({"rule_bbox":prepared.rule.raw,"page_size":p.size,"angle":p.angle,"median_height":p.em});
    let lines: Vec<_> = p.lines.iter().map(|r| r.as_ref()).collect();
    let notes = super::super::table_footnotes::collect_core_geometry(
        &request,
        &p.rows,
        &lines,
        &mut cache.notes,
        &prepared.ids,
        work,
    )?;
    cache.timings.mark("rule_materialize_notes");
    #[cfg(test)]
    if std::env::var_os("RAPIDDOC_RULE_MATERIALIZE").is_some() {
        eprintln!("typed notes {:?}", stage.elapsed());
        stage = std::time::Instant::now();
    }
    let mut annotations = prepared.annotations.iter().collect::<Vec<_>>();
    if !notes["annotation"].is_null() {
        annotations.push(&notes["annotation"]);
    }
    let mut included = prepared
        .caption_rows
        .iter()
        .map(|r| r.b.clone())
        .chain(core_rows.iter().map(|r| r.b.clone()))
        .collect::<Vec<_>>();
    for row in notes["footnote_rows"].as_array().unwrap() {
        included.push(bbox(&row["bbox"])?);
    }
    let mut outer = included[0].clone();
    for b in &included[1..] {
        outer = union(outer, b);
    }
    let mut local = union(prepared.core.clone(), &outer);
    let mut core = prepared.core;
    if cache.grids.is_none() {
        let mut grids = component_boxes(p, work)?;
        grids.retain(|b| !excluded.iter().any(|e| overlap_smaller(b, e) >= 0.5));
        cache.grids = Some(grids);
    }
    let grids = cache.grids.as_ref().unwrap();
    let tolerance = 2f64.max(p.em);
    let mut best: Option<usize> = None;
    for (i, b) in grids.iter().enumerate() {
        if super::super::graphic_captions::axis(&core, b, 0) < 0.9
            || core.n[3] < b.n[1] - tolerance
            || core.n[1] > b.n[3] + tolerance
        {
            continue;
        }
        let score = core.n[3].min(b.n[3]) - core.n[1].max(b.n[1]);
        if best.is_none_or(|j| {
            let prior = core.n[3].min(grids[j].n[3]) - core.n[1].max(grids[j].n[1]);
            score > prior
                || (score == prior
                    && super::super::graphic_arbitration::area_key(b)
                        > super::super::graphic_arbitration::area_key(&grids[j]))
        }) {
            best = Some(i);
        }
    }
    let mut ids = prepared.ids;
    if let Some(i) = best {
        core = union(core, &grids[i]);
        local = union(local, &grids[i]);
        let key = core.n.map(f64::to_bits);
        if !cache.shared_grid_members.contains_key(&key) {
            cache.shared_grid_members.insert(
                key,
                Members::from_domain(
                    Arc::clone(&cache.core_members.as_ref().unwrap().domain),
                    &grid_members(p, &core, work)?,
                ),
            );
        }
        let members = &cache.shared_grid_members[&key];
        ids.union_with(members);
    }
    let excluded_ids: Vec<_> = annotations
        .iter()
        .flat_map(|a| a["line_indices"].as_array().unwrap())
        .map(|v| v.as_i64().unwrap())
        .filter(|id| ids.contains(id))
        .collect();
    if !excluded_ids.is_empty() {
        for id in excluded_ids {
            ids.remove(&id);
        }
    }
    let b = bbox(&super::super::rule_candidate_expansion::rotate(
        &local.raw, &p.size, p.angle,
    ))?;
    let core = bbox(&super::super::rule_candidate_expansion::rotate(
        &core.raw, &p.size, p.angle,
    ))?;
    #[cfg(test)]
    if std::env::var_os("RAPIDDOC_RULE_MATERIALIZE").is_some() {
        eprintln!("typed grid {:?}", stage.elapsed());
    }
    cache.timings.mark("rule_materialize_grid");
    let candidate = super::super::table_candidate_merge::from_parts(
        b,
        local,
        core,
        p.angle,
        d.score,
        ids,
        annotations,
        work,
    )?;
    cache.timings.mark("rule_materialize_candidate");
    Ok(candidate)
}
pub(super) fn projection(d: &Draft, p: &Page) -> Value {
    let members: BTreeSet<i64> = d
        .rows
        .iter()
        .flat_map(|r| &r.fragments)
        .map(|f| f.id)
        .collect();
    json!({"kind":"draft","angle":p.angle,"median_height":p.em,"boundaries":d.boundaries,"caption":d.caption,"member_source_indices":members})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(xs: &[f64], y: f64) -> Row {
        rowprep::row(
            &json!({"bbox":[0.,y,200.,y+2.],"center_y":y+1.,
            "fragments":xs.iter().enumerate().map(|(i,x)|json!({"bbox":[x,y,x+5.,y+2.],
                "local_bbox":[x,y,x+5.,y+2.],"line_index":i})).collect::<Vec<_>>()}),
            &mut 0,
        )
        .unwrap()
    }
    #[test]
    fn borrowed_materialization_matches_owned_reference() {
        let _scope = crate::legacy_page::budget::host_page();
        let f: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(
                "../../../../../../tests/fixtures/upstream_068/shared_supplemental_tables.json.gz"
            )[..],
        ))
        .unwrap();
        let mut count = 0;
        for case in f["cases"].as_array().unwrap() {
            let source = &case["request"]["source"];
            let p = prepare_source(source, &json!({}), &mut Work(0)).unwrap();
            let mut notes = MaterializationCache::default();
            for d in drafts(source, &p, &[], &mut Work(0))
                .unwrap()
                .iter()
                .take(8)
            {
                let borrowed = super::super::super::table_candidate_merge::into_value(
                    materialize_typed(d, &p, &[], &mut notes, &mut Work(0)).unwrap(),
                );
                let owned = crate::legacy_page::budget::with_resources(
                    crate::legacy_page::budget::HostResources {
                        max_work: Some(usize::MAX - 1),
                        ..Default::default()
                    },
                    || {
                        let _scope = crate::legacy_page::budget::host_page();
                        materialize(d, &p, &[], &mut Work(0)).unwrap()
                    },
                );
                assert_eq!(borrowed, owned);
                count += 1;
            }
        }
        assert!(count > 0);
    }
    #[test]
    fn sparse_core_masks_match_fragment_union() {
        let source: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(
                "../../../../../../tests/fixtures/upstream_068/shared_supplemental_tables.json.gz"
            )[..],
        ))
        .unwrap();
        for case in source["cases"].as_array().unwrap() {
            let _scope = crate::legacy_page::budget::host_page();
            let page =
                prepare_source(&case["request"]["source"], &json!({}), &mut Work(0)).unwrap();
            let mut index = CoreMemberIndex::new(&page);
            let rows = page.rows.iter().cloned().map(Arc::new).collect::<Vec<_>>();
            for step in [1, 2, 3] {
                for start in 0..rows.len() {
                    let selected = rows[start..]
                        .iter()
                        .step_by(step)
                        .cloned()
                        .collect::<Vec<_>>();
                    let expected = selected
                        .iter()
                        .flat_map(|r| &r.fragments)
                        .map(|f| f.id)
                        .collect::<BTreeSet<_>>();
                    assert_eq!(
                        index
                            .members(&selected)
                            .unwrap()
                            .iter()
                            .collect::<BTreeSet<_>>(),
                        expected
                    );
                }
            }
            assert!(index.members(&[]).unwrap().len() == 0);
            let mut unknown = row(&[1., 2.], 0.);
            unknown.fragments[0].id = i64::MIN;
            assert!(index.members(&[Arc::new(unknown)]).is_none());
        }
    }
    #[test]
    fn stable_column_cache_matches_uncached_prefixes_and_subsets() {
        let _scope = crate::legacy_page::budget::host_page();
        let rows: Vec<_> = (0..24)
            .map(|i| {
                row(
                    &[
                        10. + (i % 5) as f64,
                        50. + (i % 3) as f64,
                        100. + (i % 7) as f64,
                    ],
                    i as f64 * 10.,
                )
            })
            .collect();
        let mut cache = StableCache::default();
        for start in 0..rows.len() {
            for end in start + 1..=rows.len() {
                for step in [1, 2, 3] {
                    let selected: Vec<_> = rows[start..end].iter().step_by(step).collect();
                    assert_eq!(
                        cache.count(&selected, 4., &mut Work(0)).unwrap(),
                        stable(&selected, 4., &mut Work(0)).unwrap()
                    );
                }
            }
        }
        assert_eq!(cache.count(&[], 4., &mut Work(0)).unwrap(), (0, 0.));
    }
    #[test]
    fn corridor_cache_preserves_admission_before_exclusion() {
        let _scope = crate::legacy_page::budget::host_page();
        let q = json!({"page_size":[400,600],"lines":[],"drawing_lines":[]});
        let mut p = prepare_source(&q, &json!({}), &mut Work(0)).unwrap();
        let mut r = row(&[10., 50.], 10.);
        r.fragments[1].local = bbox(&json!([50, 20, 55, 22])).unwrap();
        r.fragments[1].b = r.fragments[1].local.clone();
        r.b = bbox(&json!([0, 10, 200, 22])).unwrap();
        p.rows = vec![r];
        let horizontal = bbox(&json!([0, -1000, 200, 1000])).unwrap();
        let excluded = vec![bbox(&json!([45, 19, 60, 23])).unwrap()];
        let cached = prepare_corridor(&p, &horizontal, &excluded, &mut Work(0)).unwrap();
        for (top, bottom) in [(14, 18), (10, 12), (16, 16), (0, 30)] {
            let b = bbox(&json!([0, top, 200, bottom])).unwrap();
            let actual: Vec<_> = cached
                .iter()
                .filter(|r| {
                    r.source.n[1] <= b.n[3]
                        && r.source.n[3] >= b.n[1]
                        && r.gate_center >= b.n[1]
                        && r.gate_center <= b.n[3]
                })
                .map(|r| rowprep::output(&r.row))
                .collect();
            assert_eq!(actual, corridor(&p, &b, &excluded, &mut Work(0)).unwrap());
        }
    }
    #[test]
    fn prepared_band_partition_keeps_boundary_membership() {
        let _scope = crate::legacy_page::budget::host_page();
        let q = json!({"page_size":[400,600],"lines":[],"drawing_lines":[{"bbox":[0,10,200,10]},{"bbox":[0,20,200,20]},{"bbox":[0,30,200,30]}]});
        let p = prepare_source(&q, &json!({}), &mut Work(0)).unwrap();
        let group = groups(&p, &mut Work(0))
            .unwrap()
            .into_iter()
            .find(|g| g.len() >= 3)
            .unwrap();
        let mut rows: Vec<_> = group
            .iter()
            .flat_map(|i| {
                let y = center(&p.p.rules[*i].b, 1);
                [row(&[10., 50.], y - 1.), row(&[10., 50.], y - 0.5)]
            })
            .collect();
        rows.reverse();
        let refs: Vec<_> = rows.iter().collect();
        let actual = partition(&refs, &group, &p, &mut Work(0)).unwrap();
        for (i, pair) in group.windows(2).enumerate() {
            let expected: Vec<_> = refs
                .iter()
                .copied()
                .filter(|r| {
                    center(&p.p.rules[pair[0]].b, 1) <= r.center
                        && r.center <= center(&p.p.rules[pair[1]].b, 1)
                })
                .map(rowprep::output)
                .collect();
            assert_eq!(
                actual[i]
                    .iter()
                    .map(|r| rowprep::output(r))
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }
}
