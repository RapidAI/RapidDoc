//! Original four-direction nearest-content flood fill and auto-gap penalties.
use super::*;
use std::collections::VecDeque;
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Region {
    pub cells: BTreeSet<Position>,
    pub row_start: u32,
    pub row_end: u32,
    pub col_start: u32,
    pub col_end: u32,
}
impl Region {
    pub fn rows(&self) -> u32 {
        self.row_end - self.row_start + 1
    }
    pub fn cols(&self) -> u32 {
        self.col_end - self.col_start + 1
    }
}
pub fn discover(sheet: &Sheet, gap: u32) -> Vec<Region> {
    let bounds = sheet.bounds();
    let (mut visited, mut regions) = (BTreeSet::new(), vec![]);
    for start in sheet.positions() {
        if visited.contains(&start) {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        let mut region = Region {
            cells: BTreeSet::from([start]),
            row_start: start.0,
            row_end: start.0,
            col_start: start.1,
            col_end: start.1,
        };
        while let Some((r, c)) = queue.pop_front() {
            region.row_start = region.row_start.min(r);
            region.row_end = region.row_end.max(r);
            region.col_start = region.col_start.min(c);
            region.col_end = region.col_end.max(c);
            for (dr, dc) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                for step in 1..=u64::from(gap) + 1 {
                    let nr = i64::from(r) + dr * step as i64;
                    let nc = i64::from(c) + dc * step as i64;
                    if nr < 0 || nc < 0 || nr > bounds[1] as i64 || nc > bounds[3] as i64 {
                        continue;
                    }
                    let p = (nr as u32, nc as u32);
                    if region.cells.contains(&p) {
                        break;
                    }
                    if sheet.has_content(p.0, p.1) {
                        region.cells.insert(p);
                        queue.push_back(p);
                        break;
                    }
                }
            }
        }
        visited.extend(region.cells.iter().copied());
        regions.push(region);
    }
    regions
}
#[derive(Debug, Default, Clone, Serialize)]
pub struct Summary {
    pub real_singleton_ratio: f64,
    pub weighted_blank_ratio: f64,
    pub interior_blank_line_ratio: f64,
    pub sparse_large_table_ratio: f64,
    pub severe_separator_count: u64,
    pub row_overlap_excess_ratio: f64,
}
fn consecutive(flags: &[bool]) -> usize {
    let (mut current, mut best) = (0, 0);
    for b in flags {
        if *b {
            current += 1;
            best = best.max(current);
        } else {
            current = 0;
        }
    }
    best
}
pub fn summarize(sheet: &Sheet, regions: &[Region]) -> Summary {
    let (
        mut singleton,
        mut separators,
        mut sparse,
        mut area,
        mut blank,
        mut interior,
        mut possible,
    ) = (0, 0, 0, 0u64, 0., 0u64, 0u64);
    let mut rowcounts = BTreeMap::<u32, u64>::new();
    for region in regions {
        let nr = region.rows() as usize;
        let nc = region.cols() as usize;
        let size = nr as u64 * nc as u64;
        let mut mask = vec![vec![false; nc]; nr];
        for r in region.row_start..=region.row_end {
            for c in region.col_start..=region.col_end {
                if sheet.semantic(r, c) {
                    let (rs, cs) = sheet.span(r, c);
                    for rr in r..(r.saturating_add(rs)).min(region.row_end.saturating_add(1)) {
                        for cc in c..(c.saturating_add(cs)).min(region.col_end.saturating_add(1)) {
                            mask[(rr - region.row_start) as usize]
                                [(cc - region.col_start) as usize] = true;
                        }
                    }
                }
            }
        }
        let filled = mask.iter().flatten().filter(|v| **v).count();
        let ratio = 1. - filled as f64 / (size.max(1)) as f64;
        let rows = (1..nr.saturating_sub(1))
            .map(|r| !mask[r].iter().any(|v| *v))
            .collect::<Vec<_>>();
        let cols = (1..nc.saturating_sub(1))
            .map(|c| !mask.iter().any(|r| r[c]))
            .collect::<Vec<_>>();
        area += size;
        blank += size as f64 * ratio;
        interior += rows.iter().chain(&cols).filter(|v| **v).count() as u64;
        possible += nr.saturating_sub(2) as u64 + nc.saturating_sub(2) as u64;
        for r in region.row_start..=region.row_end {
            *rowcounts.entry(r).or_default() += 1;
        }
        if nr == 1 && nc == 1 && sheet.span(region.row_start, region.col_start) == (1, 1) {
            singleton += 1;
        }
        if size >= 6 && ratio > 0.35 {
            sparse += 1;
        }
        if consecutive(&rows).max(consecutive(&cols)) >= 2 {
            separators += 1;
        }
    }
    let n = regions.len().max(1) as f64;
    Summary {
        real_singleton_ratio: singleton as f64 / n,
        weighted_blank_ratio: blank / area.max(1) as f64,
        interior_blank_line_ratio: interior as f64 / possible.max(1) as f64,
        sparse_large_table_ratio: sparse as f64 / n,
        severe_separator_count: separators,
        row_overlap_excess_ratio: rowcounts.values().map(|v| v.saturating_sub(1)).sum::<u64>()
            as f64
            / rowcounts.len().max(1) as f64,
    }
}
pub fn penalty(s: &Summary) -> f64 {
    6. * s.severe_separator_count as f64
        + 2.5 * s.interior_blank_line_ratio
        + 1.5 * s.sparse_large_table_ratio
        + s.real_singleton_ratio
        + 0.5 * s.weighted_blank_ratio
        + s.row_overlap_excess_ratio
}
pub fn select(sheet: &Sheet) -> (u32, f64, Vec<Region>) {
    let mut candidates = (0..=2)
        .map(|gap| {
            let regions = discover(sheet, gap);
            let summary = summarize(sheet, &regions);
            let score = penalty(&summary);
            (gap, score, regions, summary)
        })
        .collect::<Vec<_>>();
    let min = candidates.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
    let index = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.1 <= min + 0.15)
        .min_by(|(_, a), (_, b)| {
            a.3.severe_separator_count
                .cmp(&b.3.severe_separator_count)
                .then_with(|| {
                    let preference = |gap| match gap {
                        1 => 0,
                        0 => 1,
                        _ => 2,
                    };
                    preference(a.0).cmp(&preference(b.0))
                })
                .then_with(|| {
                    a.3.interior_blank_line_ratio
                        .total_cmp(&b.3.interior_blank_line_ratio)
                })
                .then_with(|| a.1.total_cmp(&b.1))
        })
        .unwrap()
        .0;
    let (gap, score, regions, _) = candidates.swap_remove(index);
    (gap, score, regions)
}
