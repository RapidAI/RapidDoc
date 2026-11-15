use super::*;
use crate::legacy_model::table_direction;
use rapiddoc_inference::table_raster;
use std::collections::BTreeSet;

type Box4 = [f64; 4];
type Pair = (usize, usize, usize, bool);

pub(super) struct GridCell {
    pub bbox: Box4,
    pub logic: [usize; 4],
    pub tokens: Vec<usize>,
}
pub(super) struct Grid {
    pub cells: Vec<GridCell>,
    pub verified_splits: usize,
    pub width: usize,
    pub legacy: Option<Box<Grid>>,
}
impl Grid {
    pub fn new(cells: Vec<GridCell>) -> Option<Self> {
        let width = cells.iter().map(|c| c.logic[3]).max()?.checked_add(1)?;
        if cells.iter().any(|c| {
            c.bbox.iter().any(|v| !v.is_finite())
                || c.logic[0] > c.logic[1]
                || c.logic[2] > c.logic[3]
        }) {
            return None;
        }
        Some(Self {
            cells,
            verified_splits: 0,
            width,
            legacy: None,
        })
    }
}

pub(super) fn cell_bounds(b: &[f64]) -> Option<Box4> {
    match b.len() {
        4 => b.try_into().ok(),
        8 => {
            let points: Vec<_> = b.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
            Some(bounds(&points))
        }
        _ => None,
    }
}
fn bounds(points: &[[f64; 2]]) -> Box4 {
    points.iter().fold(
        [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
        |b, p| {
            [
                b[0].min(p[0]),
                b[1].min(p[1]),
                b[2].max(p[0]),
                b[3].max(p[1]),
            ]
        },
    )
}
fn center(b: Box4, axis: usize) -> f64 {
    (b[axis] + b[axis + 2]) / 2.
}
fn overlap(a: Box4, b: Box4, axis: usize) -> f64 {
    a[axis + 2].min(b[axis + 2]) - a[axis].max(b[axis])
}
fn coverage(b: Box4, cell: Box4) -> f64 {
    overlap(b, cell, 0).max(0.) * overlap(b, cell, 1).max(0.) / ((b[2] - b[0]) * (b[3] - b[1]))
}
#[derive(Clone, Copy)]
pub(super) struct Segment(pub usize, pub f64, pub f64, pub f64);

pub(super) struct Token {
    pub id: usize,
    pub bbox: Box4,
    pub confidence: f64,
    numeric: bool,
}
pub(super) struct Source {
    pub tokens: Vec<Token>,
    legacy_tokens: Vec<Token>,
    legacy_em: f64,
    pub em: f64,
    pairs: Vec<Pair>,
    lines: Vec<Segment>,
    pub ruling: [Vec<(f64, f64, f64)>; 2],
}
#[derive(Default)]
pub(super) struct Score {
    text: BTreeMap<usize, f64>,
    gaps: BTreeMap<Pair, f64>,
    lines: BTreeMap<(usize, usize), f64>,
    pub missing: std::collections::BTreeSet<usize>,
    pub missing_numbers: std::collections::BTreeSet<usize>,
}
fn delta<K: Ord>(main: &BTreeMap<K, f64>, alt: &BTreeMap<K, f64>) -> f64 {
    alt.iter()
        .filter_map(|(k, v)| main.get(k).map(|m| m - v))
        .sum::<f64>()
        / (alt.len().max(4) as f64).sqrt()
}
impl Score {
    pub fn text_penalty(&self) -> f64 {
        self.text.values().sum()
    }

    pub fn suspicious(&self) -> bool {
        self.text_penalty() >= 2f64.max(0.025 * self.text.len() as f64)
            || self.lines.values().sum::<f64>() >= 1.3
            || self.gaps.values().sum::<f64>() >= 2.
    }
    pub fn delta(&self, main: &Self) -> [f64; 3] {
        [
            delta(&main.text, &self.text),
            delta(&main.lines, &self.lines),
            delta(&main.gaps, &self.gaps),
        ]
    }
    pub fn line_gain(&self, main: &Self, axis: Option<usize>) -> f64 {
        let changes: Vec<_> = self
            .lines
            .iter()
            .filter_map(|(key, v)| {
                let change = main.lines.get(key)? - v;
                (axis.map_or(change.abs() >= 0.15, |a| key.0 == a)).then_some(change)
            })
            .collect();
        changes.iter().sum::<f64>()
            / (changes.len().max(if axis.is_some() { 4 } else { 1 }) as f64).sqrt()
    }
    pub fn separator_gains(&self, main: &Self, axis: usize) -> usize {
        self.gaps
            .iter()
            .filter(|(k, v)| {
                k.0 == axis && k.3 && **v == 0. && main.gaps.get(k).is_some_and(|m| *m > 0.)
            })
            .count()
    }
    pub fn loses_supported_lines(&self, main: &Self, line_gain: f64) -> bool {
        line_gain <= -0.8
            && main
                .lines
                .iter()
                .filter(|(key, v)| {
                    **v <= 0.1 && self.lines.get(key).is_some_and(|after| after - *v >= 0.6)
                })
                .take(4)
                .count()
                >= 4
    }
    pub fn report(&self) -> Value {
        json!({"text":self.text.values().sum::<f64>(),"lines":self.lines.values().sum::<f64>(),
               "gaps":self.gaps.values().sum::<f64>(),"missing_numbers":self.missing_numbers})
    }
}

impl Source {
    pub fn stacked_boundary(&self, rows: &Value, height: f64, width: f64) -> Option<u32> {
        let mut horizontal = self.ruling[1].clone();
        horizontal.sort_by(|a, b| a.0.total_cmp(&b.0));
        let full: Vec<_> = horizontal
            .iter()
            .filter_map(|&(y, l, r)| (r - l >= 0.85 * width).then_some(y))
            .collect();
        let partial: Vec<_> = horizontal
            .into_iter()
            .filter(|&(_, l, r)| (0.12 * width..=0.5 * width).contains(&(r - l)))
            .collect();
        if full.len() < 4 || partial.len() < 4 {
            return None;
        }
        let first_after = |top| full.iter().copied().find(|&y| y > top + self.em);
        let headers = |top: f64, bottom: f64| {
            for &(y, _, _) in &partial {
                if !(top < y && y < bottom.min(top + 2. * self.em)) {
                    continue;
                }
                let mut spans: Vec<_> = partial
                    .iter()
                    .filter_map(|&(pos, l, r)| {
                        ((pos - y).abs() <= 0.15 * self.em).then_some((l, r))
                    })
                    .collect();
                spans.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut groups: Vec<(f64, f64)> = Vec::new();
                for (l, r) in spans {
                    if let Some(last) = groups.last_mut().filter(|last| l < last.1) {
                        last.1 = last.1.max(r);
                    } else {
                        groups.push((l, r));
                    }
                }
                if groups.len() < 2 {
                    continue;
                }
                let labels: Vec<Vec<String>> = groups
                    .iter()
                    .map(|&(l, r)| {
                        let mut tokens: Vec<_> = self
                            .tokens
                            .iter()
                            .filter(|t| {
                                let (x, cy) = (center(t.bbox, 0), center(t.bbox, 1));
                                l <= x && x <= r && y < cy && cy < bottom
                            })
                            .collect();
                        tokens.sort_by(|a, b| a.bbox[0].total_cmp(&b.bbox[0]));
                        tokens
                            .iter()
                            .map(|t| {
                                rows[1][t.id]
                                    .as_str()
                                    .unwrap_or_default()
                                    .chars()
                                    .filter(|c| !c.is_whitespace())
                                    .flat_map(char::to_lowercase)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                if labels[0].len() >= 2 && labels.iter().all(|label| *label == labels[0]) {
                    return Some((labels[0].clone(), labels.len()));
                }
            }
            None
        };
        if full[0] > 2. * self.em {
            return None;
        }
        let (upper, n) = headers(full[0], first_after(full[0])?)?;
        let mut candidates = full.windows(2).filter_map(|p| {
            let cut = (p[0] + p[1]) / 2.;
            if p[1] - p[0] > 0.5 * self.em
                || cut.min(height - cut) < 5. * self.em
                || self.ruling[0].iter().any(|&(x, l, r)| {
                    0.1 * width < x
                        && x < 0.9 * width
                        && l < cut - 0.5 * self.em
                        && r > cut + 0.5 * self.em
                })
                || self
                    .tokens
                    .iter()
                    .any(|t| t.bbox[1] < cut && cut < t.bbox[3])
            {
                return None;
            }
            let (lower, m) = headers(p[1], first_after(p[1])?)?;
            (lower == upper && m != n).then_some(cut.round() as u32)
        });
        let cut = candidates.next()?;
        candidates.next().is_none().then_some(cut)
    }

    pub fn new(
        png: &[u8],
        rows: &Value,
        excluded: &[String],
        vector: &[Segment],
    ) -> Result<Self, Error> {
        let (width, height) = table_direction::dimensions(png)?;
        let mut tokens = Vec::new();
        let mut legacy_tokens = Vec::new();
        for (i, points) in array(&rows[0])?.iter().enumerate() {
            let text = rows[1][i].as_str().unwrap_or_default();
            let confidence = rows[2][i].as_f64().unwrap_or_default();
            if text.trim().is_empty() || confidence < 0.5 || excluded.iter().any(|e| e == text) {
                continue;
            }
            let points: Vec<[f64; 2]> = serde_json::from_value(points.clone()).map_err(error)?;
            let [x, y, r, b] = bounds(&points);
            if x < r && y < b {
                legacy_tokens.push(Token {
                    id: i,
                    bbox: [x, y, r, b],
                    confidence,
                    numeric: false,
                });
            }
            let bbox = [
                x.clamp(0., width as f64),
                y.clamp(0., height as f64),
                r.clamp(0., width as f64),
                b.clamp(0., height as f64),
            ];
            if bbox[0] < bbox[2] && bbox[1] < bbox[3] {
                tokens.push(Token {
                    id: i,
                    bbox,
                    confidence,
                    numeric: confidence >= 0.85
                        && rapiddoc_inference::table_transformer::is_numeric(text.trim()),
                });
            }
        }
        let mut heights: Vec<_> = tokens.iter().map(|t| t.bbox[3] - t.bbox[1]).collect();
        heights.sort_by(f64::total_cmp);
        let em = if heights.is_empty() {
            10.
        } else {
            let n = heights.len();
            ((heights[(n - 1) / 2] + heights[n / 2]) / 2.).max(2.)
        };
        let mut raw_heights: Vec<_> = legacy_tokens
            .iter()
            .map(|t| t.bbox[3] - t.bbox[1])
            .collect();
        raw_heights.sort_by(f64::total_cmp);
        let legacy_em = if raw_heights.is_empty() {
            10.
        } else {
            (raw_heights[(raw_heights.len() - 1) / 2] + raw_heights[raw_heights.len() / 2]) / 2.
        };
        let pairs = gutters(&tokens, em);
        let mut lines: Vec<Segment> = Vec::new();
        let mut ruling = [Vec::new(), Vec::new()];
        let mut tracks: Vec<(usize, f64, Vec<(f64, f64)>)> = Vec::new();
        {
            let image = image::load_from_memory(png).map_err(error)?.into_rgb8();
            let raster =
                table_raster::evidence_rules(image.as_raw(), width as usize, height as usize, em);
            let raster = raster.into_iter().map(|rule| {
                let axis = usize::from(rule.orientation == "horizontal");
                let b = rule.bbox;
                Segment(axis, center(b, axis), b[1 - axis], b[3 - axis])
            });
            // 矢量和栅格都映射到当前裁图坐标。同一轨迹只合并相交区段。
            for Segment(axis, pos, lo, hi) in vector.iter().copied().chain(raster) {
                let limits = [width as f64, height as f64];
                let lo = lo.max(0.);
                let hi = hi.min(limits[1 - axis]);
                if pos <= 0.05 * em || pos >= limits[axis] - 0.05 * em || hi - lo < 2.5 * em {
                    continue;
                }
                ruling[axis].push((pos, lo, hi));
                if tokens.len() < 2 {
                    continue;
                }
                let (mut before, mut after, mut crosses) = (0, 0, 0);
                for token in &tokens {
                    let c = center(token.bbox, 1 - axis);
                    if c < lo - em || c > hi + em {
                        continue;
                    }
                    let c = center(token.bbox, axis);
                    before += usize::from(c < pos - 0.25 * em);
                    after += usize::from(c > pos + 0.25 * em);
                    crosses += usize::from(
                        token.bbox[axis] < pos - 0.2 * em && token.bbox[axis + 2] > pos + 0.2 * em,
                    );
                }
                if before == 0 || after == 0 || crosses > before.min(after) {
                    continue;
                }
                if let Some((_, _, parts)) = tracks
                    .iter_mut()
                    .find(|(a, p, _)| *a == axis && (*p - pos).abs() <= 0.2 * em)
                {
                    parts.push((lo, hi));
                } else {
                    tracks.push((axis, pos, vec![(lo, hi)]));
                }
            }
            for (axis, pos, mut parts) in tracks {
                parts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
                for (mut lo, mut hi) in parts {
                    if let Some(last) = lines.last() {
                        if last.0 == axis && last.1 == pos && lo <= last.3 {
                            lo = last.2;
                            hi = hi.max(last.3);
                            lines.pop();
                        }
                    }
                    lines.push(Segment(axis, pos, lo, hi));
                }
            }
        }
        Ok(Self {
            tokens,
            legacy_tokens,
            legacy_em,
            em,
            pairs,
            lines,
            ruling,
        })
    }
    pub fn ruled_empty_rows(&self, grid: &Grid, empty_indices: &[usize]) -> BTreeSet<usize> {
        let mut supported = BTreeSet::new();
        if empty_indices.is_empty() || self.ruling.iter().any(Vec::is_empty) {
            return supported;
        }
        let mut by_row = BTreeMap::<usize, Vec<Box4>>::new();
        for c in &grid.cells {
            let row = c.logic[0];
            if row == c.logic[1] && empty_indices.binary_search(&row).is_ok() {
                by_row.entry(row).or_default().push(c.bbox);
            }
        }
        for (row, cells) in by_row {
            let median = |axis: usize| {
                let mut values: Vec<_> = cells.iter().map(|c| c[axis]).collect();
                values.sort_by(f64::total_cmp);
                (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.
            };
            let (top, bottom) = (median(1), median(3));
            let left = cells.iter().map(|c| c[0]).fold(f64::INFINITY, f64::min);
            let right = cells.iter().map(|c| c[2]).fold(f64::NEG_INFINITY, f64::max);
            if right <= left || bottom <= top {
                continue;
            }
            let tolerance = (0.4 * self.em).min(0.25 * (bottom - top));
            let at = |lo: f64, hi: f64| {
                self.ruling[1]
                    .iter()
                    .copied()
                    .filter(move |(y, _, _)| lo <= *y && *y <= hi)
            };
            // 分割框常止于线的内侧，按行高容许向外的小间隙。
            let outer = 0.3 * (bottom - top);
            let min_width = (2.5 * self.em).max(0.5 * (right - left));
            // 空白填写格用原图交叉框线验证，不要求边界两侧各有 OCR。
            let ruled = at(top - outer, top + tolerance).any(|(_, a, b)| {
                at(bottom - tolerance, bottom + outer).any(|(_, c, d)| {
                    let (lo, hi) = (
                        a.max(c).max(left - tolerance),
                        b.min(d).min(right + tolerance),
                    );
                    if hi - lo < min_width {
                        return false;
                    }
                    let (mut first, mut last) = (f64::INFINITY, f64::NEG_INFINITY);
                    for &(x, y0, y1) in &self.ruling[0] {
                        if lo - tolerance <= x
                            && x <= hi + tolerance
                            && y0 <= top + tolerance
                            && y1 >= bottom - tolerance
                        {
                            first = first.min(x);
                            last = last.max(x);
                        }
                    }
                    last - first >= 2.5 * self.em
                })
            });
            if ruled {
                supported.insert(row);
            }
        }
        supported
    }
    pub fn unsupported_row_splits(&self, main: &Grid, candidate: &Grid) -> usize {
        let alternate_cells = unique_token_cells(candidate);
        let tracks = boundary_tracks(candidate, 1, self.em);
        let mut witnesses = BTreeMap::<usize, Vec<(f64, f64, f64, f64)>>::new();
        let tokens: BTreeMap<_, _> = self.tokens.iter().map(|token| (token.id, token)).collect();
        for cell in &main.cells {
            if cell.logic[0] != cell.logic[1] {
                continue;
            }
            let mut assigned: Vec<_> = cell
                .tokens
                .iter()
                .filter_map(|id| {
                    let token = tokens.get(id)?;
                    let alternative = candidate
                        .cells
                        .get(alternate_cells.get(id).copied().flatten()?)?;
                    (alternative.logic[0] == alternative.logic[1]
                        && coverage(token.bbox, cell.bbox) >= 0.65)
                        .then_some((token.bbox, alternative.logic[0]))
                })
                .collect();
            assigned.sort_by(|a, b| center(a.0, 1).total_cmp(&center(b.0, 1)));
            for pair in assigned.windows(2) {
                let [(upper, before), (lower, after)] = [pair[0], pair[1]];
                if after != before.saturating_add(1)
                    || center(lower, 1) - center(upper, 1) < 0.5 * self.em
                {
                    continue;
                }
                // 用同一主候选单元格的相邻文字行定位争议边界。局部原图横线
                // 或多个格子共同存在的宽间距，才为该边界提供独立证据。
                witnesses.entry(after).or_default().push((
                    upper[3],
                    lower[1],
                    cell.bbox[0],
                    cell.bbox[2],
                ));
            }
        }
        witnesses
            .iter()
            .filter(|&(&row, segments)| {
                if segments.len() < 2 {
                    return false;
                }
                // 有内容的跨行格为邻列的一组子记录提供锚点，例如分组名称与多条明细。
                if candidate.cells.iter().any(|cell| {
                    let [start, end, left_col, right_col] = cell.logic;
                    let within_group = start < row && row <= end;
                    let starts_next_group = start == row
                        && end > row
                        && candidate.cells.iter().any(|prior| {
                            prior.logic[0] < prior.logic[1]
                                && prior.logic[1].checked_add(1) == Some(row)
                                && prior.logic[2..] == [left_col, right_col]
                                && prior.tokens.iter().any(|id| tokens.contains_key(id))
                        });
                    (within_group || starts_next_group)
                        && cell.tokens.iter().any(|id| tokens.contains_key(id))
                        && segments.iter().all(|&(_, _, left, right)| {
                            cell.bbox[2].min(right) <= cell.bbox[0].max(left) + 0.2 * self.em
                        })
                }) {
                    return false;
                }
                let local_line = |&(top, bottom, left, right): &(f64, f64, f64, f64)| {
                    self.ruling[1].iter().any(|&(y, lo, hi)| {
                        y >= top.min(bottom) - 0.2 * self.em
                            && y <= top.max(bottom) + 0.2 * self.em
                            && hi.min(right) - lo.max(left) >= 0.5 * (right - left)
                            && tracks.iter().any(|&(edge, a, b)| {
                                (edge - y).abs() <= 0.5 * self.em
                                    && b.min(hi).min(right) > a.max(lo).max(left)
                            })
                    })
                };
                let ruled = segments
                    .iter()
                    .filter(|segment| local_line(segment))
                    .count();
                let gap_top = segments
                    .iter()
                    .map(|s| s.0)
                    .fold(f64::NEG_INFINITY, f64::max);
                let gap_bottom = segments.iter().map(|s| s.1).fold(f64::INFINITY, f64::min);
                ruled * 2 < segments.len() && gap_bottom - gap_top < 0.55 * self.em
            })
            .count()
    }

    pub fn score(&self, grid: &Grid) -> Score {
        let mut result = Score::default();
        let mut assignments = BTreeMap::<usize, Vec<usize>>::new();
        for (i, c) in grid.cells.iter().enumerate() {
            for &id in &c.tokens {
                assignments.entry(id).or_default().push(i);
            }
        }
        for t in &self.tokens {
            let penalty = match assignments.get(&t.id).map(Vec::as_slice) {
                Some([i]) => {
                    let ratio = coverage(t.bbox, grid.cells[*i].bbox);
                    if ratio < 0.7 {
                        t.confidence * (1. - ratio)
                    } else {
                        0.
                    }
                }
                Some(_) => 0.,
                None => {
                    result.missing.insert(t.id);
                    if t.numeric {
                        result.missing_numbers.insert(t.id);
                    }
                    t.confidence
                }
            };
            result.text.insert(t.id, penalty);
        }
        for &key @ (axis, a, b, separated) in &self.pairs {
            let (a, b) = (&self.tokens[a], &self.tokens[b]);
            let (Some(left), Some(right)) = (assignments.get(&a.id), assignments.get(&b.id)) else {
                continue;
            };
            let start = if axis == 0 { 2 } else { 0 };
            let extent = |ids: &[usize]| {
                (
                    ids.iter()
                        .map(|&i| grid.cells[i].logic[start])
                        .min()
                        .unwrap(),
                    ids.iter()
                        .map(|&i| grid.cells[i].logic[start + 1])
                        .max()
                        .unwrap(),
                )
            };
            let (l, r) = (extent(left), extent(right));
            let conflict = if separated {
                l.1 >= r.0
            } else {
                l.1 < r.0 || r.1 < l.0
            };
            result.gaps.insert(
                key,
                if conflict {
                    a.confidence.min(b.confidence)
                } else {
                    0.
                },
            );
        }
        let tracks = [
            boundary_tracks(grid, 0, self.em),
            boundary_tracks(grid, 1, self.em),
        ];
        for (i, &Segment(axis, pos, lo, hi)) in self.lines.iter().enumerate() {
            let mut intervals: Vec<_> = tracks[axis]
                .iter()
                .filter_map(|&(x, a, b)| {
                    let (a, b) = (lo.max(a), hi.min(b));
                    ((x - pos).abs() <= 0.5 * self.em && a < b).then_some((a, b))
                })
                .collect();
            intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (mut end, mut total) = (lo, 0.);
            for (a, b) in intervals {
                if b > end {
                    total += b - a.max(end);
                    end = b;
                }
            }
            let weight = ((hi - lo) / (5. * self.em)).min(1.);
            result
                .lines
                .insert((axis, i), weight * (0.8 - total / (hi - lo)).max(0.));
        }
        result
    }
}

fn unique_token_cells(grid: &Grid) -> BTreeMap<usize, Option<usize>> {
    let mut cells = BTreeMap::new();
    for (index, cell) in grid.cells.iter().enumerate() {
        for &id in &cell.tokens {
            cells
                .entry(id)
                .and_modify(|prior| {
                    if *prior != Some(index) {
                        *prior = None;
                    }
                })
                .or_insert(Some(index));
        }
    }
    cells
}

fn boundary_tracks(grid: &Grid, axis: usize, em: f64) -> Vec<(f64, f64, f64)> {
    let index = if axis == 0 { 2 } else { 0 };
    let mut tracks = BTreeMap::<usize, (Vec<Box4>, Vec<Box4>)>::new();
    for c in &grid.cells {
        tracks.entry(c.logic[index]).or_default().1.push(c.bbox);
        tracks
            .entry(c.logic[index + 1] + 1)
            .or_default()
            .0
            .push(c.bbox);
    }
    let mut out = Vec::new();
    for (before, mut after) in tracks.into_values() {
        let along = 1 - axis;
        after.sort_by(|a, b| a[along].total_cmp(&b[along]));
        for a in before {
            let n = after.partition_point(|b| b[along] < a[along + 2]);
            for b in &after[..n] {
                let (lo, hi) = (a[along].max(b[along]), a[along + 2].min(b[along + 2]));
                if lo < hi && (a[axis + 2] - b[axis]).abs() <= 2. * em {
                    out.push(((a[axis + 2] + b[axis]) / 2., lo, hi));
                }
            }
        }
    }
    out
}

fn gutters(tokens: &[Token], em: f64) -> Vec<Pair> {
    let mut order: Vec<_> = (0..tokens.len()).collect();
    let bbox = |i: usize| tokens[i].bbox;
    order.sort_by(|&a, &b| center(bbox(a), 1).total_cmp(&center(bbox(b), 1)));
    let mut bands: Vec<(f64, Vec<usize>)> = Vec::new();
    for i in order {
        let y = center(bbox(i), 1);
        if bands.last().is_none_or(|b| y - b.0 > 0.45 * em) {
            bands.push((y, Vec::new()));
        }
        bands.last_mut().unwrap().1.push(i);
    }
    let mut pairs = Vec::new();
    let mut gaps = Vec::new();
    for (row, (_, band)) in bands.iter_mut().enumerate() {
        band.sort_by(|&a, &b| bbox(a)[0].total_cmp(&bbox(b)[0]));
        for p in band.windows(2) {
            let (a, b) = (bbox(p[0]), bbox(p[1]));
            let gap = b[0] - a[2];
            if gap >= 0. && overlap(a, b, 1) >= 0.5 * (a[3] - a[1]).min(b[3] - b[1]) {
                pairs.push((1, p[0], p[1], false));
            }
            if gap >= 0.45 * em {
                gaps.push((row, a[2], b[0], p[0], p[1]));
            }
        }
    }
    for &(row, lo, hi, a, b) in &gaps {
        let start = gaps.partition_point(|g| g.0 < row.saturating_sub(6));
        let end = gaps.partition_point(|g| g.0 <= row + 6);
        let support: std::collections::BTreeSet<_> = gaps[start..end]
            .iter()
            .filter_map(|&(r, l, h, _, _)| {
                (hi.min(h) - lo.max(l) >= 0.3 * em
                    && (hi - h).abs().min((lo - l).abs()) <= 0.65 * em)
                    .then_some(r)
            })
            .collect();
        if support.len() >= 3 {
            pairs.push((0, a, b, true));
        }
    }
    for w in bands.windows(2) {
        if w[1].0 - w[0].0 < 1.2 * em {
            continue;
        }
        let mut aligned = Vec::new();
        for &a in &w[0].1 {
            let l = bbox(a);
            let mut matches = w[1].1.iter().copied().filter(|&b| {
                let r = bbox(b);
                r[1] - l[3] >= 0.12 * em
                    && overlap(l, r, 0) > 0.
                    && (l[0] - r[0])
                        .abs()
                        .min((l[2] - r[2]).abs())
                        .min((center(l, 0) - center(r, 0)).abs())
                        < 0.45 * em
            });
            if let Some(b) = matches.next() {
                if matches.next().is_none() {
                    aligned.push((1, a, b, true));
                }
            }
        }
        if aligned
            .first()
            .is_some_and(|p| aligned.iter().any(|q| p.2 != q.2))
        {
            pairs.extend(aligned);
        }
    }
    pairs
}
impl Source {
    // 旧检查在原候选网格上以重叠率近似归属；与新评分共享 OCR 实例。
    pub fn legacy_grid(
        &self,
        grid: &Grid,
    ) -> (
        std::collections::BTreeSet<(usize, usize)>,
        std::collections::BTreeSet<(usize, usize)>,
    ) {
        let grid = grid.legacy.as_deref().unwrap_or(grid);
        let assignments: Vec<_> = self
            .legacy_tokens
            .iter()
            .map(|t| {
                grid.cells
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, coverage(t.bbox, c.bbox)))
                    .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
                    .filter(|(_, v)| *v >= 0.65)
                    .map(|(i, _)| i)
            })
            .collect();
        let mut order: Vec<_> = (0..self.legacy_tokens.len()).collect();
        order.sort_by(|&a, &b| {
            center(self.legacy_tokens[a].bbox, 1).total_cmp(&center(self.legacy_tokens[b].bbox, 1))
        });
        let mut bands: Vec<(f64, Vec<usize>)> = Vec::new();
        for i in order {
            let y = center(self.legacy_tokens[i].bbox, 1);
            if bands.last().is_none_or(|b| y - b.0 > 0.35 * self.legacy_em) {
                bands.push((y, Vec::new()));
            }
            bands.last_mut().unwrap().1.push(i);
        }
        let mut gaps = Vec::new();
        for (row, (_, band)) in bands.iter_mut().enumerate() {
            band.sort_by(|&a, &b| {
                self.legacy_tokens[a].bbox[0].total_cmp(&self.legacy_tokens[b].bbox[0])
            });
            for pair in band.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                if self.legacy_tokens[b].bbox[0] - self.legacy_tokens[a].bbox[2]
                    >= 0.5 * self.legacy_em
                {
                    gaps.push((self.legacy_tokens[b].bbox[0], row, a, b));
                }
            }
        }
        gaps.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
                .then(a.3.cmp(&b.3))
        });
        let (mut conflicts, mut supported) = (
            std::collections::BTreeSet::new(),
            std::collections::BTreeSet::new(),
        );
        let mut check = |a: usize, b: usize, axis: usize| {
            let (Some(ca), Some(cb)) = (assignments[a], assignments[b]) else {
                return;
            };
            let pair = (self.legacy_tokens[a].id, self.legacy_tokens[b].id);
            if ca == cb {
                conflicts.insert(pair);
            } else {
                let k = if axis == 0 { 2 } else { 0 };
                if grid.cells[ca].logic[k + 1] < grid.cells[cb].logic[k] {
                    supported.insert(pair);
                }
            }
        };
        for &(x, _, a, b) in &gaps {
            let start = gaps.partition_point(|g| g.0 < x - 0.6 * self.legacy_em);
            let end = gaps.partition_point(|g| g.0 <= x + 0.6 * self.legacy_em);
            if gaps[start..end]
                .iter()
                .map(|g| g.1)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                >= 3
            {
                check(a, b, 0);
            }
        }
        for w in bands.windows(2) {
            if w[1].0 - w[0].0 < 1.25 * self.legacy_em {
                continue;
            }
            for &a in &w[0].1 {
                for &b in &w[1].1 {
                    let (a_box, b_box) = (self.legacy_tokens[a].bbox, self.legacy_tokens[b].bbox);
                    if (a_box[0] - b_box[0]).abs() <= 0.4 * self.legacy_em && b_box[1] >= a_box[3] {
                        check(a, b, 1);
                    }
                }
            }
        }
        (conflicts, supported)
    }
}

pub(super) fn crop_vectors(
    lines: &Value,
    crop: &Value,
    scale: f64,
    rotation: &str,
) -> Result<Vec<Segment>, Error> {
    let coords: Vec<f64> = serde_json::from_value(crop.clone()).map_err(error)?;
    let [px, py, x0, y0, x1, y1, width, height] = coords.as_slice() else {
        return Err(error("Invalid table crop"));
    };
    let mut result = Vec::new();
    for line in array(lines)? {
        let b: Box4 = serde_json::from_value(line["bbox"].clone()).map_err(error)?;
        if b[2] < x0 / scale - 3.
            || b[0] > x1 / scale + 3.
            || b[3] < y0 / scale - 3.
            || b[1] > y1 / scale + 3.
        {
            continue;
        }
        let axis = match line["orientation"].as_str() {
            Some("vertical") => 0,
            Some("horizontal") => 1,
            _ => continue,
        };
        let (along, across) = if axis == 0 {
            (y0 - py, x0 - px)
        } else {
            (x0 - px, y0 - py)
        };
        let (lo, hi, pos) = (
            b[1 - axis] * scale - along,
            b[3 - axis] * scale - along,
            center(b, axis) * scale - across,
        );
        let segment = match (rotation, axis) {
            ("90", 0) => Segment(1, width - pos, lo, hi),
            ("90", _) => Segment(0, pos, width - hi, width - lo),
            ("270", 0) => Segment(1, pos, height - hi, height - lo),
            ("270", _) => Segment(0, height - pos, lo, hi),
            _ => Segment(axis, pos, lo, hi),
        };
        result.push(segment);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn added_row_boundary_needs_local_evidence_in_multiple_columns() {
        let make_grid = |split| {
            Grid::new(
                (0..2)
                    .flat_map(|col| {
                        let x = (col * 50) as f64;
                        if split {
                            vec![
                                GridCell {
                                    bbox: [x, 0., x + 50., 24.],
                                    logic: [0, 0, col, col],
                                    tokens: vec![col],
                                },
                                GridCell {
                                    bbox: [x, 24., x + 50., 60.],
                                    logic: [1, 1, col, col],
                                    tokens: vec![col + 2],
                                },
                            ]
                        } else {
                            vec![GridCell {
                                bbox: [x, 0., x + 50., 60.],
                                logic: [0, 0, col, col],
                                tokens: vec![col, col + 2],
                            }]
                        }
                    })
                    .collect(),
            )
            .unwrap()
        };
        let main = make_grid(false);
        let alternate = make_grid(true);
        let mut source = Source {
            tokens: (0..4)
                .map(|id| {
                    let x = (id % 2 * 50 + 5) as f64;
                    let y = (id / 2 * 13 + 11) as f64;
                    Token {
                        id,
                        bbox: [x, y, x + 35., y + 10.],
                        confidence: 1.,
                        numeric: false,
                    }
                })
                .collect(),
            legacy_tokens: Vec::new(),
            em: 10.,
            legacy_em: 10.,
            pairs: Vec::new(),
            lines: Vec::new(),
            ruling: [Vec::new(), Vec::new()],
        };
        assert_eq!(source.unsupported_row_splits(&main, &alternate), 1);
        source.ruling[1].push((24., 0., 100.));
        assert_eq!(source.unsupported_row_splits(&main, &alternate), 0);
        source.ruling[1].clear();
        for token in &mut source.tokens[2..] {
            token.bbox[1] += 6.;
            token.bbox[3] += 6.;
        }
        assert_eq!(source.unsupported_row_splits(&main, &alternate), 0);
    }

    fn grid(raw: &Value, tokens: &Value) -> Option<Grid> {
        if raw.is_null() {
            return None;
        }
        Grid::new(
            raw[0]
                .as_array()
                .unwrap()
                .iter()
                .zip(raw[1].as_array().unwrap())
                .enumerate()
                .map(|(i, (b, l))| {
                    let b: Vec<f64> = serde_json::from_value(b.clone()).unwrap();
                    GridCell {
                        bbox: cell_bounds(&b).unwrap(),
                        logic: serde_json::from_value(l.clone()).unwrap(),
                        tokens: tokens
                            .as_object()
                            .into_iter()
                            .flat_map(|m| m.iter())
                            .filter_map(|(key, v)| {
                                v.as_array()
                                    .unwrap()
                                    .iter()
                                    .any(|n| n.as_u64() == Some(i as u64))
                                    .then(|| key.parse::<usize>().unwrap())
                            })
                            .collect(),
                    }
                })
                .collect(),
        )
    }

    #[test]
    fn vector_mapping_follows_crop_padding_scale_and_rotation() {
        let lines = json!([{"bbox":[10,33,40,33],"orientation":"horizontal"}]);
        let crop = json!([4, 4, 20, 20, 112, 92, 100, 80]);
        for (rotation, expected) in [
            ("0", [1., 50., 4., 64.]),
            ("90", [0., 50., 36., 96.]),
            ("270", [0., 30., 4., 64.]),
        ] {
            let segments = crop_vectors(&lines, &crop, 2., rotation).unwrap();
            let [Segment(axis, pos, lo, hi)] = segments.as_slice() else {
                panic!("wrong segment count");
            };
            assert_eq!([*axis as f64, *pos, *lo, *hi], expected);
        }
    }

    #[test]
    fn shared_evidence_detects_column_merge_with_and_without_lines() {
        for draw in [false, true] {
            let mut image = image::RgbImage::from_pixel(140, 120, image::Rgb([255; 3]));
            if draw {
                for y in 4..112 {
                    for x in 67..69 {
                        image.put_pixel(x, y, image::Rgb([0; 3]));
                    }
                }
            }
            let mut png = std::io::Cursor::new(Vec::new());
            image.write_to(&mut png, image::ImageFormat::Png).unwrap();
            let boxes: Vec<_> = [15, 50, 85]
                .into_iter()
                .flat_map(|y| {
                    [12, 87].map(|x| [[x, y], [x + 24, y], [x + 24, y + 12], [x, y + 12]])
                })
                .collect();
            let rows = json!([
                boxes,
                ["item", "42", "item", "42", "item", "42"],
                vec![0.99; 6]
            ]);
            let source = Source::new(png.get_ref(), &rows, &[], &[]).unwrap();
            let split = Grid::new(
                (0..6)
                    .map(|i| {
                        let (r, c) = (i / 2, i % 2);
                        let y = 3. + r as f64 * 35.;
                        GridCell {
                            bbox: [c as f64 * 68., y, if c == 0 { 68. } else { 140. }, y + 35.],
                            logic: [r, r, c, c],
                            tokens: vec![i],
                        }
                    })
                    .collect(),
            )
            .unwrap();
            let merged = Grid::new(
                (0..3)
                    .map(|r| GridCell {
                        bbox: [0., 3. + r as f64 * 35., 140., 38. + r as f64 * 35.],
                        logic: [r, r, 0, 0],
                        tokens: vec![r * 2, r * 2 + 1],
                    })
                    .collect(),
            )
            .unwrap();
            let (right, wrong) = (source.score(&split), source.score(&merged));
            assert!(!right.suspicious());
            assert!(wrong.suspicious());
            let [text, lines, gaps] = right.delta(&wrong);
            assert_eq!(text, 0.);
            assert!(gaps > 0.);
            assert_eq!(lines > 0., draw);
        }
    }

    #[test]
    fn ruled_blank_rows_are_protected_from_header_only_candidate() {
        let white = image::RgbImage::from_pixel(136, 136, image::Rgb([255; 3]));
        let mut png = std::io::Cursor::new(Vec::new());
        white.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let rows = json!([
            [
                [[17, 9], [33, 9], [33, 34], [17, 34]],
                [[77, 9], [93, 9], [93, 34], [77, 34]]
            ],
            ["Name", "Amount"],
            [1., 1.]
        ]);
        let mut lines: Vec<_> = (0..=6)
            .map(|i| Segment(1, 15. + i as f64 * 18., 6., 126.))
            .collect();
        lines.extend([6., 66., 126.].map(|x| Segment(0, x, 15., 123.)));
        let source = Source::new(png.get_ref(), &rows, &[], &lines).unwrap();
        let grid = |n: usize| {
            Grid::new(
                (0..n)
                    .flat_map(|r| {
                        (0..2).map(move |c| {
                            let x = 6. + c as f64 * 60.;
                            let y = 15. + r as f64 * 18.;
                            GridCell {
                                bbox: [x, y + 1., x + 60., y + 15.],
                                logic: [r, r, c, c],
                                tokens: if r == 0 && n == 1 { vec![c] } else { vec![] },
                            }
                        })
                    })
                    .collect(),
            )
            .unwrap()
        };
        let main = grid(6);
        let backup = grid(1);
        let header = "<tr><td>Name</td><td>Amount</td></tr>";
        let html = json!(format!(
            "<table>{header}{}</table>",
            "<tr><td></td><td></td></tr>".repeat(5)
        ));
        let candidate = json!(format!("<table>{header}</table>"));
        let structure = super::super::table_fallback::structure(&html, &rows, &[]);
        let ruled = source.ruled_empty_rows(&main, &structure.empty_indices);
        assert_eq!(ruled, (1..=5).collect());
        assert!(!structure.suspicious(&ruled));
        let (selected, report) = super::super::tables::select_candidate(
            &html,
            Some(&main),
            &candidate,
            Some(&backup),
            &rows,
            &[],
            Some(&source),
            &Request::default(),
        );
        assert_eq!(report["protected_empty_rows"], true);
        assert!(!selected);
    }

    #[test]
    #[ignore = "Generate build/migration/table-evidence-sync.json with export_table_evidence_sync.py"]
    fn python_table_replay() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/table-evidence-sync.json");
        let data: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let mut failures = Vec::new();
        let mut selected = 0;
        let mut triggered = 0;
        for case in data.as_array().unwrap() {
            let _budget = crate::legacy_page::budget::host_page();
            let id = case["id"].as_str().unwrap();
            let png = std::fs::read(case["image"].as_str().unwrap()).unwrap();
            let vectors: Vec<Segment> = case["vector"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    Segment(
                        v[0].as_u64().unwrap() as usize,
                        v[1].as_f64().unwrap(),
                        v[2].as_f64().unwrap(),
                        v[3].as_f64().unwrap(),
                    )
                })
                .collect();
            let source = Source::new(&png, &case["rows"], &[], &vectors).unwrap();
            let mut grids = Vec::new();
            for j in 0..2 {
                let c = &case["candidates"][j];
                let mut g = grid(&c["evidence_grid"], &c["tokens"]).unwrap();
                g.legacy = grid(&case["grids"][j], &Value::Null).map(Box::new);
                let score = source.score(&g);
                let name = if j == 0 { "main" } else { "tatr" };
                if json!(score.missing) != c["missing"]
                    || json!(score.missing_numbers) != c["missing_numbers"]
                {
                    failures.push(format!("{id}/{name} OCR instance retention mismatch"));
                }
                let quality =
                    super::super::table_fallback::quality(&case[name], &case["rows"], &[]);
                let st = super::super::table_fallback::structure(&case[name], &case["rows"], &[]);
                let actual = json!({"split_lines":st.split_lines,"merged_values":st.merged_values,
                    "numbers":st.numbers,"full_width_rows":st.full_width_rows,"row_misalignment":st.row_misalignment,
                    "collapsed_rows":st.collapsed_rows,"row_count":st.rows,"empty_row_run":st.empty_run,"empty_rows":st.empty,"empty_indices":st.empty_indices});
                if actual != c["structure"] {
                    failures.push(format!(
                        "{id}/{name} structure actual={actual} expected={}",
                        c["structure"]
                    ));
                }
                if json!({"valid":quality.valid,"retained":quality.retained,"total":quality.total})
                    != c["quality"]
                {
                    failures.push(format!("{id}/{name} quality mismatch"));
                }
                let report = score.report();
                for (k, index) in [("text", 0), ("lines", 1), ("gaps", 2)] {
                    if (report[k].as_f64().unwrap() - c["score"][index][0].as_f64().unwrap()).abs()
                        > 0.011
                    {
                        failures.push(format!(
                            "{id}/{name} {k} actual={} expected={}",
                            report[k], c["score"][index][0]
                        ));
                    }
                }
                let (a, b) = source.legacy_grid(&g);
                if serde_json::to_value(&a).unwrap() != c["conflicts"]
                    || serde_json::to_value(&b).unwrap() != c["supported"]
                {
                    failures.push(format!(
                        "{id}/{name} legacy pairs actual={:?}/{:?} expected={}/{}",
                        a, b, c["conflicts"], c["supported"]
                    ));
                }
                grids.push(g);
            }
            let first = super::super::table_fallback::quality(&case["main"], &case["rows"], &[]);
            let st = super::super::table_fallback::structure(&case["main"], &case["rows"], &[]);
            let ruled_rows = source.ruled_empty_rows(&grids[0], &st.empty_indices);
            let ran = first.needs_fallback(0.2)
                || st.suspicious(&ruled_rows)
                || source.score(&grids[0]).suspicious()
                || source.legacy_grid(&grids[0]).0.len() >= 2;
            let request = Request::default();
            let (chosen, report) = super::super::tables::select_candidate(
                &case["main"],
                Some(&grids[0]),
                &case["tatr"],
                Some(&grids[1]),
                &case["rows"],
                &[],
                Some(&source),
                &request,
            );
            let chosen = ran && chosen;
            triggered += usize::from(ran);
            selected += usize::from(chosen);
            if ran != case["triggered"].as_bool().unwrap() || chosen != (case["selected"] == "tatr")
            {
                failures.push(format!(
                    "{id} decision actual={ran}/{chosen} expected={}/{} report={report}",
                    case["triggered"], case["selected"]
                ));
            }
        }
        for f in &failures {
            eprintln!("{f}");
        }
        println!(
            "{} tables, {triggered} fallback calls, {selected} selected, {} discrepancies",
            data.as_array().unwrap().len(),
            failures.len()
        );
        assert!(failures.is_empty());
    }
}
