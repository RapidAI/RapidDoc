//! 批量计算表格筛选、线段覆盖、网格连通和字符归属；候选裁决留在 Python。

use crate::geometry::{self, Box4};
use std::collections::HashMap;
pub type Rule = (u8, f64, f64, f64);
pub type Query = (u8, Vec<f64>, f64, f64, f64);
pub type GridIndex = (Vec<f64>, Vec<f64>, Vec<Vec<usize>>);
type CoverageCache = HashMap<(u8, Vec<u64>, u64), Vec<(f64, f64)>>;
pub type ComponentSpecs = (Vec<usize>, Option<Vec<(usize, usize, usize, usize)>>);

/// 取得矩形交集，边界相接仍视为无面积。
fn intersection(a: Box4, b: Box4) -> Option<Box4> {
    let c = [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ];
    if c[2] > c[0] && c[3] > c[1] {
        Some(c)
    } else {
        None
    }
}

/// 批量筛出中心落在表格内的源框，并预先计算局部坐标。
pub fn select_boxes(
    values: Vec<Option<Box4>>,
    table: Box4,
    angle: i32,
) -> Vec<(usize, Box4, Option<Box4>)> {
    let mut result = Vec::new();
    for (i, b) in values.into_iter().enumerate() {
        let Some(b) = geometry::normalize(b, false) else {
            continue;
        };
        let x = (b[0] + b[2]) / 2.0;
        let y = (b[1] + b[3]) / 2.0;
        if !(table[0] <= x && x <= table[2] && table[1] <= y && y <= table[3]) {
            continue;
        }
        let local = intersection(b, table).map(|b| {
            geometry::rotate(
                [
                    b[0] - table[0],
                    b[1] - table[1],
                    b[2] - table[0],
                    b[3] - table[1],
                ],
                [table[2] - table[0], table[3] - table[1]],
                angle,
            )
        });
        result.push((i, b, local));
    }
    result
}

/// 用稳定端点排序与顺序累加计算覆盖率。
fn coverage(intervals: &[(f64, f64)], start: f64, end: f64) -> f64 {
    if end <= start {
        return 0.0;
    }
    let mut clipped: Vec<_> = intervals
        .iter()
        .filter_map(|(a, b)| {
            let left = start.max(a.min(*b));
            let right = end.min(a.max(*b));
            if right > left {
                Some((left, right))
            } else {
                None
            }
        })
        .collect();
    clipped.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let Some(&(mut left, mut right)) = clipped.first() else {
        return 0.0;
    };
    let mut covered = 0.0;
    for &(a, b) in &clipped[1..] {
        if a <= right {
            right = right.max(b);
        } else {
            covered += right - left;
            left = a;
            right = b;
        }
    }
    covered += right - left;
    1.0_f64.min(covered / (end - start))
}

/// 在一次调用中复用轨道区间，避免每个 separator 单独跨语言调用。
pub fn coverage_batch(rules: Vec<Rule>, queries: Vec<Query>) -> Option<Vec<f64>> {
    if rules
        .iter()
        .any(|r| ![r.1, r.2, r.3].iter().all(|v| v.is_finite()))
        || queries.iter().any(|q| {
            q.1.iter()
                .chain([q.2, q.3, q.4].iter())
                .any(|v| !v.is_finite())
        })
    {
        return None;
    }
    let mut cache = CoverageCache::new();
    Some(
        queries
            .into_iter()
            .map(|(orientation, aliases, start, end, tolerance)| {
                let key = (
                    orientation,
                    aliases.iter().map(|v| v.to_bits()).collect(),
                    tolerance.to_bits(),
                );
                let intervals = cache.entry(key).or_insert_with(|| {
                    rules
                        .iter()
                        .filter(|r| {
                            r.0 == orientation
                                && aliases.iter().any(|a| (r.1 - *a).abs() <= tolerance)
                        })
                        .map(|r| (r.2, r.3))
                        .collect()
                });
                coverage(intervals, start, end)
            })
            .collect(),
    )
}

/// 使用 Python 已计算的簇均值合并整批线段，保持求和/舍入的版本语义。
pub fn merge_rules(
    rules: Vec<Rule>,
    coordinates: Vec<(u8, Vec<f64>)>,
    tolerance: f64,
    join: f64,
) -> Option<Vec<Rule>> {
    if rules
        .iter()
        .any(|r| ![r.1, r.2, r.3].iter().all(|v| v.is_finite()))
        || coordinates
            .iter()
            .flat_map(|r| &r.1)
            .any(|v| !v.is_finite())
    {
        return None;
    }
    let mut output = Vec::new();
    for (orientation, coords) in coordinates {
        for coordinate in coords {
            let mut intervals: Vec<_> = rules
                .iter()
                .filter(|r| r.0 == orientation && (r.1 - coordinate).abs() <= tolerance)
                .map(|r| (r.2.min(r.3), r.2.max(r.3)))
                .collect();
            intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let Some(&(mut start, mut end)) = intervals.first() else {
                continue;
            };
            for &(a, b) in &intervals[1..] {
                if a <= end + join {
                    end = end.max(b);
                } else {
                    output.push((orientation, coordinate, start, end));
                    start = a;
                    end = b;
                }
            }
            output.push((orientation, coordinate, start, end));
        }
    }
    Some(output)
}

/// 计算字符相交面积比例，维持原来的上界截断。
fn overlap(glyph: Box4, cell: Box4) -> f64 {
    let area = (glyph[2] - glyph[0]) * (glyph[3] - glyph[1]);
    let Some(b) = intersection(glyph, cell) else {
        return 0.0;
    };
    if area <= 0.0 {
        0.0
    } else {
        1.0_f64.min((b[2] - b[0]) * (b[3] - b[1]) / area)
    }
}

/// 检查中心归属，供零交叠情况下的原规则回退使用。
fn contains(cell: Box4, glyph: Box4) -> bool {
    let x = (glyph[0] + glyph[2]) / 2.0;
    let y = (glyph[1] + glyph[3]) / 2.0;
    cell[0] <= x && x <= cell[2] && cell[1] <= y && y <= cell[3]
}

/// 将二分位置限制在实际原子轨道范围内。
fn track_index(tracks: &[f64], value: f64, right: bool) -> usize {
    let at = if right {
        tracks.partition_point(|v| *v <= value)
    } else {
        tracks.partition_point(|v| *v < value)
    };
    at.saturating_sub(1).min(tracks.len() - 2)
}

/// 整张表格执行现有索引或穷举落格判定，相同比例仍优先最大单元格索引。
pub fn assign_cells(
    glyphs: Vec<Box4>,
    specs: Vec<Box4>,
    index: Option<GridIndex>,
) -> Option<Vec<(Option<usize>, bool)>> {
    if specs.is_empty()
        || glyphs
            .iter()
            .chain(&specs)
            .flatten()
            .any(|v| !v.is_finite())
    {
        return None;
    }
    if let Some((x, y, owners)) = &index {
        if x.len() < 2
            || y.len() < 2
            || x.iter().chain(y).any(|v| !v.is_finite())
            || owners.len() != y.len() - 1
            || owners
                .iter()
                .any(|row| row.len() != x.len() - 1 || row.iter().any(|i| *i >= specs.len()))
        {
            return None;
        }
    }
    Some(
        glyphs
            .into_iter()
            .map(|glyph| {
                let candidates = if let Some((x, y, owners)) = &index {
                    let left = track_index(x, glyph[0], true);
                    let right = track_index(x, glyph[2], false);
                    let top = track_index(y, glyph[1], true);
                    let bottom = track_index(y, glyph[3], false);
                    if left == right && top == bottom {
                        let i = owners[top][left];
                        return (
                            if overlap(glyph, specs[i]) > 0.0 || contains(specs[i], glyph) {
                                Some(i)
                            } else {
                                None
                            },
                            false,
                        );
                    }
                    let mut indices: Vec<_> = (top..=bottom)
                        .flat_map(|r| (left..=right).map(move |c| owners[r][c]))
                        .collect();
                    indices.sort_unstable();
                    indices.dedup();
                    indices
                } else {
                    (0..specs.len()).collect()
                };
                let mut overlaps: Vec<_> = candidates
                    .iter()
                    .map(|i| (overlap(glyph, specs[*i]), *i))
                    .collect();
                overlaps.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let Some(&(ratio, i)) = overlaps.first() else {
                    return (None, false);
                };
                if ratio <= 0.0 {
                    let containing: Vec<_> = candidates
                        .into_iter()
                        .filter(|j| contains(specs[*j], glyph))
                        .collect();
                    return (
                        if containing.len() == 1 {
                            Some(containing[0])
                        } else {
                            None
                        },
                        false,
                    );
                }
                (
                    Some(i),
                    overlaps.len() > 1
                        && overlaps[1].0 >= 0.45
                        && (ratio - overlaps[1].0).abs() <= 0.02,
                )
            })
            .collect(),
    )
}

/// 对既有父节点执行完整路径压缩，与表格并查集的递归 find 等价。
fn find(parents: &mut [usize], index: usize) -> usize {
    let mut root = index;
    while parents[root] != root {
        root = parents[root];
    }
    let mut i = index;
    while parents[i] != i {
        let next = parents[i];
        parents[i] = root;
        i = next;
    }
    root
}

/// 批量连接原子格，并保留 first-root 拥有 second-root 的规则。
pub fn grid_parents(count: usize, pairs: Vec<(usize, usize)>) -> Vec<usize> {
    let mut parents: Vec<_> = (0..count).collect();
    for (a, b) in pairs {
        let a = find(&mut parents, a);
        let b = find(&mut parents, b);
        if a != b {
            parents[b] = a;
        }
    }
    parents
}

/// 将连通格转换为矩形范围；不完整矩形仍拒绝整份候选。
pub fn component_specs(mut parents: Vec<usize>, rows: usize, cols: usize) -> ComponentSpecs {
    let mut groups: HashMap<usize, (usize, usize, usize, usize, usize)> = HashMap::new();
    for r in 0..rows {
        for c in 0..cols {
            let root = find(&mut parents, r * cols + c);
            let g = groups.entry(root).or_insert((r, c, r, c, 0));
            g.0 = g.0.min(r);
            g.1 = g.1.min(c);
            g.2 = g.2.max(r);
            g.3 = g.3.max(c);
            g.4 += 1;
        }
    }
    let mut specs = Vec::new();
    for (r, c, last_r, last_c, n) in groups.into_values() {
        if (last_r - r + 1) * (last_c - c + 1) != n {
            return (parents, None);
        }
        specs.push((r, c, last_r - r + 1, last_c - c + 1));
    }
    specs.sort_unstable();
    (parents, Some(specs))
}

/// 保持反向候选遍历、严格距离平局与前缀下界提前停止的视觉组行。
pub fn visual_rows(
    boxes: Vec<Box4>,
    ids: Vec<usize>,
    median_height: f64,
) -> Option<Vec<Vec<usize>>> {
    if boxes.len() != ids.len()
        || !median_height.is_finite()
        || boxes
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > 1e150)
    {
        return None;
    }
    let mut order: Vec<_> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| {
        let aa = boxes[a];
        let bb = boxes[b];
        ((aa[1] + aa[3]) / 2.0)
            .partial_cmp(&((bb[1] + bb[3]) / 2.0))
            .unwrap()
            .then_with(|| aa[0].partial_cmp(&bb[0]).unwrap())
            .then_with(|| ids[a].cmp(&ids[b]))
    });
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut bounds: Vec<Box4> = Vec::new();
    let mut prefix: Vec<f64> = Vec::new();
    let tolerance = 0.75_f64.max(median_height * 0.40);
    for index in order {
        let g = boxes[index];
        let cy = (g[1] + g[3]) / 2.0;
        let mut best = None;
        let mut best_distance = f64::INFINITY;
        for i in (0..bounds.len()).rev() {
            let r = bounds[i];
            let ry = (r[1] + r[3]) / 2.0;
            let distance = (cy - ry).abs();
            let overlap = g[3].min(r[3]) - g[1].max(r[1]);
            let height = (g[3] - g[1]).min(r[3] - r[1]);
            let ratio = if overlap <= 0.0 || height <= 0.0 {
                0.0
            } else {
                1.0_f64.min(overlap / height)
            };
            if (ratio >= 0.45 || distance <= tolerance) && distance < best_distance {
                best = Some(i);
                best_distance = distance;
            }
            if cy >= ry && cy - ry > tolerance && g[1] > prefix[i] {
                break;
            }
        }
        if let Some(i) = best {
            rows[i].push(index);
            let r = bounds[i];
            bounds[i] = [
                r[0].min(g[0]),
                r[1].min(g[1]),
                r[2].max(g[2]),
                r[3].max(g[3]),
            ];
            for j in i..bounds.len() {
                prefix[j] = if j == 0 {
                    bounds[j][3]
                } else {
                    prefix[j - 1].max(bounds[j][3])
                };
            }
        } else {
            rows.push(vec![index]);
            bounds.push(g);
            prefix.push(prefix.last().copied().unwrap_or(g[3]).max(g[3]));
        }
    }
    let mut row_order: Vec<_> = (0..rows.len()).collect();
    row_order.sort_by(|&a, &b| {
        bounds[a][1]
            .partial_cmp(&bounds[b][1])
            .unwrap()
            .then_with(|| bounds[a][0].partial_cmp(&bounds[b][0]).unwrap())
    });
    Some(
        row_order
            .into_iter()
            .map(|i| {
                let mut row = std::mem::take(&mut rows[i]);
                row.sort_by(|&a, &b| {
                    boxes[a][0]
                        .partial_cmp(&boxes[b][0])
                        .unwrap()
                        .then_with(|| boxes[a][1].partial_cmp(&boxes[b][1]).unwrap())
                        .then_with(|| ids[a].cmp(&ids[b]))
                });
                row
            })
            .collect(),
    )
}

/// 整表计算列占用；公共边界仍归属从左到右第一个符合区间的列。
pub fn row_occupancy(rows: Vec<Vec<f64>>, tracks: Vec<f64>) -> Option<Vec<Vec<usize>>> {
    if rows
        .iter()
        .flatten()
        .chain(tracks.iter())
        .any(|v| !v.is_finite())
    {
        return None;
    }
    Some(
        rows.into_iter()
            .map(|row| {
                let mut occupied = vec![false; tracks.len().saturating_sub(1)];
                for center in row {
                    if let Some(i) = tracks
                        .windows(2)
                        .position(|t| t[0] <= center && center <= t[1])
                    {
                        occupied[i] = true;
                    }
                }
                occupied
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, yes)| yes.then_some(i))
                    .collect()
            })
            .collect(),
    )
}

/// 单元格一次完成原视觉组的几何细分，只返回完整源索引，不缓存文字或认领结果。
pub fn cell_visual_groups(
    boxes: Vec<Box4>,
    ids: Vec<usize>,
    visual: Vec<i64>,
    median_height: f64,
) -> Option<Vec<Vec<usize>>> {
    if boxes.len() != ids.len()
        || boxes.len() != visual.len()
        || !median_height.is_finite()
        || boxes
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > 1e100)
    {
        return None;
    }
    let mut groups = std::collections::BTreeMap::<i64, Vec<usize>>::new();
    for (index, row) in visual.into_iter().enumerate() {
        groups.entry(row).or_default().push(index);
    }
    let mut refined = Vec::new();
    for original in groups.into_values() {
        // 两个合格细分行各至少含两个字形，少于四个时只能保留原组。
        if original.len() < 4 {
            refined.push(original);
            continue;
        }
        let local = visual_rows(
            original.iter().map(|i| boxes[*i]).collect(),
            original.iter().map(|i| ids[*i]).collect(),
            median_height,
        )?;
        let coherent = local.iter().all(|row| {
            let centers: Vec<f64> = row
                .iter()
                .map(|i| {
                    let b = boxes[original[*i]];
                    (b[1] + b[3]) / 2.0
                })
                .collect();
            let low = centers.iter().copied().reduce(f64::min).unwrap();
            let high = centers.iter().copied().reduce(f64::max).unwrap();
            high - low <= 0.3 * median_height
        });
        if local.len() > 1
            && coherent
            && local.iter().all(|row| {
                let height = crate::median(
                    row.iter()
                        .map(|i| {
                            let b = boxes[original[*i]];
                            b[3] - b[1]
                        })
                        .collect(),
                );
                row.len() >= 2 && 0.85 * median_height <= height && height <= 1.25 * median_height
            })
        {
            refined.extend(
                local
                    .into_iter()
                    .map(|row| row.into_iter().map(|i| original[i]).collect()),
            );
        } else {
            refined.push(original);
        }
    }
    Some(refined)
}
