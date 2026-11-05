//! 对单调追加的聚类直接读取中位位置，保持原比较顺序和成员索引。

/// 输入已稳定排序；每个簇按全局顺序追加，不需要再次排序。
fn ordered_median(indices: &[usize], values: &[f64]) -> f64 {
    let n = indices.len();
    if n % 2 == 1 {
        values[indices[n / 2]]
    } else {
        (values[indices[n / 2 - 1]] + values[indices[n / 2]]) / 2.0
    }
}

/// 按绝对或相对容差选首个簇，last_only 保留只检查末簇的原策略。
pub fn ordered_clusters(
    values: Vec<f64>,
    tolerance: f64,
    relative: f64,
    last_only: bool,
) -> Option<Vec<Vec<usize>>> {
    if values.iter().any(|v| !v.is_finite()) || !tolerance.is_finite() || !relative.is_finite() {
        return None;
    }
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].partial_cmp(&values[*b]).unwrap());
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for index in order {
        let start = if last_only {
            clusters.len().saturating_sub(1)
        } else {
            0
        };
        let target = (start..clusters.len()).find(|&i| {
            let middle = ordered_median(&clusters[i], &values);
            let limit = if relative == 0.0 {
                tolerance
            } else {
                relative * middle
            };
            (values[index] - middle).abs() <= limit
        });
        if let Some(target) = target {
            clusters[target].push(index);
        } else {
            clusters.push(vec![index]);
        }
    }
    Some(clusters)
}

use crate::geometry::Box4;
use crate::median;

/// 按调用方提供的值计算中位数，避免为了切片临时克隆整份向量。
fn median_values<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    crate::median(values.into_iter().collect())
}

pub type Typography = (
    f64,
    Option<f64>,
    Option<usize>,
    f64,
    Option<f64>,
    Option<f64>,
    Option<f64>,
);

/// 一次计算整行字体与字形统计；类别编号由 Python 按其字符串/整数语义提供。
pub fn typography(
    boxes: Vec<Option<Box4>>,
    fonts: Vec<Option<usize>>,
    weights: Vec<Option<f64>>,
    families: &[Option<usize>],
    fallback_height: f64,
) -> Option<Typography> {
    if boxes.len() != fonts.len()
        || boxes.len() != weights.len()
        || fonts.iter().flatten().any(|i| *i >= families.len())
        || !fallback_height.is_finite()
        || boxes.iter().flatten().flatten().any(|v| !v.is_finite())
        || weights.iter().flatten().any(|v| !v.is_finite())
    {
        return None;
    }
    let glyph_count = boxes.len();
    let mut heights = Vec::with_capacity(glyph_count);
    let mut widths = Vec::with_capacity(glyph_count);
    let mut counts = vec![0usize; families.len()];
    let mut font_weights = vec![Vec::new(); families.len()];
    let mut seen = Vec::new();
    let mut glyphs = Vec::with_capacity(glyph_count);
    for ((raw, font), weight) in boxes.into_iter().zip(fonts).zip(weights) {
        let Some(b) = raw else {
            continue;
        };
        heights.push(0.1_f64.max(b[3] - b[1]));
        widths.push(0.1_f64.max(b[2] - b[0]));
        if let Some(id) = font {
            if counts[id] == 0 {
                seen.push(id);
            }
            counts[id] += 1;
            if let Some(weight) = weight {
                font_weights[id].push(weight);
            }
        }
        glyphs.push((b, font, weight));
    }
    let height = if heights.is_empty() {
        fallback_height
    } else {
        median_values(heights)
    };
    let width = if widths.is_empty() {
        None
    } else {
        Some(median_values(widths))
    };
    let mut winner = None;
    for id in seen {
        if winner.is_none_or(|previous| counts[id] > counts[previous]) {
            winner = Some(id);
        }
    }
    let coverage = winner.map_or(0.0, |id| {
        counts[id] as f64 / counts.iter().sum::<usize>() as f64
    });
    let weight = winner.and_then(|id| {
        if font_weights[id].is_empty() {
            None
        } else {
            Some(median_values(font_weights[id].iter().copied()))
        }
    });
    let mut emphasis = None;
    let mut typography_width = None;
    if glyphs.len() >= 4 {
        if let Some(first) = glyphs[0].1 {
            let split = glyphs
                .iter()
                .position(|g| g.1 != Some(first))
                .unwrap_or(glyphs.len());
            if split >= 2 && glyphs.len() - split >= 2 {
                let left = glyphs[..split]
                    .iter()
                    .map(|g| g.0[0])
                    .fold(f64::INFINITY, f64::min);
                let right = glyphs[..split]
                    .iter()
                    .map(|g| g.0[2])
                    .fold(f64::NEG_INFINITY, f64::max);
                let prefix_width = 0.1_f64.max(right - left);
                let prefix: Vec<_> = glyphs[..split].iter().filter_map(|g| g.2).collect();
                let body: Vec<_> = glyphs[split..].iter().filter_map(|g| g.2).collect();
                if !prefix.is_empty() && !body.is_empty() {
                    let p = median_values(prefix);
                    let b = median_values(body);
                    // 中位数相加可能溢出，保留 Python 两个小于判断的否定，不能改写为大于等于。
                    if !(p - b < 100.0 || p < 1.15 * 1.0_f64.max(b)) {
                        emphasis = Some(prefix_width);
                    }
                }
                if let Some(first_family) = families[first] {
                    let mut body_counts = vec![0; families.len()];
                    let mut body_seen = Vec::new();
                    for (_, id, _) in &glyphs[split..] {
                        if let Some(id) = *id {
                            if body_counts[id] == 0 {
                                body_seen.push(id);
                            }
                            body_counts[id] += 1;
                        }
                    }
                    let mut body_winner = None;
                    for id in body_seen {
                        if body_winner
                            .is_none_or(|previous| body_counts[id] > body_counts[previous])
                        {
                            body_winner = Some(id);
                        }
                    }
                    if let Some(id) = body_winner {
                        if body_counts[id] >= 2 && families[id] != Some(first_family) {
                            typography_width = Some(prefix_width);
                        }
                    }
                }
            }
        }
    }
    Some((
        height,
        width,
        winner,
        coverage,
        weight,
        emphasis,
        typography_width,
    ))
}

/// 输入沿 Python 的稳定成员排序；快照只在本次调用使用，不缓存可变行。
pub fn lane_gap(
    boxes: Vec<Box4>,
    heights: Vec<f64>,
    restored: Vec<bool>,
    skip: Vec<bool>,
) -> Option<(f64, f64)> {
    let n = boxes.len();
    if heights.len() != n
        || restored.len() != n
        || skip.len() != n.saturating_sub(1)
        || boxes.iter().flatten().any(|v| !v.is_finite())
        || heights.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return None;
    }
    let mh = if n == 0 { 1.0 } else { median(heights.clone()) };
    if !mh.is_finite() {
        return None;
    }
    let mut gaps = Vec::new();
    for i in 1..n {
        if skip[i - 1] {
            continue;
        }
        let a = boxes[i - 1];
        let b = boxes[i];
        let ah = heights[i - 1];
        let bh = heights[i];
        if ah.max(bh) / ah.min(bh) > 1.35 {
            continue;
        }
        let h = ah.max(bh);
        let gap = if restored[i - 1] {
            (b[1] - a[3]).max(-0.25 * ah)
        } else {
            b[1] - (a[1] + ah)
        };
        if gap < -0.25 * h || gap > 2.0 * h {
            continue;
        }
        let overlap = 0.0_f64.max(a[2].min(b[2]) - a[0].max(b[0]));
        let shorter = (a[2] - a[0]).min(b[2] - b[0]);
        let ratio = if shorter > 0.0 {
            overlap / shorter
        } else {
            0.0
        };
        if ratio < 0.5 && (a[0] - b[0]).abs() > 1.5 * mh {
            continue;
        }
        if !gap.is_finite() {
            return None;
        }
        gaps.push(0.0_f64.max(gap));
    }
    if gaps.is_empty() {
        return Some((0.35 * mh, 0.0));
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let count = 1usize.max((gaps.len() as f64 * 0.6).ceil() as usize);
    let lower = gaps[..count].to_vec();
    let regular = median(lower.clone());
    if !regular.is_finite() {
        return None;
    }
    let mad = median(lower.into_iter().map(|g| (g - regular).abs()).collect());
    if !mad.is_finite() {
        return None;
    }
    Some((regular, mad))
}
/// 只在一次表格候选构建期间保存排序后的正文高度与来源中心范围。
pub struct NoteMetrics {
    items: Vec<(i64, f64, f64)>,
    extents: std::collections::HashMap<i64, (f64, f64)>,
    index: crate::note_index::HeightIndex,
}

impl NoteMetrics {
    /// 将高度稳定排序一次，后续区间过滤保持 Python 排序后的平局顺序。
    pub fn new(mut items: Vec<(i64, f64, f64)>) -> Option<Self> {
        if items
            .iter()
            .any(|(_, y, h)| !y.is_finite() || !h.is_finite())
        {
            return None;
        }
        let mut extents = std::collections::HashMap::<i64, (f64, f64)>::new();
        for &(id, y, _) in &items {
            extents
                .entry(id)
                .and_modify(|p| {
                    p.0 = p.0.min(y);
                    p.1 = p.1.max(y);
                })
                .or_insert((y, y));
        }
        let index = crate::note_index::HeightIndex::new(&items);
        items.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        Some(Self {
            items,
            extents,
            index,
        })
    }

    /// 建立来源中心的行范围索引，整个走廊只传输一次成员。
    pub fn rows(&self, members: Vec<Vec<i64>>) -> crate::note_index::CoreRows {
        crate::note_index::CoreRows::new(members, &self.extents)
    }

    /// 已证明核心全在排除区间时查询秩树，否则保留原生精确成员筛选。
    pub fn height_for_rows(
        &self,
        rows: &crate::note_index::CoreRows,
        start: usize,
        end: usize,
        top: f64,
        bottom: f64,
        fallback: f64,
    ) -> Option<f64> {
        if start > end || end > rows.members.len() {
            return None;
        }
        if rows.contained(start, end, top, bottom) {
            return self.index.height(top, bottom, fallback);
        }
        let core: Vec<i64> = rows.members[start..end].iter().flatten().copied().collect();
        self.height(top, bottom, &core, fallback)
    }

    /// 过滤纵向排除区间和明确核心成员，直接读取最高四分位的中位位置。
    pub fn height(&self, top: f64, bottom: f64, core: &[i64], fallback: f64) -> Option<f64> {
        if !top.is_finite() || !bottom.is_finite() || !fallback.is_finite() {
            return None;
        }
        // 处于排除区间内部的成员无需再次建哈希集合，重复来源仍按其全部中心范围检查。
        let excluded: std::collections::HashSet<i64> = core
            .iter()
            .copied()
            .filter(|id| {
                self.extents
                    .get(id)
                    .is_some_and(|(a, b)| *a < top || *b > bottom)
            })
            .collect();
        let heights: Vec<f64> = self
            .items
            .iter()
            .filter_map(|(id, y, h)| {
                ((*y < top || *y > bottom) && !excluded.contains(id)).then_some(*h)
            })
            .collect();
        let n = heights.len();
        if n < 4 {
            return Some(fallback);
        }
        let count = n.div_ceil(4);
        let start = n - count;
        let value = if count % 2 == 1 {
            heights[start + count / 2]
        } else {
            (heights[start + count / 2 - 1] + heights[start + count / 2]) / 2.0
        };
        value.is_finite().then_some(value)
    }
}
