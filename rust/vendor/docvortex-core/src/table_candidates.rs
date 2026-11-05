//! 表格走廊的连续候选计算：闭区间分配、文本证据及核心几何共享一次准备。
use crate::row_geometry::RowGeometry;
use std::collections::HashMap;

pub struct RuleCandidates {
    centers: Vec<f64>,
    bands: Vec<usize>,
    on_rule: Vec<bool>,
    fragment_counts: Vec<usize>,
    boxes: Vec<[f64; 4]>,
    row_centers: Vec<f64>,
    members: Vec<usize>,
    member_offsets: Vec<usize>,
    source_ids: Vec<i64>,
    geometry: RowGeometry,
}

impl RuleCandidates {
    /// 一次验证并保存有限纯数值走廊；横线必须严格递增，不删除或重排任何行。
    pub fn new(
        centers: Vec<f64>,
        rows: Vec<(f64, usize, [f64; 4], Vec<i64>)>,
    ) -> Result<Self, &'static str> {
        if centers.iter().any(|v| !v.is_finite())
            || centers.windows(2).any(|pair| pair[0] >= pair[1])
            || rows
                .iter()
                .any(|row| !row.0.is_finite() || row.2.iter().any(|v| !v.is_finite()))
        {
            return Err("nonfinite or unordered rule candidate geometry");
        }
        let mut bands = Vec::with_capacity(rows.len());
        let mut on_rule = Vec::with_capacity(rows.len());
        let mut fragment_counts = Vec::with_capacity(rows.len());
        let mut boxes = Vec::with_capacity(rows.len());
        let mut row_centers = Vec::with_capacity(rows.len());
        let mut members = Vec::new();
        let mut member_offsets = Vec::with_capacity(rows.len() + 1);
        let mut source_ids = Vec::new();
        let mut interned = HashMap::new();
        member_offsets.push(0);
        for (center, count, bbox, sources) in rows {
            row_centers.push(center);
            let band = centers.partition_point(|value| *value < center);
            bands.push(band);
            on_rule.push(centers.get(band) == Some(&center));
            fragment_counts.push(count);
            boxes.push(bbox);
            // 仅准备时散列原始来源；每个候选按紧凑 ID 位图去重，负数和稀疏 ID 不作下标。
            for source in sources {
                let id = *interned.entry(source).or_insert_with(|| {
                    let id = source_ids.len();
                    source_ids.push(source);
                    id
                });
                members.push(id);
            }
            member_offsets.push(members.len());
        }
        let geometry = RowGeometry::new(boxes.clone()).ok_or("invalid row geometry")?;
        Ok(Self {
            centers,
            bands,
            on_rule,
            fragment_counts,
            boxes,
            row_centers,
            members,
            member_offsets,
            source_ids,
            geometry,
        })
    }

    /// 按原横线中心完整核验短区间条件，不改变任何列统计或候选认领结果。
    pub fn short_intervals(
        &self,
        first: usize,
        count: usize,
        height: f64,
    ) -> Result<bool, &'static str> {
        if first
            .checked_add(count)
            .is_none_or(|end| end > self.centers.len())
            || !height.is_finite()
        {
            return Err("invalid interval height range");
        }
        Ok(count >= 2
            && self.centers[first..first + count]
                .windows(2)
                .all(|pair| pair[1] - pair[0] <= 6.0 * height))
    }

    /// 同一不可变走廊内稳定按行中心排序并分段，所有成员仍保留且不缓存最终候选。
    pub fn segments(
        &self,
        start: usize,
        end: usize,
        height: f64,
    ) -> Result<Vec<Vec<usize>>, &'static str> {
        if start > end || end > self.boxes.len() || !height.is_finite() {
            return Err("invalid row segment range");
        }
        let mut ordered: Vec<_> = (start..end).collect();
        ordered.sort_by(|a, b| {
            self.row_centers[*a]
                .partial_cmp(&self.row_centers[*b])
                .unwrap()
        });
        let mut output: Vec<Vec<usize>> = Vec::new();
        for index in ordered {
            if output.last().is_none_or(|group| {
                (self.boxes[index][1] - self.boxes[*group.last().unwrap()][3]).max(0.0)
                    > 3.0 * height
            }) {
                output.push(vec![index]);
            } else {
                output.last_mut().unwrap().push(index);
            }
        }
        Ok(output)
    }

    /// 保留共享横线两侧的双归属，在同一行扫描中收集多单元证据，不调用 Python。
    pub fn partition(
        &self,
        start: usize,
        end: usize,
        first: usize,
        rule_count: usize,
        height: f64,
    ) -> Result<(Vec<Vec<usize>>, bool), &'static str> {
        if start > end
            || end > self.bands.len()
            || first
                .checked_add(rule_count)
                .is_none_or(|last| last > self.centers.len())
            || !height.is_finite()
        {
            return Err("invalid rule candidate interval");
        }
        let count = rule_count.saturating_sub(1);
        let mut groups = vec![Vec::new(); count];
        let mut dense = vec![false; count];
        for row in start..end {
            let band = self.bands[row];
            if band > first && band - first - 1 < count {
                let local = band - first - 1;
                groups[local].push(row);
                dense[local] |= self.fragment_counts[row] >= 2;
            }
            if self.on_rule[row] && band >= first && band - first < count {
                let local = band - first;
                groups[local].push(row);
                dense[local] |= self.fragment_counts[row] >= 2;
            }
        }
        let accepted = rule_count >= 2
            && (0..count).all(|i| {
                dense[i]
                    || (i == 0
                        && !groups[i].is_empty()
                        && self.centers[first + 1] - self.centers[first] <= 2.5 * height)
            });
        Ok((groups, accepted))
    }

    /// 按首次出现顺序收集成员，并返回核心并集四个坐标的来源，保留 Python 坐标对象与负零。
    pub fn core(
        &self,
        start: usize,
        end: usize,
        rules: &[[f64; 4]],
    ) -> Result<(Vec<i64>, [usize; 4]), &'static str> {
        if start >= end
            || end > self.boxes.len()
            || rules.is_empty()
            || rules.iter().flatten().any(|v| !v.is_finite())
        {
            return Err("invalid rule candidate core");
        }
        let selected = &self.members[self.member_offsets[start]..self.member_offsets[end]];
        // 位图只属于本次查询；不可变快照无需共享标记或互斥，也不会污染并发查询。
        let mut seen = vec![0_u64; self.source_ids.len().div_ceil(64)];
        let mut members = Vec::with_capacity(selected.len().min(self.source_ids.len()));
        for &id in selected {
            let word = id / 64;
            let bit = 1_u64 << (id % 64);
            if seen[word] & bit == 0 {
                seen[word] |= bit;
                members.push(self.source_ids[id]);
            }
        }
        let mut sources = [0; 4];
        for index in 1..rules.len() {
            for axis in 0..4 {
                if better(rules[index][axis], rules[sources[axis]][axis], axis) {
                    sources[axis] = index;
                }
            }
        }
        let rows = self
            .geometry
            .union_indices(start, end)
            .ok_or("empty candidate core")?;
        for axis in 0..4 {
            if better(
                self.boxes[rows[axis]][axis],
                rules[sources[axis]][axis],
                axis,
            ) {
                sources[axis] = rules.len() + rows[axis];
            }
        }
        Ok((members, sources))
    }
}

/// 严格比较保持 Python min/max 的首个相等值，不使用改变负零来源的浮点 min/max。
fn better(candidate: f64, current: f64, axis: usize) -> bool {
    if axis < 2 {
        candidate < current
    } else {
        candidate > current
    }
}

/// 按原顺序筛选、去重和首个匹配分组，统计重复填充行带；不缓存最终表格认领。
pub fn repeated_fill_bands(boxes: &[[f64; 4]], rule: [f64; 4], em: f64) -> usize {
    let minimum_width = (8.0 * em).max(0.3 * (rule[2] - rule[0]));
    let mut candidates: Vec<[f64; 4]> = Vec::new();
    for &b in boxes {
        let width = b[2] - b[0];
        let height = b[3] - b[1];
        let center = (b[1] + b[3]) / 2.0;
        let overlap = 0.0_f64.max(b[2].min(rule[2]) - b[0].max(rule[0]));
        let shorter = width.min(rule[2] - rule[0]);
        let ratio = if shorter > 0.0 {
            overlap / shorter
        } else {
            0.0
        };
        if width < minimum_width
            || !(0.25 * em <= height && height <= 3.0 * em)
            || center < rule[1]
            || center > rule[3]
            || ratio < 0.8
        {
            continue;
        }
        if candidates.iter().any(|a| {
            let width = 0.0_f64.max(b[2].min(a[2]) - b[0].max(a[0]));
            let height = 0.0_f64.max(b[3].min(a[3]) - b[1].max(a[1]));
            let smaller = ((b[2] - b[0]) * (b[3] - b[1])).min((a[2] - a[0]) * (a[3] - a[1]));
            smaller > 0.0 && width * height / smaller >= 0.9
        }) {
            continue;
        }
        candidates.push(b);
    }
    let tolerance = 3.0_f64.max(em);
    let mut groups: Vec<([f64; 4], usize)> = Vec::new();
    for b in candidates {
        let target = groups.iter_mut().find(|(a, _)| {
            (b[0] - a[0]).abs() <= tolerance
                && (b[2] - a[2]).abs() <= tolerance
                && ((b[3] - b[1]) - (a[3] - a[1])).abs() <= tolerance
        });
        if let Some((_, count)) = target {
            *count += 1;
        } else {
            groups.push((b, 1));
        }
    }
    groups.iter().map(|(_, count)| *count).max().unwrap_or(0)
}
