//! 有界同行区间查询；每次只返回一小批行，不保存平方规模的配对集合。

use crate::geometry::Box4;

/// 复现有限框的水平交叠比例，零宽和反向框保持原来的零支持量。
fn x_overlap(a: Box4, b: Box4) -> f64 {
    let length = a[2].min(b[2]) - a[0].max(b[0]);
    let smaller = (a[2] - a[0]).min(b[2] - b[0]);
    if smaller > 0.0 {
        (if length > 0.0 { length } else { 0.0 }) / smaller
    } else {
        0.0
    }
}

/// 批量计算标题净空，按源顺序处理相同对象和视觉行，溢出交给 Python。
pub fn title_gaps(
    records: Vec<(usize, Option<usize>, Box4, f64, bool)>,
) -> Option<Vec<(Option<f64>, Option<f64>)>> {
    if records.iter().any(|r| {
        r.2.iter().any(|v| !v.is_finite())
            || !r.3.is_finite()
            || !(r.2[2] - r.2[0]).is_finite()
            || !(r.2[1] + r.2[3]).is_finite()
    }) {
        return None;
    }
    let centers: Vec<_> = records.iter().map(|r| (r.2[1] + r.2[3]) / 2.0).collect();
    let mut output = Vec::with_capacity(records.len());
    for (i, a) in records.iter().enumerate() {
        let (mut above, mut below): (Option<f64>, Option<f64>) = (None, None);
        for (j, b) in records.iter().enumerate() {
            if a.0 == b.0 || x_overlap(a.2, b.2) < 0.1 || (a.1.is_some() && a.1 == b.1) {
                continue;
            }
            let (first, second, target) = if centers[j] < centers[i] {
                (b, a, &mut above)
            } else if centers[j] > centers[i] {
                (a, b, &mut below)
            } else {
                continue;
            };
            let gap = if first.4 {
                (second.2[1] - first.2[3]).max(-0.25 * first.3)
            } else {
                second.2[1] - (first.2[1] + first.3)
            };
            if !gap.is_finite() {
                return None;
            }
            let gap = if gap > 0.0 { gap } else { 0.0 };
            if target.is_none_or(|previous| gap < previous) {
                *target = Some(gap);
            }
        }
        output.push((above, below));
    }
    Some(output)
}

/// 为一页几何分析选择最近上下邻行；距离相等保留第一个来源，返回索引而非复制对象。
pub fn line_neighbors(
    records: Vec<(usize, Box4, f64, f64)>,
) -> Option<Vec<(Option<usize>, Option<usize>)>> {
    if records.iter().any(|r| {
        r.1.iter().any(|v| !v.is_finite())
            || !r.2.is_finite()
            || !r.3.is_finite()
            || !(r.1[2] - r.1[0]).is_finite()
    }) {
        return None;
    }
    let mut output = Vec::with_capacity(records.len());
    for a in &records {
        let (mut above, mut below): (Option<usize>, Option<usize>) = (None, None);
        for (j, b) in records.iter().enumerate() {
            if a.0 == b.0 {
                continue;
            }
            let height = a.2.max(b.2);
            let shift = (b.3 - a.3).abs();
            if !shift.is_finite() || !(2.0 * height.max(1.0)).is_finite() {
                return None;
            }
            if !(x_overlap(a.1, b.1) >= 0.35 || (a.1[0] - b.1[0]).abs() <= 2.0 * height.max(1.0))
                || shift < 2.0_f64.max(0.6 * a.2)
            {
                continue;
            }
            if b.3 < a.3 && above.is_none_or(|k| b.3 > records[k].3) {
                above = Some(j);
            }
            if b.3 > a.3 && below.is_none_or(|k| b.3 < records[k].3) {
                below = Some(j);
            }
        }
        output.push((above, below));
    }
    Some(output)
}

#[derive(Debug)]
struct Group {
    order: Vec<usize>,
    lows: Vec<f64>,
    maxima: Vec<f64>,
    width: usize,
}

#[derive(Debug)]
pub struct IntervalIndex {
    bounds: Vec<(f64, f64)>,
    group_ids: Vec<usize>,
    groups: Vec<Group>,
}

impl IntervalIndex {
    /// 建立按左端点排序的最大右端点树，输入必须是已经验证的有限区间。
    pub fn new(bounds: Vec<(f64, f64)>, group_ids: Vec<usize>) -> Option<Self> {
        if bounds.len() != group_ids.len()
            || bounds
                .iter()
                .any(|(a, b)| !a.is_finite() || !b.is_finite() || a > b)
            || group_ids.iter().any(|g| *g >= bounds.len())
        {
            return None;
        }
        let count = group_ids.iter().max().map_or(0, |g| g + 1);
        let mut members = vec![Vec::new(); count];
        for (i, &group) in group_ids.iter().enumerate() {
            members[group].push(i);
        }
        let mut groups = Vec::with_capacity(count);
        for mut order in members {
            order.sort_by(|a, b| {
                bounds[*a]
                    .0
                    .partial_cmp(&bounds[*b].0)
                    .unwrap()
                    .then(a.cmp(b))
            });
            let width = order.len().max(1).next_power_of_two();
            let mut maxima = vec![f64::NEG_INFINITY; 2 * width];
            for (position, &index) in order.iter().enumerate() {
                maxima[width + position] = bounds[index].1;
            }
            for node in (1..width).rev() {
                maxima[node] = maxima[2 * node].max(maxima[2 * node + 1]);
            }
            let lows = order.iter().map(|i| bounds[*i].0).collect();
            groups.push(Group {
                order,
                lows,
                maxima,
                width,
            });
        }
        Some(Self {
            bounds,
            group_ids,
            groups,
        })
    }

    /// 返回当前行右侧的相交行，按原始索引排序以保持并查集合并顺序。
    pub fn query(&self, index: usize) -> Vec<usize> {
        let (low, high) = self.bounds[index];
        let group = &self.groups[self.group_ids[index]];
        let end = group.lows.partition_point(|value| *value <= high);
        let mut stack = vec![(1, 0, group.width)];
        let mut result = Vec::new();
        while let Some((node, left, right)) = stack.pop() {
            if left >= end || group.maxima[node] < low {
                continue;
            }
            if right - left == 1 {
                let other = group.order[left];
                if other > index {
                    result.push(other);
                }
            } else {
                let middle = (left + right) / 2;
                stack.push((2 * node + 1, middle, right));
                stack.push((2 * node, left, middle));
            }
        }
        result.sort_unstable();
        result
    }

    /// 批量查询有限行数；单个极密行可以超过预算，但不会缓存整个页面的行对。
    pub fn rows(&self, start: usize, count: usize, budget: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut total = 0;
        for index in start..start.saturating_add(count).min(self.bounds.len()) {
            let row = self.query(index);
            total += row.len();
            result.push(row);
            if total >= budget {
                break;
            }
        }
        result
    }
}

/// 同行安全超集过滤，保留普通框和原始连续字符框两条独立路径。
pub struct BaselineGeometry {
    index: IntervalIndex,
    boxes: Vec<Box4>,
    heights: Vec<f64>,
    sources: Vec<Option<Box4>>,
}

/// 有限正向框的参考同行几何，计算顺序与 Python 一致。
fn same_baseline(a: Box4, ah: f64, b: Box4, bh: f64) -> bool {
    let h = ah.max(bh);
    if ah.min(bh) <= 0.0 || h / ah.min(bh) > 1.35 {
        return false;
    }
    let overlap = 0.0_f64.max(a[3].min(b[3]) - a[1].max(b[1]));
    let shorter = 0.1_f64.max((a[3] - a[1]).min(b[3] - b[1]));
    if overlap / shorter < 0.7 && (a[3] - b[3]).abs() > 0.25 * h {
        return false;
    }
    let (left, right) = if a[0] <= b[0] { (a, b) } else { (b, a) };
    let gap = right[0] - left[2];
    gap >= -0.25 * h && gap <= 3.0_f64.max(0.75 * h)
}

/// 验证打包记录，限制幅度避免中间运算溢出改变必要条件。
fn safe_box(b: &Box4) -> bool {
    b.iter().all(|v| v.is_finite() && v.abs() <= 1e100) && b[2] > b[0] && b[3] > b[1]
}

impl BaselineGeometry {
    /// 每次闭包阶段构建独立只读索引，不复用已经合并的旧几何。
    pub fn new(
        bounds: Vec<(f64, f64)>,
        groups: Vec<usize>,
        boxes: Vec<Box4>,
        heights: Vec<f64>,
        sources: Vec<Option<Box4>>,
    ) -> Option<Self> {
        let n = bounds.len();
        if boxes.len() != n
            || heights.len() != n
            || sources.len() != n
            || boxes.iter().any(|b| !safe_box(b))
            || heights.iter().any(|h| !h.is_finite() || h.abs() > 1e100)
            || sources.iter().flatten().any(|b| !safe_box(b))
        {
            return None;
        }
        Some(Self {
            index: IntervalIndex::new(bounds, groups)?,
            boxes,
            heights,
            sources,
        })
    }

    /// 只剔除所有参考分支均不可能接受的行对，语义与表格判断仍由 Python 执行。
    fn accepts(&self, i: usize, j: usize) -> bool {
        let (a, b, ah, bh) = (
            self.boxes[i],
            self.boxes[j],
            self.heights[i],
            self.heights[j],
        );
        if same_baseline(a, ah, b, bh) {
            return true;
        }
        if let (Some(sa), Some(sb)) = (self.sources[i], self.sources[j]) {
            if same_baseline(sa, sa[3] - sa[1], sb, sb[3] - sb[1]) {
                return true;
            }
        }
        let h = ah.max(bh);
        let overlap = 0.0_f64.max(a[3].min(b[3]) - a[1].max(b[1]));
        let smaller = 0.1_f64.max((a[3] - a[1]).min(b[3] - b[1]));
        if overlap / smaller < 0.7 || ((a[1] + a[3]) / 2.0 - (b[1] + b[3]) / 2.0).abs() > 0.5 * h {
            return false;
        }
        let (left, right) = if a[0] <= b[0] { (a, b) } else { (b, a) };
        let gap = right[0] - left[2];
        gap >= -0.15 * h && gap <= 0.75
    }

    /// 保序返回有界行批次，单行超预算时仍返回全部成员。
    pub fn rows(&self, start: usize, count: usize, budget: usize) -> Vec<Vec<usize>> {
        let mut output = Vec::new();
        let mut total = 0;
        for i in start..start.saturating_add(count).min(self.boxes.len()) {
            let row: Vec<_> = self
                .index
                .query(i)
                .into_iter()
                .filter(|j| self.accepts(i, *j))
                .collect();
            total += row.len();
            output.push(row);
            if total >= budget {
                break;
            }
        }
        output
    }
}
