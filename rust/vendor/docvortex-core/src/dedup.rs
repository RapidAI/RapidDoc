//! 保持分桶访问次序、候选上限及最早来源规则的字符重复计算。

use crate::geometry::{Box4, Size};
use std::collections::HashMap;

pub type PaintRecord = (Option<Box4>, Option<Size>, usize, usize, bool);
pub type Pair = (usize, usize);
pub type Offset = (usize, usize, f64, f64);
pub type MappingRecord = (Option<i64>, i64, i64, usize, f64, Option<Size>, Box4, bool);

/// 批量识别连续的一对多来源映射，只返回范围，Unicode 合并仍由 Python 执行。
pub fn mapping_runs(records: Vec<MappingRecord>) -> Option<Vec<(usize, usize)>> {
    if records.iter().any(|r| {
        !r.4.is_finite()
            || r.6.iter().any(|v| !v.is_finite())
            || r.5.is_some_and(|p| p.iter().any(|v| !v.is_finite()))
    }) {
        return None;
    }
    let mut groups = Vec::new();
    let mut start = 0;
    for i in 1..records.len() {
        let a = &records[i - 1];
        let b = &records[i];
        let same = a.0.is_some()
            && a.0 == b.0
            && a.2.checked_add(1) == Some(b.1)
            && a.3 == b.3
            && a.4 == b.4
            && a.7
            && b.7
            && matches!((a.5, b.5), (Some(x), Some(y)) if close(x, y, 0.001))
            && close(a.6, b.6, 0.001);
        if !same {
            groups.push((start, i));
            start = i;
        }
    }
    if !records.is_empty() {
        groups.push((start, records.len()));
    }
    Some(groups)
}

/// 数值缺失、非有限或超出容差时不得作为相同位置的证据。
fn close<const N: usize>(a: [f64; N], b: [f64; N], epsilon: f64) -> bool {
    a.iter()
        .zip(b)
        .all(|(x, y)| x.is_finite() && y.is_finite() && (*x - y).abs() <= epsilon)
}

/// 计算覆盖较小矩形的比例，与 Python 的减法、乘除次序一致。
fn overlap(a: Box4, b: Box4) -> f64 {
    let area = ((a[2] - a[0]) * (a[3] - a[1])).min((b[2] - b[0]) * (b[3] - b[1]));
    if area > 0.0 {
        0.0_f64.max(a[2].min(b[2]) - a[0].max(b[0])) * 0.0_f64.max(a[3].min(b[3]) - a[1].max(b[1]))
            / area
    } else {
        0.0
    }
}

/// 按签名和相邻九桶生成精确重复与平移候选，保留每桶最多 64 项。
pub fn paint_pairs(records: Vec<PaintRecord>) -> Option<(Vec<Pair>, Vec<Offset>)> {
    if records
        .iter()
        .any(|r| r.0.is_some_and(|b| b.iter().any(|v| v.abs() > 1e15)))
    {
        return None;
    }
    let mut buckets: HashMap<(usize, i64, i64), Vec<usize>> = HashMap::new();
    let mut exact = Vec::new();
    let mut offsets = Vec::new();
    for (i, (box_, origin, signature, object, eligible)) in records.iter().enumerate() {
        if !eligible {
            continue;
        }
        let (Some(b), Some(co)) = (box_, origin) else {
            continue;
        };
        let bx = (b[0] / 2.5).floor() as i64;
        let by = (b[1] / 2.5).floor() as i64;
        for x in bx - 1..=bx + 1 {
            for y in by - 1..=by + 1 {
                if let Some(candidates) = buckets.get(&(*signature, x, y)) {
                    if candidates.len() >= 64 {
                        continue;
                    }
                    for &j in candidates {
                        let (a, po, _, previous_object, _) = records[j];
                        if previous_object == *object {
                            continue;
                        }
                        let a = a.unwrap();
                        let po = po.unwrap();
                        if close(a, *b, 0.001) && close(po, *co, 0.001) {
                            exact.push((j, i));
                            continue;
                        }
                        let dx = co[0] - po[0];
                        let dy = co[1] - po[1];
                        if !dx.is_finite() || !dy.is_finite() || dx.abs().max(dy.abs()) > 2.5 {
                            continue;
                        }
                        if [b[0] - a[0], b[1] - a[1], b[2] - a[2], b[3] - a[3]]
                            .iter()
                            .zip([dx, dy, dx, dy])
                            .all(|(d, t)| (*d - t).abs() <= 0.1)
                            && overlap(a, *b) >= 0.45
                        {
                            offsets.push((j, i, dx, dy));
                        }
                    }
                }
            }
        }
        let bucket = buckets.entry((*signature, bx, by)).or_default();
        if bucket.len() < 64 {
            bucket.push(i);
        }
    }
    Some((exact, offsets))
}

/// 精确复现原并查集的路径压缩与最终一次父节点跳转。
pub fn components(count: usize, pairs: &[Pair]) -> Vec<usize> {
    let mut parents: Vec<_> = (0..count).collect();
    for &(mut a, mut b) in pairs {
        while parents[a] != a {
            parents[a] = parents[parents[a]];
            a = parents[a];
        }
        while parents[b] != b {
            parents[b] = parents[parents[b]];
            b = parents[b];
        }
        parents[a.max(b)] = a.min(b);
    }
    for i in 0..count {
        parents[i] = parents[parents[i]];
    }
    parents
}

pub type EvidenceRecord = (Option<Box4>, f64, usize, usize, bool);
/// 连续证据中间仅允许空白或两端字形的副本。
fn endpoints(records: &[EvidenceRecord], roots: &[usize], start: usize, end: usize) -> bool {
    end - start <= 128
        && (start + 1..end)
            .all(|i| records[i].4 || roots[i] == roots[start] || roots[i] == roots[end])
}

/// 保留字典插入次序聚类平移证据，再以连续三字形、两种文本确认阴影。
pub fn confirmed_offsets(
    records: Vec<EvidenceRecord>,
    pairs: Vec<Offset>,
    exact: Vec<Pair>,
) -> Vec<Pair> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let all: Vec<_> = exact
        .into_iter()
        .chain(pairs.iter().map(|p| (p.0, p.1)))
        .collect();
    let roots = components(records.len(), &all);
    let mut bins: HashMap<(usize, i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut key_order = Vec::new();
    let mut clusters: Vec<Vec<Offset>> = Vec::new();
    for p in pairs {
        let r = &records[p.0];
        let normal = r.1;
        let key = (
            r.2,
            (p.2 / 0.1).floor() as i64,
            (p.3 / 0.1).floor() as i64,
            (normal / 0.1).floor() as i64,
        );
        let mut found = None;
        'search: for x in key.1 - 1..=key.1 + 1 {
            for y in key.2 - 1..=key.2 + 1 {
                for n in key.3 - 1..=key.3 + 1 {
                    if let Some(ids) = bins.get(&(key.0, x, y, n)) {
                        for &id in ids {
                            let first = clusters[id][0];
                            if (p.2 - first.2).abs() <= 0.1
                                && (p.3 - first.3).abs() <= 0.1
                                && (normal - records[first.0].1).abs() <= 0.1
                            {
                                found = Some(id);
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        if let Some(id) = found {
            clusters[id].push(p);
        } else {
            if !bins.contains_key(&key) {
                key_order.push(key);
            }
            bins.entry(key).or_default().push(clusters.len());
            clusters.push(vec![p]);
        }
    }
    let mut confirmed = Vec::new();
    for key in key_order {
        for &id in &bins[&key] {
            let ordered = &mut clusters[id];
            ordered.sort_by_key(|p| (p.0, p.1));
            let mut runs: Vec<Vec<Offset>> = Vec::new();
            for &p in ordered.iter() {
                let continuous = if let Some(prev) = runs.last().and_then(|r| r.last()) {
                    let pb = records[prev.0].0.unwrap();
                    let cb = records[p.0].0.unwrap();
                    p.0 > prev.0
                        && p.1 > prev.1
                        && cb[0] >= pb[0]
                        && cb[0] - pb[2] <= 1.0_f64.max(0.6 * (pb[3] - pb[1]).min(cb[3] - cb[1]))
                        && endpoints(&records, &roots, prev.0, p.0)
                        && endpoints(&records, &roots, prev.1, p.1)
                } else {
                    false
                };
                if !continuous {
                    runs.push(Vec::new());
                }
                runs.last_mut().unwrap().push(p);
            }
            for run in runs {
                let unique_roots: std::collections::HashSet<_> =
                    run.iter().map(|p| roots[p.0]).collect();
                let unique_text: std::collections::HashSet<_> =
                    run.iter().map(|p| records[p.0].3).collect();
                if unique_roots.len() >= 3 && unique_text.len() >= 2 {
                    confirmed.extend(run.into_iter().map(|p| (p.0, p.1)));
                }
            }
        }
    }
    confirmed
}

/// 用 Python 预计算的 sin/cos 投影，避免跨平台三角函数差异。
pub fn project(b: Box4, co: f64, si: f64) -> Box4 {
    let points = [(b[0], b[1]), (b[0], b[3]), (b[2], b[1]), (b[2], b[3])]
        .map(|(x, y)| (co * x + si * y, -si * x + co * y));
    let mut out = [points[0].0, points[0].1, points[0].0, points[0].1];
    for (x, y) in &points[1..] {
        out = [
            out[0].min(*x),
            out[1].min(*y),
            out[2].max(*x),
            out[3].max(*y),
        ];
    }
    out
}

pub type HiddenRecord = (
    Option<Box4>,
    Option<Size>,
    usize,
    i8,
    bool,
    bool,
    f64,
    f64,
    f64,
    usize,
);

/// 产生隐藏文本和可见文本的几何配对；Unicode 文本判据仍由 Python 执行。
pub fn hidden_candidates(records: Vec<HiddenRecord>) -> Option<Vec<(Vec<usize>, Vec<usize>)>> {
    if records.iter().any(|r| {
        r.0.is_some_and(|b| b.iter().any(|v| !v.is_finite() || v.abs() > 1e15))
    }) {
        return None;
    }
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    let mut key_order = Vec::new();
    let mut runs: Vec<Vec<usize>> = Vec::new();
    for (i, r) in records.iter().enumerate() {
        let Some(b) = r.0 else {
            continue;
        };
        if !r.5 {
            continue;
        }
        if [0, 1, 2, 4, 5, 6].contains(&r.3) && r.4 {
            let key = (
                ((b[0] + b[2]) / 2.0 / 32.0).floor() as i64,
                ((b[1] + b[3]) / 2.0 / 32.0).floor() as i64,
            );
            if !grid.contains_key(&key) {
                key_order.push(key);
            }
            grid.entry(key).or_default().push(i);
        } else if r.3 == 3 {
            let mut same = false;
            if let Some(previous) = runs.last().and_then(|v| v.last()).map(|j| &records[*j]) {
                if previous.2 == r.2 && (previous.6 - r.6).abs() <= 0.001 {
                    let pb = project(previous.0.unwrap(), r.7, r.8);
                    let cb = project(b, r.7, r.8);
                    if let (Some(po), Some(co)) = (previous.1, r.1) {
                        same = (-r.8 * (po[0] - co[0]) + r.7 * (po[1] - co[1])).abs() <= 0.1
                            && cb[0] >= pb[0]
                            && cb[0] - pb[2] <= 1.0_f64.max(0.6 * (cb[3] - cb[1]));
                    }
                }
            }
            if !same {
                runs.push(Vec::new());
            }
            runs.last_mut().unwrap().push(i);
        }
    }
    let mut result = Vec::new();
    for run in runs {
        let first = &records[run[0]];
        let boxes: Vec<_> = run.iter().map(|i| records[*i].0.unwrap()).collect();
        let b = crate::geometry::union(&boxes).unwrap();
        let p = project(b, first.7, first.8);
        let margin = 0.1 * (p[3] - p[1]);
        let x0 = ((b[0] - margin) / 32.0).floor() as i64;
        let x1 = ((b[2] + margin) / 32.0).floor() as i64;
        let y0 = ((b[1] - margin) / 32.0).floor() as i64;
        let y1 = ((b[3] + margin) / 32.0).floor() as i64;
        let keys: Vec<_> = if (x1 - x0 + 1) as i128 * (y1 - y0 + 1) as i128 > 4 * grid.len() as i128
        {
            key_order
                .iter()
                .copied()
                .filter(|(x, y)| x0 <= *x && *x <= x1 && y0 <= *y && *y <= y1)
                .collect()
        } else {
            (x0..=x1)
                .flat_map(|x| (y0..=y1).map(move |y| (x, y)))
                .collect()
        };
        let mut candidates = Vec::new();
        for key in keys {
            if let Some(items) = grid.get(&key) {
                for &i in items {
                    let r = &records[i];
                    if (r.6 - first.6).abs() > 0.001 {
                        continue;
                    }
                    let q = project(r.0.unwrap(), first.7, first.8);
                    if p[0] - margin <= (q[0] + q[2]) / 2.0
                        && (q[0] + q[2]) / 2.0 <= p[2] + margin
                        && p[1] <= (q[1] + q[3]) / 2.0
                        && (q[1] + q[3]) / 2.0 <= p[3]
                    {
                        candidates.push(i);
                    }
                }
            }
        }
        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|a, b| {
            let x = project(records[*a].0.unwrap(), first.7, first.8)[0];
            let y = project(records[*b].0.unwrap(), first.7, first.8)[0];
            x.partial_cmp(&y)
                .unwrap()
                .then(records[*a].9.cmp(&records[*b].9))
        });
        let projected: Vec<_> = candidates
            .iter()
            .map(|i| project(records[*i].0.unwrap(), first.7, first.8))
            .collect();
        let v = crate::geometry::union(&projected).unwrap();
        let axis = 0.0_f64.max(p[2].min(v[2]) - p[0].max(v[0]));
        let normal = 0.0_f64.max(p[3].min(v[3]) - p[1].max(v[1]));
        if axis < 0.8 * (p[2] - p[0]) || normal < 0.8 * (p[3] - p[1]).min(v[3] - v[1]) {
            continue;
        }
        result.push((run, candidates));
    }
    Some(result)
}
