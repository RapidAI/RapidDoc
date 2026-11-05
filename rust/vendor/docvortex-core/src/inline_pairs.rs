//! 行内上下标保序配对；仅计算数值关系，文字和对象物化留在 Python。
use crate::geometry::Box4;

pub type Record = (Box4, f64, f64, usize, bool, usize, usize);
type Choice = (f64, usize, bool, bool);

/// 复现主体与小字号框的低重叠外置判断。
fn detached(a: Box4, b: Box4, h: f64) -> bool {
    let width = 0.0_f64.max(a[2] - a[0]);
    let edge = (b[0] - a[2]).abs().min((a[0] - b[2]).abs());
    let gap = 0.0_f64.max(b[1] - a[3]).max(a[1] - b[3]);
    let center = ((a[1] + a[3]) / 2.0 - (b[1] + b[3]) / 2.0).abs();
    let outside = (b[1] - a[1]).max(a[3] - b[3]);
    width <= 0.75 * h
        && edge <= 1.0_f64.max(0.1 * h)
        && gap <= 0.5_f64.max(0.15 * h)
        && center >= 0.5_f64.max(0.25 * h)
        && outside >= 0.5_f64.max(0.2 * h)
}

/// 为已验证记录计算候选度量；布尔位置 true 表示前缀。
fn candidate(a: &Record, b: &Record) -> Option<(f64, bool, bool)> {
    if a.5 == b.5 || a.6 != b.6 || a.1 <= 0.0 || b.1 <= 0.0 {
        return None;
    }
    let legacy_a = 0.1_f64.max(a.1);
    let legacy_b = 0.1_f64.max(b.1);
    let ratio = legacy_a / legacy_b;
    let canonical = a.4 && !(0.35..=0.8).contains(&ratio) && (0.35..=0.8).contains(&(a.2 / b.2));
    let (small, base) = if canonical {
        (a.2, b.2)
    } else {
        (legacy_a, legacy_b)
    };
    if !(0.35..=0.8).contains(&(small / base)) || (a.3 > 8 && a.0[2] - a.0[0] > 3.0 * base) {
        return None;
    }
    let overlap = 0.0_f64.max(a.0[3].min(b.0[3]) - a.0[1].max(b.0[1]));
    let overlap_ratio = overlap / 0.1_f64.max(a.0[3] - a.0[1]);
    let outside_pair = overlap_ratio < 0.5 && detached(a.0, b.0, base);
    if overlap_ratio < 0.5 && !outside_pair {
        return None;
    }
    let center = ((a.0[1] + a.0[3]) / 2.0 - (b.0[1] + b.0[3]) / 2.0).abs();
    if center < 0.5_f64.max(0.12 * base) {
        return None;
    }
    let mut best: Option<(f64, bool, f64)> = None;
    for (prefix, gap) in [(true, b.0[0] - a.0[2]), (false, a.0[0] - b.0[2])] {
        if gap >= -0.35 * base
            && gap <= 1.5_f64.max(0.35 * base)
            && best.is_none_or(|old| gap.abs() < old.0)
        {
            best = Some((gap.abs(), prefix, gap));
        }
    }
    let (_, prefix, gap) = best?;
    let outside = (b.0[1] - a.0[1]).max(a.0[3] - b.0[3]);
    if outside < 0.5_f64.max(0.08 * base) && gap.abs() > 1.0_f64.max(0.1 * base) {
        return None;
    }
    Some((
        gap.abs() + (1.0 - overlap_ratio) * base,
        prefix,
        outside_pair,
    ))
}

/// 每页建立边缘索引，严格按小行和主体原索引裁决相同度量，存储规模为线性。
pub fn matches(records: Vec<Record>) -> Option<Vec<(usize, usize, bool, bool)>> {
    let n = records.len();
    if records.iter().any(|r| {
        r.6 >= n
            || r.0.iter().any(|x| !x.is_finite() || x.abs() > 1e100)
            || r.0[2] <= r.0[0]
            || r.0[3] <= r.0[1]
            || !r.1.is_finite()
            || r.1.abs() > 1e100
            || !r.2.is_finite()
            || r.2 <= 0.0
            || r.2 > 1e100
    }) {
        return None;
    }
    let count = records.iter().map(|r| r.6).max().map_or(0, |x| x + 1);
    let mut left = vec![Vec::new(); count];
    let mut right = left.clone();
    let mut scales = vec![0.1_f64; count];
    for (i, r) in records.iter().enumerate() {
        left[r.6].push((r.0[0], i));
        right[r.6].push((r.0[2], i));
        scales[r.6] = scales[r.6].max(r.2).max(0.1_f64.max(r.1));
    }
    for values in left.iter_mut().chain(right.iter_mut()) {
        values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    }
    let mut best_base: Vec<Option<Choice>> = vec![None; n];
    let mut best_small: Vec<[Option<(f64, usize)>; 2]> = vec![[None, None]; n];
    for (i, a) in records.iter().enumerate() {
        if a.3 == 0 {
            continue;
        }
        let padding = 0.35 * scales[a.6];
        let gap = 1.5_f64.max(padding);
        let mut bases = Vec::new();
        for (values, low, high) in [
            (&left[a.6], a.0[2] - padding, a.0[2] + gap),
            (&right[a.6], a.0[0] - gap, a.0[0] + padding),
        ] {
            let start = values.partition_point(|v| v.0 < low);
            let end = values.partition_point(|v| v.0 <= high);
            bases.extend(values[start..end].iter().map(|v| v.1));
        }
        bases.sort_unstable();
        bases.dedup();
        for j in bases {
            if i == j {
                continue;
            }
            if let Some((metric, prefix, outside)) = candidate(a, &records[j]) {
                if best_base[i].is_none_or(|old| metric < old.0) {
                    best_base[i] = Some((metric, j, prefix, outside));
                }
                let slot = usize::from(!prefix);
                if best_small[j][slot].is_none_or(|old| metric < old.0) {
                    best_small[j][slot] = Some((metric, i));
                }
            }
        }
    }
    Some(
        best_base
            .iter()
            .enumerate()
            .filter_map(|(i, best)| {
                let (_, base, prefix, outside) = (*best)?;
                (best_small[base][usize::from(!prefix)].map(|x| x.1) == Some(i))
                    .then_some((i, base, prefix, outside))
            })
            .collect(),
    )
}
