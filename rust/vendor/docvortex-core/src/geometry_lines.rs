//! 在自有样本上连续计算行字形、主基线与最终 Y 风险，避免 Python 字符往返。
use crate::{
    geometry::Box4, geometry_risk::quantile, geometry_style::Prepared, median,
    statistics::ordered_clusters,
};
use std::collections::HashMap;

pub struct World {
    pub tight: Box4,
    pub canonical: bool,
    pub y_eligible: bool,
}

pub type Summary = (Vec<((i64, i64), Box4)>, Vec<((i64, i64), f64)>, bool);

/// 保持 Python min/max 首见值，包括相等坐标及有符号零。
fn union(boxes: impl Iterator<Item = Box4>) -> Option<Box4> {
    let mut boxes = boxes;
    let mut result = boxes.next()?;
    for bbox in boxes {
        for axis in 0..4 {
            if (axis < 2 && bbox[axis] < result[axis]) || (axis >= 2 && bbox[axis] > result[axis]) {
                result[axis] = bbox[axis];
            }
        }
    }
    Some(result)
}

/// 模拟 CPython 3.12 起的补偿求和；旧版本仍按原顺序普通相加。
fn width_sum(values: impl Iterator<Item = f64>, compensated: bool) -> Option<f64> {
    let (mut hi, mut lo) = (0.0_f64, 0.0_f64);
    for value in values {
        let next = hi + value;
        if compensated {
            lo += if hi.abs() >= value.abs() {
                (hi - next) + value
            } else {
                (value - next) + hi
            };
        }
        hi = next;
        if !hi.is_finite() || !lo.is_finite() {
            return None;
        }
    }
    let total = if compensated && lo != 0.0 {
        hi + lo
    } else {
        hi
    };
    total.is_finite().then_some(total)
}

/// 共享基线聚类，但 canonical 按宽度优先、Y 风险按成员数优先，保留不同裁决语义。
pub fn summarize(prepared: &Prepared, world: &[World], compensated: bool) -> Option<Summary> {
    if world.len() != prepared.samples.len()
        || world.iter().any(|w| w.tight.iter().any(|v| !v.is_finite()))
    {
        return None;
    }
    let samples = &prepared.samples;
    let mut inks = Vec::new();
    let mut baselines = Vec::new();
    let mut ratios = Vec::new();
    let mut extremes = HashMap::<usize, usize>::new();
    let mut y_risk = false;
    for (key, line) in prepared.line_keys.iter().zip(&prepared.lines) {
        let Some(&first) = line.first() else {
            continue;
        };
        inks.push((*key, union(line.iter().map(|i| world[*i].tight))?));
        let anchors: Vec<_> = line
            .iter()
            .copied()
            .filter(|i| samples[*i].anchor)
            .collect();
        let canonical = world[first].canonical && anchors.len() >= 2;
        let risk = !y_risk && world[first].y_eligible && anchors.len() >= 4;
        if !canonical && !risk {
            continue;
        }
        let heights: Vec<_> = anchors
            .iter()
            .map(|i| samples[*i].tight[3] - samples[*i].tight[1])
            .collect();
        let tolerance = 0.5_f64.max(0.25 * quantile(heights, 0.75));
        let groups = ordered_clusters(
            anchors.iter().map(|i| samples[*i].origin[1]).collect(),
            tolerance,
            0.0,
            false,
        )?;
        if canonical {
            let mut best = None;
            for group in &groups {
                let width = width_sum(
                    group.iter().map(|i| {
                        let b = samples[anchors[*i]].tight;
                        b[2] - b[0]
                    }),
                    compensated,
                )?;
                let score = (width, group.len());
                if best.as_ref().is_none_or(|(previous, _)| score > *previous) {
                    best = Some((score, group));
                }
            }
            if let Some((_, dominant)) = best {
                if dominant.len() as f64 / anchors.len() as f64 >= 0.5 {
                    baselines.push((
                        *key,
                        median(
                            dominant
                                .iter()
                                .map(|i| samples[anchors[*i]].origin[1])
                                .collect(),
                        ),
                    ));
                }
            }
        }
        if !risk {
            continue;
        }
        if groups
            .iter()
            .filter(|g| g.len() >= 3 && g.len() as f64 / anchors.len() as f64 >= 0.20)
            .count()
            >= 2
        {
            y_risk = true;
            continue;
        }
        // max_by_key 会选最后一个并列项；显式比较以保持 Python max 的首见顺序。
        let mut dominant = &groups[0];
        for group in &groups[1..] {
            if group.len() > dominant.len() {
                dominant = group;
            }
        }
        if dominant.len() as f64 / (anchors.len() as f64) < 0.80 {
            continue;
        }
        let source = union(dominant.iter().map(|i| samples[anchors[*i]].source))?;
        let tight = union(dominant.iter().map(|i| samples[anchors[*i]].tight))?;
        let height = tight[3] - tight[1];
        if height <= 0.0 {
            continue;
        }
        let ratio = (source[3] - source[1]) / height;
        if !ratio.is_finite() {
            return None;
        }
        ratios.push(ratio);
        if ratio >= 3.0 {
            let mut counts = HashMap::<usize, usize>::new();
            for i in dominant {
                *counts.entry(samples[anchors[*i]].run).or_default() += 1;
            }
            let mut run = samples[anchors[dominant[0]]].run;
            for i in dominant {
                let candidate = samples[anchors[*i]].run;
                if counts[&candidate] > counts[&run] {
                    run = candidate;
                }
            }
            *extremes.entry(run).or_default() += 1;
        }
    }
    y_risk |= !ratios.is_empty()
        && (quantile(ratios, 0.95) >= 2.20 || extremes.values().any(|n| *n >= 3));
    Some((inks, baselines, y_risk))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// 保留 Python 3.11 与 3.12+ 的不同求和结果，避免并列主基线改变。
    fn width_sum_matches_python_versions() {
        assert_eq!(width_sum([1e16, 1.0, 1.0].into_iter(), false), Some(1e16));
        assert_eq!(
            width_sum([1e16, 1.0, 1.0].into_iter(), true),
            Some(1e16 + 2.0)
        );
        assert_eq!(width_sum([f64::MAX, f64::MAX].into_iter(), true), None);
    }

    #[test]
    /// 并集保留首次出现的有符号零，不用浮点 min/max 改写坐标位模式。
    fn union_preserves_first_equal_coordinate() {
        let result = union([[0.0, -0.0, 1.0, 2.0], [-0.0, 0.0, 1.0, 2.0]].into_iter()).unwrap();
        assert_eq!(result[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(result[1].to_bits(), (-0.0_f64).to_bits());
    }
}
