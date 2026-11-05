//! 在文档内连续累积 X、Y 和跨页样式风险，不回传逐字符中间对象。
use std::collections::{HashMap, HashSet};

use crate::{
    geometry::{anchor_pairs, Box4, Size},
    median,
    statistics::ordered_clusters,
};

pub type Entry = (usize, Box4, Box4, Size, f64);
pub type RunKey = (String, u64, i32, i32, i32, String);

/// 保留线性插值分位数，不复用其他模块的取整采样规则。
pub(crate) fn quantile(mut values: Vec<f64>, fraction: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let position = fraction * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return values[lower];
    }
    let weight = position - lower as f64;
    values[lower] * (1.0 - weight) + values[upper] * weight
}

#[derive(Default)]
struct Run {
    ratios: Vec<f64>,
    overlaps: usize,
    line_count: usize,
    inflated: HashSet<(usize, i64)>,
    extreme: usize,
}

#[derive(Default)]
pub struct Risk {
    runs: HashMap<usize, Run>,
    y_ratios: Vec<f64>,
    split: bool,
}

impl Risk {
    /// 消费一行原始锚点；校验失败返回 None，由边界选择显式参考计算。
    pub fn add_line(
        &mut self,
        page: usize,
        source: i64,
        height: f64,
        skip_y: bool,
        entries: Vec<Entry>,
    ) -> Option<bool> {
        if !height.is_finite()
            || entries.iter().any(|e| {
                !e.4.is_finite() || e.1.iter().chain(&e.2).chain(&e.3).any(|v| !v.is_finite())
            })
        {
            return None;
        }
        if entries.is_empty() {
            return Some(false);
        }
        let q75 = quantile(entries.iter().map(|e| e.2[3] - e.2[1]).collect(), 0.75);
        if !q75.is_finite() {
            return None;
        }
        let anchors: Vec<_> = entries
            .into_iter()
            .filter(|e| e.2[3] - e.2[1] >= 0.38 * q75)
            .collect();
        // 先验证相邻对；任何后续拒绝都会要求边界丢弃整个累积器。
        let pairs = anchor_pairs(anchors.clone(), false)?;
        for (index, _, ratio, overlap) in pairs {
            let run = self.runs.entry(anchors[index].0).or_default();
            run.ratios.push(ratio);
            run.overlaps += usize::from(overlap);
        }
        let mut by_run: HashMap<usize, Vec<&Entry>> = HashMap::new();
        for entry in &anchors {
            by_run.entry(entry.0).or_default().push(entry);
        }
        for (id, group) in by_run {
            let run = self.runs.entry(id).or_default();
            run.line_count += 1;
            let fonts: Vec<_> = group
                .iter()
                .filter_map(|e| (e.4 > 0.0).then_some(e.4))
                .collect();
            let tight = quantile(group.iter().map(|e| e.2[3] - e.2[1]).collect(), 0.75);
            let scale = median(fonts).max(tight);
            if scale >= 4.0 && height > 1.50 * scale && height > 1.80 * 0.1_f64.max(tight) {
                run.inflated.insert((page, source));
            }
        }
        if anchors.len() < 4 || skip_y {
            return Some(false);
        }
        let groups = ordered_clusters(
            anchors.iter().map(|e| e.3[1]).collect(),
            0.5_f64.max(0.25 * q75),
            0.0,
            true,
        )?;
        if groups
            .iter()
            .filter(|g| g.len() >= 3 && g.len() as f64 / anchors.len() as f64 >= 0.20)
            .count()
            >= 2
        {
            self.split = true;
            return Some(true);
        }
        // Python max 在并列时选择首簇，不能使用选择末项的 max_by_key。
        let mut dominant = &groups[0];
        for group in &groups[1..] {
            if group.len() > dominant.len() {
                dominant = group;
            }
        }
        if dominant.len() as f64 / (anchors.len() as f64) < 0.80 {
            return Some(false);
        }
        let first = &anchors[dominant[0]];
        let (mut source_top, mut source_bottom, mut tight_top, mut tight_bottom) =
            (first.1[1], first.1[3], first.2[1], first.2[3]);
        let mut counts = HashMap::<usize, usize>::new();
        let mut order = Vec::new();
        for index in dominant {
            let entry = &anchors[*index];
            source_top = source_top.min(entry.1[1]);
            source_bottom = source_bottom.max(entry.1[3]);
            tight_top = tight_top.min(entry.2[1]);
            tight_bottom = tight_bottom.max(entry.2[3]);
            let count = counts.entry(entry.0).or_default();
            if *count == 0 {
                order.push(entry.0);
            }
            *count += 1;
        }
        let tight_height = tight_bottom - tight_top;
        if tight_height <= 0.0 {
            return Some(false);
        }
        let ratio = (source_bottom - source_top) / tight_height;
        if !ratio.is_finite() {
            return None;
        }
        self.y_ratios.push(ratio);
        if ratio >= 3.0 {
            let mut winner = order[0];
            for id in order {
                if counts[&id] > counts[&winner] {
                    winner = id;
                }
            }
            self.runs.entry(winner).or_default().extreme += 1;
        }
        Some(false)
    }

    /// 汇总完整文档统计，保留 split 提前返回时 style 为 false 的语义。
    pub fn finish(&self) -> (bool, bool) {
        if self.split {
            return (true, false);
        }
        let layout = self.runs.values().any(|r| {
            let n = r.ratios.len();
            n >= 30
                && median(r.ratios.clone()) >= 1.30
                && r.ratios.iter().filter(|v| **v > 1.35).count() as f64 / n as f64 >= 0.30
                && r.overlaps as f64 / n as f64 >= 0.30
        }) || (!self.y_ratios.is_empty()
            && (quantile(self.y_ratios.clone(), 0.95) >= 2.20
                || self.runs.values().any(|r| r.extreme >= 3)));
        let style = self.runs.values().any(|r| {
            r.line_count >= 3
                && r.inflated.len() >= 3
                && r.inflated.len() as f64 / r.line_count as f64 >= 0.50
                && r.inflated.iter().map(|v| v.0).collect::<HashSet<_>>().len() >= 2
        });
        (layout, style)
    }
}
