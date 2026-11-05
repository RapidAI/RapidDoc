//! 复用同一批 run 与样本，连续计算跨页样式异常和全文逐行字号校准。
use crate::{
    geometry_risk::quantile,
    geometry_runs::{Run, Sample},
    median,
};
use std::collections::{HashMap, HashSet};

pub struct Metadata {
    pub page: i64,
    pub source: i64,
    pub height: f64,
}

pub struct Style {
    pub inflated: Vec<usize>,
    pub scales: Vec<(usize, f64)>,
}

/// 只复用已有数值，不物化 Python 字符；保持首锚点高度、重复来源行和并列字体规则。
pub fn prepare(
    samples: &[Sample],
    lines: &[Vec<usize>],
    runs: &[Run],
    metadata: &[Metadata],
) -> Option<Style> {
    if samples.len() != metadata.len()
        || metadata.iter().any(|m| !m.height.is_finite())
        || samples
            .iter()
            .any(|s| !(s.tight[3] - s.tight[1]).is_finite())
    {
        return None;
    }
    let mut inflated = Vec::new();
    for (id, run) in runs.iter().enumerate() {
        let mut groups: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for index in &run.members {
            if samples[*index].anchor {
                let m = &metadata[*index];
                groups.entry((m.page, m.source)).or_default().push(*index);
            }
        }
        if groups.len() < 3 {
            continue;
        }
        let mut count = 0usize;
        let mut pages = HashSet::new();
        for ((page, _), members) in &groups {
            let fonts = members
                .iter()
                .filter_map(|i| (samples[*i].size > 0.0).then_some(samples[*i].size))
                .collect();
            let tight = quantile(
                members
                    .iter()
                    .map(|i| samples[*i].tight[3] - samples[*i].tight[1])
                    .collect(),
                0.75,
            );
            let scale = median(fonts).max(tight);
            let height = metadata[members[0]].height;
            if scale >= 4.0 && height > 1.50 * scale && height > 1.80 * 0.1_f64.max(tight) {
                count += 1;
                pages.insert(*page);
            }
        }
        if count >= 3 && count as f64 / groups.len() as f64 >= 0.50 && pages.len() >= 2 {
            inflated.push(id);
        }
    }
    let mut scales = Vec::new();
    // 全文异常触发后校准所有行，不能只处理属于异常 run 的行。
    if !inflated.is_empty() {
        for (line_index, line) in lines.iter().enumerate() {
            let mut counts = HashMap::<usize, usize>::new();
            let mut order = Vec::new();
            for index in line {
                let sample = &samples[*index];
                if sample.anchor {
                    let count = counts.entry(sample.run).or_default();
                    if *count == 0 {
                        order.push(sample.run);
                    }
                    *count += 1;
                }
            }
            let Some(&first) = order.first() else {
                continue;
            };
            let mut dominant = first;
            for id in order {
                if counts[&id] > counts[&dominant] {
                    dominant = id;
                }
            }
            let members: Vec<_> = line
                .iter()
                .copied()
                .filter(|i| samples[*i].anchor && samples[*i].run == dominant)
                .collect();
            let fonts = members
                .iter()
                .filter_map(|i| (samples[*i].size > 0.0).then_some(samples[*i].size))
                .collect();
            let tight = quantile(
                members
                    .iter()
                    .map(|i| samples[*i].tight[3] - samples[*i].tight[1])
                    .collect(),
                0.75,
            );
            let scale = median(fonts).max(tight).max(0.1);
            if scale >= 4.0 {
                scales.push((line_index, scale));
            }
        }
    }
    Some(Style { inflated, scales })
}

/// 持有整本样式文档的数值样本，跨阶段不创建 Python 字符对象。
#[derive(Default)]
pub struct Document {
    samples: Vec<Sample>,
    metadata: Vec<Metadata>,
    lines: Vec<Vec<usize>>,
    line_keys: Vec<(i64, i64)>,
    line_ids: HashMap<(i64, i64), usize>,
}

pub type Report = (
    usize,
    usize,
    usize,
    Option<f64>,
    Option<f64>,
    f64,
    f64,
    bool,
    bool,
    bool,
);
pub type Result = (Vec<usize>, Vec<((i64, i64), f64)>, Vec<Report>);

/// 完成锚点、run 和样式计算后的自有数据，布局分支仅在后续修复边界物化。
pub type Legacy = (Option<crate::geometry::Box4>, crate::geometry::Size, i32);

pub struct Prepared {
    pub samples: Vec<Sample>,
    pub metadata: Vec<Metadata>,
    pub lines: Vec<Vec<usize>>,
    pub line_keys: Vec<(i64, i64)>,
    pub runs: Vec<Run>,
    pub style: Style,
    pub restored: Vec<Option<crate::geometry::Box4>>,
}

impl Document {
    /// 接收已完成坐标转换和字体编码的一行，保持重复来源行的成员追加次序。
    pub fn append(&mut self, page: i64, source: i64, height: f64, records: Vec<Sample>) {
        if records.is_empty() {
            return;
        }
        let key = (page, source);
        let line = *self.line_ids.entry(key).or_insert_with(|| {
            self.line_keys.push(key);
            self.lines.push(Vec::new());
            self.lines.len() - 1
        });
        for sample in records {
            self.lines[line].push(self.samples.len());
            self.samples.push(sample);
            self.metadata.push(Metadata {
                page,
                source,
                height,
            });
        }
    }

    /// 只在 Rust 内生成完整前置数据，样本坐标不经过 Python 往返。
    pub fn prepare(mut self, families: &[usize], legacy: Option<Vec<Legacy>>) -> Option<Prepared> {
        for line in &self.lines {
            let heights = line
                .iter()
                .filter_map(|i| {
                    self.samples[*i]
                        .anchor
                        .then_some(self.samples[*i].tight[3] - self.samples[*i].tight[1])
                })
                .collect::<Vec<_>>();
            if heights.iter().any(|h| !h.is_finite()) {
                return None;
            }
            let q75 = quantile(heights, 0.75);
            for index in line {
                let sample = &mut self.samples[*index];
                sample.anchor =
                    sample.anchor && q75 > 0.0 && sample.tight[3] - sample.tight[1] >= 0.38 * q75;
            }
        }
        let mut runs = crate::geometry_runs::group(&self.samples, families)?;
        let style = prepare(&self.samples, &self.lines, &runs, &self.metadata)?;
        let mut restored = Vec::new();
        if !style.inflated.is_empty() {
            if let Some(legacy) = legacy {
                if legacy.len() != self.samples.len() {
                    return None;
                }
                for (sample, (raw, size, angle)) in self.samples.iter_mut().zip(legacy) {
                    let bbox = crate::geometry::clip(crate::geometry::normalize(raw, false), size);
                    if let Some(bbox) = bbox {
                        sample.source = crate::geometry::rotate(bbox, size, angle);
                    }
                    restored.push(bbox);
                }
            }
        }
        crate::geometry_runs::complete(&self.samples, &self.lines, families, &mut runs)?;
        Some(Prepared {
            samples: self.samples,
            metadata: self.metadata,
            lines: self.lines,
            line_keys: self.line_keys,
            runs,
            style,
            restored,
        })
    }

    /// 样式分支只返回最终诊断，消费完成即释放全部字符数据。
    pub fn finish(self, families: Vec<usize>) -> Option<Result> {
        let Prepared {
            runs,
            style,
            line_keys,
            ..
        } = self.prepare(&families, None)?;
        let reports = reports(&runs, &style);
        Some((
            style.inflated,
            style
                .scales
                .into_iter()
                .map(|(i, v)| (line_keys[i], v))
                .collect(),
            reports,
        ))
    }
}

/// 从已完成统计的自有 run 导出诊断数值，不构造 Python run 对象。
pub fn reports(runs: &[Run], style: &Style) -> Vec<Report> {
    let inflated: HashSet<_> = style.inflated.iter().copied().collect();
    let mut reports = Vec::with_capacity(runs.len());
    for (id, run) in runs.iter().enumerate() {
        let count = run.ratios.len();
        reports.push((
            id,
            run.members.len(),
            count,
            (count > 0).then(|| median(run.ratios.clone())),
            (count > 0).then(|| quantile(run.ratios.clone(), 0.9)),
            if count > 0 {
                run.ratios.iter().filter(|v| **v > 1.35).count() as f64 / count as f64
            } else {
                0.0
            },
            if count > 0 {
                run.overlaps.iter().filter(|v| **v).count() as f64 / count as f64
            } else {
                0.0
            },
            run.strong,
            run.sibling,
            inflated.contains(&id),
        ));
    }
    reports
}
