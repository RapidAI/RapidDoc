//! 在快照字符上连续构造样式候选、匹配装饰线并输出紧凑区间。
use crate::{geometry::Box4, median};
use std::collections::HashMap;

pub struct Character {
    pub bbox: Box4,
    pub tight: Option<Box4>,
    pub index: usize,
    pub fragment: String,
    pub visible: bool,
    pub space: bool,
    pub marker: bool,
    pub bold: bool,
}
pub struct Line {
    pub bbox: Box4,
    pub source: i64,
    pub members: Vec<usize>,
}
pub struct Drawing {
    pub bbox: Box4,
    pub width: f64,
}
pub type Payload = (Box4, String, Vec<(usize, usize, u8)>, i64);
struct Candidate {
    line: Line,
    visible: Vec<usize>,
    height: f64,
    center: f64,
    bottom: f64,
    styles: Vec<u8>,
}
struct Match {
    line: usize,
    style: usize,
    start: usize,
    end: usize,
    distance: f64,
    overlap: f64,
}

/// 与 Python 严格框规范一致，零面积框不参与候选。
fn valid(b: Box4) -> bool {
    b.iter().all(|v| v.is_finite()) && b[2] > b[0] && b[3] > b[1]
}

/// 按原始字符索引稳定排序，过滤短粗体与分离的行首项目符号。
fn candidate(mut line: Line, chars: &[Character], min_bold: usize) -> Option<Candidate> {
    if !valid(line.bbox) {
        return None;
    }
    line.members.sort_by_key(|&i| chars[i].index);
    let visible: Vec<_> = line
        .members
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| (chars[m].visible && valid(chars[m].bbox)).then_some(i))
        .collect();
    if visible.is_empty() {
        return None;
    }
    let height = median(
        visible
            .iter()
            .map(|&i| {
                let b = chars[line.members[i]].bbox;
                b[3] - b[1]
            })
            .collect(),
    );
    if height <= 0.0 {
        return None;
    }
    let body: Vec<_> = visible
        .iter()
        .map(|&i| chars[line.members[i]].bbox)
        .filter(|b| b[3] - b[1] >= 0.8 * height)
        .collect();
    if body.is_empty() {
        return None;
    }
    let center = median(body.iter().map(|b| (b[1] + b[3]) / 2.0).collect());
    let bottom = median(body.iter().map(|b| b[3]).collect());
    let mut styles: Vec<u8> = line
        .members
        .iter()
        .map(|&i| u8::from(chars[i].bold))
        .collect();
    let comparable: Vec<_> = line
        .members
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| (!chars[m].fragment.is_empty()).then_some(i))
        .collect();
    let mut start = 0;
    while start < comparable.len() {
        if styles[comparable[start]] & 1 == 0 {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < comparable.len() && styles[comparable[end]] & 1 != 0 {
            end += 1;
        }
        let short = comparable[start..end]
            .iter()
            .map(|&i| chars[line.members[i]].fragment.chars().count())
            .sum::<usize>()
            < min_bold;
        let marker = start == 0
            && end < comparable.len()
            && comparable[start..end]
                .iter()
                .all(|&i| chars[line.members[i]].marker)
            && {
                let a = comparable[end - 1];
                let b = comparable[end];
                line.members[a + 1..b].iter().any(|&i| chars[i].space) || {
                    let left = chars[line.members[a]].bbox;
                    let right = chars[line.members[b]].bbox;
                    valid(left) && valid(right) && right[0] - left[2] >= 0.5 * height
                }
            };
        if short || marker {
            for &i in &comparable[start..end] {
                styles[i] &= !1;
            }
        }
        start = end;
    }
    Some(Candidate {
        line,
        visible,
        height,
        center,
        bottom,
        styles,
    })
}

/// 保留目标距离、覆盖、端点和墨迹穿越的全部阈值，返回原字符半开区间。
fn drawing_match(
    line: &Candidate,
    chars: &[Character],
    drawing: &Drawing,
    style: usize,
    thresholds: &[f64; 8],
) -> Option<Match> {
    let b = drawing.bbox;
    let length = b[2] - b[0];
    if length < thresholds[0] * line.height {
        return None;
    }
    let y = (b[1] + b[3]) / 2.0;
    let target = if style == 0 { line.bottom } else { line.center };
    let distance = (y - target).abs() / line.height;
    if distance > thresholds[1 + style] || drawing.width.max(0.0) > thresholds[3] * line.height {
        return None;
    }
    let hit: Vec<_> = line
        .visible
        .iter()
        .copied()
        .filter(|&i| {
            let c = chars[line.line.members[i]].bbox;
            b[0] <= (c[0] + c[2]) / 2.0 && (c[0] + c[2]) / 2.0 <= b[2]
        })
        .collect();
    if hit.is_empty() {
        return None;
    }
    if style == 1 {
        let boxes: Vec<_> = hit
            .iter()
            .map(|&i| chars[line.line.members[i]].tight.filter(|&b| valid(b)))
            .collect();
        if boxes.iter().all(Option::is_some)
            && !boxes.iter().flatten().any(|b| b[1] <= y && y <= b[3])
        {
            return None;
        }
    }
    let left = hit
        .iter()
        .map(|&i| chars[line.line.members[i]].bbox[0])
        .fold(f64::INFINITY, f64::min);
    let right = hit
        .iter()
        .map(|&i| chars[line.line.members[i]].bbox[2])
        .fold(f64::NEG_INFINITY, f64::max);
    if (right - left) / length < thresholds[4]
        || (b[0] - left).abs().min((b[2] - right).abs()) > thresholds[5] * line.height
    {
        return None;
    }
    let overlap = (line.line.bbox[2].min(b[2]) - line.line.bbox[0].max(b[0])).max(0.0)
        / (line.line.bbox[2] - line.line.bbox[0])
            .min(length)
            .max(0.01);
    Some(Match {
        line: 0,
        style,
        start: *hit.first().unwrap(),
        end: hit.last().unwrap() + 1,
        distance,
        overlap,
    })
}

/// 先建立相同网格超集，再逐条绘图线选择同一最优目标；极端网格范围返回参考选择。
pub fn detect(
    chars: &[Character],
    lines: Vec<Line>,
    drawings: &[Drawing],
    thresholds: [f64; 8],
    min_bold: usize,
) -> Option<Vec<Payload>> {
    let mut candidates: Vec<_> = lines
        .into_iter()
        .filter_map(|line| candidate(line, chars, min_bold))
        .collect();
    if candidates.is_empty() {
        return Some(Vec::new());
    }
    if !drawings.is_empty() {
        let grid = median(candidates.iter().map(|c| c.height).collect()).max(1.0);
        let mut anchors: HashMap<i64, Vec<(usize, usize)>> = HashMap::new();
        let mut tops: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, c) in candidates.iter().enumerate() {
            if (c.line.bbox[1] / grid).abs() > 1e15 {
                return None;
            }
            tops.entry((c.line.bbox[1] / grid).floor() as i64)
                .or_default()
                .push(i);
            for (style, target) in [c.bottom, c.center].into_iter().enumerate() {
                let tolerance = thresholds[1 + style] * c.height;
                let start = ((target - tolerance) / grid).floor();
                let end = ((target + tolerance) / grid).floor();
                if start.abs() > 1e15 || end.abs() > 1e15 || end - start > 4096.0 {
                    return None;
                }
                for cell in start as i64..=end as i64 {
                    anchors.entry(cell).or_default().push((i, style));
                }
            }
        }
        for drawing in drawings {
            if !valid(drawing.bbox) {
                continue;
            }
            let y = (drawing.bbox[1] + drawing.bbox[3]) / 2.0;
            if (y / grid).abs() > 1e15 {
                return None;
            }
            let mut best: Option<Match> = None;
            for &(i, style) in anchors
                .get(&((y / grid).floor() as i64))
                .into_iter()
                .flatten()
            {
                let c = &candidates[i];
                let Some(mut m) = drawing_match(c, chars, drawing, style, &thresholds) else {
                    continue;
                };
                if style == 0 {
                    let max_top = y + thresholds[6] * c.height;
                    let start = (y / grid).floor();
                    let end = (max_top / grid).floor();
                    if end.abs() > 1e15 || end - start > 4096.0 {
                        return None;
                    }
                    let fraction = (start as i64..=end as i64).any(|cell| {
                        tops.get(&cell).into_iter().flatten().any(|&j| {
                            let b = candidates[j].line.bbox;
                            j != i
                                && y <= b[1]
                                && b[1] <= max_top
                                && (drawing.bbox[2].min(b[2]) - drawing.bbox[0].max(b[0])).max(0.0)
                                    / (b[2] - b[0]).max(0.01)
                                    >= thresholds[7]
                        })
                    });
                    if fraction {
                        continue;
                    }
                }
                m.line = i;
                let key = (m.distance, -m.overlap, c.line.source, m.style);
                if best.as_ref().is_none_or(|b| {
                    key < (
                        b.distance,
                        -b.overlap,
                        candidates[b.line].line.source,
                        b.style,
                    )
                }) {
                    best = Some(m);
                }
            }
            if let Some(m) = best {
                for value in &mut candidates[m.line].styles[m.start..m.end] {
                    *value |= 2 << m.style;
                }
            }
        }
    }
    let mut output = Vec::new();
    for c in candidates {
        let mut text = String::new();
        let mut ranges = Vec::new();
        let mut offset = 0;
        let mut active = 0;
        let mut active_start = 0;
        for (i, &m) in c.line.members.iter().enumerate() {
            let fragment = &chars[m].fragment;
            if fragment.is_empty() {
                continue;
            }
            let style = c.styles[i];
            if style != active {
                if active != 0 {
                    ranges.push((active_start, offset, active));
                }
                active_start = offset;
                active = style;
            }
            text.push_str(fragment);
            offset += fragment.chars().count();
        }
        if active != 0 {
            ranges.push((active_start, offset, active));
        }
        if !text.is_empty() {
            output.push((c.line.bbox, text, ranges, c.line.source));
        }
    }
    if !output.iter().any(|r| !r.2.is_empty()) {
        return Some(Vec::new());
    }
    output.sort_by(|a, b| {
        a.3.cmp(&b.3)
            .then_with(|| a.0[1].partial_cmp(&b.0[1]).unwrap())
            .then_with(|| a.0[0].partial_cmp(&b.0[0]).unwrap())
    });
    Some(output)
}
