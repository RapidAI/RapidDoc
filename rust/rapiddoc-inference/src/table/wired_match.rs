//! UNET 专用匹配、跨格文本拆分及同行归并；源规则来自 utils_table_recover.py。
use crate::{error, table_recover::Polygon, table_unicode_data::RANGES};
use rapiddoc_core::{Error, ErrorCode};
use serde::{Deserialize, Serialize};
pub type Rect = [f64; 4];
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Ocr {
    pub points: Polygon,
    pub text: String,
    pub confidence: f64,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Fragment {
    pub ocr_index: Option<usize>,
    pub points: Polygon,
    pub text: String,
    pub confidence: f64,
    pub split: bool,
}
#[derive(Serialize)]
pub struct Matched {
    pub cells: Vec<Vec<Fragment>>,
    pub unmatched: Vec<usize>,
    pub cell_lines: Vec<Vec<String>>,
}

fn flags(c: char) -> u8 {
    let code = c as u32;
    let i = RANGES.partition_point(|(_, end, _)| *end < code);
    RANGES
        .get(i)
        .filter(|(start, _, _)| *start <= code)
        .map(|(_, _, f)| *f)
        .unwrap_or(0)
}
fn whitespace(c: char) -> bool {
    flags(c) & 2 != 0
}
fn decimal(c: char) -> bool {
    flags(c) & 4 != 0
}
fn strip(s: &str) -> &str {
    s.trim_matches(whitespace)
}
fn weight(c: char) -> f64 {
    if ('\u{4e00}'..='\u{9fff}').contains(&c) {
        1.8
    } else if flags(c) & 1 != 0 {
        1.
    } else if whitespace(c) {
        0.35
    } else if ",.-+%/:;'\"".contains(c) {
        0.45
    } else {
        0.8
    }
}
// Python 3.12+ sum(float) 的补偿求和；np.cumsum 仍在下方按顺序累积。
fn sum_weights(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut hi = 0f64;
    let mut lo = 0f64;
    for value in values {
        let t = hi + value;
        lo += if hi.abs() >= value.abs() {
            (hi - t) + value
        } else {
            (value - t) + hi
        };
        hi = t;
    }
    hi + lo
}
fn rect(p: Polygon) -> Result<Rect, Error> {
    if p.iter()
        .flatten()
        .any(|v| !v.is_finite() || v.abs() > rapiddoc_core::resources::limit(1_000_000.))
    {
        return Err(error(ErrorCode::InvalidInput, "UNET 匹配框无效"));
    }
    // Original box_4_2_poly_to_box_4_1 selects p0/p2 verbatim. A valid
    // strongly tilted cell may have reversed diagonal y coordinates; Python
    // retains its signed area and zero-clamped intersection in the matcher.
    Ok([p[0][0], p[0][1], p[2][0], p[2][1]])
}
fn quad(b: Rect) -> Polygon {
    [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
}
fn consume(remaining: &mut usize, n: usize) -> Result<(), Error> {
    *remaining = remaining
        .checked_sub(n)
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "UNET OCR 匹配工作量超过预算"))?;
    Ok(())
}

/// 日期、千分位金额、数字百分比、ASCII 单词、汉字；顺序和贪婪规则匹配 Python 正则。
fn token_end(chars: &[char], start: usize) -> Option<usize> {
    let len = chars.len();
    let mut date = start;
    while date < len && date < start + 4 && decimal(chars[date]) {
        date += 1;
    }
    if date == start + 4 && date < len && matches!(chars[date], '-' | '/') {
        let month = date + 1;
        let mut end = month;
        while end < len && end < month + 2 && decimal(chars[end]) {
            end += 1;
        }
        if end > month && end < len && matches!(chars[end], '-' | '/') {
            let day = end + 1;
            end = day;
            while end < len && end < day + 2 && decimal(chars[end]) {
                end += 1;
            }
            if end > day {
                return Some(end);
            }
        }
    }
    let mut number = start;
    if number < len && matches!(chars[number], '-' | '+') {
        number += 1;
    }
    let first = number;
    while number < len && decimal(chars[number]) {
        number += 1;
    }
    if number > first {
        // 首组最多三个数字，后续每组恰好三个；可选小数恰好两位。
        if number - first <= 3 {
            let mut end = number;
            let mut groups = 0;
            while end + 3 < len
                && chars[end] == ','
                && chars[end + 1..end + 4].iter().copied().all(decimal)
            {
                end += 4;
                groups += 1;
            }
            if groups > 0 {
                if end + 2 < len
                    && chars[end] == '.'
                    && decimal(chars[end + 1])
                    && decimal(chars[end + 2])
                {
                    end += 3;
                }
                if end < len && chars[end] == '%' {
                    end += 1;
                }
                return Some(end);
            }
        }
        if number + 1 < len && chars[number] == '.' && decimal(chars[number + 1]) {
            number += 2;
            while number < len && decimal(chars[number]) {
                number += 1;
            }
        }
        if number < len && chars[number] == '%' {
            number += 1;
        }
        return Some(number);
    }
    if start < len && chars[start].is_ascii_alphabetic() {
        let mut end = start + 1;
        while end < len && chars[end].is_ascii_alphabetic() {
            end += 1;
        }
        return Some(end);
    }
    if start < len && ('\u{4e00}'..='\u{9fff}').contains(&chars[start]) {
        let mut end = start + 1;
        while end < len && ('\u{4e00}'..='\u{9fff}').contains(&chars[end]) {
            end += 1;
        }
        return Some(end);
    }
    None
}
fn tokens(text: &str, count: usize) -> Vec<String> {
    let chars: Vec<_> = strip(text).chars().collect();
    let mut i = 0;
    let mut result = Vec::new();
    while i < chars.len() {
        if let Some(end) = token_end(&chars, i) {
            result.push(chars[i..end].iter().collect());
            i = end;
        } else {
            if !whitespace(chars[i]) {
                return Vec::new();
            }
            i += 1;
        }
    }
    if result.len() == count {
        result
    } else {
        Vec::new()
    }
}

fn weighted_split(text: &str, ocr: Rect, cells: &[Rect]) -> Vec<String> {
    let chars: Vec<_> = strip(text).chars().collect();
    if chars.len() < cells.len() || ocr[2] <= ocr[0] {
        return Vec::new();
    }
    let weights: Vec<_> = chars.iter().map(|c| weight(*c)).collect();
    let total = sum_weights(weights.iter().copied());
    if total <= 0. {
        return Vec::new();
    }
    let mut cumulative = Vec::with_capacity(chars.len());
    let mut sum = 0.;
    for w in &weights {
        sum += w;
        cumulative.push(sum);
    }
    let width = ocr[2] - ocr[0];
    let mut cuts = Vec::new();
    let mut boundaries = vec![ocr[0]];
    let mut previous = 0;
    for pair in cells.windows(2) {
        let boundary = (pair[0][2] + pair[1][0]) / 2.;
        let target = total * ((boundary - ocr[0]) / width).clamp(0., 1.);
        let remaining = cells.len() - cuts.len() - 1;
        let cut = (cumulative.partition_point(|v| *v < target) + 1)
            .clamp(previous + 1, chars.len() - remaining);
        if cut <= previous {
            return Vec::new();
        }
        cuts.push(cut);
        boundaries.push(boundary.clamp(ocr[0], ocr[2]));
        previous = cut;
    }
    cuts.push(chars.len());
    boundaries.push(ocr[2]);
    let mut start = 0;
    let mut segments = Vec::new();
    for (i, end) in cuts.into_iter().enumerate() {
        let text: String = chars[start..end].iter().collect();
        let text = strip(&text).to_owned();
        if text.is_empty() {
            return Vec::new();
        }
        let segment_weight = sum_weights(text.chars().map(weight));
        if boundaries[i + 1] - boundaries[i] < width * segment_weight / total * 0.6 {
            return Vec::new();
        }
        segments.push(text);
        start = end;
    }
    segments
}

fn split(
    ocr: &Ocr,
    bbox: Rect,
    pred: &[Rect],
    indices: &[usize],
    source: usize,
) -> Vec<(usize, Fragment)> {
    if indices.len() < 2 {
        return Vec::new();
    }
    let mut indices = indices.to_vec();
    indices.sort_by(|&a, &b| pred[a][0].total_cmp(&pred[b][0]));
    let boxes: Vec<_> = indices.iter().map(|&i| pred[i]).collect();
    for pair in boxes.windows(2) {
        let [a, b] = [pair[0], pair[1]];
        let overlap = a[3].min(b[3]) - a[1].max(b[1]);
        let height = (a[3] - a[1]).min(b[3] - b[1]);
        let width = (a[2] - a[0]).min(b[2] - b[0]);
        if height <= 0. || overlap / height < 0.6 || (b[0] - a[2]).abs() > 2f64.max(width * 0.3) {
            return Vec::new();
        }
    }
    let mut segments = tokens(&ocr.text, indices.len());
    if segments.is_empty() {
        segments = weighted_split(&ocr.text, bbox, &boxes);
    }
    if segments.len() != indices.len() {
        return Vec::new();
    }
    indices
        .into_iter()
        .zip(boxes)
        .zip(segments)
        .map(|((i, b), text)| {
            let mut intersection = [
                bbox[0].max(b[0]),
                bbox[1].max(b[1]),
                bbox[2].min(b[2]),
                bbox[3].min(b[3]),
            ];
            if intersection[2] <= intersection[0] {
                intersection[0] = b[0];
                intersection[2] = b[2];
            }
            if intersection[3] <= intersection[1] {
                intersection[1] = bbox[1];
                intersection[3] = bbox[3];
            }
            (
                i,
                Fragment {
                    ocr_index: Some(source),
                    points: quad(intersection),
                    text,
                    confidence: ocr.confidence,
                    split: true,
                },
            )
        })
        .collect()
}

pub fn match_cells(polygons: &[Polygon], ocr: &[Ocr]) -> Result<Matched, Error> {
    if polygons.len() > rapiddoc_core::resources::limit(2000)
        || ocr.len() > rapiddoc_core::resources::limit(1000)
        || ocr.iter().map(|v| v.text.len()).sum::<usize>()
            > rapiddoc_core::resources::limit(1_000_000)
    {
        return Err(error(ErrorCode::ResourceLimit, "UNET OCR 匹配输入超过预算"));
    }
    let pred: Vec<_> = polygons
        .iter()
        .map(|p| rect(*p))
        .collect::<Result<_, _>>()?;
    for v in ocr {
        rect(v.points)?;
        if !v.confidence.is_finite() || !(0. ..=1.).contains(&v.confidence) {
            return Err(error(ErrorCode::InvalidInput, "UNET OCR 评分无效"));
        }
    }
    let mut cells = vec![Vec::new(); pred.len()];
    let mut unmatched = Vec::new();
    let mut work = 50_000_000;
    let mut fragments = 0;
    for (index, value) in ocr.iter().enumerate() {
        consume(&mut work, pred.len())?;
        let bbox = rect(value.points)?;
        let area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
        let height = bbox[3] - bbox[1];
        let mut candidates = Vec::new();
        let mut projection = Vec::new();
        let mut coverage = Vec::new();
        let mut ious = Vec::new();
        let chars: Vec<_> = strip(&value.text).chars().map(weight).collect();
        let total = sum_weights(chars.iter().copied());
        let min = chars.iter().copied().fold(f64::INFINITY, f64::min);
        let min_width = if total > 0. && bbox[2] > bbox[0] {
            ((bbox[2] - bbox[0]) * min / total * 0.7).max((bbox[2] - bbox[0]) * 0.04)
        } else {
            f64::INFINITY
        };
        for (i, b) in pred.iter().enumerate() {
            let intersects =
                !(bbox[2] < b[0] || bbox[0] > b[2] || bbox[3] < b[1] || bbox[1] > b[3]);
            let w = (bbox[2].min(b[2]) - bbox[0].max(b[0])).max(0.);
            let h = (bbox[3].min(b[3]) - bbox[1].max(b[1])).max(0.);
            let intersection = w * h;
            let union = area + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
            let iou = if !intersects {
                0.
            } else if union != 0. {
                intersection / union
            } else {
                1.
            };
            let outside = if area > 0. {
                (area - intersection) / area
            } else {
                0.
            };
            let cover = if area > 0. { intersection / area } else { 0. };
            if (intersects && outside < 0.6) || iou > 0.8 {
                candidates.push(i);
            }
            if intersects && height > 0. && h / height >= 0.6 && w > 0. && w >= min_width {
                projection.push(i);
            }
            coverage.push(cover);
            ious.push(iou);
        }
        let mut assigned = split(value, bbox, &pred, &projection, index);
        if assigned.is_empty() && !candidates.is_empty() {
            let mut rank = candidates.clone();
            rank.sort_by(|&a, &b| {
                coverage[b]
                    .total_cmp(&coverage[a])
                    .then_with(|| ious[b].total_cmp(&ious[a]))
            });
            let best = rank[0];
            let clear = if rank.len() == 1 {
                true
            } else {
                let second = rank[1];
                let cx = (bbox[0] + bbox[2]) / 2.;
                let cy = (bbox[1] + bbox[3]) / 2.;
                let hits: Vec<_> = candidates
                    .iter()
                    .copied()
                    .filter(|&i| {
                        pred[i][0] <= cx && cx < pred[i][2] && pred[i][1] <= cy && cy <= pred[i][3]
                    })
                    .collect();
                (hits == [best]
                    && coverage[best] >= 0.55
                    && coverage[best] - coverage[second] >= 0.15)
                    || (coverage[best] >= 0.65 && coverage[best] - coverage[second] >= 0.2)
            };
            if clear {
                assigned.push((
                    best,
                    Fragment {
                        ocr_index: Some(index),
                        points: value.points,
                        text: value.text.clone(),
                        confidence: value.confidence,
                        split: false,
                    },
                ));
            } else {
                assigned = split(value, bbox, &pred, &candidates, index);
            }
        }
        if assigned.is_empty() {
            unmatched.push(index);
        } else {
            fragments += assigned.len();
            if fragments > rapiddoc_core::resources::limit(10_000) {
                return Err(error(ErrorCode::ResourceLimit, "UNET OCR 拆分片段超过预算"));
            }
            for (cell, fragment) in assigned {
                cells[cell].push(fragment);
            }
        }
    }
    let mut cell_lines = Vec::with_capacity(cells.len());
    for cell in &cells {
        cell_lines.push(gather(cell, &mut work)?);
    }
    Ok(Matched {
        cells,
        unmatched,
        cell_lines,
    })
}

fn same_row(a: Rect, b: Rect, threshold: f64) -> bool {
    let ah = a[3] - a[1];
    let bh = b[3] - b[1];
    let overlap = a[3].min(b[3]) - a[1].max(b[1]);
    (if ah > 0. { (ah - overlap) / ah } else { 0. }) < threshold
        || (if bh > 0. { (bh - overlap) / bh } else { 0. }) < threshold
}
fn gather(fragments: &[Fragment], work: &mut usize) -> Result<Vec<String>, Error> {
    consume(
        work,
        fragments
            .len()
            .saturating_mul(fragments.len())
            .saturating_mul(2),
    )?;
    let mut lines: Vec<_> = fragments
        .iter()
        .map(|v| {
            (
                [
                    v.points[0][0],
                    v.points[0][1],
                    v.points[2][0],
                    v.points[2][1],
                ],
                v.text.clone(),
            )
        })
        .collect();
    lines.sort_by(|a, b| {
        a.0[1]
            .total_cmp(&b.0[1])
            .then_with(|| a.0[0].total_cmp(&b.0[0]))
    });
    for i in 0..lines.len().saturating_sub(1) {
        for j in (0..=i).rev() {
            let a = lines[j].0;
            let b = lines[j + 1].0;
            if same_row(a, b, 0.3) && b[0] < a[0] && (a[1] - b[1]).abs() < 20. {
                lines.swap(j, j + 1);
            } else {
                break;
            }
        }
    }
    let mut alive = vec![true; lines.len()];
    for i in 0..lines.len() {
        if !alive[i] {
            continue;
        }
        for j in i + 1..lines.len() {
            if !alive[j] {
                continue;
            }
            let a = lines[i].0;
            let b = lines[j].0;
            if same_row(a, b, 0.3) {
                let spaces = ((b[0] - a[2]).max(0.) / 10.) as usize;
                if spaces > rapiddoc_core::resources::limit(100_000) {
                    return Err(error(ErrorCode::ResourceLimit, "UNET 同行空格超过预算"));
                }
                let text = lines[j].1.clone();
                lines[i].1.push_str(&" ".repeat(spaces));
                lines[i].1.push_str(&text);
                if lines[i].1.len() > rapiddoc_core::resources::limit(1_000_000) {
                    return Err(error(ErrorCode::ResourceLimit, "UNET 合并文字超过预算"));
                }
                lines[i].0 = [
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                ];
                alive[j] = false;
            }
        }
    }
    Ok(lines
        .into_iter()
        .zip(alive)
        .filter_map(|((_, text), alive)| {
            if !alive || strip(&text).is_empty() {
                None
            } else {
                Some(strip(&text.replace(['\r', '\n'], "")).to_owned())
            }
        })
        .collect())
}

pub fn fill_blank(
    matched: &mut Matched,
    polygons: &[Polygon],
    results: &[crate::blank_cell::ResultCell],
) -> Result<(), Error> {
    if matched.cells.len() != polygons.len() {
        return Err(error(ErrorCode::InvalidInput, "UNET 补识别格框数量不匹配"));
    }
    if results.len() > rapiddoc_core::resources::limit(2000) {
        return Err(error(
            ErrorCode::ResourceLimit,
            "UNET 空格补识别结果超过预算",
        ));
    }
    if results.iter().map(|r| r.text.len()).sum::<usize>()
        > rapiddoc_core::resources::limit(1_000_000)
    {
        return Err(error(ErrorCode::ResourceLimit, "UNET 补识别文字超过预算"));
    }
    let mut seen = std::collections::HashSet::new();
    for result in results {
        let i = result.cell_index;
        if i >= matched.cells.len() || !seen.insert(i) || !matched.cells[i].is_empty() {
            return Err(error(
                ErrorCode::InvalidInput,
                "UNET 空格补识别索引重复/越界或覆盖已有文字",
            ));
        }
        if !result.confidence.is_finite()
            || !(0. ..=1.).contains(&result.confidence)
            || result.text.len() > rapiddoc_core::resources::limit(1_000_000)
        {
            return Err(error(ErrorCode::InvalidInput, "UNET 补识别文字/评分无效"));
        }
        if result.accepted {
            let p = polygons[i];
            matched.cells[i].push(Fragment {
                ocr_index: None,
                points: [p[0], p[3], p[2], p[1]],
                text: result.text.clone(),
                confidence: result.confidence,
                split: false,
            });
        }
    }
    let mut remaining = rapiddoc_core::resources::limit(50_000_000);
    matched.cell_lines = matched
        .cells
        .iter()
        .map(|cell| gather(cell, &mut remaining))
        .collect::<Result<_, _>>()?;
    Ok(())
}
