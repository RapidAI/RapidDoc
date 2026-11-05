//! 字符到基础文本行的连续原生管线；不携带 Python 对象或 PDFium 指针。
use std::collections::HashMap;

/// 保存当前 Python 解释器的非 ASCII 字符属性，纯计算阶段不回调 Python。
#[derive(Default)]
pub struct UnicodeProperties(pub HashMap<char, u8>);
impl UnicodeProperties {
    /// ASCII 使用固定语义，其他字符读取绑定层去重后的 Unicode 属性。
    fn flags(&self, ch: char) -> u8 {
        if ch.is_ascii() {
            (u8::from(ch.is_ascii_alphanumeric()))
                | (u8::from(ch.is_ascii_digit()) << 1)
                | (u8::from(ch.is_ascii_digit()) << 5)
                | (u8::from(matches!(ch, '+' | '<' | '=' | '>' | '|' | '~')) << 2)
                | (u8::from((' '..='~').contains(&ch)) << 4)
                | (u8::from(matches!(ch, '\t'..='\r' | '\u{1c}'..='\u{1f}' | ' ')) << 3)
        } else {
            self.0.get(&ch).copied().unwrap_or(0)
        }
    }
    /// 保持 Python str.strip 的空白定义。
    fn is_space(&self, ch: char) -> bool {
        self.flags(ch) & 8 != 0
    }
}

#[derive(Clone, Debug)]
pub struct TextChar {
    pub bbox: [f64; 4],
    pub text: String,
    pub rotation: f64,
    pub font_id: usize,
    pub font_size: Option<f64>,
    pub signature_id: Option<usize>,
    pub font_weight: Option<f64>,
}

#[derive(Debug)]
pub struct TextSpan {
    pub bbox: [f64; 4],
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub font_id: usize,
    pub rotation: f64,
    pub superscript: bool,
    pub subscript: bool,
}

#[derive(Debug)]
pub struct TextLine {
    pub bbox: [f64; 4],
    pub rotation: f64,
    pub spans: Vec<TextSpan>,
}

/// 按 Python 累加矩形的比较顺序合并，保留非有限输入的既有行为。
fn merge(a: &mut [f64; 4], b: [f64; 4]) {
    if b[0] < a[0] {
        a[0] = b[0];
    }
    if b[1] < a[1] {
        a[1] = b[1];
    }
    if b[2] > a[2] {
        a[2] = b[2];
    }
    if b[3] > a[3] {
        a[3] = b[3];
    }
}

/// 一次遍历生成字体片段，再连续执行组行与上下标标记。
pub fn group_text_lines(
    chars: &[TextChar],
    height_threshold: f64,
    distance: f64,
    unicode: &UnicodeProperties,
) -> Vec<TextLine> {
    let mut spans: Vec<TextSpan> = Vec::new();
    for (index, ch) in chars.iter().enumerate() {
        let split = match spans.last() {
            None => true,
            Some(span) => {
                let height = span.bbox[3] - span.bbox[1];
                ch.font_id != span.font_id
                    || ch.rotation != span.rotation
                    || matches!(chars[index - 1].text.as_str(), "\x02" | "\n")
                    || (ch.bbox[1] < span.bbox[1] - height * distance
                        && ch.bbox[3] < height * height_threshold + span.bbox[1]
                        && ch.bbox[0] > span.bbox[2])
            }
        };
        if split {
            spans.push(TextSpan {
                bbox: ch.bbox,
                text: ch.text.clone(),
                start: index,
                end: index + 1,
                font_id: ch.font_id,
                rotation: ch.rotation,
                superscript: false,
                subscript: false,
            });
        } else {
            let span = spans.last_mut().unwrap();
            merge(&mut span.bbox, ch.bbox);
            span.text.push_str(&ch.text);
            span.end = index + 1;
        }
    }
    let mut lines: Vec<TextLine> = Vec::new();
    for span in spans {
        let split = match lines.last() {
            None => true,
            Some(line) => {
                let text = &line.spans.last().unwrap().text;
                let diff = (span.rotation - line.rotation).abs() % std::f64::consts::TAU;
                let diff = diff.min(std::f64::consts::TAU - diff);
                text.ends_with('\n')
                    || text.ends_with('\x02')
                    || (span.rotation != line.rotation
                        && (std::f64::consts::FRAC_PI_4..=3.0 * std::f64::consts::FRAC_PI_4)
                            .contains(&diff))
                    || span.bbox[1] > line.bbox[3]
            }
        };
        if split {
            lines.push(TextLine {
                bbox: span.bbox,
                rotation: span.rotation,
                spans: vec![span],
            });
        } else {
            let line = lines.last_mut().unwrap();
            merge(&mut line.bbox, span.bbox);
            line.spans.push(span);
        }
    }
    for line in &mut lines {
        assign_scripts(line, height_threshold, distance, unicode);
    }
    lines
}

/// 在线性时间取得两个极值，保持同值时的首索引规则。
fn extreme_two(values: impl Iterator<Item = f64>, maximum: bool) -> (f64, usize, f64) {
    let initial = if maximum {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let (mut first, mut at, mut second) = (initial, usize::MAX, initial);
    for (i, v) in values.enumerate() {
        let better = if maximum { v > first } else { v < first };
        if better {
            second = first;
            first = v;
            at = i;
        } else if if maximum { v > second } else { v < second } {
            second = v;
        }
    }
    (first, at, second)
}

/// 按基础片段几何标记上下标，Unicode 判定使用与参考后端一致的类别表。
fn assign_scripts(line: &mut TextLine, threshold: f64, distance: f64, unicode: &UnicodeProperties) {
    let n = line.spans.len();
    let line_height = line.bbox[3] - line.bbox[1];
    if n < 2 || line_height > line.bbox[2] - line.bbox[0] {
        return;
    }
    let heights: Vec<_> = line.spans.iter().map(|s| s.bbox[3] - s.bbox[1]).collect();
    let above = extreme_two(
        line.spans
            .iter()
            .zip(&heights)
            .map(|(s, h)| s.bbox[1] - h * distance),
        true,
    );
    let below = extreme_two(
        line.spans
            .iter()
            .zip(&heights)
            .map(|(s, h)| s.bbox[3] + h * distance),
        false,
    );
    for i in 0..n {
        let spans = &line.spans;
        let first = i == 0
            || spans[i - 1]
                .text
                .trim_matches(|ch| unicode.is_space(ch))
                .is_empty();
        let last = i == n - 1
            || spans[i + 1]
                .text
                .trim_matches(|ch| unicode.is_space(ch))
                .is_empty();
        let span = &spans[i];
        let height = heights[i];
        let full = height / line_height.max(1.0) <= threshold;
        let neighbor_full = (first || height / heights[i - 1].max(1.0) <= threshold)
            || (last || height / heights[i + 1].max(1.0) <= threshold);
        let top = span.bbox[1];
        let bottom = span.bbox[3];
        let text = span.text.trim_matches(|ch| unicode.is_space(ch));
        let count = text.chars().count();
        let okay = (count == 1
            || (!text.is_empty() && text.chars().all(|ch| unicode.flags(ch) & 2 != 0)))
            && ((!text.is_empty() && text.chars().all(|ch| unicode.flags(ch) & 1 != 0))
                || (count == 1 && text.chars().all(|ch| unicode.flags(ch) & 4 != 0)));
        let super_flag = neighbor_full
            && full
            && okay
            && top < if i == above.1 { above.2 } else { above.0 }
            && ((first || top < spans[i - 1].bbox[1]) || (last || top < spans[i + 1].bbox[1]));
        let sub_flag = !super_flag
            && neighbor_full
            && full
            && okay
            && bottom > if i == below.1 { below.2 } else { below.0 }
            && ((first || bottom > spans[i - 1].bbox[3])
                || (last || bottom > spans[i + 1].bbox[3]));
        line.spans[i].superscript = super_flag;
        line.spans[i].subscript = sub_flag;
    }
}

#[derive(Debug)]
pub struct VisualTextRun {
    pub text: String,
    pub bbox: [f64; 4],
    pub angle: i32,
    pub indices: Vec<usize>,
    pub row_id: usize,
    pub run_index: usize,
    pub split: bool,
    pub formula: bool,
    pub coarse_fallback: bool,
    pub paragraph_terminal: bool,
    pub typography: Option<crate::statistics::Typography>,
    pub typographic_scale: Option<f64>,
}

/// 按 Python 规则缓存正数字体尺寸中位数，缺失时留给调用方使用行高回退。
fn typographic_scale(indices: &[usize], chars: &[TextChar]) -> Option<f64> {
    let sizes: Vec<_> = indices
        .iter()
        .filter_map(|&index| chars[index].font_size)
        .filter(|size| size.is_finite() && *size > 0.0)
        .collect();
    (!sizes.is_empty()).then(|| 0.1_f64.max(crate::median(sizes)))
}

/// 返回 Python 字符串语义下的可打印、全空白标志。
fn glyph_flags(text: &str, unicode: &UnicodeProperties) -> u8 {
    let printable = text.chars().all(|ch| unicode.flags(ch) & 16 != 0);
    let space = !text.is_empty() && text.chars().all(|ch| unicode.is_space(ch));
    u8::from(printable && !space) | (u8::from(space) << 1)
}

/// 将弧度换成非负圆周角，匹配 Python 对非法方向的零度处理。
fn angle_degrees(value: f64) -> f64 {
    if value.is_finite() {
        value.to_degrees().rem_euclid(360.0)
    } else {
        0.0
    }
}

/// 计算两个角度的最短圆周距离。
fn angle_distance(a: f64, b: f64) -> f64 {
    ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs()
}

/// 用字符中心及中位字高识别仿斜体矩阵，不把真实斜排误判成正文。
fn horizontal_baseline(
    spans: &[TextSpan],
    chars: &[TextChar],
    unicode: &UnicodeProperties,
) -> bool {
    use crate::geometry;
    let mut boxes: Vec<_> = spans
        .iter()
        .flat_map(|s| s.start..s.end)
        .filter(|&i| glyph_flags(&chars[i].text, unicode) & 1 != 0)
        .filter_map(|i| geometry::normalize(Some(chars[i].bbox), false))
        .collect();
    if boxes.len() < 4 {
        return false;
    }
    let bbox = geometry::union(&boxes).unwrap();
    let width = bbox[2] - bbox[0];
    let height = bbox[3] - bbox[1];
    let median = crate::median(boxes.iter().map(|b| b[3] - b[1]).collect());
    if height <= 0.0 || median <= 0.0 || width / height < 3.0 {
        return false;
    }
    boxes.sort_by(|a, b| {
        ((a[0] + a[2]) / 2.0)
            .total_cmp(&((b[0] + b[2]) / 2.0))
            .then_with(|| ((a[1] + a[3]) / 2.0).total_cmp(&((b[1] + b[3]) / 2.0)))
    });
    let first = boxes.first().unwrap();
    let last = boxes.last().unwrap();
    let baseline_width = (last[0] + last[2]) / 2.0 - (first[0] + first[2]) / 2.0;
    if baseline_width <= 0.0 {
        return false;
    }
    let dy = (last[1] + last[3]) / 2.0 - (first[1] + first[3]) / 2.0;
    let angle = dy.atan2(baseline_width).to_degrees().abs();
    let minimum = boxes
        .iter()
        .map(|b| (b[1] + b[3]) / 2.0)
        .fold(f64::INFINITY, f64::min);
    let maximum = boxes
        .iter()
        .map(|b| (b[1] + b[3]) / 2.0)
        .fold(f64::NEG_INFINITY, f64::max);
    angle <= 2.0 && maximum - minimum <= 0.75 * median
}

/// 负字号必须同时得到源字符正向基线证实才纠正半周翻转，单字不作推断。
fn correct_negative_scale_angle(
    spans: &[TextSpan],
    visual: f64,
    chars: &[TextChar],
    unicode: &UnicodeProperties,
) -> f64 {
    let candidate = (visual + 180.0).rem_euclid(360.0);
    let cardinal = [0.0, 90.0, 180.0, 270.0]
        .into_iter()
        .min_by(|a, b| angle_distance(candidate, *a).total_cmp(&angle_distance(candidate, *b)))
        .unwrap();
    if angle_distance(candidate, cardinal) > 0.1 {
        return visual;
    }
    let (axis_x, axis_y) = match cardinal as i32 {
        0 => (1.0, 0.0),
        90 => (0.0, 1.0),
        180 => (-1.0, 0.0),
        _ => (0.0, -1.0),
    };
    let mut positions = Vec::new();
    let mut heights = Vec::new();
    // 字符、字体签名和字号共同确定描粗副层的去重身份。
    type GlyphIdentity<'a> = (&'a str, Option<usize>, u64);
    let mut seen: HashMap<GlyphIdentity<'_>, Vec<[f64; 4]>> = HashMap::new();
    for index in spans.iter().flat_map(|span| span.start..span.end) {
        let ch = &chars[index];
        if ch.text.is_empty() || !ch.text.chars().all(|c| unicode.flags(c) & 1 != 0) {
            continue;
        }
        let Some(bbox) = crate::geometry::normalize(Some(ch.bbox), false) else {
            continue;
        };
        if bbox[0] >= bbox[2] || bbox[1] >= bbox[3] {
            continue;
        }
        if !ch
            .font_size
            .is_some_and(|size| size.is_finite() && size < 0.0)
        {
            return visual;
        }
        let area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
        let previous = seen
            .entry((
                ch.text.as_str(),
                ch.signature_id,
                ch.font_size.unwrap().to_bits(),
            ))
            .or_default();
        if previous.iter().any(|old| {
            let intersection = (bbox[2].min(old[2]) - bbox[0].max(old[0])).max(0.0)
                * (bbox[3].min(old[3]) - bbox[1].max(old[1])).max(0.0);
            let union = area + (old[2] - old[0]) * (old[3] - old[1]) - intersection;
            intersection >= 0.5 * union
        }) {
            continue;
        }
        // 描粗副层只从方向证据去重，不删除或重排原始字符成员。
        previous.push(bbox);
        let (x, y) = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0);
        positions.push((x * axis_x + y * axis_y, -x * axis_y + y * axis_x));
        heights.push(if axis_x != 0.0 {
            bbox[3] - bbox[1]
        } else {
            bbox[2] - bbox[0]
        });
    }
    if positions.len() < 2 {
        return visual;
    }
    let median_height = crate::median(heights);
    let first = positions.first().unwrap();
    let last = positions.last().unwrap();
    let (forward, sideways) = (last.0 - first.0, last.1 - first.1);
    let minimum = positions.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let maximum = positions
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);
    if forward < 0.5 * median_height
        || sideways.atan2(forward).to_degrees().abs() > 2.0
        || maximum - minimum > 0.75 * median_height
        || positions
            .windows(2)
            .any(|pair| pair[1].0 < pair[0].0 - 0.1 * median_height)
    {
        return visual;
    }
    cardinal
}

/// 解析常规方向及公式专用小角度候选，在白名单前恢复已证实的负尺度方向。
fn visual_angle(
    spans: &[TextSpan],
    rotation: f64,
    chars: &[TextChar],
    unicode: &UnicodeProperties,
    page_rotation: i32,
    supported: &[f64],
) -> Option<(i32, bool)> {
    let visual = (angle_degrees(rotation) + page_rotation as f64).rem_euclid(360.0);
    let visual = correct_negative_scale_angle(spans, visual, chars, unicode);
    for &angle in supported {
        if angle_distance(visual, angle) <= 0.1 {
            return Some((angle as i32, false));
        }
    }
    if supported
        .iter()
        .any(|&angle| angle_distance(0.0, angle) <= 0.1)
        && angle_distance(visual, 0.0) <= 30.0
        && horizontal_baseline(spans, chars, unicode)
    {
        return Some((0, false));
    }
    let nearest = supported
        .iter()
        .copied()
        .min_by(|a, b| angle_distance(visual, *a).total_cmp(&angle_distance(visual, *b)))?;
    if angle_distance(visual, nearest) > 30.0 {
        return None;
    }
    let compact: String = spans
        .iter()
        .flat_map(|s| s.text.chars())
        .filter(|&ch| unicode.flags(ch) & 16 != 0 && !unicode.is_space(ch))
        .collect();
    if compact.is_empty() {
        return None;
    }
    let script = spans.iter().any(|s| s.superscript || s.subscript);
    let math = compact.chars().any(|ch| "=∑∫√±×÷".contains(ch));
    let sizes: Vec<_> = spans
        .iter()
        .filter_map(|s| chars[s.start].font_size)
        .filter(|v| *v > 0.0)
        .collect();
    let mixed = !sizes.is_empty()
        && sizes.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            >= 1.35 * sizes.iter().copied().fold(f64::INFINITY, f64::min);
    let identifier =
        compact.chars().count() <= 3 && compact.chars().all(|ch| unicode.flags(ch) & 1 != 0);
    if script || math || mixed || identifier {
        Some((nearest as i32, true))
    } else {
        None
    }
}

/// 将连续基础组行、旋转拆分、方向筛选及视觉间隙分段融合，避免中间 Python 片段对象。
pub fn prepare_visual_lines(
    chars: &[TextChar],
    size: [f64; 2],
    page_rotation: i32,
    supported: &[f64],
    unicode: &UnicodeProperties,
    families: &[Option<usize>],
) -> Vec<VisualTextRun> {
    use crate::geometry;
    let lines = group_text_lines(chars, 0.7, 0.1, unicode);
    let mut output = Vec::new();
    let mut row_id = 0;
    for line in lines {
        let mut start = 0;
        let mut angle = angle_degrees(line.rotation);
        let mut chunks = Vec::new();
        for (i, span) in line.spans.iter().enumerate() {
            let span_angle = angle_degrees(span.rotation);
            let visible = span
                .text
                .chars()
                .any(|ch| unicode.flags(ch) & 16 != 0 && !unicode.is_space(ch));
            if i > start && visible && angle_distance(span_angle, angle) >= 44.9 {
                chunks.push((start, i, angle));
                start = i;
                angle = span_angle;
            }
        }
        chunks.push((start, line.spans.len(), angle));
        for (start, end, angle) in chunks {
            let spans = &line.spans[start..end];
            let Some((visual_angle, formula)) = visual_angle(
                spans,
                angle.to_radians(),
                chars,
                unicode,
                page_rotation,
                supported,
            ) else {
                continue;
            };
            let this_row = row_id;
            row_id += 1;
            let span_boxes: Vec<_> = spans
                .iter()
                .filter_map(|s| geometry::normalize(Some(s.bbox), false))
                .collect();
            let Some(bbox) = geometry::clip(geometry::union(&span_boxes).or(Some(line.bbox)), size)
            else {
                continue;
            };
            let member_capacity = spans.iter().map(|s| s.end - s.start).sum();
            let mut indices = Vec::with_capacity(member_capacity);
            indices.extend(
                spans
                    .iter()
                    .flat_map(|s| s.start..s.end)
                    .filter(|&i| !matches!(chars[i].text.as_str(), "\r" | "\n")),
            );
            let mut raw = Vec::with_capacity(indices.len());
            raw.extend(indices.iter().map(|&i| Some(chars[i].bbox)));
            let mut flags = Vec::with_capacity(indices.len());
            flags.extend(
                indices
                    .iter()
                    .map(|&i| glyph_flags(&chars[i].text, unicode)),
            );
            let ranges =
                geometry::visual_runs(raw, vec![None; indices.len()], flags, size, visual_angle);
            if ranges.is_empty() {
                output.push(VisualTextRun {
                    text: spans.iter().map(|s| s.text.as_str()).collect(),
                    bbox,
                    angle: visual_angle,
                    indices: spans.iter().flat_map(|s| s.start..s.end).collect(),
                    row_id: this_row,
                    run_index: 0,
                    split: false,
                    formula,
                    coarse_fallback: true,
                    paragraph_terminal: false,
                    typography: None,
                    typographic_scale: None,
                });
                continue;
            }
            let split = ranges.len() > 1;
            for (run_index, (a, b, run_box)) in ranges.into_iter().enumerate() {
                let Some(run_box) = run_box else {
                    continue;
                };
                let mut selected = Vec::with_capacity(b - a);
                selected.extend_from_slice(&indices[a..b]);
                output.push(VisualTextRun {
                    text: selected.iter().map(|&i| chars[i].text.as_str()).collect(),
                    bbox: run_box,
                    angle: visual_angle,
                    indices: selected,
                    row_id: this_row,
                    run_index,
                    split,
                    formula,
                    coarse_fallback: false,
                    paragraph_terminal: false,
                    typography: None,
                    typographic_scale: None,
                });
            }
        }
    }
    for run in &mut output {
        run.text = normalize_run_text(&run.text, unicode);
        if run.coarse_fallback || run.text.is_empty() {
            continue;
        }
        run.paragraph_terminal = sentence_terminal(run, chars, unicode);
        let mut boxes = Vec::with_capacity(run.indices.len());
        boxes.extend(run.indices.iter().map(|&i| {
            if glyph_flags(&chars[i].text, unicode) & 1 == 0 {
                return None;
            }
            geometry::clip(geometry::normalize(Some(chars[i].bbox), false), size)
                .map(|b| geometry::rotate(b, size, run.angle))
        }));
        let mut font_ids = Vec::with_capacity(run.indices.len());
        font_ids.extend(run.indices.iter().map(|&i| chars[i].signature_id));
        let mut weights = Vec::with_capacity(run.indices.len());
        weights.extend(run.indices.iter().map(|&i| chars[i].font_weight));
        let local = geometry::rotate(run.bbox, size, run.angle);
        run.typography = crate::statistics::typography(
            boxes,
            font_ids,
            weights,
            families,
            0.1_f64.max(local[3] - local[1]),
        );
        run.typographic_scale = typographic_scale(&run.indices, chars);
    }
    output.retain(|run| !run.text.is_empty());
    output
}

/// 按 Python 原规则顺序清洗 run 文本，不改变字符成员或视觉分段编号。
fn normalize_run_text(text: &str, unicode: &UnicodeProperties) -> String {
    let mut translated = Vec::with_capacity(text.len());
    let mut source = text.chars().peekable();
    while let Some(ch) = source.next() {
        let ch = match ch {
            '\r' => {
                if source.peek() == Some(&'\n') {
                    source.next();
                }
                '\n'
            }
            '\u{85}' | '\u{2028}' | '\u{2029}' => '\n',
            '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}' => ' ',
            '\u{200b}' | '\u{2060}' | '\u{feff}' => continue,
            _ => ch,
        };
        translated.push(ch);
    }
    let mut normalized = String::with_capacity(text.len());
    let mut previous_space = false;
    for (index, &original) in translated.iter().enumerate() {
        let mut ch = original;
        // 软断词的前后断言必须在删除控制字符、换行和软连字符之前计算。
        if matches!(ch, '\u{2}' | '\u{ad}')
            && index > 0
            && translated[index - 1].is_ascii_alphabetic()
        {
            let mut after = index + 1;
            while after < translated.len() && matches!(translated[after], '\t' | ' ') {
                after += 1;
            }
            if after == translated.len() || translated[after] == '\n' {
                ch = '-';
            }
        }
        if matches!(ch, '\u{ad}' | '\n' | '\u{0}'..='\u{8}' | '\u{b}' | '\u{c}' | '\u{e}'..='\u{1f}' | '\u{7f}'..='\u{9f}')
        {
            continue;
        }
        if ch == '\t' {
            ch = ' ';
        }
        if ch == ' ' {
            if previous_space {
                continue;
            }
            previous_space = true;
        } else {
            previous_space = false;
        }
        normalized.push(ch);
    }
    normalized
        .trim_matches(|ch| unicode.is_space(ch))
        .to_owned()
}

/// 基于规范化文本和原字符记录判断句尾，保留引用长度按记录切片的既有语义。
fn sentence_terminal(run: &VisualTextRun, chars: &[TextChar], unicode: &UnicodeProperties) -> bool {
    let text = run.text.trim_end_matches(|ch| unicode.is_space(ch));
    let mut ending = text.chars().rev();
    let last_body = ending.find(|&ch| {
        !matches!(
            ch,
            ']' | ')' | '}' | '）' | '】' | '》' | '”' | '’' | '\'' | '"'
        )
    });
    if last_body.is_some_and(|ch| matches!(ch, '.' | '!' | '?' | '。' | '！' | '？')) {
        return true;
    }
    let mut references = 0;
    let mut first_is_decimal = false;
    let mut punctuation = false;
    for ch in text.chars().rev() {
        let decimal = unicode.flags(ch) & 32 != 0;
        if decimal || matches!(ch, ',' | '–' | '—' | '-') {
            references += 1;
            first_is_decimal = decimal;
        } else {
            punctuation = matches!(ch, '.' | '!' | '?' | '。' | '！' | '？');
            break;
        }
    }
    if references == 0 || !first_is_decimal || !punctuation {
        return false;
    }
    let axis = if matches!(run.angle, 90 | 270) { 0 } else { 1 };
    let sizes: Vec<_> = run
        .indices
        .iter()
        .filter_map(|&index| {
            let ch = &chars[index];
            if ch
                .text
                .trim_matches(|value| unicode.is_space(value))
                .is_empty()
            {
                return None;
            }
            Some(ch.bbox[axis + 2] - ch.bbox[axis])
        })
        .collect();
    if sizes.len() <= references {
        return false;
    }
    let cut = sizes.len() - references;
    let body: Vec<_> = sizes[..cut]
        .iter()
        .copied()
        .filter(|size| *size > 0.0)
        .collect();
    if body.is_empty() {
        return false;
    }
    let limit = 0.8 * crate::median(body);
    sizes[cut..]
        .iter()
        .all(|size| *size > 0.0 && *size <= limit)
}
