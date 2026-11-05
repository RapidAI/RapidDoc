//! 按 pdftext 0.7.1 pages.py 顺序移植，Apache-2.0；见 licenses/PDFTEXT.txt。
use super::{merge, Char, Font, PageChars};
use serde::Serialize;
mod unicode;

#[derive(Debug, Serialize)]
struct Span {
    bbox: [f64; 4],
    text: String,
    font: Font,
    rotation: f64,
    char_start_idx: usize,
    char_end_idx: usize,
    url: String,
    superscript: bool,
    subscript: bool,
    chars: Vec<Char>,
}
#[derive(Debug, Serialize)]
struct Line {
    spans: Vec<Span>,
    bbox: [f64; 4],
    rotation: f64,
}
#[derive(Debug, Serialize)]
struct Block {
    lines: Vec<Line>,
    bbox: [f64; 4],
    rotation: f64,
}
#[derive(Debug, Serialize)]
pub struct NativePage {
    size: [f64; 2],
    bbox: [f64; 4],
    width: u64,
    height: u64,
    rotation: i32,
    blocks: Vec<Block>,
    chars: Vec<Char>,
}
fn span(chars: &[Char], bbox: [f64; 4]) -> Span {
    let first = &chars[0];
    Span {
        bbox,
        text: chars.iter().map(|c| c.char.as_str()).collect(),
        font: first.font.clone(),
        rotation: first.rotation,
        char_start_idx: first.char_idx,
        char_end_idx: chars.last().unwrap().char_idx,
        url: String::new(),
        superscript: false,
        subscript: false,
        chars: chars.to_vec(),
    }
}
fn whitespace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}
fn text(span: &Span) -> &str {
    span.text.trim_matches(whitespace)
}
fn scripts(line: &mut Line) {
    let n = line.spans.len();
    let height = line.bbox[3] - line.bbox[1];
    if n < 2 || height > line.bbox[2] - line.bbox[0] {
        return;
    }
    let above: Vec<_> = line
        .spans
        .iter()
        .enumerate()
        .map(|(i, s)| (s.bbox[1] - (s.bbox[3] - s.bbox[1]) * 0.1, i))
        .collect();
    let below: Vec<_> = line
        .spans
        .iter()
        .enumerate()
        .map(|(i, s)| (s.bbox[3] + (s.bbox[3] - s.bbox[1]) * 0.1, i))
        .collect();
    // 两个极值等价于排除当前 span 后的 max/min，不随 span 数平方增长。
    let extremes = |values: &[(f64, usize)], maximum: bool| {
        let mut a = values[0];
        let mut b = values[1];
        let better = |x: f64, y: f64| if maximum { x > y } else { x < y };
        if better(b.0, a.0) {
            std::mem::swap(&mut a, &mut b);
        }
        for &v in &values[2..] {
            if better(v.0, a.0) {
                b = a;
                a = v;
            } else if better(v.0, b.0) {
                b = v;
            }
        }
        (a, b)
    };
    let top = extremes(&above, true);
    let bottom = extremes(&below, false);
    for i in 0..n {
        let s = &line.spans[i];
        let h = s.bbox[3] - s.bbox[1];
        let first = i == 0 || text(&line.spans[i - 1]).is_empty();
        let last = i + 1 == n || text(&line.spans[i + 1]).is_empty();
        let prev_h = if first {
            true
        } else {
            let b = line.spans[i - 1].bbox;
            h / (b[3] - b[1]).max(1.) <= 0.7
        };
        let next_h = if last {
            true
        } else {
            let b = line.spans[i + 1].bbox;
            h / (b[3] - b[1]).max(1.) <= 0.7
        };
        let value = text(s);
        let mut cs = value.chars();
        let one = cs.next().is_some() && cs.next().is_none();
        let digit = !value.is_empty() && value.chars().all(unicode::digit);
        let alnum = !value.is_empty() && value.chars().all(unicode::alnum);
        let okay = (one || digit) && (alnum || (one && value.chars().all(unicode::math)));
        if !(prev_h || next_h) || h / height.max(1.) > 0.7 || !okay {
            continue;
        }
        let high = if top.0 .1 == i { top.1 .0 } else { top.0 .0 };
        let low = if bottom.0 .1 == i {
            bottom.1 .0
        } else {
            bottom.0 .0
        };
        let prev_above = first || s.bbox[1] < line.spans[i - 1].bbox[1];
        let next_above = last || s.bbox[1] < line.spans[i + 1].bbox[1];
        let prev_below = first || s.bbox[3] > line.spans[i - 1].bbox[3];
        let next_below = last || s.bbox[3] > line.spans[i + 1].bbox[3];
        if (prev_above || next_above) && s.bbox[1] < high {
            line.spans[i].superscript = true;
        } else if (prev_below || next_below) && s.bbox[3] > low {
            line.spans[i].subscript = true;
        }
    }
}
fn overlap(a: [f64; 4], b: [f64; 4]) -> bool {
    (a[2] - a[0]) * (a[3] - a[1]) > 0.
        && (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.)
            > 0.
}
fn gap(a: [f64; 4], b: [f64; 4]) -> [f64; 2] {
    [
        (((a[0] + a[2]) / 2.) - ((b[0] + b[2]) / 2.)).abs(),
        (((a[1] + a[3]) / 2.) - ((b[1] + b[3]) / 2.)).abs(),
    ]
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    let m = if n == 0 {
        0.
    } else if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    };
    if m == 0. {
        0.1
    } else {
        m
    }
}
pub(super) fn group(page: PageChars) -> NativePage {
    let chars = &page.chars;
    let mut spans = Vec::new();
    if !chars.is_empty() {
        let mut start = 0;
        let mut bbox = chars[0].bbox;
        for j in 1..chars.len() {
            let c = &chars[j];
            let h = bbox[3] - bbox[1];
            if c.font != chars[start].font
                || c.rotation != chars[start].rotation
                || matches!(chars[j - 1].char.as_str(), "\u{2}" | "\n")
                || (c.bbox[1] < bbox[1] - h * 0.1
                    && c.bbox[3] < h * 0.7 + bbox[1]
                    && c.bbox[0] > bbox[2])
            {
                spans.push(span(&chars[start..j], bbox));
                start = j;
                bbox = c.bbox;
            } else {
                merge(&mut bbox, c.bbox);
            }
        }
        spans.push(span(&chars[start..], bbox));
    }
    let mut lines: Vec<Line> = Vec::new();
    for s in spans {
        let split = lines.last().is_none_or(|line| {
            let last = line.spans.last().unwrap();
            let d = (line.rotation - s.rotation).abs() % (2. * std::f64::consts::PI);
            let d = d.min(2. * std::f64::consts::PI - d);
            last.text.ends_with(['\n', '\u{2}'])
                || (std::f64::consts::FRAC_PI_4..=3. * std::f64::consts::FRAC_PI_4).contains(&d)
                || s.bbox[1] > line.bbox[3]
        });
        if split {
            lines.push(Line {
                bbox: s.bbox,
                rotation: s.rotation,
                spans: vec![s],
            });
        } else {
            let line = lines.last_mut().unwrap();
            merge(&mut line.bbox, s.bbox);
            line.spans.push(s);
        }
    }
    for line in &mut lines {
        scripts(line);
    }
    let gaps: Vec<_> = lines
        .windows(2)
        .map(|w| gap(w[0].bbox, w[1].bbox))
        .collect();
    let dx = median(gaps.iter().map(|g| g[0]).collect()) * 1.5;
    let dy = median(gaps.iter().map(|g| g[1]).collect()) * 1.5;
    let mut blocks: Vec<Block> = Vec::new();
    for line in lines {
        let join = blocks.last().is_some_and(|block| {
            let last = block.lines.last().unwrap();
            let g = gap(last.bbox, line.bbox);
            (g[0] <= dx && g[1] <= dy)
                || (block.lines.len() == 1 && last.bbox[0] > line.bbox[0] && g[1] <= dy)
                || (last.bbox[2] > line.bbox[2] && g[1] <= dy)
                || (g[1] < dy * 0.2 && last.bbox[2] > line.bbox[0])
                || overlap(block.bbox, line.bbox)
        });
        if join {
            let b = blocks.last_mut().unwrap();
            merge(&mut b.bbox, line.bbox);
            b.lines.push(line);
        } else {
            blocks.push(Block {
                bbox: line.bbox,
                rotation: line.rotation,
                lines: vec![line],
            });
        }
    }
    let mut merged: Vec<Block> = Vec::new();
    for b in blocks {
        if merged.last().is_some_and(|a| overlap(a.bbox, b.bbox)) {
            let a = merged.last_mut().unwrap();
            merge(&mut a.bbox, b.bbox);
            a.lines.extend(b.lines);
        } else {
            merged.push(b);
        }
    }
    NativePage {
        size: page.size,
        bbox: page.bbox,
        width: page.width,
        height: page.height,
        rotation: page.rotation,
        blocks: merged,
        chars: page.chars,
    }
}
