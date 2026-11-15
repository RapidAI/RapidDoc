//! The optional docvortex_ocr adapter: whole OCR words and observed strokes.
pub mod skew;
use rapiddoc_core::{pillow_resize::size, Error};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Word {
    pub bbox: [f64; 4],
    pub text: String,
    pub source_index: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rule {
    pub bbox: [f64; 4],
    pub width: f64,
    pub orientation: String,
}
#[derive(Serialize)]
pub struct Prepared {
    pub width: usize,
    pub height: usize,
    pub words: Vec<Word>,
    pub rules: Vec<Rule>,
    pub angle: f64,
}
pub fn gray(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|p| {
            ((p[0] as u32 * 9798 + p[1] as u32 * 19235 + p[2] as u32 * 3735 + 16384) >> 15) as u8
        })
        .collect()
}
fn ink(gray: &[u8], w: usize, h: usize) -> Vec<bool> {
    let block = 41.min((w.min(h) / 2) * 2 - 1);
    let radius = block / 2;
    let sigma = 0.3 * ((block - 1) as f64 * 0.5 - 1.) + 0.8;
    let mut kernel = (0..block)
        .map(|i| (-0.5 * ((i as f64 - radius as f64) / sigma).powi(2)).exp())
        .collect::<Vec<_>>();
    let sum = kernel.iter().sum::<f64>();
    for k in &mut kernel {
        *k /= sum;
    }
    let kernel = kernel.into_iter().map(|x| x as f32).collect::<Vec<_>>();
    let mut temp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0f32;
            for (i, &k) in kernel.iter().enumerate() {
                let xx =
                    (x as isize + i as isize - radius as isize).clamp(0, w as isize - 1) as usize;
                sum += gray[y * w + xx] as f32 * k;
            }
            temp[y * w + x] = sum;
        }
    }
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0f32;
            for (i, &k) in kernel.iter().enumerate() {
                let yy =
                    (y as isize + i as isize - radius as isize).clamp(0, h as isize - 1) as usize;
                sum += temp[yy * w + x] * k;
            }
            out[y * w + x] =
                gray[y * w + x] as i32 - sum.round_ties_even().clamp(0., 255.) as i32 <= -12;
        }
    }
    out
}
fn morphology(
    input: &[bool],
    w: usize,
    h: usize,
    len: usize,
    horizontal: bool,
    dilate: bool,
) -> Vec<bool> {
    let (outer, inner) = if horizontal { (h, w) } else { (w, h) };
    let anchor = len / 2;
    let mut out = vec![false; w * h];
    let index = |o: usize, i: usize| if horizontal { o * w + i } else { i * w + o };
    for o in 0..outer {
        let mut prefix = vec![0usize; inner + 1];
        for i in 0..inner {
            prefix[i + 1] = prefix[i] + usize::from(input[index(o, i)]);
        }
        for i in 0..inner {
            let lo = i.saturating_sub(anchor);
            let hi = (i + len - anchor).min(inner);
            let count = prefix[hi] - prefix[lo];
            out[index(o, i)] = if dilate { count > 0 } else { count == hi - lo };
        }
    }
    out
}
pub fn extract_rules(rgb: &[u8], w: usize, h: usize, words: &[Word]) -> Vec<Rule> {
    if w.min(h) < 16 {
        return vec![];
    }
    let binary = ink(&gray(rgb), w, h);
    let mut rules = vec![];
    for horizontal in [true, false] {
        let size = if horizontal { w } else { h };
        let gap = (size as f64 / 500.).round_ties_even().clamp(1., 3.) as usize;
        let minimum = (size as f64 * 0.025).round_ties_even().max(12.) as usize;
        let close = morphology(
            &morphology(&binary, w, h, gap + 1, horizontal, true),
            w,
            h,
            gap + 1,
            horizontal,
            false,
        );
        let mut mask = morphology(
            &morphology(&close, w, h, minimum, horizontal, false),
            w,
            h,
            minimum,
            horizontal,
            true,
        );
        for seed in 0..w * h {
            if !mask[seed] {
                continue;
            }
            mask[seed] = false;
            let mut pending = vec![seed];
            let (mut x0, mut y0, mut x1, mut y1, mut area) =
                (seed % w, seed / w, seed % w, seed / w, 0usize);
            while let Some(p) = pending.pop() {
                let (x, y) = (p % w, p / w);
                x0 = x0.min(x);
                x1 = x1.max(x);
                y0 = y0.min(y);
                y1 = y1.max(y);
                area += 1;
                for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                        let i = yy * w + xx;
                        if mask[i] {
                            mask[i] = false;
                            pending.push(i);
                        }
                    }
                }
            }
            let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
            let (length, thickness) = if horizontal { (bw, bh) } else { (bh, bw) };
            if length < minimum
                || length < 5 * thickness
                || thickness as f64 > 5f64.max(w.min(h) as f64 * 0.025)
                || area as f64 / ((bw * bh) as f64) < 0.15
            {
                continue;
            }
            let bbox = [x0 as f64, y0 as f64, (x1 + 1) as f64, (y1 + 1) as f64];
            if words.iter().any(|word| {
                let b = word.bbox;
                (bbox[2].min(b[2]) - bbox[0].max(b[0])).max(0.)
                    * (bbox[3].min(b[3]) - bbox[1].max(b[1])).max(0.)
                    >= 0.8 * (bw * bh) as f64
            }) {
                continue;
            }
            rules.push(Rule {
                bbox,
                width: thickness as f64,
                orientation: if horizontal { "horizontal" } else { "vertical" }.into(),
            });
        }
    }
    rules
}
pub fn prepare(png: &[u8], mut words: Vec<Word>) -> Result<Prepared, Error> {
    let (mut rgb, mut width, mut height) = crate::preprocess::decode_png(png, 16_000_000)?;
    for p in rgb.chunks_exact_mut(3) {
        p.swap(0, 2);
    }
    let angle = skew::angle(&gray(&rgb), width, height);
    if angle != 0. {
        let a = angle.to_radians();
        let (beta, alpha) = (a.sin(), a.cos());
        let (cx, cy) = (width as f64 / 2., height as f64 / 2.);
        let nw = (height as f64 * beta.abs() + width as f64 * alpha.abs()).ceil() as usize;
        let nh = (height as f64 * alpha.abs() + width as f64 * beta.abs()).ceil() as usize;
        size(nw, nh)?;
        let tx = (1. - alpha) * cx - beta * cy + (nw as f64 - width as f64) / 2.;
        let ty = beta * cx + (1. - alpha) * cy + (nh as f64 - height as f64) / 2.;
        for word in &mut words {
            let [x0, y0, x1, y1] = word.bbox;
            let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                .map(|[x, y]| [alpha * x + beta * y + tx, -beta * x + alpha * y + ty]);
            word.bbox = [
                corners.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
                corners.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
                corners
                    .iter()
                    .map(|p| p[0])
                    .fold(f64::NEG_INFINITY, f64::max),
                corners
                    .iter()
                    .map(|p| p[1])
                    .fold(f64::NEG_INFINITY, f64::max),
            ];
        }
        let determinant = alpha * alpha + beta * beta;
        let inverse = [
            alpha / determinant,
            -beta / determinant,
            (beta * ty - alpha * tx) / determinant,
            beta / determinant,
            alpha / determinant,
            (-beta * tx - alpha * ty) / determinant,
        ];
        rgb = crate::crop::affine_white(&rgb, width, height, inverse, nw, nh);
        width = nw;
        height = nh;
    }
    let rules = extract_rules(&rgb, width, height, &words);
    Ok(Prepared {
        width,
        height,
        words,
        rules,
        angle,
    })
}
/// 与 Python 三层证据的灰度、均值阈值和方向开运算一致。
pub fn evidence_rules(rgb: &[u8], width: usize, height: usize, em: f64) -> Vec<Rule> {
    if width == 0 || height == 0 {
        return vec![];
    }
    let factor = (1400. / width.max(height) as f64).min(1.);
    let (w, h) = (
        ((width as f64 * factor).round_ties_even() as usize).max(1),
        ((height as f64 * factor).round_ties_even() as usize).max(1),
    );
    let gray = gray(rgb);
    let gray = if factor < 1. {
        let bytes: Vec<_> = gray.iter().flat_map(|v| [*v; 3]).collect();
        super::unet::resize_area(&bytes, width, height, w, h)
            .chunks_exact(3)
            .map(|p| p[0])
            .collect()
    } else {
        gray
    };
    let window = ((2. * em * factor).round_ties_even() as usize | 1).max(15);
    let r = window / 2;
    // BORDER_REPLICATE 的可分离盒滤波，整数和避免浮点累计误差。
    let mut horizontal = vec![0u64; w * h];
    for y in 0..h {
        let mut prefix = vec![0u64; w + 1];
        for x in 0..w {
            prefix[x + 1] = prefix[x] + u64::from(gray[y * w + x]);
        }
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r + 1).min(w);
            horizontal[y * w + x] = prefix[hi] - prefix[lo]
                + r.saturating_sub(x) as u64 * u64::from(gray[y * w])
                + (x + r + 1).saturating_sub(w) as u64 * u64::from(gray[y * w + w - 1]);
        }
    }
    let mut binary = vec![false; w * h];
    for x in 0..w {
        let mut prefix = vec![0u64; h + 1];
        for y in 0..h {
            prefix[y + 1] = prefix[y] + horizontal[y * w + x];
        }
        for y in 0..h {
            let lo = y.saturating_sub(r);
            let hi = (y + r + 1).min(h);
            let sum = prefix[hi] - prefix[lo]
                + r.saturating_sub(y) as u64 * horizontal[x]
                + (y + r + 1).saturating_sub(h) as u64 * horizontal[(h - 1) * w + x];
            let mean = (sum as f64 / (window * window) as f64).round_ties_even() as i32;
            binary[y * w + x] = i32::from(gray[y * w + x]) <= mean - 8;
        }
    }
    let mut result = Vec::new();
    for axis in [0usize, 1] {
        let length = ((2.5 * em * factor).round_ties_even() as usize)
            .max(12)
            .max([h, w][axis] / 25);
        let horizontal = axis == 1;
        let mut mask = morphology(
            &morphology(&binary, w, h, length, horizontal, false),
            w,
            h,
            length,
            horizontal,
            true,
        );
        let mut components = Vec::new();
        for seed in 0..w * h {
            if !mask[seed] {
                continue;
            }
            mask[seed] = false;
            let mut pending = vec![seed];
            let (mut x0, mut y0, mut x1, mut y1) = (seed % w, seed / w, seed % w, seed / w);
            while let Some(p) = pending.pop() {
                let (x, y) = (p % w, p / w);
                x0 = x0.min(x);
                x1 = x1.max(x);
                y0 = y0.min(y);
                y1 = y1.max(y);
                for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                        let i = yy * w + xx;
                        if mask[i] {
                            mask[i] = false;
                            pending.push(i);
                        }
                    }
                }
            }
            let (bw, bh) = (x1 - x0 + 1, y1 - y0 + 1);
            let (long, thick) = if horizontal { (bw, bh) } else { (bh, bw) };
            if long < length || long < 6 * thick {
                continue;
            }
            components.push(Rule {
                bbox: [
                    x0 as f64 / factor,
                    y0 as f64 / factor,
                    (x1 + 1) as f64 / factor,
                    (y1 + 1) as f64 / factor,
                ],
                width: thick as f64 / factor,
                orientation: if horizontal { "horizontal" } else { "vertical" }.into(),
            });
        }
        // OpenCV 返回外轮廓的顺序是扫描顺序的逆序。
        components.reverse();
        result.extend(components);
    }
    result
}
