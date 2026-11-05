//! DB 概率图后处理。候选、轮廓和评分工作量均有显式预算。

use image::{GrayImage, Luma};
use imageproc::{contours::find_contours, point::Point};
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;

pub struct Config {
    pub threshold: f32,
    pub box_threshold: f64,
    pub unclip_ratio: f64,
    pub dilation: bool,
    pub max_candidates: usize,
    pub max_pixels: usize,
    pub max_work: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            box_threshold: 0.5,
            unclip_ratio: 1.6,
            dilation: true,
            max_candidates: 1000,
            max_pixels: 4_000_000,
            max_work: 50_000_000,
        }
    }
}

impl Config {
    /// AtomModelSingleton's body OCR defaults; table OCR explicitly uses .5/1.6.
    pub fn body() -> Self {
        Self {
            box_threshold: 0.3,
            unclip_ratio: 1.8,
            ..Self::default()
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Detection {
    /// 左上、右上、右下、左下；原图像素坐标。
    pub points: [[f32; 2]; 4],
    pub score: f64,
}

/// 与当前 RapidOCR 相同：稳定 y 排序，相邻 y 差小于 10 的连通组按 x 排序。
pub fn sort_reading_order(boxes: &mut [Detection]) {
    boxes.sort_by(|a, b| a.points[0][1].total_cmp(&b.points[0][1]));
    let mut start = 0;
    for index in 1..=boxes.len() {
        if index == boxes.len() || boxes[index].points[0][1] - boxes[index - 1].points[0][1] >= 10.
        {
            boxes[start..index].sort_by(|a, b| a.points[0][0].total_cmp(&b.points[0][0]));
            start = index;
        }
    }
}

fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}

fn consume(work: &mut usize, count: usize) -> Result<(), Error> {
    *work = work
        .checked_sub(count)
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "DB 后处理工作量超过预算"))?;
    Ok(())
}

/// 单图单通道概率图；无效概率必须报错，不能返回空检测。
pub fn boxes(
    values: &[f32],
    width: usize,
    height: usize,
    source_width: usize,
    source_height: usize,
    config: &Config,
) -> Result<Vec<Detection>, Error> {
    let count = width
        .checked_mul(height)
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "概率图尺寸溢出"))?;
    if count > rapiddoc_core::resources::limit(config.max_pixels)
        || width > rapiddoc_core::resources::limit(4096)
        || height > rapiddoc_core::resources::limit(4096)
    {
        return Err(error(ErrorCode::ResourceLimit, "概率图超过预算"));
    }
    if width == 0
        || height == 0
        || count != values.len()
        || source_width == 0
        || source_height == 0
        || source_width > rapiddoc_core::resources::limit(1_000_000)
        || source_height > rapiddoc_core::resources::limit(1_000_000)
        || values
            .iter()
            .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
        || !config.threshold.is_finite()
        || !(0. ..=1.).contains(&config.threshold)
        || !config.box_threshold.is_finite()
        || !(0. ..=1.).contains(&config.box_threshold)
        || !config.unclip_ratio.is_finite()
        || !(0. ..=10.).contains(&config.unclip_ratio)
        || config.max_candidates == 0
        || config.max_candidates > rapiddoc_core::resources::limit(10000)
    {
        return Err(error(ErrorCode::InvalidInput, "DB 概率图或配置无效"));
    }
    // 一圈零边框让触边前景也具有封闭轮廓，避免轮廓库边界前提造成漏检。
    let mut bitmap = GrayImage::new(width as u32 + 2, height as u32 + 2);
    for y in 0..height {
        for x in 0..width {
            let foreground = if config.dilation {
                // OpenCV 2x2 默认 anchor=(1,1)，向右下扩展一像素。
                (y.saturating_sub(1)..=y).any(|yy| {
                    (x.saturating_sub(1)..=x).any(|xx| values[yy * width + xx] > config.threshold)
                })
            } else {
                values[y * width + x] > config.threshold
            };
            bitmap.put_pixel(x as u32 + 1, y as u32 + 1, Luma([u8::from(foreground)]));
        }
    }
    let contours = find_contours::<i32>(&bitmap);
    let mut work = rapiddoc_core::resources::limit(config.max_work);
    let mut result = Vec::new();
    let trace = std::env::var_os("RAPIDDOC_DB_GEOMETRY_TRACE").is_some();
    // RETR_LIST 以逆扫描顺序返回，包括孔洞边界。
    for contour in contours.into_iter().rev().take(config.max_candidates) {
        let points: Vec<_> = contour
            .points
            .into_iter()
            .map(|p| Point::new(p.x - 1, p.y - 1))
            .collect();
        let Some((rect, short)) = minimum_rect(&points, &mut work)? else {
            continue;
        };
        if short < 3. {
            continue;
        }
        let score = score(values, width, height, &rect, &mut work)?;
        if score < config.box_threshold {
            continue;
        }
        // Rounded boxPoints vertices need not form an exact rectangle.
        // Preserve Polygon.area/length instead of width*height shortcuts.
        let mut area = 0.;
        for i in 1..4 {
            area += (rect[i][0] - rect[0][0]) * (rect[i - 1][1] - rect[(i + 1) % 4][1]);
        }
        let perimeter = (0..4)
            .map(|i| distance(rect[i], rect[(i + 1) % 4]))
            .sum::<f64>();
        let delta = area.abs() * 0.5 * config.unclip_ratio / perimeter;
        if trace {
            eprintln!(
                "DB rectangle={rect:?} area={area:?} perimeter={perimeter:?} delta={delta:?}"
            );
        }
        // Pyclipper 将浮点输入截断到整数，保持相同输入合同。
        let points = crate::detection_offset::expand(&rect, delta)?;
        if trace {
            eprintln!("DB expanded={points:?}");
        }
        let Some((rect, short)) = minimum_rect(&points, &mut work)? else {
            continue;
        };
        if short < 5. {
            continue;
        }
        let mut points = rect.map(|p| {
            [
                ((p[0] as f32 / width as f32 * source_width as f32).round_ties_even())
                    .clamp(0., source_width as f32),
                ((p[1] as f32 / height as f32 * source_height as f32).round_ties_even())
                    .clamp(0., source_height as f32),
            ]
        });
        // 原实现量化后再次排序，45°菱形的 x 并列不能沿用浮点排序结果。
        points.sort_by(|a, b| a[0].total_cmp(&b[0]));
        if points[0][1] > points[1][1] {
            points.swap(0, 1);
        }
        if points[2][1] > points[3][1] {
            points.swap(2, 3);
        }
        let points = [points[0], points[2], points[3], points[1]].map(|p| {
            [
                p[0].min((source_width - 1) as f32),
                p[1].min((source_height - 1) as f32),
            ]
        });
        if distance(points[0].map(f64::from), points[1].map(f64::from)) < 4.
            || distance(points[0].map(f64::from), points[3].map(f64::from)) < 4.
        {
            continue;
        }
        result.push(Detection { points, score });
    }
    Ok(result)
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

type Rectangle = ([[f64; 2]; 4], f64);

/// 计入凸包排序及原始索引匹配的保守工作预算。
fn minimum_rect(points: &[Point<i32>], work: &mut usize) -> Result<Option<Rectangle>, Error> {
    consume(
        work,
        points
            .len()
            .checked_mul(points.len())
            .ok_or_else(|| error(ErrorCode::ResourceLimit, "凸包尺寸溢出"))?,
    )?;
    Ok(crate::detection_geometry::rectangle(points))
}

pub(crate) fn score(
    values: &[f32],
    width: usize,
    height: usize,
    rect: &[[f64; 2]],
    work: &mut usize,
) -> Result<f64, Error> {
    let xmin = rect
        .iter()
        .map(|p| p[0])
        .fold(f64::INFINITY, f64::min)
        .floor()
        .clamp(0., (width - 1) as f64) as usize;
    let xmax = rect
        .iter()
        .map(|p| p[0])
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil()
        .clamp(0., (width - 1) as f64) as usize;
    let ymin = rect
        .iter()
        .map(|p| p[1])
        .fold(f64::INFINITY, f64::min)
        .floor()
        .clamp(0., (height - 1) as f64) as usize;
    let ymax = rect
        .iter()
        .map(|p| p[1])
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil()
        .clamp(0., (height - 1) as f64) as usize;
    consume(work, (xmax - xmin + 1) * (ymax - ymin + 1))?;
    let mut mask = GrayImage::new((xmax - xmin + 1) as u32, (ymax - ymin + 1) as u32);
    let points: Vec<_> = rect
        .iter()
        .map(|p| Point::new((p[0] - xmin as f64) as i32, (p[1] - ymin as f64) as i32))
        .collect();
    crate::detection_mask::fill(&mut mask, &points);
    let mut total = 0.;
    let mut count = 0;
    for y in ymin..=ymax {
        for x in xmin..=xmax {
            if mask.get_pixel((x - xmin) as u32, (y - ymin) as u32)[0] != 0 {
                total += values[y * width + x] as f64;
                count += 1;
            }
        }
    }
    Ok(if count == 0 { 0. } else { total / count as f64 })
}
