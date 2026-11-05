//! 按已检测文字行复用 PDF 原生字符。几何、Unicode 或图片覆盖不可靠时明确选择 OCR。
use crate::{error, table_recover::Polygon};
use rapiddoc_core::{document::ImageFact, geometry, Error, ErrorCode, Rect};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Facts {
    visual_width_pt: f64,
    visual_height_pt: f64,
    rotation: u16,
    characters: Vec<Glyph>,
    images: Vec<ImageFact>,
}
#[derive(Deserialize)]
struct Glyph {
    index: usize,
    text: Option<String>,
    is_generated: bool,
    angle_degrees: f64,
    visual_tight_bounds: Rect,
}
pub struct NativePage {
    glyphs: Vec<Glyph>,
    images: Vec<[f64; 4]>,
    horizontal: bool,
}
#[derive(Deserialize)]
pub struct Line {
    pub points: Polygon,
    pub score: f64,
}
#[derive(Serialize)]
pub struct Match {
    pub method: &'static str,
    pub reason: &'static str,
    pub text: Option<String>,
    pub character_indices: Vec<usize>,
}
impl Match {
    fn ocr(reason: &'static str) -> Self {
        Self {
            method: "ocr",
            reason,
            text: None,
            character_indices: Vec::new(),
        }
    }
}
#[derive(Serialize)]
pub struct Output {
    pub lines: Vec<Match>,
    pub points: Vec<Polygon>,
    pub work: usize,
}
impl NativePage {
    pub fn from_json(data: &str, width: usize, height: usize) -> Result<Self, Error> {
        if data.len() > rapiddoc_core::resources::limit(32_000_000) {
            return Err(error(ErrorCode::ResourceLimit, "原生页面 JSON 超过预算"));
        }
        crate::preprocess::check_size(width, height, 16_000_000)?;
        let mut facts: Facts =
            serde_json::from_str(data).map_err(|e| error(ErrorCode::InvalidInput, e))?;
        if !facts.visual_width_pt.is_finite()
            || !facts.visual_height_pt.is_finite()
            || facts.visual_width_pt <= 0.
            || facts.visual_height_pt <= 0.
            || ![0, 90, 180, 270].contains(&facts.rotation)
        {
            return Err(error(ErrorCode::InvalidInput, "原生页面尺寸或方向无效"));
        }
        if facts.characters.len() > rapiddoc_core::resources::limit(100_000)
            || facts.images.len() > rapiddoc_core::resources::limit(10_000)
        {
            return Err(error(
                ErrorCode::ResourceLimit,
                "原生页面字符或图片超过预算",
            ));
        }
        let scale = geometry::Affine::new([
            width as f64 / facts.visual_width_pt,
            0.,
            0.,
            height as f64 / facts.visual_height_pt,
            0.,
            0.,
        ])?;
        for (position, glyph) in facts.characters.iter_mut().enumerate() {
            if glyph.index != position
                || !glyph.angle_degrees.is_finite()
                || glyph
                    .text
                    .as_ref()
                    .is_some_and(|text| text.len() > rapiddoc_core::resources::limit(64))
            {
                return Err(error(
                    ErrorCode::InvalidInput,
                    "原生字符索引、方向或文字无效",
                ));
            }
            glyph.visual_tight_bounds = scale.rect(glyph.visual_tight_bounds)?;
        }
        let images = facts
            .images
            .into_iter()
            .map(|image| scale.rect(image.visual_bounds).map(Into::into))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            glyphs: facts.characters,
            images,
            horizontal: facts.rotation == 0,
        })
    }
    /// 输入为页面像素的 DB 四角。只支持可可靠复用的水平行，其余继续 OCR。
    pub fn match_lines(&self, lines: &[Line], max_work: usize) -> Result<Output, Error> {
        let max_work = rapiddoc_core::resources::limit(max_work);
        if lines.len() > rapiddoc_core::resources::limit(1000)
            || max_work > rapiddoc_core::resources::limit(10_000_000)
        {
            return Err(error(ErrorCode::ResourceLimit, "原生行匹配预算无效或超限"));
        }
        let boxes = lines
            .iter()
            .map(|line| {
                if !line.score.is_finite()
                    || !(0. ..=1.).contains(&line.score)
                    || line
                        .points
                        .iter()
                        .flatten()
                        .any(|v| v.abs() > rapiddoc_core::resources::limit(1_000_000.))
                {
                    return Err(error(ErrorCode::InvalidInput, "文字行评分或坐标无效"));
                }
                let b: [f64; 4] = geometry::bounds(&line.points)?.into();
                if area(b) <= 0. {
                    return Err(error(ErrorCode::InvalidInput, "文字行必须有正面积"));
                }
                Ok(b)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let work = lines
            .len()
            .checked_mul(self.glyphs.len() + self.images.len())
            .ok_or_else(|| error(ErrorCode::ResourceLimit, "原生行匹配工作量溢出"))?;
        if work > max_work {
            return Err(error(ErrorCode::ResourceLimit, "原生行匹配工作量超过预算"));
        }
        let mut assigned = vec![Vec::new(); lines.len()];
        for (index, glyph) in self.glyphs.iter().enumerate() {
            let b: [f64; 4] = glyph.visual_tight_bounds.into();
            if area(b) <= 0.
                || glyph
                    .text
                    .as_ref()
                    .is_some_and(|t| t.chars().all(char::is_whitespace))
            {
                continue;
            }
            let center = [(b[0] + b[2]) / 2., (b[1] + b[3]) / 2.];
            let mut best = None;
            for (line_index, (line, bounds)) in lines.iter().zip(&boxes).enumerate() {
                let overlap = intersection(b, *bounds) / area(b);
                if overlap >= 0.5
                    && inside(center, &line.points)
                    && best.is_none_or(|(_, score)| overlap > score)
                {
                    best = Some((line_index, overlap));
                }
            }
            if let Some((line, _)) = best {
                assigned[line].push(index);
            }
        }
        let results = lines
            .iter()
            .zip(boxes)
            .zip(assigned)
            .map(|((line, bounds), indices)| {
                if !self.horizontal
                    || line.points[1][0] <= line.points[0][0]
                    || (line.points[1][1] - line.points[0][1]).abs()
                        > (line.points[1][0] - line.points[0][0]) * 0.0875
                {
                    return Match::ocr("non_horizontal");
                }
                if self
                    .images
                    .iter()
                    .any(|image| intersection(*image, bounds) > 0.)
                {
                    return Match::ocr("raster_overlap");
                }
                self.native_line(bounds, &indices)
            })
            .collect();
        Ok(Output {
            lines: results,
            points: lines.iter().map(|line| line.points).collect(),
            work,
        })
    }
    fn native_line(&self, line: [f64; 4], indices: &[usize]) -> Match {
        let Some(&first) = indices.first() else {
            return Match::ocr("no_native_characters");
        };
        let mut covered: [f64; 4] = self.glyphs[first].visual_tight_bounds.into();
        let mut previous_x = f64::NEG_INFINITY;
        for &index in indices {
            let glyph = &self.glyphs[index];
            let Some(text) = &glyph.text else {
                return Match::ocr("invalid_unicode");
            };
            if text.contains('\u{fffd}')
                || text.chars().any(char::is_control)
                || glyph.is_generated
                || glyph.angle_degrees.abs() > 5.
            {
                return Match::ocr("unreliable_native_character");
            }
            let b: [f64; 4] = glyph.visual_tight_bounds.into();
            if b[0] + 0.5 < previous_x {
                return Match::ocr("non_monotonic_characters");
            }
            previous_x = b[0];
            covered = [
                covered[0].min(b[0]),
                covered[1].min(b[1]),
                covered[2].max(b[2]),
                covered[3].max(b[3]),
            ];
        }
        // DB 框包含扩边；宽/高覆盖不足意味着同一检测行可能混有扫描文字。
        if (covered[2] - covered[0]) / (line[2] - line[0]) < 0.8
            || (covered[3] - covered[1]) / (line[3] - line[1]) < 0.5
        {
            return Match::ocr("insufficient_native_coverage");
        }
        let last = *indices.last().unwrap();
        let mut selected = indices.iter().copied().peekable();
        let mut text = String::new();
        for index in first..=last {
            let glyph = &self.glyphs[index];
            if selected.peek() == Some(&index) {
                selected.next();
            } else if glyph.text.as_ref().is_none_or(|t| {
                !t.chars()
                    .all(|c| c.is_whitespace() && c != '\r' && c != '\n')
            }) {
                return Match::ocr("noncontiguous_characters");
            }
            text.push_str(glyph.text.as_deref().unwrap_or(""));
        }
        Match {
            method: "native",
            reason: "native_coverage",
            text: Some(text),
            character_indices: (first..=last).map(|i| self.glyphs[i].index).collect(),
        }
    }
}
fn area(b: [f64; 4]) -> f64 {
    (b[2] - b[0]) * (b[3] - b[1])
}
fn intersection(a: [f64; 4], b: [f64; 4]) -> f64 {
    (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.)
}
fn inside(p: [f64; 2], polygon: &Polygon) -> bool {
    let mut sign = 0.;
    for i in 0..4 {
        let a = polygon[i];
        let b = polygon[(i + 1) % 4];
        let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        if cross != 0. {
            if sign != 0. && sign * cross < 0. {
                return false;
            }
            sign = cross;
        }
    }
    sign != 0.
}
