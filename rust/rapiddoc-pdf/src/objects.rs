//! 有界遍历 Form、路径和图片，不把 PDFium 对象或借用内存带出锁。

use crate::map_error;
use pdfium_render::prelude::*;
use rapiddoc_core::document::{ImageFact, PathSegment, VectorPath};
use rapiddoc_core::geometry::{bounds as point_bounds, intersect, Affine};
use rapiddoc_core::{Error, ErrorCode, Limits, Rect};

#[derive(Default)]
pub(crate) struct Objects {
    pub paths: Vec<VectorPath>,
    pub images: Vec<ImageFact>,
    pub form_count: usize,
    pub object_count: usize,
    segments: usize,
}

pub(crate) struct Context<'a> {
    pub visual: Affine,
    pub page_clip: Rect,
    pub limits: &'a Limits,
}

/// 将库中的矩阵转换为有限双精度矩阵。
fn matrix(value: PdfMatrix) -> Result<Affine, Error> {
    Affine::new([
        value.a() as f64,
        value.b() as f64,
        value.c() as f64,
        value.d() as f64,
        value.e() as f64,
        value.f() as f64,
    ])
}

/// 将库矩形复制为独立的校验矩形。
pub(crate) fn rect(value: PdfRect) -> Result<Rect, Error> {
    Rect::try_from([
        value.left().value as f64,
        value.bottom().value as f64,
        value.right().value as f64,
        value.top().value as f64,
    ])
}

/// 累加对象或片段预算，溢出和超限都明确失败。
fn charge(current: &mut usize, amount: usize, limit: usize) -> Result<(), Error> {
    let limit = rapiddoc_core::resources::limit(limit);
    *current = current
        .checked_add(amount)
        .ok_or_else(|| limit_error("对象预算溢出"))?;
    if *current > limit {
        return Err(limit_error("页面对象或路径片段超过限制"));
    }
    Ok(())
}

/// 构造资源上限错误。
fn limit_error(message: &str) -> Error {
    Error {
        code: ErrorCode::ResourceLimit,
        message: message.into(),
    }
}

/// 根据父矩阵累积保守 clip 框；空交集保持为空，不丢弃父裁剪。
fn clipping(
    object: &PdfPageObject<'_>,
    parent: Affine,
    inherited: Option<Rect>,
    output: &mut Objects,
    limits: &Limits,
) -> Result<Option<Rect>, Error> {
    let mut result = inherited;
    if let Some(clip) = object.get_clip_path() {
        for index in clip.as_range() {
            let segments = clip.get(index).map_err(map_error)?;
            charge(
                &mut output.segments,
                segments.len() as usize,
                limits.max_path_segments_per_page,
            )?;
            let mut points = Vec::with_capacity(segments.len() as usize);
            for index in segments.as_range() {
                let segment = segments.get(index).map_err(map_error)?;
                let (x, y) = segment.point();
                points.push(parent.point([x.value as f64, y.value as f64])?);
            }
            if !points.is_empty() {
                let value = point_bounds(&points)?;
                result = Some(match result {
                    Some(previous) => intersect(previous, value)?,
                    None => value,
                });
            }
        }
    }
    Ok(result)
}

/// 遍历一个对象；累计矩阵作用于内容，clip 仅乘父矩阵以免重复变换。
pub(crate) fn walk(
    object: &PdfPageObject<'_>,
    object_path: &mut Vec<usize>,
    parent: Affine,
    inherited: Option<Rect>,
    output: &mut Objects,
    context: &Context<'_>,
) -> Result<(), Error> {
    let visual = context.visual;
    let limits = context.limits;
    charge(&mut output.object_count, 1, limits.max_objects_per_page)?;
    let clip = clipping(object, parent, inherited, output, limits)?;
    let combined = matrix(object.matrix().map_err(map_error)?)?.then(parent)?;
    if let Some(form) = object.as_x_object_form_object() {
        output.form_count += 1;
        if object_path.len() > limits.max_form_depth {
            return Err(limit_error("Form 嵌套超过限制"));
        }
        for index in form.as_range() {
            let child = form.get(index).map_err(map_error)?;
            object_path.push(index);
            walk(&child, object_path, combined, clip, output, context)?;
            object_path.pop();
        }
        return Ok(());
    }
    if let Some(path) = object.as_path_object() {
        let raw_segments = path.segments();
        charge(
            &mut output.segments,
            raw_segments.len() as usize,
            limits.max_path_segments_per_page,
        )?;
        let mut segments = Vec::with_capacity(raw_segments.len() as usize);
        for index in raw_segments.as_range() {
            let segment = raw_segments.get(index).map_err(map_error)?;
            let kind = match segment.segment_type() {
                PdfPathSegmentType::MoveTo => "move",
                PdfPathSegmentType::LineTo => "line",
                PdfPathSegmentType::BezierTo => "bezier",
                PdfPathSegmentType::Unknown => {
                    return Err(Error {
                        code: ErrorCode::CorruptDocument,
                        message: "未知路径片段类型".into(),
                    })
                }
            };
            let (x, y) = segment.point();
            segments.push(PathSegment {
                kind: kind.into(),
                point: combined.point([x.value as f64, y.value as f64])?,
                local_point: [x.value as f64, y.value as f64],
                close: segment.is_close(),
            });
        }
        let bounds = parent.rect(rect(object.bounds().map_err(map_error)?.to_rect())?)?;
        let fill = object.fill_color().map_err(map_error)?;
        let stroke = object.stroke_color().map_err(map_error)?;
        let width = object.stroke_width().map_err(map_error)?.value as f64;
        if !width.is_finite() || width < 0. {
            return Err(Error {
                code: ErrorCode::CorruptDocument,
                message: "无效描边宽度".into(),
            });
        }
        let fill_rule = path.fill_mode().map_err(map_error)?;
        let stroke_enabled = path.is_stroked().map_err(map_error)?;
        let fill_visible = fill_rule != PdfPathFillMode::None && fill.alpha() > 0;
        let stroke_visible = stroke_enabled && stroke.alpha() > 0;
        let evidence_bounds = if segments.is_empty() {
            None
        } else {
            let points = segments
                .iter()
                .map(|segment| visual.point(segment.point))
                .collect::<Result<Vec<_>, _>>()?;
            let raw_bounds: [f64; 4] = point_bounds(&points)?.into();
            let [a, b, c, d, _, _] = combined.values();
            let margin = if stroke_visible {
                0.5 * width * a.hypot(b).max(c.hypot(d))
            } else {
                0.
            };
            let expanded = Rect::try_from([
                raw_bounds[0] - margin,
                raw_bounds[1] - margin,
                raw_bounds[2] + margin,
                raw_bounds[3] + margin,
            ])?;
            let clipped = match clip {
                Some(clip) => intersect(expanded, visual.rect(clip)?)?,
                None => expanded,
            };
            let clipped = intersect(clipped, context.page_clip)?;
            Some(clipped)
        };
        output.paths.push(VectorPath {
            object_path: object_path.clone(),
            bounds,
            visual_bounds: visual.rect(bounds)?,
            evidence_bounds,
            clip_bounds: clip,
            matrix: combined.values(),
            segments,
            fill_rule: match fill_rule {
                PdfPathFillMode::None => "none",
                PdfPathFillMode::EvenOdd => "even_odd",
                PdfPathFillMode::Winding => "winding",
            }
            .into(),
            fill_rgba: [fill.red(), fill.green(), fill.blue(), fill.alpha()],
            stroke: stroke_enabled,
            stroke_rgba: [stroke.red(), stroke.green(), stroke.blue(), stroke.alpha()],
            stroke_width_local: width,
            fill_visible,
            stroke_visible,
        });
    } else if object.as_image_object().is_some() {
        let bounds = parent.rect(rect(object.bounds().map_err(map_error)?.to_rect())?)?;
        output.images.push(ImageFact {
            object_path: object_path.clone(),
            bounds,
            visual_bounds: visual.rect(bounds)?,
            clip_bounds: clip,
        });
    }
    Ok(())
}
