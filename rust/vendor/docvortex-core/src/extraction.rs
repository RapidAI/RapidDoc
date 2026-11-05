//! 仅处理 PDFium 已读出的数值；不持有、传递或调用任何原生 PDF 句柄。

use crate::geometry::{Box4, Size};
pub type RawGeometry = (Box4, Option<Box4>, Option<Box4>, Option<Size>);
pub type VisualGeometry = (Box4, Option<Box4>, Option<Box4>, Option<Size>);

/// 与 Python min 保持相同的有符号零及 NaN 首项语义。
fn minimum(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}
/// 与 Python max 保持相同的首项语义。
fn maximum(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// 把 PDF 原始点转为视觉页面坐标，保留浮点页面边界。
fn point(p: Size, frame: Box4, angle: i32) -> Size {
    let w = (frame[2] - frame[0]).abs();
    let h = (frame[3] - frame[1]).abs();
    let x = p[0] - minimum(frame[0], frame[2]);
    let y = maximum(frame[1], frame[3]) - p[1];
    match angle.rem_euclid(360) {
        90 => [h - y, x],
        180 => [w - x, h - y],
        270 => [y, w - x],
        _ => [x, y],
    }
}

/// 按原四角遍历次序转换扩展框，非有限或零面积返回空值。
fn visual(raw: Option<Box4>, frame: Box4, angle: i32) -> Option<Box4> {
    let b = raw?;
    let points =
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[1]], [b[2], b[3]]].map(|p| point(p, frame, angle));
    let mut out = [points[0][0], points[0][1], points[0][0], points[0][1]];
    for p in &points[1..] {
        out = [
            minimum(out[0], p[0]),
            minimum(out[1], p[1]),
            maximum(out[2], p[0]),
            maximum(out[3], p[1]),
        ];
    }
    if out.iter().all(|v| v.is_finite()) && out[2] > out[0] && out[3] > out[1] {
        Some(out)
    } else {
        None
    }
}

/// 批量物化布局框与扩展框，布局框仍使用原有整数页面高度。
pub fn materialize(
    rows: Vec<RawGeometry>,
    frame: Box4,
    rounded: Size,
    angle: i32,
) -> Vec<VisualGeometry> {
    rows.into_iter()
        .map(|(selected, loose, tight, origin)| {
            let y0 = rounded[1] - (selected[1] - frame[1]);
            let y1 = rounded[1] - (selected[3] - frame[1]);
            let mut layout = [
                minimum(selected[0], selected[2]) - frame[0],
                minimum(y0, y1),
                maximum(selected[0], selected[2]) - frame[0],
                maximum(y0, y1),
            ];
            if angle != 0 {
                layout = crate::geometry::rotate(layout, rounded, (360 - angle).rem_euclid(360));
                layout = [
                    minimum(layout[0], layout[2]),
                    minimum(layout[1], layout[3]),
                    maximum(layout[0], layout[2]),
                    maximum(layout[1], layout[3]),
                ];
            }
            let origin = origin
                .map(|p| point(p, frame, angle))
                .filter(|p| p.iter().all(|v| v.is_finite()));
            (
                layout,
                visual(loose, frame, angle),
                visual(tight, frame, angle),
                origin,
            )
        })
        .collect()
}
