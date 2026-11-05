//! 从自有 Path 事实构建轴线候选；曲线不当作直边，填充细条另由后续规则处理。

use rapiddoc_core::document::{DrawingLine, Page, PathSegment, VectorPath};
use rapiddoc_core::geometry::{intersect, page_transform, Affine};
use rapiddoc_core::{Error, Rect};

/// 从页面路径提取有色描边轴线，合并后按视觉位置稳定排序。
pub fn extract(page: &Page) -> Result<Vec<DrawingLine>, Error> {
    let visual = page_transform(page.page_bounds, page.rotation)?;
    let frame = visual.rect(page.page_bounds)?;
    let mut result = Vec::new();
    for path in &page.paths {
        if !path.stroke_visible {
            continue;
        }
        let matrix = Affine::new(path.matrix)?;
        let clip = match path.clip_bounds {
            Some(value) => intersect(frame, visual.rect(value)?)?,
            None => frame,
        };
        let mut current = None;
        let mut first = None;
        for segment in &path.segments {
            if segment.kind == "move" {
                first = Some(segment);
            } else if segment.kind == "line" {
                if let Some(start) = current {
                    add_segment(start, segment, path, &[visual, matrix], clip, &mut result)?;
                }
            }
            current = Some(segment);
            if segment.close {
                if let Some(start) = first {
                    if start.point != segment.point {
                        add_segment(segment, start, path, &[visual, matrix], clip, &mut result)?;
                    }
                    current = Some(start);
                }
            }
        }
    }
    merge(result, frame)
}

/// 对同方向线按轴坐标聚类，再合并 2 point 内相接区间；最终使用视觉位置稳定排序。
fn merge(lines: Vec<DrawingLine>, frame: Rect) -> Result<Vec<DrawingLine>, Error> {
    let mut result = Vec::new();
    for orientation in ["horizontal", "vertical"] {
        let mut values: Vec<_> = lines
            .iter()
            .filter(|line| line.orientation == orientation)
            .cloned()
            .collect();
        values.sort_by(|a, b| {
            coordinate(a)
                .total_cmp(&coordinate(b))
                .then_with(|| interval(a).0.total_cmp(&interval(b).0))
        });
        let mut clusters: Vec<Vec<DrawingLine>> = Vec::new();
        for line in values {
            let fits = clusters
                .last()
                .and_then(|cluster| cluster.first())
                .is_some_and(|first| coordinate(&line) - coordinate(first) <= 2.);
            if fits {
                if let Some(cluster) = clusters.last_mut() {
                    cluster.push(line);
                }
            } else {
                clusters.push(vec![line]);
            }
        }
        for mut cluster in clusters {
            cluster.sort_by(|a, b| interval(a).0.total_cmp(&interval(b).0));
            let mut group = Vec::new();
            let mut end = f64::NEG_INFINITY;
            for line in cluster {
                let (start, last) = interval(&line);
                if !group.is_empty() && start > end + 2. {
                    if let Some(line) = combine(&group, frame)? {
                        result.push(line);
                    }
                    group.clear();
                    end = f64::NEG_INFINITY;
                }
                group.push(line);
                end = end.max(last);
            }
            if !group.is_empty() {
                if let Some(line) = combine(&group, frame)? {
                    result.push(line);
                }
            }
        }
    }
    result.sort_by(|a, b| {
        let a_box: [f64; 4] = a.bounds.into();
        let b_box: [f64; 4] = b.bounds.into();
        a_box[1]
            .total_cmp(&b_box[1])
            .then_with(|| a_box[0].total_cmp(&b_box[0]))
            .then_with(|| a.orientation.cmp(&b.orientation))
    });
    Ok(result)
}

/// 取得垂直于线段方向的轴坐标。
fn coordinate(line: &DrawingLine) -> f64 {
    if line.orientation == "horizontal" {
        line.start[1]
    } else {
        line.start[0]
    }
}

/// 取得沿线段方向的有序区间。
fn interval(line: &DrawingLine) -> (f64, f64) {
    if line.orientation == "horizontal" {
        (line.start[0], line.end[0])
    } else {
        (line.start[1], line.end[1])
    }
}

/// 使用长度加权中心和覆盖宽度合并一组共线段，保留原求和顺序。
fn combine(lines: &[DrawingLine], frame: Rect) -> Result<Option<DrawingLine>, Error> {
    let Some(first) = lines.first() else {
        return Ok(None);
    };
    let mut total = 0.;
    let mut weighted = 0.;
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    let mut start = f64::INFINITY;
    let mut end = f64::NEG_INFINITY;
    let mut width = 0f64;
    for line in lines {
        let (a, b) = interval(line);
        let length = (b - a).max(0.01);
        let value = coordinate(line);
        total += length;
        weighted += value * length;
        low = low.min(value);
        high = high.max(value);
        start = start.min(a);
        end = end.max(b);
        width = width.max(line.width);
    }
    let center = weighted / total;
    let width = width + high - low;
    let (a, b) = if first.orientation == "horizontal" {
        ([start, center], [end, center])
    } else {
        ([center, start], [center, end])
    };
    axis_line(a, b, width, frame)
}

/// 按局部法向量换算非等比变换后的描边宽度，不使用路径整体框猜线宽。
fn add_segment(
    start: &PathSegment,
    end: &PathSegment,
    path: &VectorPath,
    transforms: &[Affine; 2],
    clip: Rect,
    result: &mut Vec<DrawingLine>,
) -> Result<(), Error> {
    let [visual, matrix] = *transforms;
    let local_start = start.local_point;
    let local_end = end.local_point;
    let dx = local_end[0] - local_start[0];
    let dy = local_end[1] - local_start[1];
    let length = dx.hypot(dy);
    if length == 0. {
        return Ok(());
    }
    let [a, b, c, d, _, _] = matrix.values();
    let nx = -dy / length;
    let ny = dx / length;
    let width = path.stroke_width_local * (a * nx + c * ny).hypot(b * nx + d * ny);
    if let Some(line) = axis_line(
        visual.point(start.point)?,
        visual.point(end.point)?,
        width,
        clip,
    )? {
        result.push(line);
    }
    Ok(())
}

/// 按既有 1 point/2% 轴线容差吸附并裁剪，斜线和不足 1 point 的线段不组成规则证据。
fn axis_line(
    start: [f64; 2],
    end: [f64; 2],
    width: f64,
    clip: Rect,
) -> Result<Option<DrawingLine>, Error> {
    let dx = (end[0] - start[0]).abs();
    let dy = (end[1] - start[1]).abs();
    let (horizontal, coordinate, first, last, width) = if dx >= dy && dy <= 1f64.max(dx * 0.02) {
        (
            true,
            (start[1] + end[1]) / 2.,
            start[0].min(end[0]),
            start[0].max(end[0]),
            width.max(dy),
        )
    } else if dy > dx && dx <= 1f64.max(dy * 0.02) {
        (
            false,
            (start[0] + end[0]) / 2.,
            start[1].min(end[1]),
            start[1].max(end[1]),
            width.max(dx),
        )
    } else {
        return Ok(None);
    };
    if last - first < 1. {
        return Ok(None);
    }
    let raw = if horizontal {
        [
            first,
            coordinate - width / 2.,
            last,
            coordinate + width / 2.,
        ]
    } else {
        [
            coordinate - width / 2.,
            first,
            coordinate + width / 2.,
            last,
        ]
    };
    let clipped = intersect(Rect::try_from(raw)?, clip)?;
    let [left, top, right, bottom] = clipped.into();
    if right <= left || bottom <= top {
        return Ok(None);
    }
    let (start, end) = if horizontal {
        let y = coordinate.clamp(top, bottom);
        ([left, y], [right, y])
    } else {
        let x = coordinate.clamp(left, right);
        ([x, top], [x, bottom])
    };
    Ok(Some(DrawingLine {
        start,
        end,
        bounds: clipped,
        width,
        orientation: if horizontal { "horizontal" } else { "vertical" }.into(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_curve_candidates_and_short_lines_are_not_rules() {
        let clip = Rect::try_from([0., 0., 100., 100.]).expect("clip");
        assert!(axis_line([0., 0.], [20., 20.], 1., clip)
            .expect("diagonal")
            .is_none());
        assert!(axis_line([2., 2.], [2.5, 2.], 1., clip)
            .expect("short")
            .is_none());
    }

    #[test]
    fn clipping_preserves_axis_and_original_width() {
        let clip = Rect::try_from([10., 10., 20., 20.]).expect("clip");
        let line = axis_line([0., 15.], [30., 15.], 2., clip)
            .expect("valid")
            .expect("line");
        assert_eq!(line.start, [10., 15.]);
        assert_eq!(line.end, [20., 15.]);
        assert_eq!(line.width, 2.);
        let empty = Rect::try_from([20., 20., 20., 20.]).expect("empty clip");
        assert!(axis_line([0., 15.], [30., 15.], 2., empty)
            .expect("clipped")
            .is_none());
    }

    #[test]
    fn fragmented_rules_merge_but_transitive_coordinate_drift_does_not() {
        let frame = Rect::try_from([0., 0., 100., 100.]).expect("frame");
        let a = axis_line([10., 10.], [20., 10.], 1., frame)
            .expect("valid")
            .expect("line");
        let b = axis_line([21., 10.], [30., 10.], 1., frame)
            .expect("valid")
            .expect("line");
        let merged = merge(vec![a, b], frame).expect("merged");
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].end, [30., 10.]);
        let lines = [10., 11.5, 13.]
            .into_iter()
            .map(|y| {
                axis_line([10., y], [20., y], 1., frame)
                    .expect("valid")
                    .expect("line")
            })
            .collect();
        assert_eq!(merge(lines, frame).expect("clusters").len(), 2);
    }
}
