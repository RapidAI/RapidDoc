//! UNET 的 8 连通域及外接框过滤；只扫描行端点求凸包，避免复制每个像素坐标。
use crate::{detection_geometry::rectangle, error, preprocess::check_size};
use imageproc::{geometry::convex_hull, point::Point};
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;

#[derive(Serialize)]
pub struct Geometry {
    pub lines: Vec<[f32; 4]>,
    /// 顺序为左上、右上、右下、左下；TableRecover 调用前需换成逆时针。
    pub polygons: Vec<[[f32; 2]; 4]>,
}
struct Component {
    points: Vec<Point<i32>>,
    bounds: [usize; 4],
    order: (usize, usize),
}

pub fn analyze(
    bytes: &[u8],
    width: usize,
    height: usize,
    axis: Option<usize>,
    line_extent: usize,
) -> Result<Geometry, Error> {
    let count = check_size(width, height, 4_000_000)?;
    if bytes.len() != count || axis.is_some_and(|v| v > 1) {
        return Err(error(ErrorCode::InvalidInput, "UNET 二值图长度或轴无效"));
    }
    if width > rapiddoc_core::resources::limit(4096)
        || height > rapiddoc_core::resources::limit(4096)
    {
        return Err(error(ErrorCode::ResourceLimit, "UNET 二值图边长超过限制"));
    }
    let mut visited = vec![false; count];
    let mut components = Vec::new();
    let mut left = vec![usize::MAX; height];
    let mut right = vec![0; height];
    let mut touched = Vec::new();
    for seed in 0..count {
        if visited[seed] || bytes[seed] == 0 {
            continue;
        }
        if components.len() >= rapiddoc_core::resources::limit(2000) {
            return Err(error(ErrorCode::ResourceLimit, "UNET 连通域超过 2000"));
        }
        let mut queue = vec![seed];
        visited[seed] = true;
        let mut cursor = 0;
        let mut bounds = [width, height, 0, 0];
        let mut order = (height, width);
        while cursor < queue.len() {
            let pixel = queue[cursor];
            cursor += 1;
            let (y, x) = (pixel / width, pixel % width);
            if left[y] == usize::MAX {
                touched.push(y);
            }
            left[y] = left[y].min(x);
            right[y] = right[y].max(x);
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].min(y);
            bounds[2] = bounds[2].max(x + 1);
            bounds[3] = bounds[3].max(y + 1);
            order = order.min((y / 2, x / 2));
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let index = ny * width + nx;
                    if !visited[index] && bytes[index] != 0 {
                        visited[index] = true;
                        queue.push(index);
                    }
                }
            }
        }
        touched.sort_unstable();
        let mut points = Vec::with_capacity(touched.len() * 2);
        for &y in &touched {
            points.push(Point::new(left[y] as i32, y as i32));
            points.push(Point::new(right[y] as i32, y as i32));
            left[y] = usize::MAX;
            right[y] = 0;
        }
        touched.clear();
        components.push(Component {
            points,
            bounds,
            order,
        });
    }
    // OpenCV 默认 BBDT 的标签按 2x2 block 首次遇见顺序分配。
    components.sort_by_key(|c| c.order);
    let mut output = Geometry {
        lines: Vec::new(),
        polygons: Vec::new(),
    };
    let mut work = 50_000_000usize;
    for component in components {
        let [x0, y0, x1, y1] = component.bounds;
        if let Some(axis) = axis {
            if (if axis == 0 { x1 - x0 } else { y1 - y0 }) <= line_extent {
                continue;
            }
        } else if (x1 - x0) * (y1 - y0) > count * 3 / 4 {
            continue;
        }
        let hull = convex_hull(component.points.as_slice());
        let cost = component.points.len().saturating_mul(hull.len() + 16);
        work = work
            .checked_sub(cost)
            .ok_or_else(|| error(ErrorCode::ResourceLimit, "UNET 凸包工作量超过预算"))?;
        let Some((points, _)) = rectangle(&component.points) else {
            // 单像素宽度直线仍可作为线段；区域模式的退化框不参与恢复。
            if axis.is_some() && hull.len() <= 2 {
                if let (Some(a), Some(b)) = (hull.first(), hull.last()) {
                    let (a, b) = if if a.x == b.x {
                        a.y >= b.y
                    } else {
                        (a.x, a.y) <= (b.x, b.y)
                    } {
                        (a, b)
                    } else {
                        (b, a)
                    };
                    output
                        .lines
                        .push([a.x as f32, a.y as f32, b.x as f32, b.y as f32]);
                }
            }
            continue;
        };
        let points = order_points(points.map(|p| p.map(|v| v as f32)));
        let (w, h) = dimensions(points);
        if axis.is_some() {
            let [a, b, c, d] = points;
            output.lines.push(if w < h {
                [
                    (a[0] + b[0]) / 2.,
                    (a[1] + b[1]) / 2.,
                    (c[0] + d[0]) / 2.,
                    (c[1] + d[1]) / 2.,
                ]
            } else {
                [
                    (a[0] + d[0]) / 2.,
                    (a[1] + d[1]) / 2.,
                    (b[0] + c[0]) / 2.,
                    (b[1] + c[1]) / 2.,
                ]
            });
        } else if w * h < count as f32 * 0.5 && w >= 15. && h >= 15. {
            output.polygons.push(points);
        }
    }
    Ok(output)
}

fn norm(a: [f32; 2], b: [f32; 2]) -> f32 {
    let x = a[0] - b[0];
    let y = a[1] - b[1];
    (x * x + y * y).sqrt()
}
fn dimensions(p: [[f32; 2]; 4]) -> (f32, f32) {
    (
        (norm(p[1], p[0]) + norm(p[2], p[3])) / 2.,
        (norm(p[1], p[2]) + norm(p[0], p[3])) / 2.,
    )
}
fn order_points(mut points: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    points.sort_by(|a, b| a[0].total_cmp(&b[0]));
    if points[0][1] > points[1][1] {
        points.swap(0, 1);
    }
    let a = points[0];
    let b = points[1];
    let c = points[2];
    let d = points[3];
    if norm(c, a) > norm(d, a) {
        [a, d, c, b]
    } else {
        [a, c, d, b]
    }
}
