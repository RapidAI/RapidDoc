//! 图上字符候选与差分像素核验；不保存 PDFium 地址或 Python 对象。
use std::collections::HashMap;

pub type Box4 = [f64; 4];
pub type Record = (u32, usize, Box4);
pub type Visibility = (usize, bool, Option<Box4>);

/// 精确使用 CPython 的空白集合，包含 Rust Unicode whitespace 未包含的控制分隔符。
pub fn python_space(code: u32) -> bool {
    matches!(code, 9..=13 | 28..=32 | 0x85 | 0xa0 | 0x1680 | 0x2000..=0x200a | 0x2028 | 0x2029 | 0x202f | 0x205f | 0x3000)
}

/// 保持 Python min 对相等值和有符号零的首项选择。
fn minimum(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// 保持 Python max 对相等值和有符号零的首项选择。
fn maximum(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// 计算非空矩形面积，不能把裁剪后的反向边界重新归一化。
fn area(b: Box4) -> f64 {
    maximum(0.0, b[2] - b[0]) * maximum(0.0, b[3] - b[1])
}

/// 计算与参考实现相同的边界交集。
fn intersect(a: Box4, b: Box4) -> Box4 {
    [
        maximum(a[0], b[0]),
        maximum(a[1], b[1]),
        minimum(a[2], b[2]),
        minimum(a[3], b[3]),
    ]
}

/// 按参考切片顺序计算矩形并集，浮点加法顺序保持不变。
fn union_area(boxes: &[Box4]) -> f64 {
    let valid: Vec<_> = boxes.iter().copied().filter(|b| area(*b) > 0.0).collect();
    let mut edges: Vec<_> = valid.iter().flat_map(|b| [b[0], b[2]]).collect();
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
    edges.dedup();
    let mut total = 0.0;
    for edge in edges.windows(2) {
        let (left, right) = (edge[0], edge[1]);
        let mut intervals: Vec<_> = valid
            .iter()
            .filter(|b| b[0] < right && b[2] > left)
            .map(|b| (b[1], b[3]))
            .collect();
        intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (mut top, mut height) = (f64::NEG_INFINITY, 0.0);
        for (bottom, end) in intervals {
            height += maximum(0.0, end - maximum(bottom, top));
            top = maximum(top, end);
        }
        total += (right - left) * height;
    }
    total
}

/// 将四个角按 Python 的 x/y 嵌套顺序变换，保留页面旋转与 CropBox 偏移。
fn visual_box(b: Box4, frame: Box4, rotation: i32) -> Box4 {
    let mut points = Vec::with_capacity(4);
    for x in [b[0], b[2]] {
        for y in [b[1], b[3]] {
            points.push(match rotation {
                90 => (y - frame[1], x - frame[0]),
                180 => (frame[2] - x, y - frame[1]),
                270 => (frame[3] - y, frame[2] - x),
                _ => (x - frame[0], frame[3] - y),
            });
        }
    }
    [
        points.iter().map(|p| p.0).reduce(minimum).unwrap(),
        points.iter().map(|p| p.1).reduce(minimum).unwrap(),
        points.iter().map(|p| p.0).reduce(maximum).unwrap(),
        points.iter().map(|p| p.1).reduce(maximum).unwrap(),
    ]
}

/// 独立候选框可在原页面和文本页关闭后继续消费，不携带原生对象地址。
pub struct Candidates {
    pub boxes: Vec<Box4>,
    pub paints: bool,
}

/// 批量应用字符过滤、视觉裁剪与图像并集覆盖；未知可见性沿用参考的保守可见默认值。
pub fn candidates(
    records: &[Record],
    frame: Box4,
    rotation: i32,
    images: &[Box4],
    visibility: &[Visibility],
    overlap: f64,
) -> Option<Candidates> {
    if frame
        .iter()
        .chain(images.iter().flatten())
        .any(|v| !v.is_finite())
        || records.iter().flat_map(|r| r.2).any(|v| !v.is_finite())
        || visibility
            .iter()
            .filter_map(|v| v.2)
            .flatten()
            .any(|v| !v.is_finite())
        || !overlap.is_finite()
        || ![0, 90, 180, 270].contains(&rotation)
    {
        return None;
    }
    let (mut width, mut height) = (frame[2] - frame[0], frame[3] - frame[1]);
    if rotation == 90 || rotation == 270 {
        std::mem::swap(&mut width, &mut height);
    }
    let lookup: HashMap<_, _> = visibility.iter().map(|v| (v.0, (v.1, v.2))).collect();
    let mut result = Candidates {
        boxes: Vec::new(),
        paints: false,
    };
    for &(code, address, bounds) in records {
        if code == 0 || code > 0x10ffff || python_space(code) || address == 0 {
            continue;
        }
        let (visible, clip) = lookup.get(&address).copied().unwrap_or((true, None));
        let mut b = visual_box(bounds, frame, rotation);
        if let Some(clip) = clip {
            b = intersect(b, clip);
        }
        b = intersect(b, [0.0, 0.0, width, height]);
        let a = area(b);
        if a > 0.0 {
            let overlaps: Vec<_> = images.iter().map(|image| intersect(b, *image)).collect();
            if union_area(&overlaps) >= a * overlap {
                result.boxes.push(b);
                result.paints |= visible;
            }
        }
    }
    Some(result)
}

/// 直接扫描每个像素区域，等价于 Pillow crop/getbbox，避免每字符创建图像对象。
pub fn painted_count(
    data: &[u8],
    width: usize,
    height: usize,
    boxes: &[Box4],
    page_width: f64,
    page_height: f64,
) -> usize {
    let (sx, sy) = (width as f64 / page_width, height as f64 / page_height);
    boxes
        .iter()
        .filter(|b| {
            let x0 = maximum(0.0, (b[0] * sx).floor()) as usize;
            let y0 = maximum(0.0, (b[1] * sy).floor()) as usize;
            let x1 = minimum(width as f64, (b[2] * sx).ceil()) as usize;
            let y1 = minimum(height as f64, (b[3] * sy).ceil()) as usize;
            x0 < x1
                && y0 < y1
                && x1 <= width
                && y1 <= height
                && (y0..y1).any(|y| data[y * width + x0..y * width + x1].iter().any(|v| *v != 0))
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 控制空白要跳过，代理码点与合法非空白字符保持参考计数规则。
    #[test]
    fn whitespace_surrogates_and_overlap() {
        let records = vec![
            (28, 1, [1.0, 1.0, 2.0, 2.0]),
            (0xd800, 1, [1.0, 1.0, 2.0, 2.0]),
            (65, 2, [2.0, 2.0, 3.0, 3.0]),
        ];
        let result = candidates(
            &records,
            [0.0, 0.0, 10.0, 10.0],
            0,
            &[[0.0, 0.0, 10.0, 10.0]],
            &[(1, false, None)],
            0.8,
        )
        .unwrap();
        assert_eq!(result.boxes.len(), 2);
        assert!(result.paints);
    }

    /// floor/ceil、空区域、边界像素与重复框都按原 Pillow 查询次数计数。
    #[test]
    fn fractional_pixels_are_counted_exactly() {
        let result = painted_count(
            &[0, 1, 0, 0],
            2,
            2,
            &[
                [0.51, 0.0, 1.01, 0.5],
                [0.0, 1.0, 0.5, 2.0],
                [0.51, 0.0, 1.01, 0.5],
            ],
            2.0,
            2.0,
        );
        assert_eq!(result, 2);
    }
}
