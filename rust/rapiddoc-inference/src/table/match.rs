//! TableMatch 的连续面积 IoU / 边缘距离匹配，保留原 OCR 索引。
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Matches {
    /// 每格按来源顺序保存 OCR 索引；未匹配的格保留空列表。
    pub cells: Vec<Vec<usize>>,
    pub unmatched: Vec<usize>,
}

fn invalid(message: &str) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.into(),
    }
}

/// 四值轴框保留其方向；八值四角归一为外接框。反序框拒绝，不猜测纠正。
pub fn bounds(values: &[f64]) -> Result<[f64; 4], Error> {
    if !matches!(values.len(), 4 | 8) || values.iter().any(|v| !v.is_finite() || v.abs() > 1e9) {
        return Err(invalid("表格匹配框必须为有限四值轴框或八值四角"));
    }
    let result = if values.len() == 4 {
        [values[0], values[1], values[2], values[3]]
    } else {
        let mut b = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for point in values.as_chunks::<2>().0 {
            b[0] = b[0].min(point[0]);
            b[1] = b[1].min(point[1]);
            b[2] = b[2].max(point[0]);
            b[3] = b[3].max(point[1]);
        }
        b
    };
    if result[0] > result[2] || result[1] > result[3] {
        return Err(invalid("表格匹配轴框反序"));
    }
    Ok(result)
}

/// 固定 min_iou=0.1^8；先比较 1-IoU，再比较边缘距离，完全相同时首格优先。
pub fn match_cells(cells: &[Vec<f64>], ocr: &[Vec<f64>]) -> Result<Matches, Error> {
    if cells.len() > rapiddoc_core::resources::limit(10_000)
        || ocr.len() > rapiddoc_core::resources::limit(1000)
        || cells.len().saturating_mul(ocr.len()) > rapiddoc_core::resources::limit(10_000_000)
    {
        return Err(Error {
            code: ErrorCode::ResourceLimit,
            message: "表格匹配超过数量或工作量预算".into(),
        });
    }
    let cells: Vec<_> = cells.iter().map(|b| bounds(b)).collect::<Result<_, _>>()?;
    let ocr: Vec<_> = ocr.iter().map(|b| bounds(b)).collect::<Result<_, _>>()?;
    let mut result = Matches {
        cells: vec![vec![]; cells.len()],
        unmatched: vec![],
    };
    let min_y = cells.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min);
    for (index, b) in ocr.iter().enumerate() {
        let mut best = None;
        let mut best_inverse = f64::INFINITY;
        let mut best_distance = f64::INFINITY;
        if b[3] >= min_y {
            for (cell_index, c) in cells.iter().enumerate() {
                let left = b[0].max(c[0]);
                let right = b[2].min(c[2]);
                let top = b[1].max(c[1]);
                let bottom = b[3].min(c[3]);
                let intersection = (right - left) * (bottom - top);
                let union =
                    (b[2] - b[0]) * (b[3] - b[1]) + (c[2] - c[0]) * (c[3] - c[1]) - intersection;
                let iou = if left < right && top < bottom && union != 0. {
                    intersection / union
                } else {
                    0.
                };
                let inverse = 1. - iou;
                let d0 = (c[0] - b[0]).abs();
                let d1 = (c[1] - b[1]).abs();
                let d2 = (c[2] - b[2]).abs();
                let d3 = (c[3] - b[3]).abs();
                let distance = d0 + d1 + d2 + d3 + (d0 + d1).min(d2 + d3);
                if inverse < best_inverse || (inverse == best_inverse && distance < best_distance) {
                    best = Some(cell_index);
                    best_inverse = inverse;
                    best_distance = distance;
                }
            }
        }
        if let Some(cell) = best.filter(|_| best_inverse < 1. - 0.1f64.powi(8)) {
            result.cells[cell].push(index);
        } else {
            result.unmatched.push(index);
        }
    }
    Ok(result)
}
