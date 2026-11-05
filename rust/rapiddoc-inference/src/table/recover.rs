//! UNET TableRecover：按 Python 的顺序顶点和轨道规则恢复逻辑格，保留输入索引。
//! 数值规则依据 RapidDoc table_structure/unet/table_recover.py，见 PADDLE-TABLE 许可。
use crate::error;
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;

pub type Polygon = [[f64; 2]; 4];

#[derive(Debug, Serialize)]
pub struct Topology {
    pub rows: usize,
    pub columns: usize,
    /// 每个源框对应 [row_start, row_end, column_start, column_end]，闭区间。
    pub logic_points: Vec<[usize; 4]>,
}

fn length(a: [f64; 2], b: [f64; 2]) -> f64 {
    let x = b[0] - a[0];
    let y = b[1] - a[1];
    (x * x + y * y).sqrt()
}

/// 与 Python 一样只比较相邻 y，不重排顶点或重新聚类。
/// 保留倾斜/退化格框、有符号轨道间距及重叠列产生的空区间；
/// 空区间不是数组索引，仍保留源格框与 OCR 对应关系。
pub fn recover(
    polygons: &[Polygon],
    row_threshold: f64,
    column_threshold: f64,
) -> Result<Topology, Error> {
    if polygons.len() > rapiddoc_core::resources::limit(2000) {
        return Err(error(ErrorCode::ResourceLimit, "UNET 单元格数超过 2000"));
    }
    if polygons.is_empty()
        || !row_threshold.is_finite()
        || !column_threshold.is_finite()
        || !(0. ..=1_000_000.).contains(&row_threshold)
        || !(0. ..=1_000_000.).contains(&column_threshold)
        || polygons
            .iter()
            .flatten()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > rapiddoc_core::resources::limit(1_000_000.))
    {
        return Err(error(ErrorCode::InvalidInput, "UNET 格框或聚类阈值无效"));
    }
    let mut rows: Vec<Vec<usize>> = vec![vec![0]];
    for i in 1..polygons.len() {
        if (polygons[i][0][1] - polygons[i - 1][0][1]).abs() > row_threshold {
            rows.push(Vec::new());
        }
        if let Some(row) = rows.last_mut() {
            row.push(i);
        }
    }
    let mut longest = &rows[0];
    for row in &rows[1..] {
        if row.len() > longest.len() {
            longest = row;
        }
    }
    let mut starts: Vec<_> = longest.iter().map(|&i| polygons[i][0][0]).collect();
    let mut min = starts[0];
    let mut max = polygons[*longest
        .last()
        .ok_or_else(|| error(ErrorCode::InvalidInput, "空基准行"))?][2][0];
    for row in &rows {
        for &i in row {
            update_columns(
                &mut starts,
                polygons[i][0][0],
                &mut min,
                &mut max,
                column_threshold,
                true,
            );
            update_columns(
                &mut starts,
                polygons[i][2][0],
                &mut min,
                &mut max,
                column_threshold,
                false,
            );
        }
    }
    if rows.len() > rapiddoc_core::resources::limit(1000)
        || starts.len() > rapiddoc_core::resources::limit(1000)
    {
        return Err(error(ErrorCode::ResourceLimit, "UNET 逻辑行列数超过限制"));
    }
    let mut widths: Vec<_> = starts.windows(2).map(|p| p[1] - p[0]).collect();
    widths.push(max - starts[starts.len() - 1]);
    let mut heights: Vec<_> = rows
        .windows(2)
        .map(|r| polygons[r[1][0]][0][1] - polygons[r[0][0]][0][1])
        .collect();
    let last_height = rows[rows.len() - 1]
        .iter()
        .map(|&i| length(polygons[i][1], polygons[i][0]))
        .fold(0., f64::max);
    heights.push(last_height);
    if widths.iter().chain(&heights).any(|v| *v <= 0.) {
        // Python sums these signed track distances; it neither divides by
        // them nor reorders the source cells. Production must retain that
        // behavior for overlapping columns and loosely sorted detector rows.
        if let Some(folder) = std::env::var_os("RAPIDDOC_UNET_TRACK_TRACE") {
            let folder = std::path::PathBuf::from(folder);
            std::fs::create_dir_all(&folder).map_err(|e| error(ErrorCode::InvalidInput, e))?;
            let data = serde_json::json!({"polygons":polygons,"row_threshold":row_threshold,
                "column_threshold":column_threshold,"widths":widths,"heights":heights});
            // Preserve the first trigger as a reproducible diagnostic fixture.
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(folder.join("tracks.json"))
            {
                use std::io::Write;
                file.write_all(serde_json::to_vec(&data).unwrap().as_slice())
                    .map_err(|e| error(ErrorCode::InvalidInput, e))?;
            }
        }
    }
    let mut points = vec![[0; 4]; polygons.len()];
    for (row_index, row) in rows.iter().enumerate() {
        let mut used = 0;
        for &i in row {
            let p = &polygons[i];
            let mut nearest = 0;
            for j in 1..starts.len() {
                if (starts[j] - p[0][0]).abs() < (starts[nearest] - p[0][0]).abs() {
                    nearest = j;
                }
            }
            // 兼容旧 one_col_result.values() 的跨度总和，并非上一格的绝对终点。
            let column = used.max(nearest);
            if column >= widths.len() {
                let row_span = span(&heights, row_index, length(p[1], p[0]))?;
                points[i] = [
                    row_index,
                    row_index + row_span - 1,
                    column,
                    widths.len() - 1,
                ];
                // Python adds col_nums - col_start, which may be negative.
                used = widths.len();
                continue;
            }
            let column_span = span(&widths, column, length(p[3], p[0]))?;
            let row_span = span(&heights, row_index, length(p[1], p[0]))?;
            points[i] = [
                row_index,
                row_index + row_span - 1,
                column,
                column + column_span - 1,
            ];
            used += column_span;
        }
    }
    Ok(Topology {
        rows: rows.len(),
        columns: starts.len(),
        logic_points: points,
    })
}

fn update_columns(
    starts: &mut Vec<f64>,
    current: f64,
    min: &mut f64,
    max: &mut f64,
    threshold: f64,
    insert_last: bool,
) {
    for i in 0..starts.len() {
        let value = starts[i];
        if current - threshold <= value && value <= current + threshold {
            break;
        }
        if current < *min {
            starts.insert(0, current);
            *min = current;
            break;
        }
        if current > *max {
            if insert_last {
                starts.push(current);
            }
            *max = current;
            break;
        }
        if current < value {
            starts.insert(i, current);
            break;
        }
    }
}

fn span(sizes: &[f64], start: usize, extent: f64) -> Result<usize, Error> {
    if start >= sizes.len() {
        return Err(error(ErrorCode::InvalidInput, "UNET 合并格起点越界"));
    }
    let mut sum = 0.;
    for (i, &size) in sizes.iter().enumerate().skip(start) {
        sum += size;
        if i == start && sum > extent {
            return Ok(1);
        }
        if (sum - extent).abs() <= 10. {
            return Ok(i + 1 - start);
        }
        if sum > extent {
            let end = if (sum - extent).abs() < (sum - size - extent).abs() {
                i + 1
            } else {
                i
            };
            if end <= start {
                return Err(error(ErrorCode::InvalidInput, "UNET 合并跨度为空"));
            }
            return Ok(end - start);
        }
    }
    Ok(sizes.len() - start)
}
