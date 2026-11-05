//! UNET 噪声边缘裁剪与空格占位；保持旧结构规则，输出文字始终转义。
use rapiddoc_core::{Error, ErrorCode};
pub type Polygon = [[f64; 2]; 4];
fn invalid(message: &str) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.into(),
    }
}
fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.
    })
}

/// 输入为已归并/清理的逐格行文字；网格重叠报错，内部缺位按旧规则输出空 td。
pub fn html(
    points: &[[usize; 4]],
    polygons: &[Polygon],
    texts: &[Vec<String>],
) -> Result<String, Error> {
    render(points, polygons, texts, true, true)
}

/// 旧 UNET 不再转义已转义文本；逻辑格重叠按原 Python 的后写覆盖处理。
pub fn legacy_html(
    points: &[[usize; 4]],
    polygons: &[Polygon],
    texts: &[Vec<String>],
) -> Result<String, Error> {
    render(points, polygons, texts, false, false)
}

fn render(
    points: &[[usize; 4]],
    polygons: &[Polygon],
    texts: &[Vec<String>],
    escape: bool,
    reject_overlaps: bool,
) -> Result<String, Error> {
    if points.len() != polygons.len() || points.len() != texts.len() {
        return Err(invalid("UNET 逻辑格/物理格/文字数量不匹配"));
    }
    if points.len() > rapiddoc_core::resources::limit(2000)
        || texts.iter().flatten().map(String::len).sum::<usize>()
            > rapiddoc_core::resources::limit(1_000_000)
    {
        return Err(Error {
            code: ErrorCode::ResourceLimit,
            message: "UNET HTML 输入超过预算".into(),
        });
    }
    if points.is_empty() {
        return Ok("<html><body><table></table></body></html>".into());
    }
    let whitespace = |c: char| matches!(c,'\u{0009}'..='\u{000d}'|'\u{001c}'..='\u{0020}'|'\u{0085}'|'\u{00a0}'|'\u{1680}'|'\u{2000}'..='\u{200a}'|'\u{2028}'|'\u{2029}'|'\u{202f}'|'\u{205f}'|'\u{3000}');
    let texts: Vec<Vec<String>> = texts
        .iter()
        .map(|lines| {
            lines
                .iter()
                .filter(|s| !s.trim_matches(whitespace).is_empty())
                .map(|s| {
                    s.replace(['\r', '\n'], "")
                        .trim_matches(whitespace)
                        .to_owned()
                })
                .collect()
        })
        .collect();
    if points.iter().any(|p| {
        p[0] > p[1]
            || (reject_overlaps && p[2] > p[3])
            || p[1] >= rapiddoc_core::resources::limit(1000)
            || p[3] >= rapiddoc_core::resources::limit(1000)
    }) || polygons
        .iter()
        .flatten()
        .flatten()
        .any(|v| !v.is_finite() || v.abs() > 1_000_000.)
    {
        return Err(invalid("UNET HTML 坐标或跨度无效"));
    }
    let rows = points
        .iter()
        .map(|p| p[1] + 1)
        .max()
        .ok_or_else(|| invalid("空格坐标"))?;
    let columns = points
        .iter()
        .map(|p| p[3] + 1)
        .max()
        .ok_or_else(|| invalid("空格坐标"))?;
    if rows * columns > rapiddoc_core::resources::limit(100_000) {
        return Err(Error {
            code: ErrorCode::ResourceLimit,
            message: "UNET HTML 网格超过预算".into(),
        });
    }
    let mut grid = vec![None; rows * columns];
    let mut row_sizes = vec![Vec::new(); rows];
    let mut column_sizes = vec![Vec::new(); columns];
    for (i, p) in points.iter().enumerate() {
        for r in p[0]..=p[1] {
            for c in p[2]..=p[3] {
                if grid[r * columns + c].replace(i).is_some() && reject_overlaps {
                    return Err(invalid("UNET 逻辑格重复覆盖"));
                }
            }
        }
        let poly = polygons[i];
        let x0 = poly.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        let y0 = poly.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
        let x1 = poly.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
        let y1 = poly.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
        // Python estimates reversed/empty column intervals with max(span, 1).
        let width = (x1 - x0) / (p[3] + 1).saturating_sub(p[2]).max(1) as f64;
        let height = (y1 - y0) / (p[1] - p[0] + 1) as f64;
        if width > 0. && p[2] <= p[3] {
            for sizes in &mut column_sizes[p[2]..=p[3]] {
                sizes.push(width);
            }
        }
        if height > 0. {
            for sizes in &mut row_sizes[p[0]..=p[1]] {
                sizes.push(height);
            }
        }
    }
    let row_sizes: Vec<_> = row_sizes.iter().map(|v| median(v)).collect();
    let column_sizes: Vec<_> = column_sizes.iter().map(|v| median(v)).collect();
    let visible: Vec<_> = texts
        .iter()
        .map(|lines| lines.iter().any(|s| !s.is_empty()))
        .collect();
    let mut range = [0, rows, 0, columns]; // 右端开区间，便于保留“全被裁掉”的状态。
    for edge in 0..4 {
        while if edge < 2 {
            range[0] < range[1]
        } else {
            range[2] < range[3]
        } {
            let axis = if edge < 2 { 0 } else { 1 };
            let index = match edge {
                0 => range[0],
                1 => range[1] - 1,
                2 => range[2],
                _ => range[3] - 1,
            };
            let positions: Vec<_> = if axis == 0 {
                (range[2]..range[3])
                    .map(|c| grid[index * columns + c])
                    .collect()
            } else {
                (range[0]..range[1])
                    .map(|r| grid[r * columns + index])
                    .collect()
            };
            if positions.iter().flatten().any(|&i| visible[i]) {
                break;
            }
            let covered = positions.iter().flatten().count();
            let total = positions.len();
            let sizes = if axis == 0 { &row_sizes } else { &column_sizes };
            let others: Vec<_> = sizes
                .iter()
                .enumerate()
                .filter_map(|(i, v)| if i != index { *v } else { None })
                .collect();
            let abnormal = match (sizes[index], median(&others)) {
                (Some(size), Some(reference)) if reference > 0. => {
                    let ratio = size / reference;
                    !(0.35..=2.5).contains(&ratio)
                }
                _ => false,
            };
            if covered != 0 && covered == total && !abnormal {
                break;
            }
            match edge {
                0 => range[0] += 1,
                1 => range[1] -= 1,
                2 => range[2] += 1,
                _ => range[3] -= 1,
            }
        }
    }
    let mut output = String::from("<html><body><table>");
    if range[0] < range[1] && range[2] < range[3] {
        for r in range[0]..range[1] {
            output.push_str("<tr>");
            for c in range[2]..range[3] {
                if let Some(i) = grid[r * columns + c] {
                    let p = points[i];
                    let top = p[0].max(range[0]);
                    let left = p[2].max(range[2]);
                    if r == top && c == left {
                        let rs = p[1].min(range[1] - 1) - top + 1;
                        let cs = p[3].min(range[3] - 1) - left + 1;
                        output.push_str(&format!("<td rowspan={rs} colspan={cs}>"));
                        output.push_str(
                            &texts[i]
                                .iter()
                                .map(|s| {
                                    if escape {
                                        crate::escape_html(s)
                                    } else {
                                        s.clone()
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("<br>"),
                        );
                        output.push_str("</td>");
                    }
                } else {
                    output.push_str("<td></td>");
                }
            }
            output.push_str("</tr>");
        }
    }
    output.push_str("</table></body></html>");
    Ok(output)
}
