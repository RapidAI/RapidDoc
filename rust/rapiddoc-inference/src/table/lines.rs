//! UNET adjust_lines/final_adjust_lines 的显式线段连接；不绘图或恢复 mask 连通域。
use crate::error;
use rapiddoc_core::{Error, ErrorCode};
use serde::{Deserialize, Serialize};
pub type Line = [f32; 4];

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub row_gap: f64,
    pub column_gap: f64,
    pub angle: f64,
    pub more_rows: bool,
    pub more_columns: bool,
    pub extend: bool,
    pub max_lines: usize,
    pub max_work: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            row_gap: 100.,
            column_gap: 15.,
            angle: 50.,
            more_rows: true,
            more_columns: true,
            extend: true,
            max_lines: 2000,
            max_work: 10_000_000,
        }
    }
}
#[derive(Debug, Serialize)]
pub struct Lines {
    pub rows: Vec<Line>,
    pub columns: Vec<Line>,
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    let x = a[0] - b[0];
    let y = a[1] - b[1];
    (x * x + y * y).sqrt()
}
fn slope_angle(a: [f32; 2], b: [f32; 2]) -> f64 {
    (((b[1] - a[1]) / (b[0] - a[0] + 1e-10)).abs() as f64).atan() * 180. / std::f64::consts::PI
}
fn fit(p: Line) -> [f32; 3] {
    [p[3] - p[1], p[0] - p[2], p[2] * p[1] - p[0] * p[3]]
}

/// 先依据原始线补连接线，再按行→列顺序原位延长，顺序与旧实现保持一致。
pub fn connect(
    mut rows: Vec<Line>,
    mut columns: Vec<Line>,
    config: &Config,
) -> Result<Lines, Error> {
    if rapiddoc_core::resources::limit(config.max_lines) == 0
        || rapiddoc_core::resources::limit(config.max_lines)
            > rapiddoc_core::resources::limit(10_000)
        || rapiddoc_core::resources::limit(config.max_work)
            > rapiddoc_core::resources::limit(50_000_000)
        || !config.row_gap.is_finite()
        || !config.column_gap.is_finite()
        || !config.angle.is_finite()
        || !(0. ..=1_000_000.).contains(&config.row_gap)
        || !(0. ..=1_000_000.).contains(&config.column_gap)
        || !(0. ..=90.).contains(&config.angle)
        || rows
            .iter()
            .chain(&columns)
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > rapiddoc_core::resources::limit(1_000_000.))
    {
        return Err(error(ErrorCode::InvalidInput, "UNET 线段或连接配置无效"));
    }
    if rows.len().saturating_add(columns.len()) > rapiddoc_core::resources::limit(config.max_lines)
    {
        return Err(error(ErrorCode::ResourceLimit, "UNET 线段超过预算"));
    }
    let mut remaining = rapiddoc_core::resources::limit(config.max_work);
    let mut row_add = if config.more_rows {
        adjust(
            &rows,
            config.row_gap,
            config.angle,
            rapiddoc_core::resources::limit(config.max_lines),
            &mut remaining,
        )?
    } else {
        Vec::new()
    };
    let mut col_add = if config.more_columns {
        adjust(
            &columns,
            config.column_gap,
            config.angle,
            rapiddoc_core::resources::limit(config.max_lines),
            &mut remaining,
        )?
    } else {
        Vec::new()
    };
    if rows.len() + columns.len() + row_add.len() + col_add.len()
        > rapiddoc_core::resources::limit(config.max_lines)
    {
        return Err(error(ErrorCode::ResourceLimit, "UNET 连接后线段超过预算"));
    }
    rows.append(&mut row_add);
    columns.append(&mut col_add);
    if config.extend {
        let work = rows
            .len()
            .checked_mul(columns.len())
            .ok_or_else(|| error(ErrorCode::ResourceLimit, "UNET 交点工作量溢出"))?;
        consume(&mut remaining, work)?;
        for row in &mut rows {
            for column in &mut columns {
                *row = extend(*row, *column);
                *column = extend(*column, *row);
            }
        }
    }
    Ok(Lines { rows, columns })
}

fn consume(remaining: &mut usize, count: usize) -> Result<(), Error> {
    *remaining = remaining
        .checked_sub(count)
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "UNET 线段连接工作量超过预算"))?;
    Ok(())
}

fn adjust(
    lines: &[Line],
    gap: f64,
    angle: f64,
    max_lines: usize,
    remaining: &mut usize,
) -> Result<Vec<Line>, Error> {
    let max_lines = rapiddoc_core::resources::limit(max_lines);
    consume(remaining, lines.len().saturating_mul(lines.len()))?;
    let mut result = Vec::new();
    for (i, a) in lines.iter().enumerate() {
        let cx1 = (a[0] + a[2]) / 2.;
        let cy1 = (a[1] + a[3]) / 2.;
        for (j, b) in lines.iter().enumerate() {
            if i == j {
                continue;
            }
            let cx2 = (b[0] + b[2]) / 2.;
            let cy2 = (b[1] + b[3]) / 2.;
            if (b[0] < cx1 && cx1 < b[2])
                || (b[1] < cy1 && cy1 < b[3])
                || (a[0] < cx2 && cx2 < a[2])
                || (a[1] < cy2 && cy2 < a[3])
            {
                continue;
            }
            for (p, q) in [
                ([a[0], a[1]], [b[0], b[1]]),
                ([a[0], a[1]], [b[2], b[3]]),
                ([a[2], a[3]], [b[0], b[1]]),
                ([a[2], a[3]], [b[2], b[3]]),
            ] {
                if (distance(p, q) as f64) < gap && slope_angle(p, q) < angle {
                    if result.len() >= max_lines {
                        return Err(error(ErrorCode::ResourceLimit, "UNET 补线超过预算"));
                    }
                    result.push([p[0], p[1], q[0], q[1]]);
                }
            }
        }
    }
    Ok(result)
}

fn extend(line: Line, other: Line) -> Line {
    let [a1, b1, c1] = fit(line);
    let [a2, b2, c2] = fit(other);
    let flag1 = a2 * line[0] + b2 * line[1] + c2;
    let flag2 = a2 * line[2] + b2 * line[3] + c2;
    if !((flag1 > 0. && flag2 > 0.) || (flag1 < 0. && flag2 < 0.)) {
        return line;
    }
    let determinant = a1 * b2 - a2 * b1;
    if determinant == 0. {
        return line;
    }
    let p = [
        (b1 * c2 - b2 * c1) / determinant,
        (a2 * c1 - a1 * c2) / determinant,
    ];
    let p0 = [line[0], line[1]];
    let p1 = [line[2], line[3]];
    let r0 = distance(p, p0);
    let r1 = distance(p, p1);
    if r0.min(r1) >= 20. {
        return line;
    }
    let angle = slope_angle(p, if r0 < r1 { p1 } else { p0 });
    if angle >= 30. && (90. - angle).abs() >= 30. {
        return line;
    }
    if r0 < r1 {
        [p[0], p[1], line[2], line[3]]
    } else {
        [line[0], line[1], p[0], p[1]]
    }
}
