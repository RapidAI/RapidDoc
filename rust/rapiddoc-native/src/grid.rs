//! 已确认物理网格的单元格拓扑；候选仲裁和轨道发现由整页规则层负责。

use crate::{cell_regions, error};
use rapiddoc_core::{Error, ErrorCode, Rect};
use serde::Serialize;
use std::collections::HashSet;

/// 页面视觉坐标中的水平或竖直分隔线。
#[derive(Debug, Clone, Copy)]
pub struct Separator {
    pub vertical: bool,
    pub coordinate: f64,
    pub start: f64,
    pub end: f64,
}

/// 原字符来源与视觉框；传入顺序可以乱序，输出仍按 index 保持来源顺序。
#[derive(Debug, Clone)]
pub struct Glyph {
    pub index: usize,
    pub bounds: Rect,
    pub text: String,
    pub generated: bool,
}

/// 物理表格单元格，保留来源索引以便输出层防止正文重复。
#[derive(Debug, Clone, Serialize)]
pub struct Cell {
    pub row: usize,
    pub column: usize,
    pub row_span: usize,
    pub column_span: usize,
    pub text: String,
    pub character_indices: Vec<usize>,
}

/// 已确认轨道的表格，坐标为左上原点 PDF point。
#[derive(Debug, Clone, Serialize)]
pub struct Table {
    pub bounds: Rect,
    pub rows: usize,
    pub columns: usize,
    pub cells: Vec<Cell>,
}

/// 与表格同时保留实测物理轨道，供不均匀行列的页面几何输出使用。
#[derive(Debug, Serialize)]
pub struct RecoveredGrid {
    pub table: Table,
    pub x_tracks: Vec<f64>,
    pub y_tracks: Vec<f64>,
    pub coverage_work: usize,
}
pub struct TrackPlan {
    pub x_tracks: Vec<f64>,
    pub y_tracks: Vec<f64>,
    pub lines: Vec<Separator>,
    pub coverage_work: usize,
}

/// 网格组装限制与覆盖阈值；阈值必须在迁移差分后冻结，不能由失败动态放宽。
#[derive(Debug, Clone)]
pub struct GridConfig {
    pub max_cells: usize,
    pub max_separators: usize,
    pub tolerance: f64,
    pub minimum_coverage: f64,
    pub max_coverage_work: usize,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            max_cells: 100_000,
            max_separators: 100_000,
            tolerance: 1.,
            minimum_coverage: 0.9,
            max_coverage_work: 50_000_000,
        }
    }
}

/// 在调用方已指定的表格区域内发现轴线轨道，再交给物理网格组装器。
/// 这不是整页表格检测器，不对没有外边界的区域猜测单元格。
pub fn recover(
    region: Rect,
    lines: &[Separator],
    glyphs: Vec<Glyph>,
    config: &GridConfig,
) -> Result<Table, Error> {
    Ok(recover_with_tracks(region, lines, glyphs, config)?.table)
}

pub fn recover_with_tracks(
    region: Rect,
    lines: &[Separator],
    glyphs: Vec<Glyph>,
    config: &GridConfig,
) -> Result<RecoveredGrid, Error> {
    let plan = plan_tracks(region, lines, config)?;
    let table = assemble(&plan.x_tracks, &plan.y_tracks, &plan.lines, glyphs, config)?;
    Ok(RecoveredGrid {
        table,
        x_tracks: plan.x_tracks,
        y_tracks: plan.y_tracks,
        coverage_work: plan.coverage_work,
    })
}

/// 先预算轨道/覆盖查询，再做拓扑，失败的候选同样需要计入调用方累计预算。
pub fn plan_tracks(
    region: Rect,
    lines: &[Separator],
    config: &GridConfig,
) -> Result<TrackPlan, Error> {
    if lines.len() > rapiddoc_core::resources::limit(config.max_separators) {
        return Err(error(ErrorCode::ResourceLimit, "分隔线超过限制"));
    }
    if !config.tolerance.is_finite() || config.tolerance < 0. {
        return Err(error(ErrorCode::InvalidInput, "无效轨道容差"));
    }
    let [left, top, right, bottom] = region.into();
    if right <= left || bottom <= top {
        return Err(error(ErrorCode::InvalidInput, "表格区域必须有正面积"));
    }
    let mut selected = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for line in lines {
        if ![line.coordinate, line.start, line.end]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(error(ErrorCode::InvalidInput, "分隔线几何无效"));
        }
        let (low, high, first, last) = if line.vertical {
            (left, right, top, bottom)
        } else {
            (top, bottom, left, right)
        };
        if line.coordinate < low - config.tolerance || line.coordinate > high + config.tolerance {
            continue;
        }
        let start = line.start.min(line.end).max(first);
        let end = line.start.max(line.end).min(last);
        if end <= start {
            continue;
        }
        selected.push(Separator {
            vertical: line.vertical,
            coordinate: line.coordinate,
            start,
            end,
        });
        if line.vertical {
            xs.push(line.coordinate);
        } else {
            ys.push(line.coordinate);
        }
    }
    let xs = tracks(xs, config.tolerance);
    let ys = tracks(ys, config.tolerance);
    let cells = xs
        .len()
        .saturating_sub(1)
        .checked_mul(ys.len().saturating_sub(1))
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "网格尺寸溢出"))?;
    let coverage_work = cells
        .checked_mul(2)
        .and_then(|n| n.checked_mul(selected.len()))
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "覆盖查询预算溢出"))?;
    if cells > rapiddoc_core::resources::limit(config.max_cells)
        || coverage_work > rapiddoc_core::resources::limit(config.max_coverage_work)
    {
        return Err(error(ErrorCode::ResourceLimit, "网格或覆盖查询超过预算"));
    }
    Ok(TrackPlan {
        x_tracks: xs,
        y_tracks: ys,
        lines: selected,
        coverage_work,
    })
}

/// 按簇首值限制总跨度，避免链式相邻把多个物理轨道合为一列。
fn tracks(mut values: Vec<f64>, tolerance: f64) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    let mut output = Vec::new();
    let mut index = 0;
    while index < values.len() {
        let first = values[index];
        let mut end = index + 1;
        while end < values.len() && values[end] - first <= tolerance {
            end += 1;
        }
        output.push(values[index..end].iter().sum::<f64>() / (end - index) as f64);
        index = end;
    }
    output
}

/// 组装已知轨道网格。外边界不完整、L 形合并、来源重复均明确拒绝。
pub fn assemble(
    xs: &[f64],
    ys: &[f64],
    lines: &[Separator],
    mut glyphs: Vec<Glyph>,
    config: &GridConfig,
) -> Result<Table, Error> {
    for tracks in [xs, ys] {
        if tracks.len() < 2
            || tracks.iter().any(|v| !v.is_finite())
            || tracks.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(error(ErrorCode::InvalidInput, "轨道必须有限且严格递增"));
        }
    }
    if !config.tolerance.is_finite()
        || config.tolerance < 0.
        || !config.minimum_coverage.is_finite()
        || config.minimum_coverage <= 0.
        || config.minimum_coverage > 1.
    {
        return Err(error(ErrorCode::InvalidInput, "无效覆盖阈值"));
    }
    let rows = ys.len() - 1;
    let columns = xs.len() - 1;
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "网格尺寸溢出"))?;
    if count > rapiddoc_core::resources::limit(config.max_cells)
        || lines.len() > rapiddoc_core::resources::limit(config.max_separators)
    {
        return Err(error(ErrorCode::ResourceLimit, "网格超过预算"));
    }
    let work = count
        .checked_mul(2)
        .and_then(|value| value.checked_mul(lines.len()))
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "覆盖查询预算溢出"))?;
    if work > rapiddoc_core::resources::limit(config.max_coverage_work) {
        return Err(error(ErrorCode::ResourceLimit, "覆盖查询超过计算预算"));
    }
    if lines.iter().any(|line| {
        ![line.coordinate, line.start, line.end]
            .iter()
            .all(|v| v.is_finite())
    }) {
        return Err(error(ErrorCode::InvalidInput, "分隔线包含非有限值"));
    }
    // 整批查询共享内核的轨道缓存，避免逐单元格复制和重新过滤所有线段。
    let mut queries = Vec::new();
    let mut connections = Vec::new();
    for (vertical, coordinate, start, end) in [
        (true, xs[0], ys[0], ys[rows]),
        (true, xs[columns], ys[0], ys[rows]),
        (false, ys[0], xs[0], xs[columns]),
        (false, ys[rows], xs[0], xs[columns]),
    ] {
        queries.push((
            u8::from(vertical),
            vec![coordinate],
            start,
            end,
            config.tolerance,
        ));
    }
    for row in 0..rows {
        for column in 0..columns - 1 {
            queries.push((
                1,
                vec![xs[column + 1]],
                ys[row],
                ys[row + 1],
                config.tolerance,
            ));
            connections.push((row * columns + column, row * columns + column + 1));
        }
    }
    for row in 0..rows - 1 {
        for column in 0..columns {
            queries.push((
                0,
                vec![ys[row + 1]],
                xs[column],
                xs[column + 1],
                config.tolerance,
            ));
            connections.push((row * columns + column, (row + 1) * columns + column));
        }
    }
    let rules = lines
        .iter()
        .map(|line| {
            (
                u8::from(line.vertical),
                line.coordinate,
                line.start,
                line.end,
            )
        })
        .collect();
    let coverages = docvortex_core::tables::coverage_batch(rules, queries)
        .ok_or_else(|| error(ErrorCode::InvalidInput, "分隔线查询无效"))?;
    if coverages[..4]
        .iter()
        .any(|value| *value < config.minimum_coverage)
    {
        return Err(error(
            ErrorCode::InvalidInput,
            "外边界不完整，不能认领物理表格",
        ));
    }
    let missing = connections
        .into_iter()
        .zip(&coverages[4..])
        .filter_map(|(edge, coverage)| (*coverage < config.minimum_coverage).then_some(edge))
        .collect();
    let regions = cell_regions(
        rows,
        columns,
        missing,
        rapiddoc_core::resources::limit(config.max_cells),
    )?;
    let mut cells: Vec<Cell> = regions
        .into_iter()
        .map(|region| Cell {
            row: region.row,
            column: region.column,
            row_span: region.row_span,
            column_span: region.column_span,
            text: String::new(),
            character_indices: Vec::new(),
        })
        .collect();
    let mut owner = vec![0usize; count];
    for (index, cell) in cells.iter().enumerate() {
        for row in cell.row..cell.row + cell.row_span {
            for column in cell.column..cell.column + cell.column_span {
                owner[row * columns + column] = index;
            }
        }
    }
    let mut seen = HashSet::new();
    glyphs.sort_by_key(|glyph| glyph.index);
    let mut assignments = Vec::with_capacity(glyphs.len());
    for glyph in &glyphs {
        if !seen.insert(glyph.index) {
            return Err(error(ErrorCode::InvalidInput, "字符来源索引重复"));
        }
        let [left, top, right, bottom] = glyph.bounds.into();
        let x = left / 2. + right / 2.;
        let y = top / 2. + bottom / 2.;
        if x < xs[0] || x > xs[columns] || y < ys[0] || y > ys[rows] {
            assignments.push(None);
            continue;
        }
        let column = xs
            .partition_point(|value| *value <= x)
            .saturating_sub(1)
            .min(columns - 1);
        let row = ys
            .partition_point(|value| *value <= y)
            .saturating_sub(1)
            .min(rows - 1);
        assignments.push(Some(owner[row * columns + column]));
    }
    // 一次正向、一次反向准备邻居，长空白串也不退化为逐字符回扫。
    let mut before = Vec::with_capacity(glyphs.len());
    let mut previous = None;
    for (glyph, assignment) in glyphs.iter().zip(&assignments) {
        before.push(previous);
        if !glyph.text.chars().all(char::is_whitespace) {
            previous = *assignment;
        }
    }
    let mut after = vec![None; glyphs.len()];
    let mut next = None;
    for index in (0..glyphs.len()).rev() {
        after[index] = next;
        if !glyphs[index].text.chars().all(char::is_whitespace) {
            next = assignments[index];
        }
    }
    for (index, glyph) in glyphs.into_iter().enumerate() {
        let assignment = if glyph.generated && glyph.text.chars().all(char::is_whitespace) {
            if before[index] == after[index] {
                before[index]
            } else {
                None
            }
        } else {
            assignments[index]
        };
        let Some(assignment) = assignment else {
            continue;
        };
        let cell = &mut cells[assignment];
        cell.text.push_str(&glyph.text);
        cell.character_indices.push(glyph.index);
    }
    Ok(Table {
        bounds: Rect::try_from([xs[0], ys[0], xs[columns], ys[rows]])?,
        rows,
        columns,
        cells,
    })
}
