//! 原生规则的校验适配层；复用固定来源内核，禁止未校验索引进入并查集。

use docvortex_core::{median, quantile};
use rapiddoc_core::{Error, ErrorCode};
pub mod bar_geometry;
pub mod geometry_plan;
pub mod inline_evidence;
pub mod raster_charts;
pub mod text_noise;
pub mod visual_text;

/// Exact upper-quartile body-height queries shared with the upstream kernel.
pub use docvortex_core::note_index::HeightIndex as TableNoteHeightIndex;
pub use docvortex_core::row_geometry::RowGeometry as TableNoteRowGeometry;

pub type NativeLaneGeometry = docvortex_core::rule_text::LaneGeometry;
pub type NativeTailLine = docvortex_core::rule_text::TailLine;
fn lane_finite(n: f64) -> bool {
    n.is_finite() && n.abs() <= 1e15
}
fn validate_lane_geometry(lines: &[NativeLaneGeometry]) -> Result<(), Error> {
    if lines.is_empty() || lines.len() > rapiddoc_core::resources::limit(512) {
        return Err(error(ErrorCode::ResourceLimit, "栏带输入行数超限"));
    }
    if lines.iter().any(|l| {
        l.0.iter().any(|&n| !lane_finite(n))
            || l.0[2] <= l.0[0]
            || l.0[3] <= l.0[1]
            || !lane_finite(l.1)
            || l.1 <= 0.
    }) {
        return Err(error(ErrorCode::InvalidInput, "栏带几何无效"));
    }
    Ok(())
}
fn validate_lane_bounds(lanes: &[(f64, f64)]) -> Result<(), Error> {
    if lanes.is_empty() || lanes.len() > rapiddoc_core::resources::limit(512) {
        return Err(error(ErrorCode::ResourceLimit, "栏带数量超限"));
    }
    if lanes
        .iter()
        .any(|&(l, r)| !lane_finite(l) || !lane_finite(r))
    {
        return Err(error(ErrorCode::InvalidInput, "栏带边界无效"));
    }
    Ok(())
}
/// 校验完成后借用普通几何调用固定内核，不接收Python指针。
pub fn text_supported_lane_intervals(
    lines: &[NativeLaneGeometry],
    width: f64,
    median: f64,
) -> Result<Vec<(f64, f64, usize)>, Error> {
    validate_lane_geometry(lines)?;
    if !lane_finite(width) || width <= 0. || !lane_finite(median) || median <= 0. {
        return Err(error(ErrorCode::InvalidInput, "栏带尺度无效"));
    }
    Ok(docvortex_core::rule_text::supported_lane_intervals(
        lines, width, median,
    ))
}
pub fn text_lane_assignments(
    lines: &[NativeLaneGeometry],
    lanes: &[(f64, f64)],
    tolerance: f64,
    nested: Option<(f64, f64)>,
) -> Result<Vec<Option<usize>>, Error> {
    validate_lane_geometry(lines)?;
    validate_lane_bounds(lanes)?;
    if !lane_finite(tolerance)
        || tolerance < 0.
        || nested.is_some_and(|(t, b)| !lane_finite(t) || !lane_finite(b) || b < t)
    {
        return Err(error(ErrorCode::InvalidInput, "栏带分配参数无效"));
    }
    Ok(docvortex_core::rule_text::lane_assignments(
        lines, lanes, tolerance, nested,
    ))
}
pub fn text_short_tail_destinations(
    lines: &[NativeTailLine],
    lanes: &[(f64, f64)],
    median: f64,
) -> Result<Vec<(usize, usize)>, Error> {
    validate_lane_bounds(lanes)?;
    if lines.len() > rapiddoc_core::resources::limit(512) {
        return Err(error(ErrorCode::ResourceLimit, "短尾行数超限"));
    }
    if !lane_finite(median)
        || median <= 0.
        || lines.iter().any(|l| {
            l.0.iter().any(|&n| !lane_finite(n))
                || !lane_finite(l.1)
                || l.1 <= 0.
                || !lane_finite(l.3)
                || l.4.is_some_and(|n| !n.is_finite())
                || l.5.is_some_and(|n| !lane_finite(n))
                || l.9 >= lanes.len()
        })
    {
        return Err(error(ErrorCode::InvalidInput, "短尾快照无效"));
    }
    Ok(docvortex_core::rule_text::short_tail_destinations(
        lines, lanes, median,
    ))
}
// The Python formula fallback permits finite degenerate/inverted rectangles.
// Keep the public strict adapters above unchanged; replay uses the same kernels.
fn validate_formula_lanes(lines: &[NativeLaneGeometry]) -> Result<(), Error> {
    if lines.is_empty() || lines.len() > rapiddoc_core::resources::limit(4096) {
        return Err(error(ErrorCode::ResourceLimit, "公式栏带行数超限"));
    }
    if lines
        .iter()
        .any(|l| l.0.iter().any(|&n| !lane_finite(n)) || !lane_finite(l.1) || l.1 <= 0.)
    {
        return Err(error(ErrorCode::InvalidInput, "公式栏带几何无效"));
    }
    Ok(())
}
fn validate_formula_bounds(lanes: &[(f64, f64)]) -> Result<(), Error> {
    if lanes.is_empty() || lanes.len() > rapiddoc_core::resources::limit(4096) {
        return Err(error(ErrorCode::ResourceLimit, "公式栏带数量超限"));
    }
    if lanes
        .iter()
        .any(|&(l, r)| !lane_finite(l) || !lane_finite(r))
    {
        return Err(error(ErrorCode::InvalidInput, "公式栏带边界无效"));
    }
    Ok(())
}
pub fn text_formula_lane_intervals(
    lines: &[NativeLaneGeometry],
    width: f64,
    median: f64,
) -> Result<Vec<(f64, f64, usize)>, Error> {
    validate_formula_lanes(lines)?;
    if !lane_finite(width) || width <= 0. || !lane_finite(median) || median <= 0. {
        return Err(error(ErrorCode::InvalidInput, "公式栏带尺度无效"));
    }
    Ok(docvortex_core::rule_text::supported_lane_intervals(
        lines, width, median,
    ))
}
pub fn text_formula_lane_assignments(
    lines: &[NativeLaneGeometry],
    lanes: &[(f64, f64)],
    tolerance: f64,
    nested: Option<(f64, f64)>,
) -> Result<Vec<Option<usize>>, Error> {
    validate_formula_lanes(lines)?;
    validate_formula_bounds(lanes)?;
    if !lane_finite(tolerance)
        || tolerance < 0.
        || nested.is_some_and(|(t, b)| !lane_finite(t) || !lane_finite(b) || b < t)
    {
        return Err(error(ErrorCode::InvalidInput, "公式栏带分配参数无效"));
    }
    Ok(docvortex_core::rule_text::lane_assignments(
        lines, lanes, tolerance, nested,
    ))
}
pub fn text_formula_tail_destinations(
    lines: &[NativeTailLine],
    lanes: &[(f64, f64)],
    median: f64,
) -> Result<Vec<(usize, usize)>, Error> {
    validate_formula_bounds(lanes)?;
    if lines.len() > rapiddoc_core::resources::limit(4096) {
        return Err(error(ErrorCode::ResourceLimit, "公式短尾行数超限"));
    }
    if !lane_finite(median)
        || median <= 0.
        || lines.iter().any(|l| {
            l.0.iter().any(|&n| !lane_finite(n))
                || !lane_finite(l.1)
                || l.1 <= 0.
                || !lane_finite(l.3)
                || l.4.is_some_and(|n| !n.is_finite())
                || l.5.is_some_and(|n| !lane_finite(n))
                || l.9 >= lanes.len()
        })
    {
        return Err(error(ErrorCode::InvalidInput, "公式短尾快照无效"));
    }
    Ok(docvortex_core::rule_text::short_tail_destinations(
        lines, lanes, median,
    ))
}
pub mod grid;
pub mod lines;
pub mod script_geometry;
pub type NativeGridIndex = (Vec<f64>, Vec<f64>, Vec<Vec<usize>>);
pub type NativeRule = (u8, f64, f64, f64);
/// statistics.mean的有限[0.5,1]子域：精确二进制和/整数分母，一次ties-to-even舍入。
pub fn table_decisiveness_mean(values: &[f64]) -> Result<f64, Error> {
    if values.is_empty()
        || values.len() > rapiddoc_core::resources::limit(20000)
        || values
            .iter()
            .any(|v| !v.is_finite() || !(0.5..=1.).contains(v))
    {
        return Err(error(ErrorCode::InvalidInput, "隔断可信度均值输入无效"));
    }
    if values.iter().all(|v| *v == 1.) {
        return Ok(1.);
    }
    let (total, power) = docvortex_core::exact_mean::exact_sum_parts(values)
        .ok_or_else(|| error(ErrorCode::InvalidInput, "隔断可信度精确和失败"))?;
    // 非全1的均值指数为-1；尾数的53位整数表示为total*2^(power+53)/count。
    let numerator = (total as u128) << (power + 53);
    let denominator = values.len() as u128;
    let mut q = numerator / denominator;
    let r = numerator % denominator;
    if 2 * r > denominator || (2 * r == denominator && q % 2 == 1) {
        q += 1
    }
    Ok(q as f64 * 2f64.powi(-53))
}
/// 旧拓扑分量语义：非矩形返回None而不是把正常候选拒绝变成错误。
pub fn table_component_specs(
    rows: usize,
    cols: usize,
    edges: Vec<(usize, usize)>,
) -> Result<Option<Vec<CellRegion>>, Error> {
    let count = rows
        .checked_mul(cols)
        .filter(|n| *n > 0 && *n <= rapiddoc_core::resources::limit(10000))
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "原生表格原子格超限"))?;
    if edges.len() > count * 2 {
        return Err(error(ErrorCode::ResourceLimit, "原生表格连接超限"));
    }
    if edges.iter().any(|&(a, b)| {
        a >= count
            || b >= count
            || !(a / cols == b / cols && a.abs_diff(b) == 1
                || a % cols == b % cols && a.abs_diff(b) == cols)
    }) {
        return Err(error(ErrorCode::InvalidInput, "原生表格连接无效"));
    }
    let parents = docvortex_core::tables::grid_parents(count, edges);
    let (_, specs) = docvortex_core::tables::component_specs(parents, rows, cols);
    Ok(specs.map(|s| {
        s.into_iter()
            .map(|(row, column, row_span, column_span)| CellRegion {
                row,
                column,
                row_span,
                column_span,
            })
            .collect()
    }))
}
/// 固定vendor合并内核；簇均值由兼容层按旧Python版本的求和语义提供。
pub fn table_merge_rules(
    rules: Vec<NativeRule>,
    coordinates: Vec<(u8, Vec<f64>)>,
    tolerance: f64,
    join: f64,
) -> Result<Vec<NativeRule>, Error> {
    if rules.len() > rapiddoc_core::resources::limit(20000)
        || coordinates.iter().map(|c| c.1.len()).sum::<usize>()
            > rapiddoc_core::resources::limit(20000)
    {
        return Err(error(ErrorCode::ResourceLimit, "表格线段超过预算"));
    }
    if !tolerance.is_finite()
        || tolerance < 0.
        || !join.is_finite()
        || join < 0.
        || rules.iter().any(|r| {
            r.0 > 1
                || [r.1, r.2, r.3]
                    .iter()
                    .any(|n| !n.is_finite() || n.abs() > 1e9)
        })
        || coordinates
            .iter()
            .any(|c| c.0 > 1 || c.1.iter().any(|n| !n.is_finite() || n.abs() > 1e9))
    {
        return Err(error(ErrorCode::InvalidInput, "表格线段无效"));
    }
    docvortex_core::tables::merge_rules(rules, coordinates, tolerance, join)
        .ok_or_else(|| error(ErrorCode::InvalidInput, "表格线段合并失败"))
}
pub fn table_assign_cells(
    glyphs: Vec<[f64; 4]>,
    specs: Vec<[f64; 4]>,
    index: Option<NativeGridIndex>,
) -> Result<Vec<(Option<usize>, bool)>, Error> {
    table_assign_cells_bounded(glyphs, specs, index, 3000)
}
pub fn table_assign_cells_page(
    glyphs: Vec<[f64; 4]>,
    specs: Vec<[f64; 4]>,
    index: Option<NativeGridIndex>,
) -> Result<Vec<(Option<usize>, bool)>, Error> {
    table_assign_cells_bounded(glyphs, specs, index, usize::MAX)
}
fn table_assign_cells_bounded(
    glyphs: Vec<[f64; 4]>,
    specs: Vec<[f64; 4]>,
    index: Option<NativeGridIndex>,
    limit: usize,
) -> Result<Vec<(Option<usize>, bool)>, Error> {
    let limit = rapiddoc_core::resources::limit(limit);
    if glyphs.len() > limit || specs.len() > rapiddoc_core::resources::limit(2000) {
        return Err(error(ErrorCode::ResourceLimit, "原生表格落格输入超过预算"));
    }
    if glyphs
        .iter()
        .chain(&specs)
        .flatten()
        .any(|x| !x.is_finite() || x.abs() > 1e9)
    {
        return Err(error(ErrorCode::InvalidInput, "原生表格落格坐标无效"));
    }
    docvortex_core::tables::assign_cells(glyphs, specs, index)
        .ok_or_else(|| error(ErrorCode::InvalidInput, "原生表格落格索引无效"))
}
mod table_latin;
mod table_unicode;
pub fn python_latin(c: char) -> bool {
    let n = c as u32;
    let a = table_latin::LATIN;
    let i = a.partition_point(|&(x, _)| x <= n);
    i > 0 && n <= a[i - 1].1
}
pub fn table_cell_visual_groups(
    boxes: Vec<[f64; 4]>,
    ids: Vec<usize>,
    rows: Vec<i64>,
    height: f64,
) -> Result<Vec<Vec<usize>>, Error> {
    table_cell_visual_groups_bounded(boxes, ids, rows, height, 3000)
}
pub fn table_cell_visual_groups_page(
    boxes: Vec<[f64; 4]>,
    ids: Vec<usize>,
    rows: Vec<i64>,
    height: f64,
) -> Result<Vec<Vec<usize>>, Error> {
    table_cell_visual_groups_bounded(boxes, ids, rows, height, usize::MAX)
}
fn table_cell_visual_groups_bounded(
    boxes: Vec<[f64; 4]>,
    ids: Vec<usize>,
    rows: Vec<i64>,
    height: f64,
    limit: usize,
) -> Result<Vec<Vec<usize>>, Error> {
    let limit = rapiddoc_core::resources::limit(limit);
    if boxes.len() > limit {
        return Err(error(ErrorCode::ResourceLimit, "单元格字形超过预算"));
    }
    if !height.is_finite()
        || height.abs() > 1e9
        || boxes
            .iter()
            .flatten()
            .any(|x| !x.is_finite() || x.abs() > 1e9)
    {
        return Err(error(ErrorCode::InvalidInput, "单元格字形几何无效"));
    }
    docvortex_core::tables::cell_visual_groups(boxes, ids, rows, height)
        .ok_or_else(|| error(ErrorCode::InvalidInput, "单元格字形组无效"))
}
pub mod text_boundary;
mod text_unicode_data;

/// CPython Unicode16 printable/decimal属性；不依赖宿主Python或Rust Unicode版本。
pub fn python_printable(c: char) -> bool {
    let n = c as u32;
    let ranges = table_unicode::PRINTABLE;
    let i = ranges.partition_point(|&(start, _)| start <= n);
    i > 0 && n <= ranges[i - 1].1
}
pub fn python_decimal(c: char) -> Option<u8> {
    table_unicode::DECIMALS
        .binary_search_by_key(&(c as u32), |&(n, _)| n)
        .ok()
        .map(|i| table_unicode::DECIMALS[i].1)
}
/// 复用固定vendor视觉分行内核；输入和二次工作上限在调用前校验。
pub fn table_visual_rows(
    boxes: Vec<[f64; 4]>,
    ids: Vec<usize>,
    height: f64,
) -> Result<Vec<Vec<usize>>, Error> {
    table_visual_rows_bounded(boxes, ids, height, 3000)
}
/// Full page path; same vendor kernel, independently bounded from diagnostics.
pub fn table_visual_rows_page(
    boxes: Vec<[f64; 4]>,
    ids: Vec<usize>,
    height: f64,
) -> Result<Vec<Vec<usize>>, Error> {
    table_visual_rows_bounded(boxes, ids, height, usize::MAX)
}
fn table_visual_rows_bounded(
    boxes: Vec<[f64; 4]>,
    ids: Vec<usize>,
    height: f64,
    limit: usize,
) -> Result<Vec<Vec<usize>>, Error> {
    let limit = rapiddoc_core::resources::limit(limit);
    if boxes.len() > limit {
        return Err(error(ErrorCode::ResourceLimit, "原生表格字形超过预算"));
    }
    if boxes.len() != ids.len()
        || !height.is_finite()
        || boxes
            .iter()
            .flatten()
            .any(|x| !x.is_finite() || x.abs() > 1e9)
    {
        return Err(error(ErrorCode::InvalidInput, "原生表格分行输入无效"));
    }
    docvortex_core::tables::visual_rows(boxes, ids, height)
        .ok_or_else(|| error(ErrorCode::InvalidInput, "原生表格分行输入无效"))
}

/// 合并单元格的矩形范围，位置为零基行列，跨度必须为正。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellRegion {
    pub row: usize,
    pub column: usize,
    pub row_span: usize,
    pub column_span: usize,
}

/// 根据缺失分隔边合并相邻原子格；拒绝越界、跨行伪邻接及 L 形连通分量。
pub fn cell_regions(
    rows: usize,
    columns: usize,
    missing_edges: Vec<(usize, usize)>,
    max_cells: usize,
) -> Result<Vec<CellRegion>, Error> {
    let max_cells = rapiddoc_core::resources::limit(max_cells);
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "网格尺寸溢出"))?;
    if rows == 0 || columns == 0 {
        return Err(error(ErrorCode::InvalidInput, "网格必须有行和列"));
    }
    if count > max_cells || missing_edges.len() > count.saturating_mul(2) {
        return Err(error(ErrorCode::ResourceLimit, "网格超过限制"));
    }
    for &(first, last) in &missing_edges {
        if first >= count || last >= count {
            return Err(error(ErrorCode::InvalidInput, "网格边索引越界"));
        }
        let adjacent = first / columns == last / columns && first.abs_diff(last) == 1
            || first % columns == last % columns && first.abs_diff(last) == columns;
        if !adjacent {
            return Err(error(ErrorCode::InvalidInput, "仅允许连接相邻单元格"));
        }
    }
    let parents = docvortex_core::tables::grid_parents(count, missing_edges);
    let (_, specs) = docvortex_core::tables::component_specs(parents, rows, columns);
    let specs =
        specs.ok_or_else(|| error(ErrorCode::InvalidInput, "非矩形连通分量不能作为合并单元格"))?;
    Ok(specs
        .into_iter()
        .map(|(row, column, row_span, column_span)| CellRegion {
            row,
            column,
            row_span,
            column_span,
        })
        .collect())
}

/// 线段在目标区间的联合覆盖率；反向线段允许，所有几何必须有限。
pub fn interval_coverage(intervals: Vec<(f64, f64)>, start: f64, end: f64) -> Result<f64, Error> {
    if !start.is_finite()
        || !end.is_finite()
        || end <= start
        || intervals
            .iter()
            .any(|(a, b)| !a.is_finite() || !b.is_finite())
    {
        return Err(error(ErrorCode::InvalidInput, "覆盖查询的几何无效"));
    }
    let rules = intervals.into_iter().map(|(a, b)| (0, 0., a, b)).collect();
    let result = docvortex_core::tables::coverage_batch(rules, vec![(0, vec![0.], start, end, 0.)])
        .ok_or_else(|| error(ErrorCode::InvalidInput, "覆盖查询内核拒绝输入"))?;
    result
        .first()
        .copied()
        .ok_or_else(|| error(ErrorCode::InvalidInput, "覆盖查询内核返回空结果"))
}

/// 构造规则层错误，保留稳定分类。
pub(crate) fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}

#[cfg(test)]
mod lane_adapter_tests {
    use super::*;
    #[test]
    fn lane_kernels_reject_empty_nonfinite_and_invalid_lane_indices() {
        assert!(text_supported_lane_intervals(&[], 100., 10.).is_err());
        let line = ([0., 0., 50., 10.], 10., true, false);
        assert!(text_supported_lane_intervals(&[line], f64::NAN, 10.).is_err());
        assert!(text_lane_assignments(&[line], &[], 1., None).is_err());
        let tail = (line.0, 10., None, 0., None, None, false, None, 0, 1);
        assert!(text_short_tail_destinations(&[tail], &[(0., 100.)], 10.).is_err());
        assert_eq!(
            text_supported_lane_intervals(&[line], 100., 10.).unwrap(),
            vec![(0., 50., 1)]
        );
    }
}
