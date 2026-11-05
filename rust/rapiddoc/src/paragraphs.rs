//! 保守的布局块内段落预览；保留每个原始行索引，不跨布局块合并。
use rapiddoc_core::{geometry, Error, ErrorCode, Rect};
use rapiddoc_native::text_boundary::{merge, python_digit, python_space};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Deserialize)]
pub struct Page {
    pub width: usize,
    pub height: usize,
    pub coordinate_space: String,
    pub blocks: Vec<Block>,
}
#[derive(Deserialize)]
pub struct Block {
    pub layout_index: usize,
    pub kind: String,
    pub region: Region,
    #[serde(default)]
    pub lines: Vec<Line>,
    pub text: Option<String>,
}
#[derive(Deserialize)]
pub struct Region {
    pub label: String,
    pub coordinate: Rect,
}
#[derive(Deserialize)]
pub struct Line {
    pub points: [[f64; 2]; 4],
    pub text: String,
    pub confidence: Option<f64>,
}
#[derive(Serialize)]
pub struct Paragraph {
    pub block_index: usize,
    pub layout_index: usize,
    pub column_index: usize,
    pub kind: String,
    pub line_indices: Vec<usize>,
    pub bbox: Rect,
    pub text: Option<String>,
    pub break_reason: &'static str,
}
#[derive(Serialize)]
pub struct Diagnostic {
    pub code: &'static str,
    pub block_index: usize,
}
#[derive(Serialize)]
pub struct Output {
    pub coordinate_space: &'static str,
    pub paragraphs: Vec<Paragraph>,
    pub text: String,
    pub diagnostics: Vec<Diagnostic>,
    pub source_line_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_order: Option<crate::reading_order::Plan>,
}

/// 块内行顺序；默认保持来源，几何模式也不跨布局块或处理竖排/算法。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LineOrder {
    Source,
    Geometry,
}

/// 按来源行序整理；保持既有段落接口的默认行为。
pub fn organize(page: &Page, split_columns: bool) -> Result<Output, Error> {
    organize_ordered(page, split_columns, LineOrder::Source)
}
fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}
fn height(b: [f64; 4]) -> f64 {
    b[3] - b[1]
}
fn horizontal(line: &Line) -> bool {
    let a = line.points[0];
    let b = line.points[1];
    b[0] > a[0] && (b[1] - a[1]).abs() <= (b[0] - a[0]) * 0.0875
}
fn marker(text: &str) -> bool {
    let s = text.trim_start_matches(python_space);
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if "•‣◦▪▫".contains(first) {
        return true;
    }
    if "-*+".contains(first) {
        return chars.next().is_some_and(python_space);
    }
    let value = s.strip_prefix('(').unwrap_or(s);
    let mut chars = value.chars().peekable();
    let mut digits = 0;
    while chars.peek().is_some_and(|c| python_digit(*c)) {
        chars.next();
        digits += 1;
    }
    if digits > 0 {
        return match chars.next() {
            Some('.') => chars.next().is_none_or(python_space),
            Some(')' | '、') => true,
            _ => false,
        };
    }
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || "一二三四五六七八九十".contains(c))
        && chars.next().is_some_and(|c| matches!(c, ')' | '、'))
}
fn columns(block: &Block, boxes: &[[f64; 4]], split: bool) -> Vec<Vec<usize>> {
    let original = || vec![(0..boxes.len()).collect()];
    if !split
        || boxes.len() < 4
        || block
            .lines
            .iter()
            .any(|line| !horizontal(line) || line.text.trim_matches(python_space).is_empty())
    {
        return original();
    }
    let mut heights: Vec<_> = boxes.iter().map(|b| height(*b)).collect();
    heights.sort_by(f64::total_cmp);
    let median = heights[heights.len() / 2];
    let mut sorted: Vec<_> = (0..boxes.len()).collect();
    sorted.sort_by(|&a, &b| boxes[a][0].total_cmp(&boxes[b][0]).then(a.cmp(&b)));
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut end = f64::NEG_INFINITY;
    for index in sorted {
        if boxes[index][0] > end + 1.5 * median {
            groups.push(Vec::new());
        }
        groups.last_mut().unwrap().push(index);
        end = end.max(boxes[index][2]);
    }
    if groups.len() < 2 || groups.iter().any(|g| g.len() < 2) {
        return original();
    }
    let spans: Vec<_> = groups
        .iter()
        .map(|g| {
            g.iter().fold([f64::INFINITY, f64::NEG_INFINITY], |a, &i| {
                [a[0].min(boxes[i][1]), a[1].max(boxes[i][3])]
            })
        })
        .collect();
    if spans
        .windows(2)
        .any(|p| p[0][1].min(p[1][1]) - p[0][0].max(p[1][0]) < median)
    {
        return original();
    }
    for group in &mut groups {
        group.sort_unstable();
    }
    groups
}
/// 仅在栏内所有行都有可靠的水平垂直顺序时排序；歧义时整栏保留原索引顺序。
/// 空行可能是人为段界，不移动；算法与竖排标签也不使用水平阅读规则。
fn order_lines(
    block: &Block,
    boxes: &[[f64; 4]],
    indices: &mut Vec<usize>,
) -> Option<&'static str> {
    if indices.len() < 2 {
        return None;
    }
    if block.region.label == "algorithm" {
        return Some("line_order_preserved_algorithm");
    }
    if block.region.label == "vertical_text" {
        return Some("line_order_preserved_vertical_text");
    }
    if indices
        .iter()
        .any(|&i| block.lines[i].text.trim_matches(python_space).is_empty())
    {
        return Some("line_order_preserved_blank");
    }
    if indices.iter().any(|&i| !horizontal(&block.lines[i])) {
        return Some("line_order_preserved_non_horizontal");
    }
    let mut sorted = indices.clone();
    sorted.sort_by(|&a, &b| boxes[a][1].total_cmp(&boxes[b][1]).then(a.cmp(&b)));
    if sorted.windows(2).any(|pair| {
        let a = boxes[pair[0]];
        let b = boxes[pair[1]];
        b[1] < a[3] - 0.2 * height(a).min(height(b))
    }) {
        return Some("line_order_preserved_overlap");
    }
    if sorted == *indices {
        return None;
    }
    *indices = sorted;
    Some("lines_reordered")
}
fn break_between(
    prev: &Line,
    next: &Line,
    a: [f64; 4],
    b: [f64; 4],
    column: [f64; 4],
    title: bool,
) -> Option<&'static str> {
    if prev.text.trim_matches(python_space).is_empty()
        || next.text.trim_matches(python_space).is_empty()
    {
        return Some("blank_line");
    }
    if !horizontal(prev) || !horizontal(next) {
        return Some("non_horizontal");
    }
    let h = height(a).max(height(b));
    let small = height(a).min(height(b));
    if b[1] < a[1] + small * 0.3 {
        return Some("overlap_or_order");
    }
    if h / small > 1.5 {
        return Some("line_height_change");
    }
    if b[1] - a[3] > 0.9 * h {
        return Some("vertical_gap");
    }
    let overlap = (a[2].min(b[2]) - a[0].max(b[0])).max(0.);
    if overlap < 0.5 * (a[2] - a[0]).min(b[2] - b[0]) {
        return Some("horizontal_separation");
    }
    if !title {
        if marker(&prev.text) || marker(&next.text) {
            return Some("list_boundary");
        }
        if b[0] - column[0] > 0.75 * h && a[0] - column[0] <= 0.4 * h {
            return Some("indent");
        }
        if column[2] - a[2] > (2. * h).max(0.25 * (column[2] - column[0])) {
            return Some("short_previous_line");
        }
    }
    None
}
/// 仅用稳定的悬挂缩进证明列表续行；遇到新标记、短尾或正文左缘立即结束条目。
/// 行高作为 em 的预览近似，不推测字体、嵌套层级或跨布局块列表关系。
fn list_break(
    block: &Block,
    boxes: &[[f64; 4]],
    members: &[usize],
    next: usize,
    column: [f64; 4],
) -> Option<&'static str> {
    let first = members[0];
    let prev = members[members.len() - 1];
    // 复用空行、方向、源顺序、间距及水平分离的公共安全边界。
    if let Some(reason) = break_between(
        &block.lines[prev],
        &block.lines[next],
        boxes[prev],
        boxes[next],
        column,
        true,
    ) {
        return Some(reason);
    }
    if marker(&block.lines[next].text) {
        return Some("list_boundary");
    }
    let em = height(boxes[first]);
    let next_height = height(boxes[next]);
    let gap = boxes[next][1] - boxes[prev][3];
    let indent = boxes[next][0] - boxes[first][0];
    if em.max(next_height) / em.min(next_height) > 1.15
        || !(-0.3 * em..=0.7 * em).contains(&gap)
        || !(0.5 * em..=2. * em).contains(&indent)
        || (members.len() > 1 && (boxes[next][0] - boxes[members[1]][0]).abs() > 0.25 * em)
    {
        return Some("list_continuation_boundary");
    }
    if column[2] - boxes[prev][2] > (2. * em).max(0.25 * (column[2] - column[0])) {
        return Some("short_previous_line");
    }
    None
}
/// 在页面像素坐标中组织块内段落，保留源行索引；不跨块合并。
/// 几何行序仅作用于可信水平栏，歧义时保留原序并提供诊断；表格保持整体。
/// 无效几何或全页累计预算（包括排序后的扫描量）超限返回错误。
pub fn organize_ordered(
    page: &Page,
    split_columns: bool,
    order: LineOrder,
) -> Result<Output, Error> {
    organize_with_order(
        page,
        split_columns,
        order,
        LineOrder::Source,
        crate::reading_order::MAX_WORK,
    )
}

/// 指定栏内行序和页面块序；默认包装入口保持既有结果字段和顺序。
/// 块序仅重排完整块，不合并来源；预算不足报错，几何歧义则整页保留块序。
pub fn organize_with_order(
    page: &Page,
    split_columns: bool,
    order: LineOrder,
    block_order: LineOrder,
    max_block_order_work: usize,
) -> Result<Output, Error> {
    if max_block_order_work == 0 || max_block_order_work > crate::reading_order::MAX_WORK {
        return Err(error(ErrorCode::InvalidInput, "页面块阅读顺序预算无效"));
    }
    if page.coordinate_space != "page_pixels"
        || page.width == 0
        || page.height == 0
        || page
            .width
            .checked_mul(page.height)
            .is_none_or(|n| n > rapiddoc_core::resources::limit(16_000_000))
    {
        return Err(error(
            ErrorCode::InvalidInput,
            "段落输入必须为有效页面像素坐标",
        ));
    }
    if page.blocks.len() > rapiddoc_core::resources::limit(1000) {
        return Err(error(ErrorCode::ResourceLimit, "段落块数超过预算"));
    }
    let mut seen = BTreeSet::new();
    let mut count = 0;
    let mut bytes = 0;
    let mut work = 0usize;
    // 整页先验证，防止尾部坏数据产生部分结果。
    for block in &page.blocks {
        if !seen.insert(block.layout_index)
            || !["text", "table", "unsupported"].contains(&block.kind.as_str())
        {
            return Err(error(ErrorCode::InvalidInput, "段落来源块类型或索引无效"));
        }
        count += block.lines.len();
        bytes += block.text.as_ref().map_or(0, String::len);
        let mut prefix = 0;
        for line in &block.lines {
            bytes += line.text.len();
            prefix += line.text.len();
            work += prefix;
            let b: [f64; 4] = geometry::bounds(&line.points)?.into();
            let cross: Vec<_> = (0..4)
                .map(|i| {
                    let a = line.points[i];
                    let b = line.points[(i + 1) % 4];
                    let c = line.points[(i + 2) % 4];
                    (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
                })
                .collect();
            let positive = cross.iter().any(|v| *v > 0.);
            let negative = cross.iter().any(|v| *v < 0.);
            if b[2] <= b[0]
                || b[3] <= b[1]
                || line
                    .points
                    .iter()
                    .flatten()
                    .any(|v| v.abs() > rapiddoc_core::resources::limit(1_000_000.))
                || line
                    .confidence
                    .is_some_and(|v| !v.is_finite() || !(0. ..=1.).contains(&v))
                || positive == negative
            {
                return Err(error(ErrorCode::InvalidInput, "段落行几何或置信度无效"));
            }
        }
        if count > rapiddoc_core::resources::limit(10_000)
            || bytes > rapiddoc_core::resources::limit(2_000_000)
            || block.lines.len() > rapiddoc_core::resources::limit(1000)
            || work > rapiddoc_core::resources::limit(32_000_000)
        {
            return Err(error(
                ErrorCode::ResourceLimit,
                "段落行数、文字或扫描工作量超过预算",
            ));
        }
    }
    let block_plan = if block_order == LineOrder::Geometry {
        let regions = page
            .blocks
            .iter()
            .map(|b| crate::reading_order::Region {
                bounds: b.region.coordinate,
                horizontal: b.region.label != "vertical_text" && b.lines.iter().all(horizontal),
            })
            .collect::<Vec<_>>();
        Some(crate::reading_order::plan(
            &regions,
            page.width,
            page.height,
            max_block_order_work,
        )?)
    } else {
        None
    };
    let block_indices = block_plan.as_ref().map_or_else(
        || (0..page.blocks.len()).collect::<Vec<_>>(),
        |p| p.indices.clone(),
    );
    let mut paragraphs = Vec::new();
    let mut diagnostics = Vec::new();
    let mut ordered_work = 0usize;
    for block_index in block_indices {
        let block = &page.blocks[block_index];
        if block.kind != "text" {
            paragraphs.push(Paragraph {
                block_index,
                layout_index: block.layout_index,
                column_index: 0,
                kind: block.kind.clone(),
                line_indices: Vec::new(),
                bbox: block.region.coordinate,
                text: block.text.clone(),
                break_reason: "layout_boundary",
            });
            continue;
        }
        if block.lines.is_empty() {
            diagnostics.push(Diagnostic {
                code: "empty_text_block",
                block_index,
            });
            if block.text.as_ref().is_some_and(|text| !text.is_empty()) {
                paragraphs.push(Paragraph {
                    block_index,
                    layout_index: block.layout_index,
                    column_index: 0,
                    kind: "text".into(),
                    line_indices: Vec::new(),
                    bbox: block.region.coordinate,
                    text: block.text.clone(),
                    break_reason: "no_line_geometry",
                });
            }
            continue;
        }
        let boxes = block
            .lines
            .iter()
            .map(|line| geometry::bounds(&line.points).map(Into::into))
            .collect::<Result<Vec<[f64; 4]>, _>>()?;
        let title = ["doc_title", "paragraph_title"].contains(&block.region.label.as_str());
        let algorithm = block.region.label == "algorithm";
        let allow_columns = split_columns
            && !title
            && !algorithm
            && !(order == LineOrder::Geometry && block.region.label == "vertical_text");
        let mut groups = columns(block, &boxes, allow_columns);
        if order == LineOrder::Geometry {
            let mut reasons = BTreeSet::new();
            for group in &mut groups {
                if let Some(reason) = order_lines(block, &boxes, group) {
                    reasons.insert(reason);
                }
            }
            diagnostics.extend(
                reasons
                    .into_iter()
                    .map(|code| Diagnostic { code, block_index }),
            );
        }
        if groups.len() > 1 {
            diagnostics.push(Diagnostic {
                code: "columns_split",
                block_index,
            });
        }
        for (column_index, indices) in groups.into_iter().enumerate() {
            let mut prefix = 0usize;
            for &i in &indices {
                prefix += block.lines[i].text.len();
                ordered_work += prefix;
                if ordered_work > rapiddoc_core::resources::limit(32_000_000) {
                    return Err(error(
                        ErrorCode::ResourceLimit,
                        "排序后的段落累计扫描工作量超过预算",
                    ));
                }
            }
            let column = indices.iter().fold(
                [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ],
                |b, &i| {
                    [
                        b[0].min(boxes[i][0]),
                        b[1].min(boxes[i][1]),
                        b[2].max(boxes[i][2]),
                        b[3].max(boxes[i][3]),
                    ]
                },
            );
            let mut parts: Vec<(Vec<usize>, &'static str)> = Vec::new();
            for index in indices {
                let previous = parts.last().and_then(|part| part.0.last()).copied();
                let reason = if algorithm {
                    None
                } else {
                    previous.and_then(|p| {
                        if !title {
                            if let Some((members, _)) = parts.last() {
                                if marker(&block.lines[members[0]].text) {
                                    return list_break(block, &boxes, members, index, column);
                                }
                            }
                        }
                        break_between(
                            &block.lines[p],
                            &block.lines[index],
                            boxes[p],
                            boxes[index],
                            column,
                            title,
                        )
                    })
                };
                if parts.is_empty() || reason.is_some() {
                    parts.push((Vec::new(), reason.unwrap_or("layout_boundary")));
                }
                parts.last_mut().unwrap().0.push(index);
            }
            for (line_indices, reason) in parts {
                let mut bounds = [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ];
                let mut contents = Vec::with_capacity(line_indices.len());
                for &i in &line_indices {
                    let a = boxes[i];
                    bounds = [
                        bounds[0].min(a[0]),
                        bounds[1].min(a[1]),
                        bounds[2].max(a[2]),
                        bounds[3].max(a[3]),
                    ];
                    contents.push(block.lines[i].text.clone());
                }
                let text = if algorithm {
                    contents.join("\n")
                } else {
                    merge(&contents)?
                };
                let kind = if title {
                    "title"
                } else if algorithm {
                    "algorithm"
                } else if marker(&contents[0]) {
                    "list_item"
                } else {
                    "text"
                };
                paragraphs.push(Paragraph {
                    block_index,
                    layout_index: block.layout_index,
                    column_index,
                    kind: kind.into(),
                    line_indices,
                    bbox: Rect::try_from(bounds)?,
                    text: Some(text),
                    break_reason: reason,
                });
            }
        }
    }
    let text = paragraphs
        .iter()
        .filter_map(|p| p.text.as_deref())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    Ok(Output {
        block_order: block_plan,
        coordinate_space: "page_pixels",
        paragraphs,
        text,
        diagnostics,
        source_line_count: page
            .blocks
            .iter()
            .filter(|b| b.kind == "text")
            .map(|b| b.lines.len())
            .sum(),
    })
}
