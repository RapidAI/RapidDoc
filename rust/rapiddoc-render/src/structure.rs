//! 只接受 SLANet 表格语法，结构和 OCR 文字均不能注入任意 HTML。
use rapiddoc_core::{Error, ErrorCode, Rect};
use rapiddoc_native::grid::{Cell, Table};

fn invalid(message: &str) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.into(),
    }
}

/// 解析模型 token 成完整逻辑网格，再由通用安全表格输出器转义文字。
/// 不纠正缺格、重叠、未闭合结构，也不接受模型输出的任意标签/属性。
pub fn html(tokens: &[String], texts: &[String]) -> Result<String, Error> {
    if tokens.len() > rapiddoc_core::resources::limit(10_010)
        || texts.len() > rapiddoc_core::resources::limit(10_000)
        || texts.iter().map(String::len).sum::<usize>() > rapiddoc_core::resources::limit(1_000_000)
        || tokens.iter().map(String::len).sum::<usize>()
            > rapiddoc_core::resources::limit(1_000_000)
    {
        return Err(Error {
            code: ErrorCode::ResourceLimit,
            message: "表格结构或文字超过预算".into(),
        });
    }
    if tokens.len() < 6
        || tokens[..3] != ["<html>", "<body>", "<table>"]
        || tokens[tokens.len() - 3..] != ["</table>", "</body>", "</html>"]
    {
        return Err(invalid("表格结构外层不完整"));
    }
    let inner = &tokens[3..tokens.len() - 3];
    let mut index = 0;
    let mut row = 0;
    let mut in_row = false;
    let mut column = 0;
    let mut columns = 0;
    let mut section: Option<&str> = None;
    let mut occupied = std::collections::HashSet::new();
    let mut cells = Vec::new();
    while index < inner.len() {
        match inner[index].as_str() {
            "<thead>" | "<tbody>" if !in_row && section.is_none() => {
                section = Some(&inner[index]);
            }
            "</thead>" if !in_row && section == Some("<thead>") => {
                section = None;
            }
            "</tbody>" if !in_row && section == Some("<tbody>") => {
                section = None;
            }
            "<tr>" if !in_row => {
                in_row = true;
                column = 0;
            }
            "</tr>" if in_row => {
                in_row = false;
                row += 1;
            }
            "<td></td>" | "<td>" | "<td" if in_row => {
                let tag = inner[index].as_str();
                let mut row_span = None;
                let mut column_span = None;
                if tag == "<td" {
                    index += 1;
                    while index < inner.len() && inner[index] != ">" {
                        let attribute = inner[index].trim();
                        let (name, value) = attribute
                            .split_once('=')
                            .ok_or_else(|| invalid("未知单元格属性"))?;
                        let number = value
                            .strip_prefix('"')
                            .and_then(|s| s.strip_suffix('"'))
                            .or_else(|| value.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
                            .ok_or_else(|| invalid("单元格跨度需要引号"))?;
                        if number.is_empty() || !number.bytes().all(|b| b.is_ascii_digit()) {
                            return Err(invalid("单元格跨度无效"));
                        }
                        let value: usize = number.parse().map_err(|_| invalid("单元格跨度溢出"))?;
                        if !(1..=1000).contains(&value) {
                            return Err(invalid("单元格跨度超限"));
                        }
                        let slot = match name {
                            "rowspan" => &mut row_span,
                            "colspan" => &mut column_span,
                            _ => return Err(invalid("未知单元格属性")),
                        };
                        if slot.replace(value).is_some() {
                            return Err(invalid("单元格属性重复"));
                        }
                        index += 1;
                    }
                    if index == inner.len() {
                        return Err(invalid("单元格开始标签未闭合"));
                    }
                }
                if tag != "<td></td>" {
                    index += 1;
                    if inner.get(index).map(String::as_str) != Some("</td>") {
                        return Err(invalid("单元格未闭合或包含未知结构"));
                    }
                }
                while occupied.contains(&(row, column)) {
                    column += 1;
                }
                let row_span = row_span.unwrap_or(1);
                let column_span = column_span.unwrap_or(1);
                if row + row_span > rapiddoc_core::resources::limit(1000)
                    || column + column_span > rapiddoc_core::resources::limit(1000)
                    || row_span * column_span + occupied.len()
                        > rapiddoc_core::resources::limit(100_000)
                {
                    return Err(invalid("表格逻辑网格超过限制"));
                }
                for r in row..row + row_span {
                    for c in column..column + column_span {
                        if !occupied.insert((r, c)) {
                            return Err(invalid("表格单元格重复覆盖"));
                        }
                    }
                }
                let text = texts
                    .get(cells.len())
                    .ok_or_else(|| invalid("结构格数和单元格框数不匹配"))?
                    .clone();
                cells.push(Cell {
                    row,
                    column,
                    row_span,
                    column_span,
                    text,
                    character_indices: vec![],
                });
                column += column_span;
                columns = columns.max(column);
            }
            _ => return Err(invalid("未知或错位的表格 token")),
        }
        index += 1;
    }
    if in_row || section.is_some() || cells.len() != texts.len() {
        return Err(invalid("结构未闭合或格数不匹配"));
    }
    // 仅供输出器验证拓扑的逻辑坐标，不作为模型像素坐标返回。
    let table = Table {
        bounds: Rect::try_from([0., 0., columns as f64, row as f64])?,
        rows: row,
        columns,
        cells,
    };
    crate::table_html(&table, 100_000)
}
