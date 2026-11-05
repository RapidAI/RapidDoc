//! 文本和表格输出器；保持 Unicode 与空格，不把输入文字解释为 HTML。

use rapiddoc_core::{Error, ErrorCode};
use rapiddoc_native::grid::Table;
pub mod quality;
pub mod structure;
pub mod wired;

/// 转义文本节点和属性值中的 HTML 特殊字符，保留 Unicode 与连续空格。
pub fn escape_html(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for value in text.chars() {
        match value {
            '&' => result.push_str("&amp;"),
            '<' => result.push_str("&lt;"),
            '>' => result.push_str("&gt;"),
            '"' => result.push_str("&quot;"),
            '\'' => result.push_str("&#39;"),
            _ => result.push(value),
        }
    }
    result
}

/// 输出带 rowspan/colspan 的 HTML 表格，适合嵌入 Markdown；不猜测首行是否表头。
pub fn table_html(table: &Table, max_cells: usize) -> Result<String, Error> {
    let max_cells = rapiddoc_core::resources::limit(max_cells);
    let count = table
        .rows
        .checked_mul(table.columns)
        .ok_or_else(|| invalid("表格尺寸溢出"))?;
    if count == 0 || count > max_cells {
        return Err(invalid("表格为空或超过输出预算"));
    }
    let mut owner = vec![false; count];
    let mut starts = vec![None; count];
    for (index, cell) in table.cells.iter().enumerate() {
        let last_row = cell
            .row
            .checked_add(cell.row_span)
            .ok_or_else(|| invalid("单元格跨度溢出"))?;
        let last_column = cell
            .column
            .checked_add(cell.column_span)
            .ok_or_else(|| invalid("单元格跨度溢出"))?;
        if cell.row_span == 0
            || cell.column_span == 0
            || last_row > table.rows
            || last_column > table.columns
        {
            return Err(invalid("单元格跨度越界"));
        }
        for row in cell.row..last_row {
            for column in cell.column..last_column {
                let slot = &mut owner[row * table.columns + column];
                if *slot {
                    return Err(invalid("单元格重复覆盖"));
                }
                *slot = true;
            }
        }
        starts[cell.row * table.columns + cell.column] = Some(index);
    }
    if owner.iter().any(|value| !*value) {
        return Err(invalid("表格存在未覆盖位置"));
    }
    let mut html = String::from("<table>\n");
    for row in 0..table.rows {
        html.push_str("<tr>");
        for column in 0..table.columns {
            if let Some(index) = starts[row * table.columns + column] {
                let cell = &table.cells[index];
                html.push_str("<td");
                if cell.row_span > 1 {
                    html.push_str(&format!(" rowspan=\"{}\"", cell.row_span));
                }
                if cell.column_span > 1 {
                    html.push_str(&format!(" colspan=\"{}\"", cell.column_span));
                }
                html.push('>');
                let value = escape_html(&cell.text)
                    .replace("\r\n", "\n")
                    .replace('\n', "<br/>");
                html.push_str(&value);
                html.push_str("</td>");
            }
        }
        html.push_str("</tr>\n");
    }
    html.push_str("</table>");
    Ok(html)
}

/// 输出器拒绝不完整 IR，避免生成看似成功但丢格子的表格。
fn invalid(message: &str) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.into(),
    }
}
