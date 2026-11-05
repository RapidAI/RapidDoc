//! Deterministic original spreadsheet HTML and singleton projection rules.
use super::*;
use std::sync::OnceLock;
pub fn html(table: &Table) -> String {
    let cells = table
        .data
        .iter()
        .map(|c| ((c.row, c.col), c))
        .collect::<BTreeMap<_, _>>();
    let mut covered = BTreeSet::new();
    let mut lines = vec!["<table>".to_owned()];
    for r in 0..table.num_rows {
        lines.push("  <tr>".into());
        for c in 0..table.num_cols {
            if covered.contains(&(r, c)) {
                continue;
            }
            let Some(cell) = cells.get(&(r, c)) else {
                lines.push("    <td></td>".into());
                continue;
            };
            let tag = if cell.row == 0 { "th" } else { "td" };
            let mut attrs = String::new();
            if cell.row_span > 1 {
                attrs.push_str(&format!(" rowspan=\"{}\"", cell.row_span));
            }
            if cell.col_span > 1 {
                attrs.push_str(&format!(" colspan=\"{}\"", cell.col_span));
            }
            for rr in 0..cell.row_span {
                for cc in 0..cell.col_span {
                    covered.insert((r + rr, c + cc));
                }
            }
            let mut text = if cell.text_is_html {
                cell.text.clone()
            } else {
                crate::document_output::html::escape(&cell.text)
            };
            if !cell.media.is_empty() {
                if !text.is_empty() {
                    text.push_str("<br>");
                }
                text.push_str(&cell.media.join("<br>"));
            }
            for eq in &cell.equations {
                text.push_str("<eq>");
                text.push_str(eq);
                text.push_str("</eq>");
            }
            static BLOCK: OnceLock<regex::Regex> = OnceLock::new();
            let re = BLOCK.get_or_init(|| {
                regex::Regex::new(r"(?i)<\s*(p|ul|ol|li|div|table|blockquote|pre|h[1-6])\b")
                    .unwrap()
            });
            let inner = if !text.is_empty() && cell.text_is_html && re.is_match(&text) {
                text
            } else {
                format!("<p>{text}</p>")
            };
            lines.push(format!("    <{tag}{attrs}>{inner}</{tag}>"));
        }
        lines.push("  </tr>".into());
    }
    lines.push("</table>".into());
    lines.join("\n")
}
pub fn block(table: &Table, singleton_text: bool) -> serde_json::Value {
    if singleton_text && table.data.len() == 1 {
        let c = &table.data[0];
        if c.row_span == 1
            && c.col_span == 1
            && c.media.is_empty()
            && c.equations.is_empty()
            && !c.text_is_html
        {
            return serde_json::json!({"type":"text","content":if c.text.is_empty(){vec![]}else{vec![serde_json::json!({"type":"text","content":c.text})]}});
        }
    }
    serde_json::json!({"type":"table","content":html(table)})
}
