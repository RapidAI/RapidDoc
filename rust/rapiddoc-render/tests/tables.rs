use rapiddoc_core::Rect;
use rapiddoc_native::grid::{Cell, Table};
use rapiddoc_render::table_html;

fn cell(row: usize, column: usize, row_span: usize, column_span: usize, text: &str) -> Cell {
    Cell {
        row,
        column,
        row_span,
        column_span,
        text: text.into(),
        character_indices: vec![],
    }
}

#[test]
fn merged_cells_keep_text_and_escape_markup() {
    let table = Table {
        bounds: Rect::try_from([0., 0., 20., 20.]).expect("box"),
        rows: 2,
        columns: 2,
        cells: vec![
            cell(0, 0, 1, 2, "中文  <b>&\nnext"),
            cell(1, 0, 1, 1, "left"),
            cell(1, 1, 1, 1, "right"),
        ],
    };
    let html = table_html(&table, 100).expect("table");
    assert!(html.contains("<td colspan=\"2\">中文  &lt;b&gt;&amp;<br/>next</td>"));
    assert_eq!(html.matches("<tr>").count(), 2);
    assert_eq!(html.matches("<td").count(), 3);
    assert!(!html.contains("<th>"));
}

#[test]
fn overlapping_or_missing_cells_cannot_render_as_success() {
    let mut table = Table {
        bounds: Rect::try_from([0., 0., 20., 20.]).expect("box"),
        rows: 2,
        columns: 2,
        cells: vec![cell(0, 0, 1, 2, "header")],
    };
    assert!(table_html(&table, 100).is_err());
    table.cells.push(cell(0, 0, 2, 2, "overlap"));
    assert!(table_html(&table, 100).is_err());
    table.cells = vec![cell(0, 0, 2, 2, "all")];
    assert!(table_html(&table, 3).is_err());
}
