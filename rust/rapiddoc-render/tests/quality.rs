use rapiddoc_core::ErrorCode;
use rapiddoc_render::quality::{metrics, select};
#[test]
fn coverage_priority_and_small_equal_table_rules() {
    let w = "<table><tr><td>A B</td><td>x</td></tr></table>";
    let l = "<table><tr><td>A</td><td>x</td></tr></table>";
    let selected = select(&["A".into(), "B".into()], w, l).unwrap();
    assert_eq!(
        (selected.route, selected.reason),
        ("wired", "wired_text_coverage")
    );
    let tied = select(&[], w, l).unwrap();
    assert_eq!((tied.route, tied.reason), ("wireless", "wireless_cell_gap"));
    let three = "<table><tr><td>A</td><td>B</td><td>C</td></tr></table>";
    assert_eq!(
        select(&[], three, three).unwrap().reason,
        "wireless_small_equal"
    );
}
#[test]
fn blanks_entities_and_legacy_header_prefix() {
    let m = metrics(
        "<table><thead><tr><td>\u{001c}<br/>\u{3000}</td><th>&amp;</th></tr></thead></table>",
    )
    .unwrap();
    assert_eq!((m.cells, m.blank, m.non_blank()), (3, 1, 2));
    assert_eq!(
        metrics("<table><tr><td>&nbsp;</td></tr></table>")
            .err()
            .unwrap()
            .code,
        ErrorCode::InvalidInput
    );
    assert!(metrics("<table><tr><td></tr></table>").is_err());
}
