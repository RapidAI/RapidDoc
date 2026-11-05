use rapiddoc_core::Rect;
use rapiddoc_native::grid::{recover, Glyph, GridConfig, Separator};

#[test]
fn region_filters_other_rules_and_keeps_cell_text() {
    let mut lines = Vec::new();
    for coordinate in [0., 10., 20.] {
        lines.push(Separator {
            vertical: true,
            coordinate,
            start: 0.,
            end: 20.,
        });
        lines.push(Separator {
            vertical: false,
            coordinate,
            start: 0.,
            end: 20.,
        });
    }
    lines.push(Separator {
        vertical: true,
        coordinate: 100.,
        start: 0.,
        end: 20.,
    });
    let glyphs = vec![Glyph {
        index: 0,
        bounds: Rect::try_from([2., 2., 8., 8.]).expect("glyph"),
        text: "中文".into(),
        generated: false,
    }];
    let table = recover(
        Rect::try_from([0., 0., 20., 20.]).expect("region"),
        &lines,
        glyphs,
        &GridConfig::default(),
    )
    .expect("recovered");
    assert_eq!(table.rows, 2);
    assert_eq!(table.columns, 2);
    assert_eq!(table.cells[0].text, "中文");
}

#[test]
fn region_without_closed_tracks_is_not_a_successful_empty_table() {
    assert!(recover(
        Rect::try_from([0., 0., 20., 20.]).expect("region"),
        &[],
        vec![],
        &GridConfig::default()
    )
    .is_err());
}
