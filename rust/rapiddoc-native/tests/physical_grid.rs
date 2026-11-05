use rapiddoc_core::Rect;
use rapiddoc_native::grid::{assemble, Glyph, GridConfig, Separator};

fn rules() -> Vec<Separator> {
    vec![
        Separator {
            vertical: true,
            coordinate: 0.,
            start: 0.,
            end: 20.,
        },
        Separator {
            vertical: true,
            coordinate: 20.,
            start: 0.,
            end: 20.,
        },
        Separator {
            vertical: true,
            coordinate: 10.,
            start: 10.,
            end: 20.,
        },
        Separator {
            vertical: false,
            coordinate: 0.,
            start: 0.,
            end: 20.,
        },
        Separator {
            vertical: false,
            coordinate: 10.,
            start: 0.,
            end: 20.,
        },
        Separator {
            vertical: false,
            coordinate: 20.,
            start: 0.,
            end: 20.,
        },
    ]
}

#[test]
fn merged_header_and_origin_order_are_preserved() {
    let glyphs = vec![
        Glyph {
            index: 2,
            bounds: Rect::try_from([12., 2., 18., 8.]).expect("box"),
            text: "B".into(),
            generated: false,
        },
        Glyph {
            index: 1,
            bounds: Rect::try_from([2., 2., 8., 8.]).expect("box"),
            text: "A".into(),
            generated: false,
        },
    ];
    let table = assemble(
        &[0., 10., 20.],
        &[0., 10., 20.],
        &rules(),
        glyphs,
        &GridConfig::default(),
    )
    .expect("grid");
    assert_eq!(table.cells.len(), 3);
    assert_eq!(table.cells[0].column_span, 2);
    assert_eq!(table.cells[0].text, "AB");
    assert_eq!(table.cells[0].character_indices, vec![1, 2]);
}

#[test]
fn incomplete_boundary_and_nonrectangular_merges_are_rejected() {
    let mut lines = rules();
    lines.remove(0);
    assert!(assemble(
        &[0., 10., 20.],
        &[0., 10., 20.],
        &lines,
        vec![],
        &GridConfig::default()
    )
    .is_err());
    let mut lines = rules();
    lines[4].start = 10.;
    assert!(assemble(
        &[0., 10., 20.],
        &[0., 10., 20.],
        &lines,
        vec![],
        &GridConfig::default()
    )
    .is_err());
}

#[test]
fn fragmented_outer_border_uses_union_coverage() {
    let mut lines = rules();
    lines.remove(0);
    lines.push(Separator {
        vertical: true,
        coordinate: 0.,
        start: 0.,
        end: 10.,
    });
    lines.push(Separator {
        vertical: true,
        coordinate: 0.,
        start: 20.,
        end: 10.,
    });
    assert!(assemble(
        &[0., 10., 20.],
        &[0., 10., 20.],
        &lines,
        vec![],
        &GridConfig::default()
    )
    .is_ok());
}

#[test]
fn generated_cell_separator_is_removed_but_real_spaces_are_preserved() {
    let glyph = |index, x, text: &str, generated| Glyph {
        index,
        bounds: Rect::try_from([x, 12., x + 1., 18.]).expect("box"),
        text: text.into(),
        generated,
    };
    let glyphs = vec![
        glyph(0, 2., "a", false),
        glyph(1, 3., " ", false),
        glyph(2, 4., "b", false),
        glyph(3, 5., " ", true),
        glyph(4, 12., "c", false),
    ];
    let table = assemble(
        &[0., 10., 20.],
        &[0., 10., 20.],
        &rules(),
        glyphs,
        &GridConfig::default(),
    )
    .expect("grid");
    assert_eq!(table.cells[1].text, "a b");
    assert_eq!(table.cells[2].text, "c");
    let glyphs = vec![
        glyph(0, 2., "a", false),
        glyph(1, 3., " ", true),
        glyph(2, 4., "b", false),
    ];
    let table = assemble(
        &[0., 10., 20.],
        &[0., 10., 20.],
        &rules(),
        glyphs,
        &GridConfig::default(),
    )
    .expect("grid");
    assert_eq!(table.cells[1].text, "a b");
}
