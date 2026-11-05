use rapiddoc::native_table::{PageTables, Proposal};
use rapiddoc_core::{
    document::{Character, DrawingLine, ImageFact, Page},
    ErrorCode, Rect,
};
fn rect(b: [f64; 4]) -> Rect {
    Rect::try_from(b).unwrap()
}
fn glyph(index: usize, text: &str, b: [f64; 4]) -> Character {
    Character {
        index,
        unicode: text.chars().next().unwrap() as u32,
        is_generated: false,
        text: Some(text.into()),
        font_name: "test".into(),
        font_size_pt: 6.,
        angle_degrees: 0.,
        origin: [b[0], b[1]],
        tight_bounds: rect(b),
        loose_bounds: rect(b),
        visual_tight_bounds: rect(b),
        visual_loose_bounds: rect(b),
        visual_origin: [b[0], b[1]],
    }
}
fn line(vertical: bool, coordinate: f64, start: f64, end: f64) -> DrawingLine {
    let (a, b, bounds) = if vertical {
        (
            [coordinate, start],
            [coordinate, end],
            [coordinate, start, coordinate, end],
        )
    } else {
        (
            [start, coordinate],
            [end, coordinate],
            [start, coordinate, end, coordinate],
        )
    };
    DrawingLine {
        start: a,
        end: b,
        bounds: rect(bounds),
        width: 1.,
        orientation: if vertical { "vertical" } else { "horizontal" }.into(),
    }
}
fn page() -> Page {
    Page {
        page_index: 0,
        width_pt: 100.,
        height_pt: 80.,
        text: "A<&BC".into(),
        object_count: 10,
        characters: vec![
            glyph(0, "A", [20., 14., 24., 21.]),
            glyph(1, "<", [25., 14., 29., 21.]),
            glyph(2, "&", [30., 14., 34., 21.]),
            glyph(3, "B", [20., 40., 24., 47.]),
            glyph(4, "C", [70., 40., 74., 47.]),
        ],
        page_bounds: rect([0., 0., 100., 80.]),
        rotation: 0,
        visual_width_pt: 100.,
        visual_height_pt: 80.,
        paths: vec![],
        images: vec![],
        links: vec![],
        form_count: 0,
        drawing_lines: vec![
            line(true, 10., 10., 70.),
            line(true, 100., 10., 70.),
            line(true, 40., 25., 70.),
            line(false, 10., 10., 100.),
            line(false, 25., 10., 100.),
            line(false, 70., 10., 100.),
        ],
    }
}
fn proposal(index: usize) -> Proposal {
    Proposal {
        layout_index: index,
        bounds: rect([20., 20., 200., 140.]),
    }
}
#[test]
fn merged_cells_keep_actual_tracks_pixel_geometry_and_safe_html() {
    let planner = PageTables::new(page(), 200, 160).unwrap();
    let plan = planner.plan(&[proposal(7)], 6., 10000).unwrap();
    let native = plan.selections[0].native.as_ref().unwrap();
    assert_eq!(plan.selections[0].layout_index, 7);
    assert_eq!(native.x_tracks, [10., 40., 100.]);
    assert_eq!(native.y_tracks, [10., 25., 70.]);
    assert_eq!(native.table.cells[0].column_span, 2);
    assert_eq!(native.table.cells[0].character_indices, [0, 1, 2]);
    assert!(native.html.contains("A&lt;&amp;"));
    assert_eq!(
        native.page_geometry.cell_polygons[1],
        [[20., 50.], [80., 50.], [80., 140.], [20., 140.]]
    );
    assert_eq!(native.page_geometry.point_to_page, [2., 0., 0., 2., 0., 0.]);
}
#[test]
fn unsafe_candidates_keep_explicit_model_reasons() {
    let mut data = page();
    data.images.push(ImageFact {
        object_path: vec![1],
        bounds: rect([20., 30., 40., 50.]),
        visual_bounds: rect([20., 30., 40., 50.]),
        clip_bounds: None,
    });
    assert_eq!(
        PageTables::new(data, 200, 160)
            .unwrap()
            .plan(&[proposal(0)], 6., 10000)
            .unwrap()
            .selections[0]
            .reason,
        "raster_overlap"
    );
    let mut data = page();
    data.drawing_lines.remove(1);
    assert_eq!(
        PageTables::new(data, 200, 160)
            .unwrap()
            .plan(&[proposal(0)], 6., 10000)
            .unwrap()
            .selections[0]
            .reason,
        "incomplete_grid"
    );
    let mut data = page();
    data.characters[0].text = None;
    assert_eq!(
        PageTables::new(data, 200, 160)
            .unwrap()
            .plan(&[proposal(0)], 6., 10000)
            .unwrap()
            .selections[0]
            .reason,
        "unreliable_native_text"
    );
    let mut data = page();
    data.characters.clear();
    assert_eq!(
        PageTables::new(data, 200, 160)
            .unwrap()
            .plan(&[proposal(0)], 6., 10000)
            .unwrap()
            .selections[0]
            .reason,
        "no_native_text"
    );
    let planner = PageTables::new(page(), 200, 160).unwrap();
    let small = Proposal {
        layout_index: 0,
        bounds: rect([55., 30., 65., 40.]),
    };
    assert!(planner.plan(&[small], 32., 10000).unwrap().selections[0]
        .native
        .is_none());
}
#[test]
fn cumulative_budget_includes_rejected_grid_queries_and_bad_facts_fail() {
    let mut data = page();
    data.drawing_lines.remove(1);
    let planner = PageTables::new(data, 200, 160).unwrap();
    let work = planner.plan(&[proposal(0)], 6., 10000).unwrap().work;
    assert!(work > 10);
    assert_eq!(
        planner
            .plan(&[proposal(0), proposal(1)], 6., work * 2 - 1)
            .err()
            .unwrap()
            .code,
        ErrorCode::ResourceLimit
    );
    assert!(planner
        .plan(&[proposal(0), proposal(0)], 6., 10000)
        .is_err());
    let mut data = page();
    data.characters[0].index = 3;
    assert!(PageTables::new(data, 200, 160).is_err());
    assert!(planner.plan(&[], f64::NAN, 100).is_err());
}
