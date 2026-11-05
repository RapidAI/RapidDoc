use rapiddoc::paragraphs::{organize, organize_ordered, Block, Line, LineOrder, Page, Region};
use rapiddoc_core::{ErrorCode, Rect};
fn line(b: [f64; 4], text: &str) -> Line {
    Line {
        points: [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]],
        text: text.into(),
        confidence: None,
    }
}
fn block(index: usize, lines: Vec<Line>) -> Block {
    Block {
        layout_index: index,
        kind: "text".into(),
        region: Region {
            label: "text".into(),
            coordinate: Rect::try_from([0., 0., 600., 500.]).unwrap(),
        },
        lines,
        text: None,
    }
}
fn page(blocks: Vec<Block>) -> Page {
    Page {
        width: 600,
        height: 500,
        coordinate_space: "page_pixels".into(),
        blocks,
    }
}
#[test]
fn column_major_paragraphs_keep_original_line_indices_and_optional_source_order() {
    let value = page(vec![block(
        9,
        vec![
            line([10., 10., 240., 30.], "left one"),
            line([330., 10., 560., 30.], "right one"),
            line([10., 40., 240., 60.], "left two"),
            line([330., 40., 560., 60.], "right two"),
        ],
    )]);
    let actual = organize(&value, true).unwrap();
    assert_eq!(actual.paragraphs.len(), 2);
    assert_eq!(actual.paragraphs[0].line_indices, [0, 2]);
    assert_eq!(actual.paragraphs[1].line_indices, [1, 3]);
    assert_eq!(actual.text, "left one left two\n\nright one right two");
    assert_eq!(actual.paragraphs[0].layout_index, 9);
    assert_eq!(organize(&value, false).unwrap().paragraphs.len(), 4);
}
#[test]
fn indent_blank_and_lists_split_without_mutating_lines_or_crossing_layout() {
    let mut value = page(vec![block(
        0,
        vec![
            line([10., 10., 240., 30.], "first"),
            line([10., 40., 240., 60.], "continued"),
            line([40., 70., 240., 90.], "indented"),
            line([10., 100., 240., 120.], "continued"),
            line([10., 130., 240., 150.], ""),
            line([10., 160., 240., 180.], "1. item"),
            line([10., 190., 240., 210.], "3.14 value"),
        ],
    )]);
    let actual = organize(&value, true).unwrap();
    assert_eq!(actual.paragraphs[0].line_indices, [0, 1]);
    assert_eq!(actual.paragraphs[1].line_indices, [2, 3]);
    assert_eq!(actual.paragraphs[2].line_indices, [4]);
    assert_eq!(actual.paragraphs[3].kind, "list_item");
    assert_eq!(actual.paragraphs[4].kind, "text");
    assert_eq!(value.blocks[0].lines.len(), 7);
    value.blocks.push(block(
        1,
        vec![line([10., 220., 240., 240.], "separate block")],
    ));
    assert_eq!(
        organize(&value, true)
            .unwrap()
            .paragraphs
            .last()
            .unwrap()
            .block_index,
        1
    );
}
#[test]
fn invalid_geometry_sources_and_aggregate_work_fail_before_output() {
    let mut value = page(vec![block(0, vec![line([0., 0., 10., 10.], "text")])]);
    value.blocks[0].lines[0].points[0][0] = f64::NAN;
    assert_eq!(
        organize(&value, true).err().unwrap().code,
        ErrorCode::InvalidInput
    );
    value = page(vec![
        block(0, vec![line([0., 0., 10., 10.], "text")]),
        block(0, vec![]),
    ]);
    assert!(organize(&value, true).is_err());
    value = page(vec![block(
        0,
        (0..1000)
            .map(|_| line([0., 0., 10., 10.], &"x".repeat(1000)))
            .collect(),
    )]);
    assert_eq!(
        organize(&value, true).err().unwrap().code,
        ErrorCode::ResourceLimit
    );
}

#[test]
fn hanging_list_continuations_join_without_absorbing_the_next_item_or_body() {
    let value = page(vec![block(
        0,
        vec![
            line([10., 10., 240., 30.], "1. A long item"),
            line([35., 40., 240., 60.], "with a continuation"),
            line([36., 70., 160., 90.], "and a final line."),
            line([10., 100., 240., 120.], "2. Next item"),
            line([35., 130., 240., 150.], "continued too"),
            line([10., 160., 240., 180.], "Ordinary body"),
        ],
    )]);
    let result = organize(&value, true).unwrap();
    assert_eq!(
        result
            .paragraphs
            .iter()
            .map(|p| p.line_indices.clone())
            .collect::<Vec<_>>(),
        vec![vec![0, 1, 2], vec![3, 4], vec![5]]
    );
    assert_eq!(result.paragraphs[0].kind, "list_item");
    assert_eq!(result.paragraphs[1].kind, "list_item");
    assert_eq!(result.paragraphs[2].kind, "text");
    assert_eq!(
        result.paragraphs[0].text.as_deref(),
        Some("1. A long item with a continuation and a final line.")
    );
    assert_eq!(result.source_line_count, 6);
}

#[test]
fn list_continuation_requires_stable_indent_height_gap_and_non_short_previous_line() {
    for (bbox, text) in [
        ([10., 40., 240., 60.], "flush left"),
        ([100., 40., 240., 60.], "too far indented"),
        ([35., 65., 240., 85.], "new paragraph after gap"),
        ([35., 40., 240., 67.], "larger text"),
        ([35., 40., 240., 60.], "• new bullet"),
        ([35., 40., 240., 60.], ""),
    ] {
        let value = page(vec![block(
            0,
            vec![line([10., 10., 240., 30.], "1. Item"), line(bbox, text)],
        )]);
        assert_eq!(
            organize(&value, false).unwrap().paragraphs.len(),
            2,
            "{text}"
        );
    }
    let value = page(vec![block(
        0,
        vec![
            line([10., 10., 240., 30.], "• Item"),
            line([35., 40., 100., 60.], "end."),
            line([35., 70., 240., 90.], "should stay separate"),
        ],
    )]);
    let result = organize(&value, false).unwrap();
    assert_eq!(result.paragraphs[0].line_indices, [0, 1]);
    assert_eq!(result.paragraphs[1].line_indices, [2]);
}

#[test]
fn geometric_line_order_repairs_shuffled_columns_without_changing_default_or_sources() {
    let value = page(vec![block(
        9,
        vec![
            line([10., 40., 240., 60.], "left two"),
            line([330., 40., 560., 60.], "right two"),
            line([10., 10., 240., 30.], "left one"),
            line([330., 10., 560., 30.], "right one"),
        ],
    )]);
    assert_eq!(organize(&value, true).unwrap().paragraphs.len(), 4);
    let actual = organize_ordered(&value, true, LineOrder::Geometry).unwrap();
    assert_eq!(actual.text, "left one left two\n\nright one right two");
    assert_eq!(actual.paragraphs[0].line_indices, [2, 0]);
    assert_eq!(actual.paragraphs[1].line_indices, [3, 1]);
    assert_eq!(
        actual
            .diagnostics
            .iter()
            .filter(|d| d.code == "lines_reordered")
            .count(),
        1
    );
    assert_eq!(value.blocks[0].lines[0].text, "left two");
}

#[test]
fn ambiguous_rows_and_semantic_barriers_preserve_source_order_with_diagnostics() {
    let same_row = page(vec![block(
        0,
        vec![
            line([100., 10., 240., 30.], "right"),
            line([10., 10., 90., 30.], "left"),
        ],
    )]);
    let actual = organize_ordered(&same_row, true, LineOrder::Geometry).unwrap();
    assert_eq!(actual.paragraphs[0].line_indices, [0]);
    assert!(actual
        .diagnostics
        .iter()
        .any(|d| d.code == "line_order_preserved_overlap"));
    for label in ["algorithm", "vertical_text"] {
        let mut value = page(vec![block(
            0,
            vec![
                line([10., 40., 240., 60.], "second"),
                line([10., 10., 240., 30.], "first"),
            ],
        )]);
        value.blocks[0].region.label = label.into();
        let actual = organize_ordered(&value, true, LineOrder::Geometry).unwrap();
        assert_eq!(actual.paragraphs[0].line_indices[0], 0);
        assert!(!actual.diagnostics.is_empty());
    }
}

#[test]
fn reordered_work_is_limited_across_blocks_not_only_in_source_order() {
    let value = page(
        (0..17)
            .map(|i| {
                block(
                    i,
                    (0..100)
                        .map(|j| {
                            let y = (99 - j) as f64 * 3.;
                            line(
                                [10., y, 240., y + 2.],
                                &if j == 99 {
                                    "x".repeat(20_000)
                                } else {
                                    "x".into()
                                },
                            )
                        })
                        .collect(),
                )
            })
            .collect(),
    );
    assert!(organize(&value, true).is_ok());
    assert_eq!(
        organize_ordered(&value, true, LineOrder::Geometry)
            .err()
            .unwrap()
            .code,
        ErrorCode::ResourceLimit
    );
}
