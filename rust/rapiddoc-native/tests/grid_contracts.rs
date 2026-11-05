use rapiddoc_core::ErrorCode;
use rapiddoc_native::{cell_regions, interval_coverage, CellRegion};

#[test]
fn merged_header_retains_other_cells() {
    assert_eq!(
        cell_regions(2, 2, vec![(0, 1)], 100).expect("valid grid"),
        vec![
            CellRegion {
                row: 0,
                column: 0,
                row_span: 1,
                column_span: 2
            },
            CellRegion {
                row: 1,
                column: 0,
                row_span: 1,
                column_span: 1
            },
            CellRegion {
                row: 1,
                column: 1,
                row_span: 1,
                column_span: 1
            }
        ]
    );
}

#[test]
fn invalid_connectivity_does_not_panic_or_create_rectangles() {
    assert!(cell_regions(2, 2, vec![(0, 1), (0, 2)], 100).is_err());
    assert!(cell_regions(2, 2, vec![(1, 2)], 100).is_err());
    assert!(cell_regions(2, 2, vec![(0, 4)], 100).is_err());
    assert_eq!(
        cell_regions(usize::MAX, 2, vec![], 100)
            .expect_err("overflow")
            .code,
        ErrorCode::ResourceLimit
    );
}

#[test]
fn coverage_unions_reversed_overlapping_and_touching_segments() {
    assert_eq!(
        interval_coverage(vec![(5., 0.), (5., 8.), (7., 10.)], 0., 10.).expect("coverage"),
        1.
    );
    assert_eq!(
        interval_coverage(vec![(0., 2.), (5., 7.)], 0., 10.).expect("partial"),
        0.4
    );
    assert!(interval_coverage(vec![(f64::NAN, 0.)], 0., 10.).is_err());
}
