use rapiddoc::reading_order::{plan, Region, MAX_WORK};
use rapiddoc_core::{ErrorCode, Rect};

fn region(b: [f64; 4]) -> Region {
    Region {
        bounds: Rect::try_from(b).unwrap(),
        horizontal: true,
    }
}

#[test]
fn spanning_header_and_footer_do_not_interleave_columns_by_row() {
    let regions = vec![
        region([0., 450., 600., 480.]),
        region([330., 200., 580., 250.]),
        region([0., 0., 600., 30.]),
        region([20., 60., 280., 120.]),
        region([330., 60., 580., 120.]),
        region([20., 200., 280., 250.]),
    ];
    let output = plan(&regions, 600, 500, MAX_WORK).unwrap();
    assert_eq!(output.indices, [2, 3, 5, 4, 1, 0]);
    assert_eq!(output.reason, "geometry");
    assert!(output.work_used > 0);
}

#[test]
fn sequential_diagonal_regions_use_vertical_order_and_parallel_regions_use_columns() {
    let regions = vec![
        region([10., 100., 100., 140.]),
        region([200., 10., 300., 50.]),
    ];
    assert_eq!(plan(&regions, 600, 500, MAX_WORK).unwrap().indices, [1, 0]);
    let regions = vec![
        region([330., 20., 580., 200.]),
        region([10., 10., 280., 150.]),
    ];
    assert_eq!(plan(&regions, 600, 500, MAX_WORK).unwrap().indices, [1, 0]);
}

#[test]
fn ambiguity_overlap_and_direction_preserve_the_entire_page() {
    for (regions, reason) in [
        (
            vec![
                region([10., 10., 200., 100.]),
                region([20., 50., 240., 120.]),
            ],
            "overlapping_regions",
        ),
        (
            vec![region([10., 50., 240., 100.]), region([10., 0., 240., 50.])],
            "ambiguous_geometry",
        ),
        (
            vec![region([10., 10., 10., 30.])],
            "invalid_region_geometry",
        ),
        (
            vec![Region {
                bounds: Rect::try_from([10., 10., 30., 40.]).unwrap(),
                horizontal: false,
            }],
            "unsupported_direction",
        ),
    ] {
        let output = plan(&regions, 600, 500, MAX_WORK).unwrap();
        assert_eq!(output.indices, (0..regions.len()).collect::<Vec<_>>());
        assert_eq!(output.reason, reason);
    }
}

#[test]
fn work_and_region_limits_are_errors_and_not_ambiguous_fallbacks() {
    let regions = vec![region([10., 10., 200., 30.]), region([10., 60., 200., 80.])];
    assert_eq!(
        plan(&regions, 600, 500, 1).err().unwrap().code,
        ErrorCode::ResourceLimit
    );
    assert_eq!(
        plan(&regions, 600, 500, 0).err().unwrap().code,
        ErrorCode::InvalidInput
    );
    let regions = (0..1001)
        .map(|_| region([10., 10., 200., 30.]))
        .collect::<Vec<_>>();
    assert_eq!(
        plan(&regions, 600, 500, MAX_WORK).err().unwrap().code,
        ErrorCode::ResourceLimit
    );
}

#[test]
fn every_permutation_of_a_spanning_double_column_page_has_the_same_reading_order() {
    let boxes = [
        [0., 450., 600., 480.],
        [330., 200., 580., 250.],
        [0., 0., 600., 30.],
        [20., 60., 280., 120.],
        [330., 60., 580., 120.],
        [20., 200., 280., 250.],
    ];
    fn check(boxes: &[[f64; 4]], permutation: &mut [usize], start: usize) {
        if start == permutation.len() {
            let regions = permutation
                .iter()
                .map(|&i| region(boxes[i]))
                .collect::<Vec<_>>();
            let result = plan(&regions, 600, 500, MAX_WORK).unwrap();
            assert_eq!(
                result
                    .indices
                    .iter()
                    .map(|&i| permutation[i])
                    .collect::<Vec<_>>(),
                [2, 3, 5, 4, 1, 0]
            );
            return;
        }
        for i in start..permutation.len() {
            permutation.swap(start, i);
            check(boxes, permutation, start + 1);
            permutation.swap(start, i);
        }
    }
    check(&boxes, &mut [0, 1, 2, 3, 4, 5], 0);
}

#[test]
fn one_thousand_disjoint_regions_fit_the_default_work_budget_without_recursive_stack_growth() {
    let regions = (0..1000)
        .rev()
        .map(|i| region([10., i as f64 * 10., 590., i as f64 * 10. + 4.]))
        .collect::<Vec<_>>();
    let result = plan(&regions, 600, 12000, MAX_WORK).unwrap();
    assert_eq!(result.indices, (0..1000).rev().collect::<Vec<_>>());
    assert!(result.work_used < MAX_WORK);
    assert_eq!(
        plan(&regions, 600, 12000, 500_000).err().unwrap().code,
        ErrorCode::ResourceLimit
    );
}
