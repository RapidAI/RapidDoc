use rapiddoc_core::ErrorCode;
use rapiddoc_inference::{
    blank_cell,
    wired_match::{fill_blank, match_cells, Ocr},
};
fn p(a: f64, b: f64, c: f64, d: f64) -> [[f64; 2]; 4] {
    [[a, b], [a, d], [c, d], [c, b]]
}

#[test]
fn blank_fill_rejects_mismatched_geometry_and_total_text_budget() {
    let polygons = [p(0., 0., 100., 80.), p(100., 0., 200., 80.)];
    let mut matched = match_cells(&polygons, &[]).unwrap();
    assert_eq!(
        fill_blank(&mut matched, &[], &[]).unwrap_err().code,
        ErrorCode::InvalidInput
    );
    let results: Vec<_> = (0..2)
        .map(|cell_index| blank_cell::ResultCell {
            cell_index,
            text: "x".repeat(500_001),
            confidence: 0.9,
            accepted: true,
            reason: "accepted".into(),
            contrast: Some(0.5),
        })
        .collect();
    assert_eq!(
        fill_blank(&mut matched, &polygons, &results)
            .unwrap_err()
            .code,
        ErrorCode::ResourceLimit
    );
    assert!(matched.cells.iter().all(Vec::is_empty));
}
#[test]
fn unicode_weight_split_and_ambiguous_ownership() {
    let text = "甲乙丙丁戊己庚辛壬癸";
    let result = match_cells(
        &[p(0., 0., 100., 80.), p(100., 0., 200., 80.)],
        &[Ocr {
            points: p(10., 10., 190., 30.),
            text: text.into(),
            confidence: 0.9,
        }],
    )
    .unwrap();
    assert_eq!(
        result.cell_lines,
        vec![vec!["甲乙丙丁戊"], vec!["己庚辛壬癸"]]
    );
    assert!(result
        .cells
        .iter()
        .all(|v| v[0].split && v[0].ocr_index == Some(0)));
    let ambiguous = match_cells(
        &[p(0., 0., 150., 80.), p(50., 0., 200., 80.)],
        &[Ocr {
            points: p(80., 10., 120., 30.),
            text: "text".into(),
            confidence: 0.9,
        }],
    )
    .unwrap();
    assert_eq!(ambiguous.unmatched, [0]);
    assert!(ambiguous.cells.iter().all(Vec::is_empty));
}
#[test]
fn invalid_score_and_blank_rec_constraints() {
    assert_eq!(
        match_cells(
            &[],
            &[Ocr {
                points: p(0., 0., 1., 1.),
                text: "A".into(),
                confidence: 2.
            }]
        )
        .err()
        .unwrap()
        .code,
        ErrorCode::InvalidInput
    );
    assert!(!blank_cell::accepted("口", 0.99));
    assert!(!blank_cell::accepted("ABC", 0.59));
    assert!(blank_cell::accepted("ABC", 0.6));
    let blank = blank_cell::crop(&vec![255; 30 * 20 * 3], 30, 20, p(0., 0., 30., 20.))
        .unwrap()
        .unwrap();
    assert_eq!(blank.contrast, 0.);
    assert_eq!((blank.width, blank.height), (28, 18));
}

#[test]
fn tilted_diagonal_is_not_rejected_before_python_intersection_rules() {
    let cells = [
        [
            [800.2001342773438, 227.4000244140625],
            [807.2001342773438, 241.4000244140625],
            [846.4000854492188, 221.80001831054688],
            [839.4000854492188, 207.80001831054688],
        ],
        p(0., 0., 100., 40.),
    ];
    let result = match_cells(
        &cells,
        &[Ocr {
            points: p(10., 10., 40., 30.),
            text: "normal".into(),
            confidence: 0.9,
        }],
    )
    .unwrap();
    assert!(result.cells[0].is_empty());
    assert_eq!(result.cell_lines[1], vec!["normal"]);
    assert!(result.unmatched.is_empty());
}
