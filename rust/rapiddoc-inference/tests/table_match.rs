use rapiddoc_inference::table_match::{bounds, match_cells};

#[test]
fn source_order_unmatched_and_first_tie() {
    let cells = vec![
        vec![0., 10., 10., 20.],
        vec![0., 10., 10., 20.],
        vec![10., 10., 20., 20.],
    ];
    let ocr = vec![
        vec![1., 1., 5., 5.],
        vec![11., 11., 15., 15.],
        vec![1., 11., 5., 15.],
        vec![3., 13., 8., 18.],
        vec![10., 10., 10., 12.],
    ];
    let result = match_cells(&cells, &ocr).unwrap();
    assert_eq!(result.cells, vec![vec![2, 3], vec![], vec![1]]);
    assert_eq!(result.unmatched, vec![0, 4]);
}

#[test]
fn quadrilateral_empty_and_invalid() {
    assert_eq!(
        bounds(&[3., 4., 0., 1., 2., 6., -1., 3.]).unwrap(),
        [-1., 1., 3., 6.]
    );
    assert_eq!(
        match_cells(&[], &[vec![1., 1., 2., 2.]]).unwrap().unmatched,
        vec![0]
    );
    assert!(bounds(&[2., 1., 1., 2.]).is_err());
    assert!(bounds(&[0., 0., f64::NAN, 1.]).is_err());
    assert!(match_cells(&[], &vec![vec![0.; 4]; 1001]).is_err());
}
