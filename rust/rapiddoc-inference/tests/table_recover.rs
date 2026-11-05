use rapiddoc_core::ErrorCode;
use rapiddoc_inference::table_recover::recover;
fn p(a: f64, b: f64, c: f64, d: f64) -> [[f64; 2]; 4] {
    [[a, b], [a, d], [c, d], [c, b]]
}
#[test]
fn merged_header_and_source_indices() {
    let result = recover(
        &[
            p(0., 0., 200., 50.),
            p(0., 50., 100., 100.),
            p(100., 50., 200., 100.),
        ],
        10.,
        15.,
    )
    .unwrap();
    assert_eq!(
        result.logic_points,
        [[0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 1]]
    );
    assert_eq!((result.rows, result.columns), (2, 2));
}
#[test]
fn invalid_input_and_limit() {
    assert!(recover(&[], 10., 15.).is_err());
    assert!(recover(&[p(0., 0., f64::NAN, 1.)], 10., 15.).is_err());
    assert!(recover(&[p(0., 0., f64::INFINITY, 1.)], 10., 15.).is_err());
    assert!(recover(&[p(0., 0., 10., 10.)], f64::NAN, 15.).is_err());
    assert_eq!(
        recover(&vec![p(0., 0., 10., 10.); 2001], 10., 15.)
            .err()
            .unwrap()
            .code,
        ErrorCode::ResourceLimit
    );
}

#[test]
fn legacy_empty_column_intervals_match_python() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/upstream_068/unet_empty_columns.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let polygons: Vec<rapiddoc_inference::table_recover::Polygon> =
            serde_json::from_value(case["polygons"].clone()).unwrap();
        let actual = recover(&polygons, 10., 15.).unwrap();
        assert_eq!(
            serde_json::to_value(actual.logic_points).unwrap(),
            case["points"],
            "{}",
            case["name"]
        );
    }
}

#[test]
fn legacy_nonpositive_tracks_preserve_python_order_and_topology() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/upstream_068/unet_nonpositive_tracks.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let polygons: Vec<rapiddoc_inference::table_recover::Polygon> =
            serde_json::from_value(case["polygons"].clone()).unwrap();
        let actual = recover(&polygons, 10., 15.).unwrap();
        assert_eq!(
            serde_json::to_value(actual.logic_points).unwrap(),
            case["points"],
            "{}",
            case["name"]
        );
    }
}

#[test]
fn production_preserves_python_collapsed_and_nonconvex_cells() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/upstream_068/unet_collapsed_polygons.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let polygons: Vec<rapiddoc_inference::table_recover::Polygon> =
            serde_json::from_value(case["polygons"].clone()).unwrap();
        let actual = recover(&polygons, 10., 15.).unwrap();
        assert_eq!(
            serde_json::to_value(actual.logic_points).unwrap(),
            case["points"],
            "{}",
            case["name"]
        );
    }
}
