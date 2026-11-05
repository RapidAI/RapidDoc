#[test]
fn legacy_empty_column_rendering_matches_python() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/upstream_068/unet_empty_columns.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let points = serde_json::from_value::<Vec<[usize; 4]>>(case["points"].clone()).unwrap();
        let polygons: Vec<rapiddoc_render::wired::Polygon> =
            serde_json::from_value(case["polygons"].clone()).unwrap();
        let texts: Vec<Vec<String>> = serde_json::from_value(case["texts"].clone()).unwrap();
        let html = rapiddoc_render::wired::legacy_html(&points, &polygons, &texts).unwrap();
        assert_eq!(html, case["html"].as_str().unwrap(), "{}", case["name"]);
        if case["name"] != "without_overlap" {
            assert!(rapiddoc_render::wired::html(&points, &polygons, &texts).is_err());
        }
    }
}
