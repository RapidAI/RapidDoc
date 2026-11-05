use serde_json::Value;
#[test]
fn old_preescaped_unet_html_is_not_encoded_twice() {
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/legacy-unet.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for c in fixture["html"].as_array().unwrap() {
        assert_eq!(
            rapiddoc_render::wired::legacy_html(
                &serde_json::from_value::<Vec<[usize; 4]>>(c["points"].clone()).unwrap(),
                &serde_json::from_value::<Vec<[[f64; 2]; 4]>>(c["polygons"].clone()).unwrap(),
                &serde_json::from_value::<Vec<Vec<String>>>(c["texts"].clone()).unwrap(),
            )
            .unwrap(),
            c["expected"].as_str().unwrap()
        );
    }
}
