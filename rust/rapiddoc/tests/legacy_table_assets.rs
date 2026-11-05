use rapiddoc::legacy_page::assets::{assemble_with_table_images, TableFillImage};
use serde_json::{json, Value};

#[test]
fn old_table_pages_and_writer_order_match_except_explicit_stable_names() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/legacy-table-assets.json")).unwrap(),
    )
    .unwrap();
    let png = std::fs::read(root.join(fixture["source"].as_str().unwrap())).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let fills: Vec<_> = case["fills"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| TableFillImage {
                uuid: f["uuid"].as_str().unwrap().to_owned(),
                bbox: std::array::from_fn(|i| f["bbox"][i].as_f64().unwrap()),
                png: std::fs::read(root.join(f["source"].as_str().unwrap())).unwrap(),
            })
            .collect();
        let parent = case["parent"].as_str().unwrap().rsplit('/').next().unwrap();
        let prefix = if parent.is_empty() {
            String::new()
        } else {
            format!("{parent}/")
        };
        let output =
            assemble_with_table_images(&case["request"], &png, &[], false, 0.9, &prefix, &fills)
                .unwrap();
        assert_eq!(
            output.page, case["expected_with_stable_names"],
            "{}",
            case["name"]
        );
        let calls = case["calls"].as_array().unwrap();
        assert_eq!(output.assets.len(), calls.len(), "{}", case["name"]);
        for (asset, expected) in output.assets.iter().zip(calls) {
            assert_eq!(asset.name, expected["name"], "{}", case["name"]);
        }
    }
}

#[test]
fn invalid_uuid_bbox_images_and_html_expansion_are_bounded() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let png =
        std::fs::read(root.join("build/migration/legacy-asset-images/RGB-48-40-0.png")).unwrap();
    let mut request = json!({"model":{"layout_dets":[{"category_id":5,"score":0.9,"original_order":0,"poly":[2,3,20,3,20,22,2,22],"html":"<td>uuid</td>"}]},"page_width":48,"page_height":40,"page_index":7,"scale":1.,"formula_enabled":true});
    for (uuid, bbox, bytes) in [
        ("", [0.; 4], png.clone()),
        ("uuid", [f64::NAN; 4], png.clone()),
        ("uuid", [0.; 4], b"invalid".to_vec()),
    ] {
        let fills = [TableFillImage {
            uuid: uuid.into(),
            bbox,
            png: bytes,
        }];
        assert!(
            assemble_with_table_images(&request, &png, &[], false, 0.9, "images/", &fills).is_err()
        );
    }
    request["model"]["layout_dets"][0]["html"] = json!("x".repeat(30_000));
    let fills = [TableFillImage {
        uuid: "x".into(),
        bbox: [0.; 4],
        png: png.clone(),
    }];
    assert!(
        assemble_with_table_images(&request, &png, &[], false, 0.9, "images/", &fills).is_err()
    );
}
