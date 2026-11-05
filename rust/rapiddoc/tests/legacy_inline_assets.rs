use rapiddoc::legacy_page::inline::replace;
use serde_json::Value;

#[test]
fn actual_old_inline_bytes_html_and_permissive_base64_match() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/legacy-inline-assets.json")).unwrap(),
    )
    .unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let (html, assets) =
            replace(c["markup"].as_str().unwrap(), c["prefix"].as_str().unwrap()).unwrap();
        assert_eq!(html, c["expected"], "{}", c["markup"]);
        let calls = c["calls"].as_array().unwrap();
        assert_eq!(assets.len(), calls.len());
        for (asset, expected) in assets.iter().zip(calls) {
            assert_eq!(asset.name, expected["name"], "{}", c["markup"]);
            let hex: String = asset.png.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(hex, expected["bytes_hex"], "{}", c["markup"]);
        }
    }
}

#[test]
fn unsafe_extensions_vector_conversion_and_resource_limits_are_explicit() {
    for fmt in ["../png", "wmf", "x-emf", "a.b", ""] {
        let result = replace(&format!("src=\"data:image/{fmt};base64,YQ==\""), "");
        if fmt.is_empty() {
            assert!(result.unwrap().1.is_empty());
        } else {
            assert!(result.is_err());
        }
    }
    assert!(replace(&"x".repeat(2_000_001), "").is_err());
    assert!(replace("", &"x".repeat(4097)).is_err());
    assert!(replace(&"src=\"data:image/png;base64,YQ==\"".repeat(1001), "").is_err());
}

#[test]
fn original_request_budget_and_malformed_model_are_checked_before_replacement() {
    use rapiddoc::legacy_page::assets::assemble;
    use serde_json::json;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let png =
        std::fs::read(root.join("build/migration/legacy-asset-images/RGB-48-40-0.png")).unwrap();
    let request = json!({"model":"wrong type"});
    assert!(assemble(&request, &png, &[], false, 0.9).is_err());
    let html = format!("src=\"data:image/png;base64,{}\"", "A".repeat(1_100_000));
    let request = json!({"page_width":48,"page_height":40,"page_index":0,"scale":1.,"formula_enabled":true,"model":{"layout_dets":[
        {"category_id":5,"poly":[2,3,20,3,20,22,2,22],"score":0.9,"original_order":0,"html":html},
        {"category_id":5,"poly":[22,3,40,3,40,22,22,22],"score":0.9,"original_order":1,"html":html}
    ]}});
    assert!(assemble(&request, &png, &[], false, 0.9).is_err());
}
