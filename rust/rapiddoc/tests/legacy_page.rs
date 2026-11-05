use rapiddoc::legacy_page::assemble;
use serde_json::{json, Value};

#[test]
fn large_integer_numeric_equality_and_decimal_rounding_keep_old_page_fields() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/migration/legacy-page-numbers.json");
    let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        assert_eq!(
            assemble(&c["request"]).unwrap(),
            c["expected"],
            "{}",
            c["name"]
        );
    }
}

#[test]
fn invalid_models_missing_order_and_unmigrated_output_paths_are_explicit() {
    use rapiddoc_core::ErrorCode;
    let mut request = json!({"model":{"layout_dets":[{"category_id":1,"poly":[0,0,100,0,100,30,0,30],"score":0.95,"original_order":null}]},"scale":1.0,"page_index":0,"page_width":600,"page_height":800,"formula_enabled":true});
    assert_eq!(assemble(&request).unwrap_err().code, ErrorCode::Unsupported);
    request["model"]["layout_dets"][0]["poly"] = json!([0, 0]);
    assert_eq!(
        assemble(&request).unwrap_err().code,
        ErrorCode::InvalidInput
    );
    let render = json!({"pages":[{"page_idx":0,"page_size":[600,800],"para_blocks":[{"type":"image","blocks":[{"type":"image_body"}]}]}],"make_mode":"mm_markdown","image_path_prefix":"images"});
    assert_eq!(
        rapiddoc::legacy_page::render(&render).unwrap_err().code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        rapiddoc::legacy_page::assemble_many(&Value::Array(vec![json!({}); 1001]))
            .unwrap_err()
            .code,
        ErrorCode::ResourceLimit
    );
}

#[test]
fn markdown_and_content_list_match_old_union_make() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/migration/legacy-output.json");
    let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        assert_eq!(
            rapiddoc::legacy_page::render(&c["request"]).unwrap(),
            c["expected"],
            "{}",
            c["name"]
        );
    }
}

#[test]
fn multiple_model_pages_and_paragraphs_match_unmodified_old_stages() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/migration/legacy-page-chain.json");
    let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        assert_eq!(
            rapiddoc::legacy_page::assemble_many(&c["requests"]).unwrap(),
            c["expected"],
            "{}",
            c["name"]
        );
    }
}

#[test]
fn raw_model_outputs_match_complete_old_page_conversion() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/migration/legacy-page.json");
    let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let actual = assemble(&c["request"]).unwrap_or_else(|e| panic!("{}: {:?}", c["name"], e));
        if actual != c["expected"] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/legacy-page-failure.json");
            std::fs::write(
                path,
                serde_json::to_vec_pretty(&json!({"case":c,"actual":actual})).unwrap(),
            )
            .unwrap();
            panic!("old complete page differs: {}", c["name"]);
        }
    }
}
