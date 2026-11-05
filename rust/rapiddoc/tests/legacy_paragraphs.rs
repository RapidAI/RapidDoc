use rapiddoc::legacy_paragraphs::split;
use rapiddoc_core::ErrorCode;
use serde_json::{json, Value};

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/migration/legacy-paragraphs.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn complete_page_json_matches_unmodified_python_for_all_frozen_cases() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 600);
    for c in cases {
        assert_eq!(split(&c["pages"]).unwrap(), c["expected"], "{}", c["name"]);
    }
}

#[test]
fn floating_geometry_unicode_and_real_cpu_ocr_keep_exact_legacy_outputs() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/migration/legacy-paragraph-boundaries.json");
    let data: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for c in data["cases"].as_array().unwrap() {
        let actual = split(&c["pages"]).unwrap();
        if actual != c["expected"] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/legacy-paragraph-failure.json");
            std::fs::write(
                path,
                serde_json::to_vec_pretty(&json!({"actual":actual,"reference":c})).unwrap(),
            )
            .unwrap();
            panic!("exact legacy JSON differs: {}", c["name"]);
        }
    }
}

#[test]
fn cross_page_moves_and_deletion_flags_preserve_the_original_inputs() {
    let fixture = fixture();
    let c = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "merge-97-")
        .unwrap();
    let before = c["pages"].clone();
    let actual = split(&before).unwrap();
    assert_eq!(before, c["pages"]);
    assert_eq!(
        actual[0]["para_blocks"][0]["lines"][4]["spans"][0]["cross_page"],
        true
    );
    assert_eq!(actual[1]["para_blocks"][0]["lines"], json!([]));
    assert_eq!(actual[1]["para_blocks"][0]["lines_deleted"], true);
    assert_eq!(actual[1]["preproc_blocks"], before[1]["preproc_blocks"]);
}

#[test]
fn invalid_shapes_duplicate_pages_and_resource_limits_are_not_silent_empty_outputs() {
    assert_eq!(split(&json!({})).unwrap_err().code, ErrorCode::InvalidInput);
    let page = json!({"page_idx":0,"page_size":[600,800],"preproc_blocks":[]});
    assert_eq!(
        split(&json!([page.clone(), page.clone()]))
            .unwrap_err()
            .code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        split(&Value::Array(vec![page; 1001])).unwrap_err().code,
        ErrorCode::ResourceLimit
    );
    assert_eq!(split(&json!([{"page_idx":0,"page_size":[600,800],"preproc_blocks":[{"type":"text","bbox":[0,0,0,10],"lines":[]}]}])).unwrap_err().code,ErrorCode::InvalidInput);
    assert_eq!(split(&json!([{"page_idx":0,"page_size":[600,800],"preproc_blocks":[],"metadata":"x".repeat(2_000_001)}])).unwrap_err().code,ErrorCode::ResourceLimit);
}
