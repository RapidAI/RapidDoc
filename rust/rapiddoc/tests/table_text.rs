use serde_json::{json, Value};
use sha2::{Digest, Sha256};
fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn fixture() -> Value {
    serde_json::from_slice(&std::fs::read(root().join("build/migration/table-text.json")).unwrap())
        .unwrap()
}
fn pixels(png: &[u8]) -> Value {
    let p = image::load_from_memory(png).unwrap().to_rgb8();
    json!({"width":p.width(),"height":p.height(),"pixels_sha256":format!("{:x}",Sha256::digest(p.as_raw()))})
}
#[test]
fn old_table_box_retry_ocr_and_html_policies() {
    let f = fixture();
    for c in f["boxes"].as_array().unwrap() {
        assert_eq!(
            rapiddoc::legacy_model::ocr_results::table_boxes(&c["request"]).unwrap(),
            c["expected"]
        );
    }
    for c in f["restores"].as_array().unwrap() {
        assert_eq!(
            rapiddoc::legacy_model::table_text::restore(&c["request"]).unwrap(),
            c["expected"]
        );
    }
    for c in f["applies"].as_array().unwrap() {
        assert_eq!(
            rapiddoc::legacy_model::table_text::apply_ocr(&c["request"]).unwrap(),
            c["expected"]
        );
    }
    for c in f["attachments"].as_array().unwrap() {
        assert_eq!(
            rapiddoc::legacy_model::table_text::apply_prediction(&c["request"]).unwrap(),
            c["expected"]
        );
    }
}
#[test]
fn old_table_mask_and_retry_pixels() {
    for c in fixture()["masks"].as_array().unwrap() {
        let source = std::fs::read(root().join(c["source"].as_str().unwrap())).unwrap();
        let p = rapiddoc::legacy_model::table_text::detection(&source, &c["request"]).unwrap();
        assert_eq!(pixels(&p.png), c["expected"]);
        let p = rapiddoc::legacy_model::table_text::padding(&p.png, 8).unwrap();
        assert_eq!(pixels(&p.png), c["padded"]);
    }
}
#[test]
fn table_stage_failures_are_atomic() {
    let f = fixture();
    let mut req = f["applies"][0]["request"].clone();
    req["results"] = json!([]);
    assert!(rapiddoc::legacy_model::table_text::apply_ocr(&req).is_err());
    assert!(rapiddoc::legacy_model::table_text::detection(b"bad", &json!({"masks":[]})).is_err());
    let req = json!({"table":{"layout_image_list":["keep"]},"html_code":"<table>x</table>","scale":0,"formulas":[],"fill_images":[]});
    assert!(rapiddoc::legacy_model::table_text::apply_prediction(&req).is_err());
    assert_eq!(req["table"]["layout_image_list"], json!(["keep"]));
}
