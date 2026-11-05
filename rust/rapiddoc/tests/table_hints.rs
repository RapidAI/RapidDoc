use rapiddoc::legacy_model::table_hints;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn fixture() -> Value {
    serde_json::from_slice(&std::fs::read(root().join("build/migration/table-hints.json")).unwrap())
        .unwrap()
}
#[test]
fn prepared_candidate_payload_is_identity_without_new_hints() {
    let mut source = Vec::new();
    let image = image::RgbImage::from_fn(17, 13, |x, y| image::Rgb([x as u8, y as u8, 83]));
    image
        .write_to(
            &mut std::io::Cursor::new(&mut source),
            image::ImageFormat::Png,
        )
        .unwrap();
    let rows = json!([
        [[[1, 1], [12, 1], [12, 8], [1, 8]]],
        ["7. 应收款项"],
        [0.998]
    ]);
    let prepared = table_hints::payload(
        &source,
        &json!({
            "ocr_result":rows,"fill_images":[],"formulas":[],
            "skip_text_in_image":true,"inline_left":"$","inline_right":"$"
        }),
    )
    .unwrap();
    assert_eq!(prepared.metadata["ocr_result"], rows);
    assert_eq!(
        image::load_from_memory(&prepared.png).unwrap().to_rgb8(),
        image
    );
}
#[test]
fn old_table_formula_and_image_hints() {
    for c in fixture()["plans"].as_array().unwrap() {
        assert_eq!(
            table_hints::plan(&c["request"]).unwrap(),
            c["expected"],
            "{}",
            c["request"]
        );
    }
}
#[test]
fn old_table_model_payload_and_inclusive_white_pixels() {
    for c in fixture()["payloads"].as_array().unwrap() {
        let source = std::fs::read(root().join(c["source"].as_str().unwrap())).unwrap();
        let out = table_hints::payload(&source, &c["request"]).unwrap();
        assert_eq!(out.metadata, c["expected"], "{}", c["request"]);
        let p = image::load_from_memory(&out.png).unwrap().to_rgb8();
        assert_eq!(
            json!({"width":p.width(),"height":p.height(),"pixels_sha256":format!("{:x}",Sha256::digest(p.as_raw()))}),
            c["image"],
            "{}",
            c["request"]
        );
    }
}
#[test]
fn table_hint_errors_and_budgets_are_atomic() {
    let f = fixture();
    let c = &f["plans"][0];
    let mut req = c["request"].clone();
    req["scale"] = json!(0);
    let before = req.clone();
    assert!(table_hints::plan(&req).is_err());
    assert_eq!(req, before);
    req = c["request"].clone();
    req["formulas"] = json!([{"bbox":[1,2]}]);
    assert!(table_hints::plan(&req).is_err());
    let c = f["payloads"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| {
            c["request"]["ocr_result"]
                .as_array()
                .is_some_and(|a| a.len() == 3 && a[0].as_array().is_some_and(|a| !a.is_empty()))
        })
        .unwrap();
    let source = std::fs::read(root().join(c["source"].as_str().unwrap())).unwrap();
    let mut req = c["request"].clone();
    req["ocr_result"][2] = json!([]);
    assert!(table_hints::payload(&source, &req).is_err());
    req = c["request"].clone();
    req["skip_text_in_image"] = json!("yes");
    assert!(table_hints::payload(&source, &req).is_err());
    req = c["request"].clone();
    req["fill_images"] = json!(vec![
        json!({"uuid":"x","ocr_bbox":[[0,0],[1,0],[1,1],[0,1]]});
        1001
    ]);
    assert!(table_hints::payload(&source, &req).is_err());
    assert!(table_hints::payload(b"bad", &c["request"]).is_err());
}
