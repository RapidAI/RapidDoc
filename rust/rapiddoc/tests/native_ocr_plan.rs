use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn load(name: &str) -> Value {
    serde_json::from_slice(
        &std::fs::read(root().join(format!("build/migration/{name}.json"))).unwrap(),
    )
    .unwrap()
}
#[test]
fn actual_old_native_boxes_and_numeric_boundaries() {
    let f = load("native-ocr-plan");
    assert_eq!(f["boxes"].as_array().unwrap().len(), 182);
    for (i, c) in f["boxes"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::native_ocr::boxes(&c["request"]).unwrap(),
            c["expected"],
            "case {i}"
        );
    }
}
#[test]
fn actual_old_native_model_and_detection_inputs() {
    let f = load("native-ocr-plan");
    let base = load("legacy-model");
    assert_eq!(f["plans"].as_array().unwrap().len(), 54);
    for (i, c) in f["plans"].as_array().unwrap().iter().enumerate() {
        let b = &base["crops"][c["crop_case"].as_u64().unwrap() as usize];
        let png = std::fs::read(root().join(b["source"].as_str().unwrap())).unwrap();
        let crops = rapiddoc::legacy_model::crops::prepare(&b["request"], &png).unwrap();
        let plan = rapiddoc::legacy_model::native_ocr::plan(&c["request"]).unwrap();
        let pngs = plan["detect"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                crops
                    .images
                    .iter()
                    .find(|(k, _)| k == t["source"].as_str().unwrap())
                    .unwrap()
                    .1
                    .clone()
            })
            .collect::<Vec<_>>();
        let output = rapiddoc::legacy_model::native_ocr::detection::prepare(
            &json!({"plan":plan,"page_index":3}),
            &pngs,
        )
        .unwrap();
        assert_eq!(output.plan, c["expected"], "plan {i}");
        let images=output.images.into_iter().map(|(key,png)|{let image=image::load_from_memory(&png).unwrap();assert_eq!(image.color(),image::ColorType::Rgb8);json!({"key":key,"width":image.width(),"height":image.height(),"pixels_sha256":format!("{:x}",Sha256::digest(image.into_bytes()))})}).collect::<Vec<_>>();
        assert_eq!(json!(images), c["images"], "pixels {i}");
    }
}
