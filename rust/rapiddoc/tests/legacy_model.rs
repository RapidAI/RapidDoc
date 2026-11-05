use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn fixture() -> Value {
    serde_json::from_slice(
        &std::fs::read(root().join("build/migration/legacy-model.json")).unwrap(),
    )
    .unwrap()
}
#[test]
fn actual_old_layout_conversion_and_filtering() {
    let f = fixture();
    assert_eq!(f["cases"].as_array().unwrap().len(), 258);
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::convert(&c["request"]).unwrap(),
            c["expected"],
            "conversion {i}"
        );
    }
    for (i, c) in f["filters"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::filter(&c["request"]).unwrap(),
            c["expected"],
            "filter {i}"
        );
    }
}
#[test]
fn actual_old_task_plans_and_crop_pixels() {
    let f = fixture();
    assert_eq!(f["crops"].as_array().unwrap().len(), 60);
    for (i, c) in f["crops"].as_array().unwrap().iter().enumerate() {
        let bytes = std::fs::read(root().join(c["source"].as_str().unwrap())).unwrap();
        let output = rapiddoc::legacy_model::crops::prepare(&c["request"], &bytes).unwrap();
        assert_eq!(output.plan, c["expected"], "plan {i}");
        let actual=output.images.into_iter().map(|(key,png)|{
            let image=image::load_from_memory(&png).unwrap();assert_eq!(image.color(),image::ColorType::Rgb8);
            json!({"key":key,"width":image.width(),"height":image.height(),"pixels_sha256":format!("{:x}",Sha256::digest(image.into_bytes()))})
        }).collect::<Vec<_>>();
        assert_eq!(json!(actual), c["images"], "crop pixels {i}");
    }
}
