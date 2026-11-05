use rapiddoc_inference::{table_raster as r, table_transformer as t};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/model_switching")
        .join(name)
}
#[test]
fn transformer_python_structure_references() {
    let cases: Value =
        serde_json::from_slice(&std::fs::read(fixture("transformer.json")).unwrap()).unwrap();
    for (i, c) in cases.as_array().unwrap().iter().enumerate() {
        let objects = serde_json::from_value::<Vec<t::Object>>(c["objects"].clone()).unwrap();
        let tokens = serde_json::from_value::<Vec<t::Token>>(c["tokens"].clone()).unwrap();
        let actual = serde_json::to_value(t::decode_objects(&objects, &tokens)).unwrap();
        assert_eq!(actual, c["expected"], "TATR case {i}");
    }
}
#[test]
fn raster_python_pixel_and_stroke_references() {
    let cases: Value =
        serde_json::from_slice(&std::fs::read(fixture("raster.json")).unwrap()).unwrap();
    for c in cases.as_array().unwrap() {
        let path = fixture(c["image"].as_str().unwrap());
        let png = std::fs::read(path).unwrap();
        let image = image::load_from_memory(&png).unwrap().to_rgb8();
        let g = r::gray(image.as_raw());
        assert_eq!(
            format!("{:x}", Sha256::digest(&g)),
            c["gray_sha256"].as_str().unwrap(),
            "gray {}",
            c["image"]
        );
        let edges = r::skew::canny(&g, image.width() as usize, image.height() as usize);
        assert_eq!(
            format!("{:x}", Sha256::digest(&edges)),
            c["canny_sha256"].as_str().unwrap(),
            "Canny {}",
            c["image"]
        );
        let words = serde_json::from_value(c["words"].clone()).unwrap();
        let p = r::prepare(&png, words).unwrap();
        assert!(
            (p.angle - c["angle"].as_f64().unwrap()).abs() < 1e-12,
            "angle {} actual {} expected {}",
            c["image"],
            p.angle,
            c["angle"]
        );
        assert_eq!(json!([p.width, p.height]), json!([c["width"], c["height"]]));
        let mut actual = serde_json::to_value(p.rules)
            .unwrap()
            .as_array()
            .unwrap()
            .clone();
        let mut expected = c["rules"].as_array().unwrap().clone();
        actual.sort_by_key(Value::to_string);
        expected.sort_by_key(Value::to_string);
        assert_eq!(actual, expected, "rules {}", c["image"]);
    }
}
