use serde_json::Value;
use sha2::{Digest, Sha256};
#[test]
fn old_pipeline_rotation_ratio_and_crop_pixels() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/model-ocr-execution.json")).unwrap(),
    )
    .unwrap();
    let mut checked = 0;
    for c in f["preparations"].as_array().unwrap() {
        let images = c["images"].as_array().unwrap();
        if images.is_empty() {
            continue;
        }
        let png = std::fs::read(root.join(c["source"].as_str().unwrap())).unwrap();
        let points = c["expected"]["processed"][0]["boxes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p[2][0].as_f64().unwrap() - p[0][0].as_f64().unwrap() >= 3.);
        for (p, e) in points.zip(images) {
            let points: [[f32; 2]; 4] = serde_json::from_value(p.clone()).unwrap();
            let (w, h, bytes) =
                rapiddoc_inference::crop::pipeline_png(&png, points, 4_000_000).unwrap();
            let image = image::load_from_memory(&bytes).unwrap();
            assert_eq!(image.color(), image::ColorType::Rgb8);
            assert_eq!(w, e["width"].as_u64().unwrap() as usize);
            assert_eq!(h, e["height"].as_u64().unwrap() as usize);
            assert_eq!(
                format!("{:x}", Sha256::digest(image.into_bytes())),
                e["pixels_sha256"].as_str().unwrap()
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 16);
}
