use serde_json::{json, Value};
use sha2::{Digest, Sha256};
#[test]
fn actual_old_table_raw_crops() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/table-text.json")).unwrap(),
    )
    .unwrap();
    for c in f["crops"].as_array().unwrap() {
        let png = std::fs::read(root.join(c["source"].as_str().unwrap())).unwrap();
        let boxes: Vec<[[f32; 2]; 4]> = serde_json::from_value(c["boxes"].clone()).unwrap();
        let out = rapiddoc_inference::crop::pipeline_png_batch(&png, &boxes).unwrap();
        let actual:Vec<_>=out.into_iter().map(|p|{let raw=image::load_from_memory(&p.png).unwrap().to_rgb8();json!({"width":p.width,"height":p.height,"pixels_sha256":format!("{:x}",Sha256::digest(raw.as_raw()))})}).collect();
        assert_eq!(json!(actual), c["expected"]);
    }
}
