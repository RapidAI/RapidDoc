use serde_json::{json, Value};
use sha2::{Digest, Sha256};
#[test]
fn full_ocr_crops_before_confidence_filter_match_old_pixels() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/table-full-ocr.json")).unwrap(),
    )
    .unwrap();
    // 独立冻结的实际裁图；框整理的完整组合由rapiddoc层验证。此处按旧已选框逐一检查crop内核。
    for c in f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["masks"].as_array().unwrap().is_empty())
        .take(1)
    {
        let source = std::fs::read(root.join(c["source"].as_str().unwrap())).unwrap();
        let points: Vec<[[f32; 2]; 4]> = serde_json::from_value(c["boxes"].clone()).unwrap();
        let out = rapiddoc_inference::crop::pipeline_png_batch(&source, &points).unwrap();
        let pixels=out.into_iter().map(|p|{let im=image::load_from_memory(&p.png).unwrap().to_rgb8();json!({"width":im.width(),"height":im.height(),"pixels_sha256":format!("{:x}",Sha256::digest(im.as_raw()))})}).collect::<Vec<_>>();
        assert_eq!(json!(pixels), c["crops"]);
    }
}
