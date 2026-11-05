use serde_json::Value;
use sha2::{Digest, Sha256};

#[test]
fn actual_old_formula_pixels_and_tensors() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/formula-preprocess.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(f["cases"].as_array().unwrap().len(), 88);
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        let png = std::fs::read(root.join(c["source"].as_str().unwrap())).unwrap();
        let p = rapiddoc_inference::formula_preprocess::from_png(
            &png,
            c["target_size"][0].as_u64().unwrap() as usize,
            c["target_size"][1].as_u64().unwrap() as usize,
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&p.shape).unwrap(),
            c["shape"],
            "shape {i}"
        );
        assert_eq!(
            serde_json::json!([p.crop_box[2] - p.crop_box[0], p.crop_box[3] - p.crop_box[1]]),
            c["cropped_size"],
            "crop {i}"
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(&p.rgb)),
            c["rgb_sha256"].as_str().unwrap(),
            "pixels {i}"
        );
        let expected = std::fs::read(root.join(c["tensor"].as_str().unwrap())).unwrap();
        assert_eq!(p.values.len() * 4, expected.len());
        for (j, (a, b)) in p.values.iter().zip(expected.as_chunks::<4>().0).enumerate() {
            let b = f32::from_le_bytes(*b);
            // 旧OpenCV 5.0的IPP与公开SIMD具有不同FMA次序；像素仍逐字节严格一致。
            assert!((a - b).abs() <= 1e-6, "tensor {i} at {j}: {a} / {b}");
        }
    }
}

#[test]
fn formula_preprocess_limits() {
    use rapiddoc_inference::formula_preprocess::prepare;
    assert!(prepare(&[0, 0, 0], 1, 1, 0, 32).is_err());
    assert!(prepare(&[0, 0, 0], 1, 1, 2049, 32).is_err());
    assert!(prepare(&[], 1, 1, 32, 32).is_err());
    // 裁边前判断输入大小，不分配超预算源图。
    assert!(prepare(&[], 4_000_001, 1, 32, 32).is_err());
    assert!(prepare(&[], 65_537, 1, 2, 2).is_err());
    // 旧先放大小边再缩略的中间图也受预算约束。
    let mut rgb = vec![0; 2000 * 3];
    rgb[0] = 255;
    assert!(prepare(&rgb, 2000, 1, 768, 768).is_err());
}
