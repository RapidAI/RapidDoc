//! Independent original pixel references plus raw/PNG detector tensor equality.
use rapiddoc::legacy_model::{crops, native_ocr};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[test]
fn native_detection_pixels_preserve_png_plans_and_original_tensors() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let read = |name: &str| -> Value {
        serde_json::from_slice(
            &std::fs::read(root.join(format!("build/migration/{name}.json"))).unwrap(),
        )
        .unwrap()
    };
    let references = read("native-ocr-plan");
    let base = read("legacy-model");
    let mut checked = 0;
    for case in references["plans"].as_array().unwrap() {
        let b = &base["crops"][case["crop_case"].as_u64().unwrap() as usize];
        let source = std::fs::read(root.join(b["source"].as_str().unwrap())).unwrap();
        let crops = crops::prepare(&b["request"], &source).unwrap();
        let plan = native_ocr::plan(&case["request"]).unwrap();
        let sources: Vec<_> = plan["detect"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                crops
                    .images
                    .iter()
                    .find(|(key, _)| key == t["source"].as_str().unwrap())
                    .unwrap()
                    .1
                    .clone()
            })
            .collect();
        let q = json!({"plan":plan,"page_index":3});
        let png = native_ocr::detection::prepare(&q, &sources).unwrap();
        let refs: Vec<_> = sources.iter().map(Vec::as_slice).collect();
        let raw = native_ocr::detection::prepare_pixels(&q, &refs).unwrap();
        assert_eq!(raw.plan, png.plan);
        assert!(raw.images.is_empty());
        assert_eq!(raw.pixels.len(), png.images.len());
        for (((id, rgb), (png_id, png)), expected) in raw
            .pixels
            .iter()
            .zip(&png.images)
            .zip(case["images"].as_array().unwrap())
        {
            assert_eq!(id, png_id);
            assert_eq!(id, expected["key"].as_str().unwrap());
            let decoded = image::load_from_memory(png).unwrap().to_rgb8();
            assert_eq!(
                (rgb.w, rgb.h),
                (decoded.width() as usize, decoded.height() as usize)
            );
            assert_eq!(&rgb.data, decoded.as_raw());
            assert_eq!(
                format!("{:x}", Sha256::digest(&rgb.data)),
                expected["pixels_sha256"].as_str().unwrap()
            );
            let mut bgr = rgb.data.clone();
            for p in bgr.as_chunks_mut::<3>().0 {
                p.swap(0, 2);
            }
            let mut decoded_bgr = decoded.into_raw();
            for p in decoded_bgr.as_chunks_mut::<3>().0 {
                p.swap(0, 2);
            }
            let config = rapiddoc_inference::detection_preprocess::DetectionConfig::default();
            let a = rapiddoc_inference::detection_preprocess::prepare(&bgr, rgb.w, rgb.h, &config)
                .unwrap();
            let b = rapiddoc_inference::detection_preprocess::prepare(
                &decoded_bgr,
                rgb.w,
                rgb.h,
                &config,
            )
            .unwrap();
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.values, b.values);
            checked += 1;
        }
    }
    assert_eq!(checked, 75);
}
