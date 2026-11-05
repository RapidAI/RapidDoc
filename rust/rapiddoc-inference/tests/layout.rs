use rapiddoc_core::ErrorCode;
use rapiddoc_inference::{layout::prepare_rgb, layout_post::rectangles};

#[test]
fn cubic_tensor_opencv_matches_and_preexisting_tiny_upscale_cases() {
    use sha2::{Digest, Sha256};
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/layout_resize_reference.json"
    ))
    .unwrap();
    let mut differences = vec![];
    for (i, c) in reference["cases"].as_array().unwrap().iter().enumerate() {
        let [w, h]: [usize; 2] = serde_json::from_value(c["source"].clone()).unwrap();
        let [tw, th]: [usize; 2] = serde_json::from_value(c["target"].clone()).unwrap();
        let pixels: Vec<u8> = (0..w * h * 3)
            .map(|i| ((i * 37 + i / 13) % 256) as u8)
            .collect();
        let tensor = prepare_rgb(&pixels, w, h, tw, th, 16_000_000).unwrap();
        let bytes: Vec<u8> = tensor.values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let digest = format!("{:x}", Sha256::digest(bytes));
        let expected = c.get("preexisting_native_sha256").unwrap_or(&c["sha256"]);
        if c.get("preexisting_native_sha256").is_some() {
            assert_ne!(expected, &c["sha256"]);
        }
        if digest != expected.as_str().unwrap() {
            differences.push(i);
        }
    }
    assert!(
        differences.is_empty(),
        "OpenCV tensor differences: {differences:?}"
    );
}

#[test]
fn layout_limits_and_malformed_results_are_explicit_errors() {
    assert_eq!(
        prepare_rgb(&[], 1, 1, 32, 32, 10000).err().unwrap().code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        prepare_rgb(&[0, 0, 0], 1, 1, 800, 800, 100)
            .err()
            .unwrap()
            .code,
        ErrorCode::ResourceLimit
    );
    let labels = vec!["text".into()];
    assert_eq!(
        rectangles(&[[0.; 7]; 1001], &labels, 640, 256, 0.5)
            .err()
            .unwrap()
            .code,
        ErrorCode::ResourceLimit
    );
    assert_eq!(
        rectangles(&[[1., 0.9, 0., 0., 10., 10., 0.]], &labels, 640, 256, 0.5)
            .err()
            .unwrap()
            .code,
        ErrorCode::InvalidInput
    );
    assert!(rectangles(&[], &labels, 640, 256, f32::NAN).is_err());
}

#[test]
fn score_ties_use_last_source_index_deterministically() {
    let result = rectangles(
        &[
            [0., 0.9, 0., 0., 100., 100., 1.],
            [0., 0.9, 0., 0., 100., 100., 0.],
        ],
        &["text".into()],
        640,
        256,
        0.5,
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].order, 1);
}
