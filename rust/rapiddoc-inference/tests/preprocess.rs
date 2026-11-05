use rapiddoc_core::ErrorCode;
use rapiddoc_inference::preprocess::{recognition_bgr, RecognitionConfig};

#[test]
fn constant_channels_padding_and_shape_have_explicit_contract() {
    let input = [0, 127, 255].repeat(12);
    let config = RecognitionConfig {
        height: 2,
        minimum_width: 8,
        maximum_width: 8,
        max_pixels: 100,
        max_elements: 100,
    };
    let result = recognition_bgr(&input, 4, 3, &config).expect("tensor");
    assert_eq!(result.shape, vec![1, 3, 2, 8]);
    assert_eq!(result.resized_width, 3);
    assert_eq!(&result.values[..8], &[-1., -1., -1., 0., 0., 0., 0., 0.]);
    assert_eq!(result.values[32], 1.);
}

#[test]
fn corrupt_length_and_oversized_images_are_rejected_before_resize() {
    let config = RecognitionConfig::default();
    assert_eq!(
        recognition_bgr(&[], 1, 1, &config)
            .err()
            .expect("length")
            .code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        recognition_bgr(&[], usize::MAX, 2, &config)
            .err()
            .expect("overflow")
            .code,
        ErrorCode::ResourceLimit
    );
    assert!(recognition_bgr(&[], 0, 1, &config).is_err());
}
