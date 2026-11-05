use rapiddoc_inference::detection_preprocess::{prepare, DetectionConfig, LimitType};

#[test]
fn minimum_and_maximum_limits_keep_existing_shape_contract() {
    let data = vec![128; 40 * 80 * 3];
    for (mode, shape) in [
        (LimitType::Minimum, vec![1, 3, 64, 128]),
        (LimitType::Maximum, vec![1, 3, 32, 64]),
    ] {
        let config = DetectionConfig {
            side: 64,
            mode,
            ..DetectionConfig::default()
        };
        let result = prepare(&data, 80, 40, &config).expect("prepared");
        assert_eq!(result.shape, shape);
        assert!(result.values.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn zero_normalizer_and_zero_aligned_sides_are_not_empty_success() {
    let config = DetectionConfig {
        std: [0., 1., 1.],
        ..DetectionConfig::default()
    };
    assert!(prepare(&[0; 3], 1, 1, &config).is_err());
    assert!(prepare(&[0; 3], 1, 1, &DetectionConfig::default()).is_err());
}
