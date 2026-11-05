use rapiddoc_inference::crop::{pipeline_quadrilateral, with_sampling, Sampling};
use sha2::{Digest, Sha256};

#[test]
fn real_skew_table_matches_opencv_413_pixels() {
    let source = image::load_from_memory(include_bytes!(
        "../../../tests/fixtures/opencv4_crop/source.png"
    ))
    .unwrap()
    .to_rgb8();
    let expected = image::load_from_memory(include_bytes!(
        "../../../tests/fixtures/opencv4_crop/expected.png"
    ))
    .unwrap()
    .to_rgb8();
    let crop = with_sampling(Sampling::OpenCv4, || {
        pipeline_quadrilateral(
            source.as_raw(),
            source.width() as usize,
            source.height() as usize,
            [[379., 16.], [518., 19.], [518., 63.], [378., 60.]],
            4_000_000,
        )
    })
    .unwrap();
    assert_eq!(
        [crop.width, crop.height],
        [expected.width() as usize, expected.height() as usize]
    );
    assert_eq!(crop.bgr, expected.as_raw().as_slice());
}

#[test]
fn independent_opencv4_geometries_and_borders() {
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/opencv4_crop/reference.json"
    ))
    .unwrap();
    let source: Vec<u8> = (0..256 * 192 * 3)
        .map(|i| ((i * 37 + i / 13) % 256) as u8)
        .collect();
    let mut differences = vec![];
    with_sampling(Sampling::OpenCv4, || {
        for (i, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
            let points = serde_json::from_value(case["points"].clone()).unwrap();
            let result = pipeline_quadrilateral(&source, 256, 192, points, 4_000_000).unwrap();
            if result.width != case["width"].as_u64().unwrap() as usize
                || result.height != case["height"].as_u64().unwrap() as usize
                || format!("{:x}", Sha256::digest(&result.bgr)) != case["sha256"].as_str().unwrap()
            {
                differences.push(i);
            }
        }
    });
    assert!(
        differences.is_empty(),
        "OpenCV pixel differences in cases {differences:?}"
    );
}

#[test]
fn sampling_scope_restores_the_previous_profile_after_panic() {
    let source = image::load_from_memory(include_bytes!(
        "../../../tests/fixtures/opencv4_crop/source.png"
    ))
    .unwrap()
    .to_rgb8();
    let crop = || {
        pipeline_quadrilateral(
            source.as_raw(),
            source.width() as usize,
            source.height() as usize,
            [[379., 16.], [518., 19.], [518., 63.], [378., 60.]],
            4_000_000,
        )
        .unwrap()
        .bgr
    };
    let original = crop();
    with_sampling(Sampling::OpenCv4, || {
        let expected = crop();
        assert_ne!(original, expected);
        let panic = std::panic::catch_unwind(|| {
            with_sampling(Sampling::OpenCv5, || {
                assert_eq!(crop(), original);
                panic!("scope restoration probe");
            })
        });
        assert!(panic.is_err());
        assert_eq!(crop(), expected);
    });
    assert_eq!(crop(), original);
}
