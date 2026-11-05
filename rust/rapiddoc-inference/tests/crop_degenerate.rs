use rapiddoc_inference::crop::{pipeline_quadrilateral, with_sampling, Sampling};
use sha2::{Digest, Sha256};

#[test]
fn collapsed_and_nonconvex_crops_match_actual_python() {
    let source: Vec<u8> = (0..64 * 48 * 3)
        .map(|i| ((i * 37 + i / 13) % 256) as u8)
        .collect();
    for (profile, reference) in [
        (
            Sampling::OpenCv4,
            include_str!("../../../tests/fixtures/opencv4_crop/degenerate4.json"),
        ),
        (
            Sampling::OpenCv5,
            include_str!("../../../tests/fixtures/opencv4_crop/degenerate5.json"),
        ),
    ] {
        let reference: serde_json::Value = serde_json::from_str(reference).unwrap();
        with_sampling(profile, || {
            for case in reference["cases"].as_array().unwrap() {
                let points = serde_json::from_value(case["points"].clone()).unwrap();
                let result = pipeline_quadrilateral(&source, 64, 48, points, 10000).unwrap();
                assert_eq!(
                    (result.width, result.height),
                    (
                        case["width"].as_u64().unwrap() as usize,
                        case["height"].as_u64().unwrap() as usize
                    ),
                    "{} {:?}",
                    case["name"],
                    profile
                );
                assert_eq!(
                    format!("{:x}", Sha256::digest(&result.bgr)),
                    case["sha256"].as_str().unwrap(),
                    "{} {:?}",
                    case["name"],
                    profile
                );
            }
        });
    }
}

#[test]
#[ignore = "requires opt-in Python capture via RAPIDDOC_CROP_REFERENCE"]
fn captured_pdf_crops_match_original_pixels() {
    let root = std::path::PathBuf::from(std::env::var("RAPIDDOC_CROP_REFERENCE").unwrap());
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("cases.json")).unwrap()).unwrap();
    for (i, case) in cases.as_array().unwrap().iter().enumerate() {
        let source = std::fs::read(root.join(format!("{i}-source.bgr"))).unwrap();
        let expected = std::fs::read(root.join(format!("{i}-expected.bgr"))).unwrap();
        let result = with_sampling(Sampling::OpenCv4, || {
            pipeline_quadrilateral(
                &source,
                case["source_shape"][1].as_u64().unwrap() as usize,
                case["source_shape"][0].as_u64().unwrap() as usize,
                serde_json::from_value(case["points"].clone()).unwrap(),
                4_000_000,
            )
        })
        .unwrap();
        assert_eq!(
            (result.width, result.height),
            (
                case["shape"][1].as_u64().unwrap() as usize,
                case["shape"][0].as_u64().unwrap() as usize
            )
        );
        assert_eq!(result.bgr, expected, "real crop {i}");
    }
}
