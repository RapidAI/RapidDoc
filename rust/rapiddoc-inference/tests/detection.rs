use rapiddoc_core::ErrorCode;
use rapiddoc_inference::detection::{boxes, sort_reading_order, Config, Detection};

#[test]
fn live_probability_map_preserves_python_boundary_and_recognition_width() {
    let data = include_bytes!("../../../tests/fixtures/detector_boundary/map.f32");
    let probabilities: Vec<f32> = data
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let default_profile = boxes(&probabilities, 384, 192, 384, 192, &Config::body()).unwrap();
    assert_eq!(
        default_profile[0].points,
        [[51., 49.], [325., 49.], [325., 93.], [51., 93.]]
    );
    let result = rapiddoc_inference::crop::with_sampling(
        rapiddoc_inference::crop::Sampling::OpenCv4,
        || boxes(&probabilities, 384, 192, 384, 192, &Config::body()),
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].points,
        [[52., 49.], [325., 49.], [325., 93.], [52., 93.]]
    );
}

#[test]
fn body_and_table_keep_the_original_distinct_box_thresholds() {
    let mut probabilities = vec![0.; 32 * 32];
    for y in 8..24 {
        for x in 10..20 {
            probabilities[y * 32 + x] = 0.43;
        }
    }
    let body = Config::body();
    assert_eq!((body.box_threshold, body.unclip_ratio), (0.3, 1.8));
    let table = Config::default();
    assert_eq!((table.box_threshold, table.unclip_ratio), (0.5, 1.6));
    assert_eq!(
        boxes(&probabilities, 32, 32, 32, 32, &body).unwrap().len(),
        1
    );
    assert!(boxes(&probabilities, 32, 32, 32, 32, &table)
        .unwrap()
        .is_empty());
}

#[test]
fn adjacent_line_sort_preserves_scores_and_stable_ties() {
    let mut detections: Vec<_> = [
        (40., 18., 0.1),
        (80., 0., 0.2),
        (20., 9., 0.3),
        (10., 40., 0.4),
    ]
    .into_iter()
    .map(|(x, y, score)| Detection {
        points: [[x, y]; 4],
        score,
    })
    .collect();
    sort_reading_order(&mut detections);
    assert_eq!(
        detections.iter().map(|d| d.score).collect::<Vec<_>>(),
        vec![0.3, 0.1, 0.2, 0.4]
    );
    sort_reading_order(&mut []);
}

#[test]
fn invalid_probabilities_and_work_overflow_never_become_empty_success() {
    let config = Config::default();
    for value in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
        assert_eq!(
            boxes(&[value], 1, 1, 1, 1, &config).unwrap_err().code,
            ErrorCode::InvalidInput
        );
    }
    assert_eq!(
        boxes(&[], usize::MAX, 2, 1, 1, &config).unwrap_err().code,
        ErrorCode::ResourceLimit
    );
    assert_eq!(
        boxes(&[], 0, 1, 1, 1, &config).unwrap_err().code,
        ErrorCode::InvalidInput
    );
    let probability = vec![0.9; 32 * 32];
    let config = Config {
        max_work: 1,
        ..config
    };
    assert_eq!(
        boxes(&probability, 32, 32, 32, 32, &config)
            .unwrap_err()
            .code,
        ErrorCode::ResourceLimit
    );
}

#[test]
fn touching_edge_foreground_is_closed_and_blank_is_empty() {
    let config = Config::default();
    assert!(boxes(&vec![0.; 32 * 32], 32, 32, 64, 64, &config)
        .unwrap()
        .is_empty());
    let result = boxes(&vec![0.9; 32 * 32], 32, 32, 64, 64, &config).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].points,
        [[0., 0.], [63., 0.], [63., 63.], [0., 63.]]
    );
    assert!((result[0].score - 0.9).abs() < 1e-6);
}
