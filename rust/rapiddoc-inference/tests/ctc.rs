use rapiddoc_inference::ctc::decode;

#[test]
fn compact_gpu_reductions_preserve_full_ctc_results_and_validation() {
    use rapiddoc_inference::ctc::decode_compact;
    let alphabet = vec!["blank".into(), "中".into(), "a".into()];
    let values = [
        0., 0.9, 0.9, 0., 0.8, 0.1, 1., 0., 0., 0., 0.7, 0.1, 1., 0., 0., 1., 0., 0., 0., 0., 1.,
        0., 1., 0.,
    ];
    let indices = [1, 1, 0, 1, 0, 0, 2, 1];
    let scores = [0.9, 0.8, 1., 0.7, 1., 1., 1., 1.];
    assert_eq!(
        serde_json::to_value(decode(2, 4, 3, &values, &alphabet).unwrap()).unwrap(),
        serde_json::to_value(decode_compact(2, 4, &indices, &scores, &alphabet).unwrap()).unwrap()
    );
    for index in [-1, 3] {
        assert!(decode_compact(1, 1, &[index], &[1.], &alphabet).is_err());
    }
    assert!(decode_compact(1, 1, &[1], &[f32::NAN], &alphabet).is_err());
    assert!(decode_compact(1, 2, &[1], &[1.], &alphabet).is_err());
    assert!(decode_compact(usize::MAX, 2, &[], &[], &alphabet).is_err());
}

#[test]
fn blank_separates_repeated_characters_and_ties_choose_first() {
    let alphabet = vec!["blank".into(), "中".into(), " ".into()];
    let values = vec![
        0., 0.9, 0.1, 0., 0.8, 0.1, 0.9, 0.1, 0., 0., 0.7, 0.1, 0., 0.6, 0.6,
    ];
    let result = decode(1, 5, 3, &values, &alphabet).expect("decode");
    assert_eq!(result[0].text, "中中");
    assert_eq!(result[0].token_indices, vec![1, 1]);
    assert_eq!(result[0].timesteps, vec![0, 3]);
    assert_eq!(result[0].confidence, 0.8);
}

#[test]
fn empty_prediction_and_invalid_dictionary_remain_distinct() {
    let alphabet = vec!["blank".into(), "a".into()];
    let result = decode(1, 2, 2, &[1., 0., 1., 0.], &alphabet).expect("blank");
    assert_eq!(result[0].text, "");
    assert_eq!(result[0].confidence, 0.);
    assert!(decode(1, 1, 3, &[0., 0., 1.], &alphabet).is_err());
    assert!(decode(1, 1, 2, &[f32::NAN, 0.], &alphabet).is_err());
}
