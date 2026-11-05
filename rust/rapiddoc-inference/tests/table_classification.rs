use rapiddoc_core::ErrorCode;
use rapiddoc_inference::table_classification::{decode, prepare_bgr, OutputType, Variant};

#[test]
fn qanything_keeps_constant_gray_and_rejects_large_intermediate() {
    let prepared = Variant::Qanything
        .prepare(&[255; 3], 1, 1, 16000000)
        .unwrap();
    assert_eq!(prepared.values[0], (1f32 - 0.485) / 0.229);
    assert_eq!(prepared.values[224 * 224], (1f32 - 0.456) / 0.224);
    assert_eq!(
        Variant::Qanything
            .prepare(&vec![0; 3 * 20000], 1, 20000, 16000000)
            .err()
            .unwrap()
            .code,
        ErrorCode::ResourceLimit
    );
}
#[test]
fn probabilities_are_not_softmaxed_twice_and_ties_choose_wired() {
    let p = decode(&[0.9, 0.1], OutputType::Probabilities).unwrap();
    assert_eq!((p.label, p.score), ("wired", 0.9f32 as f64));
    assert!(p.score < 0.9);
    assert_eq!(
        decode(&[0.5, 0.5], OutputType::Probabilities)
            .unwrap()
            .label,
        "wired"
    );
    assert_eq!(
        decode(&[-f32::MAX, f32::MAX], OutputType::Logits)
            .unwrap()
            .score,
        1.
    );
    assert_eq!(
        decode(&[2., -1.], OutputType::Probabilities)
            .err()
            .unwrap()
            .code,
        ErrorCode::InvalidInput
    );
    assert!(decode(&[f32::NAN, 0.], OutputType::Logits).is_err());
}
#[test]
fn resize_and_tensor_budgets_are_checked_before_work() {
    assert_eq!(
        prepare_bgr(&vec![0; 3 * 256], 1, 256, 16000000)
            .err()
            .unwrap()
            .code,
        ErrorCode::ResourceLimit
    );
    assert_eq!(
        prepare_bgr(&[255; 3], 1, 1, 10).err().unwrap().code,
        ErrorCode::ResourceLimit
    );
    assert_eq!(
        prepare_bgr(&[255; 2], 1, 1, 16000000).err().unwrap().code,
        ErrorCode::InvalidInput
    );
}
