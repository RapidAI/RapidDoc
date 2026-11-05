use rapiddoc_core::ErrorCode;
use rapiddoc_inference::orientation::prepare_rgb;

#[test]
fn orientation_limits_are_checked_before_resizing() {
    assert_eq!(
        prepare_rgb(&[], 1, 1, 10, 150528).err().unwrap().code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        prepare_rgb(&[0, 0, 0], 1, 1, 100, 150528)
            .err()
            .unwrap()
            .code,
        ErrorCode::ResourceLimit
    );
    assert_eq!(
        prepare_rgb(&[0, 0, 0], 1, 1, 65536, 150527)
            .err()
            .unwrap()
            .code,
        ErrorCode::ResourceLimit
    );
    let tensor = prepare_rgb(&[0, 127, 255], 1, 1, 65536, 150528).unwrap();
    assert_eq!(tensor.shape, vec![1, 3, 224, 224]);
    assert_eq!(tensor.values[0], -0.485 / 0.229);
    assert_eq!(tensor.values[224 * 224 * 2], (1. - 0.406) / 0.225);
}
