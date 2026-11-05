use rapiddoc_core::ErrorCode;
use rapiddoc_inference::unet::prepare;

#[test]
fn unet_budget_shape_and_thin_image_contract() {
    let tensor = prepare(&[0; 3], 1, 1, 3 * 1024 * 1024).unwrap();
    assert_eq!(tensor.shape, vec![1, 3, 1024, 1024]);
    assert_eq!(
        tensor.values[0],
        (-123.675f32 as f64 / 58.395f32 as f64) as f32
    );
    assert_eq!(
        prepare(&[0; 3], 1, 1, 100).err().unwrap().code,
        ErrorCode::ResourceLimit
    );
    assert_eq!(
        prepare(&[], 1, 1, 100).err().unwrap().code,
        ErrorCode::InvalidInput
    );
    assert!(prepare(&vec![0; 10_000 * 3], 1, 10_000, 16_000_000).is_err());
}
