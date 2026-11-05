use rapiddoc_core::ErrorCode;
use rapiddoc_inference::crop::quadrilateral;

#[test]
fn malformed_and_over_budget_crops_have_stable_errors() {
    let bgr = vec![255; 32 * 32 * 3];
    let region = [[0., 0.], [31., 0.], [31., 31.], [0., 31.]];
    assert_eq!(
        quadrilateral(&[], 32, 32, region, 1024).err().unwrap().code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        quadrilateral(&bgr, 32, 32, region, 100).err().unwrap().code,
        ErrorCode::ResourceLimit
    );
    let mut region = region;
    region[0][0] = f32::NAN;
    assert!(quadrilateral(&bgr, 32, 32, region, 1024).is_err());
}

#[test]
fn axis_crop_and_vertical_rotation_preserve_pixel_coordinates() {
    let bgr: Vec<_> = (0..8 * 8 * 3).map(|v| v as u8).collect();
    let image = quadrilateral(&bgr, 8, 8, [[1., 1.], [4., 1.], [4., 6.], [1., 6.]], 64).unwrap();
    assert_eq!((image.width, image.height), (5, 3));
    assert_eq!(&image.bgr[0..3], &bgr[(8 + 3) * 3..(8 + 3) * 3 + 3]);
}
