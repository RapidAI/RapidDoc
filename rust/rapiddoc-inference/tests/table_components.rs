use rapiddoc_inference::{table_components::analyze, unet::Mask, unet_lines};
#[test]
fn row_endpoint_cache_does_not_leak_between_components() {
    let (w, h) = (256, 192);
    let mut pixels = vec![0; w * h];
    for y in 30..170 {
        pixels[y * w + 180] = 1;
    }
    for i in 0..40 {
        pixels[(160 - i) * w + 20 + i] = 1;
    }
    assert!(analyze(&pixels, w, h, None, 10)
        .unwrap()
        .polygons
        .is_empty());
    assert_eq!(
        analyze(&pixels, w, h, Some(1), 10).unwrap().lines[0],
        [180., 169., 180., 30.]
    );
}
#[test]
fn invalid_mask_and_empty_geometry() {
    let mask = Mask {
        width: 8,
        height: 8,
        classes: vec![0; 64],
    };
    let result = unet_lines::prepare(&mask, 16, 16, true).unwrap();
    assert!(result
        .horizontal
        .iter()
        .chain(&result.vertical)
        .all(|v| *v == 0));
    assert!(analyze(&[0], 8, 8, None, 1).is_err());
    assert!(analyze(&[0], 1, 1, Some(2), 1).is_err());
}

#[test]
fn bbtd_component_order_is_not_individual_pixel_row_order() {
    let mut mask = vec![0; 256 * 192];
    for (x0, y0) in [(130, 2), (10, 3)] {
        for y in y0..y0 + 22 {
            for x in x0..x0 + 22 {
                mask[y * 256 + x] = 1;
            }
        }
    }
    let regions = analyze(&mask, 256, 192, None, 10).unwrap();
    assert_eq!(regions.polygons.len(), 2);
    assert_eq!(regions.polygons[0][0], [10., 3.]);
    assert_eq!(regions.polygons[1][0], [130., 2.]);
}
