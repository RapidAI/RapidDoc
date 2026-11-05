use rapiddoc_core::{
    geometry::{bounds, intersect, page_transform, Affine},
    Rect,
};

#[test]
fn cropped_rotated_pages_map_corners_and_roundtrip() {
    let page = Rect::try_from([10., 20., 210., 320.]).expect("page");
    for (rotation, expected) in [
        (0, [0., 300.]),
        (90, [0., 0.]),
        (180, [200., 0.]),
        (270, [300., 200.]),
    ] {
        let transform = page_transform(page, rotation).expect("rotation");
        assert_eq!(transform.point([10., 20.]).expect("point"), expected);
        let point = [55., 123.];
        assert_eq!(
            transform
                .inverse()
                .expect("inverse")
                .point(transform.point(point).expect("forward"))
                .expect("back"),
            point
        );
        let box_values: [f64; 4] = transform.rect(page).expect("rect").into();
        assert_eq!(
            box_values,
            if rotation == 90 || rotation == 270 {
                [0., 0., 300., 200.]
            } else {
                [0., 0., 200., 300.]
            }
        );
    }
    assert!(page_transform(page, 45).is_err());
}

#[test]
fn nested_form_transform_applies_child_then_parent() {
    let child = Affine::new([2., 0., 0., 3., 5., 7.]).expect("child");
    let parent = Affine::new([0., 1., -1., 0., 100., 200.]).expect("parent");
    assert_eq!(
        child
            .then(parent)
            .expect("combined")
            .point([2., 3.])
            .expect("point"),
        parent
            .point(child.point([2., 3.]).expect("child point"))
            .expect("parent point")
    );
    assert!(Affine::new([1., 0., 0., 1., f64::NAN, 0.]).is_err());
    assert!(Affine::new([0.; 6]).expect("singular").inverse().is_err());
    assert!(bounds(&[]).is_err());
}

#[test]
fn disjoint_clipping_stays_empty() {
    let first = Rect::try_from([0., 0., 10., 10.]).expect("first");
    let second = Rect::try_from([20., 20., 30., 30.]).expect("second");
    assert_eq!(
        <[f64; 4]>::from(intersect(first, second).expect("clip")),
        [20., 20., 20., 20.]
    );
}
