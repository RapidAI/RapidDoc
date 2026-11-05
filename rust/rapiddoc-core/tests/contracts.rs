use rapiddoc_core::{ErrorCode, Limits, PageRange, Rect};

#[test]
fn geometry_rejects_nonfinite_and_inverted_coordinates() {
    for input in [
        [0., 0., f64::NAN, 1.],
        [0., 0., 1., f64::INFINITY],
        [2., 0., 1., 1.],
    ] {
        assert!(Rect::try_from(input).is_err());
    }
    assert!(serde_json::from_str::<Rect>("[2,0,1,1]").is_err());
    let rect = Rect::try_from([0., 0., 2., 3.]).expect("valid rect");
    assert_eq!(
        serde_json::from_str::<Rect>(&serde_json::to_string(&rect).expect("serialize"))
            .expect("deserialize"),
        rect
    );
}

#[test]
fn page_range_preserves_original_indices() {
    assert_eq!(
        PageRange {
            start: 2,
            end: Some(4)
        }
        .resolve(5)
        .expect("valid range"),
        2..4
    );
    assert!(PageRange {
        start: 3,
        end: Some(2)
    }
    .resolve(5)
    .is_err());
    assert!(PageRange {
        start: 0,
        end: Some(6)
    }
    .resolve(5)
    .is_err());
}

#[test]
fn empty_and_oversized_inputs_have_distinct_codes() {
    let limits = Limits {
        max_input_bytes: 2,
        max_pages: 1,
        ..Limits::default()
    };
    assert_eq!(
        limits.check_input(&[]).expect_err("empty").code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        limits.check_input(&[1, 2, 3]).expect_err("large").code,
        ErrorCode::ResourceLimit
    );
    assert!(limits.check_input(&[1, 2]).is_ok());
}
