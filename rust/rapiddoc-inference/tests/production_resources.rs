use rapiddoc_core::resources::python_compatible;

#[test]
fn production_formula_text_is_not_truncated_at_diagnostic_length() {
    let text = "x+y ".repeat(20_000);
    assert!(rapiddoc_inference::formula::post_process(&text).is_err());
    {
        let _scope = python_compatible();
        assert_eq!(
            rapiddoc_inference::formula::post_process(&text).unwrap(),
            text
        );
    }
    assert!(rapiddoc_inference::formula::post_process(&text).is_err());
}

#[test]
fn production_ocr_keeps_all_crops_and_accepts_positive_large_batches() {
    let sizes = vec![[8, 48]; 1001];
    assert!(rapiddoc_inference::recognition_batch::plan(&sizes, 65, 32_000_000).is_err());
    let _scope = python_compatible();
    let batches = rapiddoc_inference::recognition_batch::plan(&sizes, 65, 32_000_000).unwrap();
    let mut indices: Vec<_> = batches
        .iter()
        .flat_map(|b| b.indices.iter().copied())
        .collect();
    indices.sort_unstable();
    assert_eq!(indices, (0..1001).collect::<Vec<_>>());
    assert!(rapiddoc_inference::recognition_batch::plan(&sizes, 0, 32_000_000).is_err());
    // An explicit session/tensor capacity is still enforced.
    assert!(rapiddoc_inference::recognition_batch::plan(&sizes, 65, 1).is_err());
}

#[test]
fn production_resize_accepts_real_dimensions_but_rejects_overflow() {
    assert!(rapiddoc_core::pillow_resize::size(8192, 2048).is_err());
    let _scope = python_compatible();
    assert_eq!(
        rapiddoc_core::pillow_resize::size(8192, 2048).unwrap(),
        16_777_216
    );
    assert!(rapiddoc_core::pillow_resize::size(usize::MAX, 2).is_err());
    assert!(rapiddoc_core::pillow_resize::size(0, 1).is_err());
}
