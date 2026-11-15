use rapiddoc_inference::recognition_batch::plan;
#[test]
fn mixed_widths_keep_every_crop_in_its_own_bucket() {
    let sizes: Vec<_> = (1..=40).map(|i| [i * 47, 48]).collect();
    let batches = plan(&sizes, 6, 16_000_000).unwrap();
    let mut seen = Vec::new();
    for batch in batches {
        let first = batch.indices[0];
        let bucket = |i: usize| ((sizes[i][0] as f64 / 320.).ceil() as usize).max(1);
        for &i in &batch.indices {
            assert_eq!(bucket(i), bucket(first));
            assert!(batch.width >= sizes[i][0].max(320));
            seen.push(i);
        }
        assert!(batch.indices.len() <= 6);
    }
    seen.sort();
    assert_eq!(seen, (0..sizes.len()).collect::<Vec<_>>());
}
#[test]
fn short_title_is_not_padded_to_long_text_width() {
    let sizes = [[438, 79], [960, 48], [1056, 48], [1152, 48], [480, 48], [384, 48], [240, 48]];
    let batches = plan(&sizes, 6, 16_000_000).unwrap();
    assert_eq!(batches.iter().find(|b| b.indices.contains(&0)).unwrap().width, 320);
    let mut seen: Vec<_> = batches.iter().flat_map(|b| b.indices.iter().copied()).collect();
    seen.sort();
    assert_eq!(seen, (0..sizes.len()).collect::<Vec<_>>());
    assert!(batches.iter().all(|b| b.indices.len() <= 6));
}
#[test]
fn batch_pixel_and_tensor_budgets_fail_before_allocating() {
    for n in [0, 65] {
        assert!(plan(&[], n, 16_000_000).is_err());
    }
    assert!(plan(&[[0, 48]], 6, 16_000_000).is_err());
    assert!(plan(&[[16_000_001, 1]], 6, 16_000_000).is_err());
    assert!(plan(&[[100, 48]], 6, 1).is_err());
    assert!(plan(&[[5000, 48]], 6, 16_000_000).is_err());
    assert!(plan(&vec![[10, 48]; 1001], 6, 16_000_000).is_err());
}
