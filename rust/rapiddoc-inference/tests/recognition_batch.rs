use rapiddoc_inference::recognition_batch::plan;
use serde_json::Value;
#[test]
fn actual_old_shared_widths_match_for_every_original_image() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/recognition-batch.json")).unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let sizes: Vec<_> = c["sources"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let png = std::fs::read(root.join(p.as_str().unwrap())).unwrap();
                [
                    u32::from_be_bytes(png[16..20].try_into().unwrap()) as usize,
                    u32::from_be_bytes(png[20..24].try_into().unwrap()) as usize,
                ]
            })
            .collect();
        let result = plan(
            &sizes,
            c["batch_size"].as_u64().unwrap() as usize,
            16_000_000,
        );
        for batch in result.unwrap() {
            for i in batch.indices {
                assert_eq!(
                    batch.width as u64,
                    c["tensors"][i]["shape"][2].as_u64().unwrap(),
                    "{}",
                    c["name"]
                );
            }
        }
    }
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
