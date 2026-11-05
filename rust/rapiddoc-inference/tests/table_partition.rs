//! Legacy disk-decoder descriptors are rejected; model cache uses original input.
use rapiddoc_inference::registry::{cached, Descriptor};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
#[test]
fn original_model_cache_identity_and_obsolete_disk_descriptor_rejection() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let runtime = if cfg!(windows) {
        root.join(".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll")
    } else {
        PathBuf::from(std::env::var("RAPIDDOC_ONNXRUNTIME_LIBRARY").expect("ORT runtime"))
    };
    let model = root.join("tests/fixtures/table_partition/encoder.onnx");
    let sha = format!("{:x}", Sha256::digest(std::fs::read(&model).unwrap()));
    let raw = serde_json::json!({"runtime":runtime,"model":model,"sha256":sha,"threads":1});
    let d: Descriptor = serde_json::from_value(raw.clone()).unwrap();
    let first = cached(&d).unwrap();
    assert!(Arc::ptr_eq(&first, &cached(&d).unwrap()));
    assert_eq!(
        first
            .lock()
            .unwrap()
            .run_table(vec![1, 1], vec![2.])
            .unwrap()["encoded"]
            .values,
        vec![4.]
    );
    let mut old = raw.clone();
    old["table_decoder"] = serde_json::json!({"model":"old-file","sha256":"old"});
    assert!(serde_json::from_value::<Descriptor>(old).is_err());
    let mut optimized: Descriptor = serde_json::from_value(raw).unwrap();
    optimized.optimization = Some(rapiddoc_inference::onnx::Optimization::Slanet);
    assert!(cached(&optimized).is_err());
}
