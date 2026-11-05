use rapiddoc_inference::registry::{cached, Descriptor};
use serde_json::{json, Value};
use std::{path::Path, sync::Arc};
#[test]
fn cpu_policy_is_part_of_cache_identity_and_preserves_outputs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = std::fs::read_dir(root.join("build/migration/onnx-fixtures"))
        .unwrap()
        .map(|x| x.unwrap().path())
        .find(|p| p.extension().is_some_and(|s| s == "json"))
        .unwrap();
    let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let base:Descriptor=serde_json::from_value(json!({"runtime":root.join(".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"),"model":root.join(f["model"].as_str().unwrap()),"sha256":f["sha256"],"threads":1})).unwrap();
    let original = cached(&base).unwrap();
    assert!(Arc::ptr_eq(&original, &cached(&base).unwrap()));
    let mut sleeping = base.clone();
    sleeping.cpu_spinning = Some(false);
    let a = cached(&sleeping).unwrap();
    assert!(!Arc::ptr_eq(&original, &a));
    assert!(Arc::ptr_eq(&a, &cached(&sleeping).unwrap()));
    let mut dynamic = base.clone();
    dynamic.cpu_dynamic_block_base = Some(4);
    let b = cached(&dynamic).unwrap();
    assert!(!Arc::ptr_eq(&original, &b));
    assert!(!Arc::ptr_eq(&a, &b));
    let mut combined = sleeping.clone();
    combined.cpu_dynamic_block_base = Some(4);
    let c = cached(&combined).unwrap();
    assert!(!Arc::ptr_eq(&a, &c));
    assert!(!Arc::ptr_eq(&b, &c));
    let mut stopping = base.clone();
    stopping.cpu_force_spinning_stop = Some(true);
    let d = cached(&stopping).unwrap();
    assert!(!Arc::ptr_eq(&original, &d));
    assert!(!Arc::ptr_eq(&a, &d));
    assert!(Arc::ptr_eq(&d, &cached(&stopping).unwrap()));
    for model in [&original, &a, &b, &c, &d] {
        let out = model
            .lock()
            .unwrap()
            .run(
                serde_json::from_value(f["shape"].clone()).unwrap(),
                serde_json::from_value(f["values"].clone()).unwrap(),
            )
            .unwrap();
        for (name, v) in out {
            let expected = &f["outputs"][&name];
            assert_eq!(serde_json::to_value(v.shape).unwrap(), expected["shape"]);
            let numbers = expected["values"].as_array().unwrap();
            assert_eq!(v.values.len(), numbers.len());
            for (actual, e) in v.values.iter().zip(numbers) {
                assert!((*actual - e.as_f64().unwrap() as f32).abs() < 1e-5);
            }
        }
    }
    let mut invalid = base;
    invalid.cpu_dynamic_block_base = Some(i32::MAX as usize + 1);
    assert!(cached(&invalid).is_err());
}
