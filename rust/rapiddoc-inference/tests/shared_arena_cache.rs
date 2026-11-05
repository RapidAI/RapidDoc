use rapiddoc_inference::registry::{cached, Descriptor};
use serde_json::{json, Value};
use std::{path::Path, sync::Arc};

#[test]
fn shared_arena_cache_policy_and_independent_model_outputs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixtures: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/shared-arena-fixture-20261029/models.json")).unwrap(),
    )
    .unwrap();
    let descriptor = |index: usize, shared: Option<bool>| -> Descriptor {
        serde_json::from_value(json!({"runtime":root.join(".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"),
            "model":root.join(fixtures[index]["path"].as_str().unwrap()),"sha256":fixtures[index]["sha256"],
            "threads":1,"cpu_shared_arena":shared})).unwrap()
    };
    let separate = cached(&descriptor(0, None)).unwrap();
    let a = cached(&descriptor(0, Some(true))).unwrap();
    let b = cached(&descriptor(1, Some(true))).unwrap();
    assert!(!Arc::ptr_eq(&a, &separate));
    assert!(!Arc::ptr_eq(&a, &b));
    assert!(Arc::ptr_eq(
        &a,
        &cached(&descriptor(0, Some(true))).unwrap()
    ));
    let check = |model: &rapiddoc_inference::registry::Shared, multiplier: f32| {
        let out = model
            .lock()
            .unwrap()
            .run(vec![1, 4], vec![1., 2., 3., 4.])
            .unwrap();
        assert_eq!(
            out["y"].values,
            vec![
                multiplier,
                2. * multiplier,
                3. * multiplier,
                4. * multiplier
            ]
        );
    };
    check(&a, 2.);
    check(&b, 3.);
    check(&separate, 2.);
    drop(a);
    check(&b, 3.);
    drop(b);
    let reopened = cached(&descriptor(0, Some(true))).unwrap();
    check(&reopened, 2.);
}
