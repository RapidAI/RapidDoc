use serde_json::{json, Value};

#[test]
fn actual_old_formula_stage() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/migration/model-formula-tasks.json");
    let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(f["cases"].as_array().unwrap().len(), 100);
    for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            rapiddoc::legacy_model::formulas::plan(&c["request"]).unwrap(),
            json!({"jobs":c["jobs"]}),
            "plan {i}"
        );
        assert_eq!(
            rapiddoc::legacy_model::formulas::apply(&c["request"]).unwrap(),
            c["expected"],
            "apply {i}"
        );
    }
}

#[test]
fn invalid_formula_stage_is_atomic() {
    let p = json!({"model":{"layout_dets":[{"category_id":13,"latex":"prior"}]},"formulas":[{"layout_index":0,"source":"x"}]});
    let request = json!({"plans":[p],"results":[123]});
    assert!(rapiddoc::legacy_model::formulas::apply(&request).is_err());
    assert_eq!(
        request["plans"][0]["model"]["layout_dets"][0]["latex"],
        "prior"
    );
    for results in [json!([]), json!(["x", "y"])] {
        let mut r = request.clone();
        r["results"] = results;
        assert!(rapiddoc::legacy_model::formulas::apply(&r).is_err());
    }
    let mut r = request.clone();
    r["plans"][0]["formulas"][0]["layout_index"] = json!(1);
    assert!(rapiddoc::legacy_model::formulas::plan(&r).is_err());
    r["plans"][0]["formulas"] = json!([]);
    assert!(rapiddoc::legacy_model::formulas::plan(&r).is_err());
    assert!(rapiddoc::legacy_model::formulas::validate_png(b"invalid png").is_err());
}
