use serde_json::{json, Value};
fn fixture() -> Value {
    serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-rule.json"),
        )
        .unwrap(),
    )
    .unwrap()
}
#[test]
fn old_native_table_pre_model_guards() {
    for c in fixture()["policy"].as_array().unwrap() {
        let p = rapiddoc::legacy_model::table_native_rule::plan(&c["request"]).unwrap();
        assert_eq!(p["attempt"], c["expected"]["attempt"], "{c}");
        if p["attempt"] == true {
            assert_eq!(p["table_bbox"], c["expected"]["bbox"], "{c}");
        }
    }
}
#[test]
fn old_missing_vs_explicit_engine_values() {
    let f: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../build/migration/native-table-engine-values.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let mut request = fixture()["policy"][0]["request"].clone();
        request["config"] = c["config"].clone();
        assert_eq!(
            rapiddoc::legacy_model::table_native_rule::plan(&request).unwrap()["attempt"],
            c["expected"],
            "{c}"
        );
    }
}
#[test]
fn old_native_result_acceptance_and_metadata() {
    for c in fixture()["policy"].as_array().unwrap() {
        if c["expected"]["attempt"] != true {
            continue;
        }
        let result = rapiddoc::legacy_model::table_native_rule::accept(
            &json!({"table":c["request"]["table"],"result":c["result"],"threshold":0.9}),
        )
        .unwrap();
        assert_eq!(result["accepted"], c["expected"]["accepted"]);
        assert_eq!(result["table"], c["expected"]["table"]);
    }
}
#[test]
fn old_candidate_threshold_topology_and_undercount_arbitration() {
    for c in fixture()["selection"].as_array().unwrap() {
        assert_eq!(
            rapiddoc::legacy_model::table_native_rule::select(&c["request"]).unwrap(),
            c["expected"],
            "{c}"
        );
    }
}
