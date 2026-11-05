use rapiddoc::{ParseOptions, RapidDoc, RapidDocConfig, RuntimeResources};
use serde_json::json;

#[test]
fn one_engine_parses_office_on_independent_threads_without_runtime_resources() {
    let engine = RapidDoc::new(
        RapidDocConfig::default(),
        RuntimeResources {
            bundle_dirs: vec![],
            allow_download: false,
            ..Default::default()
        },
    );
    let inputs: [&[u8]; 6] = [
        include_bytes!("../../../tests/migration/inputs/synthetic.doc"),
        include_bytes!("../../../tests/migration/inputs/synthetic.docx"),
        include_bytes!("../../../tests/migration/inputs/synthetic.xls"),
        include_bytes!("../../../tests/migration/inputs/synthetic.xlsx"),
        include_bytes!("../../../tests/migration/inputs/synthetic.ppt"),
        include_bytes!("../../../tests/migration/inputs/synthetic.pptx"),
    ];
    let expected: Vec<_> = inputs
        .iter()
        .map(|data| {
            serde_json::to_value(
                engine
                    .parse_bytes(data, &ParseOptions::default(), &mut |_| Ok(()))
                    .unwrap(),
            )
            .unwrap()
        })
        .collect();
    for count in [2, 4] {
        let barrier = std::sync::Barrier::new(count);
        std::thread::scope(|scope| {
            for _ in 0..count {
                let (engine, inputs, expected, barrier) = (&engine, &inputs, &expected, &barrier);
                scope.spawn(move || {
                    barrier.wait();
                    for (data, expected) in inputs.iter().zip(expected) {
                        let doc = engine
                            .parse_bytes(data, &ParseOptions::default(), &mut |_| Ok(()))
                            .unwrap();
                        assert_eq!(&serde_json::to_value(doc).unwrap(), expected);
                    }
                });
            }
        });
    }
    assert!(engine.loaded_model_roles().is_empty());
}

#[test]
fn explicit_default_table_word_boxes_do_not_change_model_plan_or_request() {
    let mut config = RapidDocConfig::default();
    let default_plan = config.model_plan().unwrap();
    let default_request = config
        .request(&ParseOptions::default(), &default_plan)
        .unwrap();
    for value in [json!(false), serde_json::Value::Null] {
        config.table_config["use_word_box"] = value;
        let plan = config.model_plan().unwrap();
        assert_eq!(
            serde_json::to_value(&plan).unwrap(),
            serde_json::to_value(&default_plan).unwrap()
        );
        assert_eq!(
            format!(
                "{:?}",
                config.request(&ParseOptions::default(), &plan).unwrap()
            ),
            format!("{default_request:?}")
        );
    }
    config.table_config["use_word_box"] = json!("false");
    assert_eq!(
        config.model_plan().unwrap_err().code,
        rapiddoc_core::ErrorCode::InvalidInput
    );
    config.table_config["use_word_box"] = json!(true);
    assert_eq!(
        config.model_plan().unwrap_err().code,
        rapiddoc_core::ErrorCode::Unsupported
    );
}

#[test]
fn office_formats_use_the_same_public_entry_without_pdf_or_model_runtimes() {
    let inputs: [(&str, &[u8]); 6] = [
        (
            "doc",
            include_bytes!("../../../tests/migration/inputs/synthetic.doc"),
        ),
        (
            "docx",
            include_bytes!("../../../tests/migration/inputs/synthetic.docx"),
        ),
        (
            "xls",
            include_bytes!("../../../tests/migration/inputs/synthetic.xls"),
        ),
        (
            "xlsx",
            include_bytes!("../../../tests/migration/inputs/synthetic.xlsx"),
        ),
        (
            "ppt",
            include_bytes!("../../../tests/migration/inputs/synthetic.ppt"),
        ),
        (
            "pptx",
            include_bytes!("../../../tests/migration/inputs/synthetic.pptx"),
        ),
    ];
    let resources = RuntimeResources {
        bundle_dirs: vec![],
        allow_download: false,
        ..Default::default()
    };
    let mut engine = RapidDoc::new(RapidDocConfig::default(), resources);
    for (name, data) in inputs {
        let doc = engine
            .parse_bytes(data, &ParseOptions::default(), &mut |_| Ok(()))
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert!(doc.source_page_count > 0, "{name}");
        assert!(!doc.markdown.as_str().unwrap().is_empty(), "{name}");
        assert_eq!(doc.middle_json.page_count(), doc.source_page_count);
        assert!(engine.loaded_model_roles().is_empty());
    }
}

#[test]
fn table_warmup_does_not_load_unrelated_models() {
    let mut config = RapidDocConfig {
        formula_enable: false,
        ..Default::default()
    }
    .table_model("table_transformer");
    config.ocr_config["seal_enable"] = json!(false);
    let plan = config.model_plan().unwrap();
    let request = config.request(&ParseOptions::default(), &plan).unwrap();
    let roles = plan.warmup_roles(&request);
    assert!(roles.contains(&"table_model"));
    for role in [
        "classifier",
        "secondary_classifier",
        "wired_model",
        "formula_model",
        "seal_detector",
    ] {
        assert!(!roles.contains(&role), "{role}");
    }
    config.table_config["fallback_model_type"] = json!("unet_slanet_plus");
    let plan = config.model_plan().unwrap();
    let request = config.request(&ParseOptions::default(), &plan).unwrap();
    let roles = plan.warmup_roles(&request);
    for role in [
        "classifier",
        "secondary_classifier",
        "wired_model",
        "fallback_table_model",
    ] {
        assert!(roles.contains(&role), "{role}");
    }
}

#[test]
fn public_defaults_and_selection_aliases_match_python() {
    let mut config = RapidDocConfig::default()
        .layout_model("pp_doclayoutv2")
        .formula_model("pp_formulanet_plus_l")
        .table_model("unet");
    config.table_config["model_dir_or_path"] = json!("custom.onnx");
    let plan = config.model_plan().unwrap();
    assert_eq!(plan.models["layout_model"].model_id, "pp_doclayoutv2");
    assert_eq!(
        plan.models["formula_model"].model_id,
        "pp_formulanet_plus_l"
    );
    assert_eq!(
        plan.models["table_model"].path.as_deref(),
        Some(std::path::Path::new("custom.onnx"))
    );
    let request = config.request(&ParseOptions::default(), &plan).unwrap();
    assert_eq!(request.window, 64);
    assert!(request.auto_classify);
    assert!(request.clamp_page_range);
    assert_eq!(request.table_variant, "unet");
    assert_eq!(request.dpi, 200.);
    config.table_config["fallback_text_loss_threshold"] = json!(-0.1);
    assert!(config.model_plan().is_err());
}

#[test]
fn model_validation_and_resolution_are_lazy_for_office() {
    let config = RapidDocConfig::default().layout_model("invalid-model");
    let mut engine = RapidDoc::new(config, RuntimeResources::default());
    assert!(engine
        .parse_bytes(
            include_bytes!("../../../tests/migration/inputs/synthetic.docx"),
            &ParseOptions::default(),
            &mut |_| Ok(())
        )
        .is_ok());
    assert!(engine.warmup(&ParseOptions::default()).is_err());
    assert!(engine.loaded_model_roles().is_empty());
}

#[test]
fn explicit_null_cpu_settings_override_global_defaults() {
    let mut config = RapidDocConfig {
        cpu_spinning: Some(true),
        cpu_dynamic_block_base: Some(8),
        ..Default::default()
    };
    config.layout_config =
        json!({"engine_cfg":{"cpu_spinning":null,"cpu_dynamic_block_base":null}});
    let plan = config.model_plan().unwrap();
    assert_eq!(plan.models["layout_model"].cpu_spinning, None);
    assert_eq!(plan.models["layout_model"].cpu_dynamic_block_base, None);
    assert_eq!(plan.models["detector"].cpu_spinning, Some(true));
    config.layout_config = json!({"model_type":null});
    assert!(config.model_plan().is_err());
}
#[cfg(feature = "download-runtime")]
#[test]
fn downloaded_runtime_loads_real_cpu_sessions() {
    use rapiddoc::sdk::Resource;
    use rapiddoc_inference::registry::{cached, Descriptor};
    use sha2::{Digest, Sha256};
    let resources = rapiddoc::RuntimeResources::default();
    let runtime = resources.locate(Resource::Ort).unwrap();
    assert!(runtime.is_file());
    // A checked-in small ONNX graph keeps this deployment test independent
    // of a developer's model cache or optional model downloads.
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/table_partition/encoder.onnx");
    let sha = format!("{:x}", Sha256::digest(std::fs::read(&model).unwrap()));
    let descriptor: Descriptor = serde_json::from_value(json!({
        "runtime":runtime,"model":model,"sha256":sha,"threads":1
    }))
    .unwrap();
    let session = cached(&descriptor).unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &session,
        &cached(&descriptor).unwrap()
    ));
    assert_eq!(
        session
            .lock()
            .unwrap()
            .run_table(vec![1, 1], vec![2.])
            .unwrap()["encoded"]
            .values,
        vec![4.]
    );
}
