use rapiddoc::legacy_model::page_containers;
use rapiddoc_core::Limits;
use serde_json::Value;
fn diff(a: &Value, b: &Value, path: &str) -> Option<String> {
    if a == b {
        return None;
    }
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            for k in a.keys().chain(b.keys()) {
                if !a.contains_key(k) || !b.contains_key(k) {
                    return Some(format!("{path}/{k}: missing"));
                }
                if let Some(d) = diff(&a[k], &b[k], &format!("{path}/{k}")) {
                    return Some(d);
                }
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!("{path}: length {} vs {}", a.len(), b.len()));
            }
            for (i, (a, b)) in a.iter().zip(b).enumerate() {
                if let Some(d) = diff(a, b, &format!("{path}/{i}")) {
                    return Some(d);
                }
            }
        }
        _ => return Some(format!("{path}: {a} vs {b}")),
    }
    None
}
#[test]
fn old_actual_form_tree_and_document_chain() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/native-form-runtime-canonical.json")).unwrap(),
    )
    .unwrap();
    let bytes = std::fs::read(root.join(f["pdf"].as_str().unwrap())).unwrap();
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let mut reconstructed = Vec::new();
    for c in f["geometry"].as_array().unwrap() {
        let actual = serde_json::to_value(
            rapiddoc_pdf::form_structure::read(
                &library,
                &bytes,
                c["page_index"].as_u64().unwrap() as usize,
                None,
                &Limits::default(),
            )
            .unwrap(),
        )
        .unwrap();
        if let Some(d) = diff(&actual, &c["expected"], "geometry") {
            panic!("{}: {d}", c["label"]);
        }
        let page = c["page_index"].as_u64().unwrap() as usize;
        let g = serde_json::to_value(
            rapiddoc_pdf::character_primitives::read_geometry(
                &library,
                &bytes,
                page,
                None,
                true,
                &Limits::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let v = serde_json::to_value(
            rapiddoc_pdf::legacy_vectors::read(&library, &bytes, page, None, &Limits::default())
                .unwrap(),
        )
        .unwrap();
        let lines=rapiddoc::legacy_model::native_lines::build(&serde_json::json!({"chars":g["chars"],"page_size":g["page_size"],"page_rotation":g["rotation"]})).unwrap();
        let expected = &f["request"]["sources"][page];
        let mut source = expected.clone();
        let keys = expected["lines"]
            .as_array()
            .unwrap()
            .first()
            .map(|l| l.as_object().unwrap().keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        source["lines"] = serde_json::json!(lines
            .as_array()
            .unwrap()
            .iter()
            .map(|l| Value::Object(
                keys.iter()
                    .map(|k| (k.clone(), l.get(k).cloned().unwrap_or(Value::Null)))
                    .collect()
            ))
            .collect::<Vec<_>>());
        source["chars"] = g["chars"].clone();
        source["drawing_lines"] = v["drawing_lines"].clone();
        source["path_infos"] = v["path_infos"].clone();
        source["form_bboxes"] = actual["form_bboxes"].clone();
        source["form_infos"] = actual["form_infos"].clone();
        if let Some(d) = diff(&source, expected, "live source") {
            panic!("{}: {d}", c["label"]);
        }
        reconstructed.push(source);
    }
    let actual =
        page_containers::prepare_form_document(&serde_json::json!({"sources":reconstructed}))
            .unwrap();
    if let Some(d) = diff(&actual, &f["expected"], "document") {
        panic!("{d}");
    }
    let edge: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/native-form-runtime-edges.json")).unwrap(),
    )
    .unwrap();
    let edge_bytes = std::fs::read(root.join(edge["pdf"].as_str().unwrap())).unwrap();
    for c in edge["geometry"].as_array().unwrap() {
        let actual = serde_json::to_value(
            rapiddoc_pdf::form_structure::read(
                &library,
                &edge_bytes,
                c["page_index"].as_u64().unwrap() as usize,
                None,
                &Limits::default(),
            )
            .unwrap(),
        )
        .unwrap();
        if let Some(d) = diff(&actual, &c["expected"], "geometry") {
            panic!("{}: {d}", c["label"]);
        }
    }
    let branches: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/native-form-roles-branches-canonical.json"))
            .unwrap(),
    )
    .unwrap();
    for c in branches["cases"].as_array().unwrap() {
        let actual = page_containers::prepare_form_document(&c["request"]).unwrap();
        if let Some(d) = diff(&actual, &c["expected"], "document") {
            panic!("{}: {d}", c["label"]);
        }
    }
    let images: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/native-document-images.json")).unwrap(),
    )
    .unwrap();
    for c in images["cases"].as_array().unwrap() {
        if c.get("error").is_some() {
            assert!(
                page_containers::prepare_image_document(&c["request"]).is_err(),
                "{}",
                c["label"]
            );
            continue;
        }
        let actual = page_containers::prepare_image_document(&c["request"])
            .unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
        if let Some(d) = diff(&actual, &c["expected"], "document") {
            panic!("{}: {d}", c["label"]);
        }
    }
}
