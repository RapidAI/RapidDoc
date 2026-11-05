use rapiddoc_core::Limits;
use serde_json::{json, Value};
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
fn live_pdfium_marginal_document() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/native-marginal-runtime.json")).unwrap(),
    )
    .unwrap();
    let bytes = std::fs::read(root.join(f["pdf"].as_str().unwrap())).unwrap();
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let mut sources = Vec::new();
    for i in 0..3 {
        let g = serde_json::to_value(
            rapiddoc_pdf::character_primitives::read_geometry(
                &library,
                &bytes,
                i,
                None,
                true,
                &Limits::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let v = serde_json::to_value(
            rapiddoc_pdf::legacy_vectors::read(&library, &bytes, i, None, &Limits::default())
                .unwrap(),
        )
        .unwrap();
        let forms = serde_json::to_value(
            rapiddoc_pdf::form_structure::read(&library, &bytes, i, None, &Limits::default())
                .unwrap(),
        )
        .unwrap();
        let lines = rapiddoc::legacy_model::native_lines::build(
            &json!({"chars":g["chars"],"page_size":g["page_size"],"page_rotation":g["rotation"]}),
        )
        .unwrap();
        let source = json!({"page_size":g["page_size"],"page_index":i,"lines":lines,"chars":g["chars"],"drawing_lines":v["drawing_lines"],"path_infos":v["path_infos"],"form_infos":forms["form_infos"],"form_bboxes":forms["form_bboxes"],"image_bboxes":[],"signature_bboxes":[]});
        if let Some(d) = diff(&source, &f["request"]["sources"][i], "") {
            panic!("page {i}: {d}")
        }
        sources.push(source);
    }
    let actual =
        rapiddoc::legacy_model::native_marginals::prepare(&json!({"sources":sources})).unwrap();
    if let Some(d) = diff(&actual, &f["expected"], "") {
        panic!("marginals: {d}")
    }
}
