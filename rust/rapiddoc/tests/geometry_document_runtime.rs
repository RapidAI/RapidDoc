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
fn live_pdfium_document_geometry_plan() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/native-geometry-real-document.json")).unwrap(),
    )
    .unwrap();
    let case = &f["case"];
    let form: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/native-form-runtime-canonical.json")).unwrap(),
    )
    .unwrap();
    let bytes = std::fs::read(root.join(form["pdf"].as_str().unwrap())).unwrap();
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let mut pages = Vec::new();
    for (i, expected) in case["request"]["pages"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
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
        let lines = rapiddoc::legacy_model::native_lines::build(
            &json!({"chars":g["chars"],"page_size":g["page_size"],"page_rotation":g["rotation"]}),
        )
        .unwrap();
        if let Some(d) = diff(&lines, &expected["source"]["lines"], "") {
            panic!("live native lines page {i}: {d}");
        }
        let mut geometry =
            json!({"chars":g["chars"],"tight_bboxes":[],"origins":[],"loose_bboxes":[]});
        for (key, value) in [
            ("tight_bboxes", "bbox"),
            ("origins", "origin"),
            ("loose_bboxes", "bbox"),
        ] {
            let mut rows = g[key]
                .as_object()
                .unwrap()
                .iter()
                .map(|(index, b)| {
                    let mut row = json!({"char_idx":index.parse::<usize>().unwrap()});
                    row[value] = b.clone();
                    row
                })
                .collect::<Vec<_>>();
            rows.sort_by_key(|r| r["char_idx"].as_u64().unwrap());
            geometry[key] = json!(rows);
        }
        for key in ["tight_bboxes", "origins", "loose_bboxes"] {
            assert_eq!(
                geometry[key], expected["geometry"][key],
                "live {key} page {i}"
            );
        }
        pages
            .push(json!({"source":{"lines":lines,"page_size":g["page_size"]},"geometry":geometry}));
    }
    let actual =
        rapiddoc::legacy_model::native_geometry::build_document_plan(&json!({"pages":pages}))
            .unwrap();
    assert_eq!(actual, case["expected"]);
}
