use rapiddoc::legacy_model::diagram_components as pipeline;
use serde_json::{json, Value};
fn first_difference(a: &Value, b: &Value, path: &str) -> Option<String> {
    if a == b {
        return None;
    }
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            for k in a.keys().chain(b.keys()) {
                if !a.contains_key(k) || !b.contains_key(k) {
                    return Some(format!("{path}/{k}: missing key"));
                }
                if let Some(d) = first_difference(&a[k], &b[k], &format!("{path}/{k}")) {
                    return Some(d);
                }
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!("{path}: lengths {} vs {}", a.len(), b.len()));
            }
            for (i, (a, b)) in a.iter().zip(b).enumerate() {
                if let Some(d) = first_difference(a, b, &format!("{path}/{i}")) {
                    return Some(d);
                }
            }
        }
        _ => return Some(format!("{path}: {a} vs {b}")),
    }
    None
}
fn check() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    for corpus in ["pdf-diagram-components"] {
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join(format!("build/migration/{corpus}.json"))).unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let data = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
            let page = c["page_index"].as_u64().unwrap() as usize;
            let v = serde_json::to_value(
                rapiddoc_pdf::legacy_vectors::read(&library, &data, page, None, &limits).unwrap(),
            )
            .unwrap();
            let g = serde_json::to_value(
                rapiddoc_pdf::character_primitives::read_geometry(
                    &library, &data, page, None, true, &limits,
                )
                .unwrap(),
            )
            .unwrap();
            let lines = rapiddoc::legacy_model::native_lines::build(
            &json!({"chars":g["chars"],"page_size":g["page_size"],"page_rotation":g["rotation"]}),
        )
        .unwrap();
            let fields = "text bbox source_index angle effective_height em_height style_scale_repaired font_signature font_coverage chars visual_row_id dominant_font_weight paragraph_terminal semantic_type caption_start formula_candidate_only compact_formula_cluster";
            let projected: Vec<Value> = lines
                .as_array()
                .unwrap()
                .iter()
                .map(|l| {
                    Value::Object(
                        fields
                            .split_whitespace()
                            .map(|k| {
                                (
                                    k.to_string(),
                                    l.get(k).cloned().unwrap_or_else(|| {
                                        if k == "compact_formula_cluster" {
                                            json!(false)
                                        } else {
                                            Value::Null
                                        }
                                    }),
                                )
                            })
                            .collect(),
                    )
                })
                .collect();
            let source = json!({"page_size":g["page_size"],"lines":projected,"chars":g["chars"],"drawing_lines":v["drawing_lines"],"path_infos":v["path_infos"],"form_bboxes":[],"image_bboxes":[],"signature_bboxes":[],"page_index":page});
            let mut q = c["request"].clone();
            if let Some(d) = first_difference(&source, &q["source"], "source") {
                panic!("{}: {d}", c["label"]);
            }
            q["source"] = source;
            assert_eq!(
                pipeline::detect(&q).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
}
#[test]
fn live_pdf_to_diagram_components() {
    check();
}
