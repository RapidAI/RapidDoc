use serde_json::{json, Value};
fn queries(candidates: &[Value], mode: &str) -> Value {
    let mut new = candidates.to_vec();
    if mode != "self" {
        new.clear();
        for c in candidates {
            let b = c["core_bbox"]
                .as_array()
                .unwrap_or_else(|| c["bbox"].as_array().unwrap());
            let mid = (b[1].as_f64().unwrap() + b[3].as_f64().unwrap()) / 2.;
            let halves = [
                json!([b[0], b[1], b[2], mid]),
                json!([b[0], mid, b[2], b[3]]),
            ];
            for bounds in halves.iter().take(if mode == "partial" { 1 } else { 2 }) {
                let mut c = c.clone();
                for key in ["bbox", "local_bbox", "core_bbox"] {
                    c[key] = bounds.clone();
                }
                new.push(c);
            }
        }
    }
    json!({"base":candidates,"recovered":[],"replacements":new})
}
#[test]
fn actual_pdf_materialized_candidates_to_merge_queries() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-table-candidate-merge.json")).unwrap(),
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
        let rows = rapiddoc::legacy_model::native_lines::build(
            &json!({"chars":g["chars"],"page_size":g["page_size"],"page_rotation":g["rotation"]}),
        )
        .unwrap();
        let lines:Vec<_>=rows.as_array().unwrap().iter().map(|l|json!({"text":l["text"],"bbox":l["bbox"],"source_index":l["source_index"],"chars":l["chars"]})).collect();
        let source = json!({"page_size":g["page_size"],"chars":g["chars"],"path_infos":v["path_infos"],"drawing_lines":v["drawing_lines"],"lines":lines});
        assert_eq!(source, c["source"]);
        let detected = if c["kind"] == "filled" {
            rapiddoc::legacy_model::filled_grid::detect(&json!({"path_infos":v["path_infos"],"page_size":v["page_size"],"excluded_bboxes":[]})).unwrap()
        } else {
            rapiddoc::legacy_model::tiled_tables::detect(
                &json!({"source":source,"excluded_bboxes":[]}),
            )
            .unwrap()
        };
        let q = queries(
            detected["candidates"].as_array().unwrap(),
            c["mode"].as_str().unwrap(),
        );
        let mut candidates = Vec::new();
        for key in ["base", "recovered", "replacements"] {
            for c in q[key].as_array().unwrap() {
                let mut c = c.clone();
                for a in c["annotations"].as_array_mut().unwrap() {
                    let pairs: Vec<_> = a["line_bboxes"]
                        .as_object()
                        .unwrap()
                        .iter()
                        .map(|(id, b)| json!([id.parse::<i64>().unwrap(), b]))
                        .collect();
                    a["line_bboxes"] = json!(pairs);
                }
                candidates.push(c);
            }
        }
        let q = json!({"candidates":candidates});
        assert_eq!(q, c["request"], "query {}", c["label"]);
        assert_eq!(
            rapiddoc::legacy_model::table_candidate_merge::merge(&q).unwrap(),
            c["expected"],
            "merge {}",
            c["label"]
        );
    }
}
