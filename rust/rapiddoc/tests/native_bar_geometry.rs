use serde_json::{json, Value};
#[test]
fn actual_pdf_bar_preparation_and_regenerated_axis_queries() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-bar-geometry.json")).unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let data = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
        let v = serde_json::to_value(
            rapiddoc_pdf::legacy_vectors::read(
                &library,
                &data,
                c["page_index"].as_u64().unwrap() as usize,
                None,
                &limits,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(v["path_infos"], c["request"]["path_infos"]);
        assert_eq!(v["drawing_lines"], c["request"]["drawing_lines"]);
        let mut q = json!({"path_infos":v["path_infos"],"drawing_lines":v["drawing_lines"],"em":10.,"table_bboxes":[],"axis_queries":[]});
        let prepared = rapiddoc::legacy_model::bar_geometry::prepare(&q).unwrap();
        let mut queries = Vec::new();
        for horizontal in [true, false] {
            let key = if horizontal { "horizontal" } else { "vertical" };
            let bars: Vec<_> = prepared["admitted_bars"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|b| b[key].as_bool().unwrap())
                .collect();
            if bars.is_empty() {
                continue;
            }
            let mut bounds = [
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ];
            for b in &bars {
                for i in 0..4 {
                    let n = b["bbox"][i].as_f64().unwrap();
                    bounds[i] = if i < 2 {
                        bounds[i].min(n)
                    } else {
                        bounds[i].max(n)
                    };
                }
            }
            for edge in if horizontal { [0, 2] } else { [3, 1] } {
                queries.push(json!({"bounds":bounds,"horizontal":horizontal,"baseline":bars[0]["bbox"][edge]}));
            }
        }
        q["axis_queries"] = json!(queries);
        assert_eq!(q, c["request"], "live queries {}", c["label"]);
        assert_eq!(
            rapiddoc::legacy_model::bar_geometry::prepare(&q).unwrap(),
            c["expected"],
            "output {}",
            c["label"]
        );
    }
}
