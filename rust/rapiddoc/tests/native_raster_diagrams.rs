use serde_json::{json, Value};
#[test]
fn actual_pdf_raster_diagram_and_plain_frame_evidence() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let limits = rapiddoc_core::Limits::default();
    for name in ["pdf-raster-diagrams", "pdf-raster-diagrams-protected"] {
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join(format!("build/migration/{name}.json"))).unwrap(),
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
            let im = serde_json::to_value(
                rapiddoc_pdf::image_infos::read(&library, &data, page, None, &limits).unwrap(),
            )
            .unwrap();
            let rows=rapiddoc::legacy_model::native_lines::build(&json!({"chars":g["chars"],"page_size":g["page_size"],"page_rotation":g["rotation"]})).unwrap();
            let lines:Vec<_>=rows.as_array().unwrap().iter().map(|l|json!({"text":l["text"],"bbox":l["bbox"],"angle":l["angle"],"source_index":l["source_index"],"visual_row_id":l["visual_row_id"]})).collect();
            let boxes: Vec<_> = im["image_infos"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["bbox"].clone())
                .collect();
            let source = json!({"page_size":g["page_size"],"lines":lines,"path_infos":v["path_infos"],"drawing_lines":v["drawing_lines"],"image_bboxes":boxes,"form_bboxes":[]});
            assert_eq!(source, c["source"], "source {}", c["label"]);
            let component=rapiddoc::legacy_model::drawing_graphics::detect(&json!({"drawing_lines":v["drawing_lines"],"path_infos":[],"page_size":g["page_size"],"median_height":10.})).unwrap();
            let frames: Vec<_> = component["components"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x["bbox"].clone())
                .collect();
            assert_eq!(json!(frames), c["frame_bboxes"]);
            let out = rapiddoc::legacy_model::raster_diagrams::detect(
                &json!({"source":source,"frame_bboxes":frames,"em":10.}),
            )
            .unwrap();
            assert_eq!(out, c["expected"], "output {}", c["label"]);
            if c.get("grid_bboxes").is_some() {
                assert_eq!(
                    rapiddoc::legacy_model::graphic_grid::inspect(
                        &json!({"source":source,"regions":frames,"median_height":10.})
                    )
                    .unwrap()["grid_bboxes"],
                    c["grid_bboxes"]
                );
            }
        }
    }
}
