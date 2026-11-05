use rapiddoc::pipeline::Output;
use serde_json::json;
#[test]
fn export_matches_existing_renderers_without_losing_page_metadata() {
    let page = json!({"page_idx":0,"page_size":[100,100],"para_blocks":[
            {"type":"text","lines":[{"spans":[{"type":"text","content":"A & B"}]}]},
            {"type":"table","blocks":[{"type":"table_body","lines":[{"spans":[
                {"type":"table","html":"<table><tr><td><img src=\"images/a.png\"></td></tr></table>"}
            ]}]}]}
        ],"discarded_blocks":[]});
    for mode in ["mm_markdown", "nlp_markdown", "content_list"] {
        let mut expected_pages = json!([page]);
        expected_pages[0]["para_blocks"][1]["blocks"][0]["lines"][0]["spans"][0]["html"] =
            json!("<table><tr><td><img src=\"figures/a.png\"></td></tr></table>");
        let mut output = Output {
            metadata: json!({"pages":[page],"source_page_count":1}),
            assets: vec![],
        };
        output.render_document(mode, "figures", None).unwrap();
        for (key, render_mode) in [("markdown", mode), ("content_list_json", "content_list")] {
            let expected = rapiddoc::legacy_page::render(&json!({"pages":expected_pages,"make_mode":render_mode,"image_path_prefix":"figures"})).unwrap();
            assert_eq!(output.metadata[key], expected);
        }
        assert_eq!(output.metadata["pages"], expected_pages);
        assert_eq!(output.metadata["source_page_count"], 1);
    }
}

#[test]
fn export_error_restores_pages() {
    let pages = json!([{"page_idx":0}]);
    let mut output = Output {
        metadata: json!({"pages":pages}),
        assets: vec![],
    };
    assert!(output.render_document("invalid", "images", None).is_err());
    assert_eq!(output.metadata["pages"], pages);
}

#[test]
fn production_export_has_no_diagnostic_page_cap_and_honors_explicit_limits() {
    let pages = json!(vec![
        json!({"page_idx":0,"page_size":[100,100],"para_blocks":[]});
        1001
    ]);
    let mut output = Output {
        metadata: json!({"pages":pages}),
        assets: vec![],
    };
    output
        .render_document("mm_markdown", "images", None)
        .unwrap();
    assert_eq!(output.metadata["markdown"], "");
    rapiddoc::legacy_page::with_host_resources(
        rapiddoc::legacy_page::HostResources {
            max_json_nodes: Some(1),
            ..Default::default()
        },
        || {
            assert!(output
                .render_document("mm_markdown", "images", None)
                .is_err())
        },
    );
    assert_eq!(output.metadata["pages"].as_array().unwrap().len(), 1001);
    output
        .render_document("mm_markdown", "images", None)
        .unwrap();
}
