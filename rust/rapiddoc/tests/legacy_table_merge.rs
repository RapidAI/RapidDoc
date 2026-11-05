use rapiddoc::legacy_table_merge::merge;
use serde_json::Value;
#[test]
fn complete_old_cross_page_table_json_matches() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/legacy-table-merge.json")).unwrap(),
    )
    .unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        assert_eq!(merge(&c["pages"]).unwrap(), c["expected"], "{}", c["name"]);
    }
}

#[test]
fn empty_table_states_keep_the_old_block_cleanup() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/legacy-table-merge-edges.json")).unwrap(),
    )
    .unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        assert_eq!(merge(&c["pages"]).unwrap(), c["expected"]);
    }
}

#[test]
fn extended_old_spans_headers_identity_entities_and_whitespace_match() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/legacy-table-merge-extended.json")).unwrap(),
    )
    .unwrap();
    let mut failures = Vec::new();
    for c in fixture["cases"].as_array().unwrap() {
        let actual = merge(&c["pages"]);
        if actual.as_ref().ok() != Some(&c["expected"]) {
            failures.push((c["name"].clone(), actual, c["expected"].clone()));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches; first: {:#?}",
        failures.len(),
        failures.first()
    );
}

#[test]
fn unsupported_html_and_resource_limits_do_not_return_partial_success() {
    use serde_json::json;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/legacy-table-merge.json")).unwrap(),
    )
    .unwrap();
    for html in [
        "<table><tr><td>unclosed",
        "<table><tr><td rowspan=\"0\">x</td></tr></table>",
        "<table><tr><td>&bogus;</td></tr></table>",
        "<table><tr><td><table><tr><td>x</td></tr></table></td></tr></table>",
        "<table><tr><td><script>hidden</script></td></tr></table>",
    ] {
        let mut pages = fixture["cases"][0]["pages"].clone();
        pages[1]["para_blocks"][0]["blocks"][0]["lines"][0]["spans"][0]["html"] = json!(html);
        assert!(merge(&pages).is_err(), "{html}");
    }
    let mut pages = fixture["cases"][0]["pages"].clone();
    pages[1]["para_blocks"][0]["blocks"][0]["lines"][0]["spans"][0]["html"] =
        json!("<table><tr><td colspan=\"1000\" rowspan=\"1000\">x</td></tr></table>");
    assert!(merge(&pages).is_err());
    assert!(merge(&json!({})).is_err());
}
