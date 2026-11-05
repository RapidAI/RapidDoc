use rapiddoc_office::{
    package::Package,
    pptx::{context, geometry, reading_order},
};
use serde_json::Value;
use std::io::Read;
#[path = "support/image_parity.rs"]
mod image_parity;
fn reference() -> Value {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(&include_bytes!("../../../tests/fixtures/office_rust_reference_068/pptx_context.json.gz")[..]).read_to_end(&mut raw).unwrap();
    serde_json::from_slice(&raw).unwrap()
}
fn b(v: &Value) -> Option<geometry::Bbox> {
    serde_json::from_value(v.clone()).unwrap()
}
#[test]
fn original_pptx_group_placeholder_relationship_and_picture_context() {
    for c in reference()["cases"].as_array().unwrap() {
        let bytes = base64::decode(c["package_b64"].as_str().unwrap()).unwrap();
        let mut package = Package::open(bytes.into()).unwrap();
        let actual = context::inspect_package(&mut package);
        if c.get("error").is_some() {
            assert!(actual.is_err(), "{} expected {}", c["name"], c["error"]);
            continue;
        }
        let mut actual =
            serde_json::to_value(actual.unwrap_or_else(|e| panic!("{}: {e}", c["name"]))).unwrap();
        let mut expected = c["expected"].clone();
        image_parity::normalize(&mut actual);
        image_parity::normalize(&mut expected);
        if actual != expected {
            std::fs::write(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/pptx-context-diff.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"name":c["name"],"actual":actual,"expected":expected}),
                )
                .unwrap(),
            )
            .unwrap();
            panic!("{} differs; build/pptx-context-diff.json", c["name"]);
        }
    }
}
#[test]
fn original_pptx_xycut_union_and_background_filter() {
    for (i, q) in reference()["queries"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let boxes = q["boxes"]
            .as_array()
            .unwrap()
            .iter()
            .map(b)
            .collect::<Vec<_>>();
        assert_eq!(
            serde_json::json!(reading_order::sort(&boxes)),
            q["order"],
            "query {i} order"
        );
        let later = q["later"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| (b(&v[0]), v[1].as_bool().unwrap()))
            .collect::<Vec<_>>();
        let union = geometry::union_area(&later.iter().filter_map(|(b, _)| *b).collect::<Vec<_>>());
        assert!(
            (union - q["union"].as_f64().unwrap()).abs() < 1e-7,
            "query {i} union"
        );
        assert_eq!(
            geometry::small_picture(b(&q["picture"]), 1200., 900.),
            q["small"].as_bool().unwrap(),
            "query {i} small"
        );
        assert_eq!(
            geometry::background_picture(b(&q["picture"]), later),
            q["background"].as_bool().unwrap(),
            "query {i} background"
        );
    }
}
#[test]
fn original_pptx_font_rounding_and_style_toggles() {
    for (i, q) in reference()["fonts"].as_array().unwrap().iter().enumerate() {
        let doc = rapiddoc_office::xml::parse(q["xml"].as_str().unwrap()).unwrap();
        let style = rapiddoc_office::pptx::styles::run_style(doc.root_element(), &[]);
        assert_eq!(serde_json::json!(style.styles), q["styles"], "font {i}");
        assert_eq!(
            serde_json::json!(style.font_size_pt),
            q["font_size_pt"],
            "font {i}"
        );
    }
}
#[test]
fn original_pptx_title_promotions_and_first_visible_slide() {
    use rapiddoc_office::{
        docx::body::{Block, Content, Kind},
        pptx::titles::{self, Entry, Role},
    };
    for (i, q) in reference()["titles"].as_array().unwrap().iter().enumerate() {
        let mut entries = q["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| Entry {
                block: Block {
                    kind: Kind::Text,
                    content: Content::Text(b["content"].as_str().unwrap().into()),
                    image_base64: None,
                    level: b["level"].as_i64(),
                    is_numbered_style: None,
                    anchor: None,
                    list_label: None,
                    ilevel: None,
                    attribute: None,
                    start: None,
                },
                font_size_pt: b["_effective_font_size_pt"].as_f64(),
                all_bold: b["_effective_all_bold"].as_bool().unwrap(),
                candidate: b["_pptx_title_candidate"] == true,
                role: match b["_pptx_title_role"].as_str() {
                    Some("center_title") => Some(Role::Center),
                    Some("title") => Some(Role::Title),
                    Some("subtitle") => Some(Role::Subtitle),
                    _ => None,
                },
            })
            .collect::<Vec<_>>();
        titles::finalize(&mut entries, q["first"].as_bool().unwrap());
        assert_eq!(
            serde_json::to_value(entries.into_iter().map(|e| e.block).collect::<Vec<_>>()).unwrap(),
            q["expected"],
            "title {i}"
        );
    }
}
