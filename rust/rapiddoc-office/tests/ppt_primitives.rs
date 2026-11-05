use rapiddoc_office::{
    ppt::{
        document,
        records::{self, Budget},
        styles, text,
    },
    Result,
};
use serde_json::{json, Value};
use std::io::Read;
fn frozen() -> Value {
    let mut b = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/ppt_primitives.json.gz")
            [..],
    )
    .read_to_end(&mut b)
    .unwrap();
    serde_json::from_slice(&b).unwrap()
}
fn bytes(v: &Value) -> Vec<u8> {
    v.as_str()
        .unwrap()
        .as_bytes()
        .chunks_exact(2)
        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect()
}
fn summary(r: records::Record<'_>) -> Value {
    json!({"offset":r.offset,"version":r.version,"instance":r.instance,"kind":r.kind,"payload":r.payload.iter().map(|b|format!("{b:02x}")).collect::<String>()})
}
#[test]
fn original_ppt_records_depth_opaque_and_utf16() {
    let f = frozen();
    for c in f["records"].as_array().unwrap() {
        let data = bytes(&c["data"]);
        let mut budget = Budget::default();
        let actual = records::iter(&data, 0, None, c["strict"].as_bool().unwrap(), &mut budget)
            .map(|r| r.map(summary))
            .collect::<Result<Vec<_>>>();
        if let Some(e) = c.get("error") {
            assert_eq!(
                actual.unwrap_err().kind.code(),
                e.as_str().unwrap(),
                "{}",
                c["name"]
            );
        } else {
            assert_eq!(json!(actual.unwrap()), c["records"], "{}", c["name"]);
        }
        assert_eq!(budget.count, c["count"].as_u64().unwrap() as usize);
        let mut budget = Budget::default();
        let root = records::Record {
            offset: 0,
            version: 15,
            instance: 0,
            kind: 1000,
            payload: &data,
        };
        let actual = records::descendants(root, &mut budget)
            .map(|r| r.map(summary))
            .collect::<Result<Vec<_>>>();
        if let Some(e) = c.get("descendant_error") {
            assert_eq!(
                actual.unwrap_err().kind.code(),
                e.as_str().unwrap(),
                "{}",
                c["name"]
            );
        } else {
            assert_eq!(json!(actual.unwrap()), c["descendants"], "{}", c["name"]);
            assert_eq!(
                budget.count,
                c["descendant_count"].as_u64().unwrap() as usize
            );
        }
    }
    for c in f["utf16"].as_array().unwrap() {
        assert_eq!(
            records::utf16(&bytes(&c["data"])),
            c["text"].as_str().unwrap()
        );
    }
}
#[test]
fn original_ppt_tristate_style_and_master_inputs() {
    for (i, c) in frozen()["styles"].as_array().unwrap().iter().enumerate() {
        let data = bytes(&c["data"]);
        assert_eq!(
            json!(styles::parse(&data, c["length"].as_u64().unwrap() as usize)),
            c["style"],
            "style {i}"
        );
        assert_eq!(
            json!(styles::master(
                &data,
                c["instance"].as_u64().unwrap() as u16
            )),
            c["master"],
            "master {i}"
        );
    }
}
#[test]
fn original_ppt_utf16_style_links_controls_and_paragraphs() {
    for (i, c) in frozen()["texts"].as_array().unwrap().iter().enumerate() {
        let styles = serde_json::from_value(c["styles"].clone()).unwrap();
        let masters =
            serde_json::from_value::<Vec<styles::MasterLevel>>(c["masters"].clone()).unwrap();
        let links = serde_json::from_value::<Vec<(u32, u32, String)>>(c["links"].clone()).unwrap();
        assert_eq!(
            json!(text::build(
                c["text"].as_str().unwrap(),
                &styles,
                &masters,
                &links,
                text::is_control
            )),
            c["paragraphs"],
            "paragraph {i}"
        );
    }
}
#[test]
fn original_ppt_latest_save_and_recovered_persist_layout() {
    for c in frozen()["documents"].as_array().unwrap() {
        let actual = document::inspect(&bytes(&c["data"]), &bytes(&c["current"]));
        if let Some(e) = c.get("error") {
            assert_eq!(
                actual.unwrap_err().kind.code(),
                e.as_str().unwrap(),
                "{}",
                c["name"]
            );
        } else {
            assert_eq!(json!(actual.unwrap()), c["expected"], "{}", c["name"]);
        }
    }
}
#[test]
fn original_ppt_preflight_encryption_and_complete_stream_validation() {
    for c in frozen()["prepared"].as_array().unwrap() {
        let data = bytes(&c["data"]);
        let current = bytes(&c["current"]);
        let actual = document::prepare(&data, &current);
        if let Some(e) = c.get("error") {
            assert_eq!(
                actual.err().unwrap().kind.code(),
                e.as_str().unwrap(),
                "{}",
                c["name"]
            );
        } else {
            let (layout, budget) = actual.unwrap();
            assert_eq!(
                json!({"document_offset":layout.document.offset,"persist":layout.persist,"recovered":layout.recovered,"count":budget.count}),
                c["expected"],
                "{}",
                c["name"]
            );
        }
    }
}
#[test]
fn original_ppt_master_external_text_and_notes_context() {
    for c in frozen()["contexts"].as_array().unwrap() {
        let data = bytes(&c["data"]);
        let current = bytes(&c["current"]);
        let mut budget = Budget::default();
        let layout = document::locate(&data, &current, &mut budget).unwrap();
        let masters = document::masters(&layout, &data, &mut budget).unwrap();
        let external = document::external_text(layout.document, &mut budget).unwrap();
        let links = text::targets(layout.document, &mut budget).unwrap();
        let empty = std::collections::BTreeMap::new();
        let fallback = masters
            .first()
            .map(|(_, (_, styles))| styles)
            .unwrap_or(&empty);
        let (bound, unbound) =
            rapiddoc_office::ppt::notes::bind(&layout, &data, fallback, &links, &mut budget)
                .unwrap();
        assert_eq!(
            json!(masters
                .iter()
                .map(|(id, (r, styles))| json!({"id":id,"offset":r.offset,"styles":styles}))
                .collect::<Vec<_>>()),
            c["masters"],
            "{} masters",
            c["name"]
        );
        assert_eq!(
            json!(external
                .into_iter()
                .map(|(k, v)| (k, v.into_iter().map(summary).collect::<Vec<_>>()))
                .collect::<std::collections::BTreeMap<_, _>>()),
            c["external"],
            "{} external",
            c["name"]
        );
        assert_eq!(json!(links), c["links"], "{} links", c["name"]);
        assert_eq!(json!(bound), c["notes"], "{} notes", c["name"]);
        assert_eq!(json!(unbound), c["unbound"], "{} unbound", c["name"]);
        assert_eq!(json!(budget.count), c["count"], "{} visits", c["name"]);
    }
}
