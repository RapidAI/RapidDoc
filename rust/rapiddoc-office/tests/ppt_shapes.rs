use rapiddoc_office::ppt::{
    records::{self, Budget},
    shapes,
    styles::MasterLevel,
    text::Content,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, io::Read};
fn fixture() -> Value {
    let mut b = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/ppt_shapes.json.gz")[..],
    )
    .read_to_end(&mut b)
    .unwrap();
    serde_json::from_slice(&b).unwrap()
}
fn bytes(s: &Value) -> Vec<u8> {
    s.as_str()
        .unwrap()
        .as_bytes()
        .chunks_exact(2)
        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn original_ppt_groups_geometry_text_binding_and_record_charges() {
    for c in fixture()["cases"].as_array().unwrap() {
        let data = bytes(&c["data"]);
        let mut budget = Budget::default();
        let record = records::at(&data, 0, None, false, &mut budget)
            .unwrap()
            .unwrap();
        let collection =
            shapes::collect(record, &mut budget).unwrap_or_else(|e| panic!("{} {e}", c["name"]));
        let actual=collection.shapes.iter().map(|s|json!({"key":s.key,"offset":s.record.offset,"space":s.space,"bbox":s.bbox,"order":s.order,"group_key":s.group_key})).collect::<Vec<_>>();
        assert_eq!(json!(actual), c["shapes"], "{} shapes", c["name"]);
        let groups=collection.groups.iter().map(|g|json!({"key":g.key,"offset":g.record.offset,"members":g.members.iter().map(|i|&collection.shapes[*i].key).collect::<Vec<_>>(),"order":g.order,"authoritative":g.authoritative})).collect::<Vec<_>>();
        assert_eq!(json!(groups), c["groups"], "{} groups", c["name"]);
        let masters: BTreeMap<u16, Vec<MasterLevel>> =
            serde_json::from_value(c["masters"].clone()).unwrap();
        let external: Vec<Content> = serde_json::from_value(c["external"].clone()).unwrap();
        let links = serde_json::from_value(c["links"].clone()).unwrap();
        let text = collection
            .shapes
            .iter()
            .map(|s| shapes::shape_text(s, &external, &masters, &links, &mut budget))
            .collect::<rapiddoc_office::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(json!(text), c["texts"], "{} text", c["name"]);
        let tables = collection
            .groups
            .iter()
            .map(|g| {
                rapiddoc_office::ppt::tables::from_group(
                    g,
                    &collection,
                    &external,
                    &masters,
                    &links,
                    &mut budget,
                )
            })
            .collect::<rapiddoc_office::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(json!(tables), c["tables"], "{} tables", c["name"]);
        assert_eq!(
            json!(budget.count),
            c["record_visits"],
            "{} budget",
            c["name"]
        );
    }
}
#[test]
fn original_ppt_fopt_and_numbering() {
    let f = fixture();
    for (i, c) in f["properties"].as_array().unwrap().iter().enumerate() {
        let b = bytes(&c["payload"]);
        let r = records::Record {
            offset: 0,
            version: 3,
            instance: c["instance"].as_u64().unwrap() as u16,
            kind: 0xf00b,
            payload: &b,
        };
        assert_eq!(json!(shapes::properties(r)), c["simple"], "simple {i}");
        let complex = shapes::complex(r)
            .into_iter()
            .map(|(k, v)| (k, v.iter().map(|b| format!("{b:02x}")).collect::<String>()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(json!(complex), c["complex"], "complex {i}");
    }
    for (i, c) in f["numbering"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            json!(shapes::numbering(&bytes(&c["data"]))),
            c["expected"],
            "numbering {i}"
        );
    }
}
