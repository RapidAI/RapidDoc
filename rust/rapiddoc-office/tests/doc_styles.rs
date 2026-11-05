use rapiddoc_office::{
    doc::{
        fib,
        styles::{Character, Stylesheet},
    },
    ole::OleReader,
};
#[test]
fn original_resolved_styles() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut ole = OleReader::open(
        std::fs::read(root.join("demo/ms_office_docs/docx_01.doc"))
            .unwrap()
            .into(),
    )
    .unwrap();
    let word = ole.read_stream("WordDocument", true).unwrap();
    let f = fib::parse(&word).unwrap();
    let t = ole
        .read_stream(
            if f.base.uses_1table() {
                "1Table"
            } else {
                "0Table"
            },
            true,
        )
        .unwrap();
    let styles = Stylesheet::load(&t, f.pair(1));
    let refs: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/office_rust_reference_068/doc_styles.json"))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(styles.styles.len(), refs.as_array().unwrap().len());
    for v in refs.as_array().unwrap() {
        let id = v["id"].as_u64().unwrap() as u16;
        let s = styles.get(id);
        assert_eq!(
            serde_json::json!({"id":id,"name":s.name,"character":s.character,"heading":s.heading,"title":s.title,"toc":s.toc,"code":s.code}),
            *v
        );
    }
}
#[test]
fn relative_toggles_and_visibility() {
    let base = Character {
        bold: true,
        ..Default::default()
    };
    let s = base.apply(&[0x35, 0x08, 0x81, 0x3c, 0x08, 1, 0x48, 0x2a, 1], &base);
    assert!(!s.bold);
    assert!(s.hidden);
    assert!(s.superscript);
    assert!(!s.subscript);
}
