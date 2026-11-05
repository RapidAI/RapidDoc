use rapiddoc_office::{
    doc::{fib, formatting::Formatting, paragraph::Paragraph},
    ole::OleReader,
};
#[test]
fn real_papx_matches_original_deltas() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut ole = OleReader::open(
        std::fs::read(root.join("demo/ms_office_docs/docx_01.doc"))
            .unwrap()
            .into(),
    )
    .unwrap();
    let word = ole.read_stream("WordDocument", true).unwrap();
    let f = fib::parse(&word).unwrap();
    let table = ole
        .read_stream(
            if f.base.uses_1table() {
                "1Table"
            } else {
                "0Table"
            },
            true,
        )
        .unwrap();
    let data = ole.read_stream("Data", false).unwrap();
    let runs = Formatting::load(&word, &table, f.pair(12), f.pair(13));
    let refs: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/office_rust_reference_068/doc_paragraph.json"))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(runs.paragraphs.len(), refs.as_array().unwrap().len());
    for (r, v) in runs.paragraphs.iter().zip(refs.as_array().unwrap()) {
        let mut p = Paragraph::default();
        p.apply(&r.grpprl, &data);
        assert_eq!(
            serde_json::to_value(p).unwrap(),
            v["delta"],
            "FC {}",
            r.start
        );
    }
}
#[test]
fn huge_papx_and_depth_delta() {
    let mut p = Paragraph::default();
    let data = [3, 0, 0x16, 0x24, 1];
    p.apply(
        &[
            0x46, 0x66, 0, 0, 0, 0, 0x49, 0x66, 2, 0, 0, 0, 0x4a, 0x66, 0xff, 0xff, 0xff, 0xff,
        ],
        &data,
    );
    assert_eq!(p.in_table, Some(true));
    assert_eq!(p.table_depth, Some(1));
    p.apply(&[0x40, 0x26, 9], &[]);
    assert_eq!(p.outline_level, Some(-1));
}
