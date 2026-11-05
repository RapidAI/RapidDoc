use rapiddoc_office::{
    doc::{
        fib,
        formatting::{self, Formatting},
    },
    ole::OleReader,
};
#[test]
fn real_fkp_runs_match_python() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cases: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/office_rust_reference_068/doc_formatting.json"))
            .unwrap(),
    )
    .unwrap();
    for c in cases.as_array().unwrap() {
        let mut ole = OleReader::open(
            std::fs::read(root.join(c["path"].as_str().unwrap()))
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
                false,
            )
            .unwrap();
        let runs = Formatting::load(&word, &table, f.pair(12), f.pair(13));
        let chars: Vec<_> = runs
            .characters
            .iter()
            .map(|r| serde_json::json!({"start":r.start,"end":r.end,"grpprl":r.grpprl}))
            .collect();
        let paras: Vec<_> = runs
            .paragraphs
            .iter()
            .map(|r| serde_json::json!({"start":r.start,"end":r.end,"style_id":r.style_id}))
            .collect();
        assert_eq!(serde_json::json!(chars), c["characters"]);
        assert_eq!(serde_json::json!(paras), c["paragraphs"]);
        for r in &runs.characters {
            assert!(Formatting::at(&runs.characters, r.start).is_some());
        }
    }
}
#[test]
fn sprm_widths_and_recoverable_tail() {
    let d = [
        0x16, 0x24, 1, 0x0b, 0x46, 2, 0, 0x03, 0x6a, 1, 2, 3, 4, 0x35, 0x08,
    ];
    let r = formatting::sprms(&d);
    assert_eq!(
        r,
        vec![
            (0x2416, &[1][..]),
            (0x460b, &[2, 0][..]),
            (0x6a03, &[1, 2, 3, 4][..])
        ]
    );
}
