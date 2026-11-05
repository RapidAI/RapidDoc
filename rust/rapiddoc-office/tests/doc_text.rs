use rapiddoc_office::{
    doc::{fib, pieces},
    ole::OleReader,
};

#[test]
fn surrogate_dbcs_and_ansi_positions() {
    let unicode: Vec<u8> = "A😀B".encode_utf16().flat_map(u16::to_le_bytes).collect();
    let p = pieces::Piece {
        cp_start: 0,
        cp_end: 4,
        fc: 0,
        compressed: false,
        prm: vec![],
    };
    let t = pieces::extract_mapped(&unicode, &[p], 4, 0x409);
    assert_eq!(t.chars.iter().collect::<String>(), "A😀B");
    assert_eq!(t.cps, vec![0, 1, 3]);
    assert_eq!(t.fcs, vec![0, 2, 6]);
    let p = pieces::Piece {
        cp_start: 0,
        cp_end: 3,
        fc: 0,
        compressed: true,
        prm: vec![],
    };
    let t = pieces::extract_mapped(&[0xd6, 0xd0, b'A'], &[p.clone()], 3, 0x804);
    assert_eq!(t.chars.iter().collect::<String>(), "中A");
    assert_eq!(t.cps, vec![0, 2]);
    assert_eq!(pieces::extract(&[0x80, 0xe9, b'A'], &[p], 3, 0x409), "€éA");
}
#[test]
fn real_doc_text_matches_python_cp_and_fc() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cases: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/office_rust_reference_068/doc_text.json"))
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
        let w = ole.read_stream("WordDocument", true).unwrap();
        let f = fib::parse(&w).unwrap();
        let t = ole
            .read_stream(
                if f.base.uses_1table() {
                    "1Table"
                } else {
                    "0Table"
                },
                false,
            )
            .unwrap();
        let p = pieces::parse_clx(&t, f.clx()).unwrap();
        let text = pieces::extract_mapped(&w, &p, f.ccp_text(), f.base.lid);
        assert_eq!(
            text.chars.iter().collect::<String>(),
            c["chars"].as_str().unwrap()
        );
        assert_eq!(serde_json::to_value(text.cps).unwrap(), c["cps"]);
        assert_eq!(serde_json::to_value(text.fcs).unwrap(), c["fcs"]);
    }
}
