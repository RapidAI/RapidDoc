use rapiddoc_office::{
    doc::{body, fields::Fields},
    docx::spans::{self, Span},
    ole::OleReader,
};
fn put(f: &mut Fields, out: &mut Vec<Span>, s: &str) {
    for c in s.chars() {
        f.push(
            out,
            Span::Text {
                content: c.to_string(),
                styles: vec![],
            },
        );
    }
}
#[test]
fn nested_cached_fields_and_multiline_toc() {
    let mut f = Fields::default();
    let mut out = vec![];
    f.begin();
    put(&mut f, &mut out, " REF outer ");
    f.separator();
    put(&mut f, &mut out, "A");
    f.begin();
    put(&mut f, &mut out, "PAGE");
    f.separator();
    put(&mut f, &mut out, "1");
    f.end(&mut out);
    f.end(&mut out);
    assert_eq!(spans::visible(&out), "A1");
    f.begin();
    put(&mut f, &mut out, "TOC");
    f.separator();
    put(&mut f, &mut out, "first");
    assert_eq!(spans::visible(&out), "A1first");
    out.clear();
    put(&mut f, &mut out, "second");
    f.end(&mut out);
    assert_eq!(spans::visible(&out), "second");
}
#[test]
fn hyperlink_switches_and_rejected_scheme() {
    use rapiddoc_office::doc::fields::hyperlink_target;
    assert_eq!(
        hyperlink_target(r#" HYPERLINK "https://example.com" \l "part" \o "hint" "#),
        Some("https://example.com#part".into())
    );
    assert_eq!(
        hyperlink_target(r#"HYPERLINK \l "bookmark""#),
        Some("#bookmark".into())
    );
    assert_eq!(hyperlink_target(r#"HYPERLINK "javascript:alert(1)""#), None);
}
#[test]
fn real_doc_sections_and_no_raw_field_controls() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut ole = OleReader::open(
        std::fs::read(root.join("demo/ms_office_docs/docx_01.doc"))
            .unwrap()
            .into(),
    )
    .unwrap();
    let result = body::convert(&mut ole).unwrap();
    assert_eq!(result.pages.len(), 3);
    let raw = serde_json::to_string(&result).unwrap();
    for code in ["\\u0001", "\\u0007", "\\u0013", "\\u0014", "\\u0015"] {
        assert!(!raw.contains(code), "{code}");
    }
    assert!(!result.diagnostics.is_empty());
}
#[test]
fn missing_clx_single_piece_and_fixed_slot_fib() {
    use rapiddoc_office::doc::{fib, pieces};
    let mut w = vec![0u8; 0x400];
    w[..2].copy_from_slice(&0xa5ecu16.to_le_bytes());
    w[2..4].copy_from_slice(&0xc1u16.to_le_bytes());
    w[0x4c..0x50].copy_from_slice(&3u32.to_le_bytes());
    let f = fib::parse(&w).unwrap();
    assert_eq!(f.ccp_text(), 3);
    let p = pieces::Piece {
        cp_start: 0,
        cp_end: 3,
        fc: 0,
        compressed: true,
        prm: vec![],
    };
    assert_eq!(pieces::extract(b"abc", &[p], 3, 0x409), "abc");
}
