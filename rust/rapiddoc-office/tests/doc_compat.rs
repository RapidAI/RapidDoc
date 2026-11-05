use rapiddoc_office::{
    doc::{
        body,
        pieces::{self, Piece},
    },
    ole::OleReader,
    document_output as output,
};
use serde_json::{json, Value};
use std::io::{Cursor, Read, Write};
fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn fixture(name: &str) -> Value {
    let input = std::fs::read(
        root()
            .join("tests/fixtures/office_rust_reference_068")
            .join(name),
    )
    .unwrap();
    let mut bytes = vec![];
    flate2::read::GzDecoder::new(&input[..])
        .read_to_end(&mut bytes)
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
fn bytes(v: &str) -> Vec<u8> {
    v.as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn compressed_and_utf16_characters_keep_python_cp_fc_mapping() {
    for c in fixture("doc_encoding.json.gz").as_array().unwrap() {
        let data = bytes(c["hex"].as_str().unwrap());
        let compressed = c["compressed"].as_bool().unwrap();
        let n = (data.len() / if compressed { 1 } else { 2 }) as u32;
        let mapped = pieces::extract_mapped(
            &data,
            &[Piece {
                cp_start: 0,
                cp_end: n,
                fc: 0,
                compressed,
                prm: vec![],
            }],
            n,
            c["lid"].as_u64().unwrap() as u16,
        );
        let checks = [
            (
                "text",
                json!(mapped.chars.iter().map(char::to_string).collect::<Vec<_>>()),
            ),
            ("cps", json!(mapped.cps)),
            ("fcs", json!(mapped.fcs)),
            ("piece_indexes", json!(mapped.piece_indexes)),
        ];
        for (key, actual) in checks {
            let expected = &c[if key == "text" { "chars" } else { key }];
            let a = actual.as_array().unwrap();
            let b = expected.as_array().unwrap();
            let difference = a.iter().zip(b).position(|(a, b)| a != b);
            assert!(
                a.len() == b.len() && difference.is_none(),
                "{} {key}: lengths {} vs {}; first {:?}",
                c["lid"],
                a.len(),
                b.len(),
                difference.map(|i| (i, &a[i], &b[i]))
            );
        }
    }
}
#[test]
fn invalid_doc_recovery_and_chart_aliases_match_python_end_to_end() {
    let cases = fixture("doc_compat.json.gz");
    for c in cases["cases"].as_array().unwrap() {
        let mut ole = cfb::CompoundFile::create(Cursor::new(vec![])).unwrap();
        for (name, field) in [("WordDocument", "word_hex"), ("1Table", "table_hex")] {
            ole.create_stream(name)
                .unwrap()
                .write_all(&bytes(c[field].as_str().unwrap()))
                .unwrap();
        }
        for (path, data) in c["pool"].as_object().unwrap() {
            let parent = std::path::Path::new(path).parent().unwrap();
            ole.create_storage_all(parent).unwrap();
            ole.create_stream(path)
                .unwrap()
                .write_all(&bytes(data.as_str().unwrap()))
                .unwrap();
        }
        let data = ole.into_inner().into_inner();
        let inputs = root().join("build/doc-compat-inputs");
        std::fs::create_dir_all(&inputs).unwrap();
        std::fs::write(
            inputs.join(format!("{}.doc", c["name"].as_str().unwrap())),
            &data,
        )
        .unwrap();
        let mut ole = OleReader::open(data.into()).unwrap();
        let parsed = body::convert(&mut ole);
        if c.get("error").is_some() {
            assert_eq!(
                parsed
                    .err()
                    .unwrap_or_else(|| panic!("{} unexpectedly accepted", c["name"]))
                    .kind,
                rapiddoc_office::ErrorKind::Malformed,
                "{}",
                c["name"]
            );
            continue;
        }
        let parsed = parsed.unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        assert_eq!(json!(parsed.pages), c["pages"], "{} raw", c["name"]);
        let middle = output::middle(parsed.pages, cases["version"].as_str().unwrap())
            .unwrap()
            .middle;
        assert_eq!(json!(middle), c["middle"], "{} public", c["name"]);
        for mode in [
            "mm_markdown",
            "nlp_markdown",
            "content_list",
            "content_list_v2",
        ] {
            assert_eq!(
                output::render::export(&middle, mode, "images", &Default::default()),
                c["exports"][mode],
                "{} {mode}",
                c["name"]
            );
        }
    }
}
