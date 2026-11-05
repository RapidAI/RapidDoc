use rapiddoc_office::{doc::body, ole::OleReader, document_output as output};
use serde_json::Value;
use std::io::{Cursor, Write};
fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn bytes(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn notes_captions_and_story_resets_match_python_through_public_exports() {
    verify("doc_stories.json");
}
#[test]
fn textboxes_comments_and_literal_code_match_explicit_expectations_and_python() {
    verify("doc_extensions.json");
}
fn verify(fixture: &str) {
    let cases: Value = serde_json::from_slice(
        &std::fs::read(
            root()
                .join("tests/fixtures/office_rust_reference_068")
                .join(fixture),
        )
        .unwrap(),
    )
    .unwrap();
    for c in cases["cases"].as_array().unwrap() {
        let mut ole = cfb::CompoundFile::create(Cursor::new(vec![])).unwrap();
        for (name, field) in [("WordDocument", "word_hex"), ("1Table", "table_hex")] {
            ole.create_stream(name)
                .unwrap()
                .write_all(&bytes(c[field].as_str().unwrap()))
                .unwrap();
        }
        if let Some(native) = c["native_hex"].as_str().filter(|s| !s.is_empty()) {
            ole.create_storage("ObjectPool").unwrap();
            ole.create_storage("ObjectPool/_100").unwrap();
            ole.create_stream("ObjectPool/_100/Equation Native")
                .unwrap()
                .write_all(&bytes(native))
                .unwrap();
        }
        let data = ole.into_inner().into_inner();
        let inputs = root().join("build/doc-story-inputs");
        std::fs::create_dir_all(&inputs).unwrap();
        std::fs::write(
            inputs.join(format!("{}.doc", c["name"].as_str().unwrap())),
            &data,
        )
        .unwrap();
        if c["name"] == "basic" {
            std::fs::write(root().join("build/doc-stories-synthetic.doc"), &data).unwrap();
        }
        let mut ole = OleReader::open(data.into()).unwrap();
        let parsed = body::convert(&mut ole).unwrap();
        assert_eq!(
            serde_json::to_value(&parsed.pages).unwrap(),
            c["pages"],
            "{} raw",
            c["name"]
        );
        let middle = output::middle(parsed.pages, cases["version"].as_str().unwrap())
            .unwrap()
            .middle;
        assert_eq!(
            serde_json::to_value(&middle).unwrap(),
            c["middle"],
            "{} public",
            c["name"]
        );
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
