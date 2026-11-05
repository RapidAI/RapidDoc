use rapiddoc_office::{
    ole::{casefold, OleReader},
    package::*,
    xml, ErrorKind,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    io::{Cursor, Write},
    path::PathBuf,
    sync::Arc,
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn reference() -> Value {
    serde_json::from_slice(
        &std::fs::read(root().join("tests/fixtures/office_rust_reference_068/foundation.json"))
            .unwrap(),
    )
    .unwrap()
}
fn expected(value: &Value) -> Option<&str> {
    value.as_str()
}

#[test]
fn original_python_relationship_helpers() {
    let reference = reference();
    for case in reference["relationship_targets"].as_array().unwrap() {
        assert_eq!(
            resolve_relationship_target(case["rels"].as_str().unwrap(), case["target"].as_str())
                .as_deref(),
            expected(&case["expected"]),
            "{case}"
        );
    }
    for case in reference["embedded_targets"].as_array().unwrap() {
        assert_eq!(
            resolve_internal_target(
                case["source"].as_str().unwrap(),
                case["target"].as_str().unwrap()
            )
            .as_deref(),
            expected(&case["expected"]),
            "{case}"
        );
    }
    for case in reference["relationship_members"].as_array().unwrap() {
        assert_eq!(
            part_relationships_member(case["part"].as_str().unwrap()).as_deref(),
            expected(&case["expected"])
        );
    }
    for case in reference["source_directories"].as_array().unwrap() {
        assert_eq!(
            relationship_source_base_dir(case["rels"].as_str().unwrap()).as_deref(),
            expected(&case["expected"])
        );
    }
    for case in reference["casefold"].as_array().unwrap() {
        assert_eq!(
            casefold(case["text"].as_str().unwrap()),
            case["expected"].as_str().unwrap()
        );
    }
}

#[test]
fn ole_streams_match_python_bytes_and_lookup() {
    for case in reference()["ole"].as_array().unwrap() {
        let bytes = std::fs::read(root().join(case["input"].as_str().unwrap())).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            case["sha256"].as_str().unwrap()
        );
        let mut reader = OleReader::open(bytes.into()).unwrap();
        assert_eq!(
            reader.format().unwrap().extension(),
            case["format"].as_str().unwrap()
        );
        let actual: Vec<_> = reader
            .stream_names(None)
            .into_iter()
            .map(str::to_owned)
            .collect();
        let expected: Vec<_> = case["streams"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["name"].as_str().unwrap())
            .collect();
        assert_eq!(actual, expected, "{}", case["input"]);
        for stream in case["streams"].as_array().unwrap() {
            let data = reader
                .read_stream(&stream["name"].as_str().unwrap().to_uppercase(), true)
                .unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&data)),
                stream["sha256"].as_str().unwrap()
            );
        }
    }
}

#[test]
fn twenty_baseline_containers_including_strict_and_nonstandard_parts() {
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(root().join("tests/fixtures/office_rust_reference_068/manifest.json"))
            .unwrap(),
    )
    .unwrap();
    for case in manifest["cases"].as_array().unwrap() {
        let format = case["format"].as_str().unwrap();
        if !matches!(format, "docx" | "xlsx" | "pptx") {
            continue;
        }
        let bytes: Arc<[u8]> = std::fs::read(root().join(case["input"].as_str().unwrap()))
            .unwrap()
            .into();
        let mut package = Package::open(bytes).unwrap();
        assert_eq!(
            package.format().unwrap().extension(),
            format,
            "{}",
            case["id"]
        );
        let main = package.main_part().unwrap();
        let text = package.read_xml(&main).unwrap().unwrap();
        xml::validate_stream(&text).unwrap();
        let _types = package.content_types().unwrap();
        assert!(package.contains(&main));
        assert!(package.read("absent-part").unwrap().is_none());
        assert_eq!(
            package.required("absent-part").unwrap_err().kind,
            ErrorKind::MissingPart
        );
    }
}

fn archive(members: &[(&str, &[u8])]) -> Arc<[u8]> {
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, data) in members {
        zip.start_file(*name, zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(data).unwrap();
    }
    zip.finish().unwrap().into_inner().into()
}

#[test]
fn external_relationships_never_become_package_reads_and_patches_share_bytes() {
    let rels = br#"<Relationships><Relationship Id="external" Type="a/image" Target="https://example.test/a" TargetMode="External"/><Relationship Id="internal" Type="a/officeDocument" Target="/main.xml"/></Relationships>"#;
    let mut package =
        Package::open(archive(&[(ROOT_RELS, rels), ("main.xml", b"<document/>")])).unwrap();
    let relationships = package.relationships(None).unwrap();
    assert!(relationships[0].external);
    assert!(relationships[0].resolved.is_none());
    assert_eq!(package.main_part().unwrap(), "main.xml");
    let data: Arc<[u8]> = Arc::from(b"replacement".as_slice());
    package.patch("main.xml".to_owned(), data.clone());
    let read = package.required("main.xml").unwrap();
    assert!(Arc::ptr_eq(&data, &read));
}

#[test]
fn xml_encodings_and_corruption() {
    let text = "<?xml version=\"1.0\" encoding=\"UTF-16\"?><root>中文</root>";
    for little in [true, false] {
        let mut bytes = if little {
            vec![0xff, 0xfe]
        } else {
            vec![0xfe, 0xff]
        };
        for unit in text.encode_utf16() {
            bytes.extend(if little {
                unit.to_le_bytes()
            } else {
                unit.to_be_bytes()
            });
        }
        let decoded = xml::decode(&bytes).unwrap();
        xml::validate_stream(&decoded).unwrap();
        assert_eq!(
            xml::parse(&decoded).unwrap().root_element().text(),
            Some("中文")
        );
    }
    let decoded =
        xml::decode(b"<?xml version='1.0' encoding='windows-1252'?><root>\xe9</root>").unwrap();
    assert_eq!(
        xml::parse(&decoded).unwrap().root_element().text(),
        Some("é")
    );
    assert!(xml::decode(&[0xff, 0xfe, 0]).is_err());
    assert!(xml::validate_stream("<root>").is_err());
    assert!(xml::validate_stream("<a/><b/>").is_err());
    assert!(xml::validate_stream("<a></b>").is_err());
    assert!(xml::validate_stream("<a/>text").is_err());
    assert!(xml::validate_stream("text<a/>").is_err());
    assert!(xml::validate_stream("<![CDATA[text]]><a/>").is_err());
    assert!(Package::open(Arc::from(b"not a zip".as_slice())).is_err());
}
