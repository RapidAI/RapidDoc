use rapiddoc_core::{ErrorCode, Limits};
use std::path::PathBuf;
#[test]
fn actual_legacy_pdf_vectors_and_limits() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let fixture: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-vector-primitives.json")).unwrap(),
    )
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let bytes = std::fs::read(root.join(case["pdf"].as_str().unwrap())).unwrap();
        let actual = rapiddoc_pdf::legacy_vectors::read(
            &library,
            &bytes,
            case["page_index"].as_u64().unwrap() as usize,
            None,
            &Limits::default(),
        )
        .unwrap();
        let actual = serde_json::to_value(actual).unwrap();
        if actual != case["expected"] {
            std::fs::write(
                root.join("build/pdf-vector-mismatch.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"actual":actual,"expected":case["expected"]}),
                )
                .unwrap(),
            )
            .unwrap();
        }
        assert_eq!(
            actual, case["expected"],
            "{} page {}",
            case["pdf"], case["page_index"]
        );
    }
    let bytes =
        std::fs::read(root.join("build/migration/pdf-vector-primitives/geometry.pdf")).unwrap();
    for limits in [
        Limits {
            max_objects_per_page: 1,
            ..Limits::default()
        },
        Limits {
            max_path_segments_per_page: 1,
            ..Limits::default()
        },
        Limits {
            max_pages: 1,
            ..Limits::default()
        },
    ] {
        assert_eq!(
            rapiddoc_pdf::legacy_vectors::read(&library, &bytes, 0, None, &limits)
                .unwrap_err()
                .code,
            ErrorCode::ResourceLimit
        );
    }
    assert_eq!(
        rapiddoc_pdf::legacy_vectors::read(&library, &bytes, 999, None, &Limits::default())
            .unwrap_err()
            .code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        rapiddoc_pdf::legacy_vectors::read(&library, b"invalid", 0, None, &Limits::default())
            .unwrap_err()
            .code,
        ErrorCode::CorruptDocument
    );
    rapiddoc_pdf::legacy_vectors::read(&library, &bytes, 0, None, &Limits::default()).unwrap();
}
