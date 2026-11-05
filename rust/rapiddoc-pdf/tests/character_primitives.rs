use rapiddoc_core::{ErrorCode, Limits};
use std::path::PathBuf;
#[test]
fn actual_old_character_primitives_and_limits() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    for name in ["pdf-character-primitives", "pdf-character-edges"] {
        let fixture: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join(format!("build/migration/{name}.json"))).unwrap(),
        )
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let bytes = std::fs::read(root.join(case["pdf"].as_str().unwrap())).unwrap();
            let actual = rapiddoc_pdf::character_primitives::read(
                &library,
                &bytes,
                case["page_index"].as_u64().unwrap() as usize,
                None,
                case["extended"].as_bool().unwrap(),
                &Limits::default(),
            )
            .unwrap();
            let actual = serde_json::to_value(actual).unwrap();
            if actual != case["expected"] {
                std::fs::write(
                    root.join("build/pdf-character-mismatch.json"),
                    serde_json::to_vec_pretty(
                        &serde_json::json!({"actual":actual,"expected":case["expected"]}),
                    )
                    .unwrap(),
                )
                .unwrap();
            }
            assert_eq!(
                actual, case["expected"],
                "{} page {} extended {}",
                case["pdf"], case["page_index"], case["extended"]
            );
        }
    }
    let bytes =
        std::fs::read(root.join("build/migration/pdf-vector-primitives/geometry.pdf")).unwrap();
    assert_eq!(
        rapiddoc_pdf::character_primitives::read(
            &library,
            &bytes,
            0,
            None,
            true,
            &Limits {
                max_chars_per_page: 1,
                ..Limits::default()
            }
        )
        .unwrap_err()
        .code,
        ErrorCode::ResourceLimit
    );
    assert_eq!(
        rapiddoc_pdf::character_primitives::read(
            &library,
            b"bad",
            0,
            None,
            true,
            &Limits::default()
        )
        .unwrap_err()
        .code,
        ErrorCode::CorruptDocument
    );
    rapiddoc_pdf::character_primitives::read(&library, &bytes, 0, None, true, &Limits::default())
        .unwrap();
}
