use rapiddoc_core::{ErrorCode, Limits};
use std::path::PathBuf;
#[test]
fn actual_old_pdf_link_annotations() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let f: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-link-annotations.json")).unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let bytes = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
        let actual = rapiddoc_pdf::link_annotations::read(
            &library,
            &bytes,
            c["page_index"].as_u64().unwrap() as usize,
            None,
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            c["expected"],
            "{} page {}",
            c["pdf"],
            c["page_index"]
        );
    }
    assert_eq!(
        rapiddoc_pdf::link_annotations::read(&library, b"invalid", 0, None, &Limits::default())
            .unwrap_err()
            .code,
        ErrorCode::CorruptDocument
    );
    let bytes = std::fs::read(root.join("build/migration/pdf-link-annotations/links.pdf")).unwrap();
    assert_eq!(
        rapiddoc_pdf::link_annotations::read(&library, &bytes, 999, None, &Limits::default())
            .unwrap_err()
            .code,
        ErrorCode::InvalidInput
    );
}
