use rapiddoc_core::{ErrorCode, Limits};
use std::path::PathBuf;
#[test]
fn actual_old_pdf_image_infos() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let f: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-image-infos.json")).unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let bytes = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
        let got = rapiddoc_pdf::image_infos::read(
            &library,
            &bytes,
            c["page_index"].as_u64().unwrap() as usize,
            None,
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(got).unwrap(),
            c["expected"],
            "{} page {} {:?}",
            c["pdf"],
            c["page_index"],
            c["label"]
        );
    }
    let bad = rapiddoc_pdf::image_infos::read(&library, b"invalid", 0, None, &Limits::default())
        .unwrap_err();
    assert_eq!(bad.code, ErrorCode::CorruptDocument);
    let bytes = std::fs::read(root.join("build/migration/pdf-image-infos/images.pdf")).unwrap();
    assert_eq!(
        rapiddoc_pdf::image_infos::read(&library, &bytes, 999, None, &Limits::default())
            .unwrap_err()
            .code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        rapiddoc_pdf::image_infos::read(&library, &bytes, 0, Some("\0"), &Limits::default())
            .unwrap_err()
            .code,
        ErrorCode::InvalidInput
    );
    let limits = Limits {
        max_objects_per_page: 0,
        ..Limits::default()
    };
    assert_eq!(
        rapiddoc_pdf::image_infos::read(&library, &bytes, 0, None, &limits)
            .unwrap_err()
            .code,
        ErrorCode::ResourceLimit
    );
}
