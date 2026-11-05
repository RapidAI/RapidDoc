use rapiddoc_core::Limits;
use serde_json::Value;
use sha2::{Digest, Sha256};
#[test]
fn actual_unembedded_cjk_geometry_and_pixels() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let lib = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/pdf-fixed-font-runtime.json")).unwrap(),
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let bytes = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            c["pdf_sha256"].as_str().unwrap()
        );
        let index = c["page_index"].as_u64().unwrap() as usize;
        let actual = rapiddoc_pdf::character_primitives::read(
            &lib,
            &bytes,
            index,
            None,
            true,
            &Limits::default(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            c["expected"],
            "geometry {}",
            c["label"]
        );
        for r in c["renders"].as_array().unwrap() {
            let image = rapiddoc_pdf::render::render_page(
                &lib,
                &bytes,
                index,
                r["dpi"].as_f64().unwrap(),
                2_000_000,
                None,
                &Limits::default(),
            )
            .unwrap();
            assert_eq!(
                serde_json::json!([image.width, image.height]),
                serde_json::json!([r["width"], r["height"]])
            );
            let rgba = image::load_from_memory(&image.png).unwrap().into_rgba8();
            assert_eq!(
                format!("{:x}", Sha256::digest(rgba.as_raw())),
                r["rgba_sha256"].as_str().unwrap(),
                "pixels {}",
                c["label"]
            );
        }
    }
    let info = rapiddoc_pdf::font_runtime_info(&lib).unwrap();
    assert_eq!(info["font_sha256"], rapiddoc_pdf::fonts::SHA256);
    assert!(info["bundled_requests"].as_u64().unwrap() > 0);
    assert_eq!(info["released"], false);
}
