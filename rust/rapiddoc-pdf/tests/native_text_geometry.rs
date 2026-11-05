use rapiddoc_core::Limits;
use std::path::PathBuf;
#[test]
fn complete_old_pdf_text_geometry() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    for name in ["pdf-native-geometry", "pdf-native-dedup-extra"] {
        let f: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join(format!("build/migration/{name}.json"))).unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let bytes = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
            let actual = rapiddoc_pdf::character_primitives::read_geometry(
                &library,
                &bytes,
                c["page_index"].as_u64().unwrap() as usize,
                None,
                c["extended"].as_bool().unwrap(),
                &Limits::default(),
            )
            .unwrap();
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                c["expected"],
                "{name} {} {}",
                c["page_index"],
                c["extended"]
            );
        }
    }
}
