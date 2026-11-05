use rapiddoc::ocr::{parse, Config};
use rapiddoc_core::{ErrorCode, Limits, PageRange};
use std::path::Path;

#[test]
fn invalid_ocr_config_is_rejected_before_loading_pdfium_or_calling_inference() {
    for dpi in [0., f64::NAN, 1201.] {
        let result = parse(
            Path::new("missing.dll"),
            b"",
            None,
            &PageRange {
                start: 0,
                end: None,
            },
            &Limits::default(),
            &Config {
                dpi,
                ..Default::default()
            },
            |_| panic!("invalid config must not run OCR"),
        );
        assert_eq!(result.err().unwrap().code, ErrorCode::InvalidInput);
    }
}
