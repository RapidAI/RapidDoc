use rapiddoc_core::{ErrorCode, Limits, PageRange};
use std::path::Path;

#[test]
fn invalid_original_image_options_before_runtime() {
    use rapiddoc_pdf::original_images::{extract, Config};
    for config in [
        Config {
            max_depth: 33,
            ..Config::default()
        },
        Config {
            max_pixels: 0,
            ..Config::default()
        },
        Config {
            max_pixels: 32_000_001,
            ..Config::default()
        },
    ] {
        let result = extract(
            Path::new("missing.dll"),
            &[1],
            0,
            None,
            &config,
            &Limits::default(),
        );
        assert_eq!(result.err().unwrap().code, ErrorCode::InvalidInput);
    }
}

#[test]
fn reject_invalid_input_before_loading_runtime() {
    let path = Path::new("does-not-exist-pdfium.dll");
    let error = rapiddoc_pdf::extract(path, &[], None, &PageRange::default(), &Limits::default())
        .expect_err("empty");
    assert_eq!(error.code, ErrorCode::InvalidInput);
    let limits = Limits {
        max_input_bytes: 1,
        ..Limits::default()
    };
    let error = rapiddoc_pdf::extract(path, &[1, 2], None, &PageRange::default(), &limits)
        .expect_err("large");
    assert_eq!(error.code, ErrorCode::ResourceLimit);
    let error = rapiddoc_pdf::extract(path, &[1], None, &PageRange::default(), &limits)
        .expect_err("runtime");
    assert_eq!(error.code, ErrorCode::RuntimeUnavailable);
}

#[test]
fn invalid_render_options_do_not_load_runtime() {
    let path = Path::new("missing-pdfium.dll");
    for dpi in [0., -1., f64::NAN, 1201.] {
        let result =
            rapiddoc_pdf::render::render_page(path, &[1], 0, dpi, 100, None, &Limits::default());
        assert_eq!(
            result.err().expect("invalid dpi").code,
            ErrorCode::InvalidInput
        );
    }
    let result =
        rapiddoc_pdf::render::render_page(path, &[1], 0, 144., 0, None, &Limits::default());
    assert_eq!(
        result.err().expect("invalid pixels").code,
        ErrorCode::InvalidInput
    );
}
