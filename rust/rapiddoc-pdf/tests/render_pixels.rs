use rapiddoc_core::Limits;
use rapiddoc_pdf::render::{render_model_page, render_model_page_with_pixels};
use std::path::Path;

#[test]
fn layout_render_pixels_are_exact_public_png_pixels() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
    for file in ["demo/pdfs/示例1-论文模板.pdf", "demo/pdfs/small_ocr.pdf"] {
        let bytes = std::fs::read(root.join(file)).unwrap();
        for dpi in [72., 200.] {
            let (page, pixels) = render_model_page_with_pixels(
                &library,
                &bytes,
                0,
                dpi,
                16_000_000,
                None,
                &Limits::pdf_host(),
            )
            .unwrap();
            let decoded = image::load_from_memory(&page.png).unwrap().to_rgb8();
            assert_eq!(pixels, decoded.into_raw());
            let old = render_model_page(
                &library,
                &bytes,
                0,
                dpi,
                16_000_000,
                None,
                &Limits::pdf_host(),
            )
            .unwrap();
            assert_eq!(
                (page.width, page.height, page.scale),
                (old.width, old.height, old.scale)
            );
            assert_eq!(page.png, old.png);
        }
        for dpi in [0., f64::NAN, 1201.] {
            assert_eq!(
                render_model_page_with_pixels(
                    &library,
                    &bytes,
                    0,
                    dpi,
                    16_000_000,
                    None,
                    &Limits::pdf_host()
                )
                .err()
                .unwrap()
                .code,
                render_model_page(
                    &library,
                    &bytes,
                    0,
                    dpi,
                    16_000_000,
                    None,
                    &Limits::pdf_host()
                )
                .err()
                .unwrap()
                .code
            );
        }
    }
}
