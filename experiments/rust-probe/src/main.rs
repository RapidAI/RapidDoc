//! 独立进程 PDFium 探针；不与 pypdfium2 共享初始化状态或句柄。

use pdfium_render::prelude::*;
use rapiddoc_core::Limits;

/// 使用显式 PDFium 动态库路径读取页面文字；错误返回非零状态。
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: rapiddoc-runtime-probe PDFIUM_LIBRARY INPUT_PDF".into());
    }
    let bytes = std::fs::read(&args[2])?;
    let limits = Limits::default();
    limits.check_input(&bytes)?;
    let pdfium = Pdfium::new(Pdfium::bind_to_library(&args[1])?);
    let document = pdfium.load_pdf_from_byte_slice(&bytes, None)?;
    if usize::try_from(document.pages().len())? > limits.max_pages {
        return Err("page limit exceeded".into());
    }
    let mut pages = Vec::new();
    for (index, page) in document.pages().iter().enumerate() {
        let text = page.text()?;
        let mut characters = Vec::new();
        for character in text.chars().iter() {
            let bounds = character.tight_bounds()?;
            let (x, y) = character.origin()?;
            characters.push(serde_json::json!({
                "code": character.unicode_value(),
                "text": character.unicode_string(),
                "font_name": character.font_name(),
                "angle_degrees": character.angle_degrees()?,
                "origin_pdf_points": [x.value, y.value],
                "tight_bounds_pdf_points": [bounds.left().value, bounds.bottom().value, bounds.right().value, bounds.top().value]
            }));
        }
        pages.push(
            serde_json::json!({"page_index": index, "width_pt": page.width().value,
            "height_pt": page.height().value, "text": text.all(), "char_count": text.chars().len(),
            "characters": characters, "coordinate_system": "pdfium_native_bottom_left_points"}),
        );
    }
    println!("{}", serde_json::to_string_pretty(&pages)?);
    Ok(())
}
