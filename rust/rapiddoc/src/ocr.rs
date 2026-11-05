//! 仅对无原生文字且需要 OCR 的页面回调推理，保留原始事实和页号。
use rapiddoc_core::{document::Document, ocr::TextLine, Error, ErrorCode, Limits, PageRange};
use serde::Serialize;
use std::{collections::BTreeSet, path::Path};

pub struct Config {
    pub dpi: f64,
    pub max_pixels_per_page: usize,
    pub max_total_pixels: usize,
    pub max_ocr_pages: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            dpi: 144.,
            max_pixels_per_page: 16_000_000,
            max_total_pixels: 128_000_000,
            max_ocr_pages: 256,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct PageText {
    pub page_index: usize,
    pub method: &'static str,
    pub text: String,
    /// 旋转裁剪后视觉页面左上原点 point，区别于显式 ocr_pdf_page 的像素坐标。
    pub ocr_lines: Vec<TextLine>,
}

#[derive(Debug, Serialize)]
pub struct Output {
    pub text: String,
    pub document: Document,
    pub pages: Vec<PageText>,
}

fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}

/// 仅页面级 native/OCR 路由，不对部分原生文字页面内的图片块自动补识别。
/// 回调接收 PNG 并返回像素四角；PDFium 锁在推理前已经释放。
#[allow(clippy::too_many_arguments)]
pub fn parse<F>(
    library: &Path,
    bytes: &[u8],
    password: Option<&str>,
    range: &PageRange,
    limits: &Limits,
    config: &Config,
    mut recognize: F,
) -> Result<Output, Error>
where
    F: FnMut(&[u8]) -> Result<Vec<TextLine>, Error>,
{
    if !config.dpi.is_finite()
        || config.dpi <= 0.
        || config.dpi > 1200.
        || rapiddoc_core::resources::limit(config.max_pixels_per_page) == 0
        || rapiddoc_core::resources::limit(config.max_total_pixels) == 0
        || rapiddoc_core::resources::limit(config.max_ocr_pages) == 0
    {
        return Err(error(ErrorCode::InvalidInput, "PDF OCR 配置无效"));
    }
    let output = super::parse_pdf_text(library, bytes, password, range, limits, true)?;
    let mut document = output.document;
    let selected: BTreeSet<_> = document
        .diagnostics
        .iter()
        .filter(|d| d.code == "ocr_required")
        .map(|d| d.page_index)
        .collect();
    if selected.len() > rapiddoc_core::resources::limit(config.max_ocr_pages) {
        return Err(error(ErrorCode::ResourceLimit, "需 OCR 的页数超过预算"));
    }
    let mut remaining = rapiddoc_core::resources::limit(config.max_total_pixels);
    let mut total_chars = 0usize;
    let mut pages = Vec::with_capacity(document.pages.len());
    for page in &document.pages {
        let result = if selected.contains(&page.page_index) {
            if remaining == 0 {
                return Err(error(ErrorCode::ResourceLimit, "OCR 累计像素超限"));
            }
            let image = rapiddoc_pdf::render::render_page(
                library,
                bytes,
                page.page_index,
                config.dpi,
                remaining.min(rapiddoc_core::resources::limit(config.max_pixels_per_page)),
                password,
                limits,
            )?;
            let pixels = (image.width as usize)
                .checked_mul(image.height as usize)
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "OCR 像素数量溢出"))?;
            remaining = remaining
                .checked_sub(pixels)
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "OCR 累计像素超限"))?;
            let mut lines = recognize(&image.png)?;
            if lines.len() > rapiddoc_core::resources::limit(1000) {
                return Err(error(ErrorCode::ResourceLimit, "OCR 返回区域数量超限"));
            }
            let text_count = lines
                .iter()
                .try_fold(lines.len().saturating_sub(1), |count, line| {
                    count.checked_add(line.text.chars().count())
                })
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "OCR 字符数量溢出"))?;
            if text_count > limits.max_chars_per_page {
                return Err(error(ErrorCode::ResourceLimit, "OCR 单页文字超过预算"));
            }
            for line in &mut lines {
                if !line.confidence.is_finite()
                    || !(0. ..=1.).contains(&line.confidence)
                    || !line.detection_score.is_finite()
                    || !(0. ..=1.).contains(&line.detection_score)
                    || line.points.iter().any(|p| {
                        !p[0].is_finite()
                            || !p[1].is_finite()
                            || p[0] < 0.
                            || p[1] < 0.
                            || p[0] > image.width as f64
                            || p[1] > image.height as f64
                    })
                {
                    return Err(error(ErrorCode::InvalidInput, "OCR 返回坐标或评分无效"));
                }
                for point in &mut line.points {
                    point[0] *= page.visual_width_pt / image.width as f64;
                    point[1] *= page.visual_height_pt / image.height as f64;
                }
            }
            let text = lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            PageText {
                page_index: page.page_index,
                method: "ocr",
                text,
                ocr_lines: lines,
            }
        } else {
            PageText {
                page_index: page.page_index,
                method: "native",
                text: page.text.clone(),
                ocr_lines: Vec::new(),
            }
        };
        let count = result.text.chars().count();
        total_chars = total_chars
            .checked_add(count)
            .ok_or_else(|| error(ErrorCode::ResourceLimit, "OCR 字符数量溢出"))?;
        if count > limits.max_chars_per_page || total_chars > limits.max_total_chars {
            return Err(error(ErrorCode::ResourceLimit, "合并文字超过字符预算"));
        }
        pages.push(result);
    }
    for diagnostic in &mut document.diagnostics {
        if diagnostic.code == "ocr_required" {
            diagnostic.code = "ocr_completed".into();
            diagnostic.message =
                "已执行页面 OCR；原始 facts.text 保持原生文字，识别文字位于 pages".into();
        }
    }
    let text = pages
        .iter()
        .map(|page| page.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    Ok(Output {
        text,
        document,
        pages,
    })
}
