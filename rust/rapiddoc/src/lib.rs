//! Shared Rust document SDK and parsing kernels; no Python runtime dependency.

use rapiddoc_core::document::Document;
use rapiddoc_core::{Error, ErrorCode, Limits, PageRange, Rect};
use serde::Serialize;
use std::path::Path;
// A literal pattern has one compiled instance per call site. Keep matching rules
// unchanged while avoiding compilation in per-line title/formula predicates.
macro_rules! cached_regex {
    ($pattern:literal) => {{
        static PATTERN: std::sync::LazyLock<regex::Regex> =
            std::sync::LazyLock::new(|| regex::Regex::new($pattern).unwrap());
        &*PATTERN
    }};
}
pub(crate) use cached_regex;
pub mod legacy_block_sort;
pub mod processing;
/// Compatibility path for the original migration API.
pub use processing as legacy_model;
pub mod operations;
pub mod page_assembly;
pub mod pipeline;
pub mod sdk;
pub use sdk::{
    Config as RapidDocConfig, Document as RapidDocDocument, ParseOptions, RapidDoc,
    Resources as RuntimeResources,
};
/// Compatibility path for the original migration API.
pub use page_assembly as legacy_page;
pub mod legacy_paragraphs;
pub mod legacy_post_ocr;
pub mod legacy_table_merge;
pub mod native_table;
pub mod ocr;
pub mod paragraphs;
pub mod reading_order;
pub use rapiddoc_native::text_boundary::merge as merge_text_lines;

/// 文字预览输出；布局、表格和图片导出仍需后续规则层。
#[derive(Debug, Serialize)]
pub struct TextOutput {
    pub text: String,
    pub document: Document,
}

/// 已知区域的表格恢复输出，保留单元格跨度、来源索引与安全 HTML。
#[derive(Debug, Serialize)]
pub struct TableOutput {
    pub page_index: usize,
    pub table: rapiddoc_native::grid::Table,
    pub html: String,
}

/// 复用同一事实/规则边界恢复明确指定区域；扫描页或不完整边界不会返回成功空表格。
pub fn recover_pdf_table(
    library: &Path,
    bytes: &[u8],
    password: Option<&str>,
    page_index: usize,
    region: Rect,
    limits: &Limits,
) -> Result<TableOutput, Error> {
    let end = page_index.checked_add(1).ok_or_else(|| Error {
        code: ErrorCode::InvalidInput,
        message: "页号溢出".into(),
    })?;
    let output = parse_pdf_text(
        library,
        bytes,
        password,
        &PageRange {
            start: page_index,
            end: Some(end),
        },
        limits,
        false,
    )?;
    let page = output.document.pages.first().ok_or_else(|| Error {
        code: ErrorCode::InvalidInput,
        message: "表格来源页为空".into(),
    })?;
    let separators: Vec<_> = page
        .drawing_lines
        .iter()
        .map(|line| {
            let vertical = line.orientation == "vertical";
            rapiddoc_native::grid::Separator {
                vertical,
                coordinate: if vertical {
                    line.start[0]
                } else {
                    line.start[1]
                },
                start: if vertical {
                    line.start[1]
                } else {
                    line.start[0]
                },
                end: if vertical { line.end[1] } else { line.end[0] },
            }
        })
        .collect();
    let glyphs = page
        .characters
        .iter()
        .filter_map(|character| {
            let text = character.text.as_deref().unwrap_or("\u{fffd}");
            if text.chars().all(char::is_control) {
                return None;
            }
            Some(rapiddoc_native::grid::Glyph {
                index: character.index,
                bounds: character.visual_loose_bounds,
                text: text.to_owned(),
                generated: character.is_generated,
            })
        })
        .collect();
    let config = rapiddoc_native::grid::GridConfig::default();
    let table = rapiddoc_native::grid::recover(region, &separators, glyphs, &config)?;
    let html = rapiddoc_render::table_html(&table, config.max_cells)?;
    Ok(TableOutput {
        page_index,
        table,
        html,
    })
}

/// 提取原生 PDF 文字；默认拒绝需要 OCR 的页，显式 allow_partial 才保留诊断继续。
pub fn parse_pdf_text(
    library: &Path,
    bytes: &[u8],
    password: Option<&str>,
    range: &PageRange,
    limits: &Limits,
    allow_partial: bool,
) -> Result<TextOutput, Error> {
    let mut document = rapiddoc_pdf::extract(library, bytes, password, range, limits)?;
    if !allow_partial
        && document
            .diagnostics
            .iter()
            .any(|item| item.code == "ocr_required")
    {
        return Err(Error { code: ErrorCode::ModelMissing, message: "文档包含需要 OCR 的页面；当前文字预览未接入 OCR，可显式 allow_partial=True 获取事实和诊断".into() });
    }
    for page in &mut document.pages {
        page.drawing_lines = rapiddoc_native::lines::extract(page)?;
    }
    let text = document
        .pages
        .iter()
        .map(|page| page.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    Ok(TextOutput { text, document })
}
