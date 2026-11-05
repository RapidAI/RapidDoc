//! Shared six-format dispatch and export, independent of any language binding.
use crate::{document_output as output, Format, Result};
use std::{collections::BTreeMap, sync::Arc};

#[derive(serde::Serialize)]
pub struct Document {
    pub middle_json: output::Middle,
    pub markdown: serde_json::Value,
    pub content_list_json: serde_json::Value,
    pub source_page_count: usize,
    pub diagnostics: Vec<String>,
    #[serde(skip)]
    pub assets: BTreeMap<String, Arc<[u8]>>,
}

/// Preserve the legacy format gates and empty-input policies of the public SDK.
pub fn parse_document(
    data: Arc<[u8]>,
    make_mode: &str,
    image_dir_name: &str,
    delimiters: &output::inline::Delimiters,
    docx_only: bool,
    empty_format: Option<&str>,
) -> Result<Document> {
    let mut diagnostics = vec![];
    let pages = if data.is_empty() {
        match empty_format {
            Some("pptx") => vec![],
            Some("doc" | "xls" | "ppt") => {
                return Err(crate::Error::new(
                    crate::ErrorKind::Malformed,
                    "empty legacy Office file",
                )
                .into())
            }
            None | Some("docx" | "xlsx") => vec![vec![]],
            Some(_) => {
                return Err(crate::Error::new(
                    crate::ErrorKind::Unsupported,
                    "unsupported empty Office format",
                )
                .into())
            }
        }
    } else {
        if data.starts_with(&[0xd0, 0xcf, 0x11, 0xe0]) {
            let mut ole = crate::ole::OleReader::open(data)?;
            let format = ole.format()?;
            if format == Format::Doc && !docx_only {
                let body = crate::doc::body::convert(&mut ole)?;
                diagnostics = body.diagnostics;
                body.pages
            } else {
                if format == Format::Ppt && !docx_only {
                    let body = crate::ppt::body::convert(&mut ole)?;
                    diagnostics = body.diagnostics;
                    body.pages
                } else {
                    if format != Format::Xls {
                        return Err(crate::Error::new(
                            crate::ErrorKind::Unsupported,
                            "The DOCX-only entry cannot parse this Office format",
                        )
                        .into());
                    }
                    let mut workbook = crate::xls::Workbook::from_ole(&mut ole)?;
                    diagnostics = std::mem::take(&mut workbook.diagnostics);
                    crate::spreadsheet::into_body_pages(workbook.into_pages()?)?
                }
            }
        } else {
            let mut package = crate::package::Package::open(data.clone())?;
            match package.format()? {
                Format::Docx => {
                    let body =
                        crate::docx::body::convert(crate::docx::Source::from_package(package)?)?;
                    diagnostics = body.diagnostics;
                    body.pages
                }
                Format::Xlsx if !docx_only => {
                    let result = crate::xlsx::convert_package(package, data, false, None, true)?;
                    diagnostics = result
                        .normalized_members
                        .into_iter()
                        .map(|p| format!("XLSX package normalized: {p}"))
                        .collect();
                    crate::spreadsheet::into_body_pages(result.pages)?
                }
                Format::Pptx if !docx_only => {
                    let body = crate::pptx::convert_package(package)?;
                    diagnostics = body.diagnostics;
                    body.pages
                }
                _ => {
                    return Err(crate::Error::new(
                        crate::ErrorKind::Unsupported,
                        "The DOCX-only entry cannot parse this Office format",
                    )
                    .into())
                }
            }
        }
    };
    let parsed = output::middle(pages, "0.9.11")?;
    let source_page_count = parsed.middle.pdf_info.len();
    Ok(Document {
        markdown: output::render::export(&parsed.middle, make_mode, image_dir_name, delimiters),
        content_list_json: output::render::export(
            &parsed.middle,
            "content_list",
            image_dir_name,
            delimiters,
        ),
        middle_json: parsed.middle,
        source_page_count,
        diagnostics,
        assets: parsed.assets,
    })
}
