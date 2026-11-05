//! tables adapters; algorithms remain in the shared Rust crates.
use super::*;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TableMatchInput {
    tokens: Vec<String>,
    cell_bboxes: Vec<Vec<f64>>,
    ocr: Vec<TableOcr>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TableOcr {
    bbox: Vec<f64>,
    text: String,
}

#[derive(serde::Serialize)]
pub(super) struct MatchedTable {
    matches: rapiddoc_inference::table_match::Matches,
    cell_texts: Vec<String>,
    html: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WiredMatchInput {
    polygons: Vec<rapiddoc_inference::table_recover::Polygon>,
    ocr: Vec<rapiddoc_inference::wired_match::Ocr>,
    logic_points: Vec<[usize; 4]>,
    #[serde(default)]
    blank_cells: Vec<rapiddoc_inference::blank_cell::ResultCell>,
    #[serde(default)]
    legacy_text: bool,
}

#[pyfunction]
pub(super) fn match_wired_table_json(py: Python<'_>, request: String) -> PyResult<String> {
    py.detach(move || {
        if request.len() > rapiddoc_core::resources::limit(8_000_000) {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "UNET OCR JSON 超过预算".into(),
            });
        }
        let input: WiredMatchInput =
            serde_json::from_str(&request).map_err(|e| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::InvalidInput,
                message: e.to_string(),
            })?;
        let mut matched =
            rapiddoc_inference::wired_match::match_cells(&input.polygons, &input.ocr)?;
        rapiddoc_inference::wired_match::fill_blank(
            &mut matched,
            &input.polygons,
            &input.blank_cells,
        )?;
        let render = if input.legacy_text {
            rapiddoc_render::wired::legacy_html
        } else {
            rapiddoc_render::wired::html
        };
        let html = render(&input.logic_points, &input.polygons, &matched.cell_lines)?;
        serialize(&serde_json::json!({"matches":matched,"html":html}))
    })
    .map_err(|error| python_error(py, error))
}

#[pyfunction]
pub(super) fn wired_matches_json(py: Python<'_>, request: String) -> PyResult<String> {
    py.detach(move || {
        if request.len() > rapiddoc_core::resources::limit(8_000_000) {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "UNET OCR JSON 超过预算".into(),
            });
        }
        let input: WiredMatchInput =
            serde_json::from_str(&request).map_err(|e| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::InvalidInput,
                message: e.to_string(),
            })?;
        serialize(&rapiddoc_inference::wired_match::match_cells(
            &input.polygons,
            &input.ocr,
        )?)
    })
    .map_err(|error| python_error(py, error))
}

/// 表格结构和 OCR 几何匹配，保留未匹配索引；HTML 只允许固定表格语法。
#[pyfunction]
pub(super) fn match_table_json(py: Python<'_>, request: String) -> PyResult<String> {
    py.detach(move || {
        let invalid = |message: String| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message,
        };
        if request.len() > rapiddoc_core::resources::limit(8_000_000) {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "表格匹配输入超过预算".into(),
            });
        }
        let input: TableMatchInput =
            serde_json::from_str(&request).map_err(|e| invalid(e.to_string()))?;
        if input.ocr.iter().map(|v| v.text.len()).sum::<usize>()
            > rapiddoc_core::resources::limit(1_000_000)
        {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "表格 OCR 文字超过预算".into(),
            });
        }
        let boxes: Vec<_> = input.ocr.iter().map(|v| v.bbox.clone()).collect();
        let matches = rapiddoc_inference::table_match::match_cells(&input.cell_bboxes, &boxes)?;
        let cell_texts: Vec<_> = matches
            .cells
            .iter()
            .map(|indices| {
                indices
                    .iter()
                    .map(|&i| input.ocr[i].text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        let html = rapiddoc_render::structure::html(&input.tokens, &cell_texts)?;
        serialize(&MatchedTable {
            matches,
            cell_texts,
            html,
        })
    })
    .map_err(|error| python_error(py, error))
}
