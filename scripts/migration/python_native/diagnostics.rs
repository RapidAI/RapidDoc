//! Source-only migration checkpoints. Enable migration-tools to compile.
use super::*;

#[pyfunction]
pub(super) fn deduplicate_native_characters_json(
    py: Python<'_>,
    request: String,
) -> PyResult<String> {
    py.detach(move || {
        if request.len() > rapiddoc::legacy_page::host_json_bytes() {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "字符去重JSON超过预算".into(),
            });
        }
        let chars = serde_json::from_str(&request).map_err(|_| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: "字符去重JSON无效".into(),
        })?;
        let result = rapiddoc_pdf::character_primitives::deduplicate(chars)?;
        serde_json::to_string(&result).map_err(|e| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::CorruptDocument,
            message: e.to_string(),
        })
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
pub(super) fn legacy_host_pipeline_json(
    py: Python<'_>,
    operation: &str,
    request: String,
) -> PyResult<String> {
    rapiddoc::legacy_page::with_host_page_budget(|| legacy_pipeline_json(py, operation, request))
}

#[pyfunction]
pub(super) fn sort_model_blocks_json(py: Python<'_>, request: String) -> PyResult<String> {
    py.detach(move || {
        if request.len() > rapiddoc::legacy_page::host_json_bytes() {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "旧排序 JSON 超过预算".into(),
            });
        }
        let value: serde_json::Value =
            serde_json::from_str(&request).map_err(|e| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::InvalidInput,
                message: e.to_string(),
            })?;
        serialize(&rapiddoc::legacy_block_sort::sort(&value)?)
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
pub(super) fn split_paragraphs_json(py: Python<'_>, request: String) -> PyResult<String> {
    py.detach(move || {
        if request.len() > rapiddoc::legacy_page::host_json_bytes() {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "旧段落 JSON 超过预算".into(),
            });
        }
        let pages: serde_json::Value =
            serde_json::from_str(&request).map_err(|e| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::InvalidInput,
                message: e.to_string(),
            })?;
        serialize(&rapiddoc::legacy_paragraphs::split(&pages)?)
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(request,split_columns=true,line_order="source",block_order="source",max_block_order_work=2_000_000))]
pub(super) fn organize_page_text_json(
    py: Python<'_>,
    request: String,
    split_columns: bool,
    line_order: &str,
    block_order: &str,
    max_block_order_work: usize,
) -> PyResult<String> {
    let order = match line_order {
        "source" => rapiddoc::paragraphs::LineOrder::Source,
        "geometry" => rapiddoc::paragraphs::LineOrder::Geometry,
        _ => {
            return Err(PyValueError::new_err(
                "line_order must be source or geometry",
            ))
        }
    };
    let block_order = match block_order {
        "source" => rapiddoc::paragraphs::LineOrder::Source,
        "geometry" => rapiddoc::paragraphs::LineOrder::Geometry,
        _ => {
            return Err(PyValueError::new_err(
                "block_order must be source or geometry",
            ))
        }
    };
    py.detach(move || {
        if request.len() > rapiddoc::legacy_page::host_json_bytes() {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "段落页面 JSON 超过预算".into(),
            });
        }
        let page: rapiddoc::paragraphs::Page =
            serde_json::from_str(&request).map_err(|e| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::InvalidInput,
                message: e.to_string(),
            })?;
        serialize(&rapiddoc::paragraphs::organize_with_order(
            &page,
            split_columns,
            order,
            block_order,
            max_block_order_work,
        )?)
    })
    .map_err(|e| python_error(py, e))
}
