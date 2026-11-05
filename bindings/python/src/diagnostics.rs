//! diagnostics adapters; algorithms remain in the shared Rust crates.
use super::*;

#[pyfunction]
pub(super) fn call_host_page(
    py: Python<'_>,
    callback: Py<PyAny>,
    args: Py<pyo3::types::PyTuple>,
    kwargs: Py<pyo3::types::PyDict>,
) -> PyResult<Py<PyAny>> {
    let resources = match std::env::var("RAPIDDOC_RESOURCE_LIMITS") {
        Ok(json) => serde_json::from_str::<rapiddoc::legacy_page::HostResources>(&json)
            .map_err(|e| PyValueError::new_err(format!("Invalid RAPIDDOC_RESOURCE_LIMITS: {e}")))?,
        Err(std::env::VarError::NotPresent) => Default::default(),
        Err(e) => return Err(PyValueError::new_err(e.to_string())),
    };
    rapiddoc::legacy_page::with_host_resources(resources, || {
        rapiddoc::legacy_page::with_host_page_budget(|| {
            callback
                .bind(py)
                .call(args.bind(py), Some(kwargs.bind(py)))
                .map(|value| value.unbind())
        })
    })
}

#[pyfunction]
pub(super) fn legacy_pipeline_json(
    py: Python<'_>,
    operation: &str,
    request: String,
) -> PyResult<String> {
    let operation = operation.to_owned();
    py.detach(move || {
        if request.len() > rapiddoc::legacy_page::host_json_bytes() {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "旧页面 JSON 超过预算".into(),
            });
        }
        let value: serde_json::Value =
            serde_json::from_str(&request).map_err(|e| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::InvalidInput,
                message: e.to_string(),
            })?;
        let result = rapiddoc::operations::run(&operation, value)?;
        serialize(&result)
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
pub(super) fn merge_text_lines(py: Python<'_>, lines: Vec<String>) -> PyResult<String> {
    py.detach(move || rapiddoc::merge_text_lines(&lines))
        .map_err(|e| python_error(py, e))
}

/// 检查矩形数值并返回独立坐标，不引用 Python 对象内存。
#[pyfunction]
pub(super) fn validate_rect(coordinates: [f64; 4]) -> PyResult<[f64; 4]> {
    Rect::try_from(coordinates)
        .map(Into::into)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// 检查从零开始、结束不包含的页范围；不静默截断非法输入。
#[pyfunction]
pub(super) fn resolve_page_range(
    start: usize,
    end: Option<usize>,
    page_count: usize,
) -> PyResult<(usize, usize)> {
    PageRange { start, end }
        .resolve(page_count)
        .map(|range| (range.start, range.end))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}
