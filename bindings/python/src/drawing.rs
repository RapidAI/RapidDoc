//! Optional debug API. Production draws before returning page facts to Python.
use super::*;

#[pyfunction]
#[pyo3(signature=(data,pages,kind,start=0,end=None))]
pub fn draw_pdf_bytes(
    py: Python<'_>,
    data: Vec<u8>,
    pages: String,
    kind: &str,
    start: usize,
    end: Option<usize>,
) -> PyResult<Py<PyBytes>> {
    let kind = match kind {
        "layout" => rapiddoc_pdf::drawing::Kind::Layout,
        "span" => rapiddoc_pdf::drawing::Kind::Span,
        "line_sort" => rapiddoc_pdf::drawing::Kind::LineSort,
        _ => {
            return Err(PyValueError::new_err(
                "kind must be layout, span or line_sort",
            ))
        }
    };
    let result = py
        .detach(|| {
            let pages: Vec<serde_json::Value> =
                serde_json::from_str(&pages).map_err(|e| rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message: e.to_string(),
                })?;
            rapiddoc_pdf::drawing::draw(&data, &pages, start, end, &[kind])
        })
        .map_err(|e| python_error(py, e))?;
    Ok(PyBytes::new(py, &result[0].1).unbind())
}
