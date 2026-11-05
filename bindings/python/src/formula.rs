//! formula adapters; algorithms remain in the shared Rust crates.
use super::*;

#[pyfunction]
pub(super) fn repair_formula_latex(py: Python<'_>, text: String) -> PyResult<String> {
    py.detach(move || rapiddoc_inference::formula::repair(&text))
        .map_err(|e| python_error(py, e))
}
#[pyfunction]
pub(super) fn fix_formula_text(py: Python<'_>, text: String) -> PyResult<String> {
    py.detach(move || rapiddoc_inference::formula::post_process(&text))
        .map_err(|e| python_error(py, e))
}
