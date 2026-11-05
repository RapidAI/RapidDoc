//! Source-only migration checkpoints. Enable migration-tools to compile.
use super::*;

#[pyfunction]
pub(super) fn prepare_layout_rgb_json(
    py: Python<'_>,
    data: Vec<u8>,
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
) -> PyResult<String> {
    py.detach(move || {
        serialize(&rapiddoc_inference::layout::prepare_rgb(
            &data,
            width,
            height,
            target_width,
            target_height,
            16_000_000,
        )?)
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
pub(super) fn layout_rectangles_json(
    py: Python<'_>,
    rows: Vec<[f32; 7]>,
    labels: Vec<String>,
    width: usize,
    height: usize,
    threshold: f32,
) -> PyResult<String> {
    py.detach(move || {
        serialize(&rapiddoc_inference::layout_post::rectangles(
            &rows, &labels, width, height, threshold,
        )?)
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
pub(super) fn prepare_orientation_rgb_json(
    py: Python<'_>,
    data: Vec<u8>,
    width: usize,
    height: usize,
) -> PyResult<String> {
    py.detach(move || {
        serialize(&rapiddoc_inference::orientation::prepare_rgb(
            &data, width, height, 16_000_000, 16_000_000,
        )?)
    })
    .map_err(|e| python_error(py, e))
}
