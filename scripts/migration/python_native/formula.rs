//! Source-only migration checkpoints. Enable migration-tools to compile.
use super::*;

#[pyclass(name = "FormulaDecoder")]
pub(super) struct FormulaDecoder {
    decoder: rapiddoc_inference::formula::Decoder,
}
#[pymethods]
impl FormulaDecoder {
    #[new]
    fn new(py: Python<'_>, metadata: String) -> PyResult<Self> {
        py.detach(move || {
            rapiddoc_inference::formula::Decoder::new(&metadata).map(|decoder| Self { decoder })
        })
        .map_err(|e| python_error(py, e))
    }
    fn decode(&self, py: Python<'_>, tokens: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        py.detach(move || self.decoder.decode(&tokens))
            .map_err(|e| python_error(py, e))
    }
}

#[pyfunction]
pub(super) fn prepare_formula_input_bytes(
    py: Python<'_>,
    png: Vec<u8>,
    height: usize,
    width: usize,
) -> PyResult<FormulaPrepared> {
    let p = py
        .detach(move || rapiddoc_inference::formula_preprocess::from_png(&png, height, width))
        .map_err(|e| python_error(py, e))?;
    let metadata = serde_json::json!({"shape":p.shape,"crop_box":p.crop_box,"resized_size":p.resized_size,"rgb_size":[width,height],"dtype":"float32-le","layout":"NCHW"}).to_string();
    let tensor: Vec<_> = p.values.into_iter().flat_map(f32::to_le_bytes).collect();
    Ok((
        metadata,
        PyBytes::new(py, &p.rgb).unbind(),
        PyBytes::new(py, &tensor).unbind(),
    ))
}

#[pyfunction]
pub(super) fn validate_formula_png(py: Python<'_>, png: Vec<u8>) -> PyResult<usize> {
    py.detach(move || rapiddoc::legacy_model::formulas::validate_png(&png))
        .map_err(|e| python_error(py, e))
}
