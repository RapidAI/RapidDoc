use pyo3::prelude::*;

#[pyclass]
pub struct ParseProgress(rapiddoc_core::progress::TerminalProgress);

#[pymethods]
impl ParseProgress {
    #[new]
    fn new(name: String, total: usize) -> Self {
        Self(rapiddoc_core::progress::TerminalProgress::new(name, total))
    }
    #[pyo3(signature=(stage, page=None,detail=None))]
    fn update(&mut self, stage: &str, page: Option<usize>, detail: Option<&str>) -> String {
        self.0.update_detail(stage, page, detail);
        rapiddoc_core::progress::stage_name(stage).to_owned()
    }
    fn clear(&mut self) {
        self.0.clear();
    }
    fn reset(&mut self) {
        self.0.reset();
    }
    #[pyo3(signature=(stage,completed,total,elapsed_seconds=None))]
    fn advance(
        &mut self,
        stage: &str,
        completed: usize,
        total: usize,
        elapsed_seconds: Option<f64>,
    ) -> PyResult<()> {
        let elapsed = elapsed_seconds
            .map(std::time::Duration::try_from_secs_f64)
            .transpose()
            .map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "elapsed_seconds must be finite and nonnegative",
                )
            })?;
        self.0
            .advance_with_elapsed(stage, completed, total, elapsed);
        Ok(())
    }
}
