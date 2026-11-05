//! Thin adapter for the language-independent original-model store.
use pyo3::{
    exceptions::{PyFileNotFoundError, PyOSError, PyValueError},
    prelude::*,
};
use rapiddoc_inference::model_store::{self, Catalog, StoreError, StoreErrorKind};
use std::path::PathBuf;

fn python_error(e: StoreError) -> PyErr {
    match e.kind {
        StoreErrorKind::InvalidInput | StoreErrorKind::Integrity => {
            PyValueError::new_err(e.message)
        }
        StoreErrorKind::NotFound => PyFileNotFoundError::new_err(e.message),
        _ => PyOSError::new_err(e.message),
    }
}

#[pyfunction]
pub fn model_catalog_json() -> &'static str {
    model_store::CATALOG_JSON
}

#[pyfunction]
#[pyo3(signature = (directory=None))]
pub fn model_directory_path(directory: Option<PathBuf>) -> PyResult<PathBuf> {
    model_store::model_directory(directory.as_deref()).map_err(python_error)
}

#[pyfunction]
#[pyo3(signature = (model_id, directory=None, override_path=None, allow_download=true))]
pub fn resolve_model_json(
    py: Python<'_>,
    model_id: String,
    directory: Option<PathBuf>,
    override_path: Option<PathBuf>,
    allow_download: bool,
) -> PyResult<String> {
    py.detach(move || {
        let result = Catalog::bundled()?.resolve(
            &model_id,
            directory.as_deref(),
            override_path.as_deref(),
            allow_download,
        )?;
        serde_json::to_string(&result)
            .map_err(|e| StoreError::new(StoreErrorKind::Io, e.to_string()))
    })
    .map_err(python_error)
}
