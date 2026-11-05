//! Host values/errors/progress only; configuration and model ownership are Rust.
use super::*;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::types::PyDict;
use rapiddoc::sdk::{Config, ParseOptions, RapidDoc, Resources};

fn config_error(error: rapiddoc_core::Error) -> PyErr {
    if error.code == rapiddoc_core::ErrorCode::Unsupported {
        PyNotImplementedError::new_err(error.message)
    } else {
        PyValueError::new_err(error.message)
    }
}
#[pyfunction]
pub fn model_plan_json(config: &str) -> PyResult<String> {
    let config: Config =
        serde_json::from_str(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
    serialize(&config.model_plan().map_err(config_error)?).map_err(config_error)
}
#[pyfunction]
pub fn model_optimization_json(
    role: &str,
    sha: &str,
    device: &str,
    enabled: bool,
) -> PyResult<String> {
    let plan = Config::default().model_plan().map_err(config_error)?;
    let mut model = plan
        .models
        .values()
        .find(|m| m.role == role)
        .ok_or_else(|| PyValueError::new_err("Unknown model role"))?
        .clone();
    model.device = device.into();
    model.optimize = enabled;
    serialize(&model.optimization(sha).map_err(config_error)?).map_err(config_error)
}
#[pyfunction]
pub fn locate_runtime(resources: &str, kind: &str) -> PyResult<PathBuf> {
    let resources: Resources =
        serde_json::from_str(resources).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let kind = match kind {
        "ort" => rapiddoc::sdk::Resource::Ort,
        "geos" => rapiddoc::sdk::Resource::Geos,
        "operators" => rapiddoc::sdk::Resource::Operators,
        "pdfium" => rapiddoc::sdk::Resource::Pdfium,
        _ => return Err(PyValueError::new_err("Unknown runtime kind")),
    };
    resources
        .locate(kind)
        .map_err(|e| pyo3::exceptions::PyFileNotFoundError::new_err(e.message))
}
#[pyclass]
pub struct DocumentEngine {
    engine: Mutex<Option<std::sync::Arc<RapidDoc>>>,
    active: std::sync::atomic::AtomicUsize,
}
struct ActiveRequest<'a>(&'a std::sync::atomic::AtomicUsize);
impl Drop for ActiveRequest<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::Release);
    }
}
fn event_dict<'py>(
    py: Python<'py>,
    event: rapiddoc::sdk::Progress<'_>,
) -> PyResult<Bound<'py, PyDict>> {
    let fields = PyDict::new(py);
    fields.set_item("stage", event.stage)?;
    fields.set_item("detail", event.detail)?;
    fields.set_item("page_index", event.page_index)?;
    if let Some(value) = event.total {
        fields.set_item("total", value)?;
    }
    if let Some(value) = event.completed {
        fields.set_item("completed", value)?;
    }
    if let Some(value) = event.elapsed_seconds {
        fields.set_item("elapsed_seconds", value)?;
    }
    Ok(fields)
}
#[pymethods]
impl DocumentEngine {
    #[new]
    fn new(config: &str, resources: &str) -> PyResult<Self> {
        let config =
            serde_json::from_str(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let resources =
            serde_json::from_str(resources).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            engine: Mutex::new(Some(std::sync::Arc::new(RapidDoc::new(config, resources)))),
            active: std::sync::atomic::AtomicUsize::new(0),
        })
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let mut state = self
            .engine
            .lock()
            .map_err(|_| PyValueError::new_err("Document engine lock poisoned"))?;
        if self.active.load(std::sync::atomic::Ordering::Acquire) != 0 {
            return Err(PyValueError::new_err("Document engine is already in use"));
        }
        let engine = state.take();
        drop(state);
        py.detach(move || drop(engine));
        Ok(())
    }
    fn set_config(&self, config: &str) -> PyResult<()> {
        let config =
            serde_json::from_str(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.update_engine(|engine| engine.configure(config))
            .map_err(config_error)
    }
    fn set_resources(&self, resources: &str) -> PyResult<()> {
        let resources =
            serde_json::from_str(resources).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.update_engine(|engine| {
            engine.set_resources(resources);
            Ok(())
        })
        .map_err(config_error)
    }
    fn loaded_model_roles(&self) -> PyResult<Vec<String>> {
        self.with_engine(|engine| Ok(engine.loaded_model_roles()))
            .map_err(config_error)
    }
    fn warmup(
        &self,
        py: Python<'_>,
        options: &str,
        progress: Py<PyAny>,
        device: Py<PyAny>,
    ) -> PyResult<Vec<String>> {
        match self.run(py, None, options, progress, device)? {
            RunOutput::Warmup(roles) => Ok(roles),
            _ => unreachable!(),
        }
    }
    fn parse(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        options: &str,
        progress: Py<PyAny>,
        device: Py<PyAny>,
    ) -> PyResult<(String, ImageAssets, ImageAssets)> {
        let RunOutput::Document(metadata, images, drawings) =
            self.run(py, Some(data), options, progress, device)?
        else {
            unreachable!()
        };
        Ok((
            metadata,
            images
                .into_iter()
                .map(|(name, bytes)| (name, PyBytes::new(py, &bytes).unbind()))
                .collect(),
            drawings
                .into_iter()
                .map(|(name, bytes)| (name, PyBytes::new(py, &bytes).unbind()))
                .collect(),
        ))
    }
}
enum RunOutput {
    Warmup(Vec<String>),
    Document(
        String,
        std::collections::BTreeMap<String, rapiddoc::sdk::Image>,
        std::collections::BTreeMap<String, Vec<u8>>,
    ),
}
impl DocumentEngine {
    fn update_engine<T>(
        &self,
        operation: impl FnOnce(&mut RapidDoc) -> Result<T, rapiddoc_core::Error>,
    ) -> Result<T, rapiddoc_core::Error> {
        let mut state = self.engine.lock().map_err(|_| config_lock_error())?;
        let engine = state.as_mut().ok_or_else(closed_error)?;
        operation(std::sync::Arc::make_mut(engine))
    }
    fn with_engine<T>(
        &self,
        operation: impl FnOnce(&RapidDoc) -> Result<T, rapiddoc_core::Error>,
    ) -> Result<T, rapiddoc_core::Error> {
        let engine = {
            let state = self.engine.lock().map_err(|_| config_lock_error())?;
            let engine = state.as_ref().cloned().ok_or_else(closed_error)?;
            self.active
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            engine
        };
        let _active = ActiveRequest(&self.active);
        operation(&engine)
    }

    fn run(
        &self,
        py: Python<'_>,
        data: Option<Vec<u8>>,
        options: &str,
        progress: Py<PyAny>,
        device: Py<PyAny>,
    ) -> PyResult<RunOutput> {
        let options: ParseOptions =
            serde_json::from_str(options).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let limits = py
            .import("os")?
            .call_method1("getenv", ("RAPIDDOC_RESOURCE_LIMITS",))?
            .extract::<Option<String>>()?;
        let resources = limits
            .map(|json| serde_json::from_str::<rapiddoc::legacy_page::HostResources>(&json))
            .transpose()
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .unwrap_or_default();
        let callback_error = Mutex::new(None::<PyErr>);
        let call_error = |error: PyErr| {
            let message = error.to_string();
            *callback_error.lock().unwrap() = Some(error);
            rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::InvalidInput,
                message,
            }
        };
        let output = py.detach(|| {
            rapiddoc::legacy_page::with_host_resources(resources, || {
                let mut report = |event: rapiddoc::sdk::Progress<'_>| {
                    Python::attach(|py| {
                        let fields = event_dict(py, event).map_err(&call_error)?;
                        progress
                            .bind(py)
                            .call1((fields,))
                            .map(|_| ())
                            .map_err(&call_error)
                    })
                };
                let mut prepare = |name: &str| {
                    Python::attach(|py| {
                        device
                            .bind(py)
                            .call1((name,))
                            .map(|_| ())
                            .map_err(&call_error)
                    })
                };
                self.with_engine(|engine| {
                    if let Some(data) = data {
                        let document =
                            engine.parse_with_host(&data, &options, &mut report, &mut prepare)?;
                        Ok(RunOutput::Document(
                            serialize(&document)?,
                            document.images,
                            document.drawings,
                        ))
                    } else {
                        Ok(RunOutput::Warmup(engine.warmup_with_host(
                            &options,
                            &mut report,
                            &mut prepare,
                        )?))
                    }
                })
            })
        });
        if let Some(error) = callback_error.into_inner().unwrap() {
            return Err(error);
        }
        output.map_err(|e| python_error(py, e))
    }
}

fn config_lock_error() -> rapiddoc_core::Error {
    rapiddoc_core::Error {
        code: rapiddoc_core::ErrorCode::InvalidInput,
        message: "Document engine lock poisoned".into(),
    }
}
fn closed_error() -> rapiddoc_core::Error {
    rapiddoc_core::Error {
        code: rapiddoc_core::ErrorCode::InvalidInput,
        message: "Document engine is closed".into(),
    }
}

/// Network IO is implemented once in the shared SDK and runs without the GIL.
#[pyfunction]
pub fn read_document_url(py: Python<'_>, source: String) -> PyResult<(String, Py<PyBytes>)> {
    let (name, data) = py
        .detach(|| rapiddoc::sdk::input::read_url(&source, &mut |_| Ok(())))
        .map_err(|error| python_error(py, error))?;
    Ok((name, PyBytes::new(py, &data).unbind()))
}
