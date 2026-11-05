//! Thin Python adapters over the shared parsing/export engine.
use super::*;
use std::sync::Mutex;
#[pyclass]
pub struct PdfEngine(Mutex<rapiddoc::pipeline::Engine>);
#[pymethods]
impl PdfEngine {
    #[new]
    fn new() -> Self {
        Self(Mutex::new(rapiddoc::pipeline::Engine::new(
            Default::default(),
        )))
    }
    /// Diagnostic compatibility entry: raw pages and JSON progress events.
    fn parse(
        &self,
        py: Python<'_>,
        library: String,
        data: Vec<u8>,
        request: String,
        resolver: Py<PyAny>,
        progress: Py<PyAny>,
    ) -> PyResult<PageAssets> {
        let (metadata, images, _) =
            self.run(py, library, data, request, resolver, progress, None)?;
        Ok((metadata, images))
    }
    /// Production entry: finish both exports before returning pages to Python.
    #[pyo3(signature=(library,data,request,resolver,progress,make_mode="mm_markdown",image_path_prefix="images",magika_runtime=None,draw_layout=false,draw_span=false))]
    fn parse_document(
        &self,
        py: Python<'_>,
        library: String,
        data: Vec<u8>,
        request: String,
        resolver: Py<PyAny>,
        progress: Py<PyAny>,
        make_mode: &str,
        image_path_prefix: &str,
        magika_runtime: Option<PathBuf>,
        draw_layout: bool,
        draw_span: bool,
    ) -> PyResult<(String, ImageAssets, ImageAssets)> {
        self.run(
            py,
            library,
            data,
            request,
            resolver,
            progress,
            Some((
                make_mode,
                image_path_prefix,
                magika_runtime.as_deref(),
                draw_layout,
                draw_span,
            )),
        )
    }
}
impl PdfEngine {
    fn run(
        &self,
        py: Python<'_>,
        library: String,
        data: Vec<u8>,
        request: String,
        resolver: Py<PyAny>,
        progress: Py<PyAny>,
        export: Option<(&str, &str, Option<&std::path::Path>, bool, bool)>,
    ) -> PyResult<(String, ImageAssets, ImageAssets)> {
        let resources = match std::env::var("RAPIDDOC_RESOURCE_LIMITS") {
            Ok(json) => serde_json::from_str::<rapiddoc::legacy_page::HostResources>(&json)
                .map_err(|e| {
                    PyValueError::new_err(format!("Invalid RAPIDDOC_RESOURCE_LIMITS: {e}"))
                })?,
            Err(std::env::VarError::NotPresent) => Default::default(),
            Err(e) => return Err(PyValueError::new_err(e.to_string())),
        };
        let request = serde_json::from_str::<rapiddoc::pipeline::Request>(&request)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let callback_error = Mutex::new(None::<PyErr>);
        let output = py.detach(|| {
            rapiddoc::legacy_page::with_host_resources(resources, || {
                let failure = |message: String| rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message,
                };
                let mut resolve = |role: &str| {
                    Python::attach(|py| {
                        let json = resolver
                            .bind(py)
                            .call1((role,))
                            .and_then(|v| v.extract::<String>())
                            .map_err(|e| {
                                let message =
                                    format!("Model resource resolution failed ({role}): {e}");
                                *callback_error.lock().unwrap() = Some(e);
                                failure(message)
                            })?;
                        serde_json::from_str(&json).map_err(|e| failure(e.to_string()))
                    })
                };
                let mut report = |event: rapiddoc::pipeline::Progress<'_>| {
                    Python::attach(|py| {
                        let result = (|| -> PyResult<_> {
                            if export.is_some() {
                                let fields = pyo3::types::PyDict::new(py);
                                fields.set_item("stage", event.stage)?;
                                fields.set_item("page_index", event.page_index)?;
                                fields.set_item("detail", event.detail)?;
                                if let Some(value) = event.elapsed_seconds {
                                    fields.set_item("elapsed_seconds", value)?;
                                }
                                if let Some(value) = event.completed {
                                    fields.set_item("completed", value)?;
                                }
                                if let Some(value) = event.total {
                                    fields.set_item("total", value)?;
                                }
                                progress.bind(py).call1((fields,))
                            } else {
                                let json = serde_json::to_string(&event)
                                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                                progress.bind(py).call1((json,))
                            }
                        })();
                        result.map(|_| ()).map_err(|e| {
                            let message = format!("Progress callback failed: {e}");
                            *callback_error.lock().unwrap() = Some(e);
                            failure(message)
                        })
                    })
                };
                let mut output = self
                    .0
                    .try_lock()
                    .map_err(|_| failure("PDF engine is already in use".into()))?
                    .parse(
                        &PathBuf::from(library),
                        &data,
                        &request,
                        &mut resolve,
                        &mut report,
                    )?;
                let mut drawings = vec![];
                if let Some((mode, prefix, runtime, layout, span)) = export {
                    output.render_document(mode, prefix, runtime)?;
                    let mut kinds = vec![];
                    if layout {
                        kinds.push(rapiddoc_pdf::drawing::Kind::Layout);
                    }
                    if span {
                        kinds.push(rapiddoc_pdf::drawing::Kind::Span);
                    }
                    if !kinds.is_empty() {
                        drawings = rapiddoc_pdf::drawing::draw(
                            &data,
                            output.metadata["pages"].as_array().unwrap(),
                            request.start_page,
                            request.end_page,
                            &kinds,
                        )?;
                    }
                }
                Ok((serialize(&output.metadata)?, output.assets, drawings))
            })
        });
        if let Some(error) = callback_error.into_inner().unwrap() {
            return Err(error);
        }
        let output = output.map_err(|e| python_error(py, e))?;
        Ok((
            output.0,
            output
                .1
                .into_iter()
                .map(|a| (a.name, PyBytes::new(py, &a.png).unbind()))
                .collect(),
            output
                .2
                .into_iter()
                .map(|(kind, data)| (kind.name().to_string(), PyBytes::new(py, &data).unbind()))
                .collect(),
        ))
    }
}
