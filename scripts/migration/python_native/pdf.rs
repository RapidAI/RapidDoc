//! Source-only migration checkpoints. Enable migration-tools to compile.
use super::*;

#[pyfunction]
#[pyo3(signature=(library,data,page_index=0,dpi=200.0,max_pixels=16000000,password=None))]
pub(super) fn render_pdf_model_page_bytes(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    dpi: f64,
    max_pixels: usize,
    password: Option<String>,
) -> PyResult<(u32, u32, Py<PyBytes>, f64)> {
    let output = py
        .detach(move || {
            rapiddoc_pdf::render::render_model_page(
                &PathBuf::from(library),
                &data,
                page_index,
                dpi,
                max_pixels,
                password.as_deref(),
                &rapiddoc::legacy_page::pdf_limits(),
            )
        })
        .map_err(|error| python_error(py, error))?;
    Ok((
        output.width,
        output.height,
        PyBytes::new(py, &output.png).unbind(),
        output.scale,
    ))
}

#[pyfunction]
pub(super) fn classify_pdf_font_name_bytes(face: Vec<u8>, charset: i32) -> PyResult<Option<i32>> {
    if face.len() > rapiddoc_core::resources::limit(4096) {
        return Err(PyValueError::new_err("font face exceeds 4096 bytes"));
    }
    Ok(rapiddoc_pdf::fonts::classify(&face, charset))
}

#[pyfunction]
pub(super) fn pdfium_font_runtime_info_json(py: Python<'_>, library: String) -> PyResult<String> {
    py.detach(move || {
        rapiddoc_pdf::font_runtime_info(&PathBuf::from(library)).map(|v| v.to_string())
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(library,data,page_index=0,password=None,max_depth=15,max_pixels=32000000,render=true,scale_to_original=true))]
#[allow(clippy::too_many_arguments)]
pub(super) fn extract_original_pdf_images_bytes(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    password: Option<String>,
    max_depth: usize,
    max_pixels: usize,
    render: bool,
    scale_to_original: bool,
) -> PyResult<PageAssets> {
    let output = py
        .detach(move || {
            rapiddoc_pdf::original_images::extract(
                &PathBuf::from(library),
                &data,
                page_index,
                password.as_deref(),
                &rapiddoc_pdf::original_images::Config {
                    max_depth,
                    max_pixels,
                    render,
                    scale_to_original,
                },
                &rapiddoc::legacy_page::pdf_limits(),
            )
        })
        .map_err(|e| python_error(py, e))?;
    let metadata = serde_json::json!({"images":output.images,"diagnostics":output.diagnostics});
    let images = output
        .images
        .into_iter()
        .map(|image| (image.uuid, PyBytes::new(py, &image.png).unbind()))
        .collect();
    Ok((metadata.to_string(), images))
}

#[pyfunction]
#[pyo3(signature=(library,data,page_index=0,password=None,extended=true))]
pub(super) fn read_pdf_native_text_geometry_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    password: Option<String>,
    extended: bool,
) -> PyResult<String> {
    py.detach(move || {
        let result = rapiddoc_pdf::character_primitives::read_geometry(
            &PathBuf::from(library),
            &data,
            page_index,
            password.as_deref(),
            extended,
            &rapiddoc::legacy_page::pdf_limits(),
        )?;
        let result = serde_json::to_string(&result).map_err(|e| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::CorruptDocument,
            message: e.to_string(),
        })?;
        if result.len() > rapiddoc_core::resources::limit(32_000_000) {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "PDF文字几何超过预算".into(),
            });
        }
        Ok(result)
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(library,data,page_index=0,password=None,extended=true))]
pub(super) fn read_pdf_character_primitives_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    password: Option<String>,
    extended: bool,
) -> PyResult<String> {
    py.detach(move || {
        let chars = rapiddoc_pdf::character_primitives::read(
            &PathBuf::from(library),
            &data,
            page_index,
            password.as_deref(),
            extended,
            &rapiddoc::legacy_page::pdf_limits(),
        )?;
        let result = serde_json::to_string(&chars).map_err(|e| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::CorruptDocument,
            message: e.to_string(),
        })?;
        if result.len() > rapiddoc_core::resources::limit(32_000_000) {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "PDF字符原语结果超过32MB预算".into(),
            });
        }
        Ok(result)
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(library,data,page_index=0,password=None))]
pub(super) fn read_pdf_link_annotations_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    password: Option<String>,
) -> PyResult<String> {
    py.detach(move || {
        let annotations = rapiddoc_pdf::link_annotations::read(
            &PathBuf::from(library),
            &data,
            page_index,
            password.as_deref(),
            &rapiddoc::legacy_page::pdf_limits(),
        )?;
        serde_json::to_string(&annotations).map_err(|e| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::CorruptDocument,
            message: e.to_string(),
        })
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(library,data,page_index=0,password=None))]
pub(super) fn read_pdf_image_infos_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    password: Option<String>,
) -> PyResult<String> {
    py.detach(move || {
        let images = rapiddoc_pdf::image_infos::read(
            &PathBuf::from(library),
            &data,
            page_index,
            password.as_deref(),
            &rapiddoc::legacy_page::pdf_limits(),
        )?;
        serde_json::to_string(&images).map_err(|e| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::CorruptDocument,
            message: e.to_string(),
        })
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(library,data,indices,page_index=0,password=None))]
pub(super) fn read_pdf_text_paint_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    indices: Vec<i64>,
    page_index: usize,
    password: Option<String>,
) -> PyResult<String> {
    py.detach(move || {
        let paint = rapiddoc_pdf::text_paint::read(
            &PathBuf::from(library),
            &data,
            page_index,
            &indices,
            password.as_deref(),
            &rapiddoc::legacy_page::pdf_limits(),
        )?;
        serde_json::to_string(&paint).map_err(|e| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::CorruptDocument,
            message: e.to_string(),
        })
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(library,data,page_index=0,password=None))]
pub(super) fn read_pdf_vector_geometry_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    password: Option<String>,
) -> PyResult<String> {
    py.detach(move || {
        let geometry = rapiddoc_pdf::legacy_vectors::read(
            &PathBuf::from(library),
            &data,
            page_index,
            password.as_deref(),
            &rapiddoc::legacy_page::pdf_limits(),
        )?;
        let result = serde_json::to_string(&geometry).map_err(|e| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::CorruptDocument,
            message: e.to_string(),
        })?;
        if result.len() > rapiddoc_core::resources::limit(32_000_000) {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "PDF绘图结果超过32MB预算".into(),
            });
        }
        Ok(result)
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(library,data,page_index=0,password=None))]
pub(super) fn read_pdf_form_geometry_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    password: Option<String>,
) -> PyResult<String> {
    py.detach(move || {
        let geometry = rapiddoc_pdf::form_structure::read(
            &PathBuf::from(library),
            &data,
            page_index,
            password.as_deref(),
            &rapiddoc::legacy_page::pdf_limits(),
        )?;
        let result = serde_json::to_string(&geometry).map_err(|e| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::CorruptDocument,
            message: e.to_string(),
        })?;
        if result.len() > rapiddoc_core::resources::limit(32_000_000) {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "PDF Form结果超过32MB预算".into(),
            });
        }
        Ok(result)
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(library,data,page_index=0,password=None,quote_loosebox=true))]
pub(super) fn read_native_pdf_page_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    password: Option<String>,
    quote_loosebox: bool,
) -> PyResult<String> {
    py.detach(move || {
        let page = rapiddoc_pdf::legacy_chars::read_page(
            &PathBuf::from(library),
            &data,
            page_index,
            password.as_deref(),
            quote_loosebox,
            &rapiddoc::legacy_page::pdf_limits(),
        )?;
        serde_json::to_string(&page).map_err(|e| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::CorruptDocument,
            message: e.to_string(),
        })
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(library,data,page_index=0,password=None))]
pub(super) fn read_pdf_visible_text_geometry_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    password: Option<String>,
) -> PyResult<String> {
    py.detach(move || {
        let out = rapiddoc_pdf::character_primitives::read_visible_geometry(
            PathBuf::from(library).as_path(),
            &data,
            page_index,
            password.as_deref(),
            true,
            &rapiddoc::legacy_page::pdf_limits(),
        )?;
        serde_json::to_string(&out).map_err(|_| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: "可见文字JSON无效".into(),
        })
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(library,data,page_index,password=None))]
pub(super) fn native_classification_page_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    password: Option<String>,
) -> PyResult<String> {
    py.detach(move || {
        serialize(&rapiddoc_pdf::classification::page(
            &PathBuf::from(library),
            &data,
            page_index,
            password.as_deref(),
            &rapiddoc::legacy_page::pdf_limits(),
        )?)
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(library,data,page_index,scale,password=None))]
pub(super) fn native_classification_difference_bytes(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    scale: f64,
    password: Option<String>,
) -> PyResult<Py<PyBytes>> {
    let png = py
        .detach(move || {
            rapiddoc_pdf::classification::difference(
                &PathBuf::from(library),
                &data,
                page_index,
                scale,
                password.as_deref(),
                &rapiddoc::legacy_page::pdf_limits(),
            )
        })
        .map_err(|e| python_error(py, e))?;
    Ok(PyBytes::new(py, &png).unbind())
}
