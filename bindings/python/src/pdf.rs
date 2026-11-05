//! pdf adapters; algorithms remain in the shared Rust crates.
use super::*;
#[pyfunction]
#[pyo3(signature=(count,max_pages=10))]
pub(super) fn classification_indices_json(count: usize, max_pages: usize) -> String {
    serde_json::json!(rapiddoc_pdf::classification::rules::indices(
        count, max_pages
    ))
    .to_string()
}

#[pyfunction]
#[pyo3(signature=(library,data,start=0,end=None))]
pub(super) fn classify_pdf_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    start: usize,
    end: Option<usize>,
) -> PyResult<String> {
    py.detach(move || {
        serde_json::to_string(&rapiddoc_pdf::classification::classify(
            &PathBuf::from(library),
            &data,
            start,
            end,
        ))
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature=(library,data,page_index,sample_index=None))]
pub(super) fn classification_sample_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    sample_index: Option<usize>,
) -> PyResult<String> {
    py.detach(move || {
        let facts = rapiddoc_pdf::classification::page(
            &PathBuf::from(library),
            &data,
            page_index,
            None,
            &Limits::pdf_host(),
        )?;
        Ok::<_, rapiddoc_core::Error>(
            rapiddoc_pdf::classification::rules::sample(&facts, sample_index.unwrap_or(page_index))
                .to_string(),
        )
    })
    .map_err(|e| python_error(py, e))
}

/// 渲染完成后位图已归 Rust 所有；PNG 编码也在释放 GIL 的区域执行。
#[pyfunction]
#[pyo3(signature=(library,data,page_index=0,dpi=144.0,max_pixels=16000000,password=None,rgb=false))]
#[allow(clippy::too_many_arguments)]
pub(super) fn render_pdf_page_bytes(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    dpi: f64,
    max_pixels: usize,
    password: Option<String>,
    rgb: bool,
) -> PyResult<(u32, u32, Py<PyBytes>)> {
    let output = py
        .detach(move || {
            let render = if rgb {
                rapiddoc_pdf::render::render_page_rgb
            } else {
                rapiddoc_pdf::render::render_page
            };
            render(
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
    ))
}

/// 对明确指定的页和视觉 point 区域恢复物理表格，计算阶段释放 GIL。
#[pyfunction]
#[pyo3(signature=(library,data,bounds,page_index=0,password=None))]
pub(super) fn recover_pdf_table_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    bounds: [f64; 4],
    page_index: usize,
    password: Option<String>,
) -> PyResult<String> {
    let region = Rect::try_from(bounds).map_err(|error| python_error(py, error))?;
    py.detach(move || {
        let result = rapiddoc::recover_pdf_table(
            &PathBuf::from(library),
            &data,
            password.as_deref(),
            page_index,
            region,
            &rapiddoc::legacy_page::pdf_limits(),
        )?;
        serialize(&result)
    })
    .map_err(|error| python_error(py, error))
}

/// 复制输入后释放 GIL，整次 PDFium 操作在 Rust 锁内运行，返回自有 JSON。
#[pyfunction]
#[pyo3(signature = (library, data, start=0, end=None, password=None, allow_partial=false, max_input_bytes=268435456, max_pages=10000, max_chars_per_page=1000000, max_total_chars=10000000))]
#[allow(clippy::too_many_arguments)]
pub(super) fn parse_pdf_text_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    start: usize,
    end: Option<usize>,
    password: Option<String>,
    allow_partial: bool,
    max_input_bytes: usize,
    max_pages: usize,
    max_chars_per_page: usize,
    max_total_chars: usize,
) -> PyResult<String> {
    let limits = Limits {
        max_input_bytes,
        max_pages,
        max_chars_per_page,
        max_total_chars,
        ..Limits::pdf_host()
    };
    let result = py.detach(move || {
        let output = rapiddoc::parse_pdf_text(
            &PathBuf::from(library),
            &data,
            password.as_deref(),
            &PageRange { start, end },
            &limits,
            allow_partial,
        )?;
        serialize(&output)
    });
    result.map_err(|error| python_error(py, error))
}
