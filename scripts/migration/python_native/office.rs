//! One native Office conversion/export call; no Python Office package callback.
use super::*;
use rapiddoc_office::document_output as output;
use std::sync::Arc;
#[pyfunction]
#[pyo3(signature=(data,make_mode="mm_markdown",image_dir_name="images",delimiters=None))]
pub fn parse_docx_bytes(
    py: Python<'_>,
    data: Vec<u8>,
    make_mode: &str,
    image_dir_name: &str,
    delimiters: Option<&str>,
) -> PyResult<PageAssets> {
    parse(py, data, make_mode, image_dir_name, delimiters, false, None)
}
#[pyfunction]
#[pyo3(signature=(data,make_mode="mm_markdown",image_dir_name="images",delimiters=None,empty_format=None))]
pub fn parse_office_bytes(
    py: Python<'_>,
    data: Vec<u8>,
    make_mode: &str,
    image_dir_name: &str,
    delimiters: Option<&str>,
    empty_format: Option<&str>,
) -> PyResult<PageAssets> {
    parse(
        py,
        data,
        make_mode,
        image_dir_name,
        delimiters,
        true,
        empty_format,
    )
}
fn parse(
    py: Python<'_>,
    data: Vec<u8>,
    make_mode: &str,
    image_dir_name: &str,
    delimiters: Option<&str>,
    allow_other_office: bool,
    empty_format: Option<&str>,
) -> PyResult<PageAssets> {
    let d = delimiters
        .map(serde_json::from_str::<output::inline::Delimiters>)
        .transpose()
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .unwrap_or_default();
    let result = py
        .detach(move || -> Result<_, rapiddoc_core::Error> {
            let document = rapiddoc_office::convert::parse_document(
                Arc::from(data),
                make_mode,
                image_dir_name,
                &d,
                !allow_other_office,
                empty_format,
            )?;
            Ok((serialize(&document)?, document.assets))
        })
        .map_err(|e| python_error(py, e))?;
    let images = result
        .1
        .into_iter()
        .map(|(name, data)| (name, PyBytes::new(py, &data).unbind()))
        .collect();
    Ok((result.0, images))
}
