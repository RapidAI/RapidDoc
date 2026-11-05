//! Python 3.10+ extension registration; implementation lives in domain adapters.

// Only Rust-owned allocations use this allocator. No CRT malloc override is
// enabled; PDFium, ONNX Runtime and Python retain their own allocation contracts.
#[cfg(feature = "mimalloc-allocator")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rapiddoc_core::{Limits, PageRange, Rect};
use std::path::PathBuf;
use std::sync::Mutex;
mod drawing;
mod pdf;
mod progress;
mod sdk;
use pdf::*;
mod tables;
use tables::*;
mod ocr;
use ocr::*;
mod formula;
use formula::*;
mod model_store;
mod models;
use models::*;
mod diagnostics;
use diagnostics::*;

#[cfg(feature = "migration-tools")]
#[path = "../../../scripts/migration/python_native/pdf.rs"]
mod migration_pdf;
#[cfg(feature = "migration-tools")]
use migration_pdf::*;
#[cfg(feature = "migration-tools")]
#[path = "../../../scripts/migration/python_native/diagnostics.rs"]
mod migration_diagnostics;
#[cfg(feature = "migration-tools")]
use migration_diagnostics::*;
#[cfg(feature = "migration-tools")]
#[path = "../../../scripts/migration/python_native/formula.rs"]
mod migration_formula;
#[cfg(feature = "migration-tools")]
use migration_formula::*;
#[cfg(feature = "migration-tools")]
#[path = "../../../scripts/migration/python_native/models.rs"]
mod migration_models;
#[cfg(feature = "migration-tools")]
use migration_models::*;
#[cfg(feature = "migration-tools")]
#[path = "../../../scripts/migration/python_native/ocr.rs"]
mod migration_ocr;
#[cfg(feature = "migration-tools")]
use migration_ocr::*;
#[cfg(feature = "migration-tools")]
#[path = "../../../scripts/migration/python_native/tables.rs"]
mod migration_tables;
#[cfg(feature = "migration-tools")]
use migration_tables::*;
#[cfg(feature = "migration-tools")]
#[path = "../../../scripts/migration/python_native/assets.rs"]
mod assets;
#[cfg(feature = "migration-tools")]
use assets::*;
#[cfg(feature = "migration-tools")]
#[path = "../../../scripts/migration/python_native/office.rs"]
mod office;
#[cfg(feature = "migration-tools")]
#[path = "../../../scripts/migration/python_native/pipeline.rs"]
mod pipeline;

pyo3::create_exception!(_native, RapidDocError, PyException);

#[cfg(feature = "migration-tools")]
type PageAssets = (String, Vec<(String, Py<PyBytes>)>);
type ImageAssets = Vec<(String, Py<PyBytes>)>;
#[cfg(feature = "migration-tools")]
type PageAssetBatch = (String, ImageAssets, usize, ImageAssets);
#[cfg(feature = "migration-tools")]
type TableImageInput = (String, [f64; 4], Vec<u8>);
#[cfg(feature = "migration-tools")]
type FormulaPrepared = (String, Py<PyBytes>, Py<PyBytes>);

/// 将稳定错误码附加到 Python 异常，不吞掉属性设置失败。
fn python_error(py: Python<'_>, error: rapiddoc_core::Error) -> PyErr {
    let exception = RapidDocError::new_err(error.message);
    match exception.value(py).setattr("code", error.code.as_str()) {
        Ok(()) => exception,
        Err(attribute_error) => attribute_error,
    }
}

/// 在释放 GIL 的计算区内序列化，避免大文档 JSON 编码长时间占用 Python 锁。
fn serialize<T: serde::Serialize>(value: &T) -> Result<String, rapiddoc_core::Error> {
    serde_json::to_string(value).map_err(|error| rapiddoc_core::Error {
        code: rapiddoc_core::ErrorCode::InvalidInput,
        message: error.to_string(),
    })
}

/// 注册已实现的基础接口，保持模块名与发行包名分离。
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<sdk::DocumentEngine>()?;
    module.add_function(wrap_pyfunction!(sdk::model_plan_json, module)?)?;
    module.add_function(wrap_pyfunction!(sdk::model_optimization_json, module)?)?;
    module.add_function(wrap_pyfunction!(sdk::locate_runtime, module)?)?;
    module.add_function(wrap_pyfunction!(sdk::read_document_url, module)?)?;
    module.add_function(wrap_pyfunction!(drawing::draw_pdf_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(model_store::model_catalog_json, module)?)?;
    module.add_function(wrap_pyfunction!(model_store::model_directory_path, module)?)?;
    module.add_function(wrap_pyfunction!(model_store::resolve_model_json, module)?)?;
    module.add_function(wrap_pyfunction!(classification_indices_json, module)?)?;
    module.add_function(wrap_pyfunction!(classify_pdf_json, module)?)?;
    module.add_function(wrap_pyfunction!(classification_sample_json, module)?)?;
    module.add_class::<progress::ParseProgress>()?;
    module.add_function(wrap_pyfunction!(merge_text_lines, module)?)?;
    module.add_function(wrap_pyfunction!(legacy_pipeline_json, module)?)?;
    module.add_function(wrap_pyfunction!(call_host_page, module)?)?;
    module.add_function(wrap_pyfunction!(validate_rect, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_page_range, module)?)?;
    module.add_function(wrap_pyfunction!(parse_pdf_text_json, module)?)?;
    module.add_function(wrap_pyfunction!(recover_pdf_table_json, module)?)?;
    module.add_function(wrap_pyfunction!(render_pdf_page_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(repair_formula_latex, module)?)?;
    module.add_function(wrap_pyfunction!(fix_formula_text, module)?)?;
    module.add_function(wrap_pyfunction!(ocr_png_json, module)?)?;
    module.add_function(wrap_pyfunction!(match_table_json, module)?)?;
    module.add_function(wrap_pyfunction!(match_wired_table_json, module)?)?;
    module.add_function(wrap_pyfunction!(wired_matches_json, module)?)?;
    #[cfg(feature = "migration-tools")]
    register_migration(module)?;
    module.add("MIGRATION_TOOLS", cfg!(feature = "migration-tools"))?;
    let exception_type = module.py().get_type::<RapidDocError>();
    exception_type.setattr("__module__", "rapid_doc_rust._native")?;
    module.add("RapidDocError", exception_type)?;
    module.add_class::<OnnxSession>()?;
    module.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(feature = "migration-tools")]
include!("../../../scripts/migration/python_native/register.rs");
