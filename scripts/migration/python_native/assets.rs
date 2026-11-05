//! assets adapters; algorithms remain in the shared Rust crates.
use super::*;

#[pyfunction]
#[pyo3(signature=(request,page_png,originals,extract_original=false,threshold=0.9,inline_prefix="images/",table_images=None,host_page=false))]
#[allow(clippy::too_many_arguments)]
pub(super) fn assemble_model_page_assets_bytes(
    py: Python<'_>,
    request: String,
    page_png: Vec<u8>,
    originals: Vec<([f64; 4], Vec<u8>)>,
    extract_original: bool,
    threshold: f64,
    inline_prefix: &str,
    table_images: Option<Vec<TableImageInput>>,
    host_page: bool,
) -> PyResult<PageAssetBatch> {
    let inline_prefix = inline_prefix.to_owned();
    let output = py
        .detach(move || {
            if request.len() > rapiddoc::legacy_page::host_json_bytes() {
                return Err(rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::ResourceLimit,
                    message: "资产页面 JSON 超过预算".into(),
                });
            }
            let value: serde_json::Value =
                serde_json::from_str(&request).map_err(|e| rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message: e.to_string(),
                })?;
            let originals: Vec<_> = originals
                .into_iter()
                .map(|(bbox, png)| rapiddoc::legacy_page::assets::OriginalImage { bbox, png })
                .collect();
            let fills: Vec<_> = table_images
                .unwrap_or_default()
                .into_iter()
                .map(
                    |(uuid, bbox, png)| rapiddoc::legacy_page::assets::TableFillImage {
                        uuid,
                        bbox,
                        png,
                    },
                )
                .collect();
            let run = || {
                rapiddoc::legacy_page::assets::assemble_with_table_images(
                    &value,
                    &page_png,
                    &originals,
                    extract_original,
                    threshold,
                    &inline_prefix,
                    &fills,
                )
            };
            if host_page {
                rapiddoc::legacy_page::with_host_page_budget(run)
            } else {
                run()
            }
        })
        .map_err(|e| python_error(py, e))?;
    let page = serialize(&output.page).map_err(|e| python_error(py, e))?;
    let assets = output
        .assets
        .into_iter()
        .map(|a| (a.name, PyBytes::new(py, &a.png).unbind()))
        .collect();
    let crops = output
        .ocr_images
        .into_iter()
        .map(|a| (a.name, PyBytes::new(py, &a.png).unbind()))
        .collect();
    Ok((page, assets, output.inline_count, crops))
}

#[pyfunction]
pub(super) fn fill_native_text_bytes(
    py: Python<'_>,
    request: String,
    page_png: Vec<u8>,
) -> PyResult<PageAssets> {
    let output = py
        .detach(move || {
            if request.len() > rapiddoc::legacy_page::host_json_bytes() {
                return Err(rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::ResourceLimit,
                    message: "原生字符 JSON 超过预算".into(),
                });
            }
            let value: serde_json::Value =
                serde_json::from_str(&request).map_err(|e| rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message: e.to_string(),
                })?;
            rapiddoc::legacy_page::assets::fill_native_text(&value, &page_png)
        })
        .map_err(|e| python_error(py, e))?;
    let spans = serialize(&output.spans).map_err(|e| python_error(py, e))?;
    Ok((
        spans,
        output
            .images
            .into_iter()
            .map(|a| (a.name, PyBytes::new(py, &a.png).unbind()))
            .collect(),
    ))
}

#[pyfunction]
pub(super) fn replace_inline_image_assets(
    py: Python<'_>,
    markup: String,
    prefix: String,
) -> PyResult<PageAssets> {
    let (html, assets) = py
        .detach(move || rapiddoc::legacy_page::inline::replace(&markup, &prefix))
        .map_err(|e| python_error(py, e))?;
    Ok((
        html,
        assets
            .into_iter()
            .map(|a| (a.name, PyBytes::new(py, &a.png).unbind()))
            .collect(),
    ))
}

#[pyfunction]
#[pyo3(signature=(request,page_png=None))]
pub(super) fn materialize_native_table_media_page_json(
    py: Python<'_>,
    request: String,
    page_png: Option<Vec<u8>>,
) -> PyResult<String> {
    py.detach(move || {
        if request.len() > rapiddoc::legacy_page::host_json_bytes() {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "表内媒体JSON超过预算".into(),
            });
        }
        let v = serde_json::from_str(&request).map_err(|_| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: "表内媒体JSON无效".into(),
        })?;
        let out = rapiddoc::legacy_model::table_block::images::page(&v, page_png.as_deref())?;
        serde_json::to_string(&out).map_err(|e| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: e.to_string(),
        })
    })
    .map_err(|e| python_error(py, e))
}
#[pyfunction]
#[pyo3(signature=(request,page_png=None))]
pub(super) fn native_glyph_weight_json(
    py: Python<'_>,
    request: String,
    page_png: Option<Vec<u8>>,
) -> PyResult<String> {
    py.detach(move || {
        if request.len() > rapiddoc_core::resources::limit(32_000_000) {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "字形证据超过预算".into(),
            });
        }
        let v = serde_json::from_str(&request).map_err(|_| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: "字形证据JSON无效".into(),
        })?;
        let out = rapiddoc::legacy_model::native_glyph_weight::detect(&v, page_png.as_deref())?;
        serde_json::to_string(&out).map_err(|_| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: "字形输出JSON无效".into(),
        })
    })
    .map_err(|e| python_error(py, e))
}
#[pyfunction]
#[pyo3(signature=(library,data,password=None,collect_only=false,max_pixels=16000000))]
pub(super) fn native_pdf_document_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    password: Option<String>,
    collect_only: bool,
    max_pixels: usize,
) -> PyResult<String> {
    py.detach(move || {
        let path = PathBuf::from(library);
        let out = if collect_only {
            rapiddoc::legacy_model::native_pdf_document::collect(
                &path,
                &data,
                password.as_deref(),
                max_pixels,
            )?
        } else {
            rapiddoc::legacy_model::native_pdf_document::predict(
                &path,
                &data,
                password.as_deref(),
                max_pixels,
            )?
        };
        serde_json::to_string(&out).map_err(|_| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: "原生PDF输出JSON无效".into(),
        })
    })
    .map_err(|e| python_error(py, e))
}
#[pyfunction]
#[pyo3(signature=(library,data,page_index=0,password=None))]
pub(super) fn native_pdf_table_page_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    page_index: usize,
    password: Option<String>,
) -> PyResult<String> {
    py.detach(move || {
        let out = rapiddoc::legacy_model::native_adapter::prepare_table_page(
            PathBuf::from(library).as_path(),
            &data,
            page_index,
            password.as_deref(),
        )?;
        serde_json::to_string(&out).map_err(|_| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: "原生表格页JSON无效".into(),
        })
    })
    .map_err(|e| python_error(py, e))
}
#[pyfunction]
#[pyo3(signature=(request,library,data,password=None,max_pixels=16000000))]
pub(super) fn materialize_native_table_media_pdf_json(
    py: Python<'_>,
    request: String,
    library: String,
    data: Vec<u8>,
    password: Option<String>,
    max_pixels: usize,
) -> PyResult<String> {
    py.detach(move || {
        if request.len() > rapiddoc::legacy_page::host_json_bytes() {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "表内媒体JSON超过预算".into(),
            });
        }
        let v = serde_json::from_str(&request).map_err(|_| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: "表内媒体JSON无效".into(),
        })?;
        let out = rapiddoc::legacy_model::table_block::images::pdf(
            &v,
            PathBuf::from(library).as_path(),
            &data,
            password.as_deref(),
            max_pixels,
        )?;
        serde_json::to_string(&out).map_err(|e| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: e.to_string(),
        })
    })
    .map_err(|e| python_error(py, e))
}

#[pyclass(name = "_NativeTablePage")]
pub(super) struct NativeTablePage {
    page: rapiddoc::native_table::PageTables,
}
#[pymethods]
impl NativeTablePage {
    #[new]
    fn new(py: Python<'_>, facts: String, width: usize, height: usize) -> PyResult<Self> {
        py.detach(move || {
            if facts.len() > rapiddoc_core::resources::limit(32_000_000) {
                return Err(rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::ResourceLimit,
                    message: "原生表格 JSON 超过预算".into(),
                });
            }
            let page = serde_json::from_str(&facts).map_err(|e: serde_json::Error| {
                rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message: e.to_string(),
                }
            })?;
            rapiddoc::native_table::PageTables::new(page, width, height).map(|page| Self { page })
        })
        .map_err(|e| python_error(py, e))
    }
    #[pyo3(signature=(proposals,margin_pt=6.0,max_work=10000000))]
    fn plan_json(
        &self,
        py: Python<'_>,
        proposals: String,
        margin_pt: f64,
        max_work: usize,
    ) -> PyResult<String> {
        py.detach(move || {
            if proposals.len() > rapiddoc_core::resources::limit(2_000_000) {
                return Err(rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::ResourceLimit,
                    message: "原生表格候选 JSON 超过预算".into(),
                });
            }
            let proposals: Vec<rapiddoc::native_table::Proposal> = serde_json::from_str(&proposals)
                .map_err(|e| rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message: e.to_string(),
                })?;
            serialize(&self.page.plan(&proposals, margin_pt, max_work)?)
        })
        .map_err(|e| python_error(py, e))
    }
}

#[pyclass(name = "_NativeTextPage")]
pub(super) struct NativeTextPage {
    page: rapiddoc_inference::native_text::NativePage,
}
#[pymethods]
impl NativeTextPage {
    #[new]
    fn new(py: Python<'_>, facts: String, width: usize, height: usize) -> PyResult<Self> {
        py.detach(move || {
            rapiddoc_inference::native_text::NativePage::from_json(&facts, width, height)
                .map(|page| Self { page })
        })
        .map_err(|e| python_error(py, e))
    }
    #[pyo3(signature=(lines,max_work=10000000,origin=(0,0)))]
    fn match_json(
        &self,
        py: Python<'_>,
        lines: String,
        max_work: usize,
        origin: (usize, usize),
    ) -> PyResult<String> {
        py.detach(move || {
            if lines.len() > rapiddoc_core::resources::limit(2_000_000) {
                return Err(rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::ResourceLimit,
                    message: "检测行 JSON 超过预算".into(),
                });
            }
            let mut lines: Vec<rapiddoc_inference::native_text::Line> =
                serde_json::from_str(&lines).map_err(|e| rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message: e.to_string(),
                })?;
            for point in lines.iter_mut().flat_map(|line| &mut line.points) {
                point[0] += origin.0 as f64;
                point[1] += origin.1 as f64;
            }
            serialize(&self.page.match_lines(&lines, max_work)?)
        })
        .map_err(|e| python_error(py, e))
    }
}

#[pyclass(name = "_PageImage")]
pub(super) struct PageImage {
    image: rapiddoc_inference::page_tables::PageImage,
}
#[pymethods]
impl PageImage {
    #[new]
    #[pyo3(signature=(data,max_pixels=16000000))]
    fn new(py: Python<'_>, data: Vec<u8>, max_pixels: usize) -> PyResult<Self> {
        py.detach(move || {
            rapiddoc_inference::page_tables::PageImage::decode(&data, max_pixels)
                .map(|image| Self { image })
        })
        .map_err(|e| python_error(py, e))
    }
    #[getter]
    fn width(&self) -> usize {
        self.image.width
    }
    #[getter]
    fn height(&self) -> usize {
        self.image.height
    }
    #[pyo3(signature=(regions,grid=5,max_tables=32,max_total_pixels=32000000))]
    fn plan_json(
        &self,
        py: Python<'_>,
        regions: String,
        grid: usize,
        max_tables: usize,
        max_total_pixels: usize,
    ) -> PyResult<String> {
        py.detach(move || {
            if regions.len() > rapiddoc_core::resources::limit(2_000_000) {
                return Err(rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::ResourceLimit,
                    message: "布局 JSON 超过预算".into(),
                });
            }
            let regions: Vec<rapiddoc_inference::page_tables::LayoutRegion> =
                serde_json::from_str(&regions).map_err(|e| rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message: e.to_string(),
                })?;
            serialize(
                &self
                    .image
                    .plan(&regions, grid, max_tables, max_total_pixels)?,
            )
        })
        .map_err(|e| python_error(py, e))
    }
    fn crop_png(&self, py: Python<'_>, bounds: [usize; 4]) -> PyResult<Py<PyBytes>> {
        let data = py
            .detach(move || self.image.crop_png(bounds))
            .map_err(|e| python_error(py, e))?;
        Ok(PyBytes::new(py, &data).unbind())
    }
    #[pyo3(signature=(regions,grid=5,max_regions=256,max_total_pixels=32000000))]
    fn plan_blocks_json(
        &self,
        py: Python<'_>,
        regions: String,
        grid: usize,
        max_regions: usize,
        max_total_pixels: usize,
    ) -> PyResult<String> {
        py.detach(move || {
            if regions.len() > rapiddoc_core::resources::limit(2_000_000) {
                return Err(rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::ResourceLimit,
                    message: "布局 JSON 超过预算".into(),
                });
            }
            let regions =
                serde_json::from_str::<Vec<rapiddoc_inference::page_tables::LayoutRegion>>(
                    &regions,
                )
                .map_err(|e| rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message: e.to_string(),
                })?;
            serialize(
                &self
                    .image
                    .plan_blocks(&regions, grid, max_regions, max_total_pixels)?,
            )
        })
        .map_err(|e| python_error(py, e))
    }
    fn project_ocr_json(
        &self,
        py: Python<'_>,
        lines: String,
        bounds: [usize; 4],
    ) -> PyResult<String> {
        py.detach(move || serialize(&self.image.project_ocr(&lines, bounds)?))
            .map_err(|e| python_error(py, e))
    }
    fn crop_rgb(&self, py: Python<'_>, bounds: [usize; 4]) -> PyResult<Py<PyBytes>> {
        let data = py
            .detach(move || self.image.crop_rgb(bounds))
            .map_err(|e| python_error(py, e))?;
        Ok(PyBytes::new(py, &data).unbind())
    }
    fn project_json(
        &self,
        py: Python<'_>,
        recognition: String,
        bounds: [usize; 4],
    ) -> PyResult<String> {
        py.detach(move || serialize(&self.image.project(&recognition, bounds)?))
            .map_err(|e| python_error(py, e))
    }
}
