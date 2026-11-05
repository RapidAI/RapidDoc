//! Source-only migration checkpoints. Enable migration-tools to compile.
use super::*;

#[pyfunction]
pub(super) fn prepare_table_model_payload_bytes(
    py: Python<'_>,
    png: Vec<u8>,
    request: String,
) -> PyResult<(String, Py<PyBytes>)> {
    let out = py
        .detach(move || {
            if request.len() > rapiddoc::legacy_page::host_json_bytes() {
                return Err(rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::ResourceLimit,
                    message: "表格结构输入请求超过预算".into(),
                });
            }
            let value = serde_json::from_str(&request).map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::InvalidInput,
                message: "表格结构输入JSON无效".into(),
            })?;
            rapiddoc::legacy_model::table_hints::payload(&png, &value)
        })
        .map_err(|e| python_error(py, e))?;
    Ok((
        out.metadata.to_string(),
        PyBytes::new(py, &out.png).unbind(),
    ))
}

#[pyfunction]
pub(super) fn extract_table_native_text_json(
    py: Python<'_>,
    png: Vec<u8>,
    request: String,
) -> PyResult<String> {
    py.detach(move || {
        if request.len() > rapiddoc::legacy_page::host_json_bytes() {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "表格原生文字请求超过预算".into(),
            });
        }
        let value = serde_json::from_str(&request).map_err(|_| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: "表格原生文字JSON无效".into(),
        })?;
        rapiddoc::legacy_model::table_native::extract(&png, &value).map(|v| v.to_string())
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
pub(super) fn prepare_table_direction_bytes(
    py: Python<'_>,
    png: Vec<u8>,
    angle: String,
    classifier_bgr: bool,
) -> PyResult<(String, Py<PyBytes>)> {
    let out = py
        .detach(move || {
            rapiddoc::legacy_model::table_direction::image(&png, &angle, classifier_bgr)
        })
        .map_err(|e| python_error(py, e))?;
    Ok((
        out.metadata.to_string(),
        PyBytes::new(py, &out.png).unbind(),
    ))
}

#[pyfunction]
pub(super) fn prepare_table_detection_bytes(
    py: Python<'_>,
    png: Vec<u8>,
    request: String,
) -> PyResult<(String, Py<PyBytes>)> {
    let out = py
        .detach(move || {
            if request.len() > rapiddoc_core::resources::limit(2_000_000) {
                return Err(rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::ResourceLimit,
                    message: "表格检测请求超过预算".into(),
                });
            }
            let v = serde_json::from_str(&request).map_err(|e| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::InvalidInput,
                message: e.to_string(),
            })?;
            rapiddoc::legacy_model::table_text::detection(&png, &v)
        })
        .map_err(|e| python_error(py, e))?;
    Ok((
        out.metadata.to_string(),
        PyBytes::new(py, &out.png).unbind(),
    ))
}

#[pyfunction]
pub(super) fn pad_table_detection_bytes(
    py: Python<'_>,
    png: Vec<u8>,
    padding: usize,
) -> PyResult<Py<PyBytes>> {
    let out = py
        .detach(move || rapiddoc::legacy_model::table_text::padding(&png, padding))
        .map_err(|e| python_error(py, e))?;
    Ok(PyBytes::new(py, &out.png).unbind())
}

#[pyfunction]
pub(super) fn prepare_table_recognition_bytes(
    py: Python<'_>,
    png: Vec<u8>,
    boxes: Vec<[[f32; 2]; 4]>,
) -> PyResult<Vec<Py<PyBytes>>> {
    let out = py
        .detach(move || rapiddoc_inference::crop::pipeline_png_batch(&png, &boxes))
        .map_err(|e| python_error(py, e))?;
    Ok(out
        .into_iter()
        .map(|p| PyBytes::new(py, &p.png).unbind())
        .collect())
}

#[pyfunction]
pub(super) fn select_table_candidate_json(
    py: Python<'_>,
    texts: Vec<String>,
    wired_html: String,
    wireless_html: String,
) -> PyResult<String> {
    py.detach(move || {
        let selected=rapiddoc_render::quality::select(&texts,&wired_html,&wireless_html)?;
        let metrics=|m:&rapiddoc_render::quality::Metrics|serde_json::json!({"physical_cells":m.cells,"blank_cells":m.blank,"non_blank_cells":m.non_blank(),"ocr_hits":m.ocr_hits});
        serialize(&serde_json::json!({"route":selected.route,"reason":selected.reason,"wired":metrics(&selected.wired),"wireless":metrics(&selected.wireless)}))
    }).map_err(|e|python_error(py,e))
}

#[pyfunction]
#[pyo3(signature=(data,width,height,variant="paddle"))]
pub(super) fn prepare_table_classifier_bgr_json(
    py: Python<'_>,
    data: Vec<u8>,
    width: usize,
    height: usize,
    variant: &str,
) -> PyResult<String> {
    let variant = rapiddoc_inference::table_classification::Variant::parse(variant)
        .map_err(|e| python_error(py, e))?;
    py.detach(move || serialize(&variant.prepare(&data, width, height, 16_000_000)?))
        .map_err(|e| python_error(py, e))
}

#[pyfunction]
pub(super) fn decode_table_classification_json(
    py: Python<'_>,
    values: Vec<f32>,
    output_type: &str,
) -> PyResult<String> {
    let output_type = rapiddoc_inference::table_classification::OutputType::parse(output_type)
        .map_err(|e| python_error(py, e))?;
    py.detach(move || {
        serialize(&rapiddoc_inference::table_classification::decode(
            &values,
            output_type,
        )?)
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
pub(super) fn render_wired_table_json(
    py: Python<'_>,
    points: Vec<[usize; 4]>,
    polygons: Vec<rapiddoc_render::wired::Polygon>,
    texts: Vec<Vec<String>>,
) -> PyResult<String> {
    py.detach(move || rapiddoc_render::wired::html(&points, &polygons, &texts))
        .map_err(|error| python_error(py, error))
}

#[pyfunction]
pub(super) fn blank_cell_crop_json(
    py: Python<'_>,
    bytes: Vec<u8>,
    width: usize,
    height: usize,
    points: rapiddoc_inference::table_recover::Polygon,
) -> PyResult<String> {
    py.detach(move || {
        serialize(&rapiddoc_inference::blank_cell::crop(
            &bytes, width, height, points,
        )?)
    })
    .map_err(|error| python_error(py, error))
}

#[pyfunction]
pub(super) fn match_cells_json(
    py: Python<'_>,
    cells: Vec<Vec<f64>>,
    ocr: Vec<Vec<f64>>,
) -> PyResult<String> {
    py.detach(move || serialize(&rapiddoc_inference::table_match::match_cells(&cells, &ocr)?))
        .map_err(|error| python_error(py, error))
}

#[pyfunction]
pub(super) fn prepare_unet_bgr_json(
    py: Python<'_>,
    data: Vec<u8>,
    width: usize,
    height: usize,
) -> PyResult<String> {
    py.detach(move || {
        serialize(&rapiddoc_inference::unet::prepare(
            &data, width, height, 16_000_000,
        )?)
    })
    .map_err(|error| python_error(py, error))
}

#[pyfunction]
pub(super) fn table_components_json(
    py: Python<'_>,
    data: Vec<u8>,
    width: usize,
    height: usize,
    axis: Option<usize>,
    line_extent: usize,
) -> PyResult<String> {
    py.detach(move || {
        serialize(&rapiddoc_inference::table_components::analyze(
            &data,
            width,
            height,
            axis,
            line_extent,
        )?)
    })
    .map_err(|error| python_error(py, error))
}

#[pyfunction]
pub(super) fn unet_line_geometry_json(
    py: Python<'_>,
    request: String,
    width: usize,
    height: usize,
    morph_close: bool,
    config: String,
) -> PyResult<String> {
    py.detach(move || {
        if request.len() > rapiddoc_core::resources::limit(16_000_000) {
            return Err(rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::ResourceLimit,
                message: "UNET mask JSON 超过预算".into(),
            });
        }
        let invalid = |e: serde_json::Error| rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: e.to_string(),
        };
        let mask: rapiddoc_inference::unet::Mask =
            serde_json::from_str(&request).map_err(invalid)?;
        let config: rapiddoc_inference::table_lines::Config =
            serde_json::from_str(&config).map_err(invalid)?;
        serialize(&rapiddoc_inference::unet_lines::geometry(
            &mask,
            width,
            height,
            morph_close,
            &config,
        )?)
    })
    .map_err(|error| python_error(py, error))
}

#[pyfunction]
pub(super) fn prepare_unet_line_masks_json(
    py: Python<'_>,
    classes: Vec<u8>,
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
    morph_close: bool,
) -> PyResult<String> {
    py.detach(move || {
        serialize(&rapiddoc_inference::unet_lines::prepare(
            &rapiddoc_inference::unet::Mask {
                width,
                height,
                classes,
            },
            target_width,
            target_height,
            morph_close,
        )?)
    })
    .map_err(|error| python_error(py, error))
}

#[pyfunction]
pub(super) fn draw_table_lines_bytes(
    py: Python<'_>,
    lines: Vec<rapiddoc_inference::table_lines::Line>,
    width: usize,
    height: usize,
) -> PyResult<Bound<'_, PyBytes>> {
    let result = py
        .detach(move || rapiddoc_inference::table_draw::draw(&lines, width, height))
        .map_err(|error| python_error(py, error))?;
    Ok(PyBytes::new(py, &result))
}

#[pyfunction]
pub(super) fn rotate_table_image_bytes(
    py: Python<'_>,
    data: Vec<u8>,
    width: usize,
    height: usize,
    angle: f64,
) -> PyResult<Bound<'_, PyBytes>> {
    let result = py
        .detach(move || rapiddoc_inference::table_regions::rotate(&data, width, height, angle))
        .map_err(|error| python_error(py, error))?;
    Ok(PyBytes::new(py, &result))
}

#[pyfunction]
pub(super) fn table_rotation_angle(
    py: Python<'_>,
    data: Vec<u8>,
    width: usize,
    height: usize,
) -> PyResult<f32> {
    py.detach(move || rapiddoc_inference::table_regions::angle(&data, width, height))
        .map_err(|error| python_error(py, error))
}

#[pyfunction]
#[pyo3(signature=(lines,width,height,rotated_fix,recover_topology=true))]
pub(super) fn table_cells_from_lines_json(
    py: Python<'_>,
    lines: Vec<rapiddoc_inference::table_lines::Line>,
    width: usize,
    height: usize,
    rotated_fix: bool,
    recover_topology: bool,
) -> PyResult<String> {
    py.detach(move || {
        let cells = if recover_topology {
            rapiddoc_inference::table_regions::from_lines(&lines, width, height, rotated_fix)?
        } else {
            rapiddoc_inference::table_regions::regions_from_lines(
                &lines,
                width,
                height,
                rotated_fix,
            )?
        };
        serialize(&cells)
    })
    .map_err(|error| python_error(py, error))
}

#[pyfunction]
pub(super) fn recover_table_polygons_json(
    py: Python<'_>,
    polygons: Vec<rapiddoc_inference::table_recover::Polygon>,
    row_threshold: f64,
    column_threshold: f64,
) -> PyResult<String> {
    py.detach(move || {
        serialize(&rapiddoc_inference::table_recover::recover(
            &polygons,
            row_threshold,
            column_threshold,
        )?)
    })
    .map_err(|error| python_error(py, error))
}

#[pyfunction]
pub(super) fn connect_table_lines_json(
    py: Python<'_>,
    rows: Vec<rapiddoc_inference::table_lines::Line>,
    columns: Vec<rapiddoc_inference::table_lines::Line>,
    config: String,
) -> PyResult<String> {
    py.detach(move || {
        let config: rapiddoc_inference::table_lines::Config = serde_json::from_str(&config)
            .map_err(|e| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::InvalidInput,
                message: e.to_string(),
            })?;
        serialize(&rapiddoc_inference::table_lines::connect(
            rows, columns, &config,
        )?)
    })
    .map_err(|error| python_error(py, error))
}

#[pyfunction]
pub(super) fn prepare_table_bgr_json(
    py: Python<'_>,
    data: Vec<u8>,
    width: usize,
    height: usize,
) -> PyResult<String> {
    py.detach(move || {
        serialize(&rapiddoc_inference::table_structure::prepare(
            &data, width, height, 16_000_000,
        )?)
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_table_structure_json(
    py: Python<'_>,
    bbox_shape: Vec<i64>,
    bbox_values: Vec<f32>,
    prob_shape: Vec<i64>,
    prob_values: Vec<f32>,
    dictionary: String,
    width: usize,
    height: usize,
    plus: bool,
) -> PyResult<String> {
    py.detach(move || {
        let labels = rapiddoc_inference::table_structure::alphabet(&dictionary)?;
        serialize(&rapiddoc_inference::table_structure::decode(
            &rapiddoc_inference::TensorOutput {
                shape: bbox_shape,
                values: bbox_values,
            },
            &rapiddoc_inference::TensorOutput {
                shape: prob_shape,
                values: prob_values,
            },
            &labels,
            width,
            height,
            if plus {
                rapiddoc_inference::table_structure::Variant::SlanetPlus
            } else {
                rapiddoc_inference::table_structure::Variant::Slanet1m
            },
        )?)
    })
    .map_err(|e| python_error(py, e))
}
