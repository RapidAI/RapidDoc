//! Source-only migration checkpoints. Enable migration-tools to compile.
use super::*;

#[pyfunction]
pub(super) fn fix_unicode_text(py: Python<'_>, text: String) -> PyResult<String> {
    py.detach(move || rapiddoc_inference::unicode_fix::fix_text(&text))
        .map_err(|e| python_error(py, e))
}

#[pyfunction]
pub(super) fn prepare_model_recognition_bytes(
    py: Python<'_>,
    request: String,
    pngs: Vec<Vec<u8>>,
) -> PyResult<PageAssets> {
    let (plan, images) = py
        .detach(move || {
            use rapiddoc_core::{Error, ErrorCode};
            let invalid = || Error {
                code: ErrorCode::InvalidInput,
                message: "模型识别裁图输入无效".into(),
            };
            let limit = || Error {
                code: ErrorCode::ResourceLimit,
                message: "模型识别裁图超过预算".into(),
            };
            if request.len() > rapiddoc_core::resources::limit(32_000_000)
                || pngs.len() > rapiddoc_core::resources::limit(1000)
            {
                return Err(limit());
            }
            let request: serde_json::Value =
                serde_json::from_str(&request).map_err(|_| invalid())?;
            let mut plan = rapiddoc::legacy_model::ocr_results::plan(&request)?;
            let attempts = plan["crop_attempts"].as_array().ok_or_else(invalid)?;
            if attempts.len() != pngs.len() {
                return Err(invalid());
            }
            let (mut bytes, mut pixels, mut encoded) = (0usize, 0usize, 0usize);
            let mut images = Vec::new();
            for (attempt, png) in attempts.iter().zip(pngs) {
                bytes = bytes
                    .checked_add(png.len())
                    .filter(|n| {
                        *n <= rapiddoc_core::resources::limit(128_000_000)
                            && png.len() <= rapiddoc_core::resources::limit(64_000_000)
                    })
                    .ok_or_else(limit)?;
                if png.len() < 33
                    || !png.starts_with(b"\x89PNG\r\n\x1a\n")
                    || png[24] != 8
                    || png[25] != 2
                {
                    return Err(invalid());
                }
                let points: [[f32; 2]; 4] =
                    serde_json::from_value(attempt["points"].clone()).map_err(|_| invalid())?;
                let (width, height, png) = rapiddoc_inference::crop::pipeline_png_with_limits(
                    &png,
                    points,
                    if rapiddoc::legacy_page::pdf_limits().max_chars_per_page == usize::MAX {
                        16_000_000
                    } else {
                        4_000_000
                    },
                    4_000_000,
                )?;
                pixels = pixels
                    .checked_add(width * height)
                    .filter(|n| *n <= rapiddoc_core::resources::limit(32_000_000))
                    .ok_or_else(limit)?;
                if let Some(id) = attempt["output"].as_str() {
                    encoded = encoded
                        .checked_add(png.len())
                        .filter(|n| *n <= rapiddoc_core::resources::limit(128_000_000))
                        .ok_or_else(limit)?;
                    images.push((id.to_owned(), png));
                }
            }
            plan.as_object_mut().unwrap().remove("crop_attempts");
            Ok((plan, images))
        })
        .map_err(|e| python_error(py, e))?;
    Ok((
        plan.to_string(),
        images
            .into_iter()
            .map(|(id, data)| (id, PyBytes::new(py, &data).unbind()))
            .collect(),
    ))
}

#[pyfunction]
pub(super) fn prepare_model_detection_bytes(
    py: Python<'_>,
    request: String,
    pngs: Vec<Vec<u8>>,
) -> PyResult<PageAssets> {
    let output = py
        .detach(move || {
            if request.len() > rapiddoc::legacy_page::host_json_bytes() {
                return Err(rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::ResourceLimit,
                    message: "OCR检测请求过大".into(),
                });
            }
            let v: serde_json::Value =
                serde_json::from_str(&request).map_err(|e| rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message: e.to_string(),
                })?;
            rapiddoc::legacy_model::native_ocr::detection::prepare(&v, &pngs)
        })
        .map_err(|e| python_error(py, e))?;
    Ok((
        output.plan.to_string(),
        output
            .images
            .into_iter()
            .map(|(key, data)| (key, PyBytes::new(py, &data).unbind()))
            .collect(),
    ))
}

#[pyfunction]
#[pyo3(signature=(request, png, mask_polygons=false))]
pub(super) fn prepare_model_crops_bytes(
    py: Python<'_>,
    request: String,
    png: Vec<u8>,
    mask_polygons: bool,
) -> PyResult<PageAssets> {
    let output = py
        .detach(move || {
            if request.len() > rapiddoc::legacy_page::host_json_bytes() {
                return Err(rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::ResourceLimit,
                    message: "模型路由请求过大".into(),
                });
            }
            let request: serde_json::Value =
                serde_json::from_str(&request).map_err(|e| rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message: e.to_string(),
                })?;
            let mut output = rapiddoc::legacy_model::crops::prepare(&request, &png)?;
            if mask_polygons {
                rapiddoc_inference::crop_mask::apply(
                    &output.plan,
                    &mut output.images,
                    request
                        .get("formula_bbox_expand_px")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(2),
                )?;
            }
            Ok::<_, rapiddoc_core::Error>(output)
        })
        .map_err(|e| python_error(py, e))?;
    Ok((
        output.plan.to_string(),
        output
            .images
            .into_iter()
            .map(|(key, data)| (key, PyBytes::new(py, &data).unbind()))
            .collect(),
    ))
}

#[pyfunction]
pub(super) fn prepare_recognition_bgr_json(
    py: Python<'_>,
    data: Vec<u8>,
    width: usize,
    height: usize,
) -> PyResult<String> {
    py.detach(move || {
        serialize(&rapiddoc_inference::preprocess::recognition_bgr(
            &data,
            width,
            height,
            &rapiddoc_inference::preprocess::RecognitionConfig::default(),
        )?)
    })
    .map_err(|error| python_error(py, error))
}

#[pyfunction]
pub(super) fn prepare_recognition_batch_bgr_json(
    py: Python<'_>,
    images: Vec<(Vec<u8>, usize, usize)>,
    batch_size: usize,
) -> PyResult<String> {
    py.detach(move || {
        let images:Vec<_>=images.into_iter().map(|(bgr,width,height)|rapiddoc_inference::recognition_batch::Image {bgr,width,height}).collect();
        let sizes:Vec<_>=images.iter().map(|i|[i.width,i.height]).collect();
        let plan=rapiddoc_inference::recognition_batch::plan(&sizes,batch_size,16_000_000)?;
        let mut out=Vec::new();for batch in plan {
            let tensor=rapiddoc_inference::recognition_batch::prepare(&images,&batch,16_000_000)?;
            out.push(serde_json::json!({"indices":batch.indices,"shape":tensor.shape,"values":tensor.values}));
        }serialize(&out)
    }).map_err(|e|python_error(py,e))
}

#[pyfunction]
#[pyo3(signature=(library,data,detector=None,recognizer=None,start=0,end=None,password=None,dpi=144.0,max_pixels=16000000,max_total_pixels=128000000,max_ocr_pages=256))]
#[allow(clippy::too_many_arguments)]
pub(super) fn parse_pdf_ocr_json(
    py: Python<'_>,
    library: String,
    data: Vec<u8>,
    detector: Option<PyRef<'_, OnnxSession>>,
    recognizer: Option<PyRef<'_, OnnxSession>>,
    start: usize,
    end: Option<usize>,
    password: Option<String>,
    dpi: f64,
    max_pixels: usize,
    max_total_pixels: usize,
    max_ocr_pages: usize,
) -> PyResult<String> {
    let detector = detector.as_ref().map(|s| &s.model);
    let recognizer = recognizer.as_ref().map(|s| &s.model);
    py.detach(move || {
        let config = rapiddoc::ocr::Config {
            dpi,
            max_pixels_per_page: max_pixels,
            max_total_pixels,
            max_ocr_pages,
        };
        let output = rapiddoc::ocr::parse(
            &PathBuf::from(library),
            &data,
            password.as_deref(),
            &PageRange { start, end },
            &rapiddoc::legacy_page::pdf_limits(),
            &config,
            |png| {
                let (detector, recognizer) =
                    detector
                        .zip(recognizer)
                        .ok_or_else(|| rapiddoc_core::Error {
                            code: rapiddoc_core::ErrorCode::ModelMissing,
                            message: "页面需要 OCR，请提供 detector 和 recognizer 会话".into(),
                        })?;
                Ok(run_ocr(detector, recognizer, png)?
                    .into_iter()
                    .map(|line| rapiddoc_core::ocr::TextLine {
                        points: line.points.map(|p| p.map(f64::from)),
                        text: line.recognition.text,
                        confidence: line.recognition.confidence,
                        detection_score: line.detection_score,
                    })
                    .collect())
            },
        )?;
        serialize(&output)
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
pub(super) fn seal_geometry_json(py: Python<'_>, request: &str) -> PyResult<String> {
    let request = request.to_owned();
    fn invalid_json(e: serde_json::Error) -> rapiddoc_core::Error {
        rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: e.to_string(),
        }
    }
    py.detach(move || {
        let v: serde_json::Value = serde_json::from_str(&request).map_err(invalid_json)?;
        let invalid = || rapiddoc_core::Error {
            code: rapiddoc_core::ErrorCode::InvalidInput,
            message: "Invalid seal geometry request".into(),
        };
        let number = |key: &str| {
            v[key]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(invalid)
        };
        let points =
            || serde_json::from_value::<Vec<[f64; 2]>>(v["points"].clone()).map_err(invalid_json);
        match v["operation"].as_str() {
            Some("unclip") => serialize(&rapiddoc_inference::seal::unclip(
                &points()?,
                v["ratio"].as_f64().ok_or_else(invalid)?,
            )?),
            Some("detection") => {
                let values: Vec<f32> =
                    serde_json::from_value(v["values"].clone()).map_err(invalid_json)?;
                serialize(&rapiddoc_inference::seal::polygons(
                    &values,
                    number("width")?,
                    number("height")?,
                    number("source_width")?,
                    number("source_height")?,
                )?)
            }
            Some("crop") => {
                let data: Vec<u8> =
                    serde_json::from_value(v["bgr"].clone()).map_err(invalid_json)?;
                let path = v["geos_library"].as_str().ok_or_else(invalid)?;
                serialize(
                    &rapiddoc_inference::seal::Cropper::load(std::path::Path::new(path))?.crop(
                        &data,
                        number("width")?,
                        number("height")?,
                        &points()?,
                    )?,
                )
            }
            Some("prepare") => {
                let data: Vec<u8> =
                    serde_json::from_value(v["bgr"].clone()).map_err(invalid_json)?;
                serialize(&rapiddoc_inference::seal::prepare(
                    &data,
                    number("width")?,
                    number("height")?,
                )?)
            }
            _ => Err(invalid()),
        }
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(data,width,height,points,sampling="opencv5"))]
pub(super) fn crop_bgr_json(
    py: Python<'_>,
    data: Vec<u8>,
    width: usize,
    height: usize,
    points: [[f32; 2]; 4],
    sampling: &str,
) -> PyResult<String> {
    let sampling =
        serde_json::from_value::<rapiddoc_inference::crop::Sampling>(serde_json::json!(sampling))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    py.detach(move || {
        rapiddoc_inference::crop::with_sampling(sampling, || {
            serialize(&rapiddoc_inference::crop::quadrilateral(
                &data, width, height, points, 16_000_000,
            )?)
        })
    })
    .map_err(|e| python_error(py, e))
}

#[pyfunction]
#[pyo3(signature=(values,width,height,source_width,source_height,dilation=true))]
pub(super) fn detection_boxes_json(
    py: Python<'_>,
    values: Vec<f32>,
    width: usize,
    height: usize,
    source_width: usize,
    source_height: usize,
    dilation: bool,
) -> PyResult<String> {
    py.detach(move || {
        serialize(&rapiddoc_inference::detection::boxes(
            &values,
            width,
            height,
            source_width,
            source_height,
            &rapiddoc_inference::detection::Config {
                dilation,
                ..Default::default()
            },
        )?)
    })
    .map_err(|error| python_error(py, error))
}

#[pyfunction]
#[pyo3(signature=(data,width,height,limit_side_len=960,limit_type="max"))]
pub(super) fn prepare_detection_bgr_json(
    py: Python<'_>,
    data: Vec<u8>,
    width: usize,
    height: usize,
    limit_side_len: usize,
    limit_type: &str,
) -> PyResult<String> {
    use rapiddoc_inference::detection_preprocess::{DetectionConfig, LimitType};
    let mode = match limit_type {
        "min" => LimitType::Minimum,
        "max" => LimitType::Maximum,
        _ => return Err(PyValueError::new_err("limit_type must be min or max")),
    };
    py.detach(move || {
        serialize(&rapiddoc_inference::detection_preprocess::prepare(
            &data,
            width,
            height,
            &DetectionConfig {
                side: limit_side_len,
                mode,
                ..DetectionConfig::default()
            },
        )?)
    })
    .map_err(|error| python_error(py, error))
}
