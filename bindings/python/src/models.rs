//! models adapters; algorithms remain in the shared Rust crates.
use super::*;

/// 可复用的本地 ONNX 会话；内部锁防止多线程共享同一可变 session。
#[pyclass(module = "rapid_doc_rust._native")]
pub(super) struct OnnxSession {
    pub(super) model: rapiddoc_inference::registry::Shared,
}

#[pymethods]
impl OnnxSession {
    fn table_transformer_png_json(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        tokens: &str,
    ) -> PyResult<String> {
        let tokens: Vec<rapiddoc_inference::table_transformer::Token> =
            serde_json::from_str(tokens).map_err(|e| PyValueError::new_err(e.to_string()))?;
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX session lock poisoned".into(),
            })?;
            serialize(&model.transformer_cells_png(&data, &tokens)?)
        })
        .map_err(|e| python_error(py, e))
    }
    fn recognize_blank_table_cells_png_json(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        polygons: Vec<rapiddoc_inference::table_recover::Polygon>,
        needed: Vec<bool>,
    ) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            serialize(&model.recognize_blank_cells_png(&data, &polygons, &needed)?)
        })
        .map_err(|error| python_error(py, error))
    }
    fn table_cells_png_json(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        rotated_fix: bool,
    ) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            serialize(&model.table_cells_png(&data, rotated_fix)?)
        })
        .map_err(|error| python_error(py, error))
    }
    fn table_line_mask_png_json(&self, py: Python<'_>, data: Vec<u8>) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            serialize(&model.table_line_mask_png(&data)?)
        })
        .map_err(|error| python_error(py, error))
    }
    /// 显式指定模型变体，防止把 SLANet 的 bbox 当作 plus 坐标回缩。
    #[pyo3(signature=(data,variant="slanet-plus",legacy_coordinates=false))]
    fn table_structure_png_json(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        variant: &str,
        legacy_coordinates: bool,
    ) -> PyResult<String> {
        let variant = match variant {
            "slanet-plus" => rapiddoc_inference::table_structure::Variant::SlanetPlus,
            "slanet-1m" => rapiddoc_inference::table_structure::Variant::Slanet1m,
            _ => {
                return Err(PyValueError::new_err(
                    "variant must be slanet-plus or slanet-1m",
                ))
            }
        };
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            let structure=model.table_structure_png(&data, variant)?;
            if legacy_coordinates {
                let cells=structure.cell_bboxes.iter().map(|b|b.iter().map(|v|f64::from(*v)).collect::<Vec<_>>()).collect::<Vec<_>>();
                serialize(&serde_json::json!({"tokens":structure.tokens,"confidence":structure.confidence,"cell_bboxes":cells}))
            } else {serialize(&structure)}
        })
        .map_err(|e| python_error(py, e))
    }
    /// PP-DocLayoutV3 的矩形模式预览，不生成掩码多边形或组合整份文档版面。
    #[pyo3(signature=(data,threshold=0.5))]
    fn layout_png_json(&self, py: Python<'_>, data: Vec<u8>, threshold: f32) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            serialize(&model.layout_png(&data, threshold)?)
        })
        .map_err(|e| python_error(py, e))
    }
    /// 多输入 FP32 / 有类型输出的开发入口，输入从 JSON 读取后严格验证。
    fn run_named_json(&self, py: Python<'_>, inputs: String) -> PyResult<String> {
        py.detach(move || {
            let inputs: std::collections::BTreeMap<String, rapiddoc_inference::named::Input> =
                serde_json::from_str(&inputs).map_err(|e| rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::InvalidInput,
                    message: e.to_string(),
                })?;
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            serialize(&model.run_named(inputs)?)
        })
        .map_err(|e| python_error(py, e))
    }
    /// 对 PNG 分类方向，返回模型建议角度和原始评分，不自动旋转图像。
    fn classify_orientation_png_json(&self, py: Python<'_>, data: Vec<u8>) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            serialize(&model.classify_orientation_png(&data)?)
        })
        .map_err(|e| python_error(py, e))
    }

    #[pyo3(signature=(data,variant,output_type=None))]
    fn classify_table_png_json(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        variant: &str,
        output_type: Option<&str>,
    ) -> PyResult<String> {
        let variant = rapiddoc_inference::table_classification::Variant::parse(variant)
            .map_err(|e| python_error(py, e))?;
        let output_type = match output_type {
            Some(value) => rapiddoc_inference::table_classification::OutputType::parse(value)
                .map_err(|e| python_error(py, e))?,
            None => variant.output_type(),
        };
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            serialize(&model.classify_table_png(&data, variant, output_type)?)
        })
        .map_err(|e| python_error(py, e))
    }
    /// 显式指定模型哈希、运行库及设备，加载期间释放 GIL，不下载任何资源。
    #[new]
    #[pyo3(signature = (runtime, model, sha256, device="cpu", threads=1, max_elements=isize::MAX as usize / 4, cpu_spinning=None, cpu_dynamic_block_base=None, cpu_force_spinning_stop=None, cpu_shared_arena=None, operator_libraries=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        runtime: String,
        model: String,
        sha256: String,
        device: &str,
        threads: usize,
        max_elements: usize,
        cpu_spinning: Option<bool>,
        cpu_dynamic_block_base: Option<usize>,
        cpu_force_spinning_stop: Option<bool>,
        cpu_shared_arena: Option<bool>,
        operator_libraries: Option<Vec<PathBuf>>,
    ) -> PyResult<Self> {
        let device = match device {
            "cpu" => rapiddoc_inference::Device::Cpu,
            "cuda" => rapiddoc_inference::Device::Cuda,
            _ => return Err(PyValueError::new_err("device must be cpu or cuda")),
        };
        let model = py
            .detach(move || {
                rapiddoc_inference::registry::cached(&rapiddoc_inference::registry::Descriptor {
                    runtime: PathBuf::from(runtime),
                    model: PathBuf::from(model),
                    sha256,
                    device: if device == rapiddoc_inference::Device::Cpu {
                        "cpu".into()
                    } else {
                        "cuda".into()
                    },
                    threads,
                    max_elements,
                    cpu_spinning,
                    cpu_dynamic_block_base,
                    cpu_force_spinning_stop,
                    cpu_shared_arena,
                    operator_libraries: operator_libraries.unwrap_or_default(),
                    optimization: None,
                })
            })
            .map_err(|error| python_error(py, error))?;
        Ok(Self { model })
    }

    fn character_dictionary(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            self.model
                .lock()
                .map_err(|_| rapiddoc_core::Error {
                    code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                    message: "ONNX会话锁损坏".into(),
                })?
                .character_dictionary()
        })
        .map_err(|e| python_error(py, e))
    }
    fn layout_raw_png_json(&self, py: Python<'_>, data: Vec<u8>) -> PyResult<String> {
        py.detach(|| {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX会话锁损坏".into(),
            })?;
            serialize(&model.layout_raw_png(&data)?)
        })
        .map_err(|e| python_error(py, e))
    }
    #[pyo3(signature=(data,threshold=0.3,mode="auto",geos_library=None))]
    fn layout_polygons_png_json(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        threshold: f32,
        mode: &str,
        geos_library: Option<String>,
    ) -> PyResult<String> {
        let geos = geos_library.map(PathBuf::from);
        py.detach(|| {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX session lock poisoned".into(),
            })?;
            serialize(&model.layout_polygons_png(&data, threshold, mode, geos.as_deref())?)
        })
        .map_err(|e| python_error(py, e))
    }

    /// 接收 FP32 扁平数据和形状，输出 JSON 包含每个输出的名称、shape 和完整 values。
    fn run_json(&self, py: Python<'_>, shape: Vec<usize>, values: Vec<f32>) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            let outputs = model.run(shape, values)?;
            serde_json::to_string(&outputs).map_err(|error| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::InvalidInput,
                message: error.to_string(),
            })
        })
        .map_err(|error| python_error(py, error))
    }
    /// 对单行 PNG 在 Rust 中完成前处理、推理和 CTC 解码。
    fn recognize_regions_png_json(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        regions: Vec<[[f32; 2]; 4]>,
    ) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            serialize(&model.recognize_regions_png(&data, &regions)?)
        })
        .map_err(|error| python_error(py, error))
    }

    /// 对单行 PNG 在 Rust 中完成前处理、推理和 CTC 解码。
    #[pyo3(signature=(data,sort=false))]
    fn detect_png_json(&self, py: Python<'_>, data: Vec<u8>, sort: bool) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            let mut boxes = model.detect_png(&data)?;
            if sort {
                rapiddoc_inference::detection::sort_reading_order(&mut boxes);
            }
            serialize(&boxes)
        })
        .map_err(|error| python_error(py, error))
    }

    fn detect_model_png_json(&self, py: Python<'_>, data: Vec<u8>) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX会话锁已损坏".into(),
            })?;
            serialize(&model.detect_model_png(&data)?)
        })
        .map_err(|e| python_error(py, e))
    }

    fn detect_seal_png_json(&self, py: Python<'_>, data: Vec<u8>) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX session lock poisoned".into(),
            })?;
            serialize(&model.detect_seal_png(&data)?)
        })
        .map_err(|e| python_error(py, e))
    }

    /// 对单行 PNG 在 Rust 中完成前处理、推理和 CTC 解码。
    fn recognize_png_json(&self, py: Python<'_>, data: Vec<u8>) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            serialize(&model.recognize_png(&data)?)
        })
        .map_err(|error| python_error(py, error))
    }

    #[pyo3(signature=(images,batch_size=6))]
    fn recognize_batch_png_json(
        &self,
        py: Python<'_>,
        images: Vec<Vec<u8>>,
        batch_size: usize,
    ) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            serialize(&model.recognize_batch_png(&images, batch_size)?)
        })
        .map_err(|e| python_error(py, e))
    }

    #[pyo3(signature=(images,batch_size=1))]
    fn recognize_formula_batch_png_json(
        &self,
        py: Python<'_>,
        images: Vec<Vec<u8>>,
        batch_size: usize,
    ) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX会话锁已损坏".into(),
            })?;
            serialize(&model.recognize_formula_batch_png(&images, batch_size)?)
        })
        .map_err(|e| python_error(py, e))
    }

    fn detect_table_png_json(&self, py: Python<'_>, data: Vec<u8>) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX会话锁已损坏".into(),
            })?;
            serialize(&model.detect_table_png(&data)?)
        })
        .map_err(|e| python_error(py, e))
    }

    /// 对识别张量执行模型和 CTC 解码，避免将大 logits 传回 Python。
    fn recognize_ctc_json(
        &self,
        py: Python<'_>,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) -> PyResult<String> {
        py.detach(move || {
            let mut model = self.model.lock().map_err(|_| rapiddoc_core::Error {
                code: rapiddoc_core::ErrorCode::RuntimeUnavailable,
                message: "ONNX 会话锁已损坏".into(),
            })?;
            serialize(&model.run_ctc(shape, values)?)
        })
        .map_err(|error| python_error(py, error))
    }
}
