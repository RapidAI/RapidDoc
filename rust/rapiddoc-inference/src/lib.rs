//! 有显式运行库、模型哈希与设备选择的 ONNX 会话；不隐式下载或降级设备。

use ort::{environment::Environment, session::Session, value::Tensor};
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path, PathBuf},
    sync::Mutex,
};

#[path = "onnx/mod.rs"]
pub mod onnx;

static ENVIRONMENT: Mutex<Option<(PathBuf, Environment, bool)>> = Mutex::new(None);
#[path = "table/blank_cell.rs"]
pub mod blank_cell;
#[path = "image/crop.rs"]
pub mod crop;
#[path = "image/crop_mask.rs"]
pub mod crop_mask;
#[path = "ocr/ctc.rs"]
pub mod ctc;
#[path = "ocr/detection.rs"]
pub mod detection;
#[path = "ocr/detection_geometry.rs"]
mod detection_geometry;
#[path = "ocr/detection_mask.rs"]
mod detection_mask;
#[path = "ocr/detection_offset.rs"]
mod detection_offset;
#[path = "ocr/detection_preprocess.rs"]
pub mod detection_preprocess;
#[path = "formula/mod.rs"]
pub mod formula;
#[path = "formula/preprocess.rs"]
pub mod formula_preprocess;
#[path = "layout/mod.rs"]
pub mod layout;
#[path = "layout/contour.rs"]
mod layout_contour;
#[path = "layout/geos.rs"]
pub mod layout_geos;
#[path = "layout/polygons.rs"]
pub mod layout_polygons;
#[path = "layout/post.rs"]
pub mod layout_post;
pub mod magika;
pub mod model_store;
#[path = "runtime/named.rs"]
pub mod named;
#[path = "ocr/native_text.rs"]
pub mod native_text;
#[path = "ocr/numpy_argsort.rs"]
mod numpy_argsort;
#[path = "runtime/optimize.rs"]
mod optimize_runtime;
#[path = "runtime/recognition.rs"]
mod recognition_runtime;
#[path = "runtime/private_crt.rs"]
mod private_crt;
#[path = "image/orientation.rs"]
pub mod orientation;
#[path = "table/page_tables.rs"]
pub mod page_tables;
#[path = "image/pillow_resize.rs"]
mod pillow_resize;
#[path = "ocr/preprocess.rs"]
pub mod preprocess;
#[path = "ocr/recognition_batch.rs"]
pub mod recognition_batch;
#[path = "runtime/registry.rs"]
pub mod registry;
#[path = "seal/mod.rs"]
pub mod seal;
#[path = "runtime/shared_arena.rs"]
mod shared_arena;
#[path = "table/classification.rs"]
pub mod table_classification;
#[path = "table/classifier_gray.rs"]
mod table_classifier_gray;
#[path = "table/components.rs"]
pub mod table_components;
#[path = "table/draw.rs"]
pub mod table_draw;
#[path = "table/lines.rs"]
pub mod table_lines;
#[path = "table/match.rs"]
pub mod table_match;
#[path = "table/raster/mod.rs"]
pub mod table_raster;
#[path = "table/recover.rs"]
pub mod table_recover;
#[path = "table/regions.rs"]
pub mod table_regions;
#[path = "runtime/table.rs"]
mod table_runtime;
#[path = "table/structure.rs"]
pub mod table_structure;
#[path = "table/transformer/mod.rs"]
pub mod table_transformer;
#[path = "table/unicode_data.rs"]
mod table_unicode_data;
#[path = "table/unet.rs"]
pub mod unet;
#[path = "table/unet_lines.rs"]
pub mod unet_lines;
#[path = "ocr/unicode_fix.rs"]
pub mod unicode_fix;
#[path = "table/wired_match.rs"]
pub mod wired_match;

/// 模型建议的校正角度；分类接口本身不修改输入像素或坐标。
#[derive(Debug, Serialize)]
pub struct Orientation {
    pub rotation_degrees: u16,
    pub score: f32,
    pub margin: f32,
}

/// 设备请求，CUDA 必须显式启用编译特性并且运行库能够注册提供器。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
}

/// 自有 FP32 输出，shape 和 values 可直接用于跨语言差分。
#[derive(Debug, Serialize)]
pub struct TensorOutput {
    pub shape: Vec<i64>,
    pub values: Vec<f32>,
}

/// 可复用的单输入 FP32 会话；每次推理需独占借用，防止输出寿命越过下次调用。
pub struct Model {
    session: Session,
    pub device: Device,
    max_elements: usize,
    ctc_alphabet: Option<std::sync::Arc<Vec<String>>>,
    formula_decoder: Option<formula::Decoder>,
    table_decoder: Option<Session>,
    // Fields drop in declaration order: Session must release its allocations
    // before the final arena lease unregisters the environment allocator.
    _shared_arena_lease: Option<shared_arena::Lease>,
}

/// 构造稳定错误，保存原错误文案供诊断。
fn error(code: ErrorCode, value: impl std::fmt::Display) -> Error {
    Error {
        code,
        message: value.to_string(),
    }
}

/// 流式校验已有模型文件，避免为了计算哈希额外持有整个模型。
pub fn verify_model(path: &Path, expected_sha256: &str) -> Result<(), Error> {
    if expected_sha256.len() != 64
        || !expected_sha256
            .bytes()
            .all(|value| value.is_ascii_hexdigit())
    {
        return Err(error(
            ErrorCode::InvalidInput,
            "模型 SHA256 必须为 64 位十六进制字符串",
        ));
    }
    let mut file = std::fs::File::open(path).map_err(|e| error(ErrorCode::ModelMissing, e))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|e| error(ErrorCode::InvalidInput, e))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    if format!("{:x}", hash.finalize()) != expected_sha256.to_ascii_lowercase() {
        return Err(error(ErrorCode::InvalidInput, "模型 SHA256 不匹配"));
    }
    Ok(())
}

impl Model {
    pub fn character_dictionary(&self) -> Result<String, Error> {
        self.session
            .metadata()
            .map_err(|e| error(ErrorCode::InvalidInput, e))?
            .custom("character")
            .ok_or_else(|| error(ErrorCode::Unsupported, "模型没有character元数据"))
    }
    /// 显式 SLANet 变体的 PNG 结构识别，不代替分类、UNET 或单元格 OCR matcher。
    pub fn table_structure_png(
        &mut self,
        bytes: &[u8],
        variant: table_structure::Variant,
    ) -> Result<table_structure::Structure, Error> {
        let (bgr, width, height) = preprocess::decode_png(bytes, 16_000_000)?;
        let prepared = table_structure::prepare(&bgr, width, height, self.max_elements)?;
        let dictionary = self
            .session
            .metadata()
            .map_err(|e| error(ErrorCode::InvalidInput, e))?
            .custom("character")
            .ok_or_else(|| error(ErrorCode::Unsupported, "表格模型没有 character 字典"))?;
        let labels = table_structure::alphabet(&dictionary)?;
        let mut outputs = self.run_table(prepared.tensor.shape, prepared.tensor.values)?;
        if outputs.len() != 2 {
            return Err(error(ErrorCode::Unsupported, "SLANet 需要两个输出"));
        }
        let bboxes = outputs
            .remove("save_infer_model/scale_0.tmp_0")
            .ok_or_else(|| error(ErrorCode::Unsupported, "SLANet 缺少 bbox 输出"))?;
        let probs = outputs
            .remove("save_infer_model/scale_1.tmp_0")
            .ok_or_else(|| error(ErrorCode::Unsupported, "SLANet 缺少结构概率输出"))?;
        table_structure::decode(&bboxes, &probs, &labels, width, height, variant)
    }

    /// Original V3 outputs including masks; compatibility postprocessing keeps auto polygons.
    pub fn layout_polygons_png(
        &mut self,
        bytes: &[u8],
        threshold: f32,
        mode: &str,
        geos: Option<&Path>,
    ) -> Result<Vec<serde_json::Value>, Error> {
        if bytes.len() < 24 || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
            return Err(error(ErrorCode::InvalidInput, "Invalid layout PNG"));
        }
        let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap()) as usize;
        let labels = self
            .character_dictionary()?
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let (mut rgb, decoded_width, decoded_height) = preprocess::decode_png(bytes, 16_000_000)?;
        for pixel in rgb.as_chunks_mut::<3>().0 {
            pixel.swap(0, 2);
        }
        let inputs = self.layout_inputs_rgb(&rgb, decoded_width, decoded_height)?;
        self.run_named_with(inputs, |outputs| {
            layout_polygons::outputs_ref(outputs, &labels, width, height, threshold, mode, geos)
        })
    }
    pub fn layout_polygons_rgb(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
        threshold: f32,
        mode: &str,
        geos: Option<&Path>,
    ) -> Result<Vec<serde_json::Value>, Error> {
        let labels = self
            .character_dictionary()?
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let inputs = self.layout_inputs_rgb(rgb, width, height)?;
        self.run_named_with(inputs, |outputs| {
            layout_polygons::outputs_ref(outputs, &labels, width, height, threshold, mode, geos)
        })
    }
    pub fn layout_raw_png(
        &mut self,
        bytes: &[u8],
    ) -> Result<std::collections::BTreeMap<String, named::Output>, Error> {
        let (mut rgb, width, height) = preprocess::decode_png(bytes, 16_000_000)?;
        for pixel in rgb.as_chunks_mut::<3>().0 {
            pixel.swap(0, 2);
        }
        self.layout_raw_rgb(&rgb, width, height)
    }

    /// Layout tensor preparation from the original RGB byte order, without PNG decode.
    pub fn layout_raw_rgb(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<BTreeMap<String, named::Output>, Error> {
        let inputs = self.layout_inputs_rgb(rgb, width, height)?;
        self.run_named(inputs)
    }

    fn layout_inputs_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<BTreeMap<String, named::Input>, Error> {
        let tensor = layout::prepare_rgb(rgb, width, height, 800, 800, self.max_elements)?;
        let inputs = BTreeMap::from([
            (
                "image".into(),
                named::Input {
                    shape: tensor.shape,
                    values: tensor.values,
                },
            ),
            (
                "im_shape".into(),
                named::Input {
                    shape: vec![1, 2],
                    values: vec![800., 800.],
                },
            ),
            (
                "scale_factor".into(),
                named::Input {
                    shape: vec![1, 2],
                    values: vec![(800. / height as f64) as f32, (800. / width as f64) as f32],
                },
            ),
        ]);
        Ok(inputs)
    }

    /// PP-DocLayoutV3 矩形模式：PNG、RGB cubic、三输入推理、框过滤和原图坐标。
    pub fn layout_png(
        &mut self,
        bytes: &[u8],
        threshold: f32,
    ) -> Result<Vec<layout_post::Region>, Error> {
        if !threshold.is_finite() || !(0. ..=1.).contains(&threshold) {
            return Err(error(ErrorCode::InvalidInput, "版面阈值无效"));
        }
        let (mut rgb, width, height) = preprocess::decode_png(bytes, 16_000_000)?;
        for pixel in rgb.as_chunks_mut::<3>().0 {
            pixel.swap(0, 2);
        }
        self.layout_rgb(&rgb, width, height, threshold)
    }

    pub fn layout_rgb(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
        threshold: f32,
    ) -> Result<Vec<layout_post::Region>, Error> {
        if !threshold.is_finite() || !(0. ..=1.).contains(&threshold) {
            return Err(error(ErrorCode::InvalidInput, "版面阈值无效"));
        }
        let tensor = layout::prepare_rgb(&rgb, width, height, 800, 800, self.max_elements)?;
        let dictionary = self
            .session
            .metadata()
            .map_err(|e| error(ErrorCode::InvalidInput, e))?
            .custom("character")
            .ok_or_else(|| error(ErrorCode::Unsupported, "版面模型没有 character 字典"))?;
        let labels: Vec<String> = dictionary.lines().map(str::to_owned).collect();
        let inputs = BTreeMap::from([
            (
                "image".into(),
                named::Input {
                    shape: tensor.shape,
                    values: tensor.values,
                },
            ),
            (
                "im_shape".into(),
                named::Input {
                    shape: vec![1, 2],
                    values: vec![800., 800.],
                },
            ),
            (
                "scale_factor".into(),
                named::Input {
                    shape: vec![1, 2],
                    values: vec![(800. / height as f64) as f32, (800. / width as f64) as f32],
                },
            ),
        ]);
        let mut outputs = self.run_named(inputs)?;
        let Some(named::Output::Float32 { shape, values }) = outputs.remove("fetch_name_0") else {
            return Err(error(
                ErrorCode::Unsupported,
                "版面模型缺少 FP32 fetch_name_0",
            ));
        };
        if shape.len() != 2 || shape[1] != 7 {
            return Err(error(ErrorCode::Unsupported, "当前版面框需要 Nx7 输出"));
        }
        let Some(named::Output::Int32 { values: count, .. }) = outputs.remove("fetch_name_1")
        else {
            return Err(error(
                ErrorCode::Unsupported,
                "版面模型缺少 int32 fetch_name_1",
            ));
        };
        if count.len() != 1 || count[0] < 0 || count[0] as usize > values.len() / 7 {
            return Err(error(ErrorCode::InvalidInput, "版面候选计数无效"));
        }
        let rows: Vec<[f32; 7]> = values
            .as_chunks::<7>()
            .0
            .iter()
            .copied()
            .take(count[0] as usize)
            .collect();
        layout_post::rectangles(&rows, &labels, width, height, threshold)
    }

    /// 匹配 RapidOrientationModel 的 RGB 输入路径，显式返回模型分类。
    pub fn classify_orientation_png(&mut self, bytes: &[u8]) -> Result<Orientation, Error> {
        let (mut rgb, width, height) = preprocess::decode_png(bytes, 16_000_000)?;
        for pixel in rgb.as_chunks_mut::<3>().0 {
            pixel.swap(0, 2);
        }
        let tensor = orientation::prepare_rgb(&rgb, width, height, 16_000_000, self.max_elements)?;
        let dictionary = self
            .session
            .metadata()
            .map_err(|e| error(ErrorCode::InvalidInput, e))?
            .custom("character")
            .ok_or_else(|| error(ErrorCode::Unsupported, "方向模型没有 character 元数据"))?;
        let labels = dictionary
            .lines()
            .map(|s| {
                s.parse::<u16>()
                    .ok()
                    .filter(|v| matches!(v, 0 | 90 | 180 | 270))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| error(ErrorCode::Unsupported, "方向模型标签必须为 0/90/180/270"))?;
        let outputs = self.run(tensor.shape, tensor.values)?;
        if outputs.len() != 1 {
            return Err(error(ErrorCode::Unsupported, "方向模型需要单输出"));
        }
        let output = outputs
            .into_values()
            .next()
            .ok_or_else(|| error(ErrorCode::InvalidInput, "方向输出为空"))?;
        if output.shape != vec![1, 4] || labels.len() != 4 {
            return Err(error(ErrorCode::Unsupported, "方向输出需要 1x4 和四个标签"));
        }
        let mut index = 0;
        for candidate in 1..4 {
            if output.values[candidate] > output.values[index] {
                index = candidate;
            }
        }
        Ok(Orientation {
            rotation_degrees: labels[index],
            score: output.values[index],
            margin: output.values[index]
                - output
                    .values
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != index)
                    .map(|(_, v)| *v)
                    .fold(f32::NEG_INFINITY, f32::max),
        })
    }

    /// 加载匹配哈希的本地模型；同一进程只允许一套 ORT 动态库。
    pub fn load(
        runtime: &Path,
        model: &Path,
        sha256: &str,
        device: Device,
        threads: usize,
        max_elements: usize,
    ) -> Result<Self, Error> {
        Self::load_with_cpu_options(
            runtime,
            model,
            sha256,
            device,
            threads,
            max_elements,
            None,
            None,
            None,
            None,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_with_cpu_options(
        runtime: &Path,
        model: &Path,
        sha256: &str,
        device: Device,
        threads: usize,
        max_elements: usize,
        cpu_spinning: Option<bool>,
        cpu_dynamic_block_base: Option<usize>,
        cpu_force_spinning_stop: Option<bool>,
        cpu_shared_arena: Option<bool>,
    ) -> Result<Self, Error> {
        Self::load_with_operator_libraries(
            runtime,
            model,
            sha256,
            device,
            threads,
            max_elements,
            cpu_spinning,
            cpu_dynamic_block_base,
            cpu_force_spinning_stop,
            cpu_shared_arena,
            &[],
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_with_operator_libraries(
        runtime: &Path,
        model: &Path,
        sha256: &str,
        device: Device,
        threads: usize,
        max_elements: usize,
        cpu_spinning: Option<bool>,
        cpu_dynamic_block_base: Option<usize>,
        cpu_force_spinning_stop: Option<bool>,
        cpu_shared_arena: Option<bool>,
        operator_libraries: &[std::path::PathBuf],
        optimization: Option<&onnx::Optimization>,
    ) -> Result<Self, Error> {
        if device != Device::Cpu
            && (cpu_spinning.is_some()
                || cpu_dynamic_block_base.is_some()
                || cpu_force_spinning_stop.is_some()
                || cpu_shared_arena.is_some())
        {
            return Err(error(
                ErrorCode::InvalidInput,
                "CPU session options require CPU execution",
            ));
        }
        if cpu_dynamic_block_base.is_some_and(|n| n > i32::MAX as usize) {
            return Err(error(
                ErrorCode::InvalidInput,
                "CPU dynamic block base exceeds int32 range",
            ));
        }
        if max_elements == 0 {
            return Err(error(ErrorCode::InvalidInput, "张量限制必须大于零"));
        }
        #[cfg(not(feature = "cuda"))]
        if device == Device::Cuda {
            return Err(error(
                ErrorCode::DeviceUnavailable,
                "此构建未启用 CUDA 特性",
            ));
        }
        verify_model(model, sha256)?;
        let runtime = runtime
            .canonicalize()
            .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?;
        let _crt = private_crt::activate(&runtime)?;
        let (env, shared_arena_lease) = {
            let mut slot = ENVIRONMENT
                .lock()
                .map_err(|_| error(ErrorCode::RuntimeUnavailable, "ORT 运行库锁已损坏"))?;
            if let Some((path, _, _)) = slot.as_ref() {
                if *path != runtime {
                    return Err(error(
                        ErrorCode::RuntimeUnavailable,
                        "同进程不允许切换 ORT 运行库",
                    ));
                }
            } else {
                let environment = ort::init_from(&runtime)
                    .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?
                    .with_telemetry(false)
                    .build()
                    .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?;
                *slot = Some((runtime, environment, false));
            }
            let lease = if cpu_shared_arena == Some(true) {
                let state = slot
                    .as_mut()
                    .ok_or_else(|| error(ErrorCode::RuntimeUnavailable, "ORT not initialized"))?;
                Some(shared_arena::Lease::acquire(&state.1, &mut state.2)?)
            } else {
                None
            };
            (
                slot.as_ref()
                    .ok_or_else(|| error(ErrorCode::RuntimeUnavailable, "ORT 未初始化"))?
                    .1
                    .clone(),
                lease,
            )
        };
        let builder =
            Session::builder(&env).map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?;
        let builder = if cpu_shared_arena == Some(true) {
            builder
                .with_env_allocators()
                .map_err(|e| error(ErrorCode::InvalidInput, e))?
        } else {
            builder
        };
        // Zero preserves ORT's automatic thread selection, as the original
        // Python engine does for its -1/default setting.
        let builder = if threads == 0 {
            builder
        } else {
            builder
                .with_intra_threads(threads)
                .map_err(|e| error(ErrorCode::InvalidInput, e))?
        };
        let builder = if let Some(spinning) = cpu_spinning {
            builder
                .with_intra_op_spinning(spinning)
                .and_then(|b| b.with_inter_op_spinning(spinning))
                .map_err(|e| error(ErrorCode::InvalidInput, e))?
        } else {
            builder
        };
        let builder = if let Some(base) = cpu_dynamic_block_base {
            builder
                .with_config_entry("session.dynamic_block_base", base.to_string())
                .map_err(|e| error(ErrorCode::InvalidInput, e))?
        } else {
            builder
        };
        #[cfg(feature = "cuda")]
        let builder = if device == Device::Cuda {
            builder
                .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
                .map_err(|e| error(ErrorCode::DeviceUnavailable, e))?
        } else {
            builder
        };
        let builder = if let Some(stop) = cpu_force_spinning_stop {
            builder
                .with_config_entry("session.force_spinning_stop", if stop { "1" } else { "0" })
                .map_err(|e| error(ErrorCode::InvalidInput, e))?
        } else {
            builder
        };
        let mut builder = builder;
        for library in operator_libraries {
            builder = builder
                .with_operator_library(library)
                .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?;
        }
        if optimization.is_some() && device != Device::Cuda {
            return Err(error(
                ErrorCode::InvalidInput,
                "In-memory CUDA optimization requires CUDA execution",
            ));
        }
        let (session, table_decoder) = if let Some(kind) = optimization {
            optimize_runtime::load(builder, &env, model, kind, threads)?
        } else {
            (
                builder
                    .commit_from_file(model)
                    .map_err(|e| error(ErrorCode::InvalidInput, e))?,
                None,
            )
        };
        if session.inputs().is_empty() {
            return Err(error(ErrorCode::Unsupported, "当前会话不支持无输入模型"));
        }
        Ok(Self {
            session,
            device,
            max_elements,
            ctc_alphabet: None,
            formula_decoder: None,
            table_decoder,
            _shared_arena_lease: shared_arena_lease,
        })
    }

    /// PNG 经检测前处理、DB 模型和后处理，返回原图像素四角，不执行识别。
    pub fn detect_png(&mut self, bytes: &[u8]) -> Result<Vec<detection::Detection>, Error> {
        self.detect_png_mode(bytes, false, detection::Config::default(), false)
    }

    /// 旧RapidDoc ocr_patch生效的默认：固定limit_side_len=960、unclip=1.8和内部排序。
    pub fn detect_model_png(&mut self, bytes: &[u8]) -> Result<Vec<detection::Detection>, Error> {
        self.detect_png_mode(bytes, true, detection::Config::body(), false)
    }
    /// The same model path for already prepared pipeline RGB pixels, avoiding a
    /// PNG round trip. Ownership lets RGB -> BGR reuse the allocation.
    pub fn detect_model_rgb(
        &mut self,
        image: rapiddoc_core::pillow_resize::Rgb,
    ) -> Result<Vec<detection::Detection>, Error> {
        self.detect_rgb_mode(image, detection::Config::body(), false)
    }

    /// Table RGB pixels retain the original RGB-to-BGR conversion and DB policy.
    pub fn detect_table_rgb(
        &mut self,
        image: rapiddoc_core::pillow_resize::Rgb,
    ) -> Result<Vec<detection::Detection>, Error> {
        self.detect_rgb_mode(image, detection::Config::default(), true)
    }

    fn detect_rgb_mode(
        &mut self,
        mut image: rapiddoc_core::pillow_resize::Rgb,
        post: detection::Config,
        empty_alignment_is_no_detection: bool,
    ) -> Result<Vec<detection::Detection>, Error> {
        let config = detection_preprocess::DetectionConfig {
            max_elements: self.max_elements,
            ..Default::default()
        };
        let pixels = preprocess::check_size(
            image.w,
            image.h,
            rapiddoc_core::resources::limit(config.max_pixels),
        )?;
        if image.data.len() != pixels * 3 {
            return Err(error(
                ErrorCode::InvalidInput,
                "Invalid detector RGB byte length",
            ));
        }
        for pixel in image.data.as_chunks_mut::<3>().0 {
            pixel.swap(0, 2);
        }
        self.detect_bgr_mode(
            &image.data,
            image.w,
            image.h,
            true,
            post,
            empty_alignment_is_no_detection,
            &config,
        )
    }

    /// 旧表格OCR：原RGB像素、ocr_patch固定960、unclip1.6；外层框排序/遮罩不在此方法执行。
    pub fn detect_table_png(&mut self, bytes: &[u8]) -> Result<Vec<detection::Detection>, Error> {
        self.detect_png_mode(bytes, true, detection::Config::default(), true)
    }

    fn detect_png_mode(
        &mut self,
        bytes: &[u8],
        pipeline: bool,
        post: detection::Config,
        empty_alignment_is_no_detection: bool,
    ) -> Result<Vec<detection::Detection>, Error> {
        let config = detection_preprocess::DetectionConfig {
            max_elements: self.max_elements,
            ..Default::default()
        };
        let (bgr, width, height) = if pipeline {
            preprocess::decode_pipeline_rgb(
                bytes,
                rapiddoc_core::resources::limit(config.max_pixels),
            )?
        } else {
            preprocess::decode_png(bytes, rapiddoc_core::resources::limit(config.max_pixels))?
        };
        self.detect_bgr_mode(
            &bgr,
            width,
            height,
            pipeline,
            post,
            empty_alignment_is_no_detection,
            &config,
        )
    }
    fn detect_bgr_mode(
        &mut self,
        bgr: &[u8],
        width: usize,
        height: usize,
        pipeline: bool,
        post: detection::Config,
        empty_alignment_is_no_detection: bool,
        config: &detection_preprocess::DetectionConfig,
    ) -> Result<Vec<detection::Detection>, Error> {
        if empty_alignment_is_no_detection {
            let ratio = (config.side as f64 / width.max(height) as f64).min(1.);
            if [width, height]
                .iter()
                .any(|n| ((*n as f64 * ratio).trunc() / 32.).round_ties_even() == 0.)
            {
                return Ok(Vec::new());
            }
        }
        let profile = std::env::var_os("RAPIDDOC_CUDA_PROFILE").is_some();
        let started = std::time::Instant::now();
        let tensor = detection_preprocess::prepare(bgr, width, height, config)?;
        let prepared_at = started.elapsed();
        let outputs = self.run(tensor.shape, tensor.values)?;
        let inferred_at = started.elapsed();
        if outputs.len() != 1 {
            return Err(error(ErrorCode::Unsupported, "DB 检测需要单输出模型"));
        }
        let output = outputs
            .into_values()
            .next()
            .ok_or_else(|| error(ErrorCode::InvalidInput, "检测模型输出为空"))?;
        if output.shape.len() != 4
            || output.shape[0] != 1
            || output.shape[1] != 1
            || output.shape[2] <= 0
            || output.shape[3] <= 0
        {
            return Err(error(ErrorCode::Unsupported, "DB 输出必须为 1x1xHxW"));
        }
        let mut boxes = detection::boxes(
            &output.values,
            output.shape[3] as usize,
            output.shape[2] as usize,
            width,
            height,
            &post,
        )?;
        if pipeline {
            detection::sort_reading_order(&mut boxes);
        }
        if profile {
            eprintln!(
                "RAPIDDOC_CUDA_PROFILE detector {}x{} prepare={:.6} run={:.6} post={:.6} boxes={}",
                width,
                height,
                prepared_at.as_secs_f64(),
                (inferred_at - prepared_at).as_secs_f64(),
                (started.elapsed() - inferred_at).as_secs_f64(),
                boxes.len()
            );
        }
        Ok(boxes)
    }

    pub fn detect_seal_bgr(
        &mut self,
        bytes: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<seal::Detection>, Error> {
        let input = seal::prepare(bytes, width, height)?;
        let outputs = self.run(input.shape, input.values)?;
        if outputs.len() != 1 {
            return Err(error(
                ErrorCode::Unsupported,
                "Seal DB model requires one output",
            ));
        }
        let output = outputs.into_values().next().unwrap();
        if output.shape.len() != 4
            || output.shape[0] != 1
            || output.shape[1] != 1
            || output.shape[2] <= 0
            || output.shape[3] <= 0
        {
            return Err(error(
                ErrorCode::Unsupported,
                "Seal DB output must be 1x1xHxW",
            ));
        }
        seal::polygons(
            &output.values,
            output.shape[3] as usize,
            output.shape[2] as usize,
            width,
            height,
        )
    }
    pub fn detect_seal_png(&mut self, bytes: &[u8]) -> Result<Vec<seal::Detection>, Error> {
        let (bgr, w, h) = preprocess::decode_pipeline_rgb(bytes, 64_000_000)?;
        self.detect_seal_bgr(&bgr, w, h)
    }
    /// 识别单行 PNG，使用当前识别合同完成解码、BGR 转换、resize 和 padding。
    pub fn recognize_png(&mut self, bytes: &[u8]) -> Result<Vec<ctc::Recognition>, Error> {
        let tensor = preprocess::from_png(bytes, &preprocess::RecognitionConfig::default())?;
        self.run_ctc(tensor.shape, tensor.values)
    }

    pub fn recognize_batch_png(
        &mut self,
        images: &[Vec<u8>],
        batch_size: usize,
    ) -> Result<Vec<ctc::Recognition>, Error> {
        self.recognize_batch_png_with_progress(images, batch_size, &mut |_, _| Ok(()))
    }
    pub fn recognize_batch_png_with_progress(
        &mut self,
        images: &[Vec<u8>],
        batch_size: usize,
        progress: &mut dyn FnMut(usize, usize) -> Result<(), Error>,
    ) -> Result<Vec<ctc::Recognition>, Error> {
        recognition_batch::plan(&[], batch_size, self.max_elements)?;
        if images.len() > rapiddoc_core::resources::limit(1000) {
            return Err(error(ErrorCode::ResourceLimit, "识别图片数量超过预算"));
        }
        let mut bytes = 0usize;
        let mut pixels = 0usize;
        let mut decoded = Vec::new();
        for image in images {
            bytes = bytes
                .checked_add(image.len())
                .filter(|n| *n <= rapiddoc_core::resources::limit(128_000_000))
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "识别 PNG 字节超过预算"))?;
            if image.len() > rapiddoc_core::resources::limit(64_000_000) {
                return Err(error(ErrorCode::ResourceLimit, "单张识别 PNG 超过预算"));
            }
            let (bgr, width, height) = preprocess::decode_png(
                image,
                rapiddoc_core::resources::limit(16_000_000usize)
                    .min(rapiddoc_core::resources::limit(32_000_000usize).saturating_sub(pixels)),
            )?;
            pixels += width * height;
            decoded.push(recognition_batch::Image { bgr, width, height });
        }
        self.recognize_images_with_progress(&decoded, batch_size, progress)
    }
    pub fn recognize_images(
        &mut self,
        images: &[recognition_batch::Image],
        batch_size: usize,
    ) -> Result<Vec<ctc::Recognition>, Error> {
        self.recognize_images_with_progress(images, batch_size, &mut |_, _| Ok(()))
    }
    /// Original NumPy ordering, batch padding and CTC decoder for owned BGR
    /// crops. Public PNG callers delegate here after their existing validation.
    pub fn recognize_images_with_progress(
        &mut self,
        images: &[recognition_batch::Image],
        batch_size: usize,
        progress: &mut dyn FnMut(usize, usize) -> Result<(), Error>,
    ) -> Result<Vec<ctc::Recognition>, Error> {
        recognition_runtime::recognize(
            images,
            batch_size,
            self.max_elements,
            progress,
            &mut |shape, values| self.forward_ctc(shape, values),
        )
    }

    /// 按给定区域顺序裁图和逐行识别，区域数与累计像素受预算约束。
    pub fn recognize_regions_png(
        &mut self,
        bytes: &[u8],
        regions: &[[[f32; 2]; 4]],
    ) -> Result<Vec<ctc::Recognition>, Error> {
        if regions.len() > rapiddoc_core::resources::limit(1000) {
            return Err(error(ErrorCode::ResourceLimit, "识别区域数量超过预算"));
        }
        let config = preprocess::RecognitionConfig {
            max_elements: self.max_elements,
            ..Default::default()
        };
        let (bgr, width, height) =
            preprocess::decode_png(bytes, rapiddoc_core::resources::limit(config.max_pixels))?;
        let mut remaining = rapiddoc_core::resources::limit(32_000_000usize);
        let mut results = Vec::with_capacity(regions.len());
        for &points in regions {
            let image = crop::quadrilateral(
                &bgr,
                width,
                height,
                points,
                remaining.min(rapiddoc_core::resources::limit(config.max_pixels)),
            )?;
            remaining = remaining
                .checked_sub(image.width * image.height)
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "累计识别裁图超过预算"))?;
            let tensor =
                preprocess::recognition_bgr(&image.bgr, image.width, image.height, &config)?;
            let mut recognition = self.run_ctc(tensor.shape, tensor.values)?;
            if recognition.len() != 1 {
                return Err(error(
                    ErrorCode::Unsupported,
                    "单区域识别结果 batch 必须为 1",
                ));
            }
            results.push(recognition.remove(0));
        }
        Ok(results)
    }

    /// 使用模型的 character 元数据解码 CTC；输入仍需调用方预处理。
    pub fn run_ctc(
        &mut self,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<Vec<ctc::Recognition>, Error> {
        self.forward_ctc(shape, values)?.decode()
    }
    pub(crate) fn forward_ctc(
        &mut self,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<recognition_runtime::Pending, Error> {
        if self.ctc_alphabet.is_none() {
            let metadata = self
                .session
                .metadata()
                .map_err(|e| error(ErrorCode::InvalidInput, e))?;
            let dictionary = metadata
                .custom("character")
                .ok_or_else(|| error(ErrorCode::Unsupported, "模型没有 character 字典"))?;
            if dictionary.chars().any(|c| {
                matches!(
                    c,
                    '\u{b}'
                        | '\u{c}'
                        | '\u{1c}'
                        | '\u{1d}'
                        | '\u{1e}'
                        | '\u{85}'
                        | '\u{2028}'
                        | '\u{2029}'
                )
            }) {
                return Err(error(
                    ErrorCode::Unsupported,
                    "字典含尚未适配的 Unicode 换行符",
                ));
            }
            let mut alphabet = vec!["blank".into()];
            alphabet.extend(dictionary.lines().map(str::to_owned));
            alphabet.push(" ".into());
            self.ctc_alphabet = Some(std::sync::Arc::new(alphabet));
        }
        let alphabet = self.ctc_alphabet.as_ref().unwrap().clone();
        let compact = self
            .session
            .metadata()
            .map_err(|e| error(ErrorCode::InvalidInput, e))?
            .custom("rapiddoc_ctc_compact")
            .as_deref()
            == Some("1");
        if compact {
            let input = self.session.inputs()[0].name().to_owned();
            let outputs = self.run_named_with(
                BTreeMap::from([(input, named::Input { shape, values })]),
                |outputs| {
                    Ok(outputs
                        .into_values()
                        .map(named::OutputRef::into_owned)
                        .collect())
                },
            )?;
            return Ok(recognition_runtime::Pending::Compact { alphabet, outputs });
        }
        Self::run_f32(
            &mut self.session,
            self.max_elements,
            shape,
            values,
            |outputs| {
                if outputs.len() != 1 {
                    return Err(error(ErrorCode::Unsupported, "CTC 需要单输出模型"));
                }
                let value = outputs
                    .into_iter()
                    .next()
                    .ok_or_else(|| error(ErrorCode::InvalidInput, "模型输出为空"))?
                    .1;
                Ok(recognition_runtime::Pending::Full { alphabet, value })
            },
        )
    }

    /// 运行单个 FP32 张量并复制输出；不执行模型特定预处理或后处理。
    pub fn run(
        &mut self,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<BTreeMap<String, TensorOutput>, Error> {
        Self::run_f32(
            &mut self.session,
            self.max_elements,
            shape,
            values,
            |outputs| {
                let mut result = BTreeMap::new();
                for (name, value) in outputs.iter() {
                    let (shape, values) = value
                        .try_extract_tensor::<f32>()
                        .map_err(|e| error(ErrorCode::Unsupported, e))?;
                    result.insert(
                        name.into(),
                        TensorOutput {
                            shape: shape.iter().copied().collect(),
                            values: values.to_vec(),
                        },
                    );
                }
                Ok(result)
            },
        )
    }

    // Consume borrowed ORT outputs before the session can be run again. Public run()
    // still returns owned copies; the CTC path only returns decoded text and scores.
    fn run_f32<T>(
        session: &mut Session,
        max_elements: usize,
        shape: Vec<usize>,
        values: Vec<f32>,
        consume: impl FnOnce(ort::session::SessionOutputs<'_>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        if session.inputs().len() != 1 {
            return Err(error(ErrorCode::Unsupported, "多输入模型请使用 run_named"));
        }
        let count = shape
            .iter()
            .try_fold(1usize, |count, value| count.checked_mul(*value))
            .ok_or_else(|| error(ErrorCode::ResourceLimit, "张量形状溢出"))?;
        if count > max_elements {
            return Err(error(ErrorCode::ResourceLimit, "输入张量超过限制"));
        }
        if shape.is_empty()
            || count == 0
            || count != values.len()
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(error(ErrorCode::InvalidInput, "张量形状、数量或数值无效"));
        }
        let input =
            Tensor::from_array((shape, values)).map_err(|e| error(ErrorCode::InvalidInput, e))?;
        let outputs = session
            .run(ort::inputs![input])
            .map_err(|e| error(ErrorCode::InvalidInput, e))?;
        let mut total = 0usize;
        for (_, value) in outputs.iter() {
            let (_, values) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| error(ErrorCode::Unsupported, e))?;
            total = total
                .checked_add(values.len())
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "输出大小溢出"))?;
            if total > max_elements {
                return Err(error(ErrorCode::ResourceLimit, "输出张量超过复制限制"));
            }
            if values.iter().any(|v| !v.is_finite()) {
                return Err(error(ErrorCode::InvalidInput, "推理输出包含非有限值"));
            }
        }
        consume(outputs)
    }
}
