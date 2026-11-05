//! Model selection/validation shared by native callers and every binding.
use super::*;
use rapiddoc_inference::model_store::Catalog;
use rapiddoc_inference::registry::Descriptor;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSelection {
    pub role: String,
    pub model_id: String,
    #[serde(rename = "override")]
    pub path: Option<PathBuf>,
    pub threads: usize,
    pub device: String,
    pub cpu_spinning: Option<bool>,
    pub cpu_dynamic_block_base: Option<usize>,
    pub cpu_force_spinning_stop: Option<bool>,
    pub cpu_shared_arena: Option<bool>,
    pub partition_table: bool,
    pub optimize: bool,
}
#[derive(Debug, Clone, Serialize)]
pub struct ModelPlan {
    pub models: BTreeMap<String, ModelSelection>,
    pub table_variant: String,
    pub fallback_table_variant: Option<String>,
    pub layout_shape: String,
}
fn unsupported(message: impl Into<String>) -> Error {
    Error {
        code: ErrorCode::Unsupported,
        message: message.into(),
    }
}
fn text<'a>(v: &'a Value, key: &str, default: &'a str) -> &'a str {
    match v.get(key) {
        None => default,
        Some(value) => value.as_str().unwrap_or(""),
    }
}
fn option_path(v: &Value, key: &str) -> Option<PathBuf> {
    v.get(key).and_then(Value::as_str).map(PathBuf::from)
}
fn bool_option(v: &Value, key: &str, default: Option<bool>) -> Result<Option<bool>, Error> {
    match v.get(key) {
        None => Ok(default),
        Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        _ => Err(invalid(format!("{key} must be bool"))),
    }
}
fn integer(v: &Value, key: &str, default: i64) -> Result<i64, Error> {
    v.get(key).map_or(Ok(default), |n| {
        n.as_i64()
            .ok_or_else(|| invalid(format!("{key} must be an integer")))
    })
}
fn dynamic(v: &Value, key: &str, default: Option<usize>) -> Result<Option<usize>, Error> {
    match v.get(key) {
        None => Ok(default),
        Some(Value::Null) => Ok(None),
        Some(n) => n
            .as_u64()
            .filter(|&n| n <= i32::MAX as u64)
            .map(|n| Some(n as usize))
            .ok_or_else(|| invalid(format!("{key} must be a nonnegative int32"))),
    }
}
fn truth(v: Option<&Value>, default: bool) -> bool {
    match v {
        None => default,
        Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() != Some(0.),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}
fn compatible(a: &Value, b: &Value) -> bool {
    if a.is_number() && b.is_number() {
        return a.as_f64() == b.as_f64();
    }
    if let (Some(a), Some(b)) = (a.as_array(), b.as_array()) {
        return a.len() == b.len() && a.iter().zip(b).all(|(a, b)| compatible(a, b));
    }
    a == b
}
fn variant(name: &str) -> Result<&'static str, Error> {
    match name {
        "unet_slanet_plus" => Ok("unet-slanet-plus"),
        "slanet_plus" | "slanet-plus" => Ok("slanet-plus"),
        "unet" => Ok("unet"),
        "table_transformer" => Ok("table-transformer"),
        _ => Err(invalid(format!("Unsupported table model: {name}"))),
    }
}

impl ModelPlan {
    pub fn new(c: &Config) -> Result<Self, Error> {
        if c.cpu_dynamic_block_base
            .is_some_and(|n| n > i32::MAX as usize)
        {
            return Err(invalid(
                "cpu_dynamic_block_base must be a nonnegative int32",
            ));
        }
        let (l, o, f, t) = (
            &c.layout_config,
            &c.ocr_config,
            &c.formula_config,
            &c.table_config,
        );
        let allowed=[
            "model_type model_dir_or_path docvortex_mode layout_shape_mode conf_thresh markdown_ignore_labels engine_type engine_cfg iou_thresh",
            "engine_type engine_cfg Det.engine_type Rec.engine_type Det.model_path Rec.model_path use_det_mode seal_enable onnx_ops custom_model crop_sampling EngineConfig.onnxruntime.intra_op_num_threads EngineConfig.onnxruntime.inter_op_num_threads EngineConfig.onnxruntime.use_cuda EngineConfig.onnxruntime.cuda_ep_cfg.device_id EngineConfig.onnxruntime.gpu_id Det.limit_side_len Det.limit_type Det.thresh Det.box_thresh Det.unclip_ratio Det.use_dilation Det.mean Det.std",
            "model_type model_dir_or_path engine_type engine_cfg batch_num",
            "model_type model_dir_or_path engine_type engine_cfg unet.model_dir_or_path slanet_plus.model_dir_or_path cls.model_dir_or_path cls.model_type use_compare_table cls_conf_threshold enable_blank_cell_rec force_ocr use_rule_table rule_table_engine rule_table_score_threshold fallback_model_type fallback_model_dir_or_path fallback_text_loss_threshold fallback_to_text use_word_box",
        ];
        for (config, keys) in [l, o, f, t].iter().zip(allowed) {
            if !config.is_object() {
                return Err(invalid("model config must be an object"));
            }
            for key in config.as_object().unwrap().keys() {
                if !keys.split_whitespace().any(|k| k == key) {
                    return Err(unsupported(format!("Unsupported model option: {key}")));
                }
            }
        }
        // The production table route currently uses line boxes. Explicitly
        // selecting Python's false default must be equivalent to omitting it.
        // Do not silently pretend that character-level OCR boxes are enabled.
        if bool_option(t, "use_word_box", Some(false))? == Some(true) {
            return Err(unsupported(
                "use_word_box=true is not implemented in the native table pipeline",
            ));
        }
        let fixed = json!({"Det.limit_side_len":960,"Det.limit_type":"max","Det.thresh":0.3,"Det.box_thresh":0.3,"Det.unclip_ratio":1.8,"Det.use_dilation":true,"Det.mean":[0.485,0.456,0.406],"Det.std":[0.229,0.224,0.225]});
        for (key, default) in fixed.as_object().unwrap() {
            if let Some(value) = o.get(key) {
                if !compatible(value, default) {
                    return Err(unsupported(format!(
                        "{key} custom values are not supported"
                    )));
                }
            }
        }
        if !matches!(
            text(t, "rule_table_engine", "docvortex"),
            "docvortex" | "docvortex_ocr"
        ) {
            return Err(unsupported("Unsupported rule_table_engine"));
        }
        if text(t, "cls.model_type", "paddle_q_cls") != "paddle_q_cls" {
            return Err(unsupported("Unsupported table classifier"));
        }
        for key in ["Det.engine_type", "Rec.engine_type"] {
            if text(o, key, "onnxruntime") != "onnxruntime" {
                return Err(unsupported(format!("{key} requires ONNX Runtime")));
            }
        }
        let layout = text(l, "model_type", "pp_doclayoutv3");
        if !matches!(
            layout,
            "pp_doclayoutv3" | "pp_doclayoutv2" | "pp_doclayout_plus_l" | "docvortex_rule"
        ) {
            return Err(unsupported(format!("Unsupported layout model: {layout}")));
        }
        let formula = text(f, "model_type", "pp_formulanet_plus_m");
        if !matches!(
            formula,
            "pp_formulanet_plus_s" | "pp_formulanet_plus_m" | "pp_formulanet_plus_l"
        ) {
            return Err(unsupported(format!("Unsupported formula model: {formula}")));
        }
        let table = text(t, "model_type", "unet_slanet_plus");
        let table_variant = variant(table)?.into();
        if t.get("fallback_to_text").is_some_and(|v| !v.is_boolean()) {
            return Err(invalid("fallback_to_text must be bool"));
        }
        let fallback = t
            .get("fallback_model_type")
            .filter(|v| !v.is_null())
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| invalid("fallback_model_type must be a string"))
            })
            .transpose()?;
        let fallback_table_variant = fallback.map(variant).transpose()?.map(str::to_owned);
        let loss = t
            .get("fallback_text_loss_threshold")
            .unwrap_or(&Value::Null);
        if !loss.is_null()
            && !loss
                .as_f64()
                .is_some_and(|n| n.is_finite() && (0. ..=1.).contains(&n))
        {
            return Err(invalid(
                "fallback_text_loss_threshold must be between 0 and 1",
            ));
        }
        let layout_shape = text(l, "layout_shape_mode", "auto").to_owned();
        if !matches!(layout_shape.as_str(), "auto" | "rect" | "poly" | "quad") {
            return Err(invalid("Invalid layout_shape_mode"));
        }
        if c.device.starts_with("cuda:") && c.device != "cuda:0" {
            return Err(unsupported("CUDA device 0 only"));
        }
        for v in [l, o, f, t] {
            if v.get("engine_type")
                .is_some_and(|value| !value.is_null() && value.as_str() != Some("onnxruntime"))
            {
                return Err(unsupported("ONNX Runtime is required"));
            }
            let cfg = &v["engine_cfg"];
            if ["use_dml", "use_cann", "use_npu"]
                .iter()
                .any(|k| truth(cfg.get(k), false))
            {
                return Err(unsupported("DML/NPU is not supported"));
            }
        }
        let mut models = BTreeMap::new();
        let definitions = [
            ("layout_model", "layout", l),
            ("detector", "det", o),
            ("recognizer", "rec", o),
            ("seal_detector", "seal_det", o),
            ("seal_recognizer", "rec", o),
            ("formula_model", "formula", f),
            ("orientation", "orientation", t),
            (
                "table_model",
                if table == "unet" { "wired" } else { "table" },
                t,
            ),
            ("classifier", "classifier", t),
            ("secondary_classifier", "secondary_classifier", t),
            ("wired_model", "wired", t),
        ];
        for (name, role, config) in definitions {
            models.insert(name.into(), selection(c, config, name, role)?);
        }
        {
            let m = models.get_mut("layout_model").unwrap();
            m.path = option_path(l, "model_dir_or_path");
            if !matches!(layout, "pp_doclayoutv3" | "docvortex_rule") {
                m.model_id = layout.into();
            }
            let m = models.get_mut("formula_model").unwrap();
            m.path = option_path(f, "model_dir_or_path");
            if formula != "pp_formulanet_plus_m" {
                m.model_id = formula.into();
            }
            models.get_mut("detector").unwrap().path = option_path(o, "Det.model_path");
            models.get_mut("recognizer").unwrap().path = option_path(o, "Rec.model_path");
            models.get_mut("classifier").unwrap().path = option_path(t, "cls.model_dir_or_path");
            models.get_mut("wired_model").unwrap().path = option_path(t, "unet.model_dir_or_path");
            let m = models.get_mut("table_model").unwrap();
            m.path = if table == "unet" {
                option_path(t, "unet.model_dir_or_path")
                    .or_else(|| option_path(t, "model_dir_or_path"))
            } else {
                option_path(t, "slanet_plus.model_dir_or_path").or_else(|| {
                    if matches!(table, "slanet_plus" | "slanet-plus") {
                        option_path(t, "model_dir_or_path")
                    } else {
                        None
                    }
                })
            };
            if table == "table_transformer" {
                m.model_id = "table_transformer_v1_1_all".into();
                m.path = option_path(t, "model_dir_or_path");
            }
        }
        if let Some(table) = fallback {
            let mut m = selection(
                c,
                t,
                "fallback_table_model",
                if table == "unet" { "wired" } else { "table" },
            )?;
            m.path = option_path(t, "fallback_model_dir_or_path");
            if table == "table_transformer" {
                m.model_id = "table_transformer_v1_1_all".into();
            }
            if table == "unet_slanet_plus" {
                m.path = option_path(t, "slanet_plus.model_dir_or_path");
            }
            models.insert("fallback_table_model".into(), m);
        }
        Ok(Self {
            models,
            table_variant,
            fallback_table_variant,
            layout_shape,
        })
    }
    /// Only warm models that this configuration can actually request.
    pub fn warmup_roles(&self, r: &pipeline::Request) -> Vec<&str> {
        let mut roles = vec!["detector", "recognizer"];
        if !r.native_only {
            roles.push("layout_model");
        }
        if r.formula_enabled {
            roles.push("formula_model");
        }
        if r.table_enabled {
            roles.push("orientation");
            roles.push("table_model");
            for variant in
                std::iter::once(r.table_variant.as_str()).chain(r.fallback_table_variant.as_deref())
            {
                if variant.starts_with("unet-") {
                    roles.extend(["classifier", "secondary_classifier", "wired_model"]);
                }
            }
            if r.fallback_table_variant.is_some() {
                roles.push("fallback_table_model");
            }
        }
        if r.doc_orientation_classify && !roles.contains(&"orientation") {
            roles.push("orientation");
        }
        if r.seal_enabled {
            roles.extend(["seal_detector", "seal_recognizer"]);
        }
        roles.sort_unstable();
        roles.dedup();
        roles
    }
}

fn selection(c: &Config, config: &Value, name: &str, role: &str) -> Result<ModelSelection, Error> {
    let cfg = &config["engine_cfg"];
    let ocr = matches!(
        name,
        "detector" | "recognizer" | "seal_detector" | "seal_recognizer"
    );
    let default_threads = if name == "layout_model" || name == "formula_model" || ocr {
        -1
    } else {
        4
    };
    let raw_threads = if ocr
        && config
            .get("EngineConfig.onnxruntime.intra_op_num_threads")
            .is_some()
    {
        integer(config, "EngineConfig.onnxruntime.intra_op_num_threads", -1)?
    } else {
        integer(cfg, "intra_op_num_threads", default_threads)?
    };
    if raw_threads < -1 || raw_threads > i32::MAX as i64 {
        return Err(invalid(
            "intra_op_num_threads must be -1/0 or a positive int32",
        ));
    }
    if cfg.get("use_cuda").is_some_and(|v| !v.is_boolean()) {
        return Err(invalid("use_cuda must be bool"));
    }
    let mut cuda = bool_option(
        cfg,
        "use_cuda",
        Some(matches!(c.device.as_str(), "cuda" | "cuda:0")),
    )?
    .unwrap_or(false);
    let mut device_id = integer(cfg, "gpu_id", 0)?;
    if cfg.get("cuda_ep_cfg.device_id").is_some() {
        device_id = integer(cfg, "cuda_ep_cfg.device_id", 0)?;
    }
    if ocr {
        if config
            .get("EngineConfig.onnxruntime.use_cuda")
            .is_some_and(|v| !v.is_boolean())
        {
            return Err(invalid("use_cuda must be bool"));
        }
        cuda =
            bool_option(config, "EngineConfig.onnxruntime.use_cuda", Some(cuda))?.unwrap_or(false);
        if config.get("EngineConfig.onnxruntime.gpu_id").is_some() {
            device_id = integer(config, "EngineConfig.onnxruntime.gpu_id", 0)?;
        }
        if config
            .get("EngineConfig.onnxruntime.cuda_ep_cfg.device_id")
            .is_some()
        {
            device_id = integer(config, "EngineConfig.onnxruntime.cuda_ep_cfg.device_id", 0)?;
        }
    }
    if device_id < 0 {
        return Err(invalid("CUDA device_id must be a nonnegative integer"));
    }
    if cuda && device_id != 0 {
        return Err(unsupported("CUDA device 0 only"));
    }
    let mut optimize = true;
    if role == "wired" {
        optimize = truth(cfg.get("unet_conv_cache"), true);
    }
    if role == "formula" {
        optimize = truth(cfg.get("onnx_ops"), true);
    }
    if ocr {
        optimize = truth(config.get("onnx_ops"), true);
    }
    if name == "layout_model" {
        optimize = truth(cfg.get("mask_uint8"), true);
    }
    Ok(ModelSelection {
        role: role.into(),
        model_id: role.into(),
        path: None,
        threads: raw_threads.max(0) as usize,
        device: if cuda { "cuda" } else { "cpu" }.into(),
        cpu_spinning: bool_option(cfg, "cpu_spinning", c.cpu_spinning)?,
        cpu_dynamic_block_base: dynamic(cfg, "cpu_dynamic_block_base", c.cpu_dynamic_block_base)?,
        cpu_force_spinning_stop: bool_option(cfg, "cpu_force_spinning_stop", None)?,
        cpu_shared_arena: bool_option(cfg, "cpu_shared_arena", None)?,
        partition_table: if matches!(name, "table_model" | "fallback_table_model") {
            truth(cfg.get("slanet_partition"), true)
        } else {
            true
        },
        optimize,
    })
}

impl ModelSelection {
    pub fn resolve(&self, resources: &Resources) -> Result<(Descriptor, bool), Error> {
        self.resolve_progress(resources, &mut |_| Ok(()))
    }
    pub fn resolve_progress(
        &self,
        resources: &Resources,
        progress: &mut dyn FnMut(
            &rapiddoc_inference::model_store::DownloadEvent,
        ) -> Result<(), Error>,
    ) -> Result<(Descriptor, bool), Error> {
        let catalog = Catalog::bundled().map_err(store_error)?;
        let mut callback_error = None;
        let resolved = catalog.resolve_progress(
            &self.model_id,
            resources.model_directory.as_deref(),
            self.path.as_deref(),
            resources.allow_download,
            &mut |event| {
                progress(event).map_err(|error| {
                    let message = error.to_string();
                    callback_error = Some(error);
                    rapiddoc_inference::model_store::StoreError::new(
                        rapiddoc_inference::model_store::StoreErrorKind::Io,
                        message,
                    )
                })
            },
        );
        if let Some(error) = callback_error {
            return Err(error);
        }
        let resolved = resolved.map_err(store_error)?;
        let optimization = self.optimization(&resolved.sha256)?;
        let operator_libraries = if matches!(
            optimization,
            Some(
                rapiddoc_inference::onnx::Optimization::OcrDetection
                    | rapiddoc_inference::onnx::Optimization::OcrRecognition
                    | rapiddoc_inference::onnx::Optimization::Unet
                    | rapiddoc_inference::onnx::Optimization::Formula
            )
        ) {
            vec![resources.locate(Resource::Operators)?]
        } else {
            vec![]
        };
        Ok((
            Descriptor {
                runtime: resources.locate(Resource::Ort)?,
                model: resolved.path,
                sha256: resolved.sha256,
                device: self.device.clone(),
                threads: self.threads,
                max_elements: isize::MAX as usize / 4,
                cpu_spinning: self.cpu_spinning,
                cpu_dynamic_block_base: self.cpu_dynamic_block_base,
                cpu_force_spinning_stop: self.cpu_force_spinning_stop,
                cpu_shared_arena: self.cpu_shared_arena,
                operator_libraries,
                optimization,
            },
            resolved.downloaded,
        ))
    }
    pub fn optimization(
        &self,
        sha: &str,
    ) -> Result<Option<rapiddoc_inference::onnx::Optimization>, Error> {
        if self.device != "cuda"
            || !self.optimize
            || (self.role == "table" && !self.partition_table)
        {
            return Ok(None);
        }
        let catalog = Catalog::bundled().map_err(store_error)?;
        let (_, original) = catalog.model(&self.role).map_err(store_error)?;
        if sha != original.sha256 {
            return Ok(None);
        }
        use rapiddoc_inference::onnx::Optimization as O;
        Ok(match self.role.as_str() {
            "det" | "seal_det" => Some(O::OcrDetection),
            "rec" => Some(O::OcrRecognition),
            "layout" => Some(O::Layout),
            "table" => Some(O::Slanet),
            "wired" => Some(O::Unet),
            "formula" => Some(O::Formula),
            _ => None,
        })
    }
}
