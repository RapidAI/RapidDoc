//! One configuration/default/alias policy for all supported host languages.
use super::*;
fn object() -> Value {
    json!({})
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub layout_config: Value,
    pub ocr_config: Value,
    pub formula_config: Value,
    pub table_config: Value,
    pub image_config: Value,
    pub parse_method: String,
    pub formula_enable: bool,
    pub table_enable: bool,
    pub lang: String,
    pub make_md_mode: String,
    pub pdf_pages_batch: usize,
    pub cpu_spinning: Option<bool>,
    pub cpu_dynamic_block_base: Option<usize>,
    pub device: String,
    pub doc_orientation_classify: bool,
    pub crop_sampling: rapiddoc_inference::crop::Sampling,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            layout_config: object(),
            ocr_config: object(),
            formula_config: object(),
            table_config: object(),
            image_config: object(),
            parse_method: "auto".into(),
            formula_enable: true,
            table_enable: true,
            lang: "ch".into(),
            make_md_mode: "mm_markdown".into(),
            pdf_pages_batch: 64,
            cpu_spinning: None,
            cpu_dynamic_block_base: None,
            device: std::env::var("MINERU_DEVICE_MODE").unwrap_or_else(|_| "cpu".into()),
            doc_orientation_classify: std::env::var("USE_DOC_ORIENTATION_CLASSIFY").is_ok_and(
                |v| {
                    matches!(
                        v.trim().to_lowercase().as_str(),
                        "true" | "1" | "yes" | "on"
                    )
                },
            ),
            crop_sampling: Default::default(),
        }
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParseOptions {
    pub start_page: usize,
    /// Exclusive upper bound. Python converts its inclusive end_page_id once.
    pub end_page: Option<usize>,
    pub parse_method: Option<String>,
    pub formula_enable: Option<bool>,
    pub table_enable: Option<bool>,
    pub lang: Option<String>,
    pub image_dir_name: Option<String>,
    pub doc_orientation_classify: Option<bool>,
    pub draw_layout: bool,
    pub draw_span: bool,
    pub draw_line_sort: bool,
    pub empty_format: Option<String>,
    pub latex_delimiters: Option<rapiddoc_office::document_output::inline::Delimiters>,
}
impl Config {
    pub fn layout_model(mut self, name: impl Into<String>) -> Self {
        self.layout_config["model_type"] = Value::String(name.into());
        self
    }
    pub fn formula_model(mut self, name: impl Into<String>) -> Self {
        self.formula_config["model_type"] = Value::String(name.into());
        self
    }
    pub fn table_model(mut self, name: impl Into<String>) -> Self {
        self.table_config["model_type"] = Value::String(name.into());
        self
    }
    pub fn model_plan(&self) -> Result<ModelPlan, Error> {
        ModelPlan::new(self)
    }
    pub fn request(
        &self,
        options: &ParseOptions,
        plan: &ModelPlan,
    ) -> Result<pipeline::Request, Error> {
        if self.pdf_pages_batch == 0 {
            return Err(invalid("pdf_pages_batch must be positive"));
        }
        let (l, o, f, t, i) = (
            &self.layout_config,
            &self.ocr_config,
            &self.formula_config,
            &self.table_config,
            &self.image_config,
        );
        let method = options
            .parse_method
            .as_deref()
            .unwrap_or(&self.parse_method);
        if !matches!(method, "auto" | "txt" | "ocr") {
            return Err(invalid("parse_method must be auto/txt/ocr"));
        }
        let layout = l["model_type"].as_str().unwrap_or("pp_doclayoutv3");
        let native_only = layout == "docvortex_rule";
        let mode = l["docvortex_mode"]
            .as_str()
            .unwrap_or(if native_only { "rule" } else { "off" });
        if !matches!(mode, "off" | "rule" | "rule_plus_s") || native_only && mode == "off" {
            return Err(invalid("Invalid docvortex_mode"));
        }
        if mode == "rule_plus_s" {
            return Err(Error {
                code: ErrorCode::Unsupported,
                message: "rule_plus_s is deferred".into(),
            });
        }
        let mut request = json!({
            "start_page":options.start_page,"end_page":options.end_page,"window":self.pdf_pages_batch,
            "doc_orientation_classify":options.doc_orientation_classify.unwrap_or(self.doc_orientation_classify),"docvortex_mode":mode,"native_only":native_only,
            "native_text":method!="ocr","auto_classify":method=="auto","clamp_page_range":true,
            "formula_enabled":options.formula_enable.unwrap_or(self.formula_enable),"formula_batch_size":f.get("batch_num").unwrap_or(&json!(1)),
            "table_enabled":options.table_enable.unwrap_or(self.table_enable),"seal_enabled":o.get("seal_enable").unwrap_or(&json!(true)),
            "lang":options.lang.as_deref().unwrap_or(&self.lang),"table_variant":plan.table_variant,
            "layout_shape":plan.layout_shape,"layout_model_type":layout,
            "rule_table_engine":t.get("rule_table_engine").unwrap_or(&json!("docvortex")),
            "use_rule_table":t.get("use_rule_table").and_then(Value::as_bool).unwrap_or(true)&&!t["force_ocr"].as_bool().unwrap_or(false),
            "use_compare_table":t.get("use_compare_table").and_then(Value::as_bool).unwrap_or(false),
            "cls_conf_threshold":t.get("cls_conf_threshold").unwrap_or(&json!(0.9)),
            "rule_table_score_threshold":t.get("rule_table_score_threshold").unwrap_or(&json!(0.9)),
            "fallback_table_variant":plan.fallback_table_variant,
            "fallback_text_loss_threshold":t.get("fallback_text_loss_threshold").unwrap_or(&json!(0.2)),
            "fallback_to_text":t.get("fallback_to_text").unwrap_or(&json!(false)),
            "threshold":l.get("conf_thresh").filter(|v|v.as_f64().is_some_and(|n|n!=0.)).unwrap_or(&json!(0.3)),
            "markdown_ignore_labels":l.get("markdown_ignore_labels"),"use_det_mode":o.get("use_det_mode").unwrap_or(&json!("auto")),
            "table_force_ocr":t.get("force_ocr").and_then(Value::as_bool).unwrap_or(false),
            "extract_original_image":i.get("extract_original_image").unwrap_or(&json!(false)),
            "extract_original_image_iou_thresh":i.get("extract_original_image_iou_thresh").unwrap_or(&json!(0.9)),
            "crop_sampling":o.get("crop_sampling").cloned().unwrap_or(serde_json::to_value(self.crop_sampling).unwrap()),
        });
        // A single serde boundary validates numeric and boolean fields for every
        // host. No host-specific default table or model alias implementation.
        request["dpi"] = json!(200.);
        serde_json::from_value(request).map_err(|e| invalid(e.to_string()))
    }
}
