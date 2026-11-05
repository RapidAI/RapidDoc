//! Production PDF model scheduling. No language-runtime objects or callbacks for algorithms.
use rapiddoc_core::{Error, ErrorCode, Limits, PageRange};
use rapiddoc_inference::registry::{Descriptor, Models};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};
mod ocr;
mod export;
mod orientation;
mod seals;
mod table_fallback;
mod tables;
type Images = BTreeMap<String, Arc<Vec<u8>>>;
#[derive(Default, Clone)]
struct Timings(BTreeMap<&'static str, (std::time::Duration, usize)>);
impl Timings {
    fn duration(&self, stage: &str) -> std::time::Duration {
        self.0.get(stage).map(|v| v.0).unwrap_or_default()
    }
    fn since(&self, base: &Self, stage: &str) -> f64 {
        self.duration(stage)
            .saturating_sub(base.duration(stage))
            .as_secs_f64()
    }
    fn phase(
        &mut self,
        stage: &'static str,
        start: std::time::Instant,
        loading: std::time::Duration,
        items: usize,
    ) {
        let entry = self.0.entry(stage).or_default();
        entry.0 += start.elapsed().saturating_sub(loading);
        entry.1 += items;
    }
    fn add(&mut self, stage: &'static str, start: std::time::Instant, items: usize) {
        let entry = self.0.entry(stage).or_default();
        entry.0 += start.elapsed();
        entry.1 += items;
    }
    fn rows(&self) -> Vec<Value> {
        self.0.iter().map(|(stage,(elapsed,items))|json!({"stage":stage,"seconds":elapsed.as_secs_f64(),"items":items})).collect()
    }
}
type Resolver<'a> = dyn FnMut(&str) -> Result<Descriptor, Error> + 'a;
#[derive(Clone)]
pub struct Engine {
    models: Models,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Request {
    pub start_page: usize,
    pub end_page: Option<usize>,
    pub clamp_page_range: bool,
    pub window: usize,
    pub dpi: f64,
    pub native_text: bool,
    pub auto_classify: bool,
    pub formula_enabled: bool,
    pub formula_batch_size: usize,
    pub table_enabled: bool,
    pub seal_enabled: bool,
    pub doc_orientation_classify: bool,
    pub layout_shape: String,
    pub layout_model_type: String,
    pub rule_table_engine: String,
    pub threshold: f32,
    pub docvortex_mode: String,
    pub native_only: bool,
    pub lang: Option<String>,
    pub use_det_mode: String,
    pub table_variant: String,
    pub fallback_table_variant: Option<String>,
    pub fallback_to_text: bool,
    pub fallback_text_loss_threshold: f64,
    pub use_rule_table: bool,
    pub use_compare_table: bool,
    pub cls_conf_threshold: f64,
    pub rule_table_score_threshold: f64,
    pub table_force_ocr: bool,
    pub extract_original_image: bool,
    pub extract_original_image_iou_thresh: f64,
    pub markdown_ignore_labels: Option<Vec<String>>,
    pub geos_library: Option<PathBuf>,
    pub crop_sampling: rapiddoc_inference::crop::Sampling,
}
impl Default for Request {
    fn default() -> Self {
        Self {
            start_page: 0,
            end_page: None,
            clamp_page_range: false,
            window: 16,
            dpi: 200.,
            native_text: true,
            auto_classify: false,
            formula_enabled: true,
            formula_batch_size: 1,
            table_enabled: true,
            seal_enabled: true,
            doc_orientation_classify: false,
            layout_shape: "auto".into(),
            layout_model_type: "pp_doclayoutv3".into(),
            rule_table_engine: "docvortex".into(),
            threshold: 0.3,
            docvortex_mode: "off".into(),
            native_only: false,
            lang: Some("ch".into()),
            use_det_mode: "auto".into(),
            table_variant: "unet-slanet-plus".into(),
            fallback_table_variant: None,
            fallback_to_text: false,
            fallback_text_loss_threshold: 0.2,
            use_rule_table: true,
            use_compare_table: false,
            cls_conf_threshold: 0.9,
            rule_table_score_threshold: 0.9,
            table_force_ocr: false,
            extract_original_image: false,
            extract_original_image_iou_thresh: 0.9,
            markdown_ignore_labels: None,
            geos_library: None,
            crop_sampling: Default::default(),
        }
    }
}
#[derive(Serialize)]
pub struct Progress<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<usize>,
    pub stage: &'a str,
    pub page_index: Option<usize>,
    pub detail: Option<String>,
}
impl<'a> Progress<'a> {
    fn elapsed(mut self, seconds: f64) -> Self {
        self.elapsed_seconds = Some(seconds);
        self
    }
    fn items(stage: &'a str, completed: usize, total: usize) -> Self {
        Self {
            stage,
            completed: Some(completed),
            elapsed_seconds: None,
            total: Some(total),
            page_index: None,
            detail: None,
        }
    }
}
pub struct Output {
    pub metadata: Value,
    pub assets: Vec<crate::legacy_page::assets::Asset>,
}
pub(super) struct Page {
    index: usize,
    facts: Value,
    image: rapiddoc_pdf::render::RenderedPage,
    original_image: Option<rapiddoc_pdf::render::RenderedPage>,
    orientation_label: Option<String>,
    plan: Value,
    images: Images,
    formula_pixels: BTreeMap<String, rapiddoc_core::pillow_resize::Rgb>,
    recognition_pixels: BTreeMap<String, rapiddoc_inference::recognition_batch::Image>,
    ocr_model_prefix: usize,
}
#[derive(Default)]
struct RecognitionMemory {
    images: usize,
    pixels: usize,
}
pub(super) fn error(message: impl ToString) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.to_string(),
    }
}
pub(super) fn value<T: Serialize>(v: &T) -> Result<Value, Error> {
    serde_json::to_value(v).map_err(error)
}
// Explicit developer capture only; no inputs or intermediate pages are saved by default.
pub(super) fn parity_trace(name: &str, data: &Value) -> Result<(), Error> {
    if let Some(folder) = std::env::var_os("RAPIDDOC_PARITY_TRACE") {
        let folder = PathBuf::from(folder);
        std::fs::create_dir_all(&folder).map_err(error)?;
        std::fs::write(folder.join(name), serde_json::to_vec(data).map_err(error)?)
            .map_err(error)?;
    }
    Ok(())
}
/// Preserve the former JSON bridge's FP32 promotion wherever its exact decimal
/// values are part of the legacy geometry contract. Full masks never use this.
pub(super) fn wire<T: Serialize>(v: &T) -> Result<Value, Error> {
    serde_json::from_str(&serde_json::to_string(v).map_err(error)?).map_err(error)
}
pub(super) fn op(name: &str, v: Value) -> Result<Value, Error> {
    crate::operations::run(name, v)
}
pub(super) fn array(v: &Value) -> Result<&Vec<Value>, Error> {
    v.as_array().ok_or_else(|| error("Expected pipeline array"))
}
pub(super) fn image<'a>(images: &'a Images, key: &Value) -> Result<&'a Arc<Vec<u8>>, Error> {
    images
        .get(key.as_str().ok_or_else(|| error("Invalid image key"))?)
        .ok_or_else(|| error("Pipeline image is missing"))
}
impl Engine {
    pub fn retain_model_roles(&mut self, roles: &[&str]) {
        self.models.retain_roles(roles);
    }

    pub fn warmup_models(
        &mut self,
        roles: &[&str],
        resolve: &mut Resolver<'_>,
    ) -> Result<(), Error> {
        for &role in roles {
            self.models.with_lazy(role, resolve, |_| Ok(()))?;
        }
        Ok(())
    }
    pub fn loaded_model_roles(&self) -> Vec<String> {
        self.models.loaded_roles()
    }
    pub fn new(models: BTreeMap<String, Descriptor>) -> Self {
        Self {
            models: Models::new(models),
        }
    }
    pub fn parse(
        &mut self,
        library: &Path,
        data: &[u8],
        request: &Request,
        resolve: &mut Resolver<'_>,
        progress: &mut dyn FnMut(Progress<'_>) -> Result<(), Error>,
    ) -> Result<Output, Error> {
        crate::legacy_page::with_host_page_budget(|| {
            rapiddoc_inference::crop::with_sampling(request.crop_sampling, || {
                self.parse_inner(library, data, request, resolve, progress)
            })
        })
    }
    fn parse_inner(
        &mut self,
        library: &Path,
        data: &[u8],
        r: &Request,
        resolve: &mut Resolver<'_>,
        progress: &mut dyn FnMut(Progress<'_>) -> Result<(), Error>,
    ) -> Result<Output, Error> {
        if r.layout_model_type == "pp_doclayout_plus_l" {
            return Err(Error{code:ErrorCode::Unsupported,message:"Plus-L detection is available; complete PDF parsing requires the deferred XY-cut reading order. Use pp_doclayoutv2 or pp_doclayoutv3 for PDF parsing.".into()});
        }
        if r.window == 0
            || r.window > rapiddoc_core::resources::limit(256)
            || r.formula_batch_size == 0
            || !r.dpi.is_finite()
            || !(1. ..=600.).contains(&r.dpi)
            || !matches!(r.docvortex_mode.as_str(), "off" | "rule")
            || r.native_only && r.docvortex_mode == "off"
            || !r.threshold.is_finite()
            || !(0. ..=1.).contains(&r.threshold)
            || ![
                r.cls_conf_threshold,
                r.fallback_text_loss_threshold,
                r.rule_table_score_threshold,
                r.extract_original_image_iou_thresh,
            ]
            .iter()
            .all(|v| v.is_finite() && (0. ..=1.).contains(v))
            || !matches!(r.use_det_mode.as_str(), "auto" | "txt" | "ocr")
            || !matches!(r.layout_shape.as_str(), "auto" | "rect" | "poly" | "quad")
            || !matches!(
                r.layout_model_type.as_str(),
                "pp_doclayoutv3" | "pp_doclayoutv2" | "pp_doclayout_plus_l" | "docvortex_rule"
            )
            || !matches!(r.rule_table_engine.as_str(), "docvortex" | "docvortex_ocr")
            || r.fallback_table_variant.as_deref().is_some_and(|v| {
                !matches!(
                    v,
                    "unet" | "slanet-plus" | "unet-slanet-plus" | "table-transformer"
                )
            })
            || !matches!(
                r.table_variant.as_str(),
                "unet"
                    | "slanet-plus"
                    | "slanet-1m"
                    | "unet-slanet-plus"
                    | "unet-slanet-1m"
                    | "table-transformer"
            )
        {
            return Err(error("Invalid PDF pipeline options"));
        }
        let mut timings = Timings::default();
        let initialization_base = self.models.initialization_time();
        let limits = Limits::pdf_host();
        let count = rapiddoc_pdf::page_count(library, data, None, &limits)?;
        progress(Progress {
            stage: "document_opened",
            detail: None,
            total: Some(count),
            elapsed_seconds: None,
            completed: None,
            page_index: None,
        })?;
        let mut normalized;
        let r = if r.clamp_page_range {
            if r.start_page >= count && count > 0 {
                return Err(error("start_page exceeds source page count"));
            }
            normalized = r.clone();
            normalized.end_page = Some(r.end_page.unwrap_or(count).min(count));
            &normalized
        } else {
            r
        };
        let range = PageRange {
            start: r.start_page,
            end: r.end_page,
        }
        .resolve(count)?;
        let mut all_pages = vec![];
        let mut assets = Vec::<crate::legacy_page::assets::Asset>::new();
        let mut asset_indices = BTreeMap::<String, usize>::new();
        let mut native_routes = vec![];
        let mut layout_routes = vec![];
        for first in (range.start..range.end).step_by(r.window) {
            let batch_base = timings.clone();
            let last = range.end.min(first + r.window);
            // The Python public API classifies each sliced processing window.
            // A final electronic-only window can differ from the mixed PDF.
            let mut batch_request = r.clone();
            if r.auto_classify {
                let classified =
                    rapiddoc_pdf::classification::classify(library, data, first, Some(last));
                batch_request.native_text = classified["method"] == "txt";
                if let Some(error) = classified["error"].as_str() {
                    progress(Progress {
                        completed: None,
                        elapsed_seconds: None,
                        total: None,
                        stage: "classification_fallback",
                        page_index: Some(first),
                        detail: Some(error.to_owned()),
                    })?;
                }
            }
            let r = &batch_request;
            let mut pages = vec![];
            progress(Progress {
                completed: None,
                elapsed_seconds: None,
                total: None,
                stage: "batch",
                page_index: None,
                detail: Some(format!(
                    "Batch {}/{}: {} pages/{} pages ({}-{}, {})",
                    (first - range.start) / r.window + 1,
                    range.len().div_ceil(r.window),
                    last - first,
                    range.len(),
                    first + 1,
                    last,
                    if r.native_text { "txt" } else { "ocr" }
                )),
            })?;
            let native_started = std::time::Instant::now();
            progress(Progress::items("layout", 0, last - first).elapsed(0.))?;
            let native = if r.docvortex_mode != "off" && !(first..last).is_empty() {
                progress(Progress {
                    completed: None,
                    elapsed_seconds: None,
                    total: None,
                    stage: "native_analysis",
                    page_index: Some(first),
                    detail: Some(format!("pages {}-{} of {}", first + 1, last, count)),
                })?;
                match crate::legacy_model::native_pdf_document::predict_range(
                    library,
                    data,
                    None,
                    16_000_000,
                    PageRange {
                        start: first,
                        end: Some(last),
                    },
                ) {
                    Ok(mut v) => Some(v["pages"].take()),
                    Err(e) if !r.native_only => {
                        progress(Progress {
                            completed: None,
                            elapsed_seconds: None,
                            total: None,
                            stage: "layout_fallback",
                            page_index: None,
                            detail: Some(e.to_string()),
                        })?;
                        None
                    }
                    Err(e) => return Err(e),
                }
            } else {
                None
            };
            if r.docvortex_mode != "off" {
                timings.add("native_analysis", native_started, last - first);
            }
            for index in first..last {
                progress(Progress {
                    completed: None,
                    elapsed_seconds: None,
                    total: None,
                    stage: "render",
                    page_index: Some(index),
                    detail: None,
                })?;
                let phase_started = std::time::Instant::now();
                let mut facts = value(&rapiddoc_pdf::legacy_chars::read_page(
                    library, data, index, None, true, &limits,
                )?)?;
                let (mut rendered, mut pixels) =
                    rapiddoc_pdf::render::render_model_page_with_pixels(
                        library, data, index, r.dpi, 16_000_000, None, &limits,
                    )?;
                timings.add("render", phase_started, 1);
                let mut original_image = None;
                let mut orientation_label = None;
                if r.doc_orientation_classify {
                    progress(
                        Progress::items("orientation", index - first, last - first)
                            .elapsed(timings.since(&batch_base, "orientation")),
                    )?;
                    progress(Progress {
                        completed: None,
                        elapsed_seconds: None,
                        total: None,
                        stage: "orientation",
                        page_index: Some(index),
                        detail: None,
                    })?;
                    let started = std::time::Instant::now();
                    let loading = self.models.initialization_time();
                    let label = if rendered.height as f64 / rendered.width as f64 > 1.2 {
                        self.models
                            .with_lazy("orientation", resolve, |m| {
                                m.classify_orientation_png(&rendered.png)
                            })?
                            .rotation_degrees
                            .to_string()
                    } else {
                        "0".to_owned()
                    };
                    if matches!(label.as_str(), "90" | "270") {
                        let rotated = orientation::rotate(&rendered, &label)?;
                        pixels = image::load_from_memory(&rotated.png)
                            .map_err(error)?
                            .into_rgb8()
                            .into_raw();
                        original_image = Some(std::mem::replace(&mut rendered, rotated));
                    }
                    facts["rotate_label"] = json!(label);
                    orientation_label = Some(label.clone());
                    timings.phase(
                        "orientation",
                        started,
                        self.models.initialization_time() - loading,
                        1,
                    );
                    progress(
                        Progress::items("orientation", index - first + 1, last - first)
                            .elapsed(timings.since(&batch_base, "orientation")),
                    )?;
                    progress(Progress {
                        completed: None,
                        elapsed_seconds: None,
                        total: None,
                        stage: "orientation",
                        page_index: Some(index),
                        detail: Some(format!("label={label}")),
                    })?;
                }
                let phase_started = std::time::Instant::now();
                let loading = self.models.initialization_time();
                progress(Progress {
                    completed: None,
                    elapsed_seconds: None,
                    total: None,
                    stage: "layout",
                    page_index: Some(index),
                    detail: None,
                })?;
                let (model, route) = self.layout(
                    &rendered,
                    &pixels,
                    &facts,
                    native.as_ref().and_then(|p| p.get(index - first)),
                    r,
                    resolve,
                )?;
                drop(pixels);
                timings.phase(
                    "layout",
                    phase_started,
                    self.models.initialization_time() - loading,
                    1,
                );
                if let Some(route) = route {
                    layout_routes.push(json!({"page_index":index,"method":route["method"],"reason":route["reason"]}));
                }
                pages.push(Page {
                    index,
                    facts,
                    image: rendered,
                    original_image,
                    orientation_label,
                    plan: json!({"model":model}),
                    formula_pixels: BTreeMap::new(),
                    recognition_pixels: BTreeMap::new(),
                    ocr_model_prefix: 0,
                    images: Images::new(),
                });
                progress(
                    Progress::items("layout", index - first + 1, last - first)
                        .elapsed(timings.since(&batch_base, "layout")),
                )?;
            }
            // Native blocks have already been converted into this window's layout.
            // Release native HTML/images before OCR, formula and table inference.
            drop(native);
            // Match BatchAnalyze's phase order within the bounded page window.
            // Keep one model active across pages instead of alternating layout,
            // formula and detection sessions for every page. Only the current
            // A small formula batch may borrow adjacent pages. Keep at most two
            // pages of formula crops, then release them before preparing more.
            let formula_window = if r.formula_batch_size > 1 { 2 } else { 1 };
            let formula_total = if r.formula_enabled {
                pages
                    .iter()
                    .map(|p| {
                        p.plan["model"]["layout_dets"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter(|d| matches!(d["category_id"].as_u64(), Some(8 | 13 | 14)))
                            .count()
                    })
                    .sum()
            } else {
                0
            };
            let mut formula_done = 0;
            if formula_total > 0 {
                progress(Progress::items("formula", 0, formula_total).elapsed(0.))?;
            }
            for group in pages.chunks_mut(formula_window) {
                for page in group.iter_mut() {
                    let index = page.index;
                    let phase_started = std::time::Instant::now();
                    let mut crop = crate::legacy_model::crops::prepare_with_formula_pixels(
                        &json!({"model":page.plan["model"],"page_index":index,"formula_bbox_expand_px":2}),
                        &page.image.png,
                    )?;
                    rapiddoc_inference::crop_mask::apply(&crop.plan, &mut crop.images, 2)?;
                    timings.add(
                        "region_crops",
                        phase_started,
                        crop.images.len() + crop.formula_pixels.len(),
                    );
                    if !r.table_enabled {
                        crop.plan["tables"] = json!([])
                    }
                    page.plan = crop.plan;
                    page.formula_pixels = crop.formula_pixels.into_iter().collect();
                    page.images = crop
                        .images
                        .into_iter()
                        .map(|(k, v)| (k, Arc::new(v)))
                        .collect();
                    progress(Progress {
                        completed: None,
                        elapsed_seconds: None,
                        total: None,
                        stage: "formula",
                        page_index: Some(index),
                        detail: None,
                    })?;
                }
                let phase_started = std::time::Instant::now();
                let loading = self.models.initialization_time();
                let formula_elapsed = timings.since(&batch_base, "formula");
                if r.formula_enabled {
                    match self.formulas(
                        group,
                        r.formula_batch_size,
                        resolve,
                        &mut timings,
                        &mut |done, total, seconds| {
                            progress(
                                Progress::items(
                                    "formula",
                                    formula_done + done,
                                    formula_total.max(formula_done + total),
                                )
                                .elapsed(formula_elapsed + seconds),
                            )
                        },
                    ) {
                        Err(e) if group.len() > 1 && e.code == ErrorCode::ResourceLimit => {
                            // Aggregating two otherwise valid pages must not
                            // introduce a new resource-limit failure.
                            let mut fallback_done = 0;
                            let mut fallback_seconds = 0.;
                            for page in group.iter_mut() {
                                self.formulas(
                                    std::slice::from_mut(page),
                                    r.formula_batch_size,
                                    resolve,
                                    &mut timings,
                                    &mut |done, total, seconds| {
                                        progress(
                                            Progress::items(
                                                "formula",
                                                formula_done + fallback_done + done,
                                                formula_total
                                                    .max(formula_done + fallback_done + total),
                                            )
                                            .elapsed(formula_elapsed + fallback_seconds + seconds),
                                        )
                                    },
                                )?;
                                fallback_done += array(&page.plan["formulas"])?.len();
                                fallback_seconds = phase_started
                                    .elapsed()
                                    .saturating_sub(self.models.initialization_time() - loading)
                                    .as_secs_f64();
                            }
                        }
                        result => result?,
                    }
                }
                formula_done = group
                    .iter()
                    .map(|p| array(&p.plan["formulas"]).map(Vec::len))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .sum::<usize>()
                    + formula_done;
                for page in group.iter_mut() {
                    page.formula_pixels.clear();
                }
                timings.phase(
                    "formula",
                    phase_started,
                    self.models.initialization_time() - loading,
                    if r.formula_enabled {
                        group
                            .iter()
                            .map(|page| array(&page.plan["formulas"]).map(Vec::len))
                            .collect::<Result<Vec<_>, _>>()?
                            .into_iter()
                            .sum()
                    } else {
                        0
                    },
                );
            }
            let mut recognition_memory = RecognitionMemory::default();
            progress(Progress::items("ocr_detection", 0, pages.len()).elapsed(0.))?;
            for page in &mut pages {
                let index = page.index;
                let phase_started = std::time::Instant::now();
                let loading = self.models.initialization_time();
                progress(Progress {
                    completed: None,
                    elapsed_seconds: None,
                    total: None,
                    stage: "ocr_detection",
                    page_index: Some(index),
                    detail: None,
                })?;
                self.prepare_ocr(page, r, resolve, &mut timings, &mut recognition_memory)?;
                timings.phase(
                    "ocr_detection_and_crops",
                    phase_started,
                    self.models.initialization_time() - loading,
                    array(&page.plan["detect"])?.len(),
                );
                progress(
                    Progress::items("ocr_detection", index - first + 1, last - first)
                        .elapsed(timings.since(&batch_base, "ocr_detection_and_crops")),
                )?;
                progress(Progress {
                    completed: None,
                    elapsed_seconds: None,
                    total: None,
                    stage: "page_prepared",
                    page_index: Some(index),
                    detail: Some(format!(
                        "layout={} formulas={} detections={} recognition_crops={}",
                        array(&page.plan["model"]["layout_dets"])?.len(),
                        array(&page.plan["formulas"])?.len(),
                        array(&page.plan["detect"])?.len(),
                        array(&page.plan["recognition"])?.len()
                    )),
                })?;
            }
            progress(Progress {
                completed: None,
                elapsed_seconds: None,
                total: None,
                stage: "ocr_recognition",
                page_index: None,
                detail: None,
            })?;
            let phase_started = std::time::Instant::now();
            let recognition_count = pages
                .iter()
                .map(|p| p.plan["recognition"].as_array().map(Vec::len).unwrap_or(0))
                .sum();
            self.recognize_body(&mut pages, resolve, progress)?;
            timings.add("ocr_recognition", phase_started, recognition_count);
            let mut assembled = vec![];
            let mut post_images = Images::new();
            let table_total = pages
                .iter()
                .map(|p| array(&p.plan["tables"]).map(Vec::len))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .sum();
            let mut table_done = 0;
            let seal_total = if r.seal_enabled {
                pages
                    .iter()
                    .map(|p| {
                        p.plan["model"]["layout_dets"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter(|d| d["original_label"] == "seal")
                            .count()
                    })
                    .sum()
            } else {
                0
            };
            let mut seal_done = 0;
            if table_total > 0 {
                progress(Progress::items("table", 0, table_total).elapsed(0.))?;
            }
            for mut page in pages {
                progress(Progress {
                    completed: None,
                    elapsed_seconds: None,
                    total: None,
                    stage: "table",
                    page_index: Some(page.index),
                    detail: None,
                })?;
                let phase_started = std::time::Instant::now();
                let loading = self.models.initialization_time();
                let table_elapsed = timings.since(&batch_base, "table");
                let table_count = array(&page.plan["tables"])?.len();
                let fills = self.tables(
                    library,
                    data,
                    &mut page,
                    r,
                    resolve,
                    &mut native_routes,
                    &mut timings,
                    &mut |done, _, seconds| {
                        progress(
                            Progress::items("table", table_done + done, table_total)
                                .elapsed(table_elapsed + seconds),
                        )
                    },
                )?;
                table_done += table_count;
                timings.phase(
                    "table",
                    phase_started,
                    self.models.initialization_time() - loading,
                    table_count,
                );
                if table_total > 0 {
                    progress(
                        Progress::items("table", table_done, table_total)
                            .elapsed(timings.since(&batch_base, "table")),
                    )?;
                }
                if r.seal_enabled {
                    let started = std::time::Instant::now();
                    let loading = self.models.initialization_time();
                    let seal_elapsed = timings.since(&batch_base, "seal");
                    self.seals(
                        &mut page,
                        r,
                        resolve,
                        &mut |mut event| {
                            if let Some(done) = event.completed {
                                event.completed = Some(seal_done + done);
                                event.total = Some(seal_total);
                                event.elapsed_seconds =
                                    Some(seal_elapsed + event.elapsed_seconds.unwrap_or(0.));
                            }
                            progress(event)
                        },
                        &mut timings,
                    )?;
                    timings.phase(
                        "seal",
                        started,
                        self.models.initialization_time() - loading,
                        1,
                    );
                    seal_done += page.plan["model"]["layout_dets"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|d| d["original_label"] == "seal")
                        .count();
                    if seal_total > 0 {
                        progress(
                            Progress::items("seal", seal_done, seal_total)
                                .elapsed(timings.since(&batch_base, "seal")),
                        )?;
                    }
                }
                if let Some(label) = &page.orientation_label {
                    let original = page.original_image.as_ref().unwrap_or(&page.image);
                    orientation::restore(
                        &mut page.plan["model"],
                        label,
                        original.width,
                        original.height,
                    )?;
                }
                if let Some(original) = page.original_image.take() {
                    page.image = original;
                }
                let phase_started = std::time::Instant::now();
                progress(Progress {
                    completed: None,
                    elapsed_seconds: None,
                    total: None,
                    stage: "assemble",
                    page_index: Some(page.index),
                    detail: None,
                })?;
                progress(
                    Progress::items("assemble", page.index - first, last - first)
                        .elapsed(timings.since(&batch_base, "assemble")),
                )?;
                let originals = if r.extract_original_image {
                    tables::originals(library, data, page.index, true)?
                } else {
                    vec![]
                };
                parity_trace(
                    &format!("page-{:03}-model.json", page.index),
                    &page.plan["model"],
                )?;
                let mut q = json!({"model":page.plan["model"],"page_width":page.facts["size"][0],"page_height":page.facts["size"][1],"page_index":page.index,"scale":page.image.scale,"formula_enabled":r.formula_enabled});
                if r.native_text {
                    q["native_page"] = page.facts.clone();
                }
                let output = crate::legacy_page::assets::assemble_with_table_images(
                    &q,
                    &page.image.png,
                    &originals,
                    r.extract_original_image,
                    r.extract_original_image_iou_thresh,
                    "images/",
                    &fills,
                )?;
                for asset in output.assets {
                    let name = format!("images/{}", asset.name);
                    if let Some(&i) = asset_indices.get(&name) {
                        if assets[i].png != asset.png {
                            return Err(error("Conflicting document asset"));
                        }
                    } else {
                        asset_indices.insert(name.clone(), assets.len());
                        assets.push(crate::legacy_page::assets::Asset {
                            name,
                            png: asset.png,
                        });
                    }
                }
                for crop in output.ocr_images {
                    if post_images.insert(crop.name, Arc::new(crop.png)).is_some() {
                        return Err(error("Duplicate post-OCR crop"));
                    }
                }
                assembled.push(if output.page.is_null(){json!({"preproc_blocks":[],"page_idx":page.index,"page_size":[page.facts["size"][0].as_f64().unwrap()as i64,page.facts["size"][1].as_f64().unwrap()as i64],"discarded_blocks":[]})}else{output.page});
                timings.add("assemble", phase_started, 1);
                progress(
                    Progress::items("assemble", page.index - first + 1, last - first)
                        .elapsed(timings.since(&batch_base, "assemble")),
                )?;
            }
            progress(Progress {
                completed: None,
                elapsed_seconds: None,
                total: None,
                stage: "finalize",
                page_index: None,
                detail: None,
            })?;
            let phase_started = std::time::Instant::now();
            let finalized = self.finalize(json!(assembled), post_images, resolve)?;
            timings.add("finalize", phase_started, array(&finalized)?.len());
            all_pages.extend(array(&finalized)?.iter().cloned());
        }
        for (i, page) in all_pages.iter_mut().enumerate() {
            page["page_idx"] = json!(i);
        }
        let mut pages = json!(all_pages);
        if range.len() > r.window {
            pages = crate::legacy_paragraphs::split(&pages)?;
        }
        pages = crate::legacy_table_merge::merge(&pages)?;
        let initialized = self.models.initialization_time() - initialization_base;
        if !initialized.is_zero() {
            timings.0.insert("model_initialization", (initialized, 1));
        }
        for row in timings.rows() {
            progress(Progress {
                completed: None,
                elapsed_seconds: None,
                total: None,
                stage: "summary",
                page_index: None,
                detail: Some(format!(
                    "{}: {} items in {:.3}s",
                    row["stage"].as_str().unwrap(),
                    row["items"].as_u64().unwrap(),
                    row["seconds"].as_f64().unwrap()
                )),
            })?;
        }
        let metadata = json!({"source_page_count":count,"page_range":[range.start,range.end],"pages":pages,"native_table_routes":native_routes,"native_layout_routes":layout_routes,"timings":timings.rows()});
        Ok(Output { metadata, assets })
    }
    fn layout(
        &mut self,
        image: &rapiddoc_pdf::render::RenderedPage,
        pixels: &[u8],
        facts: &Value,
        blocks: Option<&Value>,
        r: &Request,
        resolve: &mut Resolver<'_>,
    ) -> Result<(Value, Option<Value>), Error> {
        let mut ignored = r.markdown_ignore_labels.clone();
        if ignored.is_none() && r.docvortex_mode != "off" {
            ignored = Some(
                [
                    "number",
                    "header",
                    "header_image",
                    "footer",
                    "footer_image",
                    "aside_text",
                ]
                .iter()
                .map(|s| (*s).into())
                .collect(),
            );
        }
        let mut model_layout = |engine: &mut Self, filter| -> Result<Value, Error> {
            let regions = engine.models.with_lazy("layout_model", resolve, |m| {
                if r.layout_shape == "rect" && r.layout_model_type == "pp_doclayoutv3" {
                    m.layout_rgb(
                        pixels,
                        image.width as usize,
                        image.height as usize,
                        r.threshold,
                    )?
                    .iter()
                    .map(value)
                    .collect()
                } else {
                    m.layout_polygons_rgb(
                        pixels,
                        image.width as usize,
                        image.height as usize,
                        r.threshold,
                        &r.layout_shape,
                        r.geos_library.as_deref(),
                    )
                }
            })?;
            let regions = regions
                .into_iter()
                .enumerate()
                .map(|(i, mut v)| {
                    if r.layout_model_type != "pp_doclayout_plus_l" {
                        v["order"] = json!(i);
                    }
                    v
                })
                .collect::<Vec<_>>();
            op(
                "convert_layout",
                json!({"regions":regions,"layout_shape_mode":"rect","markdown_ignore_labels":ignored,"formula_enabled":r.formula_enabled,"formula_level":0,"use_custom_ocr":false,"filter_overlaps":filter,"geos_library":r.geos_library}),
            )
        };
        if r.docvortex_mode == "off" || blocks.is_none() && !r.native_only {
            return Ok((model_layout(self, true)?, None));
        }
        let scan = json!([{"type":"text","bbox":[0.,0.,1.,1.],"content":""}]);
        let blocks = if !r.native_text && array(&facts["blocks"])?.is_empty() {
            &scan
        } else {
            blocks.unwrap_or(&Value::Null)
        };
        let nonempty = blocks.as_array().is_some_and(|a| !a.is_empty());
        let mut layout = if nonempty {
            op(
                "convert_native_pdf_layout",
                json!({"blocks":blocks,"width":image.width,"height":image.height,"ignored_labels":ignored.clone().unwrap_or_default()}),
            )?
        } else {
            json!([])
        };
        if !r.native_text {
            for d in layout.as_array_mut().unwrap() {
                d.as_object_mut().unwrap().remove("html");
                d.as_object_mut().unwrap().remove("native_text");
            }
        }
        let reason = if r.native_only {
            Value::Null
        } else if !nonempty {
            json!("empty layout")
        } else {
            op(
                "native_pdf_layout_fallback",
                json!({"blocks":blocks,"layout":layout,"width":image.width,"height":image.height,"native_char_count":facts["chars"].as_array().map(Vec::len).unwrap_or(0)}),
            )?
        };
        if !reason.is_null() {
            return Ok((
                model_layout(self, false)?,
                Some(json!({"method":"model","reason":reason})),
            ));
        }
        if !r.formula_enabled {
            layout
                .as_array_mut()
                .unwrap()
                .retain(|d| d["category_id"] != 13)
        }
        Ok((
            json!({"layout_dets":layout}),
            Some(
                json!({"method":"native","reason":if nonempty{"accepted"}else{"native_unavailable"}}),
            ),
        ))
    }
}
