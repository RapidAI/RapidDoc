//! 旧 PP-DocLayoutV2/V3 类别转换、BatchAnalyze 重叠过滤；不替代掩码多边形路径。
use crate::legacy_page::{invalid, limit, unsupported, validate_tree};
use rapiddoc_core::Error;
use serde_json::{json, Value};

// Each call site owns one compiled constant pattern. Only use when the pattern
// depends on constants, never on document text or caller configuration.
macro_rules! constant_regex {
    ($pattern:expr $(,)?) => {{
        static COMPILED: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        COMPILED.get_or_init(|| regex::Regex::new($pattern).unwrap())
    }};
}

// Internal envelopes containing existing JSON trees and simple Into<Value>
// fields. Unlike json!, this never serializes a Value back into another Value
// (including reparsing every arbitrary-precision number). Deliberately does not
// accept arbitrary Serialize structs; keep json! at typed/output boundaries.
macro_rules! value_object {
    ({ $($key:literal : $value:expr),* $(,)? }) => {{
        let mut fields = serde_json::Map::new();
        $(fields.insert($key.to_owned(), serde_json::Value::from(($value).clone()));)*
        serde_json::Value::Object(fields)
    }};
}

#[cfg(test)]
mod value_transfer_tests {
    #[test]
    fn internal_envelopes_preserve_number_tokens_and_unknown_fields() {
        let source: serde_json::Value = serde_json::from_str(
            r#"{"unknown":{"big":1234567890123456789012345678901234567890,"negative_zero":-0.0,"tiny":1e-200,"fraction":1.2300},"flags":[true,false,null]}"#,
        ).unwrap();
        let page: usize = 17;
        let entries = vec![source.clone()];
        let expected = serde_json::json!({"source":source,"page":page,"entries":entries});
        let actual = value_object!({"source":source,"page":page,"entries":entries});
        assert_eq!(
            serde_json::to_vec(&actual).unwrap(),
            serde_json::to_vec(&expected).unwrap()
        );
        assert_eq!(actual["source"], source);
    }
}

/// Opt-in developer timing; no document content is emitted.
/// Attach an owned page without serializing (and recursively copying) its tree.
pub(crate) fn with_source(
    source: serde_json::Value,
    mut fields: serde_json::Value,
) -> serde_json::Value {
    fields["source"] = source;
    fields
}

pub(crate) struct NativeStageProfile(Option<std::time::Instant>);
impl NativeStageProfile {
    pub(crate) fn new() -> Self {
        Self(std::env::var_os("RAPIDDOC_NATIVE_PROFILE").map(|_| std::time::Instant::now()))
    }
    pub(crate) fn mark(&mut self, stage: &str) {
        if let Some(previous) = self.0.as_mut() {
            eprintln!(
                "RAPIDDOC_NATIVE_PROFILE {stage} {:.6}",
                previous.elapsed().as_secs_f64()
            );
            *previous = std::time::Instant::now();
        }
    }
}
// Aggregate repeated candidate stages instead of printing thousands of lines.
pub(crate) struct NativeStageTotals {
    checkpoint: Option<std::time::Instant>,
    totals: std::collections::BTreeMap<&'static str, (usize, std::time::Duration)>,
}
impl Default for NativeStageTotals {
    fn default() -> Self {
        Self {
            checkpoint: std::env::var_os("RAPIDDOC_NATIVE_PROFILE")
                .map(|_| std::time::Instant::now()),
            totals: Default::default(),
        }
    }
}
impl NativeStageTotals {
    pub(crate) fn begin(&mut self) {
        if self.checkpoint.is_some() {
            self.checkpoint = Some(std::time::Instant::now());
        }
    }
    pub(crate) fn mark(&mut self, stage: &'static str) {
        if let Some(previous) = self.checkpoint.as_mut() {
            let item = self.totals.entry(stage).or_default();
            item.0 += 1;
            item.1 += previous.elapsed();
            *previous = std::time::Instant::now();
        }
    }
}
impl Drop for NativeStageTotals {
    fn drop(&mut self) {
        for (stage, (count, time)) in &self.totals {
            eprintln!(
                "RAPIDDOC_NATIVE_TOTAL {stage} calls={count} seconds={:.6}",
                time.as_secs_f64()
            );
        }
    }
}
#[path = "graphics/bar_geometry.rs"]
pub mod bar_geometry;
#[path = "graphics/concentric_graphics.rs"]
pub mod concentric_graphics;
#[path = "ocr/crops.rs"]
pub mod crops;
#[path = "graphics/drawing_graphics.rs"]
pub mod drawing_graphics;
#[path = "formula/formulas.rs"]
pub mod formulas;
#[path = "graphics/graphic_arbitration.rs"]
pub mod graphic_arbitration;
#[path = "graphics/graphic_grid.rs"]
pub mod graphic_grid;
#[path = "graphics/image_regions.rs"]
pub mod image_regions;
#[path = "document/inline_evidence.rs"]
pub mod inline_evidence;
#[path = "document/inline_output/mod.rs"]
pub mod inline_output;
#[path = "graphics/isolated_graphics.rs"]
pub mod isolated_graphics;
#[path = "layout/native_adapter.rs"]
pub mod native_adapter;
#[path = "document/native_document.rs"]
pub mod native_document;
#[path = "text/native_document_text/mod.rs"]
pub mod native_document_text;
#[path = "document/native_final_output.rs"]
pub mod native_final_output;
#[path = "document/native_finalization.rs"]
pub mod native_finalization;
#[path = "text/native_footnotes/mod.rs"]
pub mod native_footnotes;
#[path = "formula/native_formula_components/mod.rs"]
pub mod native_formula_components;
#[path = "formula/native_formula_text.rs"]
pub mod native_formula_text;
#[path = "layout/native_geometry/mod.rs"]
pub mod native_geometry;
#[path = "text/native_glyph_weight.rs"]
pub mod native_glyph_weight;
#[path = "layout/native_grouped_order/mod.rs"]
pub mod native_grouped_order;
#[path = "document/native_inline_document.rs"]
pub mod native_inline_document;
#[path = "layout/native_layout.rs"]
pub mod native_layout;
#[path = "text/native_lines.rs"]
pub mod native_lines;
#[path = "text/native_marginals/mod.rs"]
pub mod native_marginals;
#[path = "layout/native_model_normalization.rs"]
pub mod native_model_normalization;
#[path = "ocr/native_ocr/mod.rs"]
pub mod native_ocr;
#[path = "document/native_page_preparation/mod.rs"]
pub mod native_page_preparation;
#[path = "document/native_pdf_document.rs"]
pub mod native_pdf_document;
#[path = "layout/native_precontainers/mod.rs"]
pub mod native_precontainers;
#[path = "text/native_references/mod.rs"]
pub mod native_references;
#[path = "text/native_text_assembly/mod.rs"]
pub mod native_text_assembly;
#[path = "text/native_text_roles/mod.rs"]
pub mod native_text_roles;
#[path = "text/native_titles/mod.rs"]
pub mod native_titles;
#[path = "formula/native_vector_formulas/mod.rs"]
pub mod native_vector_formulas;
#[path = "layout/native_visual_annotations/mod.rs"]
pub mod native_visual_annotations;
#[path = "ocr/ocr_results.rs"]
pub mod ocr_results;
#[path = "document/page_containers/mod.rs"]
pub mod page_containers;
#[path = "graphics/path_graphics.rs"]
pub mod path_graphics;
#[path = "graphics/raster_charts.rs"]
pub mod raster_charts;
#[path = "graphics/raster_diagrams.rs"]
pub mod raster_diagrams;
#[path = "text/source_lines.rs"]
pub mod source_lines;
#[path = "graphics/stacked_graphics.rs"]
pub mod stacked_graphics;
#[path = "table/table_block/mod.rs"]
pub mod table_block;
#[path = "table/table_direction.rs"]
pub mod table_direction;
#[path = "table/table_hints.rs"]
pub mod table_hints;
#[path = "table/table_match.rs"]
pub mod table_match;
#[path = "table/table_native.rs"]
pub mod table_native;
#[path = "table/table_native_aligned_numeric.rs"]
pub mod table_native_aligned_numeric;
#[path = "table/table_native_banded_numeric.rs"]
pub mod table_native_banded_numeric;
#[path = "table/table_native_banded_rowspan.rs"]
mod table_native_banded_rowspan;
#[path = "table/table_native_candidate.rs"]
pub mod table_native_candidate;
#[path = "table/table_native_engine.rs"]
pub mod table_native_engine;
#[path = "table/table_ocr_rule.rs"]
pub mod table_ocr_rule;
#[path = "table/table_native_engine_policy.rs"]
pub mod table_native_engine_policy;
#[path = "table/table_native_hierarchical.rs"]
mod table_native_hierarchical;
#[path = "table/table_native_multiline/mod.rs"]
pub mod table_native_multiline;
#[path = "table/table_native_rule.rs"]
pub mod table_native_rule;
#[path = "table/table_native_rule_band.rs"]
pub mod table_native_rule_band;
#[path = "table/table_native_sparse.rs"]
mod table_native_sparse;
#[path = "table/table_native_sparse_hybrid.rs"]
pub mod table_native_sparse_hybrid;
#[path = "table/table_native_text.rs"]
pub mod table_native_text;
#[path = "table/table_native_text_grid.rs"]
pub mod table_native_text_grid;
#[path = "table/table_native_text_quality.rs"]
mod table_native_text_quality;
#[path = "table/table_native_vector/mod.rs"]
pub mod table_native_vector;
#[path = "table/table_normalize.rs"]
pub mod table_normalize;
#[path = "table/table_selection.rs"]
pub mod table_selection;
#[path = "table/table_text.rs"]
pub mod table_text;
#[path = "table/table_text_styles/mod.rs"]
pub mod table_text_styles;
#[path = "text/text_noise.rs"]
pub mod text_noise;

const IGNORED: &[&str] = &[
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
];
fn category(label: &str) -> Option<u64> {
    Some(match label {
        "doc_title" | "paragraph_title" => 0,
        "chart" | "image" | "seal" | "footer_image" | "header_image" => 3,
        "table" => 5,
        "display_formula" => 14,
        "inline_formula" => 13,
        "formula_number" => 9,
        "abstract" | "algorithm" | "aside_text" | "content" | "figure_title" | "footer"
        | "footnote" | "header" | "number" | "reference" | "reference_content" | "text"
        | "vertical_text" | "vision_footnote" => 1,
        _ => return None,
    })
}
fn flag(request: &Value, key: &str, default: bool) -> Result<bool, Error> {
    request
        .get(key)
        .map_or(Ok(default), |v| v.as_bool().ok_or_else(invalid))
}
pub(super) fn coordinates(v: &Value) -> Result<[f64; 4], Error> {
    let a = v.as_array().filter(|a| a.len() == 4).ok_or_else(invalid)?;
    let mut out = [0.; 4];
    for (i, v) in a.iter().enumerate() {
        out[i] = v
            .as_f64()
            .filter(|n| n.is_finite() && n.abs() <= 1e9)
            .ok_or_else(invalid)?;
    }
    Ok(out)
}
pub(super) fn rect(d: &Value) -> Result<[f64; 4], Error> {
    let p = d["poly"]
        .as_array()
        .filter(|a| a.len() == 8)
        .ok_or_else(invalid)?;
    let b = coordinates(&json!([p[0], p[1], p[4], p[5]]))?;
    let p = p
        .iter()
        .map(|v| {
            v.as_f64()
                .filter(|n| n.is_finite() && n.abs() <= 1e9)
                .ok_or_else(invalid)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if p[2] != p[4] || p[3] != p[1] || p[6] != p[0] || p[7] != p[5] {
        return Err(unsupported("仅支持旧矩形 layout poly"));
    }
    if d.get("polygon_points")
        .is_some_and(|p| !p.is_null() && p.as_array().is_none_or(|a| !a.is_empty()))
    {
        if !crate::legacy_page::budget::current().host {
            return Err(unsupported("旧多边形布局过滤/裁图尚未迁移，不能以矩形替代"));
        }
        // The host preserves polygons; crop masking uses the native LINE_8 kernel.
    }
    Ok(b)
}
fn area(b: [f64; 4]) -> f64 {
    ((b[2] - b[0]) * (b[3] - b[1])).abs()
}
fn overlap(a: [f64; 4], b: [f64; 4]) -> f64 {
    let reference = area(a).min(area(b));
    if reference == 0. {
        0.
    } else {
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.)
            / reference
    }
}
/// 保留原列表顺序与未知字段；旧 reference 标签、6像素门槛、inline 和图片例外均按原分支。
pub fn filter(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let custom = flag(request, "use_custom_ocr", false)?;
    let input = request["layout_dets"].as_array().ok_or_else(invalid)?;
    if input.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut dets = Vec::new();
    let mut boxes = Vec::new();
    for d in input {
        let label = d["original_label"].as_str().ok_or_else(invalid)?;
        if label == "reference" {
            continue;
        }
        boxes.push(rect(d)?);
        dets.push(d);
    }
    let mut dropped = vec![false; dets.len()];
    let mut geometry = None;
    for i in 0..dets.len() {
        let b = boxes[i];
        if b[2] - b[0] < 6. || b[3] - b[1] < 6. {
            dropped[i] = true;
        }
        for j in i + 1..dets.len() {
            if dropped[i] || dropped[j] {
                continue;
            }
            let ratio = overlap(b, boxes[j]);
            let a = dets[i]["original_label"].as_str().unwrap();
            let z = dets[j]["original_label"].as_str().unwrap();
            if a == "inline_formula" || z == "inline_formula" {
                if !custom {
                    continue;
                }
                if ratio > 0.5 {
                    if a == "inline_formula" {
                        dropped[i] = true;
                    }
                    if z == "inline_formula" {
                        dropped[j] = true;
                    }
                    continue;
                }
            }
            if ratio > 0.7 {
                if let Some(points) = dets[i]
                    .get("polygon_points")
                    .and_then(Value::as_array)
                    .filter(|p| !p.is_empty())
                {
                    if geometry.is_none() {
                        let path = request["geos_library"].as_str().ok_or_else(|| {
                            unsupported("Polygon overlap requires the GEOS runtime")
                        })?;
                        geometry = Some(rapiddoc_inference::layout_geos::Context::load(
                            std::path::Path::new(path),
                        )?);
                    }
                    let a: Vec<[f64; 2]> =
                        serde_json::from_value(json!(points)).map_err(|_| invalid())?;
                    let b: Vec<[f64; 2]> =
                        serde_json::from_value(dets[j]["polygon_points"].clone())
                            .map_err(|_| invalid())?;
                    if geometry.as_ref().unwrap().overlap(&a, &b, "small")? < 0.7 {
                        continue;
                    }
                }
                if a != z
                    && [a, z]
                        .iter()
                        .any(|v| matches!(*v, "image" | "seal" | "chart"))
                {
                    continue;
                }
                if area(b) >= area(boxes[j]) {
                    dropped[j] = true;
                } else {
                    dropped[i] = true;
                }
            }
        }
    }
    Ok(json!(dets
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !dropped[*i])
        .map(|(_, d)| d.clone())
        .collect::<Vec<_>>()))
}
/// 输入为已做 PP 后处理的矩形 regions；不重新排序，也不臆造缺失的模型顺序。
pub fn convert(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    if request["layout_shape_mode"] != "rect" {
        return Err(unsupported(
            "模型路由目前仅支持显式 layout_shape_mode='rect'",
        ));
    }
    let defaults = json!(IGNORED);
    let ignored = request
        .get("markdown_ignore_labels")
        .filter(|v| !v.is_null())
        .unwrap_or(&defaults)
        .as_array()
        .ok_or_else(invalid)?;
    if ignored.len() > rapiddoc_core::resources::limit(128)
        || ignored.iter().any(|v| {
            v.as_str()
                .is_none_or(|s| s.len() > rapiddoc_core::resources::limit(256))
        })
    {
        return Err(invalid());
    }
    let formulas = flag(request, "formula_enabled", true)?;
    let level = request
        .get("formula_level")
        .map_or(Some(0), Value::as_u64)
        .filter(|v| *v <= 2)
        .ok_or_else(invalid)?;
    let apply_filter = flag(request, "filter_overlaps", true)?;
    let custom = flag(request, "use_custom_ocr", false)?;
    let regions = request["regions"].as_array().ok_or_else(invalid)?;
    if regions.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut dets = Vec::new();
    for r in regions {
        let label = r["label"].as_str().ok_or_else(invalid)?;
        let category = category(label).ok_or_else(|| unsupported("未知 PP-DocLayoutV2/V3 类别"))?;
        coordinates(&r["coordinate"])?;
        let a = r["coordinate"].as_array().unwrap();
        let order = r["order"]
            .as_u64()
            .filter(|v| *v <= rapiddoc_core::resources::limit(1_000_000_000))
            .ok_or_else(|| unsupported("必须提供非负模型 order，enhanced 阅读顺序已延期"))?;
        let score = r["score"]
            .as_f64()
            .filter(|s| (0. ..=1.).contains(s))
            .ok_or_else(invalid)?;
        let score = format!("{score:.3}")
            .parse::<f64>()
            .map_err(|_| invalid())?;
        let category = if ignored.iter().any(|v| v.as_str() == Some(label)) {
            2
        } else {
            category
        };
        let d = json!({"category_id":category,"original_label":label,"original_order":order,"poly":[a[0],a[1],a[2],a[1],a[2],a[3],a[0],a[3]],"polygon_points":r.get("polygon_points").cloned().unwrap_or(Value::Null),"score":score});
        rect(&d)?;
        dets.push(d);
    }
    let value = if apply_filter {
        filter(
            &json!({"layout_dets":dets,"use_custom_ocr":custom,"geos_library":request["geos_library"]}),
        )?
    } else {
        json!(dets)
    };
    let dets = value
        .as_array()
        .unwrap()
        .iter()
        .filter(|d| (formulas && level != 1) || d["category_id"] != 13)
        .cloned()
        .collect::<Vec<_>>();
    Ok(json!({"layout_dets":dets}))
}

#[path = "table/tiled_tables/mod.rs"]
pub mod tiled_tables;

#[path = "graphics/bar_groups.rs"]
pub mod bar_groups;

#[path = "graphics/bar_graphics.rs"]
pub mod bar_graphics;
#[path = "graphics/graphic_captions.rs"]
pub mod graphic_captions;

#[path = "table/filled_grid/mod.rs"]
pub mod filled_grid;

#[path = "table/closed_grid.rs"]
pub mod closed_grid;

#[path = "table/table_replacements.rs"]
pub mod table_replacements;

#[path = "table/table_members.rs"]
mod table_members;
#[path = "table/table_candidate_merge.rs"]
pub mod table_candidate_merge;

#[path = "table/prose_rule_conflicts.rs"]
pub mod prose_rule_conflicts;

#[path = "table/rule_annotation_preparation.rs"]
pub mod rule_annotation_preparation;

#[path = "table/table_footnotes/mod.rs"]
pub mod table_footnotes;

#[path = "table/rule_candidate_expansion.rs"]
pub mod rule_candidate_expansion;

#[path = "table/closed_rule_pipeline/mod.rs"]
pub mod closed_rule_pipeline;
#[path = "document/code_containers/mod.rs"]
pub mod code_containers;
#[path = "graphics/diagram_components.rs"]
pub mod diagram_components;
#[path = "graphics/strong_graphics.rs"]
pub mod strong_graphics;
#[path = "table/table_page.rs"]
pub mod table_page;
