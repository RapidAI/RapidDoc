//! Original native-rule guard, direction, OCR, structure routing and image hints.
use super::*;
use crate::legacy_model::{native_adapter, table_direction, table_hints, table_native, table_text};
use crate::legacy_page::assets::{OriginalImage, TableFillImage};
use rapiddoc_inference::{
    table_classification::Variant as Classifier, table_structure::Variant as Structure, wired_match,
};

struct TableCandidate {
    html: Value,
    rejected: Option<String>,
}
/// Only errors produced while interpreting candidate HTML are recoverable.
/// Session, model, device and resource errors must propagate to the caller.
fn candidate_html(result: Result<Value, Error>) -> Result<TableCandidate, Error> {
    match result {
        Ok(html) => Ok(TableCandidate {
            html,
            rejected: None,
        }),
        Err(e) if matches!(e.code, ErrorCode::Unsupported | ErrorCode::InvalidInput) => {
            Ok(TableCandidate {
                html: Value::Null,
                rejected: Some(e.message),
            })
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod candidate_tests {
    use super::*;
    #[test]
    fn html_rejection_cannot_hide_model_or_resource_errors() {
        for code in [ErrorCode::Unsupported, ErrorCode::InvalidInput] {
            let candidate = candidate_html(Err(Error {
                code,
                message: "bad candidate markup".into(),
            }))
            .unwrap();
            assert!(candidate.html.is_null());
            assert!(candidate.rejected.is_some());
        }
        for code in [
            ErrorCode::ResourceLimit,
            ErrorCode::ModelMissing,
            ErrorCode::DeviceUnavailable,
            ErrorCode::RuntimeUnavailable,
        ] {
            assert!(candidate_html(Err(Error {
                code,
                message: "must propagate".into()
            }))
            .is_err());
        }
    }
}

pub(super) fn originals(
    library: &Path,
    data: &[u8],
    index: usize,
    render: bool,
) -> Result<Vec<OriginalImage>, Error> {
    Ok(rapiddoc_pdf::original_images::extract(
        library,
        data,
        index,
        None,
        &rapiddoc_pdf::original_images::Config {
            render,
            ..Default::default()
        },
        &Limits::pdf_host(),
    )?
    .images
    .into_iter()
    .map(|i| OriginalImage {
        bbox: i.bbox,
        png: i.png,
    })
    .collect())
}
impl Engine {
    fn table_boxes(
        &mut self,
        png: &[u8],
        masks: &Value,
        resolve: &mut Resolver<'_>,
    ) -> Result<Value, Error> {
        let prepared = table_text::detection_pixels(png, &json!({"masks":masks}))?;
        let (width, height) = (prepared.width, prepared.height);
        let detect = |engine: &mut Self,
                      pixels: crate::legacy_page::assets::Pixels,
                      masks: &Value,
                      resolve: &mut Resolver<'_>|
         -> Result<Value, Error> {
            let detections = engine.models.with_lazy("detector", resolve, |m| {
                m.detect_table_rgb(rapiddoc_core::pillow_resize::Rgb {
                    w: pixels.width,
                    h: pixels.height,
                    data: pixels.data,
                })
            })?;
            op(
                "table_detection_boxes",
                json!({"boxes":wire(&detections.iter().map(|d|d.points).collect::<Vec<_>>())?,"masks":masks}),
            )
        };
        // Retain the masked original for the rare thin-image retry. Only that
        // fallback needs another allocation; the normal path transfers ownership.
        let thin = width.min(height) < 32;
        let mut boxes = detect(self, prepared, masks, resolve)?;
        if array(&boxes)?.is_empty() && thin {
            let prepared = table_text::detection_pixels(png, &json!({"masks":masks}))?;
            let padded = table_text::padding_pixels(prepared, 8)?;
            let retry = detect(self, padded, &json!([]), resolve)?;
            if !array(&retry)?.is_empty() {
                boxes = op(
                    "restore_table_detection",
                    json!({"boxes":retry,"padding":8,"width":width,"height":height}),
                )?;
            }
        }
        Ok(boxes)
    }
    fn table_rows(
        &mut self,
        png: &[u8],
        boxes: &Value,
        full: bool,
        resolve: &mut Resolver<'_>,
    ) -> Result<Value, Error> {
        let points = serde_json::from_value::<Vec<[[f32; 2]; 4]>>(boxes.clone()).map_err(error)?;
        let images = rapiddoc_inference::crop::pipeline_pixels_batch(png, &points)?;
        let results = self
            .models
            .with_lazy("recognizer", resolve, |m| m.recognize_images(&images, 6))?;
        let rows = results
            .iter()
            .map(|r| (&r.text, r.confidence))
            .collect::<Vec<_>>();
        op(
            if full {
                "apply_table_full_ocr"
            } else {
                "apply_table_ocr"
            },
            json!({"boxes":boxes,"results":rows}),
        )
    }
    fn table_text(
        &mut self,
        png: &[u8],
        page: &Page,
        table: &Value,
        useful: &Value,
        masks: &Value,
        r: &Request,
        resolve: &mut Resolver<'_>,
    ) -> Result<(Vec<u8>, Value), Error> {
        let (width, height) = table_direction::dimensions(png)?;
        let mut boxes = self.table_boxes(png, masks, resolve)?;
        let plan = op(
            "plan_table_direction",
            json!({"page":page.facts,"table":table,"width":width,"height":height,"boxes":boxes,"scale":page.image.scale}),
        )?;
        let mut label = plan["rotate_label"]
            .as_str()
            .ok_or_else(|| error("Invalid table direction"))?
            .to_owned();
        if plan["needs_classifier"] == true {
            let classifier = table_direction::image(png, "0", true)?;
            let prediction = self.models.with_lazy("orientation", resolve, |m| {
                m.classify_orientation_png(&classifier.png)
            })?;
            label = table_direction::select_model_rotation(&value(&prediction)?)?
                .as_str()
                .ok_or_else(|| error("Invalid table orientation decision"))?
                .to_owned();
        }
        let png = table_direction::oriented_png(png, &label)?;
        if matches!(label.as_str(), "90" | "270") {
            boxes = self.table_boxes(&png, masks, resolve)?;
        }
        let mut rows = json!([]);
        if !array(&boxes)?.is_empty()
            && r.native_text
            && !r.table_force_ocr
            && label == "0"
            && plan["pdf_not_rotate"] == true
        {
            rows = table_native::extract(
                &png,
                &json!({"boxes":boxes,"native_page":page.facts,"table":table,"useful_list":useful,"scale":page.image.scale,"word_boxes":false}),
            )?;
        }
        if array(&rows)?.is_empty() && !array(&boxes)?.is_empty() {
            rows = self.table_rows(&png, &boxes, false, resolve)?;
        }
        Ok((png.into_owned(), rows))
    }
    fn table_payload(
        &mut self,
        png: &[u8],
        mut rows: Value,
        fill: &Value,
        formulas: &Value,
        resolve: &mut Resolver<'_>,
    ) -> Result<Option<(Vec<u8>, Value)>, Error> {
        if rows.is_null() || array(&rows)?.is_empty() {
            let source = table_direction::oriented_png(png, "0")?;
            let raw = self
                .models
                .with_lazy("detector", resolve, |m| m.detect_table_png(&source))?;
            if raw.is_empty() {
                return Ok(None);
            }
            let boxes = op(
                "table_detection_boxes",
                json!({"boxes":wire(&raw.iter().map(|d|d.points).collect::<Vec<_>>())?,"masks":formulas}),
            )?;
            if array(&boxes)?.is_empty() {
                return Err(error(
                    "Legacy full OCR cannot recognize an empty list after formula cutting",
                ));
            }
            rows = self.table_rows(&source, &boxes, true, resolve)?;
            if array(&rows)?.is_empty() {
                return Ok(None);
            }
        }
        let payload = table_hints::payload(
            png,
            &json!({"ocr_result":rows,"fill_images":fill,"formulas":formulas,"skip_text_in_image":true,"inline_left":"$","inline_right":"$"}),
        )?;
        Ok(Some((payload.png, payload.metadata["ocr_result"].clone())))
    }
    fn structure(
        &mut self,
        png: &[u8],
        rows: &Value,
        role: &str,
        variant: &str,
        resolve: &mut Resolver<'_>,
        timings: &mut Timings,
    ) -> Result<Value, Error> {
        // table_model already prepared and validated this common payload.
        // Reapplying empty image/formula hints only copied rows and performed
        // another lossless PNG round trip for every candidate.
        if variant == "table-transformer" {
            let (width, height) = table_direction::dimensions(png)?;
            let mut tokens = vec![];
            for (i, b) in array(&rows[0])?.iter().enumerate() {
                let points: Vec<[f32; 2]> = serde_json::from_value(b.clone()).map_err(error)?;
                if points.is_empty() || points.iter().flatten().any(|v| !v.is_finite()) {
                    continue;
                }
                let bbox = [
                    points
                        .iter()
                        .map(|p| p[0])
                        .fold(f32::INFINITY, f32::min)
                        .max(0.),
                    points
                        .iter()
                        .map(|p| p[1])
                        .fold(f32::INFINITY, f32::min)
                        .max(0.),
                    points
                        .iter()
                        .map(|p| p[0])
                        .fold(f32::NEG_INFINITY, f32::max)
                        .min(width as f32),
                    points
                        .iter()
                        .map(|p| p[1])
                        .fold(f32::NEG_INFINITY, f32::max)
                        .min(height as f32),
                ]
                .map(f64::from);
                if bbox[2] > bbox[0] && bbox[3] > bbox[1] {
                    tokens.push(rapiddoc_inference::table_transformer::Token {
                        bbox,
                        text: rows[1][i]
                            .as_str()
                            .ok_or_else(|| error("Invalid TATR OCR text"))?
                            .into(),
                        block_num: 0,
                        line_num: i,
                        span_num: 0,
                    });
                }
            }
            let started = std::time::Instant::now();
            let mut cells = self
                .models
                .with_lazy(role, resolve, |m| m.transformer_cells_png(png, &tokens))?;
            timings.add("table_transformer", started, 1);
            cells.sort_by_key(|c| (c.row_nums[0], c.column_nums[0]));
            let nrows = cells
                .iter()
                .flat_map(|c| c.row_nums.iter())
                .max()
                .map_or(0, |n| n + 1);
            let mut html = String::from("<table>");
            for row in 0..nrows {
                html.push_str("<tr>");
                for c in cells.iter().filter(|c| c.row_nums[0] == row) {
                    let tag = if c.column_header { "th" } else { "td" };
                    html.push('<');
                    html.push_str(tag);
                    if c.row_nums.len() > 1 {
                        html.push_str(&format!(
                            " rowspan=\"{}\"",
                            c.row_nums.last().unwrap() - c.row_nums[0] + 1
                        ));
                    }
                    if c.column_nums.len() > 1 {
                        html.push_str(&format!(
                            " colspan=\"{}\"",
                            c.column_nums.last().unwrap() - c.column_nums[0] + 1
                        ));
                    }
                    html.push('>');
                    html.push_str(
                        &c.text
                            .replace('&', "&amp;")
                            .replace('<', "&lt;")
                            .replace('>', "&gt;"),
                    );
                    html.push_str("</");
                    html.push_str(tag);
                    html.push('>');
                }
                html.push_str("</tr>");
            }
            html.push_str("</table>");
            Ok(json!(html))
        } else if variant == "unet" {
            let started = std::time::Instant::now();
            let cells = self
                .models
                .with_lazy(role, resolve, |m| m.table_cells_png(&png, true))?;
            timings.add("table_unet_cells", started, 1);
            let cells = wire(&cells)?;
            let polygons: Vec<rapiddoc_inference::table_recover::Polygon> =
                serde_json::from_value(cells["polygons"].clone()).map_err(error)?;
            if polygons.is_empty() {
                return Ok(json!(""));
            }
            let points: Vec<[usize; 4]> =
                serde_json::from_value(cells["topology"]["logic_points"].clone()).map_err(error)?;
            let boxes = array(&rows[0])?;
            let texts = array(&rows[1])?;
            let confidences = array(&rows[2])?;
            let ocr = boxes
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    Ok(wired_match::Ocr {
                        points: serde_json::from_value(b.clone()).map_err(error)?,
                        text: texts[i]
                            .as_str()
                            .ok_or_else(|| error("Invalid table OCR text"))?
                            .into(),
                        confidence: confidences[i]
                            .as_f64()
                            .ok_or_else(|| error("Invalid table OCR confidence"))?,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let mut matched = wired_match::match_cells(&polygons, &ocr)?;
            wired_match::fill_blank(&mut matched, &polygons, &[])?;
            Ok(json!(rapiddoc_render::wired::legacy_html(
                &points,
                &polygons,
                &matched.cell_lines
            )?))
        } else {
            let variant = match variant {
                "slanet-plus" => Structure::SlanetPlus,
                "slanet-1m" => Structure::Slanet1m,
                _ => return Err(error("Unsupported table structure variant")),
            };
            let started = std::time::Instant::now();
            let structure = self
                .models
                .with_lazy(role, resolve, |m| m.table_structure_png(&png, variant))?;
            timings.add("table_slanet_structure", started, 1);
            let boxes = structure
                .cell_bboxes
                .iter()
                .map(|b| b.iter().map(|&x| x as f64).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let (width, height) = table_direction::dimensions(&png)?;
            Ok(op("legacy_table_match",json!({"structure":{"tokens":structure.tokens,"confidence":structure.confidence,"cell_bboxes":boxes},"ocr_result":rows,"width":width,"height":height}))?["raw_html"].clone())
        }
    }
    fn table_model(
        &mut self,
        png: &[u8],
        rows: Value,
        hints: &Value,
        r: &Request,
        resolve: &mut Resolver<'_>,
        timings: &mut Timings,
    ) -> Result<(Value, bool, Value), Error> {
        let started = std::time::Instant::now();
        let Some((png, rows)) = self.table_payload(
            png,
            rows,
            &hints["fill_images"],
            &hints["formulas"],
            resolve,
        )?
        else {
            return Ok((Value::Null, false, Value::Null));
        };
        timings.add("table_payload", started, 1);
        let main_candidate = self.table_candidate(
            &png,
            &rows,
            &r.table_variant,
            "table_model",
            r,
            resolve,
            timings,
        )?;
        let main = main_candidate.html;
        let rejected = main_candidate.rejected;
        if r.fallback_table_variant.is_none() && !r.fallback_to_text {
            return Ok((
                main,
                false,
                if rejected.is_some() {
                    json!({"selected":"none","main_rejected":rejected})
                } else {
                    Value::Null
                },
            ));
        }
        let excluded = super::table_fallback::excluded(hints);
        let first = super::table_fallback::quality(&main, &rows, &excluded);
        let mut report = json!({"main":r.table_variant,"main_valid":first.valid,"ocr_chars":first.total,"main_retained":first.retained,"fallback_ran":false,"selected":"main"});
        if let Some(reason) = &rejected {
            report["main_rejected"] = json!(reason);
        }
        if rejected.is_none() && !first.needs_fallback(r.fallback_text_loss_threshold) {
            return Ok((main, false, report));
        }
        if let Some(backup) = r
            .fallback_table_variant
            .as_deref()
            .filter(|v| *v != r.table_variant)
        {
            let started = std::time::Instant::now();
            let alternative = self.table_candidate(
                &png,
                &rows,
                backup,
                "fallback_table_model",
                r,
                resolve,
                timings,
            )?;
            timings.add("table_fallback", started, 1);
            if let Some(reason) = &alternative.rejected {
                report["fallback_rejected"] = json!(reason);
            }
            let alternative = alternative.html;
            let second = super::table_fallback::quality(&alternative, &rows, &excluded);
            report["fallback_ran"] = json!(true);
            report["fallback"] = json!(backup);
            report["fallback_valid"] = json!(second.valid);
            report["fallback_retained"] = json!(second.retained);
            if second.valid
                && second.retained > first.retained
                && !second.needs_fallback(r.fallback_text_loss_threshold)
            {
                report["selected"] = json!("fallback");
                return Ok((alternative, false, report));
            }
        }
        if r.fallback_to_text {
            report["selected"] = json!("text");
            Ok((
                json!(super::table_fallback::plain_html(&rows)),
                true,
                report,
            ))
        } else {
            Ok((main, false, report))
        }
    }
    fn table_candidate(
        &mut self,
        png: &[u8],
        rows: &Value,
        variant: &str,
        role: &str,
        r: &Request,
        resolve: &mut Resolver<'_>,
        timings: &mut Timings,
    ) -> Result<TableCandidate, Error> {
        let html = if let Some(variant) = variant.strip_prefix("unet-") {
            let started = std::time::Instant::now();
            let a = self.models.with_lazy("classifier", resolve, |m| {
                m.classify_table_png(&png, Classifier::Paddle, Classifier::Paddle.output_type())
            })?;
            let b = self
                .models
                .with_lazy("secondary_classifier", resolve, |m| {
                    m.classify_table_png(
                        &png,
                        Classifier::Qanything,
                        Classifier::Qanything.output_type(),
                    )
                })?;
            timings.add("table_classification", started, 1);
            let label = if a.label == b.label {
                a.label
            } else {
                "wireless"
            };
            let compare = r.use_compare_table || a.score.min(b.score) < r.cls_conf_threshold;
            let mut candidates = BTreeMap::new();
            let mut required = vec![label];
            if compare {
                required.push(if label == "wired" {
                    "wireless"
                } else {
                    "wired"
                });
            }
            for route in required {
                let html = self.structure(
                    &png,
                    &rows,
                    if route == "wired" {
                        "wired_model"
                    } else {
                        role
                    },
                    if route == "wired" { "unet" } else { variant },
                    resolve,
                    timings,
                )?;
                candidates.insert(route, html);
            }
            let selected = if compare {
                let selection = candidate_html(op(
                    "select_legacy_table",
                    json!({"texts":rows[1],"wired_html":candidates["wired"],"wireless_html":candidates["wireless"]}),
                ))?;
                if selection.rejected.is_some() {
                    return Ok(selection);
                }
                selection.html["route"]
                    .as_str()
                    .ok_or_else(|| error("Invalid table selection"))?
                    .to_owned()
            } else {
                label.into()
            };
            candidates.remove(selected.as_str()).unwrap()
        } else {
            self.structure(&png, &rows, role, variant, resolve, timings)?
        };
        candidate_html(op("normalize_legacy_table_html", html))
    }
    pub(super) fn tables(
        &mut self,
        library: &Path,
        data: &[u8],
        page: &mut Page,
        r: &Request,
        resolve: &mut Resolver<'_>,
        routes: &mut Vec<Value>,
        timings: &mut Timings,
        progress: &mut dyn FnMut(usize, usize, f64) -> Result<(), Error>,
    ) -> Result<Vec<TableFillImage>, Error> {
        let elapsed = std::time::Instant::now();
        let loading = self.models.initialization_time();
        let dets = array(&page.plan["model"]["layout_dets"])?;
        let formulas = json!(dets
            .iter()
            .filter(|d| matches!(d["category_id"].as_u64(), Some(8 | 13 | 14)))
            .cloned()
            .collect::<Vec<_>>());
        let checkboxes = json!(dets
            .iter()
            .filter(|d| d["category_id"] == 200)
            .cloned()
            .collect::<Vec<_>>());
        let mut context = page.facts.clone();
        context["ori_image_list"] = json!([]);
        let mut filled = vec![];
        let mut geometry = None;
        let started = std::time::Instant::now();
        let guard = if r.use_rule_table
            && r.native_text
            && !array(&page.facts["blocks"])?.is_empty()
            && !array(&page.plan["tables"])?.is_empty()
        {
            json!(originals(library, data, page.index, false)?
                .into_iter()
                .map(|i| json!({"bbox":i.bbox}))
                .collect::<Vec<_>>())
        } else {
            json!([])
        };
        timings.add(
            "table_native_guard",
            started,
            array(&page.plan["tables"])?.len(),
        );
        let tasks = array(&page.plan["tables"])?.clone();
        let total = tasks.len();
        for (done, task) in tasks.into_iter().enumerate() {
            progress(
                done,
                total,
                elapsed
                    .elapsed()
                    .saturating_sub(self.models.initialization_time() - loading)
                    .as_secs_f64(),
            )?;
            let index = task["layout_index"]
                .as_u64()
                .ok_or_else(|| error("Invalid table index"))? as usize;
            let table = page.plan["model"]["layout_dets"][index].clone();
            let plan = op(
                "plan_native_table_rule",
                json!({"table":table,"scale":page.image.scale,"config":{"use_rule_table":r.use_rule_table},"rotate_label":page.facts["rotate_label"],"ocr_enable":!r.native_text,"has_source_pdf":r.native_text&&!array(&page.facts["blocks"])?.is_empty(),"formulas":formulas,"checkboxes":checkboxes,"images":guard}),
            )?;
            let mut accepted = false;
            let mut reason = plan["reason"].clone();
            let mut error_type = None;
            if plan["attempt"] == true {
                let recovered = (|| -> Result<Value, Error> {
                    if geometry.is_none() {
                        let started = std::time::Instant::now();
                        geometry = Some(native_adapter::prepare_table_page(
                            library, data, page.index, None,
                        )?);
                        timings.add("table_native_geometry", started, 1);
                    }
                    let started = std::time::Instant::now();
                    let g = geometry.as_ref().unwrap();
                    let angle = op(
                        "native_pdf_table_text_angle",
                        json!({"chars":g["chars"],"table_bbox":plan["table_bbox"]}),
                    )?;
                    let recovered = native_adapter::recover_table_region_owned(
                        json!({"table_input":{"page_size":g["page_size"],"chars":g["chars"],"drawing_lines":g["drawing_lines"],"rectangles":g["rectangles"],"table_bbox":plan["table_bbox"],"angle":angle},"tight_bboxes":g["tight_bboxes"],"origins":g["origins"]}),
                    )?;
                    timings.add("table_native_recovery", started, 1);
                    if !recovered["recovery_error"].is_null() {
                        return Err(error("Native table recovery failed"));
                    }
                    let result = if recovered.is_null() {
                        Value::Null
                    } else {
                        json!({"html":recovered["content"],"source":recovered["result"]["source"],"confidence":recovered["result"]["confidence"],"diagnostics":recovered["result"]["diagnostics"]})
                    };
                    op(
                        "accept_native_table_rule",
                        json!({"table":table,"result":result,"threshold":r.rule_table_score_threshold}),
                    )
                })();
                match recovered {
                    Ok(v) => {
                        accepted = v["accepted"] == true;
                        reason = json!(if accepted { "accepted" } else { "not_accepted" });
                        if accepted {
                            page.plan["model"]["layout_dets"][index] = v["table"].clone();
                        }
                    }
                    Err(_) => {
                        reason = json!("recovery_error");
                        error_type = Some("RapidDocError");
                    }
                }
            }
            let mut route = json!({"page_index":page.index,"layout_index":index,"method":if accepted{"docvortex_native"}else{"model"},"reason":reason});
            if let Some(e) = error_type {
                route["error_type"] = json!(e)
            }
            routes.push(route);
            if accepted {
                continue;
            }
            let started = std::time::Instant::now();
            let hints = op(
                "plan_table_hints",
                json!({"page":context,"table":table,"useful_list":task["useful_list"],"formulas":formulas,"checkboxes":checkboxes,"scale":page.image.scale,"formula_enabled":r.formula_enabled,"image_enabled":true,"extract_original_image":false}),
            )?;
            timings.add("table_hints", started, 1);
            context = hints["page"].clone();
            let source = image(&page.images, &task["source"])?;
            let started = std::time::Instant::now();
            let (png, rows) = self.table_text(
                source,
                page,
                &hints["table"],
                &task["useful_list"],
                &hints["formulas"],
                r,
                resolve,
            )?;
            timings.add("table_text", started, 1);
            if r.rule_table_engine == "docvortex_ocr" && r.use_rule_table && !r.table_force_ocr {
                let eligible = op(
                    "plan_native_table_rule",
                    json!({"table":table,"scale":page.image.scale,
                    "config":{"use_rule_table":true},"rotate_label":"0","ocr_enable":false,"has_source_pdf":true,
                    "formulas":formulas,"checkboxes":checkboxes,"images":guard}),
                )?;
                if eligible["attempt"] == true {
                    let started = std::time::Instant::now();
                    let recovered = crate::legacy_model::table_ocr_rule::recover(&png, &rows);
                    timings.add("table_ocr_rule", started, 1);
                    match recovered {
                        Ok(result)
                            if !result.is_null()
                                && result["confidence"]
                                    .as_f64()
                                    .is_some_and(|v| v >= r.rule_table_score_threshold) =>
                        {
                            let mut output = hints["table"].clone();
                            output["html"] = result["html"].clone();
                            output["rule_table_score"] = result["confidence"].clone();
                            output["table_parse_method"] = json!("docvortex_ocr");
                            output.as_object_mut().unwrap().remove("layout_image_list");
                            page.plan["model"]["layout_dets"][index] = output;
                            if let Some(route) = routes.last_mut() {
                                route["method"] = json!("docvortex_ocr");
                                route["reason"] = json!("accepted");
                            }
                            continue;
                        }
                        Err(e) => {
                            if let Some(route) = routes.last_mut() {
                                route["ocr_rule_reason"] = json!(e.message);
                            }
                        }
                        _ => {
                            if let Some(route) = routes.last_mut() {
                                route["ocr_rule_reason"] = json!("not_accepted");
                            }
                        }
                    }
                }
            }
            let started = std::time::Instant::now();
            let (html, text_fallback, quality) =
                self.table_model(&png, rows, &hints, r, resolve, timings)?;
            if !quality.is_null() {
                if let Some(route) = routes.last_mut() {
                    route["text_retention"] = quality;
                }
            }
            timings.add("table_models", started, 1);
            let all_formulas = json!(array(&formulas)?
                .iter()
                .chain(array(&checkboxes)?)
                .cloned()
                .collect::<Vec<_>>());
            page.plan["model"]["layout_dets"][index] = op(
                "apply_table_prediction",
                json!({"table":hints["table"],"html_code":html,"text_fallback":text_fallback,"formulas":all_formulas,"fill_images":hints["fill_images"],"scale":page.image.scale}),
            )?;
            for f in array(&hints["fill_images"])? {
                filled.push(TableFillImage {
                    uuid: f["uuid"]
                        .as_str()
                        .ok_or_else(|| error("Invalid fill image"))?
                        .into(),
                    bbox: serde_json::from_value(f["bbox"].clone()).map_err(error)?,
                    png: (**image(&page.images, &f["source"])?).clone(),
                });
            }
        }
        Ok(filled)
    }
}
