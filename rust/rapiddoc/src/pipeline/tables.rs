//! Original native-rule guard, direction, OCR, structure routing and image hints.
use super::*;
use super::table_recovery;
use crate::legacy_model::{native_adapter, table_direction, table_hints, table_native, table_text};
use crate::legacy_page::assets::{OriginalImage, TableFillImage};
use rapiddoc_inference::{
    table_classification::Variant as Classifier, table_structure::Variant as Structure, wired_match,
};

use super::table_evidence::{Grid, GridCell, Source};

pub(super) fn select_candidate(
    main: &Value,
    main_grid: Option<&Grid>,
    alternative: &Value,
    alternate_grid: Option<&Grid>,
    rows: &Value,
    excluded: &[String],
    source: Option<&Source>,
    r: &Request,
) -> (bool, Value) {
    let first = super::table_fallback::quality(main, rows, excluded);
    let second = super::table_fallback::quality(alternative, rows, excluded);
    let primary = super::table_fallback::structure(main, rows, excluded);
    let alt = super::table_fallback::structure(alternative, rows, excluded);
    let primary_score = source.zip(main_grid).map(|(s, g)| s.score(g));
    let score = source.zip(alternate_grid).map(|(s, g)| s.score(g));
    let comparable = main_grid.is_some() && alternate_grid.is_some();
    let (gains, losses) =
        source
            .zip(main_grid.zip(alternate_grid))
            .map_or((0, 0), |(s, (a, b))| {
                let (ac, ap) = s.legacy_grid(a);
                let (_, bp) = s.legacy_grid(b);
                (ac.intersection(&bp).count(), ap.difference(&bp).count())
            });
    let better_grid = gains >= losses + 3 && gains >= 2 * losses && alt.rows <= primary.rows + 1;
    let more_consistent_rows = losses == 0
        && alt.rows < primary.rows
        && alt.full_width_rows < primary.full_width_rows
        && alt.row_misalignment < primary.row_misalignment;
    let width = |g: &Grid| g.legacy.as_deref().unwrap_or(g).width;
    let extra_columns =
        comparable && width(alternate_grid.unwrap()) > width(main_grid.unwrap()) && gains < 3;
    let margin = if better_grid && losses == 0 && gains >= 5 {
        0.05
    } else {
        0.03
    };
    let usable = second.valid && !second.needs_fallback(r.fallback_text_loss_threshold);
    let more_text = first.needs_fallback(r.fallback_text_loss_threshold)
        && second.can_replace(&first, r.fallback_text_loss_threshold, r.fallback_to_text);
    let ruled_rows = source
        .zip(main_grid)
        .map(|(s, g)| s.ruled_empty_rows(g, &primary.empty_indices))
        .unwrap_or_default();
    let protected_empty_rows = first.valid
        && !ruled_rows.is_empty()
        && primary.rows.saturating_sub(alt.rows) > primary.empty.saturating_sub(ruled_rows.len());
    let mut weak_number_losses = BTreeMap::<String, usize>::new();
    let mut lost_protected_numbers = false;
    let mut better_evidence = false;
    let mut protected_structure = false;
    let mut ruling_disagrees = false;
    let mut removes_noise_rows = alt.removes_empty_rows(&primary, &ruled_rows);
    let mut unsupported = extra_columns;
    let mut report = json!({"fallback_ran":true,"fallback_valid":second.valid,"fallback_retained":second.retained,
        "grid_gains":gains,"grid_losses":losses});
    if let Some((a, b)) = primary_score.as_ref().zip(score.as_ref()) {
        let [text_gain, line_gain, gap_gain] = b.delta(a);
        let recovered = a.missing.difference(&b.missing).count();
        let loses_ruling = b.loses_supported_lines(a, line_gain);
        // 同一份文字和间距证据没有改善时，旧的结构推断不能压过原图框线退步。
        ruling_disagrees =
            line_gain <= -0.15 && gap_gain <= 0. && text_gain <= 0. && gains <= losses;
        removes_noise_rows &= !loses_ruling;
        // 只有无框线空行被清理且 OCR 归属改善，才允许旧规则放宽新增列限制。
        let cleanup_supports_columns =
            removes_noise_rows && ruled_rows.is_empty() && text_gain >= 0.65 && gap_gain >= 0.;
        // 行结构改善需要独立间距证据，且框线和文字没有退步。
        let consistent_layout =
            more_consistent_rows && line_gain >= 0. && gap_gain >= 0.35 && text_gain >= 0.;
        unsupported &= !(cleanup_supports_columns || consistent_layout);
        let shrinks = alt.rows as f64 <= 0.75 * primary.rows as f64
            || (comparable
                && width(alternate_grid.unwrap()) as f64
                    <= 0.75 * width(main_grid.unwrap()) as f64);
        protected_structure = loses_ruling && shrinks && recovered == 0 && gap_gain < 0.5;
        let geometry = line_gain + gap_gain;
        let row_ok = alt.rows <= primary.rows
            || b.line_gain(a, Some(1)) >= 0.35
            || b.separator_gains(a, 1) >= 2
            || recovered >= 2;
        let col_ok =
            !unsupported || b.separator_gains(a, 0) >= 3 || b.line_gain(a, Some(0)) >= 0.55;
        let agrees = line_gain >= 0.35 && gap_gain >= 0.15 && text_gain >= 0.;
        let same_size = comparable
            && width(main_grid.unwrap()) == width(alternate_grid.unwrap())
            && primary.rows == alt.rows;
        let local = same_size
            && b.line_gain(a, None) >= 0.5
            && line_gain > 0.
            && gap_gain >= -0.2
            && text_gain >= 0.5;
        let restored = same_size
            && primary.empty >= alt.empty + 2
            && primary.empty * 4 >= primary.rows
            && second.retained >= first.retained
            && text_gain >= 0.65
            && line_gain >= 0.
            && gap_gain >= 0.;
        // 同一原图、同一 OCR 上比较候选；文字、框线、间距交叉验证，禁止丢失原有数字。
        // 新证据采用独立的 3% 保留率门槛，不能借用旧网格检查放宽后的 5% 门槛。
        better_evidence = usable
            && second.retained as f64 >= first.retained as f64 - first.total as f64 * 0.03
            && 0.5 * text_gain + geometry >= 0.65
            && (geometry >= 0.5 || local || recovered >= 2 && geometry >= 0.)
            && row_ok
            && col_ok
            && (losses < gains + 4 || agrees);
        better_evidence |= second.valid && restored && losses <= gains;
        for &id in b.missing_numbers.difference(&a.missing_numbers) {
            if a.weak_numbers.contains(&id) {
                // 擦边归属仍计入文字证据，只放宽这一处的数字计数。
                let number = crate::processing::inline_output::unescape_html(
                    rows[1][id].as_str().unwrap_or_default(),
                )
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
                *weak_number_losses.entry(number).or_default() += 1;
            } else {
                lost_protected_numbers = true;
            }
        }
        report["fallback_evidence"] = b.report();
        report["evidence_delta"] = json!([text_gain, line_gain, gap_gain]);
    }
    let better_structure = usable
        && first.valid
        && !protected_structure
        && !ruling_disagrees
        && (better_grid
            || (losses < gains + 4
                && !unsupported
                && (alt.improves(&primary) || removes_noise_rows)))
        && second.retained as f64 >= first.retained as f64 - first.total as f64 * margin;
    let keeps_numbers = !lost_protected_numbers
        && primary.numbers.iter().all(|(number, count)| {
            alt.numbers.get(number).copied().unwrap_or(0)
                + weak_number_losses.get(number).copied().unwrap_or(0)
                >= *count
        });
    report["keeps_numbers"] = json!(keeps_numbers);
    report["lost_protected_numbers"] = json!(lost_protected_numbers);
    report["protected_empty_rows"] = json!(protected_empty_rows);
    report["protected_structure"] = json!(protected_structure);
    report["ruling_disagrees"] = json!(ruling_disagrees);
    let repaired = !first.valid && second.valid && (usable || more_text);
    let unsupported_splits = if first.valid && !more_text && alt.rows > primary.rows {
        source
            .zip(main_grid.zip(alternate_grid))
            .map_or(0, |(source, (main, alternative))| {
                source.unsupported_row_splits(main, alternative)
            })
    } else {
        0
    };
    report["unsupported_row_splits"] = json!(unsupported_splits);
    let improves = !protected_empty_rows
        && unsupported_splits == 0
        && (more_text || repaired || better_structure || better_evidence);
    report["improves_before_number_check"] = json!(improves);
    (keeps_numbers && improves, report)
}

struct TableCandidate {
    grid: Option<Grid>,
    html: Value,
    rejected: Option<String>,
}
/// Only errors produced while interpreting candidate HTML are recoverable.
/// Session, model, device and resource errors must propagate to the caller.
fn candidate_html(result: Result<Value, Error>) -> Result<TableCandidate, Error> {
    match result {
        Ok(html) => Ok(TableCandidate {
            grid: None,
            html,
            rejected: None,
        }),
        Err(e) if matches!(e.code, ErrorCode::Unsupported | ErrorCode::InvalidInput) => {
            Ok(TableCandidate {
                grid: None,
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
    pub(super) fn probe_table_direction(
        &mut self,
        png: &[u8],
        page: &Page,
        table: &Value,
        masks: &Value,
        resolve: &mut Resolver<'_>,
    ) -> Result<(Value, String), Error> {
        let (width, height) = table_direction::dimensions(png)?;
        let boxes = self.table_boxes(png, masks, resolve)?;
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
            let prediction = value(&prediction)?;
            label = table_direction::select_model_rotation(&prediction)?
                .as_str()
                .ok_or_else(|| error("Invalid table orientation decision"))?
                .to_owned();
            if label == "0"
                && matches!(prediction["rotation_degrees"].as_u64(), Some(90 | 270))
                && prediction["score"].as_f64().unwrap_or_default() >= 0.7
                && prediction["margin"].as_f64().unwrap_or_default() >= 0.4
            {
                let model_label = if prediction["rotation_degrees"] == 90 {
                    "90"
                } else {
                    "270"
                };
                label = self
                    .verify_table_direction(png, &boxes, model_label, resolve)?
                    .into();
            }
        }
        Ok((boxes, label))
    }

    fn table_text(
        &mut self,
        png: &[u8],
        page: &mut Page,
        table: &Value,
        useful: &Value,
        masks: &mut Value,
        layout_index: usize,
        r: &Request,
        resolve: &mut Resolver<'_>,
    ) -> Result<(Vec<u8>, Value, String), Error> {
        let (width, height) = table_direction::dimensions(png)?;
        let (mut boxes, label) = match page.table_formula_probes.remove(&layout_index) {
            Some(probe) => probe,
            None => self.probe_table_direction(png, page, table, masks, resolve)?,
        };
        let png = table_direction::oriented_png(png, &label)?;
        if matches!(label.as_str(), "90" | "270") {
            super::table_formula::rotate_hints(masks, width, height, &label)?;
            let after = self.table_boxes(&png, masks, resolve)?;
            boxes = wire(&table_recovery::restore_columns(
                &serde_json::from_value::<Vec<_>>(boxes).map_err(error)?,
                serde_json::from_value(after).map_err(error)?,
                width,
                height,
                label == "270",
            ))?;
        }
        let mut rows = json!([]);
        if !array(&boxes)?.is_empty()
            && r.native_text
            && !r.table_force_ocr
            && label == "0"
            && !matches!(
                page.facts["rotate_label"].as_str(),
                Some("90" | "180" | "270")
            )
        {
            rows = table_native::extract(
                &png,
                &json!({"boxes":boxes,"native_page":page.facts,"table":table,"useful_list":useful,"scale":page.image.scale,"word_boxes":false}),
            )?;
        }
        if array(&rows)?.is_empty() && !array(&boxes)?.is_empty() {
            rows = self.table_rows(&png, &boxes, false, resolve)?;
        }
        Ok((png.into_owned(), rows, label))
    }

    fn verify_table_direction(
        &mut self,
        png: &[u8],
        boxes: &Value,
        label: &str,
        resolve: &mut Resolver<'_>,
    ) -> Result<&'static str, Error> {
        use rapiddoc_inference::{crop::PipelineImage, recognition_batch::Image};
        let boxes: Vec<[[f32; 2]; 4]> = serde_json::from_value(boxes.clone()).map_err(error)?;
        let length = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
        let mut candidates: Vec<_> = boxes
            .into_iter()
            .filter_map(|q| {
                let w = length(q[0], q[1]).max(length(q[2], q[3]));
                let h = length(q[0], q[3]).max(length(q[1], q[2]));
                (w >= 2. && h >= 1.4 * w && h <= 10. * w).then_some((h / w, q))
            })
            .collect();
        if candidates.len() < 2 {
            return Ok("0");
        }
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        let source = PipelineImage::decode(png, 4_000_000)?;
        let mut images = Vec::new();
        for (_, q) in candidates.into_iter().take(4) {
            let crop = source.crop_pixels_with_ratio(q, 4_000_000, 1.4)?;
            if crop.width > crop.height {
                images.push(Image {
                    bgr: crop.bgr,
                    width: crop.width,
                    height: crop.height,
                });
            }
        }
        let n = images.len();
        if n < 2 {
            return Ok("0");
        }
        for i in 0..n {
            let Image { bgr, width, height } = &images[i];
            let flipped = bgr.chunks_exact(3).rev().flatten().copied().collect();
            images.push(Image {
                bgr: flipped,
                width: *width,
                height: *height,
            });
        }
        let results = self
            .models
            .with_lazy("recognizer", resolve, |m| m.recognize_images(&images, 6))?;
        Ok(
            match table_recovery::direction(
                results
                    .iter()
                    .take(n)
                    .zip(results.iter().skip(n))
                    .map(|(l, r)| {
                        (
                            l.text.as_str(),
                            l.confidence.into(),
                            r.text.as_str(),
                            r.confidence.into(),
                        )
                    }),
                label,
            ) {
                "90" => "90",
                "270" => "270",
                _ => "0",
            },
        )
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
        excluded: &[String],
        resolve: &mut Resolver<'_>,
        timings: &mut Timings,
    ) -> Result<(Value, Option<Grid>), Error> {
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
                            .map(crate::processing::inline_output::unescape_html)
                            .ok_or_else(|| error("Invalid TATR OCR text"))?,
                        block_num: 0,
                        line_num: i,
                        span_num: 0,
                        confidence: rows[2][i].as_f64().unwrap_or_default(),
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
            let (width, height) = table_direction::dimensions(png)?;
            let limits = [width as f64, height as f64, width as f64, height as f64];
            let grid = Grid::new(
                cells
                    .iter()
                    .map(|cell| GridCell {
                        // 对齐 Python cells_to_result：裁图内坐标、float32 网格。
                        bbox: std::array::from_fn(|axis| {
                            cell.bbox[axis].clamp(0., limits[axis]) as f32 as f64
                        }),
                        logic: [
                            cell.row_nums[0],
                            *cell.row_nums.last().unwrap(),
                            cell.column_nums[0],
                            *cell.column_nums.last().unwrap(),
                        ],
                        tokens: cell
                            .token_indices()
                            .iter()
                            .map(|&i| tokens[i].line_num)
                            .collect(),
                    })
                    .collect(),
            );
            Ok((json!(html), grid))
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
                return Ok((json!(""), None));
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
            let rects: Vec<_> = polygons.iter().map(table_recovery::bounds).collect();
            let groups = table_recovery::cross_cell_groups(png, &rects, &points, &ocr, excluded)?;
            let mut verified = BTreeMap::new();
            if !groups.is_empty() {
                let started = std::time::Instant::now();
                let cells: Vec<_> = groups
                    .iter()
                    .flat_map(|(_, cells)| cells.iter().copied())
                    .collect();
                let crops: Vec<_> = cells.iter().map(|&cell| polygons[cell]).collect();
                let results = self.models.with_lazy("recognizer", resolve, |m| {
                    m.recognize_blank_cells_png(png, &crops, &vec![true; cells.len()])
                })?;
                timings.add("table_cross_cell_rec", started, cells.len());
                let mut results = results.into_iter();
                for (token, cells) in groups {
                    let group: Vec<_> = results.by_ref().take(cells.len()).collect();
                    if group.len() == cells.len()
                        && group.iter().all(|r| r.accepted)
                        && group
                            .iter()
                            .map(|r| table_recovery::compact(&r.text))
                            .collect::<String>()
                            == table_recovery::compact(&ocr[token].text)
                    {
                        verified.insert(token, cells.into_iter().zip(group).collect::<Vec<_>>());
                    }
                }
            }
            let originals: Vec<_> = (0..ocr.len())
                .filter(|i| !verified.contains_key(i))
                .collect();
            let remaining: Vec<_> = originals.iter().map(|&i| ocr[i].clone()).collect();
            let mut matched = wired_match::match_cells(&polygons, &remaining)?;
            for fragment in matched.cells.iter_mut().flatten() {
                fragment.ocr_index = fragment.ocr_index.map(|i| originals[i]);
            }
            matched
                .unmatched
                .iter_mut()
                .for_each(|i| *i = originals[*i]);
            let mut verified_splits = 0;
            for (token, group) in verified {
                verified_splits += group.len() - 1;
                for (cell, result) in group {
                    let p = polygons[cell];
                    matched.cells[cell].push(wired_match::Fragment {
                        ocr_index: Some(token),
                        points: [p[0], p[3], p[2], p[1]],
                        text: result.text,
                        confidence: result.confidence,
                        split: false,
                    });
                }
            }
            for index in matched.unmatched.clone() {
                let item = &ocr[index];
                if excluded.contains(&item.text) {
                    continue;
                }
                let joined = table_recovery::compact(&item.text);
                let chars: Vec<_> = joined.chars().collect();
                let Some(ids) =
                    table_recovery::joined_candidates(&rects, &points, &matched, item, chars.len())
                else {
                    continue;
                };
                let candidates: Vec<_> = ids.iter().map(|&i| polygons[i]).collect();
                let mut results = self.models.with_lazy("recognizer", resolve, |m| {
                    m.recognize_blank_cells_png(&png, &candidates, &vec![true; ids.len()])
                })?;
                let texts: Vec<_> = results
                    .iter()
                    .map(|r| {
                        if r.accepted {
                            table_recovery::compact(&r.text)
                        } else {
                            String::new()
                        }
                    })
                    .collect();
                if let Some(missing) = texts.iter().position(String::is_empty) {
                    if ids.len() < 4
                        || chars.len() != ids.len()
                        || texts
                            .iter()
                            .enumerate()
                            .any(|(i, t)| i != missing && !t.chars().eq([chars[i]]))
                    {
                        continue;
                    }
                    results[missing].text = chars[missing].to_string();
                    results[missing].confidence = item.confidence;
                    results[missing].accepted = true;
                } else if texts.concat() != joined {
                    continue;
                }
                for r in &mut results {
                    r.cell_index = ids[r.cell_index];
                }
                wired_match::fill_blank(&mut matched, &polygons, &results)?;
                for &cell in &ids {
                    for fragment in &mut matched.cells[cell] {
                        fragment.ocr_index = Some(index);
                    }
                }
            }
            wired_match::fill_blank(&mut matched, &polygons, &[])?;
            let mut emitted = Vec::new();
            let html = rapiddoc_render::wired::legacy_html_with_grid(
                &points,
                &polygons,
                &matched.cell_lines,
                &mut emitted,
            )?;
            let mut grid = Grid::new(
                emitted
                    .into_iter()
                    .map(|(i, logic)| GridCell {
                        bbox: table_recovery::bounds(&polygons[i]),
                        logic,
                        tokens: matched.cells[i]
                            .iter()
                            .filter_map(|f| f.ocr_index)
                            .collect(),
                    })
                    .collect(),
            );
            if let Some(grid) = &mut grid {
                grid.verified_splits = verified_splits;
                grid.legacy = Grid::new(
                    polygons
                        .iter()
                        .zip(&points)
                        .map(|(p, &logic)| GridCell {
                            bbox: table_recovery::bounds(p),
                            logic,
                            tokens: Vec::new(),
                        })
                        .collect(),
                )
                .map(Box::new);
            }
            Ok((json!(html), grid))
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
            let result = op(
                "legacy_table_match",
                json!({"collect_evidence":true,"structure":{"tokens":structure.tokens,"confidence":structure.confidence,"cell_bboxes":boxes},"ocr_result":rows,"width":width,"height":height}),
            )?;
            let logic: Vec<[usize; 4]> =
                serde_json::from_value(result["logic_points"].clone()).map_err(error)?;
            let grid = Grid::new(
                boxes
                    .iter()
                    .zip(logic)
                    .enumerate()
                    .filter_map(|(i, (bbox, logic))| {
                        let tokens = result["matched"][i.to_string()]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter_map(|id| {
                                result["token_ids"][id.as_u64()? as usize]
                                    .as_u64()
                                    .map(|v| v as usize)
                            })
                            .collect();
                        Some(GridCell {
                            bbox: super::table_evidence::cell_bounds(bbox)?,
                            logic,
                            tokens,
                        })
                    })
                    .collect(),
            );
            Ok((result["raw_html"].clone(), grid))
        }
    }
    fn table_model(
        &mut self,
        png: &[u8],
        rows: Value,
        hints: &Value,
        vector: &[super::table_evidence::Segment],
        r: &Request,
        resolve: &mut Resolver<'_>,
        timings: &mut Timings,
        allow_stacked_split: bool,
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
        let excluded = super::table_fallback::excluded(hints);
        let main_candidate = self.table_candidate(
            &png,
            &rows,
            &r.table_variant,
            "table_model",
            &excluded,
            r,
            resolve,
            timings,
        )?;
        let main_grid = main_candidate.grid;
        let mut main = main_candidate.html;
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
        let mut first = super::table_fallback::quality(&main, &rows, &excluded);
        if !first.valid {
            if let Some(html) = main
                .as_str()
                .and_then(super::table_fallback::repair_local_grid)
            {
                main = json!(html);
                first = super::table_fallback::quality(&main, &rows, &excluded);
            }
        }
        let mut report = json!({"main":r.table_variant,"main_valid":first.valid,"ocr_chars":first.total,"main_retained":first.retained,"fallback_ran":false,"selected":"main"});
        if let Some(reason) = &rejected {
            report["main_rejected"] = json!(reason);
        }
        let has_backup = r
            .fallback_table_variant
            .as_deref()
            .is_some_and(|v| v != r.table_variant);
        let primary = super::table_fallback::structure(&main, &rows, &excluded);
        let source = if has_backup && main_grid.is_some() {
            Some(Source::new(&png, &rows, &excluded, vector)?)
        } else {
            None
        };
        let primary_score = source
            .as_ref()
            .zip(main_grid.as_ref())
            .map(|(s, g)| s.score(g));
        let primary_legacy = source
            .as_ref()
            .zip(main_grid.as_ref())
            .map(|(s, g)| s.legacy_grid(g));
        let text_loss = first.needs_fallback(r.fallback_text_loss_threshold);
        let ruled_rows = source
            .as_ref()
            .zip(main_grid.as_ref())
            .map(|(s, g)| s.ruled_empty_rows(g, &primary.empty_indices))
            .unwrap_or_default();
        report["ruled_blank_rows"] = json!(ruled_rows.len());
        let suspicious = primary.suspicious(&ruled_rows)
            || primary_score.as_ref().is_some_and(|s| s.suspicious())
            || primary_legacy.as_ref().is_some_and(|s| s.0.len() >= 2);
        if let Some(score) = &primary_score {
            report["main_evidence"] = score.report();
        }
        if allow_stacked_split
            && excluded.is_empty()
            && primary_score.as_ref().is_some_and(|s| s.suspicious())
        {
            if let Some(html) = self.split_stacked_tables(
                &png,
                &rows,
                vector,
                source.as_ref().unwrap(),
                &first,
                primary_score.as_ref().unwrap().text_penalty(),
                r,
                resolve,
                timings,
            )? {
                report["selected"] = json!("split");
                return Ok((json!(html), false, report));
            }
        }
        if rejected.is_none() && !text_loss && !suspicious {
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
                &excluded,
                r,
                resolve,
                timings,
            )?;
            timings.add("table_fallback", started, 1);
            if let Some(reason) = &alternative.rejected {
                report["fallback_rejected"] = json!(reason);
            }
            let mut alternate_grid = alternative.grid;
            let mut alternative = alternative.html;
            if let Some(((source, main_grid), candidate_grid)) = source
                .as_ref()
                .zip(main_grid.as_ref())
                .zip(alternate_grid.as_ref())
            {
                let second = super::table_fallback::quality(&alternative, &rows, &excluded);
                let candidate_score = source.score(candidate_grid);
                let formula_source = if second.valid
                    && candidate_score.missing.is_empty()
                    && !excluded.is_empty()
                    && hints["fill_images"].as_array().is_none_or(Vec::is_empty)
                {
                    Some(Source::new(&png, &rows, &[], vector)?)
                } else {
                    None
                };
                if second.valid
                    && candidate_score.missing.is_empty()
                    && hints["fill_images"].as_array().is_none_or(Vec::is_empty)
                {
                    if let Some(joined) = table_recovery::join_aligned_row_bands(
                        alternative.as_str().unwrap_or_default(),
                        candidate_grid,
                        formula_source.as_ref().unwrap_or(source),
                        array(&rows[1])?,
                    ) {
                        let joined = json!(joined);
                        if super::table_fallback::quality(&joined, &rows, &excluded).retained
                            >= first.retained.max(second.retained)
                            && super::table_fallback::quality(&joined, &rows, &[]).retained
                                >= super::table_fallback::quality(&alternative, &rows, &[]).retained
                        {
                            report["fallback"] = json!(backup);
                            report["fallback_ran"] = json!(true);
                            report["selected"] = json!("fallback");
                            report["fallback_repair"] = json!("aligned_row_bands");
                            return Ok((joined, false, report));
                        }
                    }
                }
                let lost = candidate_score
                    .missing
                    .difference(&source.score(main_grid).missing)
                    .copied()
                    .collect();
                let restored = table_recovery::restore_trailing_row(
                    main.as_str().unwrap_or_default(),
                    alternative.as_str().unwrap_or_default(),
                    main_grid,
                    candidate_grid,
                    source,
                    &lost,
                );
                let repair = if restored.is_some() {
                    "trailing_row"
                } else {
                    "internal_rows"
                };
                let restored = restored.or_else(|| {
                    if first.valid {
                        return None;
                    }
                    table_recovery::restore_internal_rows(
                        alternative.as_str()?,
                        candidate_grid,
                        source,
                        &lost,
                        rows[1].as_array()?,
                    )
                });
                if let Some((html, grid)) = restored {
                    let html = json!(html);
                    let restored = super::table_fallback::quality(&html, &rows, &excluded);
                    if restored.valid
                        && restored.retained >= first.retained.max(second.retained)
                        && source
                            .score(&grid)
                            .missing
                            .is_subset(&source.score(main_grid).missing)
                    {
                        alternative = html;
                        alternate_grid = Some(grid);
                        report["fallback_repair"] = json!(repair);
                    }
                }
            }
            let (selected, details) = select_candidate(
                &main,
                main_grid.as_ref(),
                &alternative,
                alternate_grid.as_ref(),
                &rows,
                &excluded,
                source.as_ref(),
                r,
            );
            if let Some(fields) = details.as_object() {
                report.as_object_mut().unwrap().extend(fields.clone());
            }
            report["fallback"] = json!(backup);
            if selected {
                report["selected"] = json!("fallback");
                return Ok((alternative, false, report));
            }
            // GPU 候选的网格有改善但漏了独立数字时，用同一裁图和 OCR 重试。
            // CPU 候选仍须通过完整择优和数字保留检查。
            if backup == "table-transformer"
                && details["improves_before_number_check"] == true
                && details["lost_protected_numbers"] == true
            {
                if let Some(mut descriptor) = self.models.cuda_descriptor("fallback_table_model")? {
                    descriptor.device = "cpu".into();
                    descriptor.optimization = None;
                    descriptor.operator_libraries.clear();
                    let started = std::time::Instant::now();
                    let retry = self.table_candidate(
                        &png,
                        &rows,
                        backup,
                        "fallback_table_model_cpu_retry",
                        &excluded,
                        r,
                        &mut |_| Ok(descriptor.clone()),
                        timings,
                    );
                    timings.add("table_fallback_cpu_retry", started, 1);
                    match retry {
                        Ok(retry) => {
                            let (accepted, evidence) = select_candidate(
                                &main,
                                main_grid.as_ref(),
                                &retry.html,
                                retry.grid.as_ref(),
                                &rows,
                                &excluded,
                                source.as_ref(),
                                r,
                            );
                            if accepted {
                                report["cpu_retry"] =
                                    json!({"selected":true,"cuda_evidence":details});
                                if let Some(fields) = evidence.as_object() {
                                    report.as_object_mut().unwrap().extend(fields.clone());
                                }
                                report["selected"] = json!("fallback");
                                report["fallback_device"] = json!("cpu");
                                return Ok((retry.html, false, report));
                            }
                            report["cpu_retry"] = json!({"selected":false,"evidence":evidence});
                        }
                        Err(error) => report["cpu_retry"] = json!({"error":error.message}),
                    }
                }
            }
        }
        if r.fallback_to_text && text_loss {
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
    fn split_stacked_tables(
        &mut self,
        png: &[u8],
        rows: &Value,
        vector: &[super::table_evidence::Segment],
        source: &Source,
        first: &super::table_fallback::Quality,
        primary_text_penalty: f64,
        r: &Request,
        resolve: &mut Resolver<'_>,
        timings: &mut Timings,
    ) -> Result<Option<String>, Error> {
        use super::table_evidence::Segment;
        let (w, h) = table_direction::dimensions(png)?;
        let Some(cut) = source.stacked_boundary(rows, h as f64, w as f64) else {
            return Ok(None);
        };
        let image = image::load_from_memory(png).map_err(error)?;
        let mut sections = Vec::new();
        let mut retained = 0;
        for (top, bottom) in [(0, cut), (cut, h as u32)] {
            let (lo, hi) = (top as f64, bottom as f64);
            let mut local = [Vec::new(), Vec::new(), Vec::new()];
            for (id, item) in array(&rows[0])?.iter().enumerate() {
                let mut points: Vec<[f64; 2]> =
                    serde_json::from_value(item.clone()).map_err(error)?;
                let y0 = points.iter().map(|p| p[1]).min_by(f64::total_cmp);
                let y1 = points.iter().map(|p| p[1]).max_by(f64::total_cmp);
                let Some((y0, y1)) = y0.zip(y1) else {
                    continue;
                };
                if (lo..hi).contains(&((y0 + y1) / 2.)) {
                    if y0 < lo || y1 > hi {
                        return Ok(None);
                    }
                    points.iter_mut().for_each(|p| p[1] -= lo);
                    local[0].push(json!(points));
                    local[1].push(rows[1][id].clone());
                    local[2].push(rows[2][id].clone());
                }
            }
            if local[1].is_empty() {
                return Ok(None);
            }
            let lines: Vec<_> = vector
                .iter()
                .filter_map(|&Segment(axis, pos, a, b)| {
                    if axis == 1 {
                        (lo <= pos && pos <= hi).then_some(Segment(1, pos - lo, a, b))
                    } else {
                        (a.max(lo) < b.min(hi)).then_some(Segment(
                            0,
                            pos,
                            a.max(lo) - lo,
                            b.min(hi) - lo,
                        ))
                    }
                })
                .collect();
            let mut crop = std::io::Cursor::new(Vec::new());
            image
                .crop_imm(0, top, w as u32, bottom - top)
                .write_to(&mut crop, image::ImageFormat::Png)
                .map_err(error)?;
            let local = json!(local);
            let (html, _, _) = self.table_model(
                crop.get_ref(),
                local.clone(),
                &Value::Null,
                &lines,
                r,
                resolve,
                timings,
                false,
            )?;
            let q = super::table_fallback::quality(&html, &local, &[]);
            if !q.valid || q.needs_fallback(r.fallback_text_loss_threshold) {
                return Ok(None);
            }
            let Some(html) = html.as_str() else {
                return Ok(None);
            };
            let (Some(start), Some(end)) = (html.find("<table"), html.rfind("</table>")) else {
                return Ok(None);
            };
            retained += q.retained;
            sections.push(html[start..end + "</table>".len()].to_owned());
        }
        Ok((retained > first.retained
            || (retained == first.retained && primary_text_penalty >= 2.))
            .then(|| sections.join("\n\n")))
    }

    fn table_candidate(
        &mut self,
        png: &[u8],
        rows: &Value,
        variant: &str,
        role: &str,
        excluded: &[String],
        r: &Request,
        resolve: &mut Resolver<'_>,
        timings: &mut Timings,
    ) -> Result<TableCandidate, Error> {
        let (html, grid) = if let Some(variant) = variant.strip_prefix("unet-") {
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
                    excluded,
                    resolve,
                    timings,
                )?;
                candidates.insert(route, html);
            }
            let selected = if compare {
                let (wired_html, wired_grid) = &candidates["wired"];
                let (wireless_html, wireless_grid) = &candidates["wireless"];
                let prefer_wired = wired_grid.as_ref().zip(wireless_grid.as_ref()).is_some_and(
                    |(wired, wireless)| {
                        if wired.verified_splits == 0 {
                            return false;
                        }
                        let logical_grid = |g: &Grid| {
                            let mut points: Vec<_> = g
                                .legacy
                                .as_deref()
                                .unwrap_or(g)
                                .cells
                                .iter()
                                .map(|c| c.logic)
                                .collect();
                            points.sort_unstable();
                            points
                        };
                        let a = super::table_fallback::quality(wired_html, rows, excluded);
                        let b = super::table_fallback::quality(wireless_html, rows, excluded);
                        let loss = b.retained.saturating_sub(a.retained);
                        a.valid
                            && logical_grid(wired) == logical_grid(wireless)
                            && (loss as f64) <= wired.verified_splits as f64 / 3.
                            && (loss as f64) <= 0.08 * a.total as f64
                    },
                );
                if prefer_wired {
                    "wired".into()
                } else {
                    let selection = candidate_html(op(
                        "select_legacy_table",
                        json!({"texts":rows[1],"wired_html":candidates["wired"].0,"wireless_html":candidates["wireless"].0}),
                    ))?;
                    if selection.rejected.is_some() {
                        return Ok(selection);
                    }
                    selection.html["route"]
                        .as_str()
                        .ok_or_else(|| error("Invalid table selection"))?
                        .to_owned()
                }
            } else {
                label.into()
            };
            candidates.remove(selected.as_str()).unwrap()
        } else {
            self.structure(&png, &rows, role, variant, excluded, resolve, timings)?
        };
        let mut candidate = candidate_html(op("normalize_legacy_table_html", html))?;
        if candidate.rejected.is_none() {
            candidate.grid = grid;
        }
        Ok(candidate)
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
            let mut hints = op(
                "plan_table_hints",
                json!({"page":context,"table":table,"useful_list":task["useful_list"],"formulas":formulas,"checkboxes":checkboxes,"scale":page.image.scale,"formula_enabled":r.formula_enabled && r.table_formula_enabled,"image_enabled":true,"extract_original_image":false}),
            )?;
            timings.add("table_hints", started, 1);
            context = hints["page"].clone();
            let source = image(&page.images, &task["source"])?.clone();
            let hint_table = hints["table"].clone();
            let started = std::time::Instant::now();
            let (png, rows, table_rotation) = self.table_text(
                &source,
                page,
                &hint_table,
                &task["useful_list"],
                &mut hints["formulas"],
                index,
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
            let vectors = if r.fallback_table_variant.is_some()
                && r.native_text
                && !matches!(
                    page.facts["rotate_label"].as_str(),
                    Some("90" | "180" | "270")
                ) {
                if geometry.is_none() {
                    geometry = Some(native_adapter::prepare_table_page(
                        library, data, page.index, None,
                    )?);
                }
                super::table_evidence::crop_vectors(
                    &geometry.as_ref().unwrap()["drawing_lines"],
                    &task["useful_list"],
                    page.image.scale,
                    &table_rotation,
                )?
            } else {
                Vec::new()
            };
            let (html, text_fallback, quality) =
                self.table_model(&png, rows, &hints, &vectors, r, resolve, timings, true)?;
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
