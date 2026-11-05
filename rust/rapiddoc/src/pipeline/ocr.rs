//! Original formula -> native OCR plan -> detection -> cross-page recognition order.
use super::*;

/// Python groups detector regions over the entire page window, then collects
/// recognition crops page by page. Equal-aspect crops must enter NumPy's sort
/// in that same order: changing their padded batch neighbours changes logits.
fn order_body_groups(plans: &mut [(&mut Value, usize)]) -> Result<(), Error> {
    let mut groups = Vec::<Value>::new();
    for (plan, _) in plans.iter() {
        let mut jobs = array(&plan["detect"])?.iter().collect::<Vec<_>>();
        jobs.sort_by_key(|j| j["layout_index"].as_u64());
        for job in jobs {
            let shape = &job["target_shape"];
            if !groups.contains(shape) {
                groups.push(shape.clone());
            }
        }
    }
    for (plan, prefix) in plans {
        let mut ranks = Vec::new();
        for job in array(&plan["detect"])? {
            let index = job["layout_index"]
                .as_u64()
                .ok_or_else(|| error("Invalid detector layout index"))?
                as usize;
            let order = plan["model"]["layout_dets"][index]["original_order"].clone();
            let rank = groups
                .iter()
                .position(|g| g == &job["target_shape"])
                .unwrap();
            ranks.push((order, rank));
        }
        let dets = plan["model"]["layout_dets"]
            .as_array_mut()
            .ok_or_else(|| error("Invalid OCR model"))?;
        dets[*prefix..].sort_by_key(|span| {
            ranks
                .iter()
                .find(|(order, _)| order == &span["original_order"])
                .map(|(_, rank)| *rank)
        });
        let recognition = dets
            .iter()
            .enumerate()
            .filter_map(|(index, span)| {
                span.get("np_img")
                    .map(|source| json!({"source":source,"model_index":index}))
            })
            .collect::<Vec<_>>();
        plan["recognition"] = json!(recognition);
    }
    Ok(())
}

#[cfg(test)]
mod grouping_tests {
    use super::*;
    #[test]
    fn global_resolution_order_includes_jobs_without_recognition() {
        let mut first = json!({"detect":[{"layout_index":0,"target_shape":[64,128]}],
            "model":{"layout_dets":[{"original_order":0}]},"recognition":[]});
        let mut second = json!({"detect":[{"layout_index":0,"target_shape":[128,64]},{"layout_index":1,"target_shape":[64,128]}],
            "model":{"layout_dets":[{"original_order":0},{"original_order":1},
                {"original_order":0,"np_img":"a"},{"original_order":0,"np_img":"b"},
                {"original_order":1,"np_img":"c"}]}});
        order_body_groups(&mut [(&mut first, 1), (&mut second, 2)]).unwrap();
        assert_eq!(first["recognition"], json!([]));
        assert_eq!(
            second["recognition"],
            json!([{"source":"c","model_index":2},{"source":"a","model_index":3},{"source":"b","model_index":4}])
        );
        assert_eq!(
            second["model"]["layout_dets"][0],
            json!({"original_order":0})
        );
    }
}
impl Engine {
    pub(super) fn formulas(
        &mut self,
        pages: &mut [Page],
        batch_size: usize,
        resolve: &mut Resolver<'_>,
        timings: &mut Timings,
        progress: &mut dyn FnMut(usize, usize, f64) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let prepared = std::time::Instant::now();
        let plans = json!(pages.iter().map(|p| &p.plan).collect::<Vec<_>>());
        let jobs = op("plan_model_formulas", json!({"plans":plans}))?;
        let mut images = vec![];
        for job in array(&jobs["jobs"])? {
            let key = job["source"]
                .as_str()
                .ok_or_else(|| error("Invalid formula source"))?;
            images.push(
                pages
                    .iter()
                    .find_map(|page| {
                        page.formula_pixels
                            .get(key)
                            .map(rapiddoc_inference::formula::FormulaImage::Rgb)
                            .or_else(|| {
                                page.images.get(key).map(|png| {
                                    rapiddoc_inference::formula::FormulaImage::Png(png.as_slice())
                                })
                            })
                    })
                    .ok_or_else(|| error("Invalid formula source"))?,
            );
        }
        let mut phases = rapiddoc_inference::formula::FormulaTimings::default();
        let preparation = prepared.elapsed();
        let results = if images.is_empty() {
            Ok(vec![])
        } else {
            self.models.with_lazy("formula_model", resolve, |model| {
                let started = std::time::Instant::now();
                model.recognize_formula_images_with_progress(
                    &images,
                    batch_size,
                    &mut phases,
                    &mut |done, total| {
                        progress(done, total, (preparation + started.elapsed()).as_secs_f64())
                    },
                )
            })
        };
        for (stage, duration) in [
            ("formula_validation", phases.validation),
            ("formula_preprocessing", phases.preprocessing),
            ("formula_inference", phases.inference),
            ("formula_decoding", phases.decoding),
        ] {
            let entry = timings.0.entry(stage).or_default();
            entry.0 += duration;
            entry.1 += images.len();
        }
        drop(images);
        let results = results?;
        let mut output = op(
            "apply_model_formulas",
            json!({"plans":plans,"results":results}),
        )?;
        let plans = output
            .as_array_mut()
            .ok_or_else(|| error("Invalid formula plan output"))?;
        if plans.len() != pages.len() {
            return Err(error("Invalid formula plan count"));
        }
        for (page, plan) in pages.iter_mut().zip(plans.drain(..)) {
            page.plan = plan;
        }
        Ok(())
    }
    pub(super) fn prepare_ocr(
        &mut self,
        page: &mut Page,
        r: &Request,
        resolve: &mut Resolver<'_>,
        timings: &mut Timings,
        memory: &mut RecognitionMemory,
    ) -> Result<(), Error> {
        let started = std::time::Instant::now();
        let native = op(
            "plan_native_ocr",
            json!({"plan":{"model":page.plan["model"],"ocr":page.plan["ocr"]},"native_page":page.facts,"scale":page.image.scale,"ocr_enable":!r.native_text,"use_det_mode":if r.native_text{r.use_det_mode.as_str()}else{"ocr"}}),
        )?;
        let sources = array(&native["detect"])?
            .iter()
            .map(|task| Ok(image(&page.images, &task["source"])?.as_slice()))
            .collect::<Result<Vec<_>, Error>>()?;
        let prepared = crate::legacy_model::native_ocr::detection::prepare_pixels(
            &json!({"plan":native,"page_index":page.index}),
            &sources,
        )?;
        timings.add("ocr_detector_input", started, prepared.pixels.len());
        page.plan
            .as_object_mut()
            .ok_or_else(|| error("Invalid crop plan"))?
            .extend(prepared.plan.as_object().unwrap().clone());
        let mut pixels: BTreeMap<_, _> = prepared.pixels.into_iter().collect();
        let mut detections = vec![];
        let started = std::time::Instant::now();
        for task in array(&page.plan["detect"])? {
            let key = task["source"]
                .as_str()
                .ok_or_else(|| error("Invalid detection source"))?;
            let rgb = pixels
                .remove(key)
                .ok_or_else(|| error("Missing detector RGB pixels"))?;
            if let Some(folder) = std::env::var_os("RAPIDDOC_PARITY_TRACE") {
                let folder = std::path::PathBuf::from(folder);
                std::fs::create_dir_all(&folder).map_err(error)?;
                std::fs::write(folder.join(format!("{key}.rgb")), &rgb.data).map_err(error)?;
                super::parity_trace(
                    &format!("{key}-input.json"),
                    &json!({"width":rgb.w,"height":rgb.h,"task":task}),
                )?;
            }
            let values = self
                .models
                .with_lazy("detector", resolve, |model| model.detect_model_rgb(rgb))?;
            detections.push(wire(&values.iter().map(|d| d.points).collect::<Vec<_>>())?);
            super::parity_trace(&format!("{key}-boxes.json"), detections.last().unwrap())?;
        }
        timings.add("ocr_detector_execution", started, detections.len());
        let started = std::time::Instant::now();
        let request = json!({"plan":{"model":page.plan["model"],"detect":page.plan["detect"]},"detections":detections,"lang":r.lang,"page_index":page.index});
        page.ocr_model_prefix = array(&page.plan["model"]["layout_dets"])?.len();
        let mut recognized = op("plan_model_recognition", request)?;
        let mut decoded: Option<(String, rapiddoc_inference::crop::PipelineImage)> = None;
        for attempt in array(&recognized["crop_attempts"])? {
            let key = attempt["source"]
                .as_str()
                .ok_or_else(|| error("Invalid recognition source"))?;
            if decoded.as_ref().is_none_or(|(previous, _)| previous != key) {
                decoded = Some((
                    key.to_owned(),
                    rapiddoc_inference::crop::PipelineImage::decode(
                        image(&page.images, &attempt["source"])?,
                        16_000_000,
                    )?,
                ));
            }
            let points = serde_json::from_value(attempt["points"].clone()).map_err(error)?;
            let crop = decoded.as_ref().unwrap().1.crop_pixels(points, 4_000_000)?;
            if let Some(id) = attempt["output"].as_str() {
                memory.images += 1;
                memory.pixels += crop.width * crop.height;
                if memory.images > rapiddoc_core::resources::limit(1000)
                    || memory.pixels > rapiddoc_core::resources::limit(32_000_000)
                {
                    return Err(Error {
                        code: ErrorCode::ResourceLimit,
                        message: "Recognition window exceeds image/pixel budget".into(),
                    });
                }
                page.recognition_pixels.insert(
                    id.into(),
                    rapiddoc_inference::recognition_batch::Image {
                        bgr: crop.bgr,
                        width: crop.width,
                        height: crop.height,
                    },
                );
            }
        }
        recognized.as_object_mut().unwrap().remove("crop_attempts");
        timings.add(
            "ocr_recognition_crop_preparation",
            started,
            array(&recognized["recognition"])?.len(),
        );
        page.plan
            .as_object_mut()
            .unwrap()
            .extend(recognized.as_object().unwrap().clone());
        Ok(())
    }
    pub(super) fn recognize_body(
        &mut self,
        pages: &mut [Page],
        resolve: &mut Resolver<'_>,
        progress: &mut dyn FnMut(Progress<'_>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        order_body_groups(
            &mut pages
                .iter_mut()
                .map(|p| (&mut p.plan, p.ocr_model_prefix))
                .collect::<Vec<_>>(),
        )?;
        let mut images = vec![];
        for page in pages.iter_mut() {
            for task in array(&page.plan["recognition"])? {
                let key = task["source"]
                    .as_str()
                    .ok_or_else(|| error("Invalid recognition source"))?;
                images.push(
                    page.recognition_pixels
                        .remove(key)
                        .ok_or_else(|| error("Missing recognition pixels"))?,
                );
            }
        }
        if let Some(folder) = std::env::var_os("RAPIDDOC_PARITY_TRACE") {
            use std::io::Write;
            let folder = std::path::PathBuf::from(folder);
            std::fs::create_dir_all(&folder).map_err(error)?;
            let first = pages.first().map(|p| p.index).unwrap_or(0);
            let mut bytes = std::io::BufWriter::new(
                std::fs::File::create(folder.join(format!("body-{first:03}.bgr")))
                    .map_err(error)?,
            );
            let mut sizes = vec![];
            let mut offset = 0;
            for image in &images {
                bytes.write_all(&image.bgr).map_err(error)?;
                sizes.push(json!({"width":image.width,"height":image.height,"offset":offset,"length":image.bgr.len()}));
                offset += image.bgr.len();
            }
            bytes.flush().map_err(error)?;
            super::parity_trace(&format!("body-{first:03}.json"), &json!(sizes))?;
        }
        let results = if images.is_empty() {
            vec![]
        } else {
            let started = std::time::Instant::now();
            self.models.recognize_images_with_progress(
                "recognizer",
                resolve,
                &images,
                6,
                &mut |done, total| {
                    progress(Progress {
                        completed: Some(done),
                        elapsed_seconds: Some(started.elapsed().as_secs_f64()),
                        total: Some(total),
                        stage: "ocr_recognition",
                        page_index: None,
                        detail: Some(format!("crops {done}/{total}; batch_size=6")),
                    })
                },
            )?
        };
        drop(images);
        let mut offset = 0;
        for page in pages {
            let count = array(&page.plan["recognition"])?.len();
            let rows = results[offset..offset + count]
                .iter()
                .map(|r| (&r.text, r.confidence))
                .collect::<Vec<_>>();
            offset += count;
            page.plan["model"] = op(
                "apply_model_recognition",
                json!({"model":page.plan["model"],"results":rows}),
            )?;
        }
        Ok(())
    }
    pub(super) fn finalize(
        &mut self,
        pages: Value,
        images: Images,
        resolve: &mut Resolver<'_>,
    ) -> Result<Value, Error> {
        let targets = crate::legacy_post_ocr::plan(&pages)?;
        let pngs = targets
            .iter()
            .map(|t| Ok((**image(&images, &t.image)?).clone()))
            .collect::<Result<Vec<_>, Error>>()?;
        let results = if pngs.is_empty() {
            vec![]
        } else {
            self.models.with_lazy("recognizer", resolve, |model| {
                model.recognize_batch_png(&pngs, 6)
            })?
        };
        let rows = results
            .iter()
            .map(|r| (&r.text, r.confidence))
            .collect::<Vec<_>>();
        let pages = crate::legacy_post_ocr::apply(&json!({"pages":pages,"results":rows}))?;
        let pages = crate::legacy_page::optimize_formula_numbers(&pages)?;
        crate::legacy_paragraphs::split(&pages)
    }
}
