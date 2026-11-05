//! Shared seal scheduling. Python supplies model descriptors only.
use super::*;
impl Engine {
    pub(super) fn seals(
        &mut self,
        page: &mut Page,
        r: &Request,
        resolve: &mut Resolver<'_>,
        progress: &mut dyn FnMut(Progress<'_>) -> Result<(), Error>,
        timings: &mut Timings,
    ) -> Result<(), Error> {
        let elapsed = std::time::Instant::now();
        let loading = self.models.initialization_time();
        let indices = array(&page.plan["model"]["layout_dets"])?
            .iter()
            .enumerate()
            .filter_map(|(i, d)| (d["original_label"] == "seal").then_some(i))
            .collect::<Vec<_>>();
        if indices.is_empty() {
            return Ok(());
        }
        let path = r
            .geos_library
            .as_deref()
            .ok_or_else(|| error("Seal recognition requires the GEOS runtime"))?;
        let cropper = rapiddoc_inference::seal::Cropper::load(path)?;
        let pixels = crate::legacy_page::assets::Pixels::decode(&page.image.png)?;
        if pixels.channels != 3 {
            return Err(error("Seal page must be RGB"));
        }
        for (done, index) in indices.iter().copied().enumerate() {
            progress(Progress {
                completed: Some(done),
                elapsed_seconds: Some(
                    elapsed
                        .elapsed()
                        .saturating_sub(self.models.initialization_time() - loading)
                        .as_secs_f64(),
                ),
                total: Some(indices.len()),
                stage: "seal",
                page_index: Some(page.index),
                detail: Some(format!("regions {}/{}", done + 1, indices.len())),
            })?;
            let det = &page.plan["model"]["layout_dets"][index];
            let poly = array(&det["poly"])?;
            let mut bounds = [0i64; 4];
            for (to, i) in bounds.iter_mut().zip([0, 1, 4, 5]) {
                *to = poly
                    .get(i)
                    .and_then(Value::as_f64)
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| error("Invalid seal layout bounds"))?
                    .trunc() as i64;
            }
            let polygon = if let Some(points) = det.get("polygon_points").filter(|p| !p.is_null()) {
                serde_json::from_value::<Vec<[f64; 2]>>(points.clone())
                    .map_err(|e| error(&e.to_string()))?
            } else {
                vec![]
            };
            let started = std::time::Instant::now();
            let region = rapiddoc_inference::seal::layout_crop(
                &pixels.data,
                pixels.width,
                pixels.height,
                bounds,
                &polygon,
            )?;
            timings.add("seal_region_crop", started, 1);
            let started = std::time::Instant::now();
            let boxes = self.models.with_lazy("seal_detector", resolve, |model| {
                model.detect_seal_bgr(&region.bgr, region.width, region.height)
            })?;
            timings.add("seal_detection", started, 1);
            if boxes.is_empty() {
                continue;
            }
            let started = std::time::Instant::now();
            let images = boxes
                .iter()
                .map(|b| {
                    cropper
                        .crop(&region.bgr, region.width, region.height, &b.points)
                        .map(|c| rapiddoc_inference::recognition_batch::Image {
                            bgr: c.bgr,
                            width: c.width,
                            height: c.height,
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            timings.add("seal_rectification", started, images.len());
            let started = std::time::Instant::now();
            let recognized = self.models.with_lazy("seal_recognizer", resolve, |model| {
                model.recognize_images(&images, 6)
            })?;
            timings.add("seal_recognition", started, recognized.len());
            // Original RapidOcrModel drops scores <0.5; no surviving entries means
            // _run_seal_ocr leaves any pre-existing layout text untouched.
            let accepted = recognized
                .into_iter()
                .filter(|r| r.confidence >= 0.5)
                .collect::<Vec<_>>();
            if !accepted.is_empty() {
                page.plan["model"]["layout_dets"][index]["text"] = json!(accepted
                    .into_iter()
                    .filter_map(|r| (!r.text.is_empty()).then_some(r.text))
                    .collect::<Vec<_>>());
            }
        }
        Ok(())
    }
}
