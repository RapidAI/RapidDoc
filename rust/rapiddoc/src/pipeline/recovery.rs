use serde_json::{json, Value};
use unicode_normalization::UnicodeNormalization;

type Bounds = [f64; 4];
type Region<'a> = (Bounds, i64, Option<f64>, Option<&'a str>, Option<&'a str>);

fn intersection(line: Bounds, region: Bounds) -> f64 {
    (line[2].min(region[2]) - line[0].max(region[0])).max(0.)
        * (line[3].min(region[3]) - line[1].max(region[1])).max(0.)
}

fn area([l, t, r, b]: Bounds) -> f64 {
    (r - l) * (b - t)
}

fn covered(line: Bounds, region: Bounds) -> bool {
    intersection(line, region) > area(line) * 0.5
}

fn conflicts_with_recognized(line: Bounds, region: Region<'_>) -> bool {
    let (bbox, category, _, content, _) = region;
    if !content.is_some_and(|text| !text.trim().is_empty()) {
        return false;
    }
    let reference_area = match category {
        14 => area(bbox),
        15 => area(bbox).min(area(line)),
        _ => return false,
    };
    reference_area > 0. && intersection(line, bbox) > 0.65 * reference_area
}

fn bounds(value: &Value) -> Option<Bounds> {
    Some([
        value[0].as_f64()?,
        value[1].as_f64()?,
        value[2].as_f64()?,
        value[3].as_f64()?,
    ])
}

fn key(text: &str) -> String {
    text.nfkc()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn table_rows(html: &str) -> Vec<String> {
    let rows = crate::cached_regex!(r"(?is)<tr\b[^>]*>(.*?)</tr\s*>");
    let tags = crate::cached_regex!(r"(?s)<[^>]*>");
    rows.captures_iter(html)
        .map(|r| {
            key(&crate::processing::inline_output::unescape_html(
                &tags.replace_all(&r[1], ""),
            ))
        })
        .collect()
}

fn native_line(line: &Value, scale: f64) -> Option<(Bounds, String)> {
    let chars = line["spans"]
        .as_array()?
        .iter()
        .filter_map(|span| span["chars"].as_array())
        .flatten();
    if line["rotation"].as_f64().unwrap_or(0.) != 0. {
        let mut visible = chars
            .clone()
            .filter(|c| c["char"].as_str().is_some_and(|s| !s.trim().is_empty()));
        let a = bounds(&visible.next()?["bbox"])?;
        let b = bounds(&visible.next_back()?["bbox"])?;
        if (b[0] + b[2] - a[0] - a[2]).abs() <= 2. * (b[1] + b[3] - a[1] - a[3]).abs() {
            return None;
        }
    }
    let text: String = chars.filter_map(|c| c["char"].as_str()).collect();
    let b = bounds(&line["bbox"])?.map(|v| v * scale);
    (!text.trim().is_empty() && b[0] < b[2] && b[1] < b[3]).then_some((b, text))
}

fn poly([l, t, r, b]: Bounds) -> [f64; 8] {
    [l, t, r, t, r, b, l, b]
}

fn regions(dets: &[Value]) -> Vec<Region<'_>> {
    dets.iter()
        .filter_map(|det| {
            let p = det["poly"].as_array()?;
            let mut points = p
                .chunks_exact(2)
                .filter_map(|pair| Some((pair[0].as_f64()?, pair[1].as_f64()?)));
            let (x, y) = points.next()?;
            let b = points.fold([x, y, x, y], |[l, t, r, b], (x, y)| {
                [l.min(x), t.min(y), r.max(x), b.max(y)]
            });
            let category = det["category_id"].as_i64()?;
            Some((
                b,
                category,
                det["original_order"].as_f64(),
                det["html"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .or(det["latex"].as_str())
                    .or_else(|| (category == 15).then(|| det["text"].as_str()).flatten()),
                det["original_label"].as_str(),
            ))
        })
        .collect()
}

fn figure_regions(dets: &[Value], width: f64, height: f64) -> Vec<(usize, Bounds)> {
    let mut figures: Vec<_> = dets
        .iter()
        .enumerate()
        .filter_map(|(i, det)| {
            if det["category_id"] != 3 || det["original_label"] == "seal" {
                return None;
            }
            let r = regions(std::slice::from_ref(det)).first()?.0;
            (r[2] - r[0] < 0.9 * width || r[3] - r[1] < 0.9 * height).then_some((i, r))
        })
        .collect();
    let area = |r: Bounds| (r[2] - r[0]) * (r[3] - r[1]);
    figures.sort_by(|a, b| area(a.1).total_cmp(&area(b.1)));
    figures
}

fn format_figure_text(mut fragments: Vec<(Bounds, String)>) -> String {
    fragments.sort_by(|a, b| a.0[1].total_cmp(&b.0[1]).then(a.0[0].total_cmp(&b.0[0])));
    let mut rows: Vec<Vec<(Bounds, String)>> = Vec::new();
    for (b, text) in fragments {
        if let Some(row) = rows.last_mut() {
            let a = row[0].0;
            let overlap = b[3].min(a[3]) - b[1].max(a[1]);
            if overlap > 0.5 * (b[3] - b[1]).max(a[3] - a[1]) {
                row.push((b, text));
                continue;
            }
        }
        rows.push(vec![(b, text)]);
    }
    rows.iter_mut()
        .map(|row| {
            row.sort_by(|a, b| a.0[0].total_cmp(&b.0[0]));
            row.iter()
                .map(|(_, text)| text.trim())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn native_text(model: &mut Value, page: &Value, scale: f64, width: f64, height: f64) {
    let (Some(dets), Some(blocks)) = (
        model["layout_dets"].as_array_mut(),
        page["blocks"].as_array(),
    ) else {
        return;
    };
    let regions = regions(dets);
    let mut table_cache = std::collections::HashMap::new();
    let mut missing = Vec::new();
    for line in blocks
        .iter()
        .filter_map(|b| b["lines"].as_array())
        .flatten()
    {
        let Some((b, text)) = native_line(line, scale) else {
            continue;
        };
        let mut boundary = false;
        let owned = regions.iter().any(|&(r, cat, _, html, label)| {
            if !covered(b, r) {
                return false;
            }
            if cat == 3 {
                return label == Some("seal")
                    || r[2] - r[0] < 0.9 * width
                    || r[3] - r[1] < 0.9 * height;
            }
            if cat == 5 && ((b[1] < r[1] && r[1] < b[3]) || (b[1] < r[3] && r[3] < b[3])) {
                if let Some(html) = html {
                    let rows = table_cache.entry(html).or_insert_with(|| table_rows(html));
                    if !rows.is_empty() && !rows.iter().any(|row| row.contains(&key(&text))) {
                        boundary = true;
                        return false;
                    }
                }
            }
            true
        });
        if !owned {
            missing.push((b, text, boundary));
        }
    }
    let additions = text_detections(missing, &regions, true);
    dets.extend(additions);
}

fn text_detections(
    mut missing: Vec<(Bounds, String, bool)>,
    regions: &[Region<'_>],
    native: bool,
) -> Vec<Value> {
    missing.sort_by(|a, b| a.0[1].total_cmp(&b.0[1]).then(a.0[0].total_cmp(&b.0[0])));
    let mut groups: Vec<Vec<(Bounds, String, bool)>> = Vec::new();
    for line in missing {
        let [left, top, right, bottom] = line.0;
        let neighbor = groups
            .iter()
            .enumerate()
            .filter_map(|(i, g)| {
                let [l, t, r, b] = g.last()?.0;
                let h = (bottom - top).max(b - t);
                let gap = top - b;
                (gap >= -0.25 * h
                    && gap <= 1.5 * h
                    && (left - l).abs() <= 2. * h
                    && right.min(r) > left.max(l))
                .then_some((i, gap.abs(), (left - l).abs()))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)));
        if let Some((i, _, _)) = neighbor {
            groups[i].push(line);
        } else {
            groups.push(vec![line]);
        }
    }
    let mut additions = Vec::new();
    for group in groups {
        let b = group
            .iter()
            .fold(group[0].0, |[l, t, r, b], ([x, y, w, h], _, _)| {
                [l.min(*x), t.min(*y), r.max(*w), b.max(*h)]
            });
        if group.len() == 1 && !group[0].2 {
            let h = b[3] - b[1];
            let (mut above, mut below, mut text_neighbor) = (false, false, false);
            for &(r, cat, _, _, _) in regions {
                if matches!(cat, 0 | 1 | 3 | 5) && b[2].min(r[2]) - b[0].max(r[0]) > (b[2] - b[0]) * 0.5 {
                    let a = (-h / 2. ..=h).contains(&(b[1] - r[3]));
                    let z = (-h / 2. ..=h).contains(&(r[1] - b[3]));
                    above |= a;
                    below |= z;
                    text_neighbor |= matches!(cat, 0 | 1) && (a || z);
                }
            }
            if !(above && below && text_neighbor) {
                let (_, text, _) = &group[0];
                additions.push(json!({"category_id":15,"poly":poly(b),"score":1.,
                    "text":text,"unassigned_text":true,"original_label":"text"}));
                continue;
            }
        }
        let (x, y) = ((b[0] + b[2]) / 2., (b[1] + b[3]) / 2.);
        let local = local_column_order(b, regions);
        let order = local.or_else(|| {
            regions
                .iter()
                .filter_map(|&(r, cat, order, _, _)| {
                    let order = order.filter(|o| *o >= 0.)?;
                    if matches!(cat, 2 | 15) {
                        return None;
                    }
                    Some((
                        (r[0] - x).max(x - r[2]).max(0.),
                        y - (r[1] + r[3]) / 2.,
                        order,
                        r[3] - r[1],
                    ))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.abs().total_cmp(&b.1.abs())))
                .map(|(_, dy, order, h)| (order + dy / (2. * (dy.abs() + h.max(1.)))).max(0.))
        });
        additions.push(json!({"category_id":1,"poly":poly(b),"score":1.,"original_label":"text","original_order":order}));
        additions.extend(group.into_iter().map(|(b,text,_)| json!({
            "category_id":15,"poly":poly(b),"score":1.,"original_label":"text",
            "original_order":order,"text":text,"docvortex_native_text":native,"layout_recovered_text":true
        })));
    }
    additions
}

fn local_column_order(b: Bounds, regions: &[Region<'_>]) -> Option<f64> {
    let neighbors: Vec<_> = regions
        .iter()
        .filter_map(|&(r, cat, order, _, _)| {
            (cat == 1 && b[3].min(r[3]) > b[1].max(r[1]))
                .then_some((r, order.filter(|v| *v >= 0.)?))
        })
        .collect();
    if neighbors.is_empty() || neighbors.iter().any(|(r, _)| r[2] >= b[0] && r[0] <= b[2]) {
        return None;
    }
    let mut boxes: Vec<_> = neighbors.iter().map(|&(r, _)| r).collect();
    let slot = boxes.len();
    boxes.push(b);
    let indices = crate::processing::native_visual_annotations::recovery_order(&boxes)?;
    if indices.first() == Some(&slot) {
        let order = neighbors.iter().map(|&(_, o)| o).reduce(f64::min)?;
        (order > 0.).then_some((order - 0.5).max(0.))
    } else if indices.last() == Some(&slot) {
        neighbors
            .iter()
            .map(|&(_, o)| o)
            .reduce(f64::max)
            .map(|o| o + 0.5)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_caption_between_title_and_picture_is_recovered() {
        let regions = vec![([0., 5., 100., 18.], 0, Some(1.), None, None),
                           ([0., 32., 100., 100.], 3, Some(2.), None, None)];
        let output = text_detections(vec![([0., 20., 100., 30.], "caption".into(), false)], &regions, false);
        assert!(output.iter().any(|d| d["category_id"] == 1));
        assert!(!output.iter().any(|d| d["unassigned_text"] == true));
    }

    #[test]
    fn recovery_preserves_recognized_formulas_and_lines() {
        let ocr = [0., 0., 100., 10.];
        let formula = ([50., 0., 90., 10.], 14, None, Some("w^2"), None);
        assert!(!covered(ocr, formula.0));
        assert!(conflicts_with_recognized(ocr, formula));
        let recognized = ([20., 0., 60., 10.], 15, None, Some("existing text"), None);
        assert!(conflicts_with_recognized(ocr, recognized));
        assert!(!conflicts_with_recognized([0., 11., 100., 21.], recognized));
    }

    #[test]
    fn native_table_edge_only_recovers_unrepresented_text() {
        let page = json!({"blocks":[{"lines":[
            {"bbox":[5,94,45,104],"spans":[{"chars":[{"char":"𝑧 Axial"}]}]},
            {"bbox":[5,75,45,85],"spans":[{"chars":[{"char":"internal"}]}]}
        ]}]});
        for (cell, expected) in [("Mean", 1), ("z</td><td>Axial", 0)] {
            let mut model = json!({"layout_dets":[
                {"category_id":5,"poly":poly([0.,0.,100.,100.]),"original_order":0,
                    "html":format!("<table><tr><td>{cell}</td></tr></table>")}
            ]});
            native_text(&mut model, &page, 1., 200., 200.);
            let dets = model["layout_dets"].as_array().unwrap();
            assert_eq!(
                dets.iter().filter(|d| d["text"] == "𝑧 Axial").count(),
                expected
            );
            assert!(!dets.iter().any(|d| d["text"] == "internal"));
        }
    }

    #[test]
    fn italic_shear_is_not_vertical_writing() {
        let line = |last: Bounds| {
            json!({"bbox":[0,0,20,20],"rotation":0.24,"spans":[{"chars":[
                {"char":"A","bbox":[0,0,5,5]},{"char":"B","bbox":last}
            ]}]})
        };
        assert_eq!(native_line(&line([7., 0., 12., 5.]), 1.).unwrap().1, "AB");
        assert!(native_line(&line([0., 7., 5., 12.]), 1.).is_none());
    }
}

impl super::Engine {
    pub(super) fn recover_ocr_text(
        &mut self,
        page: &mut super::Page,
        resolve: &mut super::Resolver<'_>,
        body_ocr: bool,
        image_ocr: bool,
        native: bool,
    ) -> Result<(), rapiddoc_core::Error> {
        use rapiddoc_inference::{crop::PipelineImage, recognition_batch::Image};
        let dets = page.plan["model"]["layout_dets"].as_array_mut().unwrap();
        let regions = regions(dets);
        let width = page.image.width as f64;
        let height = page.image.height as f64;
        let figures = if image_ocr {
            figure_regions(dets, width, height)
        } else {
            vec![]
        };
        if !body_ocr && figures.is_empty() {
            return Ok(());
        }
        let mut figure_text = std::collections::BTreeMap::<usize, Vec<(Bounds, String)>>::new();
        if native {
            for line in page.facts["blocks"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|block| block["lines"].as_array())
                .flatten()
            {
                if let Some((b, text)) = native_line(line, page.image.scale) {
                    if let Some(&(i, _)) = figures.iter().find(|(_, r)| covered(b, *r)) {
                        if !regions
                            .iter()
                            .any(|&(r, cat, _, _, _)| cat != 3 && covered(b, r))
                        {
                            figure_text.entry(i).or_default().push((b, text));
                        }
                    }
                }
            }
        }
        let figures: Vec<_> = figures
            .into_iter()
            .filter(|(i, _)| !figure_text.contains_key(i))
            .collect();
        let boxes = if body_ocr || !figures.is_empty() {
            self.models.with_lazy("detector", resolve, |model| {
                model.detect_model_png(&page.image.png)
            })?
        } else {
            vec![]
        };
        let mut pending = Vec::new();
        for item in boxes {
            let [x, y] = item.points[0];
            let b = item
                .points
                .iter()
                .fold([x, y, x, y], |[l, t, r, b], &[x, y]| {
                    [l.min(x), t.min(y), r.max(x), b.max(y)]
                })
                .map(f64::from);
            let owned = regions.iter().any(|&(r, cat, _, html, label)| {
                covered(b, r)
                    && match cat {
                        0 | 1 | 16 => false,
                        3 => label == Some("seal"),
                        5 => html.is_some_and(|s| !s.is_empty()),
                        _ => true,
                    }
            });
            if b[0] < b[2] && b[1] < b[3] && !owned
                && !regions.iter().any(|&r| conflicts_with_recognized(b, r))
            {
                let inside_text = regions
                    .iter()
                    .any(|&(r, cat, _, _, _)| matches!(cat, 0 | 1) && covered(b, r));
                let figure = figures
                    .iter()
                    .find(|(_, r)| covered(b, *r))
                    .map(|(i, _)| *i);
                let in_picture = regions.iter().any(|&(r, cat, _, _, _)| {
                    cat == 3
                        && (r[2] - r[0] < 0.9 * width || r[3] - r[1] < 0.9 * height)
                        && covered(b, r)
                });
                if figure.is_some() || body_ocr && (!in_picture || inside_text) {
                    pending.push((b, item.points, inside_text, figure));
                }
            }
        }
        let mut missing = Vec::new();
        if !pending.is_empty() {
            let image = PipelineImage::decode(&page.image.png, 16_000_000)?;
            for batch in pending.chunks(6) {
                let crops = batch
                    .iter()
                    .map(|(_, points, _, _)| {
                        image.crop_pixels(*points, 4_000_000).map(|crop| Image {
                            bgr: crop.bgr,
                            width: crop.width,
                            height: crop.height,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let (items, crops): (Vec<_>, Vec<_>) = batch.iter().zip(crops)
                    .filter(|((_, _, inside, figure), crop)| *inside || figure.is_some()
                        || rapiddoc_inference::detection::has_foreground_ink(&crop.bgr))
                    .unzip();
                if crops.is_empty() { continue; }
                let recognized = self.models.with_lazy("recognizer", resolve, |model| {
                    model.recognize_images(&crops, 6)
                })?;
                for ((b, _, inside, figure), result) in items.into_iter().zip(recognized) {
                    if result.confidence < 0.5 || result.text.trim().is_empty() {
                        continue;
                    }
                    if let Some(i) = figure {
                        figure_text.entry(i).or_default().push((b, result.text));
                    } else {
                        missing.push((b, result.text, inside));
                    }
                }
            }
        }
        let additions = text_detections(missing, &regions, false);
        for (i, lines) in figure_text {
            dets[i]["image_text"] = json!(format_figure_text(lines));
        }
        dets.extend(additions);
        Ok(())
    }
}
