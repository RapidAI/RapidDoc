//! Match Python's table-owned formula crop orientation and local hint geometry.
use super::*;
use crate::legacy_model::table_direction;
use rapiddoc_core::pillow_resize::Rgb;

fn bounds(item: &Value) -> Result<[f64; 4], Error> {
    if item["category_id"] != 5 && item["bbox"].is_array() {
        serde_json::from_value(item["bbox"].clone()).map_err(error)
    } else {
        let p = &item["poly"];
        serde_json::from_value(json!([p[0], p[1], p[4], p[5]])).map_err(error)
    }
}

fn owner(box_: [f64; 4], tables: &[(usize, [f64; 4])]) -> Option<usize> {
    let [x0, y0, x1, y1] = box_;
    let area = (x1 - x0).max(0.) * (y1 - y0).max(0.);
    if area == 0. {
        return None;
    }
    tables
        .iter()
        .filter_map(|(index, [a, b, c, d])| {
            let intersection =
                (x1.min(*c) - x0.max(*a)).max(0.) * (y1.min(*d) - y0.max(*b)).max(0.);
            (*a <= (x0 + x1) / 2.
                && (x0 + x1) / 2. <= *c
                && *b <= (y0 + y1) / 2.
                && (y0 + y1) / 2. <= *d
                && intersection >= area * 0.5)
                .then_some((-intersection, (c - a) * (d - b), *index))
        })
        .min_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(a.1.total_cmp(&b.1))
                .then(a.2.cmp(&b.2))
        })
        .map(|v| v.2)
}

fn rotate_pixels(rgb: Rgb, label: &str) -> Rgb {
    if !matches!(label, "90" | "270") {
        return rgb;
    }
    let mut data = vec![0; rgb.data.len()];
    for y in 0..rgb.h {
        for x in 0..rgb.w {
            let (xx, yy) = if label == "90" {
                (y, rgb.w - 1 - x)
            } else {
                (rgb.h - 1 - y, x)
            };
            let src = (y * rgb.w + x) * 3;
            let dst = (yy * rgb.h + xx) * 3;
            data[dst..dst + 3].copy_from_slice(&rgb.data[src..src + 3]);
        }
    }
    Rgb {
        data,
        w: rgb.h,
        h: rgb.w,
    }
}

pub(super) fn rotate_hints(
    hints: &mut Value,
    width: usize,
    height: usize,
    label: &str,
) -> Result<(), Error> {
    if !matches!(label, "90" | "270") || hints.is_null() {
        return Ok(());
    }
    for hint in hints
        .as_array_mut()
        .ok_or_else(|| error("Invalid table hints"))?
    {
        let [x0, y0, x1, y1] = bounds(hint)?;
        hint["bbox"] = if label == "90" {
            json!([y0, width as f64 - x1, y1, width as f64 - x0])
        } else {
            json!([height as f64 - y1, x0, height as f64 - y0, x1])
        };
    }
    Ok(())
}

impl Engine {
    pub(super) fn prepare_table_formulas(
        &mut self,
        page: &mut Page,
        r: &Request,
        resolve: &mut Resolver<'_>,
        timings: &mut Timings,
    ) -> Result<(), Error> {
        let started = std::time::Instant::now();
        let tasks = array(&page.plan["tables"])?;
        let dets = &page.plan["model"]["layout_dets"];
        let tables = tasks
            .iter()
            .map(|t| {
                let index = t["layout_index"]
                    .as_u64()
                    .ok_or_else(|| error("Invalid table index"))?
                    as usize;
                Ok((index, bounds(&dets[index])?))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mut owned = BTreeMap::<usize, Vec<String>>::new();
        for formula in array(&page.plan["formulas"])? {
            let index = formula["layout_index"]
                .as_u64()
                .ok_or_else(|| error("Invalid formula index"))? as usize;
            if let Some(table) = owner(bounds(&dets[index])?, &tables) {
                owned.entry(table).or_default().push(
                    formula["source"]
                        .as_str()
                        .ok_or_else(|| error("Invalid formula source"))?
                        .to_owned(),
                );
            }
        }
        if owned.is_empty() {
            return Ok(());
        }
        let formulas = json!(array(dets)?
            .iter()
            .filter(|d| matches!(d["category_id"].as_u64(), Some(8 | 13 | 14)))
            .cloned()
            .collect::<Vec<_>>());
        let checkboxes = json!(array(dets)?
            .iter()
            .filter(|d| d["category_id"] == 200)
            .cloned()
            .collect::<Vec<_>>());
        let mut context = page.facts.clone();
        context["ori_image_list"] = json!([]);
        for (index, sources) in owned {
            let task = tasks
                .iter()
                .find(|t| t["layout_index"].as_u64() == Some(index as u64))
                .unwrap();
            let table = &dets[index];
            let hints = op(
                "plan_table_hints",
                json!({"page":context,"table":table,
                "useful_list":task["useful_list"],"formulas":formulas,"checkboxes":checkboxes,
                "scale":page.image.scale,"formula_enabled":r.formula_enabled,
                "image_enabled":false,"extract_original_image":false}),
            )?;
            let source = image(&page.images, &task["source"])?;
            let probe =
                self.probe_table_direction(source, page, table, &hints["formulas"], resolve)?;
            let label = &probe.1;
            if matches!(label.as_str(), "90" | "270") {
                for source in sources {
                    if let Some(pixels) = page.formula_pixels.remove(&source) {
                        page.formula_pixels
                            .insert(source, rotate_pixels(pixels, label));
                    } else if let Some(png) = page.images.get_mut(&source) {
                        *png = Arc::new(table_direction::oriented_png(png, label)?.into_owned());
                    }
                }
            }
            page.table_formula_probes.insert(index, probe);
        }
        timings.add(
            "table_formula_orientation",
            started,
            page.table_formula_probes.len(),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ownership_requires_center_and_coverage_and_prefers_smallest_table() {
        let tables = [(7, [0., 0., 10., 20.]), (8, [0., 0., 6., 10.])];
        assert_eq!(owner([1., 2., 4., 8.], &tables), Some(8));
        assert_eq!(owner([12., 2., 15., 8.], &tables), None);
        assert_eq!(owner([1., 2., 1., 8.], &tables), None);
    }
    #[test]
    fn quarter_turns_rotate_rgb_without_resampling() {
        for (label, expected) in [
            ("90", vec![1, 3, 5, 0, 2, 4]),
            ("270", vec![4, 2, 0, 5, 3, 1]),
            ("0", vec![0, 1, 2, 3, 4, 5]),
        ] {
            let rgb = Rgb {
                data: (0u8..6).flat_map(|v| [v; 3]).collect(),
                w: 2,
                h: 3,
            };
            let result = rotate_pixels(rgb, label);
            assert_eq!(
                result.data,
                expected
                    .into_iter()
                    .flat_map(|v| [v; 3])
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                (result.w, result.h),
                if label == "0" { (2, 3) } else { (3, 2) }
            );
        }
    }
    #[test]
    fn hint_rotation_preserves_latex_and_checkbox() {
        for (label, expected) in [
            ("90", json!([2., 4., 5., 7.])),
            ("270", json!([3., 1., 6., 4.])),
        ] {
            let mut hints =
                json!([{"bbox":[1,2,4,5],"latex":"\\pm 1"},{"bbox":[1,2,4,5],"checkbox":"☑"}]);
            rotate_hints(&mut hints, 8, 8, label).unwrap();
            assert_eq!(hints[0]["bbox"], expected);
            assert_eq!(hints[1]["bbox"], expected);
            assert_eq!(hints[0]["latex"], "\\pm 1");
            assert_eq!(hints[1]["checkbox"], "☑");
        }
    }
}
