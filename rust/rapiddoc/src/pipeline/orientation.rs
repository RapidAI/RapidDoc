//! Original BatchAnalyze whole-page orientation and final poly restoration.
use super::*;

pub(super) fn rotate(
    page: &rapiddoc_pdf::render::RenderedPage,
    label: &str,
) -> Result<rapiddoc_pdf::render::RenderedPage, Error> {
    let rotated = crate::processing::table_direction::image(&page.png, label, false)?;
    Ok(rapiddoc_pdf::render::RenderedPage {
        width: rotated.metadata["width"]
            .as_u64()
            .ok_or_else(|| error("Invalid rotated width"))? as u32,
        height: rotated.metadata["height"]
            .as_u64()
            .ok_or_else(|| error("Invalid rotated height"))? as u32,
        png: rotated.png,
        scale: page.scale,
    })
}

pub(super) fn restore(
    model: &mut Value,
    label: &str,
    width: u32,
    height: u32,
) -> Result<(), Error> {
    for item in model["layout_dets"]
        .as_array_mut()
        .ok_or_else(|| error("Invalid oriented layout"))?
    {
        item["rotate_label"] = json!(label);
        if !matches!(label, "90" | "270") {
            continue;
        }
        let p = item["poly"]
            .as_array()
            .filter(|p| p.len() == 8)
            .ok_or_else(|| error("Invalid oriented polygon"))?;
        let n = |i: usize| {
            p[i].as_f64()
                .filter(|n| n.is_finite())
                .ok_or_else(|| error("Invalid oriented coordinate"))
        };
        let (xmin, ymin, xmax, ymax) = (n(0)?, n(1)?, n(4)?, n(5)?);
        let [x0, y0, x1, y1] = if label == "90" {
            [
                width as f64 - 1. - ymax,
                xmin,
                width as f64 - 1. - ymin,
                xmax,
            ]
        } else {
            [
                ymin,
                height as f64 - 1. - xmax,
                ymax,
                height as f64 - 1. - xmin,
            ]
        };
        item["poly"] = json!([x0, y0, x1, y0, x1, y1, x0, y1]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_restoration_including_unrotated_180_label() {
        for (label, expected) in [
            ("90", json!([69., 10., 79., 10., 79., 40., 69., 40.])),
            ("270", json!([20., 159., 30., 159., 30., 189., 20., 189.])),
            ("180", json!([10., 20., 40., 20., 40., 30., 10., 30.])),
        ] {
            let mut m = json!({"layout_dets":[{"poly":[10.,20.,40.,20.,40.,30.,10.,30.],"bbox":[10.,20.,40.,30.]}]});
            restore(&mut m, label, 100, 200).unwrap();
            assert_eq!(m["layout_dets"][0]["poly"], expected);
            assert_eq!(m["layout_dets"][0]["rotate_label"], label);
            assert_eq!(m["layout_dets"][0]["bbox"], json!([10., 20., 40., 30.]));
        }
    }
}
