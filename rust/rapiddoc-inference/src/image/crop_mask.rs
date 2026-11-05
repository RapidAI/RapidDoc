//! Original crop_img polygon whitening, before padding, using the shared LINE_8 kernel.
use image::{codecs::png::PngEncoder, ExtendedColorType, GrayImage, ImageEncoder};
use imageproc::point::Point;
use rapiddoc_core::{Error, ErrorCode};
use serde_json::Value;

fn invalid() -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: "Invalid polygon crop".into(),
    }
}

pub fn apply(plan: &Value, images: &mut [(String, Vec<u8>)], expansion: i64) -> Result<(), Error> {
    let dets = plan["model"]["layout_dets"]
        .as_array()
        .ok_or_else(invalid)?;
    // Tables are rectangular in the original pipeline. Expanded formulas discard polygon_points.
    for kind in ["ocr", "formulas"] {
        if kind == "formulas" && expansion > 0 {
            continue;
        }
        for task in plan[kind].as_array().ok_or_else(invalid)? {
            let index = task["layout_index"].as_u64().ok_or_else(invalid)? as usize;
            let Some(polygon) = dets
                .get(index)
                .and_then(|d| d.get("polygon_points"))
                .and_then(Value::as_array)
            else {
                continue;
            };
            if polygon.is_empty() {
                continue;
            }
            if polygon.len() > rapiddoc_core::resources::limit(100_000) {
                return Err(invalid());
            }
            let u = task["useful_list"]
                .as_array()
                .filter(|a| a.len() == 8)
                .ok_or_else(invalid)?;
            let mut bounds = [0i64; 6];
            for (out, v) in bounds.iter_mut().zip(u) {
                *out = v.as_i64().ok_or_else(invalid)?;
            }
            let [px, py, x0, y0, x1, y1] = bounds;
            let (cw, ch) = (x1 - x0, y1 - y0);
            if px < 0
                || py < 0
                || cw <= 0
                || ch <= 0
                || cw > rapiddoc_core::resources::limit(16_000_000) / ch
            {
                return Err(invalid());
            }
            let mut points = Vec::with_capacity(polygon.len());
            for v in polygon {
                let p = v.as_array().filter(|a| a.len() == 2).ok_or_else(invalid)?;
                let mut xy = [0i32; 2];
                for (i, origin) in [x0, y0].into_iter().enumerate() {
                    let coordinate = p[i]
                        .as_f64()
                        .filter(|n| n.is_finite() && n.abs() <= 1e9)
                        .ok_or_else(invalid)?;
                    xy[i] =
                        i32::try_from(coordinate.trunc() as i64 - origin).map_err(|_| invalid())?;
                }
                points.push(Point::new(xy[0], xy[1]));
            }
            let source = task["source"].as_str().ok_or_else(invalid)?;
            let (_, png) = images
                .iter_mut()
                .find(|(id, _)| id == source)
                .ok_or_else(invalid)?;
            let (mut bgr, w, h) = super::preprocess::decode_pipeline_rgb(png, 16_000_000)?;
            if px + cw > w as i64 || py + ch > h as i64 {
                return Err(invalid());
            }
            let mut mask = GrayImage::new(cw as u32, ch as u32);
            super::detection_mask::fill(&mut mask, &points);
            for y in 0..ch as usize {
                for x in 0..cw as usize {
                    if mask.get_pixel(x as u32, y as u32)[0] == 0 {
                        let offset = ((py as usize + y) * w + px as usize + x) * 3;
                        bgr[offset..offset + 3].fill(255);
                    }
                }
            }
            for pixel in bgr.as_chunks_mut::<3>().0 {
                pixel.swap(0, 2);
            }
            let mut output = Vec::new();
            PngEncoder::new(&mut output)
                .write_image(&bgr, w as u32, h as u32, ExtendedColorType::Rgb8)
                .map_err(|_| invalid())?;
            *png = output;
        }
    }
    Ok(())
}
