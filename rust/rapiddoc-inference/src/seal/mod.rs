//! Original RapidDoc/Paddle seal detection: polygon DB, Clipper 6 round offset,
//! and the separate polygon rectification path. No ordinary quad approximation.
use image::{GrayImage, Luma};
use imageproc::{contours::find_contours, point::Point};
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;
mod crop;
pub use crop::Cropper;
type P = [f64; 2];
fn error(message: impl ToString) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.to_string(),
    }
}
#[derive(Debug, Serialize)]
pub struct Detection {
    pub points: Vec<P>,
    pub score: f64,
}
unsafe extern "C" {
    fn rd_seal_offset(
        xy: *const i64,
        n: usize,
        delta: f64,
        result: *mut i64,
        capacity: usize,
    ) -> isize;
}

pub fn unclip(points: &[P], ratio: f64) -> Result<Option<Vec<P>>, Error> {
    if points.len() < 3
        || points.len() > rapiddoc_core::resources::limit(1_000_000)
        || !ratio.is_finite()
        || !(0. ..=10.).contains(&ratio)
        || points
            .iter()
            .flatten()
            .any(|x| !x.is_finite() || x.abs() > rapiddoc_core::resources::limit(10_000_000.))
    {
        return Err(error("Invalid seal polygon"));
    }
    let mut area = 0.;
    let mut perimeter = 0.;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        area += (a[0] + b[0]) * (a[1] - b[1]);
        let dx = a[0] as f32 - b[0] as f32;
        let dy = a[1] as f32 - b[1] as f32;
        perimeter += (dx * dx + dy * dy).sqrt() as f64;
    }
    if perimeter == 0. {
        return Err(error("Degenerate seal polygon perimeter"));
    }
    let delta = area.abs() * 0.5 * ratio / perimeter;
    offset_polygon(points, delta)
}

/// Original Clipper offset shared by ordinary DB quads and seal polygons.
pub(crate) fn offset_polygon(points: &[P], delta: f64) -> Result<Option<Vec<P>>, Error> {
    let input = points
        .iter()
        .flat_map(|p| p.map(|v| v as i64))
        .collect::<Vec<_>>();
    // Output can grow at rounded corners; bounded by input length and offset arc steps.
    let capacity = points
        .len()
        .checked_mul(64)
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(|| error("Seal offset size overflow"))?;
    let mut output = vec![0i64; capacity * 2];
    // SAFETY: both buffers own the advertised number of pairs; exceptions stay in C++.
    let count = unsafe {
        rd_seal_offset(
            input.as_ptr(),
            points.len(),
            delta,
            output.as_mut_ptr(),
            capacity,
        )
    };
    match count {
        0 | -1 => Ok(None),
        -2 => Err(Error {
            code: ErrorCode::ResourceLimit,
            message: "Seal offset allocation budget exceeded".into(),
        }),
        n if n < 0 => Err(error("Clipper seal offset failed")),
        n => Ok(Some(
            output[..n as usize * 2]
                .chunks_exact(2)
                .map(|p| [p[0] as f64, p[1] as f64])
                .collect(),
        )),
    }
}
fn simple(points: &[P]) -> Vec<P> {
    if points.len() < 3 {
        return points.to_vec();
    }
    (0..points.len())
        .filter_map(|i| {
            let a = points[(i + points.len() - 1) % points.len()];
            let b = points[i];
            let c = points[(i + 1) % points.len()];
            let before = [
                ((b[0] - a[0]) as i64).signum(),
                ((b[1] - a[1]) as i64).signum(),
            ];
            let after = [
                ((c[0] - b[0]) as i64).signum(),
                ((c[1] - b[1]) as i64).signum(),
            ];
            (before != after).then_some(b)
        })
        .collect()
}
pub fn prepare(
    bytes: &[u8],
    width: usize,
    height: usize,
) -> Result<crate::preprocess::PreparedTensor, Error> {
    crate::detection_preprocess::prepare(
        bytes,
        width,
        height,
        &crate::detection_preprocess::DetectionConfig {
            side: 736,
            mode: crate::detection_preprocess::LimitType::Minimum,
            maximum_side: 32768,
            max_elements: rapiddoc_core::resources::limit(32_000_000),
            ..Default::default()
        },
    )
}
/// crop_img's tight RGB crop and polygon whitening, converted directly to BGR.
pub fn layout_crop(
    rgb: &[u8],
    width: usize,
    height: usize,
    bounds: [i64; 4],
    polygon: &[P],
) -> Result<crate::recognition_batch::Image, Error> {
    let [x0, y0, x1, y1] = bounds;
    if x0 < 0
        || y0 < 0
        || x1 > x0 + width as i64
        || x1 > width as i64
        || y1 > height as i64
        || x1 <= x0
        || y1 <= y0
        || width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(rgb.len())
    {
        return Err(error("Seal layout crop is empty or out of bounds"));
    }
    let w = (x1 - x0) as usize;
    let h = (y1 - y0) as usize;
    crate::preprocess::check_size(w, h, 64_000_000)?;
    let mut bgr = Vec::with_capacity(w * h * 3);
    for y in y0 as usize..y1 as usize {
        for p in rgb[(y * width + x0 as usize) * 3..(y * width + x1 as usize) * 3].chunks_exact(3) {
            bgr.extend_from_slice(&[p[2], p[1], p[0]]);
        }
    }
    if !polygon.is_empty() {
        let points = polygon
            .iter()
            .map(|p| Point::new(p[0] as i32 - x0 as i32, p[1] as i32 - y0 as i32))
            .collect::<Vec<_>>();
        let mut mask = GrayImage::new(w as u32, h as u32);
        crate::detection_mask::fill(&mut mask, &points);
        for (pixel, mask) in bgr.chunks_exact_mut(3).zip(mask.as_raw()) {
            if *mask == 0 {
                pixel.fill(255)
            }
        }
    }
    Ok(crate::recognition_batch::Image {
        bgr,
        width: w,
        height: h,
    })
}
pub fn polygons(
    values: &[f32],
    width: usize,
    height: usize,
    source_width: usize,
    source_height: usize,
) -> Result<Vec<Detection>, Error> {
    crate::preprocess::check_size(width, height, 16_000_000)?;
    crate::preprocess::check_size(source_width, source_height, 64_000_000)?;
    if width.checked_mul(height) != Some(values.len())
        || values
            .iter()
            .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
    {
        return Err(error("Invalid seal probability map"));
    }
    let mut bitmap = GrayImage::new(width as u32 + 2, height as u32 + 2);
    for y in 0..height {
        for x in 0..width {
            bitmap.put_pixel(
                x as u32 + 1,
                y as u32 + 1,
                Luma([u8::from(values[y * width + x] > 0.2)]),
            );
        }
    }
    let mut boxes = vec![];
    let mut work = usize::MAX;
    for contour in find_contours::<i32>(&bitmap).into_iter().rev().take(1000) {
        let raw = contour
            .points
            .into_iter()
            .map(|p| [(p.x - 1) as f64, (p.y - 1) as f64])
            .collect::<Vec<_>>();
        let points = simple(&raw);
        let perimeter = (0..points.len())
            .map(|i| {
                let a = points[i];
                let b = points[(i + 1) % points.len()];
                let dx = (b[0] - a[0]) as f32;
                let dy = (b[1] - a[1]) as f32;
                (dx * dx + dy * dy).sqrt() as f64
            })
            .sum::<f64>();
        let points = crate::layout_contour::approximate(&points, perimeter * 0.002)?;
        if points.len() < 4 {
            continue;
        }
        let score = crate::detection::score(values, width, height, &points, &mut work)?;
        if score < 0.6 {
            continue;
        }
        let Some(mut points) = unclip(&points, 0.5)? else {
            continue;
        };
        let integer = points
            .iter()
            .map(|p| Point::new(p[0] as i32, p[1] as i32))
            .collect::<Vec<_>>();
        if crate::detection_geometry::rectangle(&integer).is_none_or(|(_, short)| short < 5.) {
            continue;
        }
        let sx = source_width as f64 / width as f64;
        let sy = source_height as f64 / height as f64;
        for p in &mut points {
            p[0] = (p[0] * sx).round_ties_even().clamp(0., source_width as f64);
            p[1] = (p[1] * sy)
                .round_ties_even()
                .clamp(0., source_height as f64);
        }
        boxes.push(Detection { points, score });
    }
    // patched TextDetector retains raw polygon order whenever a polygon has >4 vertices.
    if boxes.iter().all(|b| b.points.len() == 4) {
        let mut quads = boxes
            .iter()
            .map(|b| crate::detection::Detection {
                points: std::array::from_fn(|i| b.points[i].map(|v| v as f32)),
                score: b.score,
            })
            .collect::<Vec<_>>();
        crate::detection::sort_reading_order(&mut quads);
        let mut sorted = Vec::with_capacity(boxes.len());
        for q in quads {
            let i = boxes
                .iter()
                .position(|b| {
                    b.points
                        .iter()
                        .zip(q.points)
                        .all(|(a, b)| a.map(|v| v as f32) == b)
                })
                .ok_or_else(|| error("Seal box ordering failed"))?;
            sorted.push(boxes.remove(i));
        }
        boxes = sorted;
    }
    let y = boxes
        .iter()
        .map(|b| b.points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min))
        .collect::<Vec<_>>();
    let order = crate::numpy_argsort::indices(&y)?;
    let mut slots = boxes.into_iter().map(Some).collect::<Vec<_>>();
    Ok(order
        .into_iter()
        .map(|i| slots[i].take().unwrap())
        .collect())
}
