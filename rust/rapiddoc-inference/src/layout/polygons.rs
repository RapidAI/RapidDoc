//! PP-DocLayoutV3 full mask/auto/poly/quad postprocessing, preserving source order.
use super::{
    layout_contour, layout_geos,
    named::{Output, OutputRef},
};
use rapiddoc_core::{Error, ErrorCode};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
type P = [f64; 2];
fn error(message: &str) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.into(),
    }
}
fn iou(a: &[f32; 7], b: &[f32; 7]) -> f32 {
    let intersection = (a[4].min(b[4]) - a[2].max(b[2]) + 1.).max(0.)
        * (a[5].min(b[5]) - a[3].max(b[3]) + 1.).max(0.);
    let aa = (a[4] - a[2] + 1.) * (a[5] - a[3] + 1.);
    let ba = (b[4] - b[2] + 1.) * (b[5] - b[3] + 1.);
    intersection / (aa + ba - intersection)
}
fn contained(a: &[f32; 7], b: &[f32; 7]) -> bool {
    let area = (a[4] - a[2]) * (a[5] - a[3]);
    let inter =
        (a[4].min(b[4]) - a[2].max(b[2])).max(0.) * (a[5].min(b[5]) - a[3].max(b[3])).max(0.);
    area > 0. && inter / area >= 0.9
}
fn selected(
    rows: &[[f32; 7]],
    labels: &[String],
    width: usize,
    height: usize,
    threshold: f32,
) -> Vec<usize> {
    let mut indices = (0..rows.len())
        .filter(|&i| rows[i][0] > -1. && rows[i][1] > threshold)
        .collect::<Vec<_>>();
    indices.sort_by(|&a, &b| rows[b][1].total_cmp(&rows[a][1]).then(b.cmp(&a)));
    let mut kept = Vec::<usize>::new();
    for i in indices {
        if kept
            .iter()
            .all(|&j| iou(&rows[i], &rows[j]) < if rows[i][0] == rows[j][0] { 0.6 } else { 0.98 })
        {
            kept.push(i)
        }
    }
    if kept.len() > 1 {
        let image = labels.iter().position(|s| s == "image");
        let limit =
            ((width as f64 * height as f64) * if width > height { 0.82 } else { 0.93 }) as f32;
        let filtered = kept
            .iter()
            .copied()
            .filter(|&i| {
                let r = rows[i];
                Some(r[0] as usize) != image
                    || (r[4].min(width as f32) - r[2].max(0.))
                        * (r[5].min(height as f32) - r[3].max(0.))
                        <= limit
            })
            .collect::<Vec<_>>();
        if !filtered.is_empty() {
            kept = filtered
        }
    }
    let formula = labels.iter().position(|s| s == "formula");
    let mut dropped = vec![false; rows.len()];
    for &i in &kept {
        for &j in &kept {
            if i == j {
                continue;
            }
            if Some(rows[i][0] as usize) == formula && Some(rows[j][0] as usize) != formula {
                continue;
            }
            let large: &[usize] = if labels.len() == 20 {
                &[0, 1, 7, 16]
            } else {
                &[3, 5, 6, 15, 17]
            };
            if large.contains(&(rows[j][0] as usize)) && contained(&rows[i], &rows[j]) {
                dropped[i] = true
            }
        }
    }
    kept.retain(|&i| !dropped[i]);
    kept.sort_by(|&a, &b| rows[a][6].total_cmp(&rows[b][6]));
    kept
}
fn quad(points: &[P]) -> Option<Vec<P>> {
    let mut q = super::detection_geometry::rectangle_float(points)?.map(|p| p.map(|v| v as f32));
    let cx = ((q[0][0] + q[1][0]) + q[2][0] + q[3][0]) / 4.;
    let cy = ((q[0][1] + q[1][1]) + q[2][1] + q[3][1]) / 4.;
    q.sort_by(|a, b| {
        (a[1] - cy)
            .atan2(a[0] - cx)
            .total_cmp(&(b[1] - cy).atan2(b[0] - cx))
    });
    let first = (0..4).min_by(|&a, &b| (q[a][0] + q[a][1]).total_cmp(&(q[b][0] + q[b][1])))?;
    q.rotate_left(first);
    Some(q.into_iter().map(|p| p.map(f64::from)).collect())
}
fn polygon_json(points: &[P], float: bool) -> Value {
    if float {
        json!(points)
    } else {
        json!(points
            .iter()
            .map(|p| p.map(|v| v as i64))
            .collect::<Vec<_>>())
    }
}
// Convert each used integer at the same FP32 precision as the original array cast.
// Do not allocate/cast the fixed 300-mask output before selecting actual regions.
trait MaskElement: Copy {
    fn fp32(self) -> f32;
}
impl MaskElement for u8 {
    fn fp32(self) -> f32 {
        self as f32
    }
}
impl MaskElement for f32 {
    #[inline]
    fn fp32(self) -> f32 {
        self
    }
}
impl MaskElement for i32 {
    #[inline]
    fn fp32(self) -> f32 {
        self as f32
    }
}
impl MaskElement for i64 {
    #[inline]
    fn fp32(self) -> f32 {
        self as f32
    }
}
/// Preserve INTER_NEAREST's reciprocal coordinate arithmetic and uint8 cast,
/// but calculate repeated horizontal samples only once per distinct source row.
#[allow(clippy::too_many_arguments)]
fn resize_mask<T: MaskElement>(
    values: &[T],
    mw: usize,
    offset: usize,
    xs: [usize; 2],
    ys: [usize; 2],
    bw: usize,
    bh: usize,
    pixels: usize,
) -> Vec<u8> {
    let crop_w = xs[1] - xs[0];
    let crop_h = ys[1] - ys[0];
    let x_scale = 1. / (bw as f64 / crop_w as f64);
    let y_scale = 1. / (bh as f64 / crop_h as f64);
    let columns: Vec<_> = (0..bw)
        .map(|x| ((x as f64 * x_scale) as usize).min(crop_w - 1) + xs[0])
        .collect();
    let mut resized = vec![0u8; pixels];
    let mut previous = None;
    for y in 0..bh {
        let source_y = ((y as f64 * y_scale) as usize).min(crop_h - 1) + ys[0];
        if previous == Some(source_y) {
            resized.copy_within((y - 1) * bw..y * bw, y * bw);
        } else {
            for (x, &source_x) in columns.iter().enumerate() {
                resized[y * bw + x] = (values[offset + source_y * mw + source_x].fp32().trunc()
                    as i64)
                    .rem_euclid(256) as u8;
            }
        }
        previous = Some(source_y);
    }
    resized
}

#[allow(clippy::too_many_arguments)]
pub fn process(
    rows: &[[f32; 7]],
    masks: Option<(&[f32], usize, usize)>,
    labels: &[String],
    width: usize,
    height: usize,
    threshold: f32,
    mode: &str,
    geos: Option<&Path>,
) -> Result<Vec<Value>, Error> {
    process_typed(rows, masks, labels, width, height, threshold, mode, geos)
}
#[allow(clippy::too_many_arguments)]
fn process_typed<T: MaskElement>(
    rows: &[[f32; 7]],
    masks: Option<(&[T], usize, usize)>,
    labels: &[String],
    width: usize,
    height: usize,
    threshold: f32,
    mode: &str,
    geos: Option<&Path>,
) -> Result<Vec<Value>, Error> {
    if !matches!(mode, "auto" | "rect" | "poly" | "quad")
        || width == 0
        || height == 0
        || !threshold.is_finite()
        || !(0. ..=1.).contains(&threshold)
        || rows
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > 1e9)
    {
        return Err(error("Invalid layout postprocessing input"));
    }
    if rows
        .iter()
        .any(|r| r[0] >= 0. && (r[0].fract() != 0. || r[0] as usize >= labels.len()))
    {
        return Err(error("Unknown layout label"));
    }
    if let Some((values, mw, mh)) = masks {
        if mw.checked_mul(mh).and_then(|n| n.checked_mul(rows.len())) != Some(values.len())
            || values.iter().any(|v| !v.fp32().is_finite())
        {
            return Err(error("Invalid layout mask dimensions"));
        }
    }
    let ids = selected(rows, labels, width, height, threshold);
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let max_width = ids
        .iter()
        .map(|&i| rows[i][4] - rows[i][3])
        .fold(f32::NEG_INFINITY, f32::max);
    let use_masks = mode != "rect" && masks.is_some();
    let mut previous: Option<Vec<P>> = None;
    let mut context = None;
    let mut result = vec![];
    for (order, &i) in ids.iter().enumerate() {
        let r = rows[i];
        let mut polygon = None;
        if use_masks {
            let (values, mw, mh) = masks.unwrap();
            let [x0, y0, x1, y1] = [r[2], r[3], r[4], r[5]].map(|v| v as i32);
            let (bw, bh) = (x1 - x0, y1 - y0);
            let rect = vec![
                [x0 as f64, y0 as f64],
                [x1 as f64, y0 as f64],
                [x1 as f64, y1 as f64],
                [x0 as f64, y1 as f64],
            ];
            let mut points = rect.clone();
            let mut floating = true;
            if bw > 0 && bh > 0 {
                let sx = 800. / width as f64 / 4.;
                let sy = 800. / height as f64 / 4.;
                let xs = [x0, x1].map(|x| {
                    ((x as f64 * sx).round_ties_even() as i64).clamp(0, mw as i64) as usize
                });
                let ys = [y0, y1].map(|y| {
                    ((y as f64 * sy).round_ties_even() as i64).clamp(0, mh as i64) as usize
                });
                let crop_w = xs[1].saturating_sub(xs[0]);
                let crop_h = ys[1].saturating_sub(ys[0]);
                let offset = i * mw * mh;
                let mut sum = 0f32;
                for y in ys[0]..ys[1] {
                    for x in xs[0]..xs[1] {
                        sum += values[offset + y * mw + x].fp32()
                    }
                }
                if crop_w > 0 && crop_h > 0 && sum != 0. {
                    let pixels = (bw as usize)
                        .checked_mul(bh as usize)
                        .filter(|&n| n <= rapiddoc_core::resources::limit(16_000_000))
                        .ok_or_else(|| Error {
                            code: ErrorCode::ResourceLimit,
                            message: "Layout mask exceeds pixel budget".into(),
                        })?;
                    let resized =
                        resize_mask(values, mw, offset, xs, ys, bw as usize, bh as usize, pixels);
                    let max_distance = if bw as f64 > (max_width * 0.6) as f64 {
                        bw as f64 * 0.3
                    } else {
                        (max_width * 0.3) as f64
                    };
                    let found = layout_contour::mask_polygon(
                        &resized,
                        bw as usize,
                        bh as usize,
                        max_distance,
                    )?;
                    if let Some((mut poly, float)) = found {
                        if poly.len() >= 4 {
                            for p in &mut poly {
                                p[0] += x0 as f64;
                                p[1] += y0 as f64;
                            }
                            match mode {
                                "poly" => {
                                    points = poly;
                                    floating = float
                                }
                                "quad" => {
                                    points = quad(&poly).unwrap_or(rect.clone());
                                }
                                "auto" => {
                                    if context.is_none() {
                                        context =
                                            Some(layout_geos::Context::load(geos.ok_or_else(
                                                || error("GEOS runtime required for auto layout"),
                                            )?)?)
                                    }
                                    let context = context.as_ref().unwrap();
                                    if let Some(mut q) = quad(&poly) {
                                        let original = q.clone();
                                        if context.overlap(&rect, &original, "union")? >= 0.95 {
                                            q = rect.clone()
                                        }
                                        let overlap = context.overlap(&poly, &original, "union")?;
                                        let before = previous
                                            .as_ref()
                                            .map(|p| context.overlap(p, &rect, "small"))
                                            .transpose()?
                                            .unwrap_or(0.);
                                        if overlap >= 0.8 && before < 0.01 {
                                            points = q
                                        } else {
                                            points = poly;
                                            floating = float
                                        }
                                    } else {
                                        points = poly;
                                        floating = float
                                    }
                                }
                                _ => unreachable!(),
                            }
                        }
                    } else if mode != "quad" {
                        previous = None;
                        continue;
                    }
                }
            }
            previous = Some(points.clone());
            polygon = Some(polygon_json(&points, floating));
        }
        let w = r[4] - r[2];
        let h = r[5] - r[3];
        let cx = r[2] + w / 2.;
        let cy = r[3] + h / 2.;
        let bounds = [
            (cx - w / 2.).max(0.),
            (cy - h / 2.).max(0.),
            (cx + w / 2.).min(width as f32),
            (cy + h / 2.).min(height as f32),
        ];
        if bounds[2] <= bounds[0] || bounds[3] <= bounds[1] {
            continue;
        }
        let mut region = json!({"cls_id":r[0]as usize,"label":labels[r[0]as usize],"score":r[1]as f64,"coordinate":bounds.map(f64::from),"order":order+1});
        if let Some(p) = polygon {
            region["polygon_points"] = p
        }
        result.push(region);
    }
    Ok(result)
}

pub fn outputs(
    outputs: BTreeMap<String, Output>,
    labels: &[String],
    width: usize,
    height: usize,
    threshold: f32,
    mode: &str,
    geos: Option<&Path>,
) -> Result<Vec<Value>, Error> {
    outputs_ref(
        outputs
            .iter()
            .map(|(name, value)| (name.clone(), value.as_ref()))
            .collect(),
        labels,
        width,
        height,
        threshold,
        mode,
        geos,
    )
}

pub(crate) fn outputs_ref(
    outputs: BTreeMap<String, OutputRef<'_>>,
    labels: &[String],
    width: usize,
    height: usize,
    threshold: f32,
    mode: &str,
    geos: Option<&Path>,
) -> Result<Vec<Value>, Error> {
    let mut outputs = outputs;
    let count = match outputs.remove("fetch_name_1") {
        Some(OutputRef::Int32 { values, .. }) => values.first().copied().map(i64::from),
        Some(OutputRef::Int64 { values, .. }) => values.first().copied(),
        _ => None,
    }
    .filter(|&n| n >= 0)
    .ok_or_else(|| error("Missing layout count"))? as usize;
    let ordered;
    let rows = match outputs.remove("fetch_name_0") {
        Some(OutputRef::Float32 { shape, values })
            if shape.len() == 2 && matches!(shape[1], 6 | 7 | 8) =>
        {
            ordered = shape[1] >= 7;
            let raw = values
                .chunks_exact(shape[1] as usize)
                .take(count)
                .collect::<Vec<_>>();
            let mut ranks = (0..raw.len()).collect::<Vec<_>>();
            if shape[1] == 8 {
                ranks.sort_by(|&a, &b| {
                    raw[a][6]
                        .total_cmp(&raw[b][6])
                        .then(raw[b][7].total_cmp(&raw[a][7]))
                });
            }
            let mut order = vec![0f32; raw.len()];
            for (rank, &i) in ranks.iter().enumerate() {
                order[i] = rank as f32;
            }
            raw.into_iter()
                .enumerate()
                .map(|(i, r)| {
                    [
                        r[0],
                        r[1],
                        r[2],
                        r[3],
                        r[4],
                        r[5],
                        if shape[1] == 8 {
                            order[i]
                        } else if ordered {
                            r[6]
                        } else {
                            -r[1]
                        },
                    ]
                })
                .collect::<Vec<[f32; 7]>>()
        }
        _ => return Err(error("Layout boxes must be FP32 Nx6, Nx7 or Nx8")),
    };
    if rows.len() != count {
        return Err(error("Layout count exceeds rows"));
    }
    let mut result = match outputs.remove("fetch_name_2") {
        Some(OutputRef::UInt8 { shape, values }) => process_mask_output(
            &rows, shape, values, labels, width, height, threshold, mode, geos,
        ),
        Some(OutputRef::Float32 { shape, values }) => process_mask_output(
            &rows, shape, values, labels, width, height, threshold, mode, geos,
        ),
        Some(OutputRef::Int32 { shape, values }) => process_mask_output(
            &rows, shape, values, labels, width, height, threshold, mode, geos,
        ),
        Some(OutputRef::Int64 { shape, values }) => process_mask_output(
            &rows, shape, values, labels, width, height, threshold, mode, geos,
        ),
        None => process(&rows, None, labels, width, height, threshold, "rect", geos),
    }?;
    if !ordered {
        for region in &mut result {
            region.as_object_mut().unwrap().remove("order");
        }
    }
    Ok(result)
}
#[allow(clippy::too_many_arguments)]
fn process_mask_output<T: MaskElement>(
    rows: &[[f32; 7]],
    dims: &[i64],
    values: &[T],
    labels: &[String],
    width: usize,
    height: usize,
    threshold: f32,
    mode: &str,
    geos: Option<&Path>,
) -> Result<Vec<Value>, Error> {
    if dims.len() != 3 || dims.iter().any(|&n| n < 0) {
        return Err(error("Invalid layout mask tensor"));
    }
    let count_values = rows
        .len()
        .checked_mul(dims[1] as usize)
        .and_then(|n| n.checked_mul(dims[2] as usize))
        .ok_or_else(|| error("Layout tensor overflow"))?;
    let values = values
        .get(..count_values)
        .ok_or_else(|| error("Truncated layout masks"))?;
    process_typed(
        rows,
        Some((values, dims[2] as usize, dims[1] as usize)),
        labels,
        width,
        height,
        threshold,
        mode,
        geos,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    #[test]
    fn binary_uint8_masks_preserve_int32_polygons() {
        let labels = vec!["text".to_owned()];
        let base = BTreeMap::from([
            (
                "fetch_name_0".into(),
                Output::Float32 {
                    shape: vec![1, 7],
                    values: vec![0., 0.9, 0., 0., 800., 800., 1.],
                },
            ),
            (
                "fetch_name_1".into(),
                Output::Int32 {
                    shape: vec![1],
                    values: vec![1],
                },
            ),
        ]);
        for fill in [0u8, 1] {
            for mode in ["rect", "poly", "quad"] {
                let build = |mask| {
                    let mut out = base.clone();
                    out.insert("fetch_name_2".into(), mask);
                    outputs(out, &labels, 800, 800, 0.3, mode, None).unwrap()
                };
                assert_eq!(
                    build(Output::UInt8 {
                        shape: vec![1, 200, 200],
                        values: vec![fill; 40_000]
                    }),
                    build(Output::Int32 {
                        shape: vec![1, 200, 200],
                        values: vec![fill as i32; 40_000]
                    })
                );
            }
        }
    }
    #[test]
    fn original_polygon_layout_outputs() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let fixture: Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/layout-polygons.json")).unwrap(),
        )
        .unwrap();
        let labels = fixture["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        let geos =
            std::fs::read_dir(root.join("bindings/python/python/rapid_doc_rust/runtime/geos"))
                .unwrap()
                .filter_map(Result::ok)
                .map(|e| e.path())
                .find(|p| {
                    p.file_name()
                        .unwrap()
                        .to_string_lossy()
                        .starts_with("geos_c")
                })
                .unwrap();
        for c in fixture["cases"].as_array().unwrap() {
            let bytes = std::fs::read(root.join(c["masks"].as_str().unwrap())).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                c["mask_sha256"].as_str().unwrap()
            );
            let masks = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>();
            let shape = c["mask_shape"].as_array().unwrap();
            let rows = c["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    v.as_array()
                        .unwrap()
                        .iter()
                        .map(|n| n.as_f64().unwrap() as f32)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<[f32; 7]>>();
            let got = process(
                &rows,
                Some((
                    &masks,
                    shape[2].as_u64().unwrap() as usize,
                    shape[1].as_u64().unwrap() as usize,
                )),
                &labels,
                c["width"].as_u64().unwrap() as usize,
                c["height"].as_u64().unwrap() as usize,
                c["threshold"].as_f64().unwrap() as f32,
                c["mode"].as_str().unwrap(),
                Some(&geos),
            )
            .unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
            if json!(got) != c["expected"] {
                std::fs::write(
                    root.join("build/layout-polygon-native-difference.json"),
                    serde_json::to_vec_pretty(
                        &json!({"case":c["name"],"actual":got,"expected":c["expected"]}),
                    )
                    .unwrap(),
                )
                .ok();
            }
            assert_eq!(json!(got), c["expected"], "{}", c["name"]);
        }
    }
}

#[cfg(test)]
mod resize_tests {
    use super::*;
    #[test]
    fn repeated_mask_rows_preserve_nearest_coordinates_and_casts() {
        let width = 29;
        let values: Vec<f32> = (0..width * 23 * 2)
            .map(|i| ((i * 31) % 999) as f32 / 2. - 150.25)
            .collect();
        for (xs, ys) in [([0, 29], [0, 23]), ([3, 17], [2, 19]), ([8, 9], [7, 8])] {
            for (bw, bh) in [(1, 1), (4, 7), (37, 53), (127, 261), (800, 533)] {
                let mut expected = vec![0u8; bw * bh];
                let (cw, ch) = (xs[1] - xs[0], ys[1] - ys[0]);
                let sx = 1. / (bw as f64 / cw as f64);
                let sy = 1. / (bh as f64 / ch as f64);
                for y in 0..bh {
                    for x in 0..bw {
                        let yy = ((y as f64 * sy) as usize).min(ch - 1) + ys[0];
                        let xx = ((x as f64 * sx) as usize).min(cw - 1) + xs[0];
                        expected[y * bw + x] = (values[width * 23 + yy * width + xx].trunc() as i64)
                            .rem_euclid(256) as u8;
                    }
                }
                assert_eq!(
                    resize_mask(&values, width, width * 23, xs, ys, bw, bh, bw * bh),
                    expected
                );
                let binary: Vec<u8> = values.iter().map(|v| u8::from(*v > 0.)).collect();
                let floating: Vec<f32> = binary.iter().map(|v| *v as f32).collect();
                assert_eq!(
                    resize_mask(&binary, width, 0, xs, ys, bw, bh, bw * bh),
                    resize_mask(&floating, width, 0, xs, ys, bw, bh, bw * bh)
                );
            }
        }
    }
}
