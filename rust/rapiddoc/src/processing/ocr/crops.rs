//! 旧 get_res_list_from_layout_res / BatchAnalyze 区域收集与 crop_img 像素规划。
use super::rect;
use crate::legacy_page::{assets::Pixels, invalid, limit, unsupported, validate_tree};
use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder};
use rapiddoc_core::Error;
use serde_json::{json, Value};

pub struct Output {
    pub plan: Value,
    pub images: Vec<(String, Vec<u8>)>,
    pub formula_pixels: Vec<(String, rapiddoc_core::pillow_resize::Rgb)>,
}
struct Crops {
    page: Pixels,
    index: u64,
    images: Vec<(String, Vec<u8>)>,
    formula_pixels: Vec<(String, rapiddoc_core::pillow_resize::Rgb)>,
    raw_formula: bool,
    pixels: usize,
    bytes: usize,
}
impl Crops {
    fn image(
        &mut self,
        b: [i64; 4],
        padding: i64,
        numpy: bool,
    ) -> Result<(String, [i64; 8]), Error> {
        self.image_impl(b, padding, numpy, false)
    }
    fn image_impl(
        &mut self,
        b: [i64; 4],
        padding: i64,
        numpy: bool,
        raw: bool,
    ) -> Result<(String, [i64; 8]), Error> {
        let (w, h) = (self.page.width as i64, self.page.height as i64);
        let slice = |v: i64, n: i64| if v < 0 { (v + n).max(0) } else { v.min(n) };
        let clipped = if numpy {
            [
                slice(b[0], w),
                slice(b[1], h),
                slice(b[2], w),
                slice(b[3], h),
            ]
        } else {
            if b[0] < 0 || b[1] < 0 || b[2] > w || b[3] > h {
                return Err(unsupported(
                    "旧 crop_img 对越界区域赋值会失败，不能补零替代",
                ));
            }
            b
        };
        let cw = (clipped[2] - clipped[0]).max(0);
        let ch = (clipped[3] - clipped[1]).max(0);
        if cw == 0 || ch == 0 {
            return Err(unsupported("空模型裁图不能编码为PNG"));
        }
        let out_w = cw + padding * 2;
        let out_h = ch + padding * 2;
        let count = (out_w as usize)
            .checked_mul(out_h as usize)
            .filter(|n| {
                *n <= if crate::legacy_page::budget::current().host {
                    rapiddoc_core::resources::limit(16_000_000)
                } else {
                    4_000_000
                }
            })
            .ok_or_else(limit)?;
        self.pixels = self
            .pixels
            .checked_add(count)
            .filter(|n| *n <= rapiddoc_core::resources::limit(32_000_000))
            .ok_or_else(limit)?;
        let serial = self.images.len() + self.formula_pixels.len();
        if serial >= rapiddoc_core::resources::limit(1000) {
            return Err(limit());
        }
        let mut pixels = vec![255; count * 3];
        for y in 0..ch as usize {
            let src = ((clipped[1] as usize + y) * self.page.width + clipped[0] as usize) * 3;
            let dst = ((padding as usize + y) * out_w as usize + padding as usize) * 3;
            pixels[dst..dst + cw as usize * 3]
                .copy_from_slice(&self.page.data[src..src + cw as usize * 3]);
        }
        let id = format!("model-{}-{}", self.index, serial);
        let bytes = if raw {
            pixels.len()
        } else {
            let mut png = Vec::new();
            PngEncoder::new(&mut png)
                .write_image(&pixels, out_w as u32, out_h as u32, ExtendedColorType::Rgb8)
                .map_err(|_| invalid())?;
            let bytes = png.len();
            self.images.push((id.clone(), png));
            bytes
        };
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .filter(|n| *n <= rapiddoc_core::resources::limit(128_000_000))
            .ok_or_else(limit)?;
        if raw {
            self.formula_pixels.push((
                id.clone(),
                rapiddoc_core::pillow_resize::Rgb {
                    data: pixels,
                    w: out_w as usize,
                    h: out_h as usize,
                },
            ));
        }
        Ok((id, [padding, padding, b[0], b[1], b[2], b[3], out_w, out_h]))
    }
}
fn int_rect(b: [f64; 4]) -> [i64; 4] {
    b.map(|v| v as i64)
}
fn rounded_rect(d: &Value) -> Result<[i64; 4], Error> {
    Ok(rect(d)?.map(|v| v.round_ties_even() as i64))
}
fn intersects(a: [i64; 2], b: [i64; 2]) -> bool {
    a[0].max(b[0]) < a[1].min(b[1])
}
fn expand(
    index: usize,
    dets: &[Value],
    width: i64,
    height: i64,
    px: i64,
) -> Result<[i64; 4], Error> {
    let original = int_rect(rect(&dets[index])?);
    if px <= 0 {
        return Ok(original);
    }
    let b = rounded_rect(&dets[index])?;
    let mut out = [
        (b[0] - px).max(0),
        (b[1] - px).max(0),
        (b[2] + px).min(width),
        (b[3] + px).min(height),
    ];
    for (i, d) in dets.iter().enumerate() {
        if i == index {
            continue;
        }
        let o = rounded_rect(d)?;
        if o[2] <= b[0] && intersects([out[1], out[3]], [o[1], o[3]]) {
            out[0] = out[0].max(o[2]);
        }
        if o[0] >= b[2] && intersects([out[1], out[3]], [o[1], o[3]]) {
            out[2] = out[2].min(o[0]);
        }
        if o[3] <= b[1] && intersects([out[0], out[2]], [o[0], o[2]]) {
            out[1] = out[1].max(o[3]);
        }
        if o[1] >= b[3] && intersects([out[0], out[2]], [o[0], o[2]]) {
            out[3] = out[3].min(o[1]);
        }
    }
    Ok(if out[0] >= out[2] || out[1] >= out[3] {
        original
    } else {
        out
    })
}
fn inside(a: [i64; 4], b: [i64; 4]) -> bool {
    let x = a[0].max(b[0]);
    let y = a[1].max(b[1]);
    let r = a[2].min(b[2]);
    let t = a[3].min(b[3]);
    r > x && t > y && ((r - x) * (t - y)) as f64 >= 0.8 * ((a[2] - a[0]) * (a[3] - a[1])) as f64
}

/// 仅规划/导出区域；模型推理执行器必须继续按旧公式→OCR→表格→后处理顺序执行。
pub fn prepare(request: &Value, png: &[u8]) -> Result<Output, Error> {
    prepare_impl(request, png, false)
}
/// Production callers consume formula pixels directly. Polygon masks when
/// expansion is zero still use the original PNG route and masking kernel.
pub(crate) fn prepare_with_formula_pixels(request: &Value, png: &[u8]) -> Result<Output, Error> {
    prepare_impl(request, png, true)
}
fn prepare_impl(request: &Value, png: &[u8], raw_formula: bool) -> Result<Output, Error> {
    validate_tree(request)?;
    let index = request["page_index"]
        .as_u64()
        .filter(|n| *n <= rapiddoc_core::resources::limit(1_000_000_000))
        .ok_or_else(invalid)?;
    let expansion = request
        .get("formula_bbox_expand_px")
        .map_or(Some(2), Value::as_i64)
        .filter(|n| (0..=1024).contains(n))
        .ok_or_else(invalid)?;
    let mut model = request["model"].clone();
    let dets = model["layout_dets"].as_array_mut().ok_or_else(invalid)?;
    if dets.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let page = Pixels::decode(png)?;
    if page.channels != 3 {
        return Err(unsupported("模型路由页图仅支持8-bit RGB PNG"));
    }
    let mut crops = Crops {
        page,
        index,
        images: Vec::new(),
        formula_pixels: Vec::new(),
        raw_formula: raw_formula && expansion > 0,
        pixels: 0,
        bytes: 0,
    };
    let (mut text, mut tables, mut formulas, mut images) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (i, d) in dets.iter_mut().enumerate() {
        let b = rect(d)?;
        if d.get("layout_image_list").is_some() {
            return Err(unsupported(
                "当前任务收集不接受已有 layout_image_list，须由页面图片区域生成",
            ));
        }
        let category = d["category_id"].as_i64().ok_or_else(invalid)?;
        if category == 3 {
            images.push(i);
        }
        match category {
            0 | 1 | 2 | 4 | 6 | 7 | 9 => text.push(i),
            5 => tables.push(i),
            8 | 13 | 14 => {
                d["bbox"] = json!(int_rect(b));
                formulas.push(i);
            }
            _ => {}
        }
    }
    // 与原函数一致：图片外层、表格内层；同图属于多个表格时重复裁图，不合并引用。
    for &image in &images {
        for &table in &tables {
            if inside(int_rect(rect(&dets[image])?), int_rect(rect(&dets[table])?)) {
                let (source, _) = crops.image(int_rect(rect(&dets[image])?), 0, false)?;
                let entry = json!({"uuid":format!("table-image-{index}-{image}-{table}"),"poly":dets[image]["poly"],"source":source});
                if dets[table].get("layout_image_list").is_none() {
                    dets[table]["layout_image_list"] = json!([]);
                }
                dets[table]["layout_image_list"]
                    .as_array_mut()
                    .unwrap()
                    .push(entry);
            }
        }
    }
    let mut table_tasks = Vec::new();
    for i in tables {
        if dets[i]
            .get("html")
            .is_some_and(|v| v.as_str().is_some_and(|s| !s.is_empty()))
        {
            continue;
        }
        let b = rect(&dets[i])?;
        let b = [
            (b[0] / 5.).floor() as i64 * 5,
            (b[1] / 5.).floor() as i64 * 5,
            (b[2] / 5.).ceil() as i64 * 5,
            (b[3] / 5.).ceil() as i64 * 5,
        ];
        let (source, useful) = crops.image(b, 0, true)?;
        table_tasks.push(json!({"layout_index":i,"source":source,"useful_list":useful}));
    }
    let mut formula_tasks = Vec::new();
    for i in formulas {
        let b = expand(
            i,
            dets,
            crops.page.width as i64,
            crops.page.height as i64,
            expansion,
        )?;
        let (source, useful) = crops.image_impl(b, 0, false, crops.raw_formula)?;
        formula_tasks.push(json!({"layout_index":i,"source":source,"useful_list":useful}));
    }
    let mut text_tasks = Vec::new();
    for i in text {
        let (source, useful) = crops.image(int_rect(rect(&dets[i])?), 50, false)?;
        text_tasks.push(json!({"layout_index":i,"source":source,"useful_list":useful}));
    }
    Ok(Output {
        plan: json!({"model":model,"ocr":text_tasks,"tables":table_tasks,"formulas":formula_tasks}),
        images: crops.images,
        formula_pixels: crops.formula_pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_formula_crops_preserve_original_pixels_plans_and_source_ids() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/legacy-model.json")).unwrap(),
        )
        .unwrap();
        for case in f["crops"].as_array().unwrap() {
            let png = std::fs::read(root.join(case["source"].as_str().unwrap())).unwrap();
            for expansion in [0, 2] {
                let mut request = case["request"].clone();
                request["formula_bbox_expand_px"] = json!(expansion);
                let original = prepare(&request, &png).unwrap();
                let raw = prepare_with_formula_pixels(&request, &png).unwrap();
                assert_eq!(raw.plan, original.plan);
                assert_eq!(
                    raw.images.len() + raw.formula_pixels.len(),
                    original.images.len()
                );
                if expansion == 0 {
                    assert!(raw.formula_pixels.is_empty());
                }
                for (key, encoded) in &raw.images {
                    assert_eq!(
                        encoded,
                        &original.images.iter().find(|(k, _)| k == key).unwrap().1
                    );
                }
                for (key, rgb) in &raw.formula_pixels {
                    let encoded = &original.images.iter().find(|(k, _)| k == key).unwrap().1;
                    let decoded = Pixels::decode(encoded).unwrap();
                    assert_eq!((rgb.w, rgb.h), (decoded.width, decoded.height));
                    assert_eq!(rgb.data, decoded.data);
                    let a = rapiddoc_inference::formula_preprocess::prepare(
                        &rgb.data, rgb.w, rgb.h, 384, 384,
                    )
                    .unwrap();
                    let b = rapiddoc_inference::formula_preprocess::from_png(encoded, 384, 384)
                        .unwrap();
                    assert_eq!(a.rgb, b.rgb);
                    assert_eq!(a.values, b.values);
                }
            }
        }
    }
}
