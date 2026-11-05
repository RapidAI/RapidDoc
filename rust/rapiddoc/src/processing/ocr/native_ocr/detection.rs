//! 旧 _apply_mask_boxes_to_image 和 64像素分辨率分组；生成独立检测PNG，不执行模型。
use super::{number, useful};
use crate::legacy_page::{assets::Pixels, invalid, limit, unsupported, validate_tree};
use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder};
use rapiddoc_core::Error;
use serde_json::{json, Value};

pub struct Output {
    pub plan: Value,
    pub images: Vec<(String, Vec<u8>)>,
    pub pixels: Vec<(String, rapiddoc_core::pillow_resize::Rgb)>,
}
pub fn prepare(request: &Value, pngs: &[Vec<u8>]) -> Result<Output, Error> {
    let sources: Vec<_> = pngs.iter().map(Vec::as_slice).collect();
    prepare_impl(request, &sources, false)
}
/// Production detection consumes owned RGB pixels. Keep the public PNG adapter
/// and the original grouping/masks/padding unchanged.
pub fn prepare_pixels(request: &Value, pngs: &[&[u8]]) -> Result<Output, Error> {
    prepare_impl(request, pngs, true)
}
fn prepare_impl(request: &Value, pngs: &[&[u8]], raw: bool) -> Result<Output, Error> {
    validate_tree(request)?;
    let index = request["page_index"]
        .as_u64()
        .filter(|n| *n <= rapiddoc_core::resources::limit(1_000_000_000))
        .ok_or_else(invalid)?;
    let plan = &request["plan"];
    let tasks = plan["detect"].as_array().ok_or_else(invalid)?;
    if tasks.len() > rapiddoc_core::resources::limit(1000)
        || pngs.len() > rapiddoc_core::resources::limit(1000)
        || pngs.iter().map(|p| p.len()).sum::<usize>()
            > rapiddoc_core::resources::limit(128_000_000)
    {
        return Err(limit());
    }
    if tasks.len() != pngs.len() {
        return Err(invalid());
    }
    let mut dimensions = Vec::new();
    let mut groups: Vec<((usize, usize), Vec<usize>)> = Vec::new();
    let mut pixels = 0usize;
    let mut masking = 0usize;
    // 先预检每张输入尺寸、白边目标和全批预算，再解码或编码任何像素。
    for (i, (t, png)) in tasks.iter().zip(pngs).enumerate() {
        if png.len() < 33 || !png.starts_with(b"\x89PNG\r\n\x1a\n") {
            return Err(invalid());
        }
        if png[24] != 8 || png[25] != 2 {
            return Err(unsupported("OCR检测输入仅支持8-bit RGB PNG"));
        }
        if png.len() > rapiddoc_core::resources::limit(64_000_000) {
            return Err(limit());
        }
        let w = u32::from_be_bytes(png[16..20].try_into().unwrap()) as usize;
        let h = u32::from_be_bytes(png[20..24].try_into().unwrap()) as usize;
        if w == 0 || h == 0 {
            return Err(invalid());
        }
        let u = useful(&t["useful_list"])?;
        if number(&u[6])? != w as f64 || number(&u[7])? != h as f64 {
            return Err(invalid());
        }
        let target = (h.div_ceil(64) * 64, w.div_ceil(64) * 64);
        let count = target
            .0
            .checked_mul(target.1)
            .filter(|n| {
                *n <= if crate::legacy_page::budget::current().host {
                    rapiddoc_core::resources::limit(16_000_000)
                } else {
                    4_000_000
                }
            })
            .ok_or_else(limit)?;
        pixels = pixels
            .checked_add(count)
            .filter(|n| *n <= rapiddoc_core::resources::limit(32_000_000))
            .ok_or_else(limit)?;
        dimensions.push((w, h));
        let masks = t["masks"].as_array().ok_or_else(invalid)?;
        if masks.len() > rapiddoc_core::resources::limit(1000) {
            return Err(limit());
        }
        for m in masks {
            let b = crate::legacy_model::coordinates(&m["bbox"])?;
            let left = b[0].floor().clamp(0., w as f64) as usize;
            let top = b[1].floor().clamp(0., h as f64) as usize;
            let right = b[2].ceil().clamp(0., w as f64) as usize;
            let bottom = b[3].ceil().clamp(0., h as f64) as usize;
            masking = masking
                .checked_add(right.saturating_sub(left) * bottom.saturating_sub(top))
                .filter(|n| *n <= rapiddoc_core::resources::limit(64_000_000))
                .ok_or_else(limit)?;
        }
        if let Some((_, items)) = groups.iter_mut().find(|(key, _)| *key == target) {
            items.push(i);
        } else {
            groups.push((target, vec![i]));
        }
    }
    let mut jobs = Vec::new();
    let mut images = Vec::new();
    let mut raw_pixels = Vec::new();
    let mut bytes = 0usize;
    for ((height, width), indices) in groups {
        for i in indices {
            let page = Pixels::decode(&pngs[i])?;
            let (w, h) = dimensions[i];
            if page.channels != 3 || page.width != w || page.height != h {
                return Err(invalid());
            }
            let mut data = vec![255; width * height * 3];
            for y in 0..h {
                data[y * width * 3..(y * width + w) * 3]
                    .copy_from_slice(&page.data[y * w * 3..(y + 1) * w * 3]);
            }
            let masks = tasks[i]["masks"].as_array().ok_or_else(invalid)?;
            if masks.len() > rapiddoc_core::resources::limit(1000) {
                return Err(limit());
            }
            for m in masks {
                let b = crate::legacy_model::coordinates(&m["bbox"])?;
                let b = [
                    b[0].floor().clamp(0., w as f64) as usize,
                    b[1].floor().clamp(0., h as f64) as usize,
                    b[2].ceil().clamp(0., w as f64) as usize,
                    b[3].ceil().clamp(0., h as f64) as usize,
                ];
                if b[2] <= b[0] || b[3] <= b[1] {
                    continue;
                }
                for y in b[1]..b[3] {
                    data[(y * width + b[0]) * 3..(y * width + b[2]) * 3].fill(255);
                }
            }
            let source = format!("detect-{index}-{}", images.len() + raw_pixels.len());
            let size = if raw {
                let size = data.len();
                raw_pixels.push((
                    source.clone(),
                    rapiddoc_core::pillow_resize::Rgb {
                        data,
                        w: width,
                        h: height,
                    },
                ));
                size
            } else {
                let mut png = Vec::new();
                PngEncoder::new(&mut png)
                    .write_image(&data, width as u32, height as u32, ExtendedColorType::Rgb8)
                    .map_err(|_| invalid())?;
                let size = png.len();
                images.push((source.clone(), png));
                size
            };
            bytes = bytes
                .checked_add(size)
                .filter(|n| *n <= rapiddoc_core::resources::limit(128_000_000))
                .ok_or_else(limit)?;
            let mut task = tasks[i].clone();
            task["crop_source"] = task["source"].clone();
            task["source"] = json!(source);
            task["target_shape"] = json!([height, width]);
            jobs.push(task);
        }
    }
    let mut plan = plan.clone();
    plan["detect"] = json!(jobs);
    Ok(Output {
        plan,
        images,
        pixels: raw_pixels,
    })
}
