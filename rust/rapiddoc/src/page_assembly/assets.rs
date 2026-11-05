//! 旧 cut_image_and_table/cut_image：保留 PNG 像素、裁图补零与哈希命名，I/O 由宿主执行。
use super::*;
use image::{
    codecs::png::PngEncoder, DynamicImage, ExtendedColorType, ImageDecoder, ImageEncoder,
    ImageFormat, ImageReader,
};
use md5::Md5;
use sha2::{Digest, Sha256};
use std::io::Cursor;

pub struct OriginalImage {
    pub bbox: [f64; 4],
    pub png: Vec<u8>,
}
pub struct TableFillImage {
    pub uuid: String,
    pub bbox: [f64; 4],
    pub png: Vec<u8>,
}
pub struct Asset {
    pub name: String,
    pub png: Vec<u8>,
}
pub struct Output {
    pub page: Value,
    pub inline_count: usize,
    pub ocr_images: Vec<Asset>,
    /// 保留旧 writer 的调用顺序；同名写入也不合并。
    pub assets: Vec<Asset>,
}
pub struct NativeOutput {
    pub spans: Value,
    pub images: Vec<Asset>,
}
pub(crate) struct NativeCrops<'a> {
    page: &'a Pixels,
    scale: f64,
    prefix: String,
    images: Vec<Asset>,
    pixels: usize,
    bytes: usize,
}
impl<'a> NativeCrops<'a> {
    pub(crate) fn new(page: &'a Pixels, request: &Value) -> Result<Self, Error> {
        if page.channels != 3 {
            return Err(unsupported("原生字符补识别页图仅支持 8-bit RGB PNG"));
        }
        let scale = request["scale"]
            .as_f64()
            .filter(|s| s.is_finite() && *s > 0. && *s <= 1000.)
            .ok_or_else(invalid)?;
        let index = request["page_index"].as_u64().ok_or_else(invalid)?;
        Ok(Self {
            page,
            scale,
            prefix: format!("ocr-{index}-"),
            images: Vec::new(),
            pixels: 0,
            bytes: 0,
        })
    }
    pub(crate) fn crop(&mut self, bbox: &Value) -> Result<Option<String>, Error> {
        let b = bb(bbox).map(|n| (n * self.scale) as i64);
        let bound = |n: i64, size: usize| {
            if n < 0 {
                (n + size as i64).max(0)
            } else {
                n.min(size as i64)
            }
        };
        let b = [
            bound(b[0], self.page.width),
            bound(b[1], self.page.height),
            bound(b[2], self.page.width),
            bound(b[3], self.page.height),
        ];
        if b[2] <= b[0] || b[3] <= b[1] {
            return Ok(None);
        }
        let count = ((b[2] - b[0]) as usize)
            .checked_mul((b[3] - b[1]) as usize)
            .filter(|n| *n <= rapiddoc_core::resources::limit(4_000_000))
            .ok_or_else(limit)?;
        self.pixels = self
            .pixels
            .checked_add(count)
            .filter(|n| *n <= rapiddoc_core::resources::limit(32_000_000))
            .ok_or_else(limit)?;
        let mut crop = self.page.crop(b)?;
        if super::native_text::contrast(&crop.data) < 0.17 {
            return Ok(None);
        }
        if self.images.len() >= rapiddoc_core::resources::limit(1000) {
            return Err(limit());
        }
        // np.asarray(PIL RGB) 不做色彩变换/透明色合成；传给 OCR 的 PNG 只保留原 RGB 值。
        crop.icc = None;
        crop.transparency = None;
        let png = crop.png()?;
        self.bytes = self
            .bytes
            .checked_add(png.len())
            .filter(|n| *n <= rapiddoc_core::resources::limit(128_000_000))
            .ok_or_else(limit)?;
        let name = format!("{}{}", self.prefix, self.images.len());
        self.images.push(Asset {
            name: name.clone(),
            png,
        });
        Ok(Some(name))
    }
}

/// 独立旧字符填充 stage；blocks/discarded_blocks 是 prepare_block_bboxes 的旧数组。
pub fn fill_native_text(request: &Value, page_png: &[u8]) -> Result<NativeOutput, Error> {
    super::validate_tree(request)?;
    let page = Pixels::decode(page_png)?;
    let mut crops = NativeCrops::new(&page, request)?;
    let source = request["spans"].as_array().ok_or_else(invalid)?;
    if source.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut spans = source.clone();
    let blocks = request["blocks"].as_array().ok_or_else(invalid)?;
    let discarded = request["discarded_blocks"].as_array().ok_or_else(invalid)?;
    super::native_text::fill(
        &request["native_page"],
        &mut spans,
        blocks,
        discarded,
        &mut Work(0),
        |bbox| crops.crop(bbox),
    )?;
    Ok(NativeOutput {
        spans: Value::Array(spans),
        images: crops.images,
    })
}
pub(crate) struct Pixels {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) channels: usize,
    pub(crate) data: Vec<u8>,
    icc: Option<Vec<u8>>,
    transparency: Option<Vec<u8>>,
}
impl Pixels {
    pub(crate) fn decode(png: &[u8]) -> Result<Self, Error> {
        if png.len() > rapiddoc_core::resources::limit(64_000_000) {
            return Err(limit());
        }
        if png.len() < 33 || !png.starts_with(b"\x89PNG\r\n\x1a\n") {
            return Err(invalid());
        }
        // 不把 P/16-bit PNG 隐式转换为另一 PIL 模式，否则 raw pixels MD5 会改变。
        if png[24] != 8 || !matches!(png[25], 0 | 2 | 4 | 6) {
            return Err(unsupported("图片资产只支持 8-bit L/LA/RGB/RGBA PNG"));
        }
        let (width, height) = ImageReader::with_format(Cursor::new(png), ImageFormat::Png)
            .into_dimensions()
            .map_err(|_| invalid())?;
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .filter(|n| *n > 0 && *n <= rapiddoc_core::resources::limit(16_000_000))
            .ok_or_else(limit)?;
        let mut transparency = None;
        let mut offset = 8usize;
        while offset + 12 <= png.len() {
            let length = u32::from_be_bytes(png[offset..offset + 4].try_into().unwrap()) as usize;
            let end = offset
                .checked_add(length)
                .and_then(|n| n.checked_add(12))
                .filter(|n| *n <= png.len())
                .ok_or_else(invalid)?;
            if &png[offset + 4..offset + 8] == b"tRNS" {
                if !((png[25] == 0 && length == 2) || (png[25] == 2 && length == 6)) {
                    return Err(invalid());
                }
                transparency = Some(png[offset..end].to_vec());
            }
            offset = end;
        }
        let mut decoder = ImageReader::with_format(Cursor::new(png), ImageFormat::Png)
            .into_decoder()
            .map_err(|_| invalid())?;
        let icc = decoder.icc_profile().map_err(|_| invalid())?;
        if icc
            .as_ref()
            .is_some_and(|v| v.len() > rapiddoc_core::resources::limit(1_000_000))
        {
            return Err(limit());
        }
        let image = DynamicImage::from_decoder(decoder).map_err(|_| invalid())?;
        let channels = match png[25] {
            0 => 1,
            2 => 3,
            4 => 2,
            6 => 4,
            _ => unreachable!(),
        };
        let mut data = image.into_bytes();
        // PNG 解码器会把 RGB/L 的透明色展开成 alpha；Pillow 保留原模式及 tRNS。
        if transparency.is_some() && data.len() == pixels * (channels + 1) {
            data = data
                .chunks_exact(channels + 1)
                .flat_map(|p| p[..channels].iter().copied())
                .collect();
        }
        if data.len() != pixels * channels {
            return Err(invalid());
        }
        Ok(Self {
            width: width as usize,
            height: height as usize,
            channels,
            data,
            icc,
            transparency,
        })
    }
    pub(crate) fn crop(&self, b: [i64; 4]) -> Result<Self, Error> {
        let width = b[2]
            .checked_sub(b[0])
            .filter(|n| *n > 0)
            .ok_or_else(invalid)? as usize;
        let height = b[3]
            .checked_sub(b[1])
            .filter(|n| *n > 0)
            .ok_or_else(invalid)? as usize;
        let length = width
            .checked_mul(height)
            .filter(|n| *n <= rapiddoc_core::resources::limit(4_000_000))
            .ok_or_else(limit)?;
        let mut data = vec![0; length * self.channels];
        let x0 = b[0].max(0).min(self.width as i64);
        let x1 = b[2].max(0).min(self.width as i64);
        let y0 = b[1].max(0).min(self.height as i64);
        let y1 = b[3].max(0).min(self.height as i64);
        if x1 > x0 && y1 > y0 {
            let row_len = (x1 - x0) as usize * self.channels;
            for y in y0..y1 {
                let src = (y as usize * self.width + x0 as usize) * self.channels;
                let dst = ((y - b[1]) as usize * width + (x0 - b[0]) as usize) * self.channels;
                data[dst..dst + row_len].copy_from_slice(&self.data[src..src + row_len]);
            }
        }
        Ok(Self {
            width,
            height,
            channels: self.channels,
            data,
            icc: self.icc.clone(),
            transparency: self.transparency.clone(),
        })
    }
    pub(crate) fn rotate_ccw(self, angle: i64) -> Result<Self, Error> {
        if !matches!(angle, 90 | 180 | 270) {
            return Ok(self);
        }
        let (width, height) = if angle == 180 {
            (self.width, self.height)
        } else {
            (self.height, self.width)
        };
        let mut data = vec![0; self.data.len()];
        for y in 0..self.height {
            for x in 0..self.width {
                let (dx, dy) = match angle {
                    90 => (y, self.width - 1 - x),
                    180 => (self.width - 1 - x, self.height - 1 - y),
                    _ => (self.height - 1 - y, x),
                };
                let from = (y * self.width + x) * self.channels;
                let to = (dy * width + dx) * self.channels;
                data[to..to + self.channels]
                    .copy_from_slice(&self.data[from..from + self.channels]);
            }
        }
        Ok(Self {
            width,
            height,
            channels: self.channels,
            data,
            icc: self.icc,
            transparency: self.transparency,
        })
    }
    pub(crate) fn png(&self) -> Result<Vec<u8>, Error> {
        let color = match self.channels {
            1 => ExtendedColorType::L8,
            2 => ExtendedColorType::La8,
            3 => ExtendedColorType::Rgb8,
            4 => ExtendedColorType::Rgba8,
            _ => return Err(invalid()),
        };
        let mut png = Vec::new();
        let mut encoder = PngEncoder::new(&mut png);
        if let Some(icc) = &self.icc {
            encoder
                .set_icc_profile(icc.clone())
                .map_err(|_| invalid())?;
        }
        // 旧 Pillow PNG save 未显式传 exif 时不从 image.info 继承 EXIF。
        encoder
            .write_image(&self.data, self.width as u32, self.height as u32, color)
            .map_err(|_| invalid())?;
        if let Some(chunk) = &self.transparency {
            png.splice(33..33, chunk.iter().copied());
        }
        Ok(png)
    }
}

/// 输入页 PNG 必须与旧 PIL 图像的模式/像素一致；不合成透明背景。
/// 原图选择沿用旧逻辑：仅 image、IoU>=阈值、最后一个匹配项获胜。
pub fn assemble(
    request: &Value,
    page_png: &[u8],
    originals: &[OriginalImage],
    extract_original: bool,
    threshold: f64,
) -> Result<Output, Error> {
    assemble_with_inline(
        request,
        page_png,
        originals,
        extract_original,
        threshold,
        "images/",
    )
}

pub fn assemble_with_inline(
    request: &Value,
    page_png: &[u8],
    originals: &[OriginalImage],
    extract_original: bool,
    threshold: f64,
    prefix: &str,
) -> Result<Output, Error> {
    assemble_with_table_images(
        request,
        page_png,
        originals,
        extract_original,
        threshold,
        prefix,
        &[],
    )
}

pub fn assemble_with_table_images(
    request: &Value,
    page_png: &[u8],
    originals: &[OriginalImage],
    extract_original: bool,
    threshold: f64,
    prefix: &str,
    fills: &[TableFillImage],
) -> Result<Output, Error> {
    super::validate_tree(request)?;
    if fills.len() > rapiddoc_core::resources::limit(256)
        || prefix.len() > rapiddoc_core::resources::limit(4096)
    {
        return Err(limit());
    }
    let mut fill_bytes = 0usize;
    for fill in fills {
        if fill.uuid.is_empty() || fill.uuid.len() > rapiddoc_core::resources::limit(256) {
            return Err(invalid());
        }
        coord(&json!(fill.bbox), false)?;
        fill_bytes = fill_bytes
            .checked_add(fill.png.len())
            .filter(|n| *n <= rapiddoc_core::resources::limit(128_000_000))
            .ok_or_else(limit)?;
        if fill.png.len() > rapiddoc_core::resources::limit(64_000_000) {
            return Err(limit());
        }
    }
    if !threshold.is_finite() || !(0. ..=1.).contains(&threshold) {
        return Err(invalid());
    }
    if originals.len() > rapiddoc_core::resources::limit(256) {
        return Err(limit());
    }
    let page = Pixels::decode(page_png)?;
    // The full-page hash is only an asset-name seed. Text-only pages still
    // undergo the same PNG/native validation, but need no pixel checksum.
    let page_md5 = std::cell::OnceCell::<String>::new();
    let mut native = if request.get("native_page").is_some() {
        Some(NativeCrops::new(&page, request)?)
    } else {
        None
    };
    let mut original_pixels = Vec::new();
    let mut original_total = 0usize;
    let mut original_bytes = 0usize;
    let mut metadata_bytes = page.icc.as_ref().map_or(0, Vec::len);
    if extract_original {
        for original in originals {
            coord(&json!(original.bbox), true)?;
            original_bytes = original_bytes
                .checked_add(original.png.len())
                .filter(|n| *n <= rapiddoc_core::resources::limit(128_000_000))
                .ok_or_else(limit)?;
            let pixels = Pixels::decode(&original.png)?;
            metadata_bytes = metadata_bytes
                .checked_add(pixels.icc.as_ref().map_or(0, Vec::len))
                .filter(|n| *n <= rapiddoc_core::resources::limit(8_000_000))
                .ok_or_else(limit)?;
            original_total = original_total
                .checked_add(pixels.width * pixels.height)
                .filter(|n| *n <= rapiddoc_core::resources::limit(64_000_000))
                .ok_or_else(limit)?;
            original_pixels.push(pixels);
        }
    }
    let mut request = request.clone();
    let mut assets = Vec::new();
    let dets = request
        .get_mut("model")
        .and_then(|m| m.get_mut("layout_dets"))
        .and_then(Value::as_array_mut)
        .ok_or_else(invalid)?;
    if dets.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut inline_bytes = 0usize;
    for det in dets {
        if let Some(markup) = det["html"].as_str().filter(|s| s.contains("data:image/")) {
            let (html, images) = super::inline::replace(markup, prefix)?;
            det["html"] = json!(html);
            for image in images {
                inline_bytes += image.png.len();
                if inline_bytes > rapiddoc_core::resources::limit(2_000_000)
                    || assets.len() >= rapiddoc_core::resources::limit(1000)
                {
                    return Err(limit());
                }
                assets.push(image);
            }
        }
    }
    let inline_count = assets.len();
    let mut total_pixels = 0usize;
    let mut total_png = inline_bytes;
    let mut fill_pixels: Vec<Option<Pixels>> = (0..fills.len()).map(|_| None).collect();
    let mut fill_decoded = 0usize;
    let mut fill_work = Work(0);
    let mut writer = |span: &mut Value, stage: AssetStage| -> Result<(), Error> {
        if matches!(stage, AssetStage::TableFill) {
            if span["category_id"] != 5 {
                return Ok(());
            }
            let Some(html) = span["html"].as_str().filter(|s| !s.is_empty()) else {
                return Ok(());
            };
            let mut html = html.to_owned();
            for (i, fill) in fills.iter().enumerate() {
                fill_work.charge(html.len())?;
                if !html.contains(&fill.uuid) {
                    continue;
                }
                if assets.len() >= rapiddoc_core::resources::limit(1000) {
                    return Err(limit());
                }
                if fill_pixels[i].is_none() {
                    let decoded = Pixels::decode(&fill.png)?;
                    fill_decoded += decoded.width * decoded.height;
                    metadata_bytes += decoded.icc.as_ref().map_or(0, Vec::len);
                    if fill_decoded > rapiddoc_core::resources::limit(64_000_000)
                        || metadata_bytes > rapiddoc_core::resources::limit(8_000_000)
                    {
                        return Err(limit());
                    }
                    fill_pixels[i] = Some(decoded);
                }
                let pixels = fill_pixels[i].as_ref().unwrap();
                let count = pixels.width * pixels.height;
                total_pixels += count;
                if count > rapiddoc_core::resources::limit(4_000_000)
                    || total_pixels > rapiddoc_core::resources::limit(32_000_000)
                {
                    return Err(limit());
                }
                let b = fill.bbox.map(|n| n as i64);
                let index = request["page_index"].as_u64().ok_or_else(invalid)?;
                // 有意修复旧函数对象 repr（含进程地址）参与哈希的不稳定命名。
                let page_md5 = page_md5.get_or_init(|| format!("{:X}", Md5::digest(&page.data)));
                let seed = format!(
                    "table_fill/{page_md5}_{index}_{}_{}_{}_{}",
                    b[0], b[1], b[2], b[3]
                );
                let name = format!("{:x}.png", Sha256::digest(seed.as_bytes()));
                let png = pixels.png()?;
                total_png += png.len();
                if total_png > rapiddoc_core::resources::limit(128_000_000) {
                    return Err(limit());
                }
                let fill_prefix = if prefix.is_empty() { "/" } else { prefix };
                let replacement = format!("<img src=\"{fill_prefix}{name}\" alt=\"Image\" />");
                let occurrences = html.matches(&fill.uuid).count();
                let size = html
                    .len()
                    .checked_sub(occurrences * fill.uuid.len())
                    .and_then(|n| n.checked_add(occurrences.checked_mul(replacement.len())?))
                    .ok_or_else(limit)?;
                if size > rapiddoc_core::resources::limit(2_000_000) {
                    return Err(limit());
                }
                html = html.replace(&fill.uuid, &replacement);
                assets.push(Asset { name, png });
            }
            span["html"] = json!(html);
            return Ok(());
        }
        if assets.len() >= rapiddoc_core::resources::limit(1000) {
            return Err(limit());
        }
        coord(&span["bbox"], true)?;
        let bbox = bb(&span["bbox"]);
        let scale = num(&request["scale"]);
        let b = bbox.map(|n| (n * scale) as i64);
        let selected = if extract_original && span["type"] == "image" {
            originals
                .iter()
                .enumerate()
                .filter(|(_, o)| iou(&span["bbox"], &json!(o.bbox)) >= threshold)
                .map(|(i, _)| i)
                .next_back()
        } else {
            None
        };
        let cropped;
        let pixels = if let Some(i) = selected {
            &original_pixels[i]
        } else {
            cropped = page.crop(b)?;
            &cropped
        };
        let count = pixels.width * pixels.height;
        // 原图提取也受单张/累计像素约束，不按裁图小框漏计。
        if count > rapiddoc_core::resources::limit(4_000_000) {
            return Err(limit());
        }
        total_pixels = total_pixels
            .checked_add(count)
            .filter(|n| *n <= rapiddoc_core::resources::limit(32_000_000))
            .ok_or_else(limit)?;
        let kind = span["type"].as_str().ok_or_else(invalid)?;
        let index = request["page_index"].as_u64().ok_or_else(invalid)?;
        let name = format!(
            "{kind}/{}_{index}_{}_{}_{}_{}",
            page_md5.get_or_init(|| format!("{:X}", Md5::digest(&page.data))),
            bbox[0] as i64,
            bbox[1] as i64,
            bbox[2] as i64,
            bbox[3] as i64
        );
        let name = format!("{:x}.png", Sha256::digest(name.as_bytes()));
        let png = pixels.png()?;
        total_png = total_png
            .checked_add(png.len())
            .filter(|n| *n <= rapiddoc_core::resources::limit(128_000_000))
            .ok_or_else(limit)?;
        span["image_path"] = json!(name);
        assets.push(Asset { name, png });
        Ok(())
    };
    let mut native_fill =
        |spans: &mut Vec<Value>, blocks: &[Value], discarded: &[Value], work: &mut Work| {
            let crops = native.as_mut().ok_or_else(invalid)?;
            super::native_text::fill(
                &request["native_page"],
                spans,
                blocks,
                discarded,
                work,
                |bbox| crops.crop(bbox),
            )
        };
    let use_native = request.get("native_page").is_some();
    let filler = if use_native {
        Some(
            &mut native_fill
                as &mut dyn FnMut(
                    &mut Vec<Value>,
                    &[Value],
                    &[Value],
                    &mut Work,
                ) -> Result<(), Error>,
        )
    } else {
        None
    };
    let page = super::assemble_with_native(&request, &mut Work(0), Some(&mut writer), filler)?;
    Ok(Output {
        page,
        assets,
        inline_count,
        ocr_images: native.map_or_else(Vec::new, |c| c.images),
    })
}
