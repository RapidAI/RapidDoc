//! Office body raster encoding. Keep Pillow's mode-dependent PNG/JPEG choice;
//! the metafile's smaller canvas limits must not leak into ordinary pictures.
use crate::{
    metafile::{dib_decode::RgbaImage, emfplus::image as codecs, output},
    Error, ErrorKind, Result,
};
use std::io::Cursor;
pub(crate) fn dimensions(w: i64, h: i64) -> Result<[usize; 2]> {
    if w <= 0 || h <= 0 {
        return Err(Error::malformed("invalid raster dimensions"));
    }
    // Pillow Image.MAX_IMAGE_PIXELS=89478485, error only above twice this.
    if w as u128 * h as u128 > 178_956_970 {
        return Err(Error::new(
            ErrorKind::ResourceLimit,
            "Pillow decompression bomb pixel threshold exceeded",
        ));
    }
    Ok([w as usize, h as usize])
}
fn error(e: image_codecs::ImageError) -> Error {
    if matches!(e, image_codecs::ImageError::Unsupported(_)) {
        Error::new(ErrorKind::Unsupported, e.to_string())
    } else {
        Error::malformed(e)
    }
}
pub fn serialize(bytes: &[u8]) -> Result<Option<String>> {
    let decoded = decode(bytes);
    let (image, alpha) = match decoded {
        Ok(v) => v,
        Err(e) if e.kind == ErrorKind::Malformed => return Ok(None),
        Err(e) => return Err(e),
    };
    let (mime, data) = if alpha {
        ("image/png", output::png(&image, None)?)
    } else {
        ("image/jpeg", super::jpeg(&image)?)
    };
    Ok(Some(format!("data:{mime};base64,{}", base64::encode(data))))
}
fn decode(bytes: &[u8]) -> Result<(RgbaImage, bool)> {
    if bytes.starts_with(b"8BPS")
        || bytes.starts_with(b"\0\0\0\x0cjP  \r\n\x87\n")
        || bytes.starts_with(b"\xff\x4f\xff\x51")
        || bytes.starts_with(b"icns")
        || bytes.starts_with(b"\x01\xda")
        || bytes.len() >= 128 && bytes[0] == 10 && [0, 2, 3, 4, 5].contains(&bytes[1])
    {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "recognized Office raster format still requires a compatible decoder",
        ));
    }
    if bytes.starts_with(&[0, 0, 1, 0]) && bytes.len() >= 6 {
        let count = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
        let table = bytes
            .get(6..6 + 16 * count)
            .ok_or_else(|| Error::malformed("truncated ICO directory"))?;
        let entry = table.chunks_exact(16).max_by_key(|e| {
            let w = if e[0] == 0 { 256 } else { e[0] as usize };
            let h = if e[1] == 0 { 256 } else { e[1] as usize };
            (w * h, u16::from_le_bytes([e[6], e[7]]))
        });
        if let Some(e) = entry {
            let size = u32::from_le_bytes(e[8..12].try_into().unwrap()) as usize;
            let offset = u32::from_le_bytes(e[12..16].try_into().unwrap()) as usize;
            let data = bytes
                .get(offset..offset + size)
                .ok_or_else(|| Error::malformed("ICO entry outside payload"))?;
            if data.starts_with(b"\x89PNG\r\n\x1a\n") {
                let (image, _) = decode(data)?;
                return Ok((image, matches!(data.get(25), Some(4 | 6))));
            }
        }
    }
    if bytes.starts_with(b"Pf") {
        return pfm(bytes).map(|image| (image, false));
    }
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return codecs::png_pillow_with_mode(bytes, dimensions);
    }
    if bytes.starts_with(b"\xff\xd8") {
        return codecs::jpeg_with(bytes, dimensions).map(|i| (i, false));
    }
    if bytes.starts_with(b"II*\0") || bytes.starts_with(b"MM\0*") {
        return super::tiff::decode(bytes);
    }
    if bytes.len() >= 18 && bytes[1] == 0 && bytes[2] == 3 && bytes[16] == 1 {
        let w = u16::from_le_bytes([bytes[12], bytes[13]]) as usize;
        let h = u16::from_le_bytes([bytes[14], bytes[15]]) as usize;
        dimensions(w as i64, h as i64)?;
        let start = 18 + bytes[0] as usize;
        let stride = w.div_ceil(8);
        let data = bytes
            .get(start..start + stride * h)
            .ok_or_else(|| Error::malformed("truncated monochrome TGA"))?;
        let mut pixels = vec![0; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let sy = if bytes[17] & 32 != 0 { y } else { h - 1 - y };
                let sx = if bytes[17] & 16 != 0 { w - 1 - x } else { x };
                let v = if data[sy * stride + sx / 8] & (128 >> (sx % 8)) != 0 {
                    255
                } else {
                    0
                };
                pixels[(y * w + x) * 4..][..4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        return Ok((
            RgbaImage {
                width: w,
                height: h,
                pixels,
            },
            false,
        ));
    }
    let mut reader = image_codecs::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(Error::malformed)?;
    if reader.format().is_none()
        && bytes.len() >= 18
        && bytes[1] < 2
        && [1, 2, 3, 9, 10, 11].contains(&bytes[2])
        && [1, 8, 16, 24, 32].contains(&bytes[16])
    {
        reader.set_format(image_codecs::ImageFormat::Tga);
    }
    let format = reader
        .format()
        .ok_or_else(|| Error::malformed("unidentified raster image"))?;
    let mut limits = image_codecs::Limits::no_limits();
    limits.max_alloc = None;
    reader.limits(limits);
    let decoder = reader.into_decoder().map_err(error)?;
    use image_codecs::ImageDecoder;
    let (w, h) = decoder.dimensions();
    dimensions(w as i64, h as i64)?;
    let image = image_codecs::DynamicImage::from_decoder(decoder).map_err(error)?;
    let mut alpha = image.color().has_alpha();
    if format == image_codecs::ImageFormat::Gif {
        // Pillow keeps first-frame P mode. A GIF frame is PNG only when the
        // first frame's GCE declares transparency, even if all pixels are opaque.
        alpha = gif_transparency(bytes)?;
    }
    let [w, h] = [w as usize, h as usize];
    let rgba = match image {
        image_codecs::DynamicImage::ImageLuma16(gray) => gray
            .pixels()
            .flat_map(|p| {
                let v = p[0].min(255) as u8;
                [v, v, v, 255]
            })
            .collect(),
        image => image.to_rgba8().into_raw(),
    };
    Ok((
        RgbaImage {
            width: w,
            height: h,
            pixels: rgba,
        },
        alpha,
    ))
}
fn pfm(bytes: &[u8]) -> Result<RgbaImage> {
    fn token<'a>(data: &'a [u8], offset: &mut usize) -> Result<&'a str> {
        loop {
            while data.get(*offset).is_some_and(u8::is_ascii_whitespace) {
                *offset += 1;
            }
            if data.get(*offset) == Some(&b'#') {
                while data.get(*offset).is_some_and(|b| *b != b'\n') {
                    *offset += 1;
                }
            } else {
                break;
            }
        }
        let start = *offset;
        while data.get(*offset).is_some_and(|b| !b.is_ascii_whitespace()) {
            *offset += 1;
        }
        if start == *offset {
            return Err(Error::malformed("truncated PFM header"));
        }
        std::str::from_utf8(&data[start..*offset]).map_err(Error::malformed)
    }
    let mut offset = 0;
    token(bytes, &mut offset)?;
    let w = token(bytes, &mut offset)?
        .parse::<i64>()
        .map_err(Error::malformed)?;
    let h = token(bytes, &mut offset)?
        .parse::<i64>()
        .map_err(Error::malformed)?;
    let scale = token(bytes, &mut offset)?
        .parse::<f64>()
        .map_err(Error::malformed)?;
    if !scale.is_finite() || scale == 0. {
        return Err(Error::malformed("invalid PFM scale"));
    }
    let [w, h] = dimensions(w, h)?;
    offset += 1;
    let data = bytes
        .get(offset..offset + w * h * 4)
        .ok_or_else(|| Error::malformed("truncated PFM pixels"))?;
    let mut pixels = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let at = ((h - 1 - y) * w + x) * 4;
            let p = data[at..at + 4].try_into().unwrap();
            let v = if scale < 0. {
                f32::from_le_bytes(p)
            } else {
                f32::from_be_bytes(p)
            }
            .clamp(0., 255.) as u8;
            pixels.extend([v, v, v, 255]);
        }
    }
    Ok(RgbaImage {
        width: w,
        height: h,
        pixels,
    })
}
fn gif_transparency(data: &[u8]) -> Result<bool> {
    if data.len() < 13 {
        return Err(Error::malformed("truncated GIF"));
    }
    let mut offset = 13
        + if data[10] & 128 != 0 {
            3usize << (usize::from(data[10] & 7) + 1)
        } else {
            0
        };
    let mut transparent = false;
    loop {
        let marker = *data
            .get(offset)
            .ok_or_else(|| Error::malformed("truncated GIF block"))?;
        offset += 1;
        match marker {
            0x2c => return Ok(transparent),
            0x3b => return Err(Error::malformed("GIF has no frame")),
            0x21 => {
                let kind = *data
                    .get(offset)
                    .ok_or_else(|| Error::malformed("truncated GIF extension"))?;
                offset += 1;
                if kind == 0xf9 {
                    transparent = data.get(offset + 1).is_some_and(|b| b & 1 != 0);
                }
                loop {
                    let n = *data
                        .get(offset)
                        .ok_or_else(|| Error::malformed("truncated GIF extension"))?
                        as usize;
                    offset += 1;
                    if n == 0 {
                        break;
                    }
                    offset = offset
                        .checked_add(n)
                        .filter(|o| *o <= data.len())
                        .ok_or_else(|| Error::malformed("truncated GIF extension"))?;
                }
            }
            _ => return Err(Error::malformed("invalid GIF block")),
        }
    }
}
