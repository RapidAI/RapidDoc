// metafile-render 0.3.0 emfplus/objects.py bitmap normalization (MIT).
use super::super::{dib_decode::RgbaImage, malformed, output, resource};
use super::cursor::Cursor;
use crate::{Error, Result};
use std::io::Cursor as Io;
pub fn size(w: i64, h: i64) -> Result<[usize; 2]> {
    if w <= 0 || h <= 0 {
        return Err(malformed());
    }
    if w.max(h) > 8192 || w * h > 16_000_000 {
        return Err(resource("EMF+ image exceeds Python pixel budget"));
    }
    Ok([w as usize, h as usize])
}
pub fn png(bytes: &[u8]) -> Result<RgbaImage> {
    if bytes.len() > 128 * 1024 * 1024 {
        return Err(resource("EMF+ normalized image exceeds Python byte budget"));
    }
    png_with(bytes, size)
}
pub(crate) fn png_with(
    bytes: &[u8],
    check: impl Fn(i64, i64) -> Result<[usize; 2]>,
) -> Result<RgbaImage> {
    png_with_mode(bytes, check).map(|(image, _)| image)
}
pub(crate) fn png_with_mode(
    bytes: &[u8],
    check: impl Fn(i64, i64) -> Result<[usize; 2]>,
) -> Result<(RgbaImage, bool)> {
    png_mode(bytes, check, false)
}
/// Pillow validates pre-image chunk CRCs, but load() skips IDAT CRC and can
/// finish at the decompressed frame before a malformed/missing IEND tail.
pub(crate) fn png_pillow_with_mode(
    bytes: &[u8],
    check: impl Fn(i64, i64) -> Result<[usize; 2]>,
) -> Result<(RgbaImage, bool)> {
    match png_mode(bytes, &check, true) {
        Ok(image) => Ok(image),
        Err(e) if e.kind == crate::ErrorKind::ResourceLimit => Err(e),
        Err(original) => {
            let Some(repaired) = pillow_png_tail(bytes, &check)? else {
                return Err(original);
            };
            png_mode(&repaired, check, true)
        }
    }
}
/// Only after the ordinary decoder fails: recover a complete deflate frame
/// whose chunk CRC/IEND or *incomplete* Adler trailer Pillow does not require.
/// A present incorrect Adler checksum still fails. Pixel data is not modified
/// or recompressed, and decompressed data is streamed solely for verification.
fn pillow_png_tail(
    bytes: &[u8],
    check: impl Fn(i64, i64) -> Result<[usize; 2]>,
) -> Result<Option<Vec<u8>>> {
    if bytes.len() < 33 || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Ok(None);
    }
    // Recovery never changes a bad header CRC into a dimension-budget error.
    let mut header = 8usize;
    while header + 8 <= bytes.len() {
        let n = u32::from_be_bytes(bytes[header..header + 4].try_into().unwrap()) as usize;
        if &bytes[header + 4..header + 8] == b"IDAT" {
            break;
        }
        let Some(end) = header
            .checked_add(8)
            .and_then(|p| p.checked_add(n))
            .filter(|p| p.checked_add(4).is_some_and(|p| p <= bytes.len()))
        else {
            return Ok(None);
        };
        let mut crc = flate2::Crc::new();
        crc.update(&bytes[header + 4..end]);
        if crc.sum() != u32::from_be_bytes(bytes[end..end + 4].try_into().unwrap()) {
            return Ok(None);
        }
        header = end + 4;
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    check(width as i64, height as i64)?;
    let max_output = u64::from(width) * u64::from(height) * 8 + u64::from(height) * 7 + 128;
    let (mut offset, mut prefix) = (8usize, None);
    let mut compressed = vec![];
    while offset + 8 <= bytes.len() {
        let n = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        let kind = &bytes[offset + 4..offset + 8];
        if kind == b"IDAT" || prefix.is_some() && kind == b"DDAT" {
            prefix.get_or_insert(offset);
            let end = (offset + 8).saturating_add(n).min(bytes.len());
            compressed.extend_from_slice(&bytes[offset + 8..end]);
        } else if prefix.is_some() {
            break;
        }
        let Some(next) = offset.checked_add(12).and_then(|p| p.checked_add(n)) else {
            return Ok(None);
        };
        offset = next;
    }
    let Some(prefix) = prefix else {
        return Ok(None);
    };
    if compressed.len() < 2 {
        return Ok(None);
    }
    if compressed[0] & 15 != 8
        || compressed[0] >> 4 > 7
        || u16::from_be_bytes([compressed[0], compressed[1]]) % 31 != 0
        || compressed[1] & 32 != 0
    {
        return Ok(None);
    }
    let mut inflater = flate2::Decompress::new(false);
    let mut buffer = [0; 65536];
    let (mut cursor, mut sum_a, mut sum_b, mut total) = (2usize, 1u32, 0u32, 0u64);
    loop {
        let before_in = inflater.total_in();
        let before_out = inflater.total_out();
        let status = match inflater.decompress(
            &compressed[cursor..],
            &mut buffer,
            flate2::FlushDecompress::None,
        ) {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        cursor += (inflater.total_in() - before_in) as usize;
        let written = (inflater.total_out() - before_out) as usize;
        total += written as u64;
        if total > max_output {
            return Ok(None);
        }
        for chunk in buffer[..written].chunks(5552) {
            for byte in chunk {
                sum_a += *byte as u32;
                sum_b += sum_a;
            }
            sum_a %= 65521;
            sum_b %= 65521;
        }
        if status == flate2::Status::StreamEnd {
            break;
        }
        if inflater.total_in() == before_in && written == 0 {
            return Ok(None);
        }
    }
    let checksum = (sum_b << 16) | sum_a;
    if compressed.len() - cursor >= 4
        && u32::from_be_bytes(compressed[cursor..cursor + 4].try_into().unwrap()) != checksum
    {
        return Ok(None);
    }
    let mut out = bytes[..prefix].to_vec();
    let length = u32::try_from(cursor + 4).map_err(Error::malformed)?;
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(b"IDAT");
    out.extend_from_slice(&compressed[..cursor]);
    out.extend_from_slice(&checksum.to_be_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&[0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xae, 0x42, 0x60, 0x82]);
    Ok(Some(out))
}
fn png_mode(
    bytes: &[u8],
    check: impl Fn(i64, i64) -> Result<[usize; 2]>,
    pillow: bool,
) -> Result<(RgbaImage, bool)> {
    let gray16 = bytes.get(24) == Some(&16) && bytes.get(25) == Some(&0);
    let mut options = png::DecodeOptions::default();
    if pillow {
        let mut offset = 8usize;
        while offset + 8 <= bytes.len() {
            let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            let name = &bytes[offset + 4..offset + 8];
            if matches!(name, b"IDAT" | b"fdAT") {
                break;
            }
            let end = offset
                .checked_add(8)
                .and_then(|p| p.checked_add(length))
                .filter(|p| p.checked_add(4).is_some_and(|p| p <= bytes.len()))
                .ok_or_else(malformed)?;
            let mut crc = flate2::Crc::new();
            crc.update(&bytes[offset + 4..end]);
            if crc.sum() != u32::from_be_bytes(bytes[end..end + 4].try_into().unwrap()) {
                return Err(malformed());
            }
            offset = end + 4;
        }
        options.set_ignore_crc(true);
        options.set_ignore_adler32(false);
    }
    let mut decoder = png::Decoder::new_with_options(Io::new(bytes), options);
    decoder.set_limits(png::Limits { bytes: usize::MAX });
    decoder.set_transformations(if gray16 {
        png::Transformations::EXPAND
    } else {
        png::Transformations::EXPAND | png::Transformations::STRIP_16
    });
    let mut reader = decoder.read_info().map_err(Error::malformed)?;
    let original = reader.info();
    let alpha = matches!(
        original.color_type,
        png::ColorType::GrayscaleAlpha | png::ColorType::Rgba
    ) || original.color_type == png::ColorType::Indexed && original.trns.is_some();
    let [w, h] = check(reader.info().width as i64, reader.info().height as i64)?;
    let mut raw = vec![0; reader.output_buffer_size().ok_or_else(malformed)?];
    let info = reader.next_frame(&mut raw).map_err(Error::malformed)?;
    raw.truncate(info.buffer_size());
    let mut rgba = Vec::with_capacity(w * h * 4);
    if gray16 {
        let step = if info.color_type == png::ColorType::GrayscaleAlpha {
            4
        } else {
            2
        };
        for p in raw.chunks_exact(step) {
            let v = u16::from_be_bytes([p[0], p[1]]).min(255) as u8;
            rgba.extend([v, v, v, if step == 4 { p[2] } else { 255 }]);
        }
    } else {
        match info.color_type {
            png::ColorType::Rgba => rgba = raw,
            png::ColorType::Rgb => {
                for p in raw.chunks_exact(3) {
                    rgba.extend([p[0], p[1], p[2], 255]);
                }
            }
            png::ColorType::Grayscale => {
                for v in raw {
                    rgba.extend([v, v, v, 255]);
                }
            }
            png::ColorType::GrayscaleAlpha => {
                for p in raw.chunks_exact(2) {
                    rgba.extend([p[0], p[0], p[0], p[1]]);
                }
            }
            _ => return Err(malformed()),
        }
    }
    Ok((
        RgbaImage {
            width: w,
            height: h,
            pixels: rgba,
        },
        alpha,
    ))
}
pub fn decode(c: &mut Cursor<'_>, warn: &mut impl FnMut(&str, &str)) -> Result<Option<RgbaImage>> {
    if c.u32()? != 1 {
        warn(
            "emfplus_unsupported_image",
            "nested metafile images are not supported",
        );
        return Ok(None);
    }
    let (w, h, stride) = (c.i32()?, c.i32()?, c.i32()?);
    let format = c.u32()?;
    let encoding = c.u32()?;
    let payload = c.take(c.remaining()?)?;
    if payload.len() > 128 * 1024 * 1024 {
        return Err(resource("EMF+ image payload exceeds Python byte budget"));
    }
    let image = match encoding {
        1 => {
            if payload.starts_with(b"\x89PNG\r\n\x1a\n") {
                png(payload)?
            } else if payload.starts_with(b"\xff\xd8") {
                jpeg_with(payload, size)?
            } else {
                let format = if payload.starts_with(b"GIF8") {
                    Some("GIF")
                } else if payload.starts_with(b"BM") {
                    Some("BMP")
                } else if payload.starts_with(b"II*\0") || payload.starts_with(b"MM\0*") {
                    Some("TIFF")
                } else if payload.starts_with(b"RIFF") && payload.get(8..12) == Some(b"WEBP") {
                    Some("WEBP")
                } else {
                    None
                };
                if let Some(format) = format {
                    warn(
                        "emfplus_unsupported_image",
                        &format!("compressed image format {format} skipped"),
                    );
                    return Ok(None);
                }
                return Err(malformed());
            }
        }
        0 => {
            let [w, h] = size(w as i64, h as i64)?;
            let bpp = (format >> 8) & 255;
            if format & 0x10000 != 0 || !matches!(bpp, 24 | 32) {
                warn(
                    "emfplus_unsupported_image",
                    &format!("raw pixel format {format:#x} skipped"),
                );
                return Ok(None);
            }
            let pitch = i64::from(stride).unsigned_abs() as usize;
            let row = w * (bpp as usize / 8);
            if pitch < row || pitch * h > payload.len() {
                return Err(malformed());
            }
            if pitch * h > 128 * 1024 * 1024 {
                return Err(resource("EMF+ raw bitmap exceeds Python byte budget"));
            }
            let mut rgba = Vec::with_capacity(w * h * 4);
            for y in 0..h {
                let sy = if stride > 0 { y } else { h - 1 - y };
                for x in 0..w {
                    let p = &payload[sy * pitch + x * (bpp as usize / 8)..];
                    let alpha = if bpp == 32 && format & 0x40000 != 0 {
                        p[3]
                    } else {
                        255
                    };
                    let mut pixel = [p[2], p[1], p[0], alpha];
                    if bpp == 32 && format & 0xc0000 == 0xc0000 && alpha != 0 && alpha != 255 {
                        for v in &mut pixel[..3] {
                            *v = (*v as u32 * 255 / alpha as u32).min(255) as u8;
                        }
                    }
                    rgba.extend(pixel);
                }
            }
            RgbaImage::new(w, h, rgba)?
        }
        _ => return Err(malformed()),
    };
    Ok(Some(image))
}
pub fn normalize(image: &RgbaImage) -> Result<Vec<u8>> {
    let bytes = output::png(image, None)?;
    if bytes.len() > 128 * 1024 * 1024 {
        Err(resource(
            "EMF+ normalized bitmap exceeds Python byte budget",
        ))
    } else {
        Ok(bytes)
    }
}

pub(crate) fn jpeg_with(
    payload: &[u8],
    check: impl Fn(i64, i64) -> Result<[usize; 2]>,
) -> Result<RgbaImage> {
    let mut decoder = turbojpeg::Decompressor::new().map_err(Error::malformed)?;
    let info = decoder.read_header(payload).map_err(Error::malformed)?;
    let [w, h] = check(info.width as i64, info.height as i64)?;
    let cmyk = matches!(
        info.colorspace,
        turbojpeg::Colorspace::CMYK | turbojpeg::Colorspace::YCCK
    );
    let mut rgba = vec![0; w * h * 4];
    decoder
        .decompress(
            payload,
            turbojpeg::Image {
                pixels: rgba.as_mut_slice(),
                width: w,
                height: h,
                pitch: w * 4,
                format: if cmyk {
                    turbojpeg::PixelFormat::CMYK
                } else {
                    turbojpeg::PixelFormat::RGBA
                },
            },
        )
        .map_err(Error::malformed)?;
    // TurboJPEG's CMYK output has inverted components, like Pillow's
    // JPEG raw decoder. Pillow then uses Convert.c cmyk2rgb.
    if cmyk {
        for p in rgba.chunks_exact_mut(4) {
            let k = p[3] as u32;
            let mul = |v: u8| {
                let t = v as u32 * k + 128;
                ((t + (t >> 8)) >> 8) as u8
            };
            p.copy_from_slice(&[mul(p[0]), mul(p[1]), mul(p[2]), 255]);
        }
    }
    Ok(RgbaImage {
        width: w,
        height: h,
        pixels: rgba,
    })
}
