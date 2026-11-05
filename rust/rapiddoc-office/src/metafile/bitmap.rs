// Copyright (c) 2026 Xiaomeng Zhao (myhloli). SPDX-License-Identifier: MIT
//! Original GDI bitmap records and DIB validation. Raster placement is separate.
use super::{binary::Reader, malformed, model::*, playback::Playback, resource};
use crate::Result;
use std::sync::Arc;
const MAX_BYTES: usize = 128 * 1024 * 1024;
const SRCCOPY: u32 = 0x00cc0020;

pub fn dimensions(header: &[u8]) -> Result<[usize; 2]> {
    let r = Reader::new(header);
    let size = r.u32(0)? as usize;
    let (w, h) = if size == 12 {
        r.bytes(0, 12)?;
        (i64::from(r.u16(4)?), i64::from(r.u16(6)?))
    } else if size >= 40 {
        r.bytes(0, size)?;
        (i64::from(r.i32(4)?), i64::from(r.i32(8)?).abs())
    } else {
        return Err(malformed());
    };
    if w <= 0 || h <= 0 || w * h > 16_000_000 {
        return Err(resource("DIB dimensions exceed Python pixel budget"));
    }
    Ok([w as usize, h as usize])
}
pub fn validate(image: &DibPayload) -> Result<usize> {
    let size = image.header.len() + image.bits.len();
    if size > MAX_BYTES {
        return Err(resource("DIB exceeds Python embedded bitmap byte budget"));
    }
    let [w, h] = dimensions(&image.header)?;
    let r = Reader::new(&image.header);
    let core = r.u32(0)? == 12;
    let bpp = r.u16(if core { 10 } else { 14 })? as usize;
    let compression = if core { 0 } else { r.u32(16)? };
    if matches!(compression, 0 | 3 | 6) {
        if !matches!(bpp, 1 | 4 | 8 | 16 | 24 | 32)
            || (w * bpp).div_ceil(32) * 4 * h > image.bits.len()
        {
            return Err(malformed());
        }
    }
    Ok(w * h * 4 + size)
}
pub fn bits_offset(dib: &[u8]) -> Result<usize> {
    let r = Reader::new(dib);
    let size = r.u32(0)? as usize;
    if size == 12 {
        let bpp = r.u16(10)?;
        return Ok(size + if bpp <= 8 { (1usize << bpp) * 3 } else { 0 });
    }
    if size < 40 {
        return Ok(size);
    }
    let bpp = r.u16(14)?;
    let compression = r.u32(16)?;
    let used = r.u32(32)? as usize;
    let palette = if used != 0 {
        used
    } else if bpp <= 8 {
        1usize << bpp
    } else {
        0
    };
    let masks = if size == 40 {
        match compression {
            3 => 12,
            6 => 16,
            _ => 0,
        }
    } else {
        0
    };
    Ok(size + masks + palette * 4)
}
fn rect(x: i64, y: i64, w: i64, h: i64) -> Rect {
    Rect {
        left: x as f64,
        top: y as f64,
        right: (x + w) as f64,
        bottom: (y + h) as f64,
    }
}
fn append(
    p: &mut Playback,
    image: DibPayload,
    destination: Rect,
    source: Rect,
    rop: u32,
    alpha: u8,
) -> Result<()> {
    let destination = [
        p.map_point([destination.left, destination.top])?,
        p.map_point([destination.right, destination.top])?,
        p.map_point([destination.right, destination.bottom])?,
        p.map_point([destination.left, destination.bottom])?,
    ];
    p.append(ImageCommand {
        image: image.into(),
        destination,
        source: Some(source),
        rop,
        stretch_mode: p.state.stretch_mode,
        constant_alpha: alpha,
        clip: Arc::clone(&p.state.clip),
        location: Location::default(),
    })
}
pub fn emf(kind: u32, r: Reader<'_>, p: &mut Playback) -> Result<()> {
    let n = |i| r.i32(i).map(i64::from);
    let (dest, src, offsets, rop, alpha, use_source_alpha) = match kind {
        76 | 77 => {
            let (w, h) = if kind == 77 && r.data.len() >= 108 {
                (n(100)?, n(104)?)
            } else {
                (n(32)?, n(36)?)
            };
            (
                rect(n(24)?, n(28)?, n(32)?, n(36)?),
                rect(n(44)?, n(48)?, w, h),
                [84, 88, 92, 96],
                r.u32(40)?,
                255,
                false,
            )
        }
        80 | 81 => {
            let (w, h) = if kind == 81 {
                (n(72)?, n(76)?)
            } else {
                (n(40)?, n(44)?)
            };
            (
                rect(n(24)?, n(28)?, w, h),
                rect(n(32)?, n(36)?, n(40)?, n(44)?),
                [48, 52, 56, 60],
                if kind == 81 { r.u32(68)? } else { SRCCOPY },
                255,
                false,
            )
        }
        114 => (
            rect(n(24)?, n(28)?, n(32)?, n(36)?),
            rect(n(44)?, n(48)?, n(100)?, n(104)?),
            [84, 88, 92, 96],
            SRCCOPY,
            r.u8(42)?,
            r.u8(43)? & 1 != 0,
        ),
        _ => return Err(malformed()),
    };
    let [a, b, c, d] = offsets;
    let (off_header, len_header, off_bits, len_bits) = (
        r.u32(a)? as usize,
        r.u32(b)? as usize,
        r.u32(c)? as usize,
        r.u32(d)? as usize,
    );
    if len_header == 0 || len_bits == 0 {
        p.warn(
            "empty_dib",
            "bitmap record does not contain a DIB header and pixel payload",
            true,
        );
        return Ok(());
    }
    if !matches!(
        rop,
        SRCCOPY | 0x008800c6 | 0x00ee0086 | 0x00660046 | 0x00550009 | 0x00000042 | 0x00ff0062
    ) {
        p.warn(
            "unsupported_rop3",
            &format!("unsupported ROP3 will use source-over approximation: {rop:#x}"),
            true,
        );
    }
    if len_header > MAX_BYTES || len_bits > MAX_BYTES || len_header + len_bits > MAX_BYTES {
        return Err(resource("DIB exceeds Python embedded bitmap byte budget"));
    }
    let image = DibPayload {
        header: r.bytes(off_header, len_header)?.into(),
        bits: r.bytes(off_bits, len_bits)?.into(),
        use_source_alpha,
    };
    append(p, image, dest, src, rop, alpha)
}
pub fn wmf(r: Reader<'_>, p: &mut Playback) -> Result<()> {
    r.bytes(0, 22)?;
    let dib = r.bytes(22, r.remaining(22)?)?;
    let header_size = Reader::new(dib).u32(0)? as usize;
    if dib.len() < 12 || header_size < 12 || header_size > dib.len() {
        return Err(malformed());
    }
    let offset = bits_offset(dib)?;
    let r_dib = Reader::new(dib);
    let image = DibPayload {
        header: r_dib.bytes(0, offset)?.into(),
        bits: r_dib.bytes(offset, r_dib.remaining(offset)?)?.into(),
        use_source_alpha: false,
    };
    let n = |i| r.i16(i).map(i64::from);
    append(
        p,
        image,
        rect(n(20)?, n(18)?, n(16)?, n(14)?),
        rect(n(12)?, n(10)?, n(8)?, n(6)?),
        r.u32(0)?,
        255,
    )
}

/// AC_SRC_ALPHA DIBs store premultiplied BGRA. Pillow RGBa -> RGBA floors
/// the unpremultiply and clamps invalid channels; retain RGB when alpha is zero.
/// Other DIB codecs deliberately remain with the pending general decoder.
pub fn raw_alpha(image: &DibPayload) -> Result<Option<Vec<u8>>> {
    if image.header.len() < 40 {
        return Ok(None);
    }
    let r = Reader::new(&image.header);
    let (size, w, h, bpp, compression) = (r.u32(0)?, r.i32(4)?, r.i32(8)?, r.u16(14)?, r.u32(16)?);
    if size < 40 || w <= 0 || h == 0 || bpp != 32 || !matches!(compression, 0 | 3 | 6) {
        return Ok(None);
    }
    let (w, height) = (w as usize, i64::from(h).unsigned_abs() as usize);
    if w as u64 * height as u64 > 16_000_000 || image.bits.len() < w * height * 4 {
        return Err(resource("32-bit DIB exceeds pixel budget or is truncated"));
    }
    let mut rgba = vec![0; w * height * 4];
    for y in 0..height {
        let src_y = if h > 0 { height - 1 - y } else { y };
        for x in 0..w {
            let src = &image.bits[(src_y * w + x) * 4..][..4];
            let dst = &mut rgba[(y * w + x) * 4..][..4];
            for channel in 0..3 {
                dst[channel] = if src[3] == 0 || src[3] == 255 {
                    src[2 - channel]
                } else {
                    (u32::from(src[2 - channel]) * 255 / u32::from(src[3])).min(255) as u8
                };
            }
            dst[3] = src[3];
        }
    }
    Ok(Some(rgba))
}

/// Source intersection, integer crop box and GDI mirror flags, following Pillow
/// crop/transpose order. No pixels are allocated until the decoder uses this.
#[derive(Debug, serde::Serialize)]
pub struct Crop {
    pub bounds: [usize; 4],
    pub flip_x: bool,
    pub flip_y: bool,
    pub fraction: Rect,
}
pub fn crop_source(size: [usize; 2], source: Option<Rect>) -> Option<Crop> {
    let Some(s) = source else {
        return Some(Crop {
            bounds: [0, 0, size[0], size[1]],
            flip_x: false,
            flip_y: false,
            fraction: Rect {
                left: 0.,
                top: 0.,
                right: 1.,
                bottom: 1.,
            },
        });
    };
    if s.width().abs() <= 1e-9 || s.height().abs() <= 1e-9 {
        return None;
    }
    let n = s.normalized();
    let (left, top, right, bottom) = (
        n.left.floor().max(0.),
        n.top.floor().max(0.),
        n.right.ceil().min(size[0] as f64),
        n.bottom.ceil().min(size[1] as f64),
    );
    if right <= left || bottom <= top {
        return None;
    }
    let (x1, x2, y1, y2) = (
        (left - s.left) / s.width(),
        (right - s.left) / s.width(),
        (top - s.top) / s.height(),
        (bottom - s.top) / s.height(),
    );
    Some(Crop {
        bounds: [left as usize, top as usize, right as usize, bottom as usize],
        flip_x: s.width() < 0.,
        flip_y: s.height() < 0.,
        fraction: Rect {
            left: x1.min(x2).max(0.),
            right: x1.max(x2).min(1.),
            top: y1.min(y2).max(0.),
            bottom: y1.max(y2).min(1.),
        },
    })
}
pub fn crop_destination(destination: [Point; 4], f: Rect) -> [Point; 4] {
    let [a, b, _, d] = destination;
    let mapped = |u, v| {
        [
            a[0] + (b[0] - a[0]) * u + (d[0] - a[0]) * v,
            a[1] + (b[1] - a[1]) * u + (d[1] - a[1]) * v,
        ]
    };
    [
        mapped(f.left, f.top),
        mapped(f.right, f.top),
        mapped(f.right, f.bottom),
        mapped(f.left, f.bottom),
    ]
}
