// Copyright (c) 2026 Xiaomeng Zhao (myhloli). SPDX-License-Identifier: MIT
use super::{
    binary::{plus_records, Reader, Records},
    malformed,
    model::*,
    resource,
};
use crate::Result;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Wmf,
    Emf,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlusMode {
    None,
    Dual,
    Only,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub source_format: Source,
    pub record_start: usize,
    pub record_end: usize,
    pub bounds: Option<Rect>,
    pub pixel_size: Option<[u64; 2]>,
    pub emfplus_mode: PlusMode,
    pub device_pixels_per_mm: Point,
    pub wmf_object_count: usize,
}
pub fn prepare(data: &[u8], dpi: u32, size: Option<[u64; 2]>) -> Result<Header> {
    if data.is_empty() {
        return Err(malformed());
    }
    if data.len() > 128 * 1024 * 1024 {
        return Err(resource("metafile exceeds Python max_metafile_bytes"));
    }
    if !(1..=1200).contains(&dpi) || size.is_some_and(|s| s[0] == 0 || s[1] == 0) {
        return Err(malformed());
    }
    let r = Reader::new(data);
    if data.len() >= 44 && r.u32(0)? == 1 && r.u32(40)? == 0x464d4520 {
        emf(data, dpi, size)
    } else {
        wmf(data, dpi, size)
    }
}
fn pixel(value: f64) -> u64 {
    value.round_ties_even().max(1.) as u64
}
fn emf(data: &[u8], dpi: u32, mut size: Option<[u64; 2]>) -> Result<Header> {
    let r = Reader::new(data);
    if data.len() < 88 || r.u32(0)? != 1 {
        return Err(malformed());
    }
    let header = r.u32(4)? as usize;
    if header < 88 || header % 4 != 0 || header > data.len() || r.u32(40)? != 0x464d4520 {
        return Err(malformed());
    }
    let end = r.u32(48)? as usize;
    let count = r.u32(52)?;
    if end < header || end > data.len() {
        return Err(malformed());
    }
    if count == 0 || count > 1_000_000 {
        return Err(resource("invalid or excessive EMF declared record count"));
    }
    let mut bounds = r.rect32(8)?;
    let frame = r.rect32(24)?;
    let dx = r.i32(72)?;
    let dy = r.i32(76)?;
    let mx = r.i32(80)?;
    let my = r.i32(84)?;
    let scale = if dx > 0 && dy > 0 && mx > 0 && my > 0 {
        [dx as f64 / mx as f64, dy as f64 / my as f64]
    } else {
        [dpi as f64 / 25.4, dpi as f64 / 25.4]
    };
    if size.is_none() && frame.width() != 0. && frame.height() != 0. {
        size = Some([
            pixel(frame.width().abs() * dpi as f64 / 2540.),
            pixel(frame.height().abs() * dpi as f64 / 2540.),
        ]);
    }
    if size.is_none() && bounds.width() != 0. && bounds.height() != 0. {
        size = Some([pixel(bounds.width().abs()), pixel(bounds.height().abs())]);
    }
    let mut mode = PlusMode::None;
    let mut plus_count = 0;
    let mut eof = false;
    for record in Records::new(data, 0, end, Source::Emf)? {
        let record = record?;
        if record.kind != 70 {
            continue;
        }
        for item in plus_records(record.data)? {
            let item = item?;
            plus_count += 1;
            if plus_count > 1_000_000 {
                return Err(resource("EMF+ exceeds Python record budget"));
            }
            if eof || mode == PlusMode::None && item.kind != 0x4001 {
                return Err(malformed());
            }
            if item.kind == 0x4001 {
                if mode != PlusMode::None
                    || item.data.data.len() != 16
                    || item.data.u32(0)? >> 12 != 0xdbc01
                {
                    return Err(malformed());
                }
                let dx = item.data.u32(8)?;
                let dy = item.data.u32(12)?;
                if dx == 0 || dy == 0 || dx.max(dy) > 100000 {
                    return Err(malformed());
                }
                mode = if item.flags & 1 != 0 {
                    PlusMode::Dual
                } else {
                    PlusMode::Only
                };
            } else if item.kind == 0x4002 {
                if !item.data.data.is_empty() {
                    return Err(malformed());
                }
                eof = true;
            }
        }
    }
    if mode != PlusMode::None && !eof {
        return Err(malformed());
    }
    if mode == PlusMode::Only && frame.width() > 0. && frame.height() > 0. {
        bounds = Rect {
            left: frame.left * scale[0] / 100.,
            top: frame.top * scale[1] / 100.,
            right: frame.right * scale[0] / 100.,
            bottom: frame.bottom * scale[1] / 100.,
        };
    }
    Ok(Header {
        source_format: Source::Emf,
        record_start: 0,
        record_end: end,
        bounds: Some(bounds),
        pixel_size: size,
        emfplus_mode: mode,
        device_pixels_per_mm: scale,
        wmf_object_count: 0,
    })
}
fn wmf(data: &[u8], dpi: u32, mut size: Option<[u64; 2]>) -> Result<Header> {
    let r = Reader::new(data);
    let mut start = 0;
    let mut bounds = None;
    if data.len() >= 22 && r.u32(0)? == 0x9ac6cdd7 {
        let mut checksum = 0;
        for offset in (0..20).step_by(2) {
            checksum ^= r.u16(offset)?;
        }
        if checksum != r.u16(20)? {
            return Err(malformed());
        }
        let rect = Rect {
            left: r.i16(6)? as f64,
            top: r.i16(8)? as f64,
            right: r.i16(10)? as f64,
            bottom: r.i16(12)? as f64,
        };
        let units = r.u16(14)?;
        if units == 0 {
            return Err(malformed());
        }
        if size.is_none() {
            size = Some([
                pixel(rect.width().abs() * dpi as f64 / f64::from(units)),
                pixel(rect.height().abs() * dpi as f64 / f64::from(units)),
            ]);
        }
        bounds = Some(rect);
        start = 22;
    }
    let h = r.sub(start, 18)?;
    if !matches!(h.u16(0)?, 1 | 2) || h.u16(2)? != 9 {
        return Err(malformed());
    }
    let length = (h.u32(6)? as usize).checked_mul(2).ok_or_else(malformed)?;
    if length < 18 || length > data.len() - start {
        return Err(malformed());
    }
    Ok(Header {
        source_format: Source::Wmf,
        record_start: start + 18,
        record_end: start + length,
        bounds,
        pixel_size: size,
        emfplus_mode: PlusMode::None,
        device_pixels_per_mm: [dpi as f64 / 25.4, dpi as f64 / 25.4],
        wmf_object_count: h.u16(10)? as usize,
    })
}
pub fn bounded_canvas(size: [u64; 2]) -> Result<([u64; 2], bool)> {
    let [w, h] = size;
    if w == 0 || h == 0 {
        return Err(malformed());
    }
    let (w, h) = (w as f64, h as f64);
    let mut scale = 1f64.min(8192. / w).min(8192. / h);
    if w * h * scale * scale > 16_000_000. {
        scale = scale.min((16_000_000. / (w * h)).sqrt());
    }
    Ok(([pixel(w * scale), pixel(h * scale)], scale < 1.))
}
pub fn check_render_size(width: u64, height: u64, work: u64) -> Result<()> {
    if width == 0
        || height == 0
        || width.max(height) > 8192
        || u128::from(width) * u128::from(height) > 16_000_000
    {
        return Err(resource(
            "metafile render dimensions exceed Python canvas budget",
        ));
    }
    if u128::from(width) * u128::from(height) * u128::from(work) > 512_000_000 {
        return Err(resource("metafile exceeds Python max_render_work_pixels"));
    }
    Ok(())
}
