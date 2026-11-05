// Copyright (c) 2026 Xiaomeng Zhao (myhloli). SPDX-License-Identifier: MIT
// source_over adapts Pillow AlphaComposite.c, MIT-CMU (licenses/PILLOW.txt).
//! GDI logical operations; source-over and raster placement are separate layers.
use super::malformed;
use crate::Result;
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bitwise {
    Xor,
    And,
    Or,
    NotOr,
    AndNotSource,
    NotSource,
    SourceAndNotDestination,
    NotAnd,
    NotXor,
    DestinationOrNotSource,
    SourceOrNotDestination,
    Invert,
    Black,
    White,
}
impl Bitwise {
    fn channel(self, d: u8, s: u8) -> u8 {
        match self {
            Self::Xor => d ^ s,
            Self::And => d & s,
            Self::Or => d | s,
            Self::NotOr => !(d | s),
            Self::AndNotSource => d & !s,
            Self::NotSource => !s,
            Self::SourceAndNotDestination => s & !d,
            Self::NotAnd => !(d & s),
            Self::NotXor => !(d ^ s),
            Self::DestinationOrNotSource => d | !s,
            Self::SourceOrNotDestination => s | !d,
            Self::Invert => !d,
            Self::Black => 0,
            Self::White => 255,
        }
    }
}
// Pillow's masked paste uses DIV255 with nearest rounding for every channel,
// including alpha. Bitwise RGB and alpha=max(destination, mask) are then pasted.
fn blend(d: u8, s: u8, m: u8) -> u8 {
    let value = u32::from(d) * (255 - u32::from(m)) + u32::from(s) * u32::from(m) + 128;
    (((value >> 8) + value) >> 8) as u8
}
pub fn paste_bitwise(canvas: &mut [u8], layer: &[u8], operation: Bitwise) -> Result<()> {
    if canvas.len() != layer.len() || canvas.len() % 4 != 0 {
        return Err(malformed());
    }
    for (d, s) in canvas.chunks_exact_mut(4).zip(layer.chunks_exact(4)) {
        let mask = s[3];
        if mask == 0 {
            continue;
        }
        for c in 0..3 {
            d[c] = blend(d[c], operation.channel(d[c], s[c]), mask);
        }
        d[3] = blend(d[3], d[3].max(mask), mask);
    }
    Ok(())
}

/// Pillow's seven-bit integer source-over, preserving its channel rounding.
pub fn source_over(canvas: &mut [u8], layer: &[u8]) -> Result<()> {
    if canvas.len() != layer.len() || canvas.len() % 4 != 0 {
        return Err(malformed());
    }
    let div255 = |v: u32| ((v >> 8) + v) >> 8;
    for (d, s) in canvas.chunks_exact_mut(4).zip(layer.chunks_exact(4)) {
        if s[3] == 0 {
            continue;
        }
        let blend = u32::from(d[3]) * (255 - u32::from(s[3]));
        let alpha = u32::from(s[3]) * 255 + blend;
        let c1 = u32::from(s[3]) * 255 * 255 * 128 / alpha;
        let c2 = 255 * 128 - c1;
        for c in 0..3 {
            d[c] = (div255(u32::from(s[c]) * c1 + u32::from(d[c]) * c2 + (128 << 7)) >> 7) as u8;
        }
        d[3] = div255(alpha + 128) as u8;
    }
    Ok(())
}
pub fn image(canvas: &mut [u8], layer: &[u8], rop: u32) -> Result<()> {
    let operation = match rop {
        0x008800c6 => Bitwise::And,
        0x00ee0086 => Bitwise::Or,
        0x00660046 => Bitwise::Xor,
        0x00550009 => Bitwise::Invert,
        0x00000042 => Bitwise::Black,
        0x00ff0062 => Bitwise::White,
        _ => return source_over(canvas, layer),
    };
    paste_bitwise(canvas, layer, operation)
}
pub fn path(canvas: &mut [u8], layer: &[u8], rop: u32) -> Result<()> {
    use Bitwise::*;
    let operation = match rop {
        1 => Black,
        2 => NotOr,
        3 => AndNotSource,
        4 => NotSource,
        5 => SourceAndNotDestination,
        6 => Invert,
        7 => Xor,
        8 => NotAnd,
        9 => And,
        10 => NotXor,
        11 => {
            return if canvas.len() == layer.len() && canvas.len() % 4 == 0 {
                Ok(())
            } else {
                Err(malformed())
            }
        }
        12 => DestinationOrNotSource,
        13 => return source_over(canvas, layer),
        14 => SourceOrNotDestination,
        15 => Or,
        16 => White,
        _ => Xor,
    };
    paste_bitwise(canvas, layer, operation)
}
