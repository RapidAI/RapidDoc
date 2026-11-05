// Port of Pillow 12.2.0 BmpImagePlugin.py and Unpack.c.
// Copyright Secret Labs AB, Fredrik Lundh, Jeffrey A. Clark and contributors.
// SPDX-License-Identifier: MIT-CMU. See licenses/PILLOW.txt.
use super::{binary::Reader, bitmap, malformed, model::DibPayload, resource};
use crate::Result;

#[derive(Debug, Clone)]
pub struct RgbaImage {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) pixels: Vec<u8>,
}
impl RgbaImage {
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self> {
        if width == 0 || height == 0 || width as u128 * height as u128 > 16_000_000 {
            return Err(resource("bitmap dimensions exceed Python pixel budget"));
        }
        if pixels.len() != width * height * 4 {
            return Err(malformed());
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }
    pub fn blank(size: [usize; 2]) -> Result<Self> {
        let [w, h] = size;
        if w == 0 || h == 0 || w as u128 * h as u128 > 16_000_000 {
            return Err(resource("bitmap dimensions exceed Python pixel budget"));
        }
        Self::new(w, h, vec![0; w * h * 4])
    }
    pub fn size(&self) -> [usize; 2] {
        [self.width, self.height]
    }
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

// BMP palette/mask reads can cross the supplied header into bits. Keep the
// original BMP file-offset semantics without concatenating a second large BMP.
struct Joined<'a>(&'a [u8], &'a [u8]);
impl Joined<'_> {
    fn byte(&self, i: usize) -> Option<u8> {
        if i < self.0.len() {
            self.0.get(i).copied()
        } else {
            self.1.get(i - self.0.len()).copied()
        }
    }
    fn u32(&self, i: usize) -> Result<u32> {
        Ok(u32::from_le_bytes([
            self.byte(i).ok_or_else(malformed)?,
            self.byte(i + 1).ok_or_else(malformed)?,
            self.byte(i + 2).ok_or_else(malformed)?,
            self.byte(i + 3).ok_or_else(malformed)?,
        ]))
    }
}

pub fn decode(image: &DibPayload) -> Result<RgbaImage> {
    bitmap::validate(image)?;
    let [w, h] = bitmap::dimensions(&image.header)?;
    if image.use_source_alpha {
        if let Some(pixels) = bitmap::raw_alpha(image)? {
            return RgbaImage::new(w, h, pixels);
        }
    }
    let r = Reader::new(&image.header);
    let size = r.u32(0)? as usize;
    if !matches!(size, 12 | 40 | 52 | 56 | 64 | 108 | 124) {
        return Err(malformed());
    }
    let core = size == 12;
    let bpp = r.u16(if core { 10 } else { 14 })? as usize;
    if !matches!(bpp, 1 | 4 | 8 | 16 | 24 | 32) {
        return Err(malformed());
    }
    let compression = if core { 0 } else { r.u32(16)? };
    if !matches!(compression, 0..=3) {
        return Err(malformed());
    }
    let top_down = !core && r.u8(11)? == 255;
    let joined = Joined(&image.header, &image.bits);
    let mut palette_start = size;
    let mut masks = [0xff0000, 0xff00, 0xff, 0];
    if bpp == 16 {
        masks = [0x7c00, 0x3e0, 0x1f, 0];
    }
    if compression == 3 {
        masks = [
            joined.u32(40)?,
            joined.u32(44)?,
            joined.u32(48)?,
            if size >= 56 { joined.u32(52)? } else { 0 },
        ];
        if size == 40 {
            palette_start += 12;
        }
        let supported = match bpp {
            16 => matches!(masks[..3], [0xf800, 0x7e0, 0x1f] | [0x7c00, 0x3e0, 0x1f]),
            24 => masks[..3] == [0xff0000, 0xff00, 0xff],
            32 => matches!(
                masks,
                [0xff0000, 0xff00, 0xff, 0]
                    | [0xff000000, 0xff0000, 0xff00, 0]
                    | [0xff000000, 0xff00, 0xff, 0]
                    | [0xff000000, 0xff0000, 0xff00, 0xff]
                    | [0xff, 0xff00, 0xff0000, 0xff000000]
                    | [0xff0000, 0xff00, 0xff, 0xff000000]
                    | [0xff000000, 0xff00, 0xff, 0xff0000]
                    | [0, 0, 0, 0]
            ),
            _ => false,
        };
        if !supported {
            return Err(malformed());
        }
        if masks == [0; 4] {
            masks = [0xff0000, 0xff00, 0xff, 0xff000000];
        }
    }
    let colors = if bpp > 8 {
        0 // Not a paletted image; avoid an unused 1 << 32 on 32-bit targets.
    } else if core {
        1usize << bpp
    } else {
        let c = r.u32(32)? as usize;
        if c == 0 {
            1usize << bpp
        } else {
            c
        }
    };
    let padding = if core { 3 } else { 4 };
    let mut offset = 0;
    let mut palette = [[0u8; 3]; 256];
    let mut gray = false;
    if bpp <= 8 {
        if colors == 0 || colors > 65536 {
            return Err(malformed());
        }
        if image.header.len() == size {
            offset = padding * colors;
        }
        gray = true;
        for i in 0..colors {
            let pos = palette_start + i * padding;
            let entry = [joined.byte(pos), joined.byte(pos + 1), joined.byte(pos + 2)];
            let expected = if colors == 2 {
                if i == 0 {
                    0
                } else {
                    255
                }
            } else {
                i
            };
            if entry != [Some(expected as u8); 3] {
                gray = false;
            }
            if i < 256 {
                if let [Some(b), Some(g), Some(r)] = entry {
                    palette[i] = [r, g, b];
                }
            }
        }
        let available = (image.header.len() + image.bits.len())
            .saturating_sub(palette_start)
            .min(colors * padding);
        if !gray && available / padding > 256 {
            return Err(malformed());
        }
    }
    let bits = image.bits.get(offset..).ok_or_else(malformed)?;
    if matches!(compression, 1 | 2) && bpp > 8 {
        return Err(malformed());
    }
    if matches!(compression, 1 | 2) && gray && colors == 2 {
        return Err(malformed());
    }
    let indexed = if matches!(compression, 1 | 2) {
        Some(rle(
            bits,
            w,
            h,
            compression == 2,
            image.header.len() + offset,
        )?)
    } else {
        None
    };
    let stride = (w * bpp).div_ceil(32) * 4;
    let effective_bpp = if gray {
        if colors == 2 {
            1
        } else {
            8
        }
    } else {
        bpp
    };
    let row_len = (w * effective_bpp).div_ceil(8);
    if indexed.is_none() && (stride < row_len || (h - 1) * stride + row_len > bits.len()) {
        return Err(malformed());
    }
    let mut pixels = vec![0; w * h * 4];
    for y in 0..h {
        let src_y = if top_down { y } else { h - 1 - y };
        for x in 0..w {
            let dst = &mut pixels[(y * w + x) * 4..][..4];
            if bpp <= 8 {
                let value = if let Some(indices) = &indexed {
                    indices[src_y * w + x]
                } else {
                    let bit = x * effective_bpp;
                    (bits[src_y * stride + bit / 8] >> (8 - effective_bpp - bit % 8))
                        & ((1u16 << effective_bpp) - 1) as u8
                };
                let rgb = if gray {
                    [if colors == 2 {
                        if value == 0 {
                            0
                        } else {
                            255
                        }
                    } else {
                        value
                    }; 3]
                } else {
                    palette[value as usize]
                };
                dst[..3].copy_from_slice(&rgb);
                dst[3] = 255;
            } else {
                let pos = src_y * stride + x * (bpp / 8);
                let mut word = 0u32;
                for i in 0..bpp / 8 {
                    word |= u32::from(bits[pos + i]) << (i * 8);
                }
                for c in 0..3 {
                    let mask = masks[c];
                    let value = (word & mask) >> mask.trailing_zeros();
                    dst[c] = (value * 255 / (mask >> mask.trailing_zeros())) as u8;
                }
                dst[3] = if bpp == 32 && image.use_source_alpha && masks[3] != 0 {
                    ((word & masks[3]) >> masks[3].trailing_zeros()) as u8
                } else {
                    255
                };
            }
        }
    }
    RgbaImage::new(w, h, pixels)
}

// Preserve Pillow's scanline clamping, delta gaps, absolute RLE4 odd-count
// truncation, word alignment and early EOF behavior, rather than a new decoder.
fn rle(bits: &[u8], w: usize, h: usize, four: bool, base: usize) -> Result<Vec<u8>> {
    let mut data = Vec::with_capacity(w * h);
    let (mut pos, mut x) = (0, 0usize);
    while data.len() < w * h && pos + 2 <= bits.len() {
        let (count, value) = (bits[pos] as usize, bits[pos + 1] as usize);
        pos += 2;
        if count != 0 {
            let count = count.min(w.saturating_sub(x));
            for i in 0..count {
                data.push(if four {
                    if i % 2 == 0 {
                        (value >> 4) as u8
                    } else {
                        (value & 15) as u8
                    }
                } else {
                    value as u8
                });
            }
            x += count;
        } else {
            match value {
                0 => {
                    let n = (w - data.len() % w) % w;
                    data.resize(data.len() + n, 0);
                    x = 0;
                }
                1 => break,
                2 => {
                    if pos + 2 > bits.len() {
                        break;
                    }
                    let n = bits[pos] as usize + bits[pos + 1] as usize * w;
                    pos += 2;
                    data.resize((data.len() + n).min(w * h), 0);
                    x = data.len() % w;
                }
                n => {
                    let count = if four { n / 2 } else { n };
                    let actual = count.min(bits.len() - pos);
                    for &v in &bits[pos..pos + actual] {
                        if four {
                            data.extend([v >> 4, v & 15]);
                        } else {
                            data.push(v);
                        }
                    }
                    pos += actual;
                    if actual < count {
                        break;
                    }
                    x += n;
                    if (base + pos) % 2 != 0 {
                        pos += 1;
                    }
                }
            }
        }
    }
    if data.len() < w * h {
        return Err(malformed());
    }
    data.truncate(w * h);
    Ok(data)
}
