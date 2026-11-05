//! Pillow mode conversion around the published TIFF decoder. Palette TIFFs
//! reuse its compression/strip decoder through a two-byte photometric view;
//! no image payload is cloned and no compression algorithm is replaced.
use super::raster::dimensions;
use crate::{metafile::dib_decode::RgbaImage, Error, ErrorKind, Result};
use ::tiff::{
    decoder::{Decoder, DecodingResult, Limits},
    tags::Tag,
};
use std::io::{Cursor, Read, Seek, SeekFrom};
struct View<'a> {
    cursor: Cursor<&'a [u8]>,
    patch: Option<(usize, [u8; 2])>,
}
impl Read for View<'_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let start = self.cursor.position() as usize;
        let n = self.cursor.read(out)?;
        if let Some((offset, value)) = self.patch {
            for (i, v) in value.iter().enumerate() {
                if offset + i >= start && offset + i < start + n {
                    out[offset + i - start] = *v;
                }
            }
        }
        Ok(n)
    }
}
impl Seek for View<'_> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.cursor.seek(pos)
    }
}
fn err(e: ::tiff::TiffError) -> Error {
    if matches!(e, ::tiff::TiffError::UnsupportedError(_)) {
        Error::new(ErrorKind::Unsupported, e.to_string())
    } else {
        Error::malformed(e)
    }
}
pub fn decode(bytes: &[u8]) -> Result<(RgbaImage, bool)> {
    let mut d = Decoder::new(Cursor::new(bytes))
        .map_err(err)?
        .with_limits(Limits::unlimited());
    let (w, h) = d.dimensions().map_err(err)?;
    let [w, h] = dimensions(w as i64, h as i64)?;
    let mut photo = d.get_tag_u32(Tag::PhotometricInterpretation).map_err(err)?;
    let samples = d.get_tag_u32(Tag::SamplesPerPixel).unwrap_or(1) as usize;
    let depths = d
        .get_tag_u16_vec(Tag::BitsPerSample)
        .unwrap_or_else(|_| vec![1]);
    let depth = *depths
        .first()
        .ok_or_else(|| Error::malformed("TIFF has no bits per sample"))?;
    if samples == 0 {
        return Err(Error::malformed("TIFF has no samples"));
    }
    let extra = d.get_tag_u16_vec(Tag::ExtraSamples).unwrap_or_default();
    let orientation = d.get_tag_u32(Tag::Orientation).unwrap_or(1);
    let palette = if photo == 3 {
        Some(d.get_tag_u16_vec(Tag::ColorMap).map_err(err)?)
    } else {
        None
    };
    let patch = if photo == 3 {
        Some(photometric_patch(bytes)?)
    } else {
        None
    };
    let mut d = Decoder::new(View {
        cursor: Cursor::new(bytes),
        patch,
    })
    .map_err(err)?
    .with_limits(Limits::unlimited());
    let compression = d.get_tag_u32(Tag::Compression).unwrap_or(1);
    let values: Vec<u8> = if compression == 7 {
        jpeg_samples(&mut d, bytes, w, h, photo, samples)?
    } else {
        match d.read_image().map_err(err)? {
            DecodingResult::U8(v) => {
                if depth < 8 {
                    let stride = (w * samples * depth as usize).div_ceil(8);
                    let mask = (1u16 << depth) - 1;
                    let mut out = Vec::with_capacity(w * h * samples);
                    for row in v.chunks_exact(stride) {
                        for i in 0..w * samples {
                            let shift = 8 - depth as usize - (i * depth as usize % 8);
                            out.push(((row[i * depth as usize / 8] >> shift) as u16 & mask) as u8);
                        }
                    }
                    out
                } else {
                    v
                }
            }
            DecodingResult::U16(v) => v
                .into_iter()
                .map(|v| {
                    if samples == 1 && palette.is_none() {
                        v.min(255) as u8
                    } else {
                        (v >> 8) as u8
                    }
                })
                .collect(),
            DecodingResult::I32(v) => v.into_iter().map(|v| v.clamp(0, 255) as u8).collect(),
            DecodingResult::U32(v) => v
                .into_iter()
                .map(|v| (v as i32).clamp(0, 255) as u8)
                .collect(),
            DecodingResult::F32(v) => v.into_iter().map(|v| v.clamp(0., 255.) as u8).collect(),
            _ => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "TIFF sample type not yet connected",
                ))
            }
        }
    };
    if compression == 7 && photo == 6 {
        photo = 2;
    }
    if values.len() != w * h * samples {
        return Err(Error::malformed("TIFF decoded sample count mismatch"));
    }
    let alpha = matches!(photo, 0 | 1 | 2) && !extra.is_empty() && extra[0] != 0;
    let mut rgba = Vec::with_capacity(w * h * 4);
    for p in values.chunks_exact(samples) {
        let mut pixel = match photo {
            0 | 1 => {
                let v = if depth == 1 {
                    if p[0] != 0 {
                        255
                    } else {
                        0
                    }
                } else {
                    p[0]
                };
                [
                    v,
                    v,
                    v,
                    if alpha {
                        *p.get(1).unwrap_or(&255)
                    } else {
                        255
                    },
                ]
            }
            2 => {
                if samples < 3 {
                    return Err(Error::malformed("TIFF RGB sample count"));
                }
                [
                    p[0],
                    p[1],
                    p[2],
                    if alpha {
                        *p.get(3).unwrap_or(&255)
                    } else {
                        255
                    },
                ]
            }
            3 => {
                let palette = palette.as_ref().unwrap();
                let n = palette.len() / 3;
                let i = p[0] as usize;
                if i >= n {
                    return Err(Error::malformed("TIFF palette index"));
                }
                [
                    (palette[i] >> 8) as u8,
                    (palette[n + i] >> 8) as u8,
                    (palette[2 * n + i] >> 8) as u8,
                    255,
                ]
            }
            5 => {
                if samples < 4 {
                    return Err(Error::malformed("TIFF CMYK sample count"));
                }
                let k = 255 - p[3] as u32;
                let channel = |v: u8| {
                    let t = v as u32 * k + 128;
                    (k - ((t + (t >> 8)) >> 8)) as u8
                };
                [channel(p[0]), channel(p[1]), channel(p[2]), 255]
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "TIFF photometric conversion not yet connected",
                ))
            }
        };
        if alpha && extra[0] == 1 && pixel[3] != 0 && pixel[3] != 255 {
            let a = pixel[3] as u32;
            for v in &mut pixel[..3] {
                *v = (*v as u32 * 255 / a).min(255) as u8;
            }
        }
        rgba.extend(pixel);
    }
    let (width, height) = if (5..=8).contains(&orientation) {
        (h, w)
    } else {
        (w, h)
    };
    if (2..=8).contains(&orientation) {
        let mut transformed = vec![0; rgba.len()];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = match orientation {
                    2 => (w - 1 - x, y),
                    3 => (w - 1 - x, h - 1 - y),
                    4 => (x, h - 1 - y),
                    5 => (y, x),
                    6 => (h - 1 - y, x),
                    7 => (h - 1 - y, w - 1 - x),
                    8 => (y, w - 1 - x),
                    _ => (x, y),
                };
                transformed[(dy * width + dx) * 4..][..4]
                    .copy_from_slice(&rgba[(y * w + x) * 4..][..4]);
            }
        }
        rgba = transformed;
    }
    Ok((
        RgbaImage {
            width,
            height,
            pixels: rgba,
        },
        alpha,
    ))
}
fn jpeg_samples(
    d: &mut Decoder<View<'_>>,
    bytes: &[u8],
    w: usize,
    h: usize,
    photo: u32,
    samples: usize,
) -> Result<Vec<u8>> {
    let offsets = d.get_tag_u64_vec(Tag::StripOffsets).map_err(err)?;
    let counts = d.get_tag_u64_vec(Tag::StripByteCounts).map_err(err)?;
    let rows = d.get_tag_u32(Tag::RowsPerStrip).unwrap_or(h as u32) as usize;
    let tables = d.get_tag_u8_vec(Tag::JPEGTables).unwrap_or_default();
    if rows == 0
        || offsets.len() != counts.len()
        || !matches!((photo, samples), (1, 1) | (2, 3) | (5, 4) | (6, 3))
    {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "JPEG TIFF sample layout not yet connected",
        ));
    }
    if offsets.len() != h.div_ceil(rows) {
        return Err(Error::malformed("JPEG TIFF strip count mismatch"));
    }
    let mut values = vec![0; w * h * samples];
    for (i, (offset, count)) in offsets.into_iter().zip(counts).enumerate() {
        let offset = usize::try_from(offset).map_err(Error::malformed)?;
        let count = usize::try_from(count).map_err(Error::malformed)?;
        let strip = bytes
            .get(
                offset
                    ..offset
                        .checked_add(count)
                        .ok_or_else(|| Error::malformed("JPEG TIFF strip overflow"))?,
            )
            .ok_or_else(|| Error::malformed("JPEG TIFF strip bounds"))?;
        let mut data = Vec::with_capacity(tables.len() + strip.len());
        if !tables.is_empty() {
            let tables = tables.strip_suffix(&[255, 217]).unwrap_or(&tables);
            data.extend_from_slice(tables);
            data.extend_from_slice(strip.strip_prefix(&[255, 216]).unwrap_or(strip));
        } else {
            data.extend_from_slice(strip);
        }
        let mut decoder = turbojpeg::Decompressor::new().map_err(Error::malformed)?;
        let info = decoder.read_header(&data).map_err(Error::malformed)?;
        dimensions(info.width as i64, info.height as i64)?;
        let y = i
            .checked_mul(rows)
            .ok_or_else(|| Error::malformed("JPEG TIFF strip position"))?;
        if y >= h || info.width != w || info.height < (h - y).min(rows) {
            return Err(Error::malformed("JPEG TIFF strip dimensions"));
        }
        let mut pixels = vec![0; info.width * info.height * samples];
        let format = match samples {
            1 => turbojpeg::PixelFormat::GRAY,
            3 => turbojpeg::PixelFormat::RGB,
            _ => turbojpeg::PixelFormat::CMYK,
        };
        decoder
            .decompress(
                &data,
                turbojpeg::Image {
                    pixels: pixels.as_mut_slice(),
                    width: w,
                    height: info.height,
                    pitch: w * samples,
                    format,
                },
            )
            .map_err(Error::malformed)?;
        let n = w * (h - y).min(rows) * samples;
        values[y * w * samples..y * w * samples + n].copy_from_slice(&pixels[..n]);
    }
    Ok(values)
}
fn photometric_patch(bytes: &[u8]) -> Result<(usize, [u8; 2])> {
    let little = bytes.starts_with(b"II");
    let read16 = |at: usize| -> Result<u16> {
        let b: [u8; 2] = bytes
            .get(at..at + 2)
            .ok_or_else(|| Error::malformed("TIFF tag bounds"))?
            .try_into()
            .unwrap();
        Ok(if little {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        })
    };
    let read32 = |at: usize| -> Result<u32> {
        let b: [u8; 4] = bytes
            .get(at..at + 4)
            .ok_or_else(|| Error::malformed("TIFF IFD bounds"))?
            .try_into()
            .unwrap();
        Ok(if little {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    };
    if read16(2)? != 42 {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "palette BigTIFF metadata view not yet connected",
        ));
    }
    let ifd = read32(4)? as usize;
    let count = read16(ifd)? as usize;
    for i in 0..count {
        let at = ifd + 2 + i * 12;
        if read16(at)? == 262 && read16(at + 2)? == 3 && read32(at + 4)? == 1 {
            return Ok((
                at + 8,
                if little {
                    1u16.to_le_bytes()
                } else {
                    1u16.to_be_bytes()
                },
            ));
        }
    }
    Err(Error::malformed("TIFF photometric SHORT tag missing"))
}
