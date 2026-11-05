//! Shared Office image policy, ported from DocVortex office/image.py.
//! The metafile parser keeps strict errors; this caller chooses its fallback.
use crate::{
    metafile::{self, dib_decode::RgbaImage, fonts, model::Color, text_render},
    Error, Result,
};
use std::sync::{Arc, OnceLock};
pub mod raster;
pub mod svg;
mod tiff;

pub fn serialize(data: &[u8], part: Option<&str>, mime: Option<&str>) -> Result<Option<Arc<str>>> {
    serialize_with_decoder(
        data,
        part,
        mime,
        &mut crate::equation::image_mtef::Decoder::default(),
    )
}
pub fn serialize_with_decoder(
    data: &[u8],
    part: Option<&str>,
    mime: Option<&str>,
    probe: &mut crate::equation::image_mtef::Decoder,
) -> Result<Option<Arc<str>>> {
    probe.probe(data, part, mime)?;
    if probe.limit_reason(data, part, mime).is_some() {
        return image_limit_placeholder().map(Some);
    }
    if is_vector(data, part, mime) {
        return vector_unchecked(data, None).map(|r| Some(r.source));
    }
    if part.is_some_and(|p| p.to_ascii_lowercase().ends_with(".svg"))
        || mime.is_some_and(|s| {
            s.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("image/svg+xml")
        })
        || data[..data.len().min(4096)]
            .windows(4)
            .any(|s| s == b"<svg")
    {
        return svg::serialize(data).map(|s| s.map(Arc::from));
    }
    raster::serialize(data).map(|s| s.map(Arc::from))
}

pub fn is_wmf(data: &[u8]) -> bool {
    data.starts_with(&[0xd7, 0xcd, 0xc6, 0x9a])
        || data.len() >= 4 && matches!(&data[..2], [1, 0] | [2, 0]) && data[2..4] == [9, 0]
}
pub fn is_emf(data: &[u8]) -> bool {
    data.len() >= 44 && data[..4] == [1, 0, 0, 0] && &data[40..44] == b" EMF"
}
pub fn is_vector_part(part: Option<&str>, mime: Option<&str>) -> bool {
    let extension = part
        .unwrap_or("")
        .rsplit('/')
        .next()
        .unwrap_or("")
        .rsplit_once('.')
        .map(|(_, s)| s.to_ascii_lowercase());
    matches!(extension.as_deref(), Some("wmf" | "emf"))
        || matches!(
            mime.unwrap_or("")
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "image/x-wmf" | "image/wmf" | "image/x-emf" | "image/emf" | "application/x-msmetafile"
        )
}
pub fn is_vector(data: &[u8], part: Option<&str>, mime: Option<&str>) -> bool {
    is_wmf(data) || is_emf(data) || is_vector_part(part, mime)
}
/// Same integer, ties-to-even EMU sizing as the Python serializer.
pub fn size_from_emu(size: Option<[i64; 2]>, dpi: u32) -> Option<[u64; 2]> {
    let [w, h] = size?;
    if w <= 0 || h <= 0 {
        return None;
    }
    Some([w, h].map(|v| {
        let n = v as u128 * dpi as u128;
        let (q, r) = (n / 914400, n % 914400);
        (q + u128::from(r * 2 > 914400 || (r * 2 == 914400 && q % 2 != 0))).max(1) as u64
    }))
}
fn multiline(
    face: &fonts::Face,
    lines: &[&str],
    spacing: f64,
    origin: [f64; 2],
) -> Result<(Vec<[f64; 2]>, [f64; 4])> {
    let widths = lines
        .iter()
        .map(|s| face.length(s))
        .collect::<Result<Vec<_>>>()?;
    let max = widths.iter().copied().fold(0., f64::max);
    let step = face.bounds("A", "la")?[3] as f64 + spacing;
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut positions = vec![];
    for (i, (line, width)) in lines.iter().zip(widths).enumerate() {
        let p = [origin[0] + (max - width) / 2., origin[1] + i as f64 * step];
        let b = face.bounds(line, "la")?;
        bounds[0] = bounds[0].min(p[0] + b[0] as f64);
        bounds[1] = bounds[1].min(p[1] + b[1] as f64);
        bounds[2] = bounds[2].max(p[0] + b[2] as f64);
        bounds[3] = bounds[3].max(p[1] + b[3] as f64);
        positions.push(p);
    }
    Ok((positions, bounds))
}
/// Original rectangle, shrinking font selection and multiline placement.
pub fn text_placeholder(size: [usize; 2], lines: &[&str]) -> Result<RgbaImage> {
    let [w, h] = size.map(|v| v.max(1));
    let mut image = RgbaImage::blank([w, h])?;
    for p in image.pixels.chunks_exact_mut(4) {
        p.copy_from_slice(&[240, 240, 240, 255]);
    }
    let border = (w.min(h) / 80).max(1);
    for y in 0..h {
        for x in 0..w {
            if x < border
                || y < border
                || x >= w.saturating_sub(border)
                || y >= h.saturating_sub(border)
            {
                image.pixels[(y * w + x) * 4..][..4].copy_from_slice(&[190, 190, 190, 255]);
            }
        }
    }
    let text = lines
        .iter()
        .filter(|s| !s.is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join("\n");
    let mut lines = text.split('\n').collect::<Vec<_>>();
    if text.is_empty() {
        lines = vec!["WMF/EMF"];
    }
    let mut selected = None;
    for size in (8..=(w.min(h) / 7).max(10)).rev() {
        let face = fonts::office_placeholder(size as u32)?;
        let spacing = (size / 4).max(2) as f64;
        let (_, b) = multiline(&face, &lines, spacing, [0., 0.])?;
        if b[2] - b[0] <= w.saturating_sub(16).max(1) as f64
            && b[3] - b[1] <= h.saturating_sub(16).max(1) as f64
        {
            selected = Some((face, spacing, b));
            break;
        }
    }
    let (face, spacing, bounds) = if let Some(v) = selected {
        v
    } else {
        lines = vec!["WMF/EMF"];
        let face = fonts::office_placeholder((w.min(h) / 5).max(10) as u32)?;
        let (_, b) = multiline(&face, &lines, 2., [0., 0.])?;
        (face, 2., b)
    };
    let origin = [
        (w as f64 - (bounds[2] - bounds[0])) / 2.,
        (h as f64 - (bounds[3] - bounds[1])) / 2.,
    ];
    let (positions, _) = multiline(&face, &lines, spacing, origin)?;
    for (line, pos) in lines.iter().zip(positions) {
        text_render::draw_single(
            &mut image,
            pos,
            line,
            &face,
            Color {
                red: 90,
                green: 90,
                blue: 90,
                alpha: 255,
            },
            "la",
        )?;
    }
    Ok(image)
}
pub fn jpeg(image: &RgbaImage) -> Result<Vec<u8>> {
    let [width, height] = image.size();
    let mut encoder = turbojpeg::Compressor::new().map_err(Error::malformed)?;
    encoder.set_quality(75).map_err(Error::malformed)?;
    encoder
        .set_subsamp(turbojpeg::Subsamp::Sub2x2)
        .map_err(Error::malformed)?;
    encoder
        .compress_to_vec(turbojpeg::Image {
            pixels: image.pixels(),
            width,
            height,
            pitch: width * 4,
            format: turbojpeg::PixelFormat::RGBA,
        })
        .map_err(Error::malformed)
}
pub fn standard_vector_placeholder() -> Result<Arc<str>> {
    static VALUE: OnceLock<Arc<str>> = OnceLock::new();
    if let Some(value) = VALUE.get() {
        return Ok(Arc::clone(value));
    }
    let image = text_placeholder(
        [320, 180],
        &[
            "WMF/EMF placeholder",
            "Metafile preview unavailable",
            "See the original document",
        ],
    )?;
    let value: Arc<str> = Arc::from(format!(
        "data:image/jpeg;base64,{}",
        base64::encode(jpeg(&image)?)
    ));
    let _ = VALUE.set(Arc::clone(&value));
    Ok(Arc::clone(VALUE.get().unwrap()))
}
/// Fixed safe dimensions; never inspect rejected image geometry.
pub fn image_limit_placeholder() -> Result<Arc<str>> {
    static VALUE: OnceLock<Arc<str>> = OnceLock::new();
    if let Some(value) = VALUE.get() {
        return Ok(Arc::clone(value));
    }
    let image = text_placeholder(
        [320, 180],
        &[
            "Image omitted",
            "Safety limit exceeded",
            "See the original document",
        ],
    )?;
    let value: Arc<str> = Arc::from(format!(
        "data:image/jpeg;base64,{}",
        base64::encode(jpeg(&image)?)
    ));
    let _ = VALUE.set(value);
    Ok(Arc::clone(VALUE.get().unwrap()))
}
pub struct VectorImage {
    pub source: Arc<str>,
    pub rendered: Option<metafile::Rendered>,
    pub failure: Option<Error>,
}
/// Preserve success/partial diagnostics, but degrade metafile errors only at
/// the Office serialization boundary, just as the Python caller does.
pub fn vector(data: &[u8], size_emu: Option<[i64; 2]>) -> Result<VectorImage> {
    vector_with_decoder(
        data,
        size_emu,
        &mut crate::equation::image_mtef::Decoder::default(),
    )
}
pub fn vector_with_decoder(
    data: &[u8],
    size_emu: Option<[i64; 2]>,
    probe: &mut crate::equation::image_mtef::Decoder,
) -> Result<VectorImage> {
    probe.probe(data, None, None)?;
    if let Some(reason) = probe.limit_reason(data, None, None) {
        return Ok(VectorImage {
            source: image_limit_placeholder()?,
            rendered: None,
            failure: Some(Error::new(crate::ErrorKind::ResourceLimit, reason)),
        });
    }
    vector_unchecked(data, size_emu)
}
fn vector_unchecked(data: &[u8], size_emu: Option<[i64; 2]>) -> Result<VectorImage> {
    let result = if is_wmf(data) || is_emf(data) {
        metafile::office_image(data, size_from_emu(size_emu, 200))
    } else {
        Err(Error::malformed(
            "vector payload does not match WMF/EMF signature",
        ))
    };
    match result {
        Ok(rendered) => {
            let source = Arc::from(format!(
                "data:{};base64,{}",
                rendered.media_type,
                base64::encode(&rendered.data)
            ));
            Ok(VectorImage {
                source,
                rendered: Some(rendered),
                failure: None,
            })
        }
        Err(error) => Ok(VectorImage {
            source: standard_vector_placeholder()?,
            rendered: None,
            failure: Some(error),
        }),
    }
}
