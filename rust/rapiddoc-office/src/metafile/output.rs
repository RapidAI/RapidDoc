//! Native PNG/SVG encoding of replayed commands; original metafile-render
//! 0.3.0 structure, clipping and raster fallback rules (MIT, Xiaomeng Zhao).
use super::{
    dib_decode::RgbaImage,
    geometry::{self, FlattenBudget},
    image_session::ImageSession,
    model::*,
    playback::GeometryDocument,
    raster, resource,
};
use crate::{Error, Result};
const MAX_SVG: usize = 64 * 1024 * 1024;
pub fn png(image: &RgbaImage, scale: Option<u32>) -> Result<Vec<u8>> {
    png_dpi(image, scale.map(|s| s * 96))
}
pub fn png_dpi(image: &RgbaImage, dpi: Option<u32>) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    {
        let mut encoder =
            png::Encoder::new(&mut output, image.size()[0] as u32, image.size()[1] as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        if let Some(dpi) = dpi {
            let per_meter = (dpi as f64 / 0.0254 + 0.5) as u32;
            encoder.set_pixel_dims(Some(png::PixelDimensions {
                xppu: per_meter,
                yppu: per_meter,
                unit: png::Unit::Meter,
            }));
        }
        let mut writer = encoder.write_header().map_err(Error::malformed)?;
        writer
            .write_image_data(image.pixels())
            .map_err(Error::malformed)?;
    }
    Ok(output)
}
fn check(size: usize) -> Result<()> {
    if size > MAX_SVG {
        Err(resource("generated SVG exceeds Python byte budget"))
    } else {
        Ok(())
    }
}
pub fn number(value: f64) -> String {
    let mut s = format!("{value:.4}");
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    if s == "-0" {
        s = "0".into();
    }
    s
}
fn color(c: Color) -> String {
    format!("#{:02x}{:02x}{:02x}", c.red, c.green, c.blue)
}
pub fn path_data(path: &Path, matrix: Matrix) -> Result<String> {
    let mut out = String::new();
    for segment in geometry::transform(path, matrix).segments.iter() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push(match segment.verb {
            Verb::M => 'M',
            Verb::L => 'L',
            Verb::C => 'C',
            Verb::Z => 'Z',
        });
        for p in &segment.points {
            out.push(' ');
            out.push_str(&number(p[0]));
            out.push(' ');
            out.push_str(&number(p[1]));
        }
        check(out.len())?;
    }
    Ok(out)
}
pub fn pen_attributes(pen: &Pen, matrix: Matrix, miter: f64) -> String {
    if pen.null {
        return "stroke=\"none\"".into();
    }
    let scale = (matrix.a.abs() + matrix.d.abs()) / 2.;
    let width = if pen.cosmetic {
        pen.width
    } else {
        pen.width * scale.max(1e-9)
    };
    let mut out=format!("stroke=\"{}\" stroke-width=\"{}\" stroke-linecap=\"{}\" stroke-linejoin=\"{}\" stroke-miterlimit=\"{}\"",color(pen.color),number(width.max(1.)),pen.cap,pen.join,number(miter.max(1.)));
    if pen.color.alpha != 255 {
        out.push_str(&format!(
            " stroke-opacity=\"{}\"",
            number(pen.color.alpha as f64 / 255.)
        ));
    }
    if !pen.dashes.is_empty() {
        let scale = if pen.cosmetic { 1. } else { scale };
        out.push_str(&format!(
            " stroke-dasharray=\"{}\"",
            pen.dashes
                .iter()
                .map(|v| number(v * scale))
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    out
}
pub fn brush_attributes(brush: &Brush) -> String {
    if brush.kind == "null" {
        return "fill=\"none\"".into();
    }
    let mut out = format!("fill=\"{}\"", color(brush.color));
    if brush.color.alpha != 255 {
        out.push_str(&format!(
            " fill-opacity=\"{}\"",
            number(brush.color.alpha as f64 / 255.)
        ));
    }
    out
}
fn clips(command: &Command) -> &[Clip] {
    match command {
        Command::Path(c) => &c.clip,
        Command::Image(c) => &c.clip,
        Command::Text(c) => &c.clip,
        Command::Clear(c) => &c.clip,
    }
}
pub fn requires_raster(doc: &GeometryDocument) -> bool {
    doc.commands.iter().enumerate().any(|(index, c)| {
        clips(c)
            .iter()
            .any(|c| !matches!(c.mode.as_str(), "copy" | "and"))
            || match c {
                Command::Path(c) => c.rop2 != 13,
                Command::Image(c) => !matches!(c.rop, 0 | 0xcc0020),
                Command::Text(_) => false,
                Command::Clear(c) => index > 0 || !c.clip.is_empty(),
            }
    })
}
fn metadata(encoded: &str) -> String {
    format!("<metadata id=\"metafile-render-raster-fallback\" data-mime=\"image/png\">{encoded}</metadata>")
}
fn root(doc: &GeometryDocument) -> String {
    format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" data-metafile-render=\"wmf-emf\">",doc.width,doc.height,doc.width,doc.height)
}
fn encode(markup: String) -> Result<Vec<u8>> {
    check(markup.len())?;
    Ok(markup.into_bytes())
}
pub fn svg(
    doc: &GeometryDocument,
    budget: &mut FlattenBudget,
    session: &mut ImageSession,
) -> Result<Vec<u8>> {
    if requires_raster(doc) {
        let canvas = raster::render(doc, budget, session)?;
        let encoded = base64::encode(png(&canvas, Some(1))?);
        check(encoded.len() * 2)?;
        return encode(format!("{}{}<image width=\"{}\" height=\"{}\" href=\"data:image/png;base64,{encoded}\"/></svg>",root(doc),metadata(&encoded),doc.width,doc.height));
    }
    let bounds = doc.bounds.normalized();
    let (sx, sy) = (
        doc.width as f64 / bounds.width(),
        doc.height as f64 / bounds.height(),
    );
    let matrix = Matrix {
        a: sx,
        d: sy,
        e: -bounds.left * sx,
        f: -bounds.top * sy,
        ..Matrix::default()
    };
    let mut clip_ids: Vec<&Clip> = vec![];
    let mut definitions = String::new();
    for command in &doc.commands {
        for op in clips(command) {
            if clip_ids.contains(&op) {
                continue;
            }
            clip_ids.push(op);
            definitions.push_str(&format!("<clipPath id=\"metafile-render-clip-{}\"><path d=\"{}\" fill-rule=\"{}\"/></clipPath>",clip_ids.len(),path_data(&op.path,matrix)?,op.fill_rule));
            check(definitions.len())?;
        }
    }
    let fallback_scale = raster::choose_scale(doc, 2);
    let fallback = if fallback_scale == 1 {
        raster::render(doc, budget, session)?
    } else {
        raster::render_once(doc, fallback_scale, budget, session)?
    };
    let encoded = base64::encode(png(&fallback, Some(fallback_scale))?);
    let mut out = root(doc);
    out.push_str(&metadata(&encoded));
    if !definitions.is_empty() {
        out.push_str("<defs>");
        out.push_str(&definitions);
        out.push_str("</defs>");
    }
    check(out.len() + 256)?;
    for command in &doc.commands {
        let mut element = match command {
            Command::Clear(c) => format!(
                "<rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"{}\"{}/>",
                doc.width,
                doc.height,
                color(c.color),
                if c.color.alpha == 255 {
                    String::new()
                } else {
                    format!(" opacity=\"{}\"", number(c.color.alpha as f64 / 255.))
                }
            ),
            Command::Text(c) => super::text_render::svg(c, matrix, session)?,
            Command::Path(c) => format!(
                "<path d=\"{}\" {} {} fill-rule=\"{}\"/>",
                path_data(&c.path, matrix)?,
                if c.stroke {
                    pen_attributes(&c.pen, matrix, c.miter_limit)
                } else {
                    "stroke=\"none\"".into()
                },
                if c.fill {
                    brush_attributes(&c.brush)
                } else {
                    "fill=\"none\"".into()
                },
                c.fill_rule
            ),
            Command::Image(c) => {
                let decoded = session.decode(&c.image)?;
                if let Some(crop) = super::bitmap::crop_source(decoded.size(), c.source) {
                    let [l, t, r, b] = crop.bounds;
                    let (w, h) = (r - l, b - t);
                    let mut pixels = Vec::with_capacity(w * h * 4);
                    for y in 0..h {
                        for x in 0..w {
                            let (x, y) = (
                                if crop.flip_x { r - 1 - x } else { l + x },
                                if crop.flip_y { b - 1 - y } else { t + y },
                            );
                            let mut p: [u8; 4] = decoded.pixels()
                                [(y * decoded.size()[0] + x) * 4..][..4]
                                .try_into()
                                .unwrap();
                            p[3] = (u32::from(p[3]) * u32::from(c.constant_alpha) / 255) as u8;
                            pixels.extend(p);
                        }
                    }
                    let cropped = RgbaImage::new(w, h, pixels)?;
                    let dest = super::bitmap::crop_destination(c.destination, crop.fraction)
                        .map(|p| matrix.point(p));
                    let [first, second, _, fourth] = dest;
                    let values = [
                        (second[0] - first[0]) / w as f64,
                        (second[1] - first[1]) / w as f64,
                        (fourth[0] - first[0]) / h as f64,
                        (fourth[1] - first[1]) / h as f64,
                        first[0],
                        first[1],
                    ];
                    format!("<image width=\"{w}\" height=\"{h}\" transform=\"matrix({})\" href=\"data:image/png;base64,{}\"/>",values.into_iter().map(number).collect::<Vec<_>>().join(" "),base64::encode(png(&cropped,None)?))
                } else {
                    String::new()
                }
            }
        };
        let mut active = vec![];
        for op in clips(command) {
            if op.mode == "copy" {
                active.clear();
                active.push(op);
            } else if op.mode == "and" {
                active.push(op);
            }
        }
        for op in active.iter().rev() {
            let id = clip_ids.iter().position(|c| c == op).unwrap() + 1;
            element = format!("<g clip-path=\"url(#metafile-render-clip-{id})\">{element}</g>");
        }
        check(out.len() + element.len())?;
        out.push_str(&element);
    }
    out.push_str("</svg>");
    encode(out)
}
pub fn png_document(
    doc: &GeometryDocument,
    budget: &mut FlattenBudget,
    session: &mut ImageSession,
) -> Result<Vec<u8>> {
    png(&raster::render(doc, budget, session)?, Some(1))
}
