//! Port of metafile-render 0.3.0 replay semantics. File/geometry/GDI layers are
//! implemented first, with mixed path/bitmap/clip single-pass rasterization.
//! GDI and EMF+ Only share typed paths, bitmap/text rasterization and PNG/SVG.
//! Unsupported source operations retain the original diagnostics and partial flag.
pub mod binary;
pub mod bitmap;
pub mod clipper;
pub mod composite;
pub mod dib_decode;
pub mod emf;
pub mod emfplus;
pub mod filters;
pub mod fonts;
pub mod geometry;
pub mod header;
pub mod image_session;
pub mod mask;
pub mod model;
pub mod output;
pub mod paths;
pub mod playback;
pub mod raster;
pub mod sampling;
pub mod text;
pub mod text_render;
pub mod wmf;
use crate::{Error, ErrorKind};
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    Svg,
    Png,
}
pub struct Rendered {
    pub data: Vec<u8>,
    pub media_type: &'static str,
    pub width: u64,
    pub height: u64,
    pub partial: bool,
    pub diagnostics: Vec<model::Diagnostic>,
}
/// Native shared Office image entry, including EMF+ Only replay.
pub fn render(
    data: &[u8],
    format: OutputFormat,
    dpi: u32,
    size: Option<[u64; 2]>,
) -> crate::Result<Rendered> {
    let h = header::prepare(data, dpi, size)?;
    let doc = match h.source_format {
        header::Source::Wmf => wmf::replay(data, h, size),
        header::Source::Emf => emf::replay(data, h, size),
    }?;
    let mut session = image_session::ImageSession::default();
    let mut budget = geometry::FlattenBudget::default();
    let (data, media_type) = match format {
        OutputFormat::Svg => (
            output::svg(&doc, &mut budget, &mut session)?,
            "image/svg+xml",
        ),
        OutputFormat::Png => (
            output::png_dpi(&raster::render(&doc, &mut budget, &mut session)?, Some(dpi))?,
            "image/png",
        ),
    };
    let mut diagnostics = doc.diagnostics;
    let mut omitted = session.font_diagnostics_omitted;
    if diagnostics
        .last()
        .is_some_and(|d| d.code == "diagnostics_truncated")
    {
        let d = diagnostics.pop().unwrap();
        omitted += d
            .message
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
    }
    for d in session.font_diagnostics {
        if diagnostics.len() < 256 {
            diagnostics.push(d);
        } else {
            omitted += 1;
        }
    }
    if omitted > 0 {
        diagnostics.push(model::Diagnostic {
            code: "diagnostics_truncated".into(),
            level: "warning".into(),
            message: format!("omitted {omitted} additional metafile diagnostics"),
            record_type: None,
            record_index: None,
            offset: None,
        });
    }
    Ok(Rendered {
        data,
        media_type,
        width: doc.width,
        height: doc.height,
        partial: doc.partial,
        diagnostics,
    })
}
/// Original Office serializer prefers SVG, retries PNG only on resource errors.
/// A new Unsupported result is propagated rather than disguised as a placeholder.
pub fn office_image(data: &[u8], size: Option<[u64; 2]>) -> crate::Result<Rendered> {
    match render(data, OutputFormat::Svg, 200, size) {
        Err(e) if e.kind == ErrorKind::ResourceLimit => render(data, OutputFormat::Png, 200, size),
        result => result,
    }
}
fn malformed() -> Error {
    Error::new(
        ErrorKind::Malformed,
        "metafile field or record exceeds its boundary",
    )
}
fn resource(message: &str) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}
