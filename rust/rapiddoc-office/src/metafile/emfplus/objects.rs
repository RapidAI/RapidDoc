// metafile-render 0.3.0 emfplus/objects.py (MIT, Xiaomeng Zhao).
use super::super::{geometry::Builder, malformed, model::*};
use super::{
    cursor::{argb, Cursor},
    image,
};
use crate::Result;
use std::sync::Arc;
#[derive(Clone)]
pub enum Object {
    Brush(Option<Color>),
    Pen {
        pen: Pen,
        unit: u32,
        miter: f64,
    },
    Path(Path),
    Image {
        png: Arc<[u8]>,
        width: usize,
        height: usize,
    },
    Font {
        font: Arc<Font>,
        unit: u32,
    },
    Format {
        flags: u32,
        alignment: u32,
        vertical: u32,
    },
    Attributes,
    Unsupported,
}
pub fn brush(c: &mut Cursor<'_>, warn: &mut impl FnMut(&str, &str)) -> Result<Option<Color>> {
    c.version()?;
    let kind = c.u32()?;
    let color = match kind {
        0 => return Ok(Some(argb(c.u32()?))),
        1 => {
            c.u32()?;
            let color = argb(c.u32()?);
            c.u32()?;
            color
        }
        4 => {
            c.u32()?;
            c.u32()?;
            c.rect(false)?;
            let color = argb(c.u32()?);
            c.take(12)?;
            color
        }
        3 => {
            c.u32()?;
            c.u32()?;
            let color = argb(c.u32()?);
            c.point(false)?;
            let n = c.count()?;
            c.take(n * 4)?;
            color
        }
        _ => {
            warn(
                "emfplus_unsupported_brush",
                &format!(
                    "brush type {kind} has no supported representative color; drawing skipped"
                ),
            );
            return Ok(None);
        }
    };
    warn(
        "emfplus_brush_approximation",
        &format!("brush type {kind} uses a representative solid color"),
    );
    Ok(Some(color))
}
fn path(c: &mut Cursor<'_>) -> Result<Path> {
    let n = c.count()?;
    let flags = c.u32()?;
    let points = c.points(n, flags as u16)?;
    let mut types = Vec::with_capacity(n);
    if flags & 0x800 != 0 {
        while types.len() < n {
            let run = c.u8()?;
            let kind = c.u8()?;
            let count = (run & 63) as usize;
            if count == 0 || types.len() + count > n || run & 64 == 0 {
                return Err(malformed());
            }
            types.extend(std::iter::repeat_n(kind, count));
        }
    } else {
        types.extend_from_slice(c.take(n)?);
    }
    let mut b = Builder::default();
    let mut i = 0;
    while i < n {
        let kind = types[i] & 7;
        let mut close = types[i] & 128 != 0;
        match kind {
            0 => b.move_to(points[i]),
            1 if b.current.is_some() => b.line_to(points[i]),
            3 if b.current.is_some() => {
                if i + 2 >= n || types[i..i + 3].iter().any(|v| v & 7 != 3) {
                    return Err(malformed());
                }
                b.cubic_to(points[i], points[i + 1], points[i + 2]);
                close = types[i + 2] & 128 != 0;
                i += 2;
            }
            _ => return Err(malformed()),
        };
        if close {
            b.close();
        }
        i += 1;
    }
    c.finish()?;
    Ok(b.take())
}
fn pen(c: &mut Cursor<'_>, warn: &mut impl FnMut(&str, &str)) -> Result<Object> {
    if c.u32()? != 0 {
        return Err(malformed());
    }
    let flags = c.u32()?;
    let unit = c.u32()?;
    let width = c.f32()?;
    if flags & !0x1fff != 0 || width < 0. || unit > 6 {
        return Err(malformed());
    }
    let (mut start, mut end, mut join, mut style) = (0, 0, 0, 0);
    let mut miter = 10.;
    let mut dashes = vec![];
    if flags & 1 != 0 {
        c.matrix()?;
        warn(
            "emfplus_pen_approximation",
            "pen-specific transform ignored",
        );
    }
    if flags & 2 != 0 {
        start = c.u32()?;
    }
    if flags & 4 != 0 {
        end = c.u32()?;
    }
    if flags & 8 != 0 {
        join = c.u32()?;
    }
    if flags & 16 != 0 {
        miter = c.f32()?.max(1.);
    }
    if flags & 32 != 0 {
        style = c.u32()?;
    }
    if flags & 64 != 0 && c.u32()? != 0 {
        warn(
            "emfplus_pen_approximation",
            "dash caps use the standard stroke caps",
        );
    }
    if flags & 128 != 0 && c.f32()? != 0. {
        warn("emfplus_pen_approximation", "dash offset ignored");
    }
    if flags & 256 != 0 {
        let n = c.count()?;
        dashes = (0..n).map(|_| c.f32()).collect::<Result<Vec<_>>>()?;
        if dashes.iter().any(|&v| v <= 0.) {
            return Err(malformed());
        }
    }
    if flags & 512 != 0 && c.u32()? != 0 {
        warn(
            "emfplus_pen_approximation",
            "pen alignment uses centered strokes",
        );
    }
    if flags & 1024 != 0 {
        let n = c.count()?;
        c.take(n * 4)?;
        warn(
            "emfplus_pen_approximation",
            "compound pen uses a single stroke",
        );
    }
    for f in [2048, 4096] {
        if flags & f != 0 {
            let n = c.u32()? as usize;
            c.take(n)?;
            warn("emfplus_pen_approximation", "custom line cap ignored");
        }
    }
    let color = brush(c, warn)?;
    if start != end || start > 2 || join > 2 {
        warn(
            "emfplus_pen_approximation",
            "unsupported or asymmetric caps/joins use standard strokes",
        );
    }
    if dashes.is_empty() {
        dashes = match style {
            0 => vec![],
            1 => vec![3., 1.],
            2 => vec![1., 1.],
            3 => vec![3., 1., 1., 1.],
            4 => vec![3., 1., 1., 1., 1., 1.],
            _ => {
                warn(
                    "emfplus_pen_approximation",
                    "unknown dash style uses a solid stroke",
                );
                vec![]
            }
        };
    }
    Ok(Object::Pen {
        pen: Pen {
            color: color.unwrap_or(Color {
                red: 0,
                green: 0,
                blue: 0,
                alpha: 0,
            }),
            width,
            cosmetic: false,
            null: color.is_none(),
            cap: match start {
                1 => "square",
                2 => "round",
                _ => "flat",
            }
            .into(),
            join: match join {
                1 => "bevel",
                2 => "round",
                _ => "miter",
            }
            .into(),
            dashes: dashes.into(),
            ..Pen::default()
        },
        unit,
        miter,
    })
}
pub fn parse(kind: u8, c: &mut Cursor<'_>, warn: &mut impl FnMut(&str, &str)) -> Result<Object> {
    if kind == 1 {
        return brush(c, warn).map(Object::Brush);
    }
    c.version()?;
    match kind {
        2 => pen(c, warn),
        3 => path(c).map(Object::Path),
        5 => match image::decode(c, warn)? {
            Some(image) => {
                let [width, height] = image.size();
                Ok(Object::Image {
                    png: image::normalize(&image)?.into(),
                    width,
                    height,
                })
            }
            None => Ok(Object::Unsupported),
        },
        6 => {
            let size = c.f32()?;
            let unit = c.u32()?;
            let style = c.u32()?;
            c.u32()?;
            let n = c.count()?;
            let name = c.text(n)?.trim_end_matches('\0').to_string();
            if size <= 0. || unit > 6 {
                return Err(malformed());
            }
            c.finish()?;
            Ok(Object::Font {
                font: Arc::new(Font {
                    face_name: name,
                    height: -size,
                    weight: if style & 1 != 0 { 700 } else { 400 },
                    italic: style & 2 != 0,
                    underline: style & 4 != 0,
                    strikeout: style & 8 != 0,
                    ..Font::default()
                }),
                unit,
            })
        }
        7 => {
            let flags = c.u32()?;
            c.u32()?;
            let alignment = c.u32()?;
            let vertical = c.u32()?;
            c.take(12)?;
            let hotkey = c.u32()?;
            c.f32()?;
            c.f32()?;
            let tracking = c.f32()?;
            let trimming = c.u32()?;
            if hotkey != 0 || (tracking - 1.).abs() > 1e-6 || trimming != 0 {
                warn(
                    "emfplus_text_approximation",
                    "advanced string formatting uses basic alignment and clipping",
                );
            }
            let tabs = c.count()?;
            let ranges = c.count()?;
            c.take(tabs * 4 + ranges * 8)?;
            c.finish()?;
            if alignment > 2 || vertical > 2 {
                return Err(malformed());
            }
            if tabs != 0 || ranges != 0 {
                warn(
                    "emfplus_text_approximation",
                    "tab stops and character ranges use basic text layout",
                );
            }
            Ok(Object::Format {
                flags,
                alignment,
                vertical,
            })
        }
        8 => {
            c.u32()?;
            c.u32()?;
            c.take(12)?;
            c.finish()?;
            Ok(Object::Attributes)
        }
        _ => {
            warn(
                "emfplus_unsupported_object",
                &format!("object type {kind} is not implemented"),
            );
            Ok(Object::Unsupported)
        }
    }
}
