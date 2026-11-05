//! WMF state/object/path/DIB records. Text drawing stays an explicit pending stage;
//! this internal geometry reader must not be advertised as the full renderer.
use super::{
    binary::{Reader, Records},
    geometry::{self, Builder},
    header::{Header, Source},
    malformed,
    model::*,
    playback::{self, GeometryDocument, Playback},
};
use crate::Result;
use std::sync::Arc;
pub fn replay(data: &[u8], header: Header, size: Option<[u64; 2]>) -> Result<GeometryDocument> {
    if header.source_format != Source::Wmf {
        return Err(malformed());
    }
    let mut p = Playback::new(header.clone());
    for record in Records::new(data, header.record_start, header.record_end, Source::Wmf)? {
        let record = record?;
        p.set_record(record.kind, record.index, record.offset);
        handle(
            record.kind,
            record.data.sub(6, record.data.data.len() - 6)?,
            &mut p,
        )?;
    }
    p.finish(size)
}
fn point(r: Reader<'_>, pos: usize) -> Result<Point> {
    Ok([r.i16(pos + 2)? as f64, r.i16(pos)? as f64])
}
fn rect(r: Reader<'_>, pos: usize) -> Result<Rect> {
    Ok(Rect {
        left: r.i16(pos + 6)? as f64,
        top: r.i16(pos + 4)? as f64,
        right: r.i16(pos + 2)? as f64,
        bottom: r.i16(pos)? as f64,
    })
}
fn points(r: Reader<'_>, count: usize, pos: usize, p: &mut Playback) -> Result<Vec<Point>> {
    r.bytes(pos, count.checked_mul(4).ok_or_else(malformed)?)?;
    (0..count)
        .map(|i| p.map_point([r.i16(pos + i * 4)? as f64, r.i16(pos + i * 4 + 2)? as f64]))
        .collect()
}
pub fn handle(kind: u32, r: Reader<'_>, p: &mut Playback) -> Result<()> {
    match kind {
        0 => (),
        0x1e => p.save_dc()?,
        0x127 => p.restore_dc(r.i16(0)?.into()),
        0x201 => p.state.background_color = Color::colorref(r.u32(0)?),
        0x209 => p.state.text_color = Color::colorref(r.u32(0)?),
        0x102 => p.state.background_mode = r.u16(0)?.into(),
        0x103 => p.state.map_mode = r.i16(0)?.into(),
        0x104 => {
            let v = u32::from(r.u16(0)?);
            p.state.rop2 = if (1..=16).contains(&v) {
                v
            } else {
                p.warn(
                    "unsupported_rop2",
                    &format!("invalid WMF ROP2 will use COPYPEN approximation: {v}"),
                    true,
                );
                13
            };
        }
        0x106 => p.state.polygon_fill_mode = r.u16(0)?.into(),
        0x107 => p.state.stretch_mode = r.u16(0)?.into(),
        0x12e => p.state.text_align = r.u16(0)?.into(),
        0x20b => p.state.window_origin = point(r, 0)?,
        0x20c => p.state.window_extent = point(r, 0)?,
        0x20d => p.state.viewport_origin = point(r, 0)?,
        0x20e => p.state.viewport_extent = point(r, 0)?,
        0x20f | 0x211 => {
            let v = point(r, 0)?;
            let origin = if kind == 0x20f {
                &mut p.state.window_origin
            } else {
                &mut p.state.viewport_origin
            };
            origin[0] += v[0];
            origin[1] += v[1];
        }
        0x410 | 0x412 => {
            let yn = r.i16(0)?;
            let yd = r.i16(2)?;
            let xn = r.i16(4)?;
            let xd = r.i16(6)?;
            if xd == 0 || yd == 0 {
                p.warn(
                    "zero_scale_denominator",
                    "WMF viewport/window scale record has a zero denominator",
                    true,
                );
            } else {
                let extent = if kind == 0x410 {
                    &mut p.state.window_extent
                } else {
                    &mut p.state.viewport_extent
                };
                extent[0] = extent[0] * f64::from(xn) / f64::from(xd);
                extent[1] = extent[1] * f64::from(yn) / f64::from(yd);
            }
        }
        0x12d => p.select_wmf(r.u16(0)?.into()),
        0x1f0 => {
            let i = usize::from(r.u16(0)?);
            if i < p.wmf_objects.len() {
                p.wmf_objects[i] = None;
            }
        }
        0x2fa => p.create_wmf_object(Object::Pen(Arc::new(playback::make_pen(
            r.u16(0)?.into(),
            r.i16(2)? as f64,
            r.u32(6)?,
            false,
        ))))?,
        0x2fc => p.create_wmf_object(Object::Brush(Arc::new(playback::make_brush(
            r.u16(0)?.into(),
            r.u32(2)?,
            r.u16(6)?.into(),
        ))))?,
        0x2fb => {
            r.bytes(0, 18)?;
            let name = r.bytes(18, r.data.len().saturating_sub(18).min(32))?;
            let (name, _, _) = encoding_rs::WINDOWS_1252.decode(name);
            let name = name.replace(['\u{81}', '\u{8d}', '\u{8f}', '\u{90}', '\u{9d}'], "�");
            let name = name
                .split('\0')
                .next()
                .unwrap_or("")
                .trim_matches(|c: char| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}'));
            let font = Font {
                face_name: if name.is_empty() { "Arial" } else { name }.into(),
                height: r.i16(0)? as f64,
                width: r.i16(2)? as f64,
                escapement: r.i16(4)? as f64,
                orientation: r.i16(6)? as f64,
                weight: r.i16(8)?.into(),
                italic: r.u8(10)? != 0,
                underline: r.u8(11)? != 0,
                strikeout: r.u8(12)? != 0,
                charset: r.u8(13)?,
            };
            p.create_wmf_object(Object::Font(Arc::new(font)))?;
        }
        0x214 => p.move_to(point(r, 0)?)?,
        0x213 => p.line_to(point(r, 0)?)?,
        0x324 | 0x325 => {
            let count = usize::from(r.u16(0)?);
            p.charge_points(count)?;
            let points = points(r, count, 2, p)?;
            p.emit(
                geometry::polygon(&points, kind == 0x324),
                true,
                kind == 0x324,
            )?;
        }
        0x538 => {
            let count = usize::from(r.u16(0)?);
            if count > 1_000_000 || count > r.remaining(2)? / 2 {
                return Err(malformed());
            }
            let counts = (0..count)
                .map(|i| r.u16(2 + i * 2).map(usize::from))
                .collect::<Result<Vec<_>>>()?;
            let total = counts.iter().sum();
            p.charge_points(total)?;
            let points = points(r, total, 2 + count * 2, p)?;
            let mut b = Builder::default();
            let mut start = 0;
            for count in counts {
                b.extend(&geometry::polygon(&points[start..start + count], true));
                start += count;
            }
            p.emit(b.take(), true, true)?;
        }
        0x41b | 0x418 => {
            let rect = rect(r, 0)?;
            p.emit_logical(
                if kind == 0x41b {
                    geometry::rectangle(rect)
                } else {
                    geometry::ellipse(rect)
                },
                true,
                true,
            )?;
        }
        0x61c => p.emit_logical(
            geometry::round_rectangle(
                rect(r, 4)?,
                f64::from(r.i16(2)?).abs() / 2.,
                f64::from(r.i16(0)?).abs() / 2.,
            ),
            true,
            true,
        )?,
        0x817 | 0x81a | 0x830 => {
            let end = point(r, 0)?;
            let start = point(r, 4)?;
            let rect = rect(r, 8)?;
            let close = match kind {
                0x81a => "pie",
                0x830 => "chord",
                _ => "open",
            };
            p.emit_logical(
                geometry::arc(rect, start, end, 2, close),
                true,
                close != "open",
            )?;
        }
        0x415 | 0x416 => {
            let path = geometry::transform(&geometry::rectangle(rect(r, 0)?), p.logical_matrix()?);
            p.add_clip(path, if kind == 0x415 { 4 } else { 1 })?;
        }
        0x41f => {
            let point = p.map_point(point(r, 4)?)?;
            p.append(PathCommand {
                path: geometry::rectangle(Rect {
                    left: point[0],
                    top: point[1],
                    right: point[0] + 1.,
                    bottom: point[1] + 1.,
                }),
                pen: Pen {
                    null: true,
                    ..Pen::default()
                },
                brush: Arc::new(Brush {
                    color: Color::colorref(r.u32(0)?),
                    ..Brush::default()
                }),
                stroke: false,
                fill: true,
                fill_rule: "evenodd".into(),
                clip: Arc::clone(&p.state.clip),
                rop2: 13,
                miter_limit: 10.,
                location: Location::default(),
            })?;
        }
        0xf43 => super::bitmap::wmf(r, p)?,
        0x521 | 0xa32 => super::text::wmf(kind, r, p)?,
        0x940 | 0xb41 => p.warn(
            "unsupported_wmf_dib_record",
            &format!("WMF DIB record will be skipped: {kind:#x}"),
            true,
        ),
        0x108 | 0x20a | 0x231 | 0x626 | 0x234 | 0x35 | 0x436 | 0x37 | 0x139 => p.warn(
            "ignored_wmf_state_record",
            &format!("WMF state/control record was ignored: {kind:#x}"),
            false,
        ),
        _ => p.warn(
            "unsupported_wmf_record",
            &format!("unsupported WMF record was skipped: {kind:#x}"),
            true,
        ),
    }
    Ok(())
}
