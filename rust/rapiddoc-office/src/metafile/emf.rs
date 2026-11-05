//! EMF geometry/state/object replay. No replacement raster/font algorithm.
use super::{
    binary::{Reader, Records},
    geometry::{self, Builder},
    header::{Header, PlusMode, Source},
    malformed,
    model::*,
    playback::{self, GeometryDocument, Playback},
    resource,
};
use crate::Result;
use std::sync::Arc;
pub fn replay(data: &[u8], header: Header, size: Option<[u64; 2]>) -> Result<GeometryDocument> {
    if header.source_format != Source::Emf {
        return Err(malformed());
    }
    let mut plus = if header.emfplus_mode == PlusMode::Only {
        Some(super::emfplus::playback::Plus::new(
            header.bounds.unwrap_or(Rect {
                left: 0.,
                top: 0.,
                right: 1.,
                bottom: 1.,
            }),
            header.device_pixels_per_mm.map(|v| v * 25.4),
        ))
    } else {
        None
    };
    let mut p = Playback::new(header.clone());
    for record in Records::new(data, header.record_start, header.record_end, Source::Emf)? {
        let record = record?;
        p.set_record(record.kind, record.index, record.offset);
        if let Some(plus) = &mut plus {
            if record.kind == 70 {
                plus.comment(record.data, &mut p)?;
            } else if plus.gdi_active && !plus.halted && !matches!(record.kind, 1 | 14) {
                handle(record.kind, record.data, &mut p)?;
            }
        } else {
            handle(record.kind, record.data, &mut p)?;
        }
    }
    if let Some(plus) = plus {
        plus.finish(&p)?;
    }
    p.finish(size)
}
fn point(r: Reader<'_>, offset: usize) -> Result<Point> {
    Ok([r.i32(offset)? as f64, r.i32(offset + 4)? as f64])
}
fn read_points(r: Reader<'_>, count: usize, offset: usize, compact: bool) -> Result<Vec<Point>> {
    let stride = if compact { 4 } else { 8 };
    if count > 1_000_000 || count > r.remaining(offset)? / stride {
        return Err(malformed());
    }
    (0..count)
        .map(|i| {
            if compact {
                Ok([
                    r.i16(offset + i * 4)? as f64,
                    r.i16(offset + i * 4 + 2)? as f64,
                ])
            } else {
                point(r, offset + i * 8)
            }
        })
        .collect()
}
fn poly(kind: u32, r: Reader<'_>, p: &mut Playback) -> Result<bool> {
    if !matches!(kind,2..=8|85..=91) {
        return Ok(false);
    }
    let compact = kind >= 85;
    if !matches!(kind, 7 | 8 | 90 | 91) {
        let count = r.u32(24)? as usize;
        p.charge_points(count)?;
        let points = read_points(r, count, 28, compact)?;
        if matches!(kind, 2 | 5 | 85 | 88) {
            if matches!(kind, 5 | 88) {
                p.bezier_to(&points)?;
                return Ok(true);
            }
            if points.is_empty() {
                return Ok(true);
            }
            let mut b = Builder::default();
            b.move_to(p.map_point(points[0])?);
            let rest = &points[1..];
            if rest.len() % 3 != 0 {
                p.warn(
                    "invalid_bezier_points",
                    "PolyBezier point count is not a multiple of three",
                    true,
                );
            }
            for chunk in rest[..rest.len() - rest.len() % 3].chunks_exact(3) {
                b.cubic_to(
                    p.map_point(chunk[0])?,
                    p.map_point(chunk[1])?,
                    p.map_point(chunk[2])?,
                );
            }
            p.emit(b.take(), true, false)?;
            return Ok(true);
        }
        let mapped = points
            .iter()
            .map(|v| p.map_point(*v))
            .collect::<Result<Vec<_>>>()?;
        if matches!(kind, 6 | 89) {
            p.polyline_to(&points)?;
        } else {
            let closed = matches!(kind, 3 | 86);
            p.emit(geometry::polygon(&mapped, closed), true, closed)?;
        }
        return Ok(true);
    }
    let count = r.u32(24)? as usize;
    let total = r.u32(28)? as usize;
    if count > 1_000_000 || count > r.remaining(32)? / 4 {
        return Err(malformed());
    }
    let counts = (0..count)
        .map(|i| r.u32(32 + i * 4).map(|v| v as usize))
        .collect::<Result<Vec<_>>>()?;
    if counts.iter().sum::<usize>() != total {
        return Err(malformed());
    }
    p.charge_points(total)?;
    let points = read_points(r, total, 32 + count * 4, compact)?;
    let mut b = Builder::default();
    let mut cursor = 0;
    let closed = matches!(kind, 8 | 91);
    for count in counts {
        let mapped = points[cursor..cursor + count]
            .iter()
            .map(|v| p.map_point(*v))
            .collect::<Result<Vec<_>>>()?;
        b.extend(&geometry::polygon(&mapped, closed));
        cursor += count;
    }
    p.emit(b.take(), true, closed)?;
    Ok(true)
}
fn validate_handle(handle: u32) -> Result<()> {
    if handle == 0 || handle & 0x80000000 != 0 || handle > 65536 {
        Err(resource("invalid or excessive EMF object handle"))
    } else {
        Ok(())
    }
}
pub fn handle(kind: u32, r: Reader<'_>, p: &mut Playback) -> Result<()> {
    if matches!(kind, 1 | 14 | 70) {
        return Ok(());
    }
    if poly(kind, r, p)? {
        return Ok(());
    }
    match kind {
        9 => p.state.window_extent = point(r, 8)?,
        10 => p.state.window_origin = point(r, 8)?,
        11 => p.state.viewport_extent = point(r, 8)?,
        12 => p.state.viewport_origin = point(r, 8)?,
        13 => p.state.brush_origin = point(r, 8)?,
        17 => p.state.map_mode = r.i32(8)?,
        18 => p.state.background_mode = r.u32(8)?,
        19 => p.state.polygon_fill_mode = r.u32(8)?,
        21 => p.state.stretch_mode = r.u32(8)?,
        22 => p.state.text_align = r.u32(8)?,
        24 => p.state.text_color = Color::colorref(r.u32(8)?),
        25 => p.state.background_color = Color::colorref(r.u32(8)?),
        20 => {
            let value = r.u32(8)?;
            p.state.rop2 = if (1..=16).contains(&value) {
                value
            } else {
                p.warn(
                    "unsupported_rop2",
                    &format!("invalid ROP2 will use COPYPEN approximation: {value}"),
                    true,
                );
                13
            };
        }
        31 | 32 => {
            let xn = r.i32(8)?;
            let xd = r.i32(12)?;
            let yn = r.i32(16)?;
            let yd = r.i32(20)?;
            if xd == 0 || yd == 0 {
                p.warn(
                    "zero_scale_denominator",
                    "viewport/window scale record has a zero denominator",
                    true,
                );
            } else {
                let extent = if kind == 31 {
                    &mut p.state.viewport_extent
                } else {
                    &mut p.state.window_extent
                };
                extent[0] = extent[0] * f64::from(xn) / f64::from(xd);
                extent[1] = extent[1] * f64::from(yn) / f64::from(yd);
            }
        }
        33 => p.save_dc()?,
        34 => p.restore_dc(r.i32(8)?),
        35 => p.state.world_transform = r.matrix(8)?,
        36 => {
            let m = r.matrix(8)?;
            let mode = r.u32(32)?;
            match mode {
                1 => p.state.world_transform = Matrix::default(),
                2 => p.state.world_transform = p.state.world_transform.then(m),
                3 => p.state.world_transform = m.then(p.state.world_transform),
                4 => p.state.world_transform = m,
                _ => p.warn(
                    "unsupported_transform_mode",
                    &format!("unsupported ModifyWorldTransform mode: {mode}"),
                    true,
                ),
            }
        }
        57 => p.state.arc_direction = r.u32(8)?,
        58 => {
            let v = r.f32(8)?;
            if v.is_finite() && v > 0. {
                p.state.miter_limit = v;
            } else {
                p.warn(
                    "invalid_miter_limit",
                    &format!("invalid miter limit: {v:?}").replace("NaN", "nan"),
                    true,
                );
            }
        }
        26 => {
            let offset = p.map_vector(point(r, 8)?)?;
            let m = Matrix {
                e: offset[0],
                f: offset[1],
                ..Matrix::default()
            };
            p.state.clip = p
                .state
                .clip
                .iter()
                .map(|op| Clip {
                    path: geometry::transform(&op.path, m),
                    ..op.clone()
                })
                .collect::<Vec<_>>()
                .into();
        }
        28 => p.reset_clip(),
        29 | 30 => {
            let path = geometry::transform(&geometry::rectangle(r.rect32(8)?), p.logical_matrix()?);
            p.add_clip(path, if kind == 29 { 4 } else { 1 })?;
        }
        67 => {
            let path = p.consume_path();
            if !path.segments.is_empty() {
                p.add_clip(path, r.u32(8)?)?;
            }
        }
        75 => {
            let size = r.u32(8)? as usize;
            let mode = r.u32(12)?;
            if size == 0 {
                if mode == 5 {
                    p.reset_clip();
                } else {
                    p.warn(
                        "empty_clip_region",
                        &format!("empty clip region used with combine mode {mode}"),
                        true,
                    );
                }
                return Ok(());
            }
            if size > r.remaining(16)? || size < 32 {
                return Err(malformed());
            }
            let region = r.sub(16, size)?;
            let header = region.u32(0)? as usize;
            let count = region.u32(8)? as usize;
            let bytes = region.u32(12)? as usize;
            if header < 32
                || region.u32(4)? != 1
                || count > 1_000_000
                || bytes > region.remaining(header)?
                || count > region.remaining(header)? / 16
            {
                return Err(malformed());
            }
            p.charge_points(count * 4)?;
            let mut b = Builder::default();
            for i in 0..count {
                b.extend(&geometry::transform(
                    &geometry::rectangle(region.rect32(header + i * 16)?),
                    p.logical_matrix()?,
                ));
            }
            let path = b.take();
            if !path.segments.is_empty() {
                p.add_clip(path, mode)?;
            }
        }
        27 => p.move_to(point(r, 8)?)?,
        54 => p.line_to(point(r, 8)?)?,
        15 => {
            let v = p.map_point(point(r, 8)?)?;
            p.append(PathCommand {
                path: geometry::rectangle(Rect {
                    left: v[0],
                    top: v[1],
                    right: v[0] + 1.,
                    bottom: v[1] + 1.,
                }),
                pen: Pen {
                    null: true,
                    ..Pen::default()
                },
                brush: Arc::new(Brush {
                    color: Color::colorref(r.u32(16)?),
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
        42..=44 => {
            let rect = r.rect32(8)?;
            let path = match kind {
                42 => geometry::ellipse(rect),
                44 => geometry::round_rectangle(
                    rect,
                    f64::from(r.i32(24)?).abs() / 2.,
                    f64::from(r.i32(28)?).abs() / 2.,
                ),
                _ => geometry::rectangle(rect),
            };
            p.emit_logical(path, true, true)?;
        }
        45..=47 | 55 => {
            let close = match kind {
                46 => "chord",
                47 => "pie",
                _ => "open",
            };
            let path = geometry::arc(
                r.rect32(8)?,
                point(r, 24)?,
                point(r, 32)?,
                p.state.arc_direction,
                close,
            );
            if kind == 55 {
                p.connected_arc(&path)?;
            } else {
                p.emit_logical(path, true, close != "open")?;
            }
        }
        41 => {
            let c = point(r, 8)?;
            let radius = f64::from(r.u32(16)?);
            let start = r.f32(20)?;
            let sweep = r.f32(24)?;
            if !start.is_finite() || !sweep.is_finite() {
                return Err(malformed());
            }
            // Preserve math.radians' multiplication order for large f32 angles.
            let radians = std::f64::consts::PI / 180.;
            let angle = start * radians;
            let end = (start + sweep) * radians;
            let path = geometry::arc(
                Rect {
                    left: c[0] - radius,
                    top: c[1] - radius,
                    right: c[0] + radius,
                    bottom: c[1] + radius,
                },
                [c[0] + radius * angle.cos(), c[1] - radius * angle.sin()],
                [c[0] + radius * end.cos(), c[1] - radius * end.sin()],
                if sweep >= 0. { 1 } else { 2 },
                "open",
            );
            p.connected_arc(&path)?;
        }
        37 => p.select_emf(r.u32(8)?),
        38 => {
            let h = r.u32(8)?;
            validate_handle(h)?;
            p.emf_objects.insert(
                h,
                Object::Pen(Arc::new(playback::make_pen(
                    r.u32(12)?,
                    r.i32(16)? as f64,
                    r.u32(24)?,
                    false,
                ))),
            );
        }
        39 => {
            let h = r.u32(8)?;
            validate_handle(h)?;
            p.emf_objects.insert(
                h,
                Object::Brush(Arc::new(playback::make_brush(
                    r.u32(12)?,
                    r.u32(16)?,
                    r.u32(20)?,
                ))),
            );
        }
        40 => {
            p.emf_objects.remove(&r.u32(8)?);
        }
        82 => {
            let h = r.u32(8)?;
            validate_handle(h)?;
            let text = String::from_utf16_lossy(
                &r.bytes(40, 64)?
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            );
            let text = text
                .split('\0')
                .next()
                .unwrap_or("")
                .trim_matches(|c: char| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}'));
            p.emf_objects.insert(
                h,
                Object::Font(Arc::new(Font {
                    face_name: if text.is_empty() { "Arial" } else { text }.into(),
                    height: r.i32(12)? as f64,
                    width: r.i32(16)? as f64,
                    escapement: r.i32(20)? as f64,
                    orientation: r.i32(24)? as f64,
                    weight: r.i32(28)?,
                    italic: r.u8(32)? != 0,
                    underline: r.u8(33)? != 0,
                    strikeout: r.u8(34)? != 0,
                    charset: r.u8(35)?,
                })),
            );
        }
        94 => {
            let h = r.u32(8)?;
            validate_handle(h)?;
            let bmi = r.u32(20)? as usize;
            let bits = r.u32(28)? as usize;
            let pattern = if bmi + bits <= 128 * 1024 * 1024 {
                [
                    r.bytes(r.u32(16)? as usize, bmi)?,
                    r.bytes(r.u32(24)? as usize, bits)?,
                ]
                .concat()
            } else {
                vec![]
            };
            p.emf_objects.insert(
                h,
                Object::Brush(Arc::new(Brush {
                    kind: "pattern".into(),
                    pattern: Some(pattern.into()),
                    ..Brush::default()
                })),
            );
            p.warn(
                "pattern_brush_approximation",
                "DIB pattern brush will use a solid-color approximation",
                true,
            );
        }
        95 => {
            let h = r.u32(8)?;
            validate_handle(h)?;
            let mut pen = playback::make_pen(r.u32(28)?, f64::from(r.u32(32)?), r.u32(40)?, true);
            let count = r.u32(48)? as usize;
            if count != 0 && count <= r.remaining(52)? / 4 {
                pen.dashes = (0..count)
                    .map(|i| r.u32(52 + i * 4).map(|n| f64::from(n).max(1.)))
                    .collect::<Result<Vec<_>>>()?
                    .into();
            }
            p.emf_objects.insert(h, Object::Pen(Arc::new(pen)));
        }
        120 => {
            let mut brush = (*p.state.brush).clone();
            brush.color = Color::colorref(r.u32(8)?);
            p.state.brush = Arc::new(brush);
        }
        121 => {
            let mut pen = (*p.state.pen).clone();
            pen.color = Color::colorref(r.u32(8)?);
            p.state.pen = Arc::new(pen);
        }
        59 => p.begin_path(),
        60 => p.end_path(),
        61 => {
            if p.path_active && p.path_builder.figure_open() {
                p.path_builder.close();
            } else {
                p.warn(
                    "close_figure_without_path",
                    "CloseFigure ignored without an active path",
                    true,
                );
            }
        }
        62..=64 => {
            let path = p.consume_path();
            p.emit(
                if kind == 63 {
                    geometry::close_open(&path)
                } else {
                    path
                },
                matches!(kind, 63 | 64),
                matches!(kind, 62 | 63),
            )?;
        }
        68 => {
            p.path_builder.clear();
            p.path_active = false;
            p.path_ready = false;
        }
        76 | 77 | 80 | 81 | 114 => super::bitmap::emf(kind, r, p)?,
        83 | 84 => super::text::emf(kind, r, p)?,
        16 | 23 | 48..=52 | 98..=101 | 104 | 107 | 109..=113 | 115 | 117 | 119 => p.warn(
            "ignored_state_record",
            &format!("EMF state/control record was ignored: {kind}"),
            false,
        ),
        _ => p.warn(
            "unsupported_emf_record",
            &format!("unsupported EMF record was skipped: {kind}"),
            true,
        ),
    }
    Ok(())
}
