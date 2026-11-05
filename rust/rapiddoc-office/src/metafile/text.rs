// metafile-render 0.3.0 gdi/text.py and backends/text.py (MIT, Xiaomeng Zhao).
use super::{binary::Reader, fonts, geometry, malformed, model::*, playback::Playback, resource};
use crate::Result;
use std::sync::Arc;
pub fn bounds(c: &TextCommand) -> Option<Rect> {
    // ReplayContext.command_bounds uses an approximate device rectangle,
    // independently of font rasterization; preserve its exact original formula.
    if let Some(r) = c
        .bounds
        .filter(|r| r.width().abs() > 0. && r.height().abs() > 0.)
    {
        return Some(r);
    }
    let mut width = (c.font_height * 0.6 * c.text.chars().count() as f64).max(c.font_height);
    if let Some(end) = c.advance_end {
        width = width.max((end[0] - c.origin[0]).abs());
    }
    let factor = match c.alignment.horizontal.as_str() {
        "center" => 0.5,
        "right" => 1.,
        _ => 0.,
    };
    let left = c.origin[0] - width * factor;
    Some(Rect {
        left,
        top: c.origin[1] - c.font_height,
        right: left + width,
        bottom: c.origin[1] + c.font_height * 0.25,
    })
}
pub fn emit(
    p: &mut Playback,
    mut text: String,
    reference: Point,
    options: u32,
    bounds: Option<Rect>,
    advances: Option<Vec<Point>>,
) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    if options & 16 != 0 {
        p.warn(
            "glyph_index_text",
            "glyph-index text was replaced with visible placeholder glyphs",
            true,
        );
        text = "□".repeat(text.chars().count());
    }
    let update = p.state.text_align & 1 != 0;
    let logical = if update {
        p.state.current_position
    } else {
        reference
    };
    let origin = p.map_point(logical)?;
    let mut positions = vec![];
    let mut advance_end = None;
    let font = Arc::clone(&p.state.font);
    if let Some(advances) = advances.filter(|a| !a.is_empty()) {
        let mut cursor = logical;
        for (i, _) in text.chars().enumerate() {
            positions.push(p.map_point(cursor)?);
            if let Some(a) = advances.get(i) {
                cursor[0] += a[0];
                cursor[1] += a[1];
            }
        }
        advance_end = Some(p.map_point(cursor)?);
        if update {
            p.state.current_position = cursor;
        }
    } else if update {
        let advance = fonts::advance(&font, &text)?;
        let angle = font.escapement / 10. * (std::f64::consts::PI / 180.);
        p.state.current_position = [
            logical[0] + advance * angle.cos(),
            logical[1] - advance * angle.sin(),
        ];
    }
    let mapped = p.map_vector([0., if font.height == 0. { -12. } else { font.height }])?;
    let baseline = p.map_vector([1., 0.])?;
    let rotation =
        baseline[1].atan2(baseline[0]) * (180. / std::f64::consts::PI) + font.escapement / 10.;
    let mapped_bounds = if options & 6 != 0 {
        if let Some(r) = bounds {
            let points = [
                p.map_point([r.left, r.top])?,
                p.map_point([r.right, r.top])?,
                p.map_point([r.right, r.bottom])?,
                p.map_point([r.left, r.bottom])?,
            ];
            Some(Rect {
                left: points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
                top: points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
                right: points
                    .iter()
                    .map(|p| p[0])
                    .fold(f64::NEG_INFINITY, f64::max),
                bottom: points
                    .iter()
                    .map(|p| p[1])
                    .fold(f64::NEG_INFINITY, f64::max),
            })
        } else {
            None
        }
    } else {
        None
    };
    let mut clip = Arc::clone(&p.state.clip);
    if let Some(r) = mapped_bounds.filter(|_| options & 4 != 0) {
        if clip.len() >= 64 {
            return Err(resource("metafile exceeds Python max_clip_operations"));
        }
        let mut ops = clip.to_vec();
        ops.push(Clip {
            path: geometry::rectangle(r),
            mode: "and".into(),
            fill_rule: "evenodd".into(),
        });
        clip = ops.into();
    }
    p.append(TextCommand {
        text,
        origin,
        positions,
        font,
        font_height: mapped[0].hypot(mapped[1]).max(1.),
        rotation,
        alignment: TextAlignment::from_gdi(p.state.text_align),
        color: p.state.text_color,
        background_color: p.state.background_color,
        opaque: options & 2 != 0 || p.state.background_mode == 2,
        bounds: mapped_bounds,
        clip,
        advance_end,
        location: Location::default(),
    })
}
pub fn ansi(raw: &[u8], charset: u8) -> String {
    if !matches!(charset, 128 | 129 | 134 | 136) {
        static TABLES: std::sync::OnceLock<std::collections::BTreeMap<String, Vec<u32>>> =
            std::sync::OnceLock::new();
        let tables = TABLES
            .get_or_init(|| serde_json::from_str(include_str!("ansi_singlebyte.json")).unwrap());
        let table = tables.get(&charset.to_string()).unwrap_or(&tables["0"]);
        return raw
            .iter()
            .map(|&b| char::from_u32(table[b as usize]).unwrap())
            .collect();
    }
    struct Tables {
        maps: Vec<Vec<u32>>,
        ranges: Vec<[u32; 3]>,
    }
    static TABLES: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    let tables = TABLES.get_or_init(|| {
        use std::io::Read;
        let mut bytes = vec![];
        flate2::read::GzDecoder::new(&include_bytes!("ansi_dbcs.bin.gz")[..])
            .read_to_end(&mut bytes)
            .unwrap();
        assert_eq!(&bytes[..5], b"RDCS1");
        let mut words = bytes[5..]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
        let mut maps = vec![];
        for _ in 0..4 {
            maps.push(words.by_ref().take(256 + 65536).collect());
        }
        let count = words.next().unwrap();
        let ranges = (0..count)
            .map(|_| {
                [
                    words.next().unwrap(),
                    words.next().unwrap(),
                    words.next().unwrap(),
                ]
            })
            .collect();
        Tables { maps, ranges }
    });
    let map = &tables.maps[match charset {
        128 => 0,
        129 => 1,
        134 => 2,
        136 => 3,
        _ => unreachable!(),
    }];
    let mut text = String::new();
    let mut i = 0;
    while i < raw.len() {
        let b = raw[i];
        let single = map[b as usize];
        if single != u32::MAX {
            text.push(char::from_u32(single).unwrap());
            i += 1;
            continue;
        }
        if charset == 134 && b >= 0x80 && i + 1 < raw.len() && raw[i + 1].is_ascii_digit() {
            if raw.len() - i < 4 {
                text.push('�');
                break;
            }
            if (0x81..=0xfe).contains(&b) && (0x81..=0xfe).contains(&raw[i + 2]) {
                if raw[i + 3].is_ascii_digit() {
                    let pointer = (((u32::from(b - 0x81) * 10 + u32::from(raw[i + 1] - 0x30))
                        * 126
                        + u32::from(raw[i + 2] - 0x81))
                        * 10)
                        + u32::from(raw[i + 3] - 0x30);
                    let n = tables.ranges.partition_point(|r| r[0] <= pointer);
                    if let Some(r) = n
                        .checked_sub(1)
                        .and_then(|n| tables.ranges.get(n))
                        .filter(|r| pointer <= r[1])
                    {
                        text.push(char::from_u32(r[2] + pointer - r[0]).unwrap());
                        i += 4;
                        continue;
                    }
                }
            }
        }
        if i + 1 < raw.len() {
            let pair = map[256 + ((b as usize) << 8) + raw[i + 1] as usize];
            if pair != u32::MAX {
                text.push(char::from_u32(pair).unwrap());
                i += 2;
                continue;
            }
        }
        text.push('�');
        i += 1;
    }
    text
}

pub fn emf(kind: u32, r: Reader<'_>, p: &mut Playback) -> Result<()> {
    r.bytes(0, 76)?;
    let count = r.u32(44)? as usize;
    if count > 1_000_000 {
        return Err(resource("EMF text exceeds Python max characters"));
    }
    let reference = [r.i32(36)? as f64, r.i32(40)? as f64];
    let options = r.u32(52)?;
    let bounds = r.rect32(56)?;
    let dx = r.u32(72)? as usize;
    let raw = r.bytes(r.u32(48)? as usize, count * if kind == 84 { 2 } else { 1 })?;
    let text = if kind == 84 {
        String::from_utf16_lossy(
            &raw.chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect::<Vec<_>>(),
        )
    } else {
        ansi(raw, p.state.font.charset)
    };
    let advances = if dx != 0 {
        let stride = if options & 8192 != 0 { 8 } else { 4 };
        if count > r.remaining(dx)? / stride {
            return Err(malformed());
        }
        Some(
            (0..count)
                .map(|i| {
                    Ok([
                        r.i32(dx + i * stride)? as f64,
                        if stride == 8 {
                            r.i32(dx + i * stride + 4)? as f64
                        } else {
                            0.
                        },
                    ])
                })
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };
    p.charge_points(count)?;
    emit(p, text, reference, options, Some(bounds), advances)
}
pub fn wmf(kind: u32, r: Reader<'_>, p: &mut Playback) -> Result<()> {
    if kind == 0x521 {
        let count = r.u16(0)? as usize;
        let raw = r.bytes(2, count)?;
        let offset = 2 + count + (count & 1);
        let reference = [r.i16(offset + 2)? as f64, r.i16(offset)? as f64];
        p.charge_points(count)?;
        return emit(p, ansi(raw, p.state.font.charset), reference, 0, None, None);
    }
    let reference = [r.i16(2)? as f64, r.i16(0)? as f64];
    let count = r.u16(4)? as usize;
    let options = r.u16(6)? as u32;
    let mut offset = 8;
    let bounds = if options & 6 != 0 {
        let rect = Rect {
            left: r.i16(offset)? as f64,
            top: r.i16(offset + 2)? as f64,
            right: r.i16(offset + 4)? as f64,
            bottom: r.i16(offset + 6)? as f64,
        };
        offset += 8;
        Some(rect)
    } else {
        None
    };
    let raw = r.bytes(offset, count)?;
    offset += count + (count & 1);
    let advances = if r.remaining(offset)? >= count * 2 {
        Some(
            (0..count)
                .map(|i| r.i16(offset + i * 2).map(|x| [x as f64, 0.]))
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };
    p.charge_points(count)?;
    emit(
        p,
        ansi(raw, p.state.font.charset),
        reference,
        options,
        bounds,
        advances,
    )
}
