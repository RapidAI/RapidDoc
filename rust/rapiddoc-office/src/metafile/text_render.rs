// Port of metafile-render 0.3.0 backends/text.py and svg.py (MIT, Xiaomeng Zhao).
use super::{
    dib_decode::RgbaImage, fonts, geometry::FlattenBudget, image_session::ImageSession, model::*,
};
use crate::Result;
fn lines<'a>(
    text: &'a str,
    face: &fonts::Face,
    anchor: &str,
    pos: Point,
) -> Result<Vec<(Point, &'a str)>> {
    let lines: Vec<_> = text.split('\n').collect();
    if lines.len() == 1 {
        return Ok(vec![(pos, text)]);
    }
    if matches!(anchor.as_bytes()[1], b't' | b'b') {
        return Err(crate::Error::malformed(
            "anchor not supported for multiline text",
        ));
    }
    let widths = lines
        .iter()
        .map(|s| face.length(s))
        .collect::<Result<Vec<_>>>()?;
    let max = widths.iter().copied().fold(0., f64::max);
    let spacing = face.bounds("A", "la")?[3] as f64 + 4.;
    let mut top = pos[1];
    if anchor.as_bytes()[1] == b'm' {
        top -= (lines.len() - 1) as f64 * spacing / 2.;
    } else if anchor.as_bytes()[1] == b'd' {
        top -= (lines.len() - 1) as f64 * spacing;
    }
    Ok(lines
        .into_iter()
        .zip(widths)
        .enumerate()
        .map(|(i, (text, width))| {
            let factor = match anchor.as_bytes()[0] {
                b'm' => 0.5,
                b'r' => 1.,
                _ => 0.,
            };
            (
                [pos[0] - (max - width) * factor, top + i as f64 * spacing],
                text,
            )
        })
        .collect())
}
fn text_bounds(text: &str, face: &fonts::Face, anchor: &str) -> Result<[f64; 4]> {
    let mut bbox = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for (pos, line) in lines(text, face, anchor, [0., 0.])? {
        let b = face.bounds(line, anchor)?;
        bbox[0] = bbox[0].min(pos[0] + b[0] as f64);
        bbox[1] = bbox[1].min(pos[1] + b[1] as f64);
        bbox[2] = bbox[2].max(pos[0] + b[2] as f64);
        bbox[3] = bbox[3].max(pos[1] + b[3] as f64);
    }
    Ok(bbox)
}
pub fn anchor(a: &TextAlignment) -> String {
    format!(
        "{}{}",
        match a.horizontal.as_str() {
            "center" => 'm',
            "right" => 'r',
            _ => 'l',
        },
        match a.vertical.as_str() {
            "bottom" => 'd',
            "baseline" => 's',
            _ => 'a',
        }
    )
}
pub fn positions(c: &TextCommand, m: Matrix) -> (Vec<Point>, bool) {
    let mut positions = if c.positions.is_empty() {
        vec![m.point(c.origin)]
    } else {
        c.positions.iter().map(|&p| m.point(p)).collect()
    };
    let positioned = !c.positions.is_empty() && c.advance_end.is_some();
    if let Some(end) = c.advance_end.filter(|_| positioned) {
        let origin = m.point(c.origin);
        let end = m.point(end);
        let factor = match c.alignment.horizontal.as_str() {
            "center" => 0.5,
            "right" => 1.,
            _ => 0.,
        };
        for p in &mut positions {
            p[0] -= (end[0] - origin[0]) * factor;
            p[1] -= (end[1] - origin[1]) * factor;
        }
    }
    (positions, positioned)
}
pub fn background(
    c: &TextCommand,
    m: Matrix,
    face: &fonts::Face,
    positions: &[Point],
    texts: &[String],
    anchor: &str,
) -> Result<Rect> {
    if let Some(r) = c.bounds {
        let p = [
            [r.left, r.top],
            [r.right, r.top],
            [r.right, r.bottom],
            [r.left, r.bottom],
        ]
        .map(|p| m.point(p));
        return Ok(Rect {
            left: p.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
            top: p.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
            right: p.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max),
            bottom: p.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max),
        }
        .normalized());
    }
    let mut r = Rect {
        left: f64::INFINITY,
        top: f64::INFINITY,
        right: f64::NEG_INFINITY,
        bottom: f64::NEG_INFINITY,
    };
    for (pos, text) in positions.iter().zip(texts) {
        let b = text_bounds(text, face, anchor)?;
        r.left = r.left.min(pos[0] + b[0]);
        r.top = r.top.min(pos[1] + b[1]);
        r.right = r.right.max(pos[0] + b[2]);
        r.bottom = r.bottom.max(pos[1] + b[3]);
    }
    Ok(r)
}
fn rgba(c: Color) -> [u8; 4] {
    [c.red, c.green, c.blue, c.alpha]
}
fn rectangle(layer: &mut RgbaImage, r: Rect, color: Color) {
    let [w, h] = layer.size();
    let (l, t, r, b) = (r.left as i64, r.top as i64, r.right as i64, r.bottom as i64);
    for y in t.max(0)..=b.min(h as i64 - 1) {
        for x in l.max(0)..=r.min(w as i64 - 1) {
            layer.pixels[(y as usize * w + x as usize) * 4..][..4].copy_from_slice(&rgba(color));
        }
    }
}
fn draw(
    layer: &mut RgbaImage,
    pos: Point,
    text: &str,
    face: &fonts::Face,
    color: Color,
    anchor: &str,
) -> Result<()> {
    for (pos, line) in lines(text, face, anchor, pos)? {
        draw_single(layer, pos, line, face, color, anchor)?;
    }
    Ok(())
}
pub(crate) fn draw_single(
    layer: &mut RgbaImage,
    pos: Point,
    text: &str,
    face: &fonts::Face,
    color: Color,
    anchor: &str,
) -> Result<()> {
    let [w, h] = layer.size();
    let (offset, [mw, mh], mask) = face.mask(text, anchor, [pos[0].fract(), pos[1].fract()])?;
    let (x, y) = (
        pos[0] as i64 + offset[0] as i64,
        pos[1] as i64 + offset[1] as i64,
    );
    let ink = rgba(color);
    for my in 0..mh {
        let dy = y + my as i64;
        if dy < 0 || dy >= h as i64 {
            continue;
        }
        for mx in 0..mw {
            let dx = x + mx as i64;
            if dx < 0 || dx >= w as i64 {
                continue;
            }
            let m = u32::from(mask[my * mw + mx]);
            let p = &mut layer.pixels[(dy as usize * w + dx as usize) * 4..][..4];
            for c in 0..4 {
                let cm = if c != 3 && m != 0 && p[3] == 0 {
                    255
                } else {
                    m
                };
                let v = u32::from(p[c]) * (255 - cm) + u32::from(ink[c]) * cm + 128;
                p[c] = (((v >> 8) + v) >> 8) as u8;
            }
        }
    }
    Ok(())
}
fn decorations(
    layer: &mut RgbaImage,
    pos: Point,
    b: [f64; 4],
    width: i64,
    color: Color,
    font: &Font,
) {
    let left = pos[0] + b[0] as f64;
    let right = pos[0] + b[2] as f64;
    let mut line = |y: f64| {
        let top = y as i64 - (width - 1) / 2;
        rectangle(
            layer,
            Rect {
                left: left as i64 as f64,
                top: top as f64,
                right: right as i64 as f64,
                bottom: (top + width - 1) as f64,
            },
            color,
        );
    };
    if font.underline {
        line(pos[1] + b[3] as f64 + width as f64);
    }
    if font.strikeout {
        line(pos[1] + (b[1] + b[3]) as f64 / 2.);
    }
}
fn rotated(
    layer: &mut RgbaImage,
    pos: Point,
    text: &str,
    face: &fonts::Face,
    c: &TextCommand,
    anchor: &str,
) -> Result<()> {
    let bbox = text_bounds(text, face, anchor)?;
    let width = (bbox[2] - bbox[0]).ceil().max(1.);
    let height = (bbox[3] - bbox[1]).ceil().max(1.);
    let line_width = (height / 14.).round_ties_even().max(1.) as i64;
    if c.rotation.abs() < 1e-6 {
        draw(layer, pos, text, face, c.color, anchor)?;
        decorations(layer, pos, bbox, line_width, c.color, &c.font);
        return Ok(());
    }
    let padding = (width.max(height) * 0.1).ceil().max(4.);
    let bottom = if c.font.underline {
        bbox[3] + line_width as f64 * 2.
    } else {
        bbox[3]
    };
    let mut radius = 0f64;
    for x in [bbox[0], bbox[2]] {
        for y in [bbox[1], bottom] {
            radius = radius.max(x.hypot(y));
        }
    }
    let radius = radius.ceil() + padding;
    let size = (radius * 2. + 1.) as usize;
    let mut patch = RgbaImage::blank([size, size])?;
    draw(&mut patch, [radius, radius], text, face, c.color, anchor)?;
    decorations(
        &mut patch,
        [radius, radius],
        bbox,
        line_width,
        c.color,
        &c.font,
    );
    let angle = -(-c.rotation).rem_euclid(360.) * (std::f64::consts::PI / 180.);
    let r15 = |v: f64| format!("{v:.15}").parse::<f64>().unwrap();
    let (a, b, d, e) = (
        r15(angle.cos()),
        r15(angle.sin()),
        r15(-angle.sin()),
        r15(angle.cos()),
    );
    let patch = super::sampling::transform(
        &patch,
        [
            a,
            b,
            a * (-radius) + b * (-radius) + radius,
            d,
            e,
            d * (-radius) + e * (-radius) + radius,
        ],
        [size, size],
        4,
    )?;
    let target = [
        (pos[0] - radius).round_ties_even() as i64,
        (pos[1] - radius).round_ties_even() as i64,
    ];
    let [w, h] = layer.size();
    for y in 0..size {
        let dy = target[1] + y as i64;
        if dy < 0 || dy >= h as i64 {
            continue;
        }
        let left = 0i64.max(-target[0]) as usize;
        let right = size.min((w as i64 - target[0]).max(0) as usize);
        if right <= left {
            continue;
        }
        let dst = (dy as usize * w + (target[0] + left as i64) as usize) * 4;
        let src = (y * size + left) * 4;
        super::composite::source_over(
            &mut layer.pixels[dst..dst + (right - left) * 4],
            &patch.pixels[src..src + (right - left) * 4],
        )?;
    }
    Ok(())
}
pub fn render(
    c: &TextCommand,
    m: Matrix,
    size: [usize; 2],
    budget: &mut FlattenBudget,
    session: &mut ImageSession,
) -> Result<RgbaImage> {
    let font_size = (c.font_height * m.d.abs().max(1e-9))
        .round_ties_even()
        .max(1.) as u32;
    let face = session.font(c, font_size)?;
    let (positions, positioned) = positions(c, m);
    let mut anchor = anchor(&c.alignment);
    if positioned {
        anchor.replace_range(..1, "l");
    }
    let texts: Vec<String> = if c.positions.is_empty() {
        vec![c.text.clone()]
    } else {
        c.text.chars().map(|ch| ch.to_string()).collect()
    };
    let mut layer = RgbaImage::blank(size)?;
    if c.opaque {
        rectangle(
            &mut layer,
            background(c, m, &face, &positions, &texts, &anchor)?,
            c.background_color,
        );
    }
    for (pos, text) in positions.iter().zip(&texts) {
        rotated(&mut layer, *pos, text, &face, c, &anchor)?;
    }
    super::paths::apply_clip(&mut layer, &c.clip, m, budget, session)?;
    Ok(layer)
}
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}
pub fn svg(c: &TextCommand, m: Matrix, session: &mut ImageSession) -> Result<String> {
    let size = (c.font_height * m.d.abs().max(1e-9)).max(1.);
    let face = session.font(c, size.round_ties_even().max(1.) as u32)?;
    let (positions, positioned) = positions(c, m);
    let texts: Vec<String> = if c.positions.is_empty() {
        vec![c.text.clone()]
    } else {
        c.text.chars().map(|ch| ch.to_string()).collect()
    };
    let (text_anchor, baseline) = (
        if positioned {
            "start"
        } else {
            match c.alignment.horizontal.as_str() {
                "center" => "middle",
                "right" => "end",
                _ => "start",
            }
        },
        match c.alignment.vertical.as_str() {
            "bottom" => "text-after-edge",
            "baseline" => "alphabetic",
            _ => "hanging",
        },
    );
    let number = super::output::number;
    let color = |c: Color| format!("#{:02x}{:02x}{:02x}", c.red, c.green, c.blue);
    let opacity = |c: Color| {
        if c.alpha == 255 {
            String::new()
        } else {
            format!(" opacity=\"{}\"", number(c.alpha as f64 / 255.))
        }
    };
    let decorations = match (c.font.underline, c.font.strikeout) {
        (true, true) => " text-decoration=\"underline line-through\"",
        (true, false) => " text-decoration=\"underline\"",
        (false, true) => " text-decoration=\"line-through\"",
        _ => "",
    };
    let common=format!("fill=\"{}\"{} font-family=\"{}\" font-size=\"{}\" font-weight=\"{}\" font-style=\"{}\" text-anchor=\"{text_anchor}\" dominant-baseline=\"{baseline}\"{decorations}",color(c.color),opacity(c.color),escape(&c.font.face_name),number(size),c.font.weight,if c.font.italic{"italic"}else{"normal"});
    let mut out = String::new();
    if c.opaque {
        let mut anchor = anchor(&c.alignment);
        if positioned {
            anchor.replace_range(..1, "l");
        }
        let b = background(c, m, &face, &positions, &texts, &anchor)?;
        out.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\"{}/>",
            number(b.left),
            number(b.top),
            number(b.width()),
            number(b.height()),
            color(c.background_color),
            opacity(c.background_color)
        ));
    }
    for (p, text) in positions.iter().zip(texts) {
        let rotation = if c.rotation.abs() > 1e-6 {
            format!(
                " transform=\"rotate({} {} {})\"",
                number(c.rotation),
                number(p[0]),
                number(p[1])
            )
        } else {
            String::new()
        };
        out.push_str(&format!(
            "<text x=\"{}\" y=\"{}\" {common}{rotation}>{}</text>",
            number(p[0]),
            number(p[1]),
            escape(&text)
        ));
    }
    Ok(out)
}
