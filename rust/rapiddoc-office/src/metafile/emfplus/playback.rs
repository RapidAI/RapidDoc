// metafile-render 0.3.0 emfplus/playback.py and table.py (MIT, Xiaomeng Zhao).
use super::super::{
    binary::{plus_records, PlusRecord, Reader},
    fonts,
    geometry::{self, Builder},
    malformed,
    model::*,
    playback::Playback,
    resource,
};
use super::{
    cursor::{argb, finite, Cursor},
    objects::{self, Object},
};
use crate::{Error, ErrorKind, Result};
use std::{collections::BTreeMap, sync::Arc};
#[derive(Clone, serde::Serialize)]
pub struct PlusState {
    pub world: Matrix,
    pub container: Matrix,
    pub unit: u32,
    pub page_scale: f64,
    pub clip: Arc<[Clip]>,
}
impl Default for PlusState {
    fn default() -> Self {
        Self {
            world: Matrix::default(),
            container: Matrix::default(),
            unit: 1,
            page_scale: 1.,
            clip: Arc::from([]),
        }
    }
}
struct Continued {
    slot: u8,
    kind: u8,
    size: usize,
    data: Vec<u8>,
}
pub struct Plus {
    pub state: PlusState,
    stack: Vec<(u32, &'static str, PlusState)>,
    objects: BTreeMap<u8, Object>,
    continued: Option<Continued>,
    pub record_count: usize,
    pub saw_header: bool,
    pub saw_eof: bool,
    pub gdi_active: bool,
    pub halted: bool,
    bounds: Rect,
    dpi: Point,
    bridge: Option<(State, Vec<State>)>,
}
impl Plus {
    pub fn new(bounds: Rect, dpi: Point) -> Self {
        Self {
            state: PlusState::default(),
            stack: vec![],
            objects: BTreeMap::new(),
            continued: None,
            record_count: 0,
            saw_header: false,
            saw_eof: false,
            gdi_active: false,
            halted: false,
            bounds,
            dpi,
            bridge: None,
        }
    }
    fn stop(&mut self, p: &mut Playback, message: &str) {
        self.halted = true;
        self.gdi_active = false;
        p.warn("emfplus_state_unsupported", message, true);
    }
    fn leave(&mut self, p: &mut Playback) {
        if let Some((state, stack)) = self.bridge.take() {
            p.state = state;
            p.state_stack = stack;
        }
    }
    pub fn units(&self, unit: u32) -> Result<Point> {
        let divisor = match unit {
            0..=2 => return Ok([1., 1.]),
            3 => 72.,
            4 => 1.,
            5 => 300.,
            6 => 25.4,
            _ => return Err(malformed()),
        };
        Ok([self.dpi[0] / divisor, self.dpi[1] / divisor])
    }
    pub fn matrix(&self) -> Result<Matrix> {
        let [x, y] = self.units(self.state.unit)?;
        let m = self
            .state
            .world
            .then(Matrix {
                a: x * self.state.page_scale,
                d: y * self.state.page_scale,
                ..Matrix::default()
            })
            .then(self.state.container);
        for v in [m.a, m.b, m.c, m.d, m.e, m.f] {
            finite(v)?;
        }
        Ok(m)
    }
    fn point(&self, p: Point, m: Matrix) -> Result<Point> {
        let p = m.point(p);
        Ok([finite(p[0])?, finite(p[1])?])
    }
    fn get(&self, slot: u32, expected: u8, p: &mut Playback) -> Result<Option<Object>> {
        if slot >= 64 {
            return Err(malformed());
        }
        let value = self.objects.get(&(slot as u8)).ok_or_else(malformed)?;
        if matches!(value, Object::Unsupported) {
            p.warn(
                "emfplus_object_skipped",
                &format!("drawing references unsupported object {slot}"),
                true,
            );
            return Ok(None);
        }
        let kind = match value {
            Object::Brush(_) => 1,
            Object::Pen { .. } => 2,
            Object::Path(_) => 3,
            Object::Image { .. } => 5,
            Object::Font { .. } => 6,
            Object::Format { .. } => 7,
            Object::Attributes => 8,
            Object::Unsupported => unreachable!(),
        };
        if kind != expected {
            return Err(malformed());
        }
        Ok(Some(value.clone()))
    }
    fn brush(&self, value: u32, flags: u16, p: &mut Playback) -> Result<Option<Color>> {
        if flags & 0x8000 != 0 {
            Ok(Some(argb(value)))
        } else {
            match self.get(value, 1, p)? {
                Some(Object::Brush(c)) => Ok(c),
                _ => Ok(None),
            }
        }
    }
    fn define(&mut self, flags: u16, c: &mut Cursor<'_>, p: &mut Playback) -> Result<()> {
        let slot = flags as u8;
        let kind = ((flags >> 8) & 127) as u8;
        let more = flags & 0x8000 != 0;
        if slot >= 64 || kind == 0 {
            return Err(malformed());
        }
        let total = if more {
            let n = c.u32()? as usize;
            if n == 0 || n > 128 * 1024 * 1024 {
                return Err(resource("EMF+ continued object exceeds Python byte budget"));
            }
            Some(n)
        } else {
            None
        };
        let chunk = c.take(c.remaining()?)?;
        p.budget.object_bytes += chunk.len();
        if p.budget.object_bytes > 128 * 1024 * 1024 {
            return Err(resource(
                "EMF+ cumulative objects exceed Python byte budget",
            ));
        }
        if let Some(current) = &self.continued {
            if current.slot != slot
                || current.kind != kind
                || total.is_some_and(|n| n != current.size)
            {
                return Err(malformed());
            }
        } else {
            self.objects.insert(slot, Object::Unsupported);
            if let Some(size) = total {
                self.continued = Some(Continued {
                    slot,
                    kind,
                    size,
                    data: vec![],
                });
            }
        }
        let owned;
        if let Some(current) = &mut self.continued {
            let remaining = current.size - current.data.len();
            if chunk.len() > remaining && (more || chunk.len() - remaining > 3) {
                return Err(malformed());
            }
            current
                .data
                .extend_from_slice(&chunk[..chunk.len().min(remaining)]);
            if more {
                return Ok(());
            }
            if current.data.len() != current.size {
                return Err(malformed());
            }
            owned = std::borrow::Cow::Owned(self.continued.take().unwrap().data);
        } else {
            owned = std::borrow::Cow::Borrowed(chunk);
        }
        let mut cursor = Cursor::new(Reader {
            data: &owned,
            base: c.r.base,
        });
        let value = objects::parse(kind, &mut cursor, &mut |code, msg| p.warn(code, msg, true))?;
        if let Object::Path(path) = &value {
            p.charge_points(path.segments.iter().map(|s| s.points.len()).sum())?;
        }
        if let Object::Image { png, width, height } = &value {
            p.budget.decoded_bytes += width * height * 4 + png.len();
            if p.budget.decoded_bytes > 128 * 1024 * 1024 {
                return Err(resource(
                    "EMF+ cumulative decoded images exceed Python byte budget",
                ));
            }
        }
        self.objects.insert(slot, value);
        Ok(())
    }
    fn save(&mut self, index: u32, kind: &'static str) -> Result<()> {
        if self.stack.len() >= 64 {
            return Err(resource("EMF+ state stack exceeds Python depth budget"));
        }
        self.stack.push((index, kind, self.state.clone()));
        Ok(())
    }
    fn restore(&mut self, index: u32, kind: &str) -> Result<()> {
        let i = self
            .stack
            .iter()
            .rposition(|(i, k, _)| *i == index && *k == kind)
            .ok_or_else(malformed)?;
        self.state = self.stack[i].2.clone();
        self.stack.truncate(i);
        Ok(())
    }
    fn clip(&mut self, path: Path, mode: u16, p: &mut Playback) -> Result<()> {
        let name = match mode {
            0 => "copy",
            1 => "and",
            2 => "or",
            3 => "xor",
            4 => "diff",
            _ => {
                self.stop(
                    p,
                    &format!("clip combine mode {mode} cannot be approximated safely"),
                );
                return Ok(());
            }
        };
        let transformed = geometry::transform(&path, self.matrix()?);
        p.charge_points(path.segments.iter().map(|s| s.points.len()).sum())?;
        let mut clip = if mode == 0 {
            vec![]
        } else {
            self.state.clip.to_vec()
        };
        if clip.len() >= 64 {
            return Err(resource("EMF+ clip operations exceed Python budget"));
        }
        clip.push(Clip {
            path: transformed,
            mode: name.into(),
            fill_rule: "evenodd".into(),
        });
        self.state.clip = clip.into();
        Ok(())
    }
    pub fn comment(&mut self, r: Reader<'_>, p: &mut Playback) -> Result<()> {
        for item in plus_records(r)? {
            let item = item?;
            self.gdi_active = false;
            self.leave(p);
            self.record_count += 1;
            if self.record_count > 1_000_000 {
                return Err(resource("EMF+ records exceed Python record budget"));
            }
            p.set_record(item.kind as u32, self.record_count, item.offset);
            if self.saw_eof
                || (!self.saw_header && item.kind != 0x4001)
                || (self.continued.is_some() && item.kind != 0x4008)
            {
                return Err(malformed());
            }
            self.record(item, p)?;
        }
        Ok(())
    }
    pub fn finish(&self, p: &Playback) -> Result<()> {
        if !self.saw_header || !self.saw_eof || self.continued.is_some() {
            return Err(malformed());
        }
        if p.commands.is_empty() {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "EMF+ Only contains no supported drawing operations",
            ));
        }
        Ok(())
    }
    fn record(&mut self, r: PlusRecord<'_>, p: &mut Playback) -> Result<()> {
        let kind = r.kind;
        let flags = r.flags;
        let mut c = Cursor::new(r.data);
        match kind {
            0x4001 => {
                if self.saw_header {
                    return Err(malformed());
                }
                c.version()?;
                c.u32()?;
                let (x, y) = (c.u32()?, c.u32()?);
                if x == 0 || y == 0 || x.max(y) > 100000 {
                    return Err(malformed());
                }
                self.dpi = [x as f64, y as f64];
                self.saw_header = true;
                c.finish()?;
                return Ok(());
            }
            0x4002 => {
                c.finish()?;
                self.saw_eof = true;
                return Ok(());
            }
            _ => {}
        }
        if self.halted {
            return Ok(());
        }
        if kind == 0x4003 {
            return Ok(());
        }
        if kind == 0x4004 {
            c.finish()?;
            self.leave(p);
            self.bridge = Some((p.state.clone(), std::mem::take(&mut p.state_stack)));
            p.state.world_transform = self.matrix()?;
            p.state.clip = Arc::clone(&self.state.clip);
            self.gdi_active = true;
            return Ok(());
        }
        if kind == 0x4008 {
            return self.define(flags, &mut c, p);
        }
        if self.control(kind, flags, &mut c, p)? {
            return c.finish();
        }
        match kind {
            0x401c => self.draw_string(flags, &mut c, p),
            0x4036 => self.driver_string(flags, &mut c, p),
            0x401a | 0x401b => self.draw_image(kind, flags, &mut c, p),
            0x4009..=0x4019 | 0x4037 => self.shape(kind, flags, &mut c, p),
            0x4038 => {
                p.warn(
                    "emfplus_effect_ignored",
                    "image effects are not implemented",
                    true,
                );
                Ok(())
            }
            _ => {
                self.stop(
                    p,
                    &format!("unknown EMF+ record {kind:#x} may change graphics state"),
                );
                Ok(())
            }
        }
    }
    fn control(
        &mut self,
        kind: u16,
        flags: u16,
        c: &mut Cursor<'_>,
        p: &mut Playback,
    ) -> Result<bool> {
        match kind {
            0x4025 | 0x4028 => {
                self.save(c.u32()?, if kind == 0x4025 { "save" } else { "container" })?
            }
            0x4026 | 0x4029 => {
                self.restore(c.u32()?, if kind == 0x4026 { "save" } else { "container" })?
            }
            0x4027 => {
                let dest = c.rect(false)?;
                let src = c.rect(false)?;
                let index = c.u32()?;
                if src.width() == 0. || src.height() == 0. {
                    return Err(malformed());
                }
                self.units(flags as u8 as u32)?;
                let parent = self.matrix()?;
                self.save(index, "container")?;
                let mapping = Matrix {
                    a: dest.width() / src.width(),
                    d: dest.height() / src.height(),
                    e: dest.left - src.left * dest.width() / src.width(),
                    f: dest.top - src.top * dest.height() / src.height(),
                    ..Matrix::default()
                };
                self.state = PlusState {
                    container: mapping.then(parent),
                    unit: 2,
                    clip: Arc::clone(&self.state.clip),
                    ..PlusState::default()
                };
            }
            0x402b => self.state.world = Matrix::default(),
            0x402a | 0x402c..=0x402f => {
                let m = match kind {
                    0x402a | 0x402c => c.matrix()?,
                    0x402d => Matrix {
                        e: c.f32()?,
                        f: c.f32()?,
                        ..Matrix::default()
                    },
                    0x402e => Matrix {
                        a: c.f32()?,
                        d: c.f32()?,
                        ..Matrix::default()
                    },
                    _ => {
                        let angle = c.f32()? * (std::f64::consts::PI / 180.);
                        Matrix {
                            a: angle.cos(),
                            b: angle.sin(),
                            c: -angle.sin(),
                            d: angle.cos(),
                            ..Matrix::default()
                        }
                    }
                };
                self.state.world = if kind == 0x402a {
                    m
                } else if flags & 0x2000 != 0 {
                    self.state.world.then(m)
                } else {
                    m.then(self.state.world)
                };
                self.matrix()?;
            }
            0x4030 => {
                let unit = flags as u8 as u32;
                self.units(unit)?;
                let scale = c.f32()?;
                if scale <= 0. {
                    return Err(malformed());
                }
                self.state.unit = unit;
                self.state.page_scale = scale;
            }
            0x4031 => self.state.clip = Arc::from([]),
            0x4032 => {
                let path = geometry::rectangle(c.rect(false)?);
                self.clip(path, (flags >> 8) & 15, p)?;
            }
            0x4033 => match self.get(flags as u8 as u32, 3, p)? {
                Some(Object::Path(path)) => self.clip(path, (flags >> 8) & 15, p)?,
                _ => self.stop(p, "clip path object is unsupported"),
            },
            0x4034 => {
                self.get(flags as u8 as u32, 0, p)?;
                self.stop(p, "complex region clipping is unsupported");
            }
            0x4035 => {
                let [x, y] = self.matrix()?.vector(c.point(false)?);
                let m = Matrix {
                    e: finite(x)?,
                    f: finite(y)?,
                    ..Matrix::default()
                };
                self.state.clip = self
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
            0x401d => {
                let (x, y) = (c.i32()?, c.i32()?);
                if x != 0 || y != 0 {
                    p.warn(
                        "emfplus_rendering_approximation",
                        "rendering origin ignored",
                        true,
                    );
                }
            }
            0x401e..=0x4024 => {
                let value = flags as u8;
                let standard = match kind {
                    0x401e => value <= 2,
                    0x4020 => value == 4,
                    0x4021 | 0x4022 => value <= 1,
                    _ => value == 0,
                };
                if !standard {
                    p.warn(
                        "emfplus_rendering_approximation",
                        &format!("rendering setting {kind:#x}={value} uses the standard renderer"),
                        true,
                    );
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
    fn emit_path(
        &self,
        path: Path,
        brush: Option<Color>,
        pen: Option<Object>,
        p: &mut Playback,
    ) -> Result<()> {
        if path.segments.is_empty() {
            return Ok(());
        }
        let m = self.matrix()?;
        let transformed = geometry::transform(&path, m);
        for s in transformed.segments.iter() {
            for &[x, y] in &s.points {
                finite(x)?;
                finite(y)?;
            }
        }
        let fill = brush.is_some();
        let stroke = matches!(&pen,Some(Object::Pen{pen,..}) if !pen.null);
        if !fill && !stroke {
            return Ok(());
        }
        p.charge_points(path.segments.iter().map(|s| s.points.len()).sum())?;
        let (resolved, miter) = if let Some(Object::Pen {
            mut pen,
            unit,
            miter,
        }) = pen
        {
            let mut scale = (m.a.hypot(m.b) + m.c.hypot(m.d)) / 2.;
            if unit != 0 {
                let u = self.units(unit)?;
                let page = self.units(self.state.unit)?;
                scale *= (u[0] + u[1]) / (page[0] + page[1]);
            }
            let width = finite(pen.width * scale)?.max(1.);
            pen.width = width;
            pen.dashes = pen
                .dashes
                .iter()
                .map(|d| d * width)
                .collect::<Vec<_>>()
                .into();
            (pen, miter)
        } else {
            (
                Pen {
                    null: true,
                    ..Pen::default()
                },
                10.,
            )
        };
        p.append(PathCommand {
            path: transformed,
            pen: resolved,
            brush: Arc::new(if let Some(color) = brush {
                Brush {
                    color,
                    ..Brush::default()
                }
            } else {
                Brush {
                    kind: "null".into(),
                    ..Brush::default()
                }
            }),
            stroke,
            fill,
            fill_rule: "evenodd".into(),
            clip: Arc::clone(&self.state.clip),
            rop2: 13,
            miter_limit: miter,
            location: Location::default(),
        })
    }
    fn shape(&mut self, kind: u16, flags: u16, c: &mut Cursor<'_>, p: &mut Playback) -> Result<()> {
        if kind == 0x4009 {
            p.append(ClearCommand {
                color: argb(c.u32()?),
                bounds: self.bounds,
                clip: Arc::clone(&self.state.clip),
                location: Location::default(),
            })?;
            return c.finish();
        }
        if matches!(kind, 0x4013 | 0x4016..=0x4018 | 0x4037) {
            p.warn(
                "emfplus_drawing_skipped",
                &format!("drawing record {kind:#x} is not implemented"),
                true,
            );
            return Ok(());
        }
        let fill = matches!(kind, 0x400a | 0x400c | 0x400e | 0x4010 | 0x4014);
        let brush = if fill {
            self.brush(c.u32()?, flags, p)?
        } else {
            None
        };
        let pen = if kind == 0x4015 {
            self.get(c.u32()?, 2, p)?
        } else if fill {
            None
        } else {
            self.get(flags as u8 as u32, 2, p)?
        };
        let paths = match kind {
            0x4014 | 0x4015 => match self.get(flags as u8 as u32, 3, p)? {
                Some(Object::Path(path)) => vec![path],
                _ => vec![],
            },
            0x400a | 0x400b => {
                let n = c.count()?;
                p.charge_points(n * 4)?;
                (0..n)
                    .map(|_| c.rect(flags & 0x4000 != 0).map(geometry::rectangle))
                    .collect::<Result<Vec<_>>>()?
            }
            0x400e | 0x400f => vec![geometry::ellipse(c.rect(flags & 0x4000 != 0)?)],
            0x4010..=0x4012 => {
                let start = c.f32()?;
                let sweep = c.f32()?;
                let rect = c.rect(flags & 0x4000 != 0)?;
                if sweep.abs() >= 360. {
                    vec![geometry::ellipse(rect)]
                } else if sweep == 0. {
                    vec![]
                } else {
                    let center = [(rect.left + rect.right) / 2., (rect.top + rect.bottom) / 2.];
                    let point = |angle: f64| {
                        let a = angle * (std::f64::consts::PI / 180.);
                        [
                            center[0] + rect.width() / 2. * a.cos(),
                            center[1] + rect.height() / 2. * a.sin(),
                        ]
                    };
                    vec![geometry::arc(
                        rect,
                        point(start),
                        point(start + sweep),
                        if sweep > 0. { 2 } else { 1 },
                        if kind == 0x4012 { "open" } else { "pie" },
                    )]
                }
            }
            _ => {
                let n = c.count()?;
                let min = if kind == 0x4019 {
                    4
                } else if kind == 0x400c {
                    3
                } else {
                    2
                };
                if n < min {
                    return Err(malformed());
                }
                let points = c.points(n, flags)?;
                let mut b = Builder::default();
                b.move_to(points[0]);
                if kind == 0x4019 {
                    if (n - 1) % 3 != 0 {
                        return Err(malformed());
                    }
                    for chunk in points[1..].chunks_exact(3) {
                        b.cubic_to(chunk[0], chunk[1], chunk[2]);
                    }
                } else {
                    for &point in &points[1..] {
                        b.line_to(point);
                    }
                    if kind == 0x400c || flags & 0x2000 != 0 {
                        b.close();
                    }
                }
                vec![b.take()]
            }
        };
        c.finish()?;
        for path in paths {
            self.emit_path(path, brush, pen.clone(), p)?;
        }
        Ok(())
    }
    fn draw_image(
        &self,
        kind: u16,
        flags: u16,
        c: &mut Cursor<'_>,
        p: &mut Playback,
    ) -> Result<()> {
        let image = self.get(flags as u8 as u32, 5, p)?;
        let attributes = c.u32()?;
        let unit = c.u32()?;
        let source = c.rect(false)?;
        if unit != 2 {
            return Err(malformed());
        }
        if attributes != u32::MAX {
            self.get(attributes, 8, p)?;
            p.warn(
                "emfplus_image_approximation",
                "image attributes use standard bitmap sampling",
                true,
            );
        }
        let points = if kind == 0x401a {
            let r = c.rect(flags & 0x4000 != 0)?;
            vec![[r.left, r.top], [r.right, r.top], [r.left, r.bottom]]
        } else {
            let n = c.count()?;
            if n != 3 {
                return Err(malformed());
            }
            c.points(n, flags)?
        };
        c.finish()?;
        if flags & 0x2000 != 0 {
            p.warn("emfplus_effect_ignored", "image effect ignored", true);
        }
        if let Some(Object::Image { png, .. }) = image {
            let m = self.matrix()?;
            let a = self.point(points[0], m)?;
            let b = self.point(points[1], m)?;
            let d = self.point(points[2], m)?;
            let third = [finite(b[0] + d[0] - a[0])?, finite(b[1] + d[1] - a[1])?];
            p.append(ImageCommand {
                image: ImagePayload::Encoded { png },
                destination: [a, b, third, d],
                source: Some(source),
                rop: 0,
                stretch_mode: 3,
                constant_alpha: 255,
                clip: Arc::clone(&self.state.clip),
                location: Location::default(),
            })?;
        }
        Ok(())
    }
    fn text_command(
        &self,
        text: String,
        font: Object,
        brush: Option<Color>,
        origin: Point,
        positions: Vec<Point>,
        alignment: u32,
        m: Matrix,
        clip: Arc<[Clip]>,
        p: &mut Playback,
    ) -> Result<()> {
        let Some(color) = brush else {
            return Ok(());
        };
        if text.is_empty() {
            return Ok(());
        }
        let Object::Font { font, unit } = font else {
            return Err(malformed());
        };
        let logical = font.height.abs() * self.units(unit)?[1] / self.units(self.state.unit)?[1];
        let height = finite(logical * m.c.hypot(m.d))?;
        if height <= 0. {
            return Ok(());
        }
        if (m.a.hypot(m.b) - m.c.hypot(m.d)).abs() > 1e-5 || (m.a * m.c + m.b * m.d).abs() > 1e-5 {
            p.warn(
                "emfplus_text_approximation",
                "nonuniform text transform uses rotated glyphs",
                true,
            );
        }
        p.charge_points(text.chars().count())?;
        p.append(TextCommand {
            text,
            origin: self.point(origin, m)?,
            positions: positions
                .into_iter()
                .map(|pt| self.point(pt, m))
                .collect::<Result<Vec<_>>>()?,
            font,
            font_height: height,
            rotation: m.b.atan2(m.a) * (180. / std::f64::consts::PI),
            alignment: TextAlignment::from_gdi(24 | alignment),
            color,
            background_color: color,
            opaque: false,
            bounds: None,
            clip,
            advance_end: None,
            location: Location::default(),
        })
    }
    fn draw_string(&self, flags: u16, c: &mut Cursor<'_>, p: &mut Playback) -> Result<()> {
        let font = self.get(flags as u8 as u32, 6, p)?;
        let brush = self.brush(c.u32()?, flags, p)?;
        let format_id = c.u32()?;
        let n = c.count()?;
        let rect = c.rect(false)?;
        let text = c.text(n)?;
        c.finish()?;
        let format = if format_id == u32::MAX {
            Some(Object::Format {
                flags: 0,
                alignment: 0,
                vertical: 0,
            })
        } else {
            self.get(format_id, 7, p)?
        };
        let (
            Some(Object::Font { font, unit }),
            Some(Object::Format {
                flags,
                alignment,
                vertical,
            }),
        ) = (font, format)
        else {
            return Ok(());
        };
        if flags & 3 != 0 || text.contains('\t') {
            p.warn(
                "emfplus_text_skipped",
                "vertical, right-to-left or tabbed text is not supported",
                true,
            );
            return Ok(());
        }
        let em = font.height.abs() * self.units(unit)?[1] / self.units(self.state.unit)?[1];
        let size = em.round_ties_even().max(1.) as u32;
        let loaded = fonts::resolve(&font, size)?;
        let (ascent, descent) = loaded.metrics();
        let line_height = (ascent + descent) as f64 * em / size as f64;
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        let lines: Vec<_> = text.split('\n').collect();
        let offset =
            (rect.height() - line_height * lines.len() as f64).max(0.) * (vertical as f64 / 2.);
        let mut baseline = rect.top + offset + ascent as f64 * em / size as f64;
        let x = rect.left + rect.width() * (alignment as f64 / 2.);
        let mut clip = Arc::clone(&self.state.clip);
        if flags & 0x4000 == 0 && rect.width() > 0. && rect.height() > 0. {
            if clip.len() >= 64 {
                return Err(resource("EMF+ text clip exceeds Python budget"));
            }
            let mut v = clip.to_vec();
            v.push(Clip {
                path: geometry::transform(&geometry::rectangle(rect), self.matrix()?),
                mode: "and".into(),
                fill_rule: "evenodd".into(),
            });
            clip = v.into();
        }
        let logical = Font {
            height: -em,
            ..(*font).clone()
        };
        for line in lines {
            if flags & 0x1000 == 0
                && rect.width() > 0.
                && fonts::advance(&logical, line)? > rect.width()
            {
                p.warn(
                    "emfplus_text_approximation",
                    "automatic line wrapping uses explicit line breaks only",
                    true,
                );
            }
            self.text_command(
                line.into(),
                Object::Font {
                    font: Arc::clone(&font),
                    unit,
                },
                brush,
                [x, baseline],
                vec![],
                match alignment {
                    1 => 6,
                    2 => 2,
                    _ => 0,
                },
                self.matrix()?,
                Arc::clone(&clip),
                p,
            )?;
            baseline += line_height;
        }
        Ok(())
    }
    fn driver_string(&self, flags: u16, c: &mut Cursor<'_>, p: &mut Playback) -> Result<()> {
        let font = self.get(flags as u8 as u32, 6, p)?;
        let brush = self.brush(c.u32()?, flags, p)?;
        let options = c.u32()?;
        let has_matrix = c.u32()?;
        let n = c.count()?;
        if has_matrix > 1 {
            return Err(malformed());
        }
        let raw = c.take(n * 2)?;
        let mut points = c.points(if options & 4 != 0 && n > 0 { 1 } else { n }, 0)?;
        let extra = if has_matrix != 0 {
            c.matrix()?
        } else {
            Matrix::default()
        };
        c.finish()?;
        if options & 1 == 0 || options & 2 != 0 {
            p.warn(
                "emfplus_text_skipped",
                "glyph-index or vertical driver strings are not supported",
                true,
            );
            return Ok(());
        }
        let Some(Object::Font { font, unit }) = font else {
            return Ok(());
        };
        let data: Vec<_> = raw
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let text = String::from_utf16(&data).map_err(|_| malformed())?;
        if text.chars().count() != n {
            p.warn(
                "emfplus_text_skipped",
                "surrogate-pair driver positioning is not supported",
                true,
            );
            return Ok(());
        }
        if options & 4 != 0 {
            if points.is_empty() {
                return Ok(());
            }
            let [mut x, y] = points[0];
            points.clear();
            let em = font.height.abs() * self.units(unit)?[1] / self.units(self.state.unit)?[1];
            let logical = Font {
                height: -em,
                ..(*font).clone()
            };
            for ch in text.chars() {
                points.push([x, y]);
                x += fonts::advance(&logical, &ch.to_string())?;
            }
            p.warn(
                "emfplus_text_approximation",
                "driver advance uses installed font metrics",
                true,
            );
        }
        if !points.is_empty() {
            self.text_command(
                text,
                Object::Font { font, unit },
                brush,
                points[0],
                points,
                0,
                extra.then(self.matrix()?),
                Arc::clone(&self.state.clip),
                p,
            )?;
        }
        Ok(())
    }
}
