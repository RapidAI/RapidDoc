// Copyright (c) 2026 Xiaomeng Zhao (myhloli). SPDX-License-Identifier: MIT
use super::{
    geometry::{self, Builder},
    header::{self, Header},
    model::*,
    resource,
};
use crate::{Error, ErrorKind, Result};
use serde::Serialize;
use std::{collections::BTreeMap, sync::Arc};
#[derive(Default, Serialize)]
pub struct Budget {
    pub points: usize,
    pub commands: usize,
    pub clip_operations: usize,
    pub object_bytes: usize,
    pub decoded_bytes: usize,
}
#[derive(Serialize)]
pub struct GeometryDocument {
    pub source_format: header::Source,
    pub emfplus_mode: header::PlusMode,
    pub bounds: Rect,
    pub width: u64,
    pub height: u64,
    pub commands: Vec<Command>,
    pub diagnostics: Vec<Diagnostic>,
    pub partial: bool,
}
pub struct Playback {
    pub header: Header,
    pub state: State,
    pub state_stack: Vec<State>,
    pub emf_objects: BTreeMap<u32, Object>,
    pub wmf_objects: Vec<Option<Object>>,
    pub path_builder: Builder,
    pub path_active: bool,
    pub path_ready: bool,
    pub commands: Vec<Command>,
    pub partial: bool,
    pub budget: Budget,
    diagnostics: Vec<Diagnostic>,
    omitted: usize,
    location: Location,
}
impl Playback {
    pub fn new(header: Header) -> Self {
        let state = State {
            device_pixels_per_mm: header.device_pixels_per_mm,
            ..State::default()
        };
        let objects = header.wmf_object_count.min(65536);
        Self {
            header,
            state,
            state_stack: vec![],
            emf_objects: BTreeMap::new(),
            wmf_objects: vec![None; objects],
            path_builder: Builder::default(),
            path_active: false,
            path_ready: false,
            commands: vec![],
            partial: false,
            budget: Budget::default(),
            diagnostics: vec![],
            omitted: 0,
            location: Location::default(),
        }
    }
    pub fn set_record(&mut self, kind: u32, index: usize, offset: usize) {
        self.location = Location {
            record_type: Some(kind),
            record_index: Some(index),
            offset: Some(offset),
        };
    }
    fn add_diagnostic(&mut self, code: &str, message: &str, location: Location) {
        if self.diagnostics.len() >= 256 {
            self.omitted += 1;
            return;
        }
        self.diagnostics.push(Diagnostic {
            code: code.into(),
            level: "warning".into(),
            message: message.into(),
            record_type: location.record_type,
            record_index: location.record_index,
            offset: location.offset,
        });
    }
    pub fn warn(&mut self, code: &str, message: &str, partial: bool) {
        self.partial |= partial;
        self.add_diagnostic(code, message, self.location);
    }
    pub fn diagnostics(&self) -> Vec<Diagnostic> {
        let mut out = self.diagnostics.clone();
        if self.omitted > 0 {
            out.push(Diagnostic {
                code: "diagnostics_truncated".into(),
                level: "warning".into(),
                message: format!("omitted {} additional metafile diagnostics", self.omitted),
                record_type: None,
                record_index: None,
                offset: None,
            });
        }
        out
    }
    pub fn charge_points(&mut self, count: usize) -> Result<()> {
        if count > 1_000_000 {
            return Err(resource(
                "metafile record exceeds Python max_points_per_record",
            ));
        }
        self.budget.points += count;
        if self.budget.points > 5_000_000 {
            return Err(resource("metafile exceeds Python max_total_points"));
        }
        Ok(())
    }
    pub fn save_dc(&mut self) -> Result<()> {
        if self.state_stack.len() >= 64 {
            return Err(resource("metafile exceeds Python max_state_depth"));
        }
        self.state_stack.push(self.state.clone());
        Ok(())
    }
    pub fn restore_dc(&mut self, level: i32) {
        if self.state_stack.is_empty() {
            self.warn(
                "restore_dc_underflow",
                "RestoreDC ignored because the state stack is empty",
                true,
            );
            return;
        }
        let target = if level < 0 {
            self.state_stack.len() as i64 + i64::from(level)
        } else if level > 0 {
            i64::from(level) - 1
        } else {
            self.warn("restore_dc_zero", "RestoreDC level 0 is invalid", true);
            return;
        };
        if target < 0 || target as usize >= self.state_stack.len() {
            self.warn(
                "restore_dc_out_of_range",
                &format!("RestoreDC level is out of range: {level}"),
                true,
            );
            return;
        }
        self.state = self.state_stack[target as usize].clone();
        self.state_stack.truncate(target as usize);
    }
    pub fn logical_matrix(&mut self) -> Result<Matrix> {
        let s = &self.state;
        let (wx, wy) = (s.window_origin[0], s.window_origin[1]);
        let (vx, vy) = (s.viewport_origin[0], s.viewport_origin[1]);
        let scale = match s.map_mode {
            7 | 8 => {
                if s.window_extent.contains(&0.) {
                    self.warn(
                        "zero_window_extent",
                        "zero window extent was replaced by identity mapping",
                        true,
                    );
                    [1., 1.]
                } else {
                    [
                        s.viewport_extent[0] / s.window_extent[0],
                        s.viewport_extent[1] / s.window_extent[1],
                    ]
                }
            }
            2 => [
                s.device_pixels_per_mm[0] / 10.,
                -s.device_pixels_per_mm[1] / 10.,
            ],
            3 => [
                s.device_pixels_per_mm[0] / 100.,
                -s.device_pixels_per_mm[1] / 100.,
            ],
            4 => [
                s.device_pixels_per_mm[0] * 25.4 / 100.,
                -s.device_pixels_per_mm[1] * 25.4 / 100.,
            ],
            5 => [
                s.device_pixels_per_mm[0] * 25.4 / 1000.,
                -s.device_pixels_per_mm[1] * 25.4 / 1000.,
            ],
            6 => [
                s.device_pixels_per_mm[0] * 25.4 / 1440.,
                -s.device_pixels_per_mm[1] * 25.4 / 1440.,
            ],
            _ => [1., 1.],
        };
        let matrix = self.state.world_transform.then(Matrix {
            a: scale[0],
            d: scale[1],
            e: vx - wx * scale[0],
            f: vy - wy * scale[1],
            ..Matrix::default()
        });
        if [matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f]
            .iter()
            .any(|v| !v.is_finite() || v.abs() > 1e12)
        {
            return Err(resource(
                "metafile cumulative transform exceeds Python coordinate budget",
            ));
        }
        Ok(matrix)
    }
    pub fn map_point(&mut self, p: Point) -> Result<Point> {
        let p = self.logical_matrix()?.point(p);
        Self::finite_point(p)
    }
    pub fn map_vector(&mut self, p: Point) -> Result<Point> {
        let p = self.logical_matrix()?.vector(p);
        Self::finite_point(p)
    }
    fn finite_point(p: Point) -> Result<Point> {
        if p.iter().any(|v| !v.is_finite() || v.abs() > 1e12) {
            Err(resource(
                "metafile coordinate exceeds Python coordinate budget",
            ))
        } else {
            Ok(p)
        }
    }
    pub fn resolved_pen(&mut self) -> Result<Pen> {
        let mut pen = (*self.state.pen).clone();
        if pen.null {
            return Ok(pen);
        }
        if pen.cosmetic {
            pen.width = pen.width.max(1.);
        } else {
            let p = self.map_vector([if pen.width == 0. { 1. } else { pen.width }, 0.])?;
            pen.width = p[0].hypot(p[1]).max(1.);
        }
        Ok(pen)
    }
    pub fn append(&mut self, command: impl Into<Command>) -> Result<()> {
        let mut command = command.into();
        let clips = match &command {
            Command::Path(c) => c.clip.len(),
            Command::Image(c) => c.clip.len(),
            Command::Text(c) => c.clip.len(),
            Command::Clear(c) => c.clip.len(),
        };
        self.budget.commands += 1;
        self.budget.clip_operations += clips;
        if self.budget.commands > 100_000 {
            return Err(resource("metafile exceeds Python max_commands"));
        }
        if self.budget.clip_operations > 100_000 {
            return Err(resource(
                "metafile exceeds Python max_total_clip_operations",
            ));
        }
        match &mut command {
            Command::Path(c) => c.location = self.location,
            Command::Text(c) => c.location = self.location,
            Command::Clear(c) => c.location = self.location,
            Command::Image(c) => {
                if let ImagePayload::Dib(image) = &c.image {
                    self.budget.decoded_bytes += super::bitmap::validate(image)?;
                }
                if self.budget.decoded_bytes > 128 * 1024 * 1024 {
                    return Err(resource("metafile exceeds Python max_decoded_bitmap_bytes"));
                }
                c.location = self.location;
            }
        }
        self.commands.push(command);
        Ok(())
    }
    pub fn emit(&mut self, path: Path, stroke: bool, fill: bool) -> Result<()> {
        if self.path_active {
            self.path_builder.extend(&path);
            return Ok(());
        }
        if path.segments.is_empty() {
            return Ok(());
        }
        let pen = self.resolved_pen()?;
        self.append(PathCommand {
            path,
            pen,
            brush: Arc::clone(&self.state.brush),
            stroke: stroke && !self.state.pen.null,
            fill: fill && self.state.brush.kind != "null",
            fill_rule: self.fill_rule().into(),
            clip: Arc::clone(&self.state.clip),
            rop2: self.state.rop2,
            miter_limit: self.state.miter_limit,
            location: Location::default(),
        })
    }
    pub fn fill_rule(&self) -> &str {
        if self.state.polygon_fill_mode == 1 {
            "evenodd"
        } else {
            "nonzero"
        }
    }
    pub fn emit_logical(&mut self, path: Path, stroke: bool, fill: bool) -> Result<()> {
        let m = self.logical_matrix()?;
        self.emit(geometry::transform(&path, m), stroke, fill)
    }
    pub fn move_to(&mut self, p: Point) -> Result<()> {
        self.state.current_position = p;
        if self.path_active {
            let mapped = self.map_point(p)?;
            self.path_builder.move_to(mapped);
        }
        Ok(())
    }
    fn ensure_current(&mut self) -> Result<()> {
        let mapped = self.map_point(self.state.current_position)?;
        if self.path_builder.current != Some(mapped) {
            self.path_builder.move_to(mapped);
        }
        Ok(())
    }
    pub fn line_to(&mut self, p: Point) -> Result<()> {
        if self.path_active {
            self.ensure_current()?;
            let mapped = self.map_point(p)?;
            self.path_builder.line_to(mapped);
        } else {
            let points = [
                self.map_point(self.state.current_position)?,
                self.map_point(p)?,
            ];
            self.emit(geometry::polygon(&points, false), true, false)?;
        }
        self.state.current_position = p;
        Ok(())
    }
    pub fn polyline_to(&mut self, points: &[Point]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        if self.path_active {
            self.ensure_current()?;
            for p in points {
                let mapped = self.map_point(*p)?;
                self.path_builder.line_to(mapped);
            }
        } else {
            let mut mapped = vec![self.map_point(self.state.current_position)?];
            for p in points {
                mapped.push(self.map_point(*p)?);
            }
            self.emit(geometry::polygon(&mapped, false), true, false)?;
        }
        self.state.current_position = *points.last().unwrap();
        Ok(())
    }
    pub fn bezier_to(&mut self, points: &[Point]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        if points.len() % 3 != 0 {
            self.warn(
                "invalid_bezier_points",
                "PolyBezierTo point count is not a multiple of three",
                true,
            );
        }
        let points = &points[..points.len() - points.len() % 3];
        if points.is_empty() {
            return Ok(());
        }
        if self.path_active {
            self.ensure_current()?;
        }
        let mut b = Builder::default();
        if !self.path_active {
            b.move_to(self.map_point(self.state.current_position)?);
        }
        for p in points.chunks_exact(3) {
            let a = self.map_point(p[0])?;
            let c = self.map_point(p[1])?;
            let end = self.map_point(p[2])?;
            if self.path_active {
                self.path_builder.cubic_to(a, c, end);
            } else {
                b.cubic_to(a, c, end);
            }
        }
        if !self.path_active {
            self.emit(b.take(), true, false)?;
        }
        self.state.current_position = *points.last().unwrap();
        Ok(())
    }
    pub fn connected_arc(&mut self, path: &Path) -> Result<()> {
        if path.segments.is_empty() {
            return Ok(());
        }
        let m = self.logical_matrix()?;
        let mapped = geometry::transform(path, m);
        let start = mapped.segments[0].points[0];
        let tail = Path {
            segments: mapped.segments[1..].into(),
        };
        if self.path_active {
            self.ensure_current()?;
            self.path_builder.line_to(start);
            self.path_builder.extend(&tail);
        } else {
            let mut b = Builder::default();
            b.move_to(self.map_point(self.state.current_position)?);
            b.line_to(start);
            b.extend(&tail);
            self.emit(b.take(), true, false)?;
        }
        self.state.current_position = *path
            .segments
            .last()
            .and_then(|s| s.points.last())
            .ok_or_else(super::malformed)?;
        Ok(())
    }
    pub fn begin_path(&mut self) {
        if self.path_active {
            self.warn(
                "begin_path_while_active",
                "BeginPath replaced an already active path bracket",
                true,
            );
        }
        self.path_builder.clear();
        self.path_active = true;
        self.path_ready = false;
    }
    pub fn end_path(&mut self) {
        if !self.path_active {
            self.warn(
                "end_path_without_begin",
                "EndPath ignored without a matching BeginPath",
                true,
            );
            return;
        }
        self.path_active = false;
        self.path_ready = true;
    }
    pub fn consume_path(&mut self) -> Path {
        if !self.path_ready {
            self.warn(
                "path_not_ready",
                "path operation ignored because no completed path exists",
                true,
            );
            return Path::default();
        }
        self.path_ready = false;
        self.path_builder.take()
    }
    pub fn add_clip(&mut self, path: Path, mode: u32) -> Result<()> {
        let name = match mode {
            1 => "and",
            2 => "or",
            3 => "xor",
            4 => "diff",
            5 => "copy",
            _ => {
                self.warn(
                    "unsupported_clip_mode",
                    &format!("unsupported clip combine mode: {mode}"),
                    true,
                );
                return Ok(());
            }
        };
        let clip = Clip {
            path,
            mode: name.into(),
            fill_rule: self.fill_rule().into(),
        };
        if mode == 5 {
            self.state.clip = Arc::from([clip]);
        } else {
            if self.state.clip.len() >= 64 {
                return Err(resource("metafile exceeds Python max_clip_operations"));
            }
            let mut clips = self.state.clip.to_vec();
            clips.push(clip);
            self.state.clip = clips.into();
        }
        Ok(())
    }
    pub fn reset_clip(&mut self) {
        self.state.clip = Arc::from([]);
    }
    pub fn create_wmf_object(&mut self, object: Object) -> Result<()> {
        if let Some(slot) = self.wmf_objects.iter_mut().find(|s| s.is_none()) {
            *slot = Some(object);
            return Ok(());
        }
        if self.wmf_objects.len() >= 65536 {
            return Err(resource("WMF object table exceeds Python max_objects"));
        }
        self.wmf_objects.push(Some(object));
        Ok(())
    }
    fn select(&mut self, object: Object) {
        match object {
            Object::Pen(p) => self.state.pen = p,
            Object::Brush(b) => self.state.brush = b,
            Object::Font(f) => self.state.font = f,
        }
    }
    pub fn select_wmf(&mut self, handle: u32) {
        self.select_handle(handle, false);
    }
    pub fn select_emf(&mut self, handle: u32) {
        self.select_handle(handle, true);
    }
    fn select_handle(&mut self, handle: u32, emf: bool) {
        let mask = if emf { 0x80000000 } else { 0x8000 };
        let label = if emf { "EMF" } else { "WMF" };
        let object = if handle & mask != 0 {
            match stock(handle & (mask - 1)) {
                Some(o) => o,
                None => {
                    self.warn(
                        "unknown_stock_object",
                        &format!("unknown {label} stock object: {handle:#x}"),
                        true,
                    );
                    return;
                }
            }
        } else {
            match if emf {
                self.emf_objects.get(&handle).cloned()
            } else {
                self.wmf_objects.get(handle as usize).cloned().flatten()
            } {
                Some(o) => o,
                None => {
                    self.warn(
                        "missing_object",
                        &format!("{label} object handle does not exist: {handle}"),
                        true,
                    );
                    return;
                }
            }
        };
        self.select(object);
    }
    pub fn finish(mut self, size: Option<[u64; 2]>) -> Result<GeometryDocument> {
        let command_bounds = geometry::union(self.commands.iter().map(|c| match c {
            Command::Path(c) => geometry::bounds(&c.path),
            Command::Image(c) => geometry::bounds(&geometry::polygon(&c.destination, true)),
            Command::Text(c) => super::text::bounds(c),
            Command::Clear(c) => Some(c.bounds),
        }));
        let mut bounds = self.header.bounds;
        if bounds.is_none_or(|b| b.width().abs() < 1e-9 || b.height().abs() < 1e-9) {
            bounds = command_bounds;
        }
        let bounds = bounds
            .filter(|b| b.width().abs() >= 1e-9 && b.height().abs() >= 1e-9)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Unsupported,
                    "metafile contains no visible output",
                )
            })?
            .normalized();
        let requested = size.or(self.header.pixel_size).unwrap_or_else(|| {
            [
                bounds.width().round_ties_even().max(1.) as u64,
                bounds.height().round_ties_even().max(1.) as u64,
            ]
        });
        let ([width, height], downscaled) = header::bounded_canvas(requested)?;
        if downscaled {
            self.add_diagnostic(
                "canvas_downscaled",
                &format!(
                    "metafile canvas was downscaled from {}x{} to {width}x{height}",
                    requested[0], requested[1]
                ),
                Location::default(),
            );
        }
        let diagnostics = self.diagnostics();
        Ok(GeometryDocument {
            source_format: self.header.source_format,
            emfplus_mode: self.header.emfplus_mode,
            bounds,
            width,
            height,
            commands: self.commands,
            diagnostics,
            partial: self.partial,
        })
    }
}
pub fn make_pen(style: u32, width: f64, color: u32, extended: bool) -> Pen {
    let unit = width.abs().max(1.);
    let dashes: Vec<f64> = match style & 15 {
        1 => vec![6., 3.],
        2 => vec![1., 2.],
        3 => vec![6., 2., 1., 2.],
        4 => vec![6., 2., 1., 2., 1., 2.],
        8 => vec![1., 1.],
        _ => vec![],
    }
    .into_iter()
    .map(|v| v * unit)
    .collect();
    Pen {
        color: Color::colorref(color),
        width: unit,
        style,
        cosmetic: !if extended {
            style & 0x10000 != 0
        } else {
            width.abs() > 1.
        },
        null: style & 15 == 5,
        cap: if style & 0x100 != 0 {
            "square"
        } else if style & 0x200 != 0 {
            "flat"
        } else {
            "round"
        }
        .into(),
        join: if style & 0x1000 != 0 {
            "bevel"
        } else if style & 0x2000 != 0 {
            "miter"
        } else {
            "round"
        }
        .into(),
        dashes: dashes.into(),
    }
}
pub fn make_brush(style: u32, color: u32, hatch: u32) -> Brush {
    if style == 1 {
        return Brush {
            kind: "null".into(),
            ..Brush::default()
        };
    }
    Brush {
        kind: match style {
            2 => "hatch",
            3 | 5..=8 => "pattern",
            _ => "solid",
        }
        .into(),
        color: Color::colorref(color),
        hatch,
        pattern: None,
    }
}
pub fn stock(index: u32) -> Option<Object> {
    Some(match index {
        0..=6 | 18 => Object::Brush(Arc::new(match index {
            1..=3 => {
                let v = [192, 128, 64][(index - 1) as usize];
                Brush {
                    color: Color {
                        red: v,
                        green: v,
                        blue: v,
                        alpha: 255,
                    },
                    ..Brush::default()
                }
            }
            4 => Brush {
                color: BLACK,
                ..Brush::default()
            },
            5 | 6 => Brush {
                kind: "null".into(),
                ..Brush::default()
            },
            _ => Brush::default(),
        })),
        7 | 8 | 9 | 19 => Object::Pen(Arc::new(Pen {
            color: if index == 7 { WHITE } else { BLACK },
            null: index == 9,
            ..Pen::default()
        })),
        10..=17 => Object::Font(Arc::new(Font {
            face_name: if matches!(index, 10 | 11 | 16) {
                "Courier New"
            } else {
                "Arial"
            }
            .into(),
            ..Font::default()
        })),
        _ => return None,
    })
}
