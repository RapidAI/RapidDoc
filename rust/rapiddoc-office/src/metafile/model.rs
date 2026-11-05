// Copyright (c) 2026 Xiaomeng Zhao (myhloli). SPDX-License-Identifier: MIT
use serde::{Deserialize, Serialize};
use std::sync::Arc;
pub type Point = [f64; 2];
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub left: f64,
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
}
impl Rect {
    pub fn width(self) -> f64 {
        self.right - self.left
    }
    pub fn height(self) -> f64 {
        self.bottom - self.top
    }
    pub fn normalized(self) -> Self {
        Self {
            left: self.left.min(self.right),
            top: self.top.min(self.bottom),
            right: self.left.max(self.right),
            bottom: self.top.max(self.bottom),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}
impl Default for Matrix {
    fn default() -> Self {
        Self {
            a: 1.,
            b: 0.,
            c: 0.,
            d: 1.,
            e: 0.,
            f: 0.,
        }
    }
}
impl Matrix {
    pub fn point(self, p: Point) -> Point {
        [
            self.a * p[0] + self.c * p[1] + self.e,
            self.b * p[0] + self.d * p[1] + self.f,
        ]
    }
    pub fn vector(self, p: Point) -> Point {
        [self.a * p[0] + self.c * p[1], self.b * p[0] + self.d * p[1]]
    }
    pub fn then(self, n: Self) -> Self {
        Self {
            a: n.a * self.a + n.c * self.b,
            b: n.b * self.a + n.d * self.b,
            c: n.a * self.c + n.c * self.d,
            d: n.b * self.c + n.d * self.d,
            e: n.a * self.e + n.c * self.f + n.e,
            f: n.b * self.e + n.d * self.f + n.f,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verb {
    M,
    L,
    C,
    Z,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub verb: Verb,
    pub points: Vec<Point>,
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Path {
    pub segments: Arc<[Segment]>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Color {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub alpha: u8,
}
pub const BLACK: Color = Color {
    red: 0,
    green: 0,
    blue: 0,
    alpha: 255,
};
pub const WHITE: Color = Color {
    red: 255,
    green: 255,
    blue: 255,
    alpha: 255,
};
impl Color {
    pub fn colorref(value: u32) -> Self {
        Self {
            red: value as u8,
            green: (value >> 8) as u8,
            blue: (value >> 16) as u8,
            alpha: 255,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pen {
    pub color: Color,
    pub width: f64,
    pub style: u32,
    pub cosmetic: bool,
    pub null: bool,
    pub cap: String,
    pub join: String,
    pub dashes: Arc<[f64]>,
}
impl Default for Pen {
    fn default() -> Self {
        Self {
            color: BLACK,
            width: 1.,
            style: 0,
            cosmetic: true,
            null: false,
            cap: "round".into(),
            join: "round".into(),
            dashes: Arc::from([]),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Brush {
    pub kind: String,
    pub color: Color,
    pub hatch: u32,
    pub pattern: Option<Arc<[u8]>>,
}
impl Default for Brush {
    fn default() -> Self {
        Self {
            kind: "solid".into(),
            color: WHITE,
            hatch: 0,
            pattern: None,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Font {
    pub face_name: String,
    pub height: f64,
    pub width: f64,
    pub weight: i32,
    pub italic: bool,
    pub underline: bool,
    pub strikeout: bool,
    pub charset: u8,
    pub escapement: f64,
    pub orientation: f64,
}
impl Default for Font {
    fn default() -> Self {
        Self {
            face_name: "Arial".into(),
            height: -12.,
            width: 0.,
            weight: 400,
            italic: false,
            underline: false,
            strikeout: false,
            charset: 1,
            escapement: 0.,
            orientation: 0.,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    pub path: Path,
    pub mode: String,
    pub fill_rule: String,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Location {
    pub record_type: Option<u32>,
    pub record_index: Option<usize>,
    pub offset: Option<usize>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub level: String,
    pub message: String,
    pub record_type: Option<u32>,
    pub record_index: Option<usize>,
    pub offset: Option<usize>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathCommand {
    pub path: Path,
    pub pen: Pen,
    pub brush: Arc<Brush>,
    pub stroke: bool,
    pub fill: bool,
    pub fill_rule: String,
    pub clip: Arc<[Clip]>,
    pub rop2: u32,
    pub miter_limit: f64,
    pub location: Location,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DibPayload {
    pub header: Arc<[u8]>,
    pub bits: Arc<[u8]>,
    pub use_source_alpha: bool,
}
#[derive(Debug, Clone, Serialize)]
pub struct ImageCommand {
    pub image: ImagePayload,
    pub destination: [Point; 4],
    pub source: Option<Rect>,
    pub rop: u32,
    pub stretch_mode: u32,
    pub constant_alpha: u8,
    pub clip: Arc<[Clip]>,
    pub location: Location,
}
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Command {
    Path(PathCommand),
    Image(ImageCommand),
    Text(TextCommand),
    Clear(ClearCommand),
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ImagePayload {
    Dib(DibPayload),
    Encoded { png: Arc<[u8]> },
}
impl From<DibPayload> for ImagePayload {
    fn from(v: DibPayload) -> Self {
        Self::Dib(v)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearCommand {
    pub color: Color,
    pub bounds: Rect,
    pub clip: Arc<[Clip]>,
    pub location: Location,
}
impl From<ClearCommand> for Command {
    fn from(v: ClearCommand) -> Self {
        Self::Clear(v)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextAlignment {
    pub horizontal: String,
    pub vertical: String,
}
impl TextAlignment {
    pub fn from_gdi(v: u32) -> Self {
        Self {
            horizontal: if v & 6 == 6 {
                "center"
            } else if v & 2 != 0 {
                "right"
            } else {
                "left"
            }
            .into(),
            vertical: if v & 24 == 24 {
                "baseline"
            } else if v & 8 != 0 {
                "bottom"
            } else {
                "top"
            }
            .into(),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextCommand {
    pub text: String,
    pub origin: Point,
    pub positions: Vec<Point>,
    pub font: Arc<Font>,
    pub font_height: f64,
    pub rotation: f64,
    pub alignment: TextAlignment,
    pub color: Color,
    pub background_color: Color,
    pub opaque: bool,
    pub bounds: Option<Rect>,
    pub clip: Arc<[Clip]>,
    pub advance_end: Option<Point>,
    pub location: Location,
}
impl From<TextCommand> for Command {
    fn from(v: TextCommand) -> Self {
        Self::Text(v)
    }
}
impl From<PathCommand> for Command {
    fn from(value: PathCommand) -> Self {
        Self::Path(value)
    }
}
impl From<ImageCommand> for Command {
    fn from(value: ImageCommand) -> Self {
        Self::Image(value)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub pen: Arc<Pen>,
    pub brush: Arc<Brush>,
    pub font: Arc<Font>,
    pub text_color: Color,
    pub background_color: Color,
    pub background_mode: u32,
    pub text_align: u32,
    pub map_mode: i32,
    pub window_origin: Point,
    pub window_extent: Point,
    pub viewport_origin: Point,
    pub viewport_extent: Point,
    pub world_transform: Matrix,
    pub current_position: Point,
    pub polygon_fill_mode: u32,
    pub rop2: u32,
    pub stretch_mode: u32,
    pub arc_direction: u32,
    pub miter_limit: f64,
    pub brush_origin: Point,
    pub clip: Arc<[Clip]>,
    pub device_pixels_per_mm: Point,
}
impl Default for State {
    fn default() -> Self {
        Self {
            pen: Arc::new(Pen::default()),
            brush: Arc::new(Brush::default()),
            font: Arc::new(Font::default()),
            text_color: BLACK,
            background_color: WHITE,
            background_mode: 2,
            text_align: 0,
            map_mode: 1,
            window_origin: [0., 0.],
            window_extent: [1., 1.],
            viewport_origin: [0., 0.],
            viewport_extent: [1., 1.],
            world_transform: Matrix::default(),
            current_position: [0., 0.],
            polygon_fill_mode: 1,
            rop2: 13,
            stretch_mode: 1,
            arc_direction: 2,
            miter_limit: 10.,
            brush_origin: [0., 0.],
            clip: Arc::from([]),
            device_pixels_per_mm: [96. / 25.4, 96. / 25.4],
        }
    }
}
#[derive(Debug, Clone)]
pub enum Object {
    Pen(Arc<Pen>),
    Brush(Arc<Brush>),
    Font(Arc<Font>),
}
