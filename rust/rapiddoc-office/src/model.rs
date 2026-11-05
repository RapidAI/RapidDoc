//! Shared semantic values. XML/media payloads are shared; JSON is an output boundary.
use crate::Format;
use std::sync::Arc;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct TextStyle {
    pub font_family: Option<Arc<str>>,
    pub size_pt: Option<f64>,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strike: bool,
    pub color: Option<Arc<str>>,
    pub vertical: VerticalAlign,
}
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum VerticalAlign {
    #[default]
    Baseline,
    Superscript,
    Subscript,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextRun {
    pub text: String,
    pub style: Arc<TextStyle>,
    pub hyperlink: Option<String>,
}
#[derive(Debug, Clone, PartialEq)]
pub enum Inline {
    Text(TextRun),
    Equation { latex: String, display: bool },
    Image { asset: usize, description: String },
    Break,
    Tab,
}
#[derive(Debug)]
pub struct Paragraph {
    pub content: Vec<Inline>,
    pub heading_level: Option<u8>,
    pub list: Option<ListItem>,
}
#[derive(Debug)]
pub struct ListItem {
    pub definition: String,
    pub level: u32,
    pub label: String,
}
#[derive(Debug)]
pub struct Table {
    pub rows: usize,
    pub columns: usize,
    pub cells: Vec<Cell>,
}
#[derive(Debug)]
pub struct Cell {
    pub row: usize,
    pub column: usize,
    pub row_span: usize,
    pub column_span: usize,
    pub content: Vec<Block>,
}
#[derive(Debug)]
pub enum Block {
    Paragraph(Paragraph),
    Table(Table),
    Image { asset: usize, description: String },
    Equation { latex: String },
}
#[derive(Debug)]
pub struct Asset {
    pub package_part: String,
    pub content_type: String,
    pub bytes: Arc<[u8]>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitKind {
    LogicalPage,
    Worksheet,
    Slide,
}
#[derive(Debug)]
pub struct Unit {
    pub kind: UnitKind,
    pub name: String,
    pub blocks: Vec<Block>,
}
#[derive(Debug)]
pub struct Diagnostic {
    pub code: String,
    pub part: Option<String>,
    pub message: String,
}
#[derive(Debug)]
pub struct Document {
    pub format: Format,
    pub units: Vec<Unit>,
    pub assets: Vec<Asset>,
    pub diagnostics: Vec<Diagnostic>,
}
