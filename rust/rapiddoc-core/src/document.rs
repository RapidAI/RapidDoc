//! PDF 事实层的自有数据契约；不把原生阅读顺序误称为最终版面顺序。

use crate::Rect;
use serde::{Deserialize, Serialize};

/// 错误以外的能力诊断，页号为原始文档的零基索引。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub page_index: usize,
    pub message: String,
}

/// 单字符原始 PDFium 事实，字框和原点为 PDF 原生左下原点 point。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Character {
    pub index: usize,
    pub unicode: u32,
    pub is_generated: bool,
    pub text: Option<String>,
    pub font_name: String,
    pub font_size_pt: f64,
    pub angle_degrees: f64,
    pub origin: [f64; 2],
    pub tight_bounds: Rect,
    pub loose_bounds: Rect,
    pub visual_tight_bounds: Rect,
    pub visual_loose_bounds: Rect,
    pub visual_origin: [f64; 2],
}

/// 已释放 PDFium 页句柄后仍可安全传递的页面事实。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page {
    pub page_index: usize,
    pub width_pt: f64,
    pub height_pt: f64,
    pub text: String,
    pub object_count: usize,
    pub characters: Vec<Character>,
    pub page_bounds: Rect,
    pub rotation: u16,
    pub visual_width_pt: f64,
    pub visual_height_pt: f64,
    pub paths: Vec<VectorPath>,
    pub images: Vec<ImageFact>,
    pub links: Vec<LinkFact>,
    pub form_count: usize,
    pub drawing_lines: Vec<DrawingLine>,
}

/// 表格规则可用的轴线候选；bounds 为经过页面与 Form clip 的视觉 point 框。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawingLine {
    pub start: [f64; 2],
    pub end: [f64; 2],
    pub bounds: Rect,
    pub width: f64,
    pub orientation: String,
}

/// 路径片段保留控制点和闭合标记，point 为累积 Form 变换后的原生 PDF 坐标。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathSegment {
    pub kind: String,
    pub point: [f64; 2],
    pub local_point: [f64; 2],
    pub close: bool,
}

/// 原始矢量路径事实；clip_bounds 是保守矩形，不冒充任意形状的精确裁剪。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorPath {
    pub object_path: Vec<usize>,
    pub bounds: Rect,
    pub visual_bounds: Rect,
    pub evidence_bounds: Option<Rect>,
    pub clip_bounds: Option<Rect>,
    pub matrix: [f64; 6],
    pub segments: Vec<PathSegment>,
    pub fill_rule: String,
    pub fill_rgba: [u8; 4],
    pub stroke: bool,
    pub stroke_rgba: [u8; 4],
    pub stroke_width_local: f64,
    pub fill_visible: bool,
    pub stroke_visible: bool,
}

/// 图片边界事实，不触发位图解码或把未导出的图片伪装成资产。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageFact {
    pub object_path: Vec<usize>,
    pub bounds: Rect,
    pub visual_bounds: Rect,
    pub clip_bounds: Option<Rect>,
}

/// 链接注解的原始框与 URI；内部跳转的 uri 为 None，保留注解记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkFact {
    pub index: usize,
    pub bounds: Rect,
    pub visual_bounds: Rect,
    pub uri: Option<String>,
}

/// 当前版本的 PDF 事实集合；字符坐标明确保留原生坐标系。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub schema_version: u32,
    pub coordinate_system: String,
    pub source_page_count: usize,
    pub pages: Vec<Page>,
    pub diagnostics: Vec<Diagnostic>,
}
