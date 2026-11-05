// Portions derived from pdftext 0.7.1, Copyright Vik Paruchuri, Apache-2.0.
// Extended DocVortex character metadata and surrogate restoration: MIT.
//! DocVortex get_chars 的去重前契约。与 pdftext 字符分组接口独立。
use crate::{legacy_vectors::hypot, with_raw_symbols};
use pdfium_render::prelude::*;
use rapiddoc_core::{Error, ErrorCode, Limits};
use serde::{Deserialize, Serialize};
mod dedup;
mod visibility;
pub use dedup::{deduplicate, read_geometry, read_visible_geometry};
use std::{collections::HashMap, path::Path};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Font {
    name: String,
    flags: i32,
    size: f64,
    weight: i32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Character {
    bbox: [f64; 4],
    char: String,
    rotation: f64,
    font: Font,
    char_idx: usize,
    source_indices: Vec<usize>,
    raw_code: u32,
    text_object_id: Option<usize>,
    text_render_mode: Option<i32>,
    writing_angle: f64,
    origin: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_generated: Option<bool>,
    // 外层None表示未请求扩展几何，内层None表示已请求但没有有效框。
    #[serde(
        default,
        deserialize_with = "optional_box",
        skip_serializing_if = "Option::is_none"
    )]
    loose_bbox: Option<Option<[f64; 4]>>,
    #[serde(
        default,
        deserialize_with = "optional_box",
        skip_serializing_if = "Option::is_none"
    )]
    tight_bbox: Option<Option<[f64; 4]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text_is_visible: Option<bool>,
}
fn optional_box<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Option<[f64; 4]>>, D::Error> {
    Option::<[f64; 4]>::deserialize(deserializer).map(Some)
}
#[derive(Debug, Serialize)]
pub struct Primitives {
    pub page_size: [f64; 2],
    pub page_bbox: [f64; 4],
    pub rotation: i32,
    pub raw_char_count: usize,
    pub chars: Vec<Character>,
    pub restored_chars: Vec<Character>,
}
fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}
struct Handles<'a> {
    api: &'a dyn PdfiumLibraryBindings,
    document: FPDF_DOCUMENT,
    page: FPDF_PAGE,
    text: FPDF_TEXTPAGE,
}
impl Drop for Handles<'_> {
    fn drop(&mut self) {
        // SAFETY: 同库句柄在全局锁内按text/page/document逆序关闭；输入字节仍存活。
        unsafe {
            if !self.text.is_null() {
                self.api.FPDFText_ClosePage(self.text);
            }
            if !self.page.is_null() {
                self.api.FPDF_ClosePage(self.page);
            }
            if !self.document.is_null() {
                self.api.FPDF_CloseDocument(self.document);
            }
        }
    }
}
fn point(p: [f64; 2], f: [f64; 4], rotation: i32) -> [f64; 2] {
    let (w, h) = ((f[2] - f[0]).abs(), (f[3] - f[1]).abs());
    let (x, y) = (p[0] - f[0].min(f[2]), f[1].max(f[3]) - p[1]);
    match rotation {
        90 => [h - y, x],
        180 => [w - x, h - y],
        270 => [y, w - x],
        _ => [x, y],
    }
}
fn visual(b: [f64; 4], f: [f64; 4], rotation: i32) -> Option<[f64; 4]> {
    let ps =
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[1]], [b[2], b[3]]].map(|p| point(p, f, rotation));
    let mut out = [ps[0][0], ps[0][1], ps[0][0], ps[0][1]];
    for p in &ps[1..] {
        out = [
            out[0].min(p[0]),
            out[1].min(p[1]),
            out[2].max(p[0]),
            out[3].max(p[1]),
        ];
    }
    (out.iter().all(|v| v.is_finite()) && out[2] > out[0] && out[3] > out[1]).then_some(out)
}
fn layout(b: [f64; 4], f: [f64; 4], rotation: i32) -> [f64; 4] {
    let (w, h) = ((f[2] - f[0]).abs().ceil(), (f[3] - f[1]).abs().ceil());
    let ys = [h - (b[1] - f[1]), h - (b[3] - f[1])];
    let [x0, y0, x1, y1] = [
        b[0].min(b[2]) - f[0],
        ys[0].min(ys[1]),
        b[0].max(b[2]) - f[0],
        ys[0].max(ys[1]),
    ];
    let b = match rotation {
        90 => [h - y1, x0, h - y0, x1],
        180 => [w - x1, h - y1, w - x0, h - y0],
        270 => [y0, w - x1, y1, w - x0],
        _ => [x0, y0, x1, y1],
    };
    [
        b[0].min(b[2]),
        b[1].min(b[3]),
        b[0].max(b[2]),
        b[1].max(b[3]),
    ]
}
fn writing_angles(chars: &mut [Character]) {
    let mut start = 0;
    while start < chars.len() {
        let mut end = start + 1;
        if let Some(object) = chars[start].text_object_id {
            while end < chars.len() && chars[end].text_object_id == Some(object) {
                end += 1;
            }
            if let Some(origin) = chars[start].origin {
                for next in start + 1..end {
                    if let Some(p) = chars[next].origin {
                        let (dx, dy) = (p[0] - origin[0], p[1] - origin[1]);
                        if hypot(dx, dy) > 0.001 {
                            let angle = dy.atan2(dx);
                            for c in &mut chars[start..end] {
                                c.writing_angle = angle;
                            }
                            break;
                        }
                    }
                }
            }
        }
        start = end;
    }
}
/// 无过滤原始序列的代理对恢复；来源索引和代表字符的原始码值/几何保持旧行为。
fn restore(chars: &[Character]) -> Vec<Character> {
    let mut consumed = vec![false; chars.len()];
    let mut out = Vec::with_capacity(chars.len());
    for c in chars {
        let i = c.char_idx;
        if consumed[i] {
            continue;
        }
        let mut c = c.clone();
        if c.char == "\u{fffd}" && (0xd800..=0xdbff).contains(&c.raw_code) && i + 1 < chars.len() {
            let low = chars[i + 1].raw_code;
            if (0xdc00..=0xdfff).contains(&low) {
                c.char = char::from_u32(0x10000 + ((c.raw_code - 0xd800) << 10) + (low - 0xdc00))
                    .unwrap()
                    .to_string();
                c.source_indices.push(i + 1);
                consumed[i + 1] = true;
            }
        }
        out.push(c);
    }
    out
}
pub fn read(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
    extended: bool,
    limits: &Limits,
) -> Result<Primitives, Error> {
    read_mode(
        library, bytes, page_index, password, extended, false, limits,
    )
}
pub(super) fn read_visible(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
    extended: bool,
    limits: &Limits,
) -> Result<Primitives, Error> {
    read_mode(library, bytes, page_index, password, extended, true, limits)
}
fn read_mode(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
    extended: bool,
    visible_only: bool,
    limits: &Limits,
) -> Result<Primitives, Error> {
    limits.check_input(bytes)?;
    if password.is_some_and(|s| s.contains('\0')) {
        return Err(error(ErrorCode::InvalidInput, "密码包含NUL"));
    }
    with_raw_symbols(library, |api, symbols| {
        // SAFETY: 同一运行库、全程持锁；全部索引/分配受限，结果不携带原生地址。
        unsafe {
            let mut owner = Handles {
                api,
                document: api.FPDF_LoadMemDocument64(bytes, password),
                page: std::ptr::null_mut(),
                text: std::ptr::null_mut(),
            };
            if owner.document.is_null() {
                return Err(error(
                    if matches!(api.FPDF_GetLastError(), 4 | 5) {
                        ErrorCode::Encrypted
                    } else {
                        ErrorCode::CorruptDocument
                    },
                    "无法打开PDF",
                ));
            }
            let pages = api.FPDF_GetPageCount(owner.document);
            if pages < 0 {
                return Err(error(ErrorCode::CorruptDocument, "无效页数"));
            }
            if pages as usize > limits.max_pages {
                return Err(error(ErrorCode::ResourceLimit, "页数超过限制"));
            }
            if page_index >= pages as usize {
                return Err(error(ErrorCode::InvalidInput, "页号越界"));
            }
            owner.page = api.FPDF_LoadPage(owner.document, page_index as i32);
            if owner.page.is_null() {
                return Err(error(ErrorCode::CorruptDocument, "无法打开PDF页"));
            }
            let mut rect = FS_RECTF {
                left: 0.,
                bottom: 0.,
                right: 0.,
                top: 0.,
            };
            if api.FPDF_GetPageBoundingBox(owner.page, &mut rect) == 0 {
                return Err(error(ErrorCode::CorruptDocument, "无法读取页框"));
            }
            let frame = [
                rect.left as f64,
                rect.bottom as f64,
                rect.right as f64,
                rect.top as f64,
            ];
            let size = [
                api.FPDF_GetPageWidthF(owner.page) as f64,
                api.FPDF_GetPageHeightF(owner.page) as f64,
            ];
            let rotation = api.FPDFPage_GetRotation(owner.page) * 90;
            if frame
                .iter()
                .chain(&size)
                .any(|v| !v.is_finite() || v.abs() > 1e9)
                || !matches!(rotation, 0 | 90 | 180 | 270)
            {
                return Err(error(ErrorCode::CorruptDocument, "页面几何无效"));
            }
            owner.text = api.FPDFText_LoadPage(owner.page);
            if owner.text.is_null() {
                return Err(error(ErrorCode::CorruptDocument, "无法打开文字页"));
            }
            let count = api.FPDFText_CountChars(owner.text);
            if count < 0 {
                return Err(error(ErrorCode::CorruptDocument, "无效字符数"));
            }
            if count as usize
                > limits
                    .max_chars_per_page
                    .min(limits.max_total_chars)
                    .min(limits.max_native_chars_per_page)
            {
                return Err(error(ErrorCode::ResourceLimit, "字符数量超过限制"));
            }
            let mut chars = Vec::with_capacity(count as usize);
            let mut objects = HashMap::new();
            let mut font_bytes = 0usize;
            for i in 0..count {
                let code = api.FPDFText_GetUnicode(owner.text, i);
                if code > 0x10ffff {
                    return Err(error(ErrorCode::CorruptDocument, "无效Unicode码位"));
                }
                let angle = api.FPDFText_GetCharAngle(owner.text, i) as f64;
                let loose = if angle == 0. || extended {
                    (api.FPDFText_GetLooseCharBox(owner.text, i, &mut rect) != 0).then_some([
                        rect.left as f64,
                        rect.bottom as f64,
                        rect.right as f64,
                        rect.top as f64,
                    ])
                } else {
                    None
                };
                let tight = if angle != 0. || extended {
                    let (mut l, mut r, mut b, mut t) = (0., 0., 0., 0.);
                    (api.FPDFText_GetCharBox(owner.text, i, &mut l, &mut r, &mut b, &mut t) != 0)
                        .then_some([l, b, r, t])
                } else {
                    None
                };
                let selected = if angle == 0. { loose } else { tight }
                    .ok_or_else(|| error(ErrorCode::CorruptDocument, "无法读取字符框"))?;
                let mut buffer = vec![0u8; 256];
                let mut flags = 0;
                let length = api.FPDFText_GetFontInfo(
                    owner.text,
                    i,
                    buffer.as_mut_ptr().cast(),
                    buffer.len() as _,
                    &mut flags,
                ) as usize;
                if length > rapiddoc_core::resources::limit(1_048_576) {
                    return Err(error(ErrorCode::ResourceLimit, "字体名超过限制"));
                }
                if length > buffer.len() {
                    buffer.resize(length, 0);
                    api.FPDFText_GetFontInfo(
                        owner.text,
                        i,
                        buffer.as_mut_ptr().cast(),
                        length as _,
                        &mut flags,
                    );
                }
                let name = if length == 0 {
                    flags = 0;
                    String::new()
                } else {
                    String::from_utf8_lossy(
                        &buffer[..buffer.iter().position(|b| *b == 0).unwrap_or(buffer.len())],
                    )
                    .into_owned()
                };
                font_bytes = font_bytes
                    .checked_add(name.len())
                    .filter(|n| *n <= rapiddoc_core::resources::limit(8_000_000))
                    .ok_or_else(|| error(ErrorCode::ResourceLimit, "累计字体名超过限制"))?;
                let font = Font {
                    name,
                    flags,
                    size: api.FPDFText_GetFontSize(owner.text, i),
                    weight: crate::font_weight::normalize(
                        api.FPDFText_GetFontWeight(owner.text, i),
                    ),
                };
                if selected
                    .iter()
                    .chain([angle, font.size].iter())
                    .any(|v| !v.is_finite() || v.abs() > 1e15)
                {
                    return Err(error(ErrorCode::CorruptDocument, "字符几何无效"));
                }
                let object = api.FPDFText_GetTextObject(owner.text, i);
                let (object_id, mode) = if object.is_null() {
                    (None, None)
                } else {
                    let next = objects.len();
                    let (id, mode) = *objects
                        .entry(object as usize)
                        .or_insert_with(|| (next, api.FPDFTextObj_GetTextRenderMode(object)));
                    (Some(id), Some(mode))
                };
                let (mut x, mut y) = (0., 0.);
                let origin = if api.FPDFText_GetCharOrigin(owner.text, i, &mut x, &mut y) != 0 {
                    let p = point([x, y], frame, rotation);
                    p.iter().all(|v| v.is_finite()).then_some(p)
                } else {
                    None
                };
                let generated = match api.FPDFText_IsGenerated(owner.text, i) {
                    0 => Some(false),
                    1 => Some(true),
                    _ => None,
                };
                chars.push(Character {
                    bbox: layout(selected, frame, rotation),
                    char: char::from_u32(code).unwrap_or('\u{fffd}').to_string(),
                    rotation: angle,
                    font,
                    char_idx: i as usize,
                    source_indices: vec![i as usize],
                    raw_code: code,
                    text_object_id: object_id,
                    text_render_mode: mode,
                    writing_angle: (rotation as f64) * (std::f64::consts::PI / 180.) - angle,
                    origin,
                    is_generated: generated,
                    loose_bbox: extended.then(|| loose.and_then(|b| visual(b, frame, rotation))),
                    tight_bbox: extended.then(|| tight.and_then(|b| visual(b, frame, rotation))),
                    text_is_visible: None,
                });
            }
            writing_angles(&mut chars);
            if visible_only {
                let visible = visibility::read(
                    api, symbols, owner.page, frame, rotation, bytes, page_index, limits,
                )?;
                let mut addresses = vec![0usize; objects.len()];
                for (address, (id, _)) in &objects {
                    addresses[*id] = *address;
                }
                chars = chars
                    .into_iter()
                    .filter_map(|mut c| {
                        let address = c.text_object_id.map(|i| addresses[i]).unwrap_or(0);
                        visibility::admit(&mut c, visible.get(&address).copied()).then_some(c)
                    })
                    .collect();
            }
            if chars.iter().any(|c| c.text_render_mode == Some(3)) {
                let mut visibility = HashMap::new();
                for c in &mut chars {
                    let visible = *visibility.entry(c.text_object_id).or_insert_with(|| {
                        if c.text_object_id.is_none() {
                            return false;
                        }
                        let (mut r, mut g, mut b, mut a) = (0, 0, 0, 0);
                        let mode = c.text_render_mode.unwrap_or(-1);
                        let mut visible = false;
                        if [0, 2, 4, 6].contains(&mode) {
                            visible = api.FPDFText_GetFillColor(
                                owner.text,
                                c.char_idx as i32,
                                &mut r,
                                &mut g,
                                &mut b,
                                &mut a,
                            ) != 0
                                && a > 0;
                        }
                        if !visible && [1, 2, 5, 6].contains(&mode) {
                            visible = api.FPDFText_GetStrokeColor(
                                owner.text,
                                c.char_idx as i32,
                                &mut r,
                                &mut g,
                                &mut b,
                                &mut a,
                            ) != 0
                                && a > 0;
                        }
                        visible
                    });
                    c.text_is_visible = Some(visible);
                }
            }
            if chars.iter().any(|c| c.raw_code == 0) {
                let replacements =
                    crate::unicode_fallback::recover(api, owner.text, bytes, page_index);
                for c in &mut chars {
                    if c.raw_code == 0 {
                        if let Some(code) = replacements.get(&c.char_idx) {
                            c.char = char::from_u32(*code).unwrap().to_string();
                        }
                    }
                }
            }
            let restored_chars = if visible_only {
                visibility::restore(api, owner.text, &chars, count as usize)
            } else {
                restore(&chars)
            };
            Ok(Primitives {
                page_size: size,
                page_bbox: frame,
                rotation,
                raw_char_count: count as usize,
                chars,
                restored_chars,
            })
        }
    })
}
