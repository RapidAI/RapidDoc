//! 旧 pdftext/RapidDoc 字符读取契约；独立于预览事实层的 f32 几何。
use crate::with_raw;
use pdfium_render::prelude::*;
use rapiddoc_core::{Error, ErrorCode, Limits};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::Path};
mod grouping;
pub use grouping::NativePage;

pub fn read_page(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
    quote_loosebox: bool,
    limits: &Limits,
) -> Result<NativePage, Error> {
    Ok(grouping::group(read(
        library,
        bytes,
        page_index,
        password,
        quote_loosebox,
        limits,
    )?))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Font {
    name: String,
    flags: i32,
    size: f64,
    weight: i32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Char {
    bbox: [f64; 4],
    char: String,
    rotation: f64,
    font: Font,
    char_idx: usize,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct PageChars {
    size: [f64; 2],
    bbox: [f64; 4],
    width: u64,
    height: u64,
    rotation: i32,
    char_count: usize,
    chars: Vec<Char>,
}
fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}

// 原始句柄按 text/page/document 顺序销毁，bytes 在此 owner 之后才释放。
struct Handles<'a> {
    api: &'a dyn PdfiumLibraryBindings,
    document: FPDF_DOCUMENT,
    page: FPDF_PAGE,
    text: FPDF_TEXTPAGE,
}
impl Drop for Handles<'_> {
    fn drop(&mut self) {
        // SAFETY: 句柄由同一 bindings 创建，持全局 PDFium 锁，非空句柄只关闭一次。
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

/// 返回旧 get_page_chars 的去重字符，尚未生成 span/line/block。
pub fn read(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
    quote_loosebox: bool,
    limits: &Limits,
) -> Result<PageChars, Error> {
    limits.check_input(bytes)?;
    if password.is_some_and(|s| s.contains('\0')) {
        return Err(error(ErrorCode::InvalidInput, "密码包含 NUL"));
    }
    with_raw(library, |api| {
        // SAFETY: 输入字节持续存活，所有索引和分配在使用前检查；全程持全局锁。
        unsafe {
            let mut owner = Handles {
                api,
                document: api.FPDF_LoadMemDocument64(bytes, password),
                page: std::ptr::null_mut(),
                text: std::ptr::null_mut(),
            };
            if owner.document.is_null() {
                let code = if matches!(api.FPDF_GetLastError(), 4 | 5) {
                    ErrorCode::Encrypted
                } else {
                    ErrorCode::CorruptDocument
                };
                return Err(error(code, "无法打开 PDF"));
            }
            let count = api.FPDF_GetPageCount(owner.document);
            if count < 0 {
                return Err(error(ErrorCode::CorruptDocument, "无效页数"));
            }
            if count as usize > limits.max_pages {
                return Err(error(ErrorCode::ResourceLimit, "文档页数超过限制"));
            }
            if page_index >= count as usize {
                return Err(error(ErrorCode::InvalidInput, "页号越界"));
            }
            owner.page = api.FPDF_LoadPage(owner.document, page_index as i32);
            if owner.page.is_null() {
                return Err(error(ErrorCode::CorruptDocument, "无法读取 PDF 页"));
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
            let bbox = [
                rect.left as f64,
                rect.bottom as f64,
                rect.right as f64,
                rect.top as f64,
            ];
            let width = (bbox[2] - bbox[0]).abs().ceil();
            let height = (bbox[1] - bbox[3]).abs().ceil();
            let size = [
                api.FPDF_GetPageWidthF(owner.page) as f64,
                api.FPDF_GetPageHeightF(owner.page) as f64,
            ];
            if bbox.iter().chain(size.iter()).any(|n| !n.is_finite()) {
                return Err(error(ErrorCode::CorruptDocument, "非有限页框"));
            }
            if width > rapiddoc_core::resources::limit(1_000_000_000.)
                || height > rapiddoc_core::resources::limit(1_000_000_000.)
            {
                return Err(error(ErrorCode::ResourceLimit, "页框尺寸超过限制"));
            }
            let rotation = api.FPDFPage_GetRotation(owner.page) * 90;
            if !matches!(rotation, 0 | 90 | 180 | 270) {
                return Err(error(ErrorCode::CorruptDocument, "无效页面旋转"));
            }
            owner.text = api.FPDFText_LoadPage(owner.page);
            if owner.text.is_null() {
                return Err(error(ErrorCode::CorruptDocument, "无法读取文字页"));
            }
            let n = api.FPDFText_CountChars(owner.text);
            if n < 0 {
                return Err(error(ErrorCode::CorruptDocument, "无效字符数"));
            }
            if n as usize
                > limits
                    .max_chars_per_page
                    .min(limits.max_total_chars)
                    .min(100_000)
            {
                return Err(error(ErrorCode::ResourceLimit, "字符数量超过限制"));
            }
            let mut chars = Vec::with_capacity(n as usize);
            let mut buffer = vec![0u8; 256];
            let mut name_bytes = 0usize;
            for i in 0..n {
                let code = api.FPDFText_GetUnicode(owner.text, i);
                if code > 0x10ffff {
                    return Err(error(ErrorCode::CorruptDocument, "无效 Unicode 码位"));
                }
                let c = char::from_u32(code).unwrap_or('\u{fffd}');
                let angle = api.FPDFText_GetCharAngle(owner.text, i) as f64;
                let (mut l, mut b, mut r, mut t) = (0., 0., 0., 0.);
                let ok = if angle == 0. && (code != 39 || quote_loosebox) {
                    let ok = api.FPDFText_GetLooseCharBox(owner.text, i, &mut rect);
                    (l, b, r, t) = (
                        rect.left as f64,
                        rect.bottom as f64,
                        rect.right as f64,
                        rect.top as f64,
                    );
                    ok
                } else {
                    api.FPDFText_GetCharBox(owner.text, i, &mut l, &mut r, &mut b, &mut t)
                };
                if ok == 0 {
                    return Err(error(ErrorCode::CorruptDocument, "无法读取字符框"));
                }
                let mut flags = 0;
                buffer.fill(0);
                let length = api.FPDFText_GetFontInfo(
                    owner.text,
                    i,
                    buffer.as_mut_ptr().cast(),
                    buffer.len() as _,
                    &mut flags,
                ) as usize;
                if length > rapiddoc_core::resources::limit(1_048_576) {
                    return Err(error(ErrorCode::ResourceLimit, "字体名称超过限制"));
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
                name_bytes += name.len();
                if name_bytes > rapiddoc_core::resources::limit(32_000_000) {
                    return Err(error(ErrorCode::ResourceLimit, "累计字体名称超过限制"));
                }
                let font = Font {
                    name,
                    flags,
                    size: api.FPDFText_GetFontSize(owner.text, i),
                    weight: crate::font_weight::normalize(
                        api.FPDFText_GetFontWeight(owner.text, i),
                    ),
                };
                if [l, b, r, t, angle, font.size]
                    .iter()
                    .any(|n| !n.is_finite())
                {
                    return Err(error(ErrorCode::CorruptDocument, "非有限字符事实"));
                }
                let (xa, xb, ya, yb) = (
                    l - bbox[0],
                    r - bbox[0],
                    height - (b - bbox[1]),
                    height - (t - bbox[1]),
                );
                let (x0, y0, x1, y1) = (xa.min(xb), ya.min(yb), xa.max(xb), ya.max(yb));
                let bbox = match rotation {
                    90 => [height - y1, x0, height - y0, x1],
                    180 => [width - x1, height - y1, width - x0, height - y0],
                    270 => [y0, width - x1, y1, width - x0],
                    _ => [x0, y0, x1, y1],
                };
                chars.push(Char {
                    bbox,
                    char: c.to_string(),
                    rotation: angle,
                    font,
                    char_idx: i as usize,
                });
            }
            Ok(PageChars {
                size,
                bbox,
                width: width as u64,
                height: height as u64,
                rotation,
                char_count: n as usize,
                chars: deduplicate(chars),
            })
        }
    })
}

fn bits(n: f64) -> u64 {
    if n == 0. {
        0
    } else {
        n.to_bits()
    }
}
fn merge(a: &mut [f64; 4], b: [f64; 4]) {
    a[0] = a[0].min(b[0]);
    a[1] = a[1].min(b[1]);
    a[2] = a[2].max(b[2]);
    a[3] = a[3].max(b[3]);
}
fn adjacent(a: &Char, b: &Char) -> bool {
    let rounded = |n: f64| format!("{n:.3}").parse::<f64>().expect("finite angle");
    if a.char != b.char || a.font != b.font || rounded(a.rotation) != rounded(b.rotation) {
        return false;
    }
    let delta = [
        b.bbox[0] - a.bbox[0],
        b.bbox[1] - a.bbox[1],
        b.bbox[2] - a.bbox[2],
        b.bbox[3] - a.bbox[3],
    ];
    if (delta[0] - delta[2]).abs() > 0.1
        || (delta[1] - delta[3]).abs() > 0.1
        || !(delta[0].abs() > 1.
            && delta[0].abs() <= 2.5
            && delta[1].abs() > 1.
            && delta[1].abs() <= 2.5)
    {
        return false;
    }
    let area = |r: [f64; 4]| (r[2] - r[0]).max(0.) * (r[3] - r[1]).max(0.);
    let smaller = area(a.bbox).min(area(b.bbox));
    smaller > 0.
        && (a.bbox[2].min(b.bbox[2]) - a.bbox[0].max(b.bbox[0])).max(0.)
            * (a.bbox[3].min(b.bbox[3]) - a.bbox[1].max(b.bbox[1])).max(0.)
            / smaller
            >= 0.45
}
fn deduplicate(chars: Vec<Char>) -> Vec<Char> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let mut end = start + 1;
        let mut bbox = chars[start].bbox;
        while end < chars.len()
            && !matches!(chars[end - 1].char.as_str(), "\u{2}" | "\n" | " ")
            && chars[end].font == chars[end - 1].font
            && chars[end].rotation == chars[end - 1].rotation
        {
            merge(&mut bbox, chars[end].bbox);
            end += 1;
        }
        let first = &chars[start];
        let key = (
            bbox.map(|n| bits(n.round_ties_even())),
            chars[start..end]
                .iter()
                .map(|c| c.char.as_str())
                .collect::<String>(),
            bits(first.rotation),
            first.font.name.clone(),
            first.font.flags,
            bits(first.font.size),
            first.font.weight,
        );
        if seen.insert(key) {
            for c in &chars[start..end] {
                let whitespace = c
                    .char
                    .chars()
                    .all(|c| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}'));
                if whitespace || !out.last().is_some_and(|last| adjacent(last, c)) {
                    out.push(c.clone());
                }
            }
        }
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frozen_old_dedup_and_group_boundaries() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/legacy-native-groups.json");
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 268);
        for (i, c) in cases.iter().enumerate() {
            let mut page: PageChars = serde_json::from_value(c["request"].clone()).unwrap();
            page.chars = deduplicate(page.chars);
            assert_eq!(
                serde_json::to_value(grouping::group(page)).unwrap(),
                c["expected"],
                "case {i}"
            );
        }
    }
}
