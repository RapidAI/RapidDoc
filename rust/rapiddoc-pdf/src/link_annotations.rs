//! 旧外部URI Link注解提取；所有FFI句柄限制在唯一运行库锁中。
use crate::with_raw;
use pdfium_render::prelude::*;
use rapiddoc_core::{Error, ErrorCode, Limits};
use serde::Serialize;
use std::{collections::HashSet, path::Path};
mod uri;
pub use uri::validate as validate_external_uri;
#[derive(Debug, Serialize)]
pub struct Annotation {
    target: String,
    bboxes: Vec<[f64; 4]>,
    source_index: usize,
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
}
impl Drop for Handles<'_> {
    fn drop(&mut self) {
        unsafe {
            if !self.page.is_null() {
                self.api.FPDF_ClosePage(self.page)
            }
            if !self.document.is_null() {
                self.api.FPDF_CloseDocument(self.document)
            }
        }
    }
}
pub(crate) fn visual(points: &[[f64; 2]], frame: [f64; 4], rotation: i32) -> Option<[f64; 4]> {
    if points.is_empty() || points.iter().flatten().any(|n| !n.is_finite()) {
        return None;
    }
    let [left, bottom, right, top] = frame;
    let mut out = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for &[x, y] in points {
        let p = match rotation {
            90 => [y - bottom, x - left],
            180 => [right - x, y - bottom],
            270 => [top - y, right - x],
            _ => [x - left, top - y],
        };
        out = [
            out[0].min(p[0]),
            out[1].min(p[1]),
            out[2].max(p[0]),
            out[3].max(p[1]),
        ];
    }
    let (w, h) = if matches!(rotation, 90 | 270) {
        (top - bottom, right - left)
    } else {
        (right - left, top - bottom)
    };
    let b = [out[0].max(0.), out[1].max(0.), out[2].min(w), out[3].min(h)];
    (b[2] > b[0] && b[3] > b[1]).then_some(b)
}
unsafe fn regions(
    api: &dyn PdfiumLibraryBindings,
    link: FPDF_LINK,
    frame: [f64; 4],
    rotation: i32,
    total: &mut usize,
) -> Result<Vec<[f64; 4]>, Error> {
    let count = unsafe { api.FPDFLink_CountQuadPoints(link) }.max(0) as usize;
    *total = total.saturating_add(count.max(1));
    if count > rapiddoc_core::resources::limit(5000)
        || *total > rapiddoc_core::resources::limit(20000)
    {
        return Err(error(ErrorCode::ResourceLimit, "链接区域超过预算"));
    }
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    for i in 0..count {
        let mut q = FS_QUADPOINTSF {
            x1: 0.,
            y1: 0.,
            x2: 0.,
            y2: 0.,
            x3: 0.,
            y3: 0.,
            x4: 0.,
            y4: 0.,
        };
        if unsafe { api.FPDFLink_GetQuadPoints(link, i as i32, &mut q) } == 0 {
            continue;
        }
        if let Some(b) = visual(
            &[
                [q.x1 as f64, q.y1 as f64],
                [q.x2 as f64, q.y2 as f64],
                [q.x3 as f64, q.y3 as f64],
                [q.x4 as f64, q.y4 as f64],
            ],
            frame,
            rotation,
        ) {
            let key = b.map(|n| if n == 0. { 0 } else { n.to_bits() });
            if seen.insert(key) {
                result.push(b)
            }
        }
    }
    if result.is_empty() {
        let mut rect = FS_RECTF {
            left: 0.,
            bottom: 0.,
            right: 0.,
            top: 0.,
        };
        if unsafe { api.FPDFLink_GetAnnotRect(link, &mut rect) } != 0 {
            if let Some(b) = visual(
                &[
                    [rect.left as f64, rect.bottom as f64],
                    [rect.left as f64, rect.top as f64],
                    [rect.right as f64, rect.bottom as f64],
                    [rect.right as f64, rect.top as f64],
                ],
                frame,
                rotation,
            ) {
                result.push(b)
            }
        }
    }
    result.sort_by(|a, b| {
        a[1].total_cmp(&b[1])
            .then(a[0].total_cmp(&b[0]))
            .then(a[3].total_cmp(&b[3]))
            .then(a[2].total_cmp(&b[2]))
    });
    Ok(result)
}
pub fn read(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
    limits: &Limits,
) -> Result<Vec<Annotation>, Error> {
    limits.check_input(bytes)?;
    if password.is_some_and(|s| s.contains('\0')) {
        return Err(error(ErrorCode::InvalidInput, "密码包含NUL"));
    }
    with_raw(library, |api| unsafe {
        // SAFETY: 输入字节在整个持锁调用中存活，所有局部句柄由同库逆序关闭。
        let mut owner = Handles {
            api,
            document: api.FPDF_LoadMemDocument64(bytes, password),
            page: std::ptr::null_mut(),
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
            (rect.left as f64).min(rect.right as f64),
            (rect.bottom as f64).min(rect.top as f64),
            (rect.left as f64).max(rect.right as f64),
            (rect.bottom as f64).max(rect.top as f64),
        ];
        let rotation = api.FPDFPage_GetRotation(owner.page) * 90;
        if frame.iter().any(|n| !n.is_finite() || n.abs() > 1e9)
            || frame[2] <= frame[0]
            || frame[3] <= frame[1]
            || !matches!(rotation, 0 | 90 | 180 | 270)
        {
            return Err(error(ErrorCode::CorruptDocument, "页面几何无效"));
        }
        let mut position = 0i32;
        let mut source_index = 0usize;
        let mut output = Vec::new();
        let mut total_regions = 0usize;
        let mut uri_bytes = 0usize;
        loop {
            let mut link = std::ptr::null_mut();
            let previous = position;
            if api.FPDFLink_Enumerate(owner.page, &mut position, &mut link) == 0 {
                break;
            }
            let current = source_index;
            source_index += 1;
            if position <= previous {
                break;
            }
            if source_index > rapiddoc_core::resources::limit(20000) {
                return Err(error(ErrorCode::ResourceLimit, "链接枚举超过预算"));
            }
            if link.is_null() {
                continue;
            }
            let annot = api.FPDFLink_GetAnnot(owner.page, link);
            if annot.is_null() {
                continue;
            }
            let flags = api.FPDFAnnot_GetFlags(annot);
            api.FPDFPage_CloseAnnot(annot);
            if flags & 35 != 0 {
                continue;
            }
            let action = api.FPDFLink_GetAction(link);
            if action.is_null() || api.FPDFAction_GetType(action) != 3 {
                continue;
            }
            let required =
                api.FPDFAction_GetURIPath(owner.document, action, std::ptr::null_mut(), 0) as usize;
            if required <= 1 {
                continue;
            }
            uri_bytes = uri_bytes.saturating_add(required);
            if required > rapiddoc_core::resources::limit(1_000_000)
                || uri_bytes > rapiddoc_core::resources::limit(8_000_000)
            {
                return Err(error(ErrorCode::ResourceLimit, "链接目标超过预算"));
            }
            let mut buffer = vec![0u8; required];
            let actual = api.FPDFAction_GetURIPath(
                owner.document,
                action,
                buffer.as_mut_ptr().cast(),
                required as _,
            ) as usize;
            if actual <= 1 || actual > required {
                continue;
            }
            let Ok(raw) = std::str::from_utf8(&buffer[..actual - 1]) else {
                continue;
            };
            let Some(target) = uri::validate(raw) else {
                continue;
            };
            let bboxes = regions(api, link, frame, rotation, &mut total_regions)?;
            if bboxes.is_empty() {
                continue;
            }
            if output.len() >= rapiddoc_core::resources::limit(5000) {
                return Err(error(ErrorCode::ResourceLimit, "有效链接超过预算"));
            }
            output.push(Annotation {
                target,
                bboxes,
                source_index: current,
            });
        }
        Ok(output)
    })
}
