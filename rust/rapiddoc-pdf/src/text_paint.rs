//! 按索引读取纯填色文字RGBA；描边/裁剪文字不作为浅色删除证据。
use crate::with_raw;
use pdfium_render::prelude::*;
use rapiddoc_core::{Error, ErrorCode, Limits};
use std::{collections::BTreeMap, path::Path};
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
        unsafe {
            if !self.text.is_null() {
                self.api.FPDFText_ClosePage(self.text)
            }
            if !self.page.is_null() {
                self.api.FPDF_ClosePage(self.page)
            }
            if !self.document.is_null() {
                self.api.FPDF_CloseDocument(self.document)
            }
        }
    }
}
pub fn read(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    indices: &[i64],
    password: Option<&str>,
    limits: &Limits,
) -> Result<BTreeMap<i64, [u32; 4]>, Error> {
    limits.check_input(bytes)?;
    if indices.len() > rapiddoc_core::resources::limit(20000) {
        return Err(error(ErrorCode::ResourceLimit, "文字颜色请求超过预算"));
    }
    if password.is_some_and(|v| v.contains('\0'))
        || indices.iter().any(|i| i32::try_from(*i).is_err())
    {
        return Err(error(ErrorCode::InvalidInput, "文字颜色读取参数无效"));
    }
    with_raw(library, |api| unsafe {
        // SAFETY: 同库句柄在唯一运行库锁内依text/page/document逆序关闭，输入字节仍存活。
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
        let count = api.FPDF_GetPageCount(owner.document);
        if count < 0 {
            return Err(error(ErrorCode::CorruptDocument, "无效页数"));
        }
        if count as usize > limits.max_pages {
            return Err(error(ErrorCode::ResourceLimit, "文档页数超过预算"));
        }
        if page_index >= count as usize {
            return Err(error(ErrorCode::InvalidInput, "页号越界"));
        }
        owner.page = api.FPDF_LoadPage(owner.document, page_index as i32);
        if owner.page.is_null() {
            return Err(error(ErrorCode::CorruptDocument, "无法打开PDF页"));
        }
        owner.text = api.FPDFText_LoadPage(owner.page);
        if owner.text.is_null() {
            return Err(error(ErrorCode::CorruptDocument, "无法读取PDF文字页"));
        }
        let count = api.FPDFText_CountChars(owner.text);
        if count < 0 {
            return Err(error(ErrorCode::CorruptDocument, "无效字符数"));
        }
        if count as usize > limits.max_chars_per_page {
            return Err(error(ErrorCode::ResourceLimit, "页面字符数超过预算"));
        }
        let mut result = BTreeMap::new();
        for &index in indices {
            if index < 0 || index >= count as i64 || result.contains_key(&index) {
                continue;
            }
            let obj = api.FPDFText_GetTextObject(owner.text, index as i32);
            if obj.is_null() || api.FPDFTextObj_GetTextRenderMode(obj) != 0 {
                continue;
            }
            let (mut r, mut g, mut b, mut a) = (0, 0, 0, 0);
            if api.FPDFText_GetFillColor(owner.text, index as i32, &mut r, &mut g, &mut b, &mut a)
                != 0
            {
                result.insert(index, [r, g, b, a]);
            }
        }
        Ok(result)
    })
}
