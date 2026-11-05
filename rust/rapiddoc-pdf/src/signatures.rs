//! Visible signature widgets: inherited field type, appearance proof and visual-frame clipping.
use crate::{link_annotations::visual, with_raw};
// PDFium 7350 fpdf_annot.h / fpdf_formfill.h constants; bindgen does not re-export them.
const FPDF_ANNOT_WIDGET: i32 = 20;
const FPDF_FORMFIELD_SIGNATURE: i32 = 7;
const FPDF_FORMFIELD_UNKNOWN: i32 = 0;
const FPDF_ANNOT_APPEARANCEMODE_NORMAL: i32 = 0;
use pdfium_render::prelude::*;
use rapiddoc_core::{Error, ErrorCode, Limits};
use std::path::Path;
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
    form: FPDF_FORMHANDLE,
}
impl Drop for Handles<'_> {
    fn drop(&mut self) {
        unsafe {
            if !self.page.is_null() {
                self.api.FPDF_ClosePage(self.page)
            }
            if !self.form.is_null() {
                self.api.FPDFDOC_ExitFormFillEnvironment(self.form)
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
    password: Option<&str>,
    limits: &Limits,
) -> Result<Vec<[f64; 4]>, Error> {
    limits.check_input(bytes)?;
    if password.is_some_and(|p| p.contains('\0')) {
        return Err(error(ErrorCode::InvalidInput, "密码包含NUL"));
    }
    with_raw(library, |api| unsafe {
        // All read-only form/page/annotation handles and the callback record stay under the sole runtime lock.
        let mut info: FPDF_FORMFILLINFO = std::mem::zeroed();
        info.version = 1;
        let mut h = Handles {
            api,
            document: api.FPDF_LoadMemDocument64(bytes, password),
            page: std::ptr::null_mut(),
            form: std::ptr::null_mut(),
        };
        if h.document.is_null() {
            return Err(error(
                if matches!(api.FPDF_GetLastError(), 4 | 5) {
                    ErrorCode::Encrypted
                } else {
                    ErrorCode::CorruptDocument
                },
                "无法打开PDF",
            ));
        }
        let count = api.FPDF_GetPageCount(h.document);
        if count < 0 {
            return Err(error(ErrorCode::CorruptDocument, "无效页数"));
        }
        if count as usize > limits.max_pages {
            return Err(error(ErrorCode::ResourceLimit, "页数超过限制"));
        }
        if page_index >= count as usize {
            return Err(error(ErrorCode::InvalidInput, "页号越界"));
        }
        h.form = api.FPDFDOC_InitFormFillEnvironment(h.document, &mut info);
        h.page = api.FPDF_LoadPage(h.document, page_index as i32);
        if h.page.is_null() {
            return Err(error(ErrorCode::CorruptDocument, "无法读取PDF页"));
        }
        let mut r = FS_RECTF {
            left: 0.,
            bottom: 0.,
            right: 0.,
            top: 0.,
        };
        if api.FPDF_GetPageBoundingBox(h.page, &mut r) == 0 {
            return Err(error(ErrorCode::CorruptDocument, "无法读取页框"));
        }
        let frame = [
            f64::from(r.left.min(r.right)),
            f64::from(r.bottom.min(r.top)),
            f64::from(r.left.max(r.right)),
            f64::from(r.bottom.max(r.top)),
        ];
        let rotation = api.FPDFPage_GetRotation(h.page) * 90;
        let count = api.FPDFPage_GetAnnotCount(h.page).max(0) as usize;
        if count > limits.max_links_per_page {
            return Err(error(ErrorCode::ResourceLimit, "注释数量超过限制"));
        }
        let mut out = vec![];
        for i in 0..count {
            let a = api.FPDFPage_GetAnnot(h.page, i as i32);
            if a.is_null() {
                continue;
            }
            let candidate = (|| {
                if api.FPDFAnnot_GetSubtype(a) != FPDF_ANNOT_WIDGET as i32
                    || api.FPDFAnnot_GetFlags(a) & (1 | 2 | 32) != 0
                {
                    return None;
                }
                let field = if h.form.is_null() {
                    None
                } else {
                    Some(api.FPDFAnnot_GetFormFieldType(h.form, a))
                };
                if field != Some(FPDF_FORMFIELD_SIGNATURE as i32) {
                    if field.is_some_and(|f| f > FPDF_FORMFIELD_UNKNOWN as i32) {
                        return None;
                    }
                    let n = api.FPDFAnnot_GetStringValue(a, "FT", std::ptr::null_mut(), 0);
                    if n < 2 || n > rapiddoc_core::resources::limit(20000) || n % 2 != 0 {
                        return None;
                    }
                    let mut text = vec![0u16; n as usize / 2];
                    let got = api.FPDFAnnot_GetStringValue(a, "FT", text.as_mut_ptr(), n);
                    if got < 2
                        || got > n
                        || String::from_utf16(&text[..got as usize / 2 - 1])
                            .ok()
                            .as_deref()
                            != Some("Sig")
                    {
                        return None;
                    }
                }
                if api.FPDFAnnot_GetAP(
                    a,
                    FPDF_ANNOT_APPEARANCEMODE_NORMAL as i32,
                    std::ptr::null_mut(),
                    0,
                ) <= 2
                {
                    return None;
                }
                let mut r = FS_RECTF {
                    left: 0.,
                    bottom: 0.,
                    right: 0.,
                    top: 0.,
                };
                if api.FPDFAnnot_GetRect(a, &mut r) == 0 {
                    return None;
                }
                let (l, b, r, t) = (
                    f64::from(r.left.min(r.right)),
                    f64::from(r.bottom.min(r.top)),
                    f64::from(r.left.max(r.right)),
                    f64::from(r.bottom.max(r.top)),
                );
                visual(&[[l, b], [l, t], [r, b], [r, t]], frame, rotation)
            })();
            api.FPDFPage_CloseAnnot(a);
            if let Some(b) = candidate {
                out.push(b)
            }
        }
        out.sort_by(|a, b| {
            a[1].total_cmp(&b[1])
                .then(a[0].total_cmp(&b[0]))
                .then(a[3].total_cmp(&b[3]))
                .then(a[2].total_cmp(&b[2]))
        });
        Ok(out)
    })
}
