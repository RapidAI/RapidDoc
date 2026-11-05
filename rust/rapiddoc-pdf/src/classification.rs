//! Owned PDFium evidence for the unchanged Python auto-classifier compatibility layer.
use crate::with_raw;
use pdfium_render::prelude::*;
use rapiddoc_core::{Error, ErrorCode, Limits};
use serde_json::{json, Value};
use std::path::Path;
mod fonts;
mod probe;
pub mod rules;

/// Complete original auto classification. Python binds this entry point only.
pub fn classify(library: &Path, data: &[u8], first: usize, last: Option<usize>) -> Value {
    match classify_inner(library, data, first, last) {
        Ok(value) => value,
        Err(error) => json!({"method":"ocr","error":error.message}),
    }
}
fn classify_inner(
    library: &Path,
    data: &[u8],
    first: usize,
    last: Option<usize>,
) -> Result<Value, Error> {
    let limits = Limits::pdf_host();
    let count = crate::page_count(library, data, None, &limits)?;
    let end = last.unwrap_or(count).min(count);
    if first > end {
        return Err(error("Invalid classification page range"));
    }
    let ids = rules::indices(end - first, 10);
    if ids.is_empty() {
        return Ok(json!({"method":"ocr","samples":[]}));
    }
    let mut facts = Vec::new();
    let mut samples = Vec::new();
    for &local in &ids {
        let mut f = page(library, data, first + local, None, &limits)?;
        let width = f["size"][0].as_f64().unwrap();
        let height = f["size"][1].as_f64().unwrap();
        if width > 0. && height > 0. && (width / height).max(height / width) > 10. {
            return Ok(json!({"method":"ocr","reason":"extreme_aspect_ratio"}));
        }
        let s = rules::sample(&f, local);
        if s["cleaned_text"].as_str().unwrap().chars().count() < 50 && probe::sparse_bad(&f, &s) {
            return Ok(json!({"method":"ocr","reason":"sparse_page"}));
        }
        f["chars"] = Value::Null;
        facts.push(f);
        samples.push(s);
    }
    let errors = rules::map_errors(&samples);
    if errors["unicode_map_error_ratio"].as_f64().unwrap() >= 0.04
        && rules::widespread(&samples, |s| {
            s["char_count"].as_f64().unwrap() > 0.
                && s["unicode_map_error_count"].as_f64().unwrap()
                    / s["char_count"].as_f64().unwrap()
                    >= 0.04
        })
    {
        return Ok(json!({"method":"ocr","samples":samples,"reason":"unicode_mapping"}));
    }
    let selected = ids
        .iter()
        .map(|&local| (local, first + local))
        .collect::<Vec<_>>();
    let font_signals = fonts::signals(data, &selected)
        .map_err(|e| error(&format!("Font resource analysis failed: {e}")))?;
    if rules::decoded_text_bad(&samples, &font_signals) {
        return Ok(json!({"method":"ocr","samples":samples,"reason":"text_quality"}));
    }
    if facts.iter().any(probe::high_coverage) {
        for &local in &ids {
            let f = page(library, data, first + local, None, &limits)?;
            let signal = probe::image_text(library, data, first + local, &f, &limits)?;
            if signal["image_backed_ocr"] == true {
                return Ok(
                    json!({"method":"ocr","samples":samples,"reason":"unpainted_image_text"}),
                );
            }
        }
    }
    Ok(json!({"method":"txt","samples":samples}))
}

#[cfg(test)]
mod native_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    fn root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }
    fn fixture(name: &str) -> Value {
        serde_json::from_slice(
            &std::fs::read(root().join(format!("build/migration/{name}.json"))).unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn original_classifier_rules() {
        for c in fixture("native-classification-rules")["cases"]
            .as_array()
            .unwrap()
        {
            let q = &c["request"];
            let samples = q["samples"].as_array().map(Vec::as_slice).unwrap_or(&[]);
            let actual = match c["operation"].as_str().unwrap() {
                "indices" => json!(rules::indices(
                    q["count"].as_u64().unwrap() as usize,
                    q["max"].as_u64().unwrap() as usize
                )),
                "quality" => rules::quality(samples),
                "map_errors" => rules::map_errors(samples),
                "u72" => rules::u72(samples),
                "cross" => rules::cross_script(samples),
                "punctuation" => rules::punctuation(samples),
                "cid" => rules::cid_usage(samples, &q["fonts"]),
                "latin" => rules::latin_usage(samples, &q["fonts"], 30, 0.01, 0.8),
                _ => panic!("Unknown rule"),
            };
            assert_eq!(actual, c["expected"], "{c}");
        }
    }
    #[test]
    fn original_classifier_documents() {
        let root = root();
        let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
        for c in fixture("native-classification-documents")["cases"]
            .as_array()
            .unwrap()
        {
            let data = std::fs::read(root.join(c["source"].as_str().unwrap())).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&data)),
                c["sha256"].as_str().unwrap()
            );
            let got = classify(&library, &data, 0, None);
            assert_eq!(got["method"], c["expected"], "{}: {got}", c["label"]);
            assert!(got.get("error").is_none(), "{}: {got}", c["label"]);
        }
    }
    #[test]
    fn original_classifier_user_samples() {
        let root = root();
        let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
        for c in fixture("api-classification")["cases"].as_array().unwrap() {
            let data = std::fs::read(root.join(c["source"].as_str().unwrap())).unwrap();
            let got = classify(&library, &data, 0, None);
            assert_eq!(got["method"], c["expected"], "{}: {got}", c["source"]);
            assert!(got.get("error").is_none(), "{}: {got}", c["source"]);
            for s in c["samples"].as_array().unwrap() {
                let i = s["page_index"].as_u64().unwrap() as usize;
                let facts = page(&library, &data, i, None, &Limits::pdf_host()).unwrap();
                assert_eq!(rules::sample(&facts, i), *s, "{} sample {i}", c["source"]);
            }
        }
    }
    #[test]
    fn native_sampling_does_not_cut_off_at_one_hundred_thousand() {
        let chars = (0..100_003)
            .map(|_| json!([65, false, false, "ABCDEF+Helvetica", [0., 0., 1., 1.]]))
            .collect::<Vec<_>>();
        let facts = json!({"text":"A".repeat(chars.len()),"chars":chars});
        let s = rules::sample(&facts, 0);
        assert_eq!(s["char_count"], 100_003);
        assert_eq!(s["cleaned_text"].as_str().unwrap().len(), 100_003);
        assert_eq!(s["font_non_generated_char_counts"]["Helvetica"], 100_003);
    }
}
// fpdfview.h constants not exported by the safe prelude.
const FPDFBITMAP_BGR: i32 = 2;
const FPDF_ANNOT: i32 = 1;
fn error(message: &str) -> Error {
    Error {
        code: ErrorCode::ResourceLimit,
        message: message.into(),
    }
}
struct Handles<'a> {
    api: &'a dyn PdfiumLibraryBindings,
    doc: FPDF_DOCUMENT,
    page: FPDF_PAGE,
    text: FPDF_TEXTPAGE,
    bitmap: FPDF_BITMAP,
}
impl Drop for Handles<'_> {
    fn drop(&mut self) {
        unsafe {
            if !self.bitmap.is_null() {
                self.api.FPDFBitmap_Destroy(self.bitmap)
            }
            if !self.text.is_null() {
                self.api.FPDFText_ClosePage(self.text)
            }
            if !self.page.is_null() {
                self.api.FPDF_ClosePage(self.page)
            }
            if !self.doc.is_null() {
                self.api.FPDF_CloseDocument(self.doc)
            }
        }
    }
}
unsafe fn objects(
    api: &dyn PdfiumLibraryBindings,
    page: FPDF_PAGE,
    parent: Option<FPDF_PAGEOBJECT>,
    depth: usize,
    max_depth: usize,
    out: &mut Vec<(FPDF_PAGEOBJECT, i32, [f64; 4])>,
) -> Result<(), Error> {
    let count = if let Some(o) = parent {
        api.FPDFFormObj_CountObjects(o)
    } else {
        api.FPDFPage_CountObjects(page)
    };
    if count < 0
        || out.len().saturating_add(count as usize) > rapiddoc_core::resources::limit(100_000)
    {
        return Err(error("分类页面对象超过预算"));
    }
    for i in 0..count {
        let o = if let Some(p) = parent {
            api.FPDFFormObj_GetObject(p, i as _)
        } else {
            api.FPDFPage_GetObject(page, i)
        };
        if o.is_null() {
            continue;
        }
        let kind = api.FPDFPageObj_GetType(o);
        let (mut l, mut b, mut r, mut t) = (0., 0., 0., 0.);
        if api.FPDFPageObj_GetBounds(o, &mut l, &mut b, &mut r, &mut t) != 0 {
            out.push((o, kind, [l as f64, b as f64, r as f64, t as f64]));
        }
        if kind == 5 && depth < max_depth {
            objects(api, page, Some(o), depth + 1, max_depth, out)?
        }
    }
    Ok(())
}
pub fn page(
    library: &Path,
    data: &[u8],
    index: usize,
    password: Option<&str>,
    limits: &Limits,
) -> Result<Value, Error> {
    limits.check_input(data)?;
    with_raw(library, |api| unsafe {
        let mut h = Handles {
            api,
            doc: api.FPDF_LoadMemDocument64(data, password),
            page: std::ptr::null_mut(),
            text: std::ptr::null_mut(),
            bitmap: std::ptr::null_mut(),
        };
        if h.doc.is_null() {
            return Err(error("无法打开分类PDF"));
        }
        let count = api.FPDF_GetPageCount(h.doc);
        if index >= count.max(0) as usize {
            return Err(error("分类页号越界"));
        }
        h.page = api.FPDF_LoadPage(h.doc, index as i32);
        if h.page.is_null() {
            return Err(error("无法读取分类页"));
        }
        let mut bbox = FS_RECTF {
            left: 0.,
            bottom: 0.,
            right: 0.,
            top: 0.,
        };
        if api.FPDF_GetPageBoundingBox(h.page, &mut bbox) == 0 {
            return Err(error("无法读取分类页框"));
        }
        h.text = api.FPDFText_LoadPage(h.page);
        if h.text.is_null() {
            return Err(error("无法读取分类字符"));
        }
        let n = api.FPDFText_CountChars(h.text);
        if n < 0 {
            return Err(error("分类字符超过预算"));
        }
        let len = api.FPDFText_GetBoundedText(
            h.text,
            bbox.left as _,
            bbox.top as _,
            bbox.right as _,
            bbox.bottom as _,
            std::ptr::null_mut(),
            0,
        );
        if len < 0 || (len as usize).saturating_mul(2) > limits.max_input_bytes {
            return Err(error("分类文本超过预算"));
        }
        let mut text = vec![0u16; len as usize];
        if len > 0 {
            api.FPDFText_GetBoundedText(
                h.text,
                bbox.left as _,
                bbox.top as _,
                bbox.right as _,
                bbox.bottom as _,
                text.as_mut_ptr(),
                len,
            );
            if text.last() == Some(&0) {
                text.pop();
            }
        }
        let text = std::char::decode_utf16(text)
            .filter_map(Result::ok)
            .collect::<String>();
        let mut chars = Vec::new();
        chars
            .try_reserve(n as usize)
            .map_err(|_| error("Classification allocation failed"))?;
        let mut name_bytes = 0usize;
        for i in 0..n {
            let mut flags = 0;
            let len =
                api.FPDFText_GetFontInfo(h.text, i, std::ptr::null_mut(), 0, &mut flags) as usize;
            if len > rapiddoc_core::resources::limit(65536) {
                return Err(error("分类字体名超过预算"));
            }
            name_bytes += len;
            if name_bytes > rapiddoc_core::resources::limit(32_000_000) {
                return Err(error("分类字体总量超过预算"));
            }
            let mut font = vec![0u8; len];
            if len > 0 {
                api.FPDFText_GetFontInfo(h.text, i, font.as_mut_ptr().cast(), len as _, &mut flags);
                if font.last() == Some(&0) {
                    font.pop();
                }
            }
            let font = String::from_utf8_lossy(&font).replace('\u{fffd}', "");
            let (mut l, mut r, mut b, mut t) = (0., 0., 0., 0.);
            api.FPDFText_GetCharBox(h.text, i, &mut l, &mut r, &mut b, &mut t);
            chars.push(json!([
                api.FPDFText_GetUnicode(h.text, i),
                api.FPDFText_IsGenerated(h.text, i) == 1,
                api.FPDFText_HasUnicodeMapError(h.text, i) != 0,
                font,
                [l, b, r, t]
            ]));
        }
        let mut os = vec![];
        objects(api, h.page, None, 0, 3, &mut os)?;
        Ok(
            json!({"text":text,"chars":chars,"size":[api.FPDF_GetPageWidthF(h.page),api.FPDF_GetPageHeightF(h.page)],"bbox":[bbox.left,bbox.bottom,bbox.right,bbox.top],"rotation":api.FPDFPage_GetRotation(h.page)*90,"objects":os.iter().map(|(_,k,b)|json!([k,b])).collect::<Vec<_>>()}),
        )
    })
}
struct PaintRestore<'a> {
    api: &'a dyn PdfiumLibraryBindings,
    items: Vec<(FPDF_PAGEOBJECT, i32)>,
}
impl Drop for PaintRestore<'_> {
    fn drop(&mut self) {
        for (o, m) in self.items.iter().rev() {
            unsafe {
                self.api.FPDFTextObj_SetTextRenderMode(*o, *m);
            }
        }
    }
}
pub fn difference(
    library: &Path,
    data: &[u8],
    index: usize,
    scale: f64,
    password: Option<&str>,
    limits: &Limits,
) -> Result<Vec<u8>, Error> {
    use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder};
    limits.check_input(data)?;
    if !scale.is_finite() || scale <= 0. || scale > 1.5 {
        return Err(error("分类像素探测比例无效"));
    }
    let (w, hh, pixels) = with_raw(library, |api| unsafe {
        let mut h = Handles {
            api,
            doc: api.FPDF_LoadMemDocument64(data, password),
            page: std::ptr::null_mut(),
            text: std::ptr::null_mut(),
            bitmap: std::ptr::null_mut(),
        };
        if h.doc.is_null() {
            return Err(error("无法打开像素探测PDF"));
        }
        let count = api.FPDF_GetPageCount(h.doc);
        if index >= count.max(0) as usize {
            return Err(error("像素探测页号越界"));
        }
        h.page = api.FPDF_LoadPage(h.doc, index as _);
        if h.page.is_null() {
            return Err(error("无法读取像素探测页"));
        }
        let w = (api.FPDF_GetPageWidthF(h.page) as f64 * scale).ceil() as i32;
        let hh = (api.FPDF_GetPageHeightF(h.page) as f64 * scale).ceil() as i32;
        if w <= 0
            || hh <= 0
            || (w as usize).saturating_mul(hh as usize) > rapiddoc_core::resources::limit(4_000_000)
        {
            return Err(error("分类像素超过预算"));
        }
        h.bitmap = api.FPDFBitmap_CreateEx(w, hh, FPDFBITMAP_BGR as _, std::ptr::null_mut(), 0);
        if h.bitmap.is_null() {
            return Err(error("无法创建分类位图"));
        }
        api.FPDFBitmap_FillRect(h.bitmap, 0, 0, w, hh, 0xffffffff);
        api.FPDF_RenderPageBitmap(h.bitmap, h.page, 0, 0, w, hh, 0, FPDF_ANNOT as _);
        let before = api.FPDFBitmap_GetBuffer_as_vec(h.bitmap);
        let mut os = vec![];
        objects(api, h.page, None, 0, 10, &mut os)?;
        let mut restore = PaintRestore { api, items: vec![] };
        for (o, k, _) in os {
            if k == 1 {
                let m = api.FPDFTextObj_GetTextRenderMode(o);
                if matches!(m, 0 | 1 | 2 | 4 | 5 | 6) {
                    if api.FPDFTextObj_SetTextRenderMode(o, if m >= 4 { 7 } else { 3 }) == 0 {
                        return Err(error("无法抑制分类文本绘制"));
                    }
                    restore.items.push((o, m));
                }
            }
        }
        api.FPDFBitmap_FillRect(h.bitmap, 0, 0, w, hh, 0xffffffff);
        api.FPDF_RenderPageBitmap(h.bitmap, h.page, 0, 0, w, hh, 0, FPDF_ANNOT as _);
        let after = api.FPDFBitmap_GetBuffer_as_vec(h.bitmap);
        let stride = api.FPDFBitmap_GetStride(h.bitmap) as usize;
        let mut out = vec![0u8; w as usize * hh as usize];
        for y in 0..hh as usize {
            for x in 0..w as usize {
                let p = y * stride + x * 3;
                out[y * w as usize + x] = (0..3)
                    .map(|c| before[p + c].abs_diff(after[p + c]))
                    .max()
                    .unwrap();
            }
        }
        drop(restore);
        Ok((w as u32, hh as u32, out))
    })?;
    let mut png = vec![];
    PngEncoder::new(&mut png)
        .write_image(&pixels, w, hh, ExtendedColorType::L8)
        .map_err(|e| error(&e.to_string()))?;
    Ok(png)
}
