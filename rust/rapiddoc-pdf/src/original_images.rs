//! 旧 get_ori_image 默认路径：保持对象遍历、局部字框和原始分辨率渲染规则。
//! 原始分辨率缩放改写自 pypdfium2；来源见 licenses/MIGRATION-NOTICES.txt 的 PYPDFIUM2-NOTICE.txt 节，完整许可见 licenses/PYPDFIUM2.txt。
use crate::with_raw;
use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder};
use pdfium_render::prelude::*;
use rapiddoc_core::{document::Diagnostic, Error, ErrorCode, Limits};
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
pub struct OriginalImage {
    pub uuid: String,
    pub bbox: [f64; 4],
    pub width: u32,
    pub height: u32,
    pub mode: String,
    #[serde(skip)]
    pub png: Vec<u8>,
}
pub struct Output {
    pub images: Vec<OriginalImage>,
    pub diagnostics: Vec<Diagnostic>,
}
pub struct Config {
    pub max_depth: usize,
    pub max_pixels: usize,
    pub render: bool,
    pub scale_to_original: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            max_depth: 15,
            max_pixels: 32_000_000,
            render: true,
            scale_to_original: true,
        }
    }
}
fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}
struct PageOwner<'a> {
    api: &'a dyn PdfiumLibraryBindings,
    document: FPDF_DOCUMENT,
    page: FPDF_PAGE,
}
impl Drop for PageOwner<'_> {
    fn drop(&mut self) {
        // SAFETY: 句柄属于本次调用，析构期间全局锁仍持有。
        unsafe {
            if !self.page.is_null() {
                self.api.FPDF_ClosePage(self.page);
            }
            if !self.document.is_null() {
                self.api.FPDF_CloseDocument(self.document);
            }
        }
    }
}
struct Bitmap<'a> {
    api: &'a dyn PdfiumLibraryBindings,
    raw: FPDF_BITMAP,
}
impl Drop for Bitmap<'_> {
    fn drop(&mut self) {
        // SAFETY: 非空 bitmap 由 Get[Rendered]Bitmap 分配且只销毁一次。
        unsafe {
            self.api.FPDFBitmap_Destroy(self.raw);
        }
    }
}

/// 返回独立 PNG 和诊断。对象级解码失败沿用旧跳过行为，但留下诊断；超预算则整次失败。
pub fn extract(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
    config: &Config,
    limits: &Limits,
) -> Result<Output, Error> {
    limits.check_input(bytes)?;
    if config.max_depth > 32
        || rapiddoc_core::resources::limit(config.max_pixels) == 0
        || rapiddoc_core::resources::limit(config.max_pixels)
            > rapiddoc_core::resources::limit(32_000_000)
        || password.is_some_and(|s| s.contains('\0'))
    {
        return Err(error(ErrorCode::InvalidInput, "提图参数无效"));
    }
    with_raw(library, |api| {
        // SAFETY: bytes 生命周期覆盖原始 document/page，全程持唯一 PDFium 锁。
        unsafe {
            let mut owner = PageOwner {
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
                    "无法打开 PDF",
                ));
            }
            let count = api.FPDF_GetPageCount(owner.document);
            if count < 0 {
                return Err(error(ErrorCode::CorruptDocument, "无效页数"));
            }
            if count as usize > limits.max_pages {
                return Err(error(ErrorCode::ResourceLimit, "页数超过限制"));
            }
            if page_index >= count as usize {
                return Err(error(ErrorCode::InvalidInput, "页号越界"));
            }
            owner.page = api.FPDF_LoadPage(owner.document, page_index as i32);
            if owner.page.is_null() {
                return Err(error(ErrorCode::CorruptDocument, "无法打开页面"));
            }
            let height = api.FPDF_GetPageHeightF(owner.page) as f64;
            if !height.is_finite() {
                return Err(error(ErrorCode::CorruptDocument, "非有限页高"));
            }
            let mut state = State {
                owner: &owner,
                config,
                page_index,
                height,
                objects: 0,
                candidates: 0,
                pixels: 0,
                bytes: 0,
                output: Output {
                    images: Vec::new(),
                    diagnostics: Vec::new(),
                },
            };
            state.walk(None, 0, limits)?;
            Ok(state.output)
        }
    })
}
struct State<'a, 'b> {
    owner: &'a PageOwner<'b>,
    config: &'a Config,
    page_index: usize,
    height: f64,
    objects: usize,
    candidates: usize,
    pixels: usize,
    bytes: usize,
    output: Output,
}
impl State<'_, '_> {
    fn walk(
        &mut self,
        form: Option<FPDF_PAGEOBJECT>,
        level: usize,
        limits: &Limits,
    ) -> Result<(), Error> {
        let api = self.owner.api;
        // SAFETY: 所有对象借用本页，递归深度/总对象数有界，页持续存活。
        unsafe {
            let count = if let Some(form) = form {
                api.FPDFFormObj_CountObjects(form)
            } else {
                api.FPDFPage_CountObjects(self.owner.page)
            };
            if count < 0 {
                return Err(error(ErrorCode::CorruptDocument, "无法枚举页面对象"));
            }
            self.objects = self
                .objects
                .checked_add(count as usize)
                .filter(|n| *n <= limits.max_objects_per_page)
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "对象数量超过限制"))?;
            for i in 0..count {
                let object = if let Some(form) = form {
                    api.FPDFFormObj_GetObject(form, i as _)
                } else {
                    api.FPDFPage_GetObject(self.owner.page, i)
                };
                if object.is_null() {
                    return Err(error(ErrorCode::CorruptDocument, "无效页面对象"));
                }
                let kind = api.FPDFPageObj_GetType(object);
                if kind == FPDF_PAGEOBJ_IMAGE as i32 {
                    match self.image(object) {
                        Ok(Some(image)) => self.output.images.push(image),
                        Ok(None) => {}
                        Err(e) if e.code == ErrorCode::CorruptDocument => {
                            self.output.diagnostics.push(Diagnostic {
                                code: "original_image_skipped".into(),
                                page_index: self.page_index,
                                message: e.message,
                            })
                        }
                        Err(e) => return Err(e),
                    }
                }
                if kind == FPDF_PAGEOBJ_FORM as i32 && level + 1 < self.config.max_depth {
                    self.walk(Some(object), level + 1, limits)?;
                }
            }
        }
        Ok(())
    }
    fn image(&mut self, object: FPDF_PAGEOBJECT) -> Result<Option<OriginalImage>, Error> {
        let api = self.owner.api;
        // SAFETY: object 为仍存活页面的 IMAGE；所有像素/stride/指针访问先验证。
        unsafe {
            let (mut l, mut b, mut r, mut t) = (0f32, 0f32, 0f32, 0f32);
            if api.FPDFPageObj_GetBounds(object, &mut l, &mut b, &mut r, &mut t) == 0 {
                return Err(error(ErrorCode::CorruptDocument, "无法读取原图框"));
            }
            let (l, b, r, t) = (l as f64, b as f64, r as f64, t as f64);
            if [l, b, r, t].iter().any(|n| !n.is_finite()) {
                return Err(error(ErrorCode::CorruptDocument, "非有限原图框"));
            }
            let (w, h) = ((r - l).abs(), (t - b).abs());
            if w < 5. || h < 5. {
                return Ok(None);
            }
            self.candidates += 1;
            if self.candidates > rapiddoc_core::resources::limit(256) {
                return Err(error(ErrorCode::ResourceLimit, "图片数量超过限制"));
            }
            let (mut pw, mut ph) = (0, 0);
            if api.FPDFImageObj_GetImagePixelSize(object, &mut pw, &mut ph) == 0
                || pw == 0
                || ph == 0
            {
                return Err(error(ErrorCode::CorruptDocument, "无法读取原图尺寸"));
            }
            self.check_pixels(pw as f64, ph as f64)?;
            let scale = if self.config.render && self.config.scale_to_original {
                if (pw < ph) != (w < h) {
                    std::mem::swap(&mut pw, &mut ph);
                }
                (pw as f64 / w).max(ph as f64 / h)
            } else {
                1.
            };
            // 在调用 PDFium 分配位图前预检预计分辨率；真实尺寸还要再次验证。
            let expected = if self.config.render {
                [(w * scale).ceil(), (h * scale).ceil()]
            } else {
                [pw as f64, ph as f64]
            };
            self.check_pixels(expected[0], expected[1])?;
            let mut matrix = FS_MATRIX {
                a: 0.,
                b: 0.,
                c: 0.,
                d: 0.,
                e: 0.,
                f: 0.,
            };
            let scaled = self.config.render && self.config.scale_to_original;
            if scaled {
                if api.FPDFPageObj_GetMatrix(object, &mut matrix) == 0 {
                    return Err(error(ErrorCode::CorruptDocument, "无法读取图片矩阵"));
                }
                let next = FS_MATRIX {
                    a: (matrix.a as f64 * scale) as f32,
                    b: (matrix.b as f64 * scale) as f32,
                    c: (matrix.c as f64 * scale) as f32,
                    d: (matrix.d as f64 * scale) as f32,
                    e: (matrix.e as f64 * scale) as f32,
                    f: (matrix.f as f64 * scale) as f32,
                };
                if [next.a, next.b, next.c, next.d, next.e, next.f]
                    .iter()
                    .any(|n| !n.is_finite())
                {
                    return Err(error(ErrorCode::ResourceLimit, "图片矩阵溢出"));
                }
                if api.FPDFPageObj_SetMatrix(object, &next) == 0 {
                    return Err(error(ErrorCode::CorruptDocument, "无法设置图片矩阵"));
                }
            }
            let raw = if self.config.render {
                api.FPDFImageObj_GetRenderedBitmap(self.owner.document, self.owner.page, object)
            } else {
                api.FPDFImageObj_GetBitmap(object)
            };
            // 尽早还原矩阵，即使渲染返回空句柄也执行；不改变后续同一对象的读取。
            let restored = !scaled || api.FPDFPageObj_SetMatrix(object, &matrix) != 0;
            let bitmap = if raw.is_null() {
                None
            } else {
                Some(Bitmap { api, raw })
            };
            if !restored {
                return Err(error(ErrorCode::CorruptDocument, "无法还原图片矩阵"));
            }
            let bitmap = bitmap.ok_or_else(|| error(ErrorCode::CorruptDocument, "无法解码原图"))?;
            let width = api.FPDFBitmap_GetWidth(bitmap.raw);
            let height = api.FPDFBitmap_GetHeight(bitmap.raw);
            let pixels = self.check_pixels(width as f64, height as f64)?;
            let (format, channels, color, mode) = match api.FPDFBitmap_GetFormat(bitmap.raw) {
                1 => (1, 1, ExtendedColorType::L8, "L"),
                2 => (2, 3, ExtendedColorType::Rgb8, "RGB"),
                4 => (4, 4, ExtendedColorType::Rgba8, "RGBA"),
                _ => return Err(error(ErrorCode::Unsupported, "原图位图模式暂不支持PNG导出")),
            };
            let stride = api.FPDFBitmap_GetStride(bitmap.raw);
            let row = (width as usize)
                .checked_mul(channels)
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "图片行溢出"))?;
            if stride < 0 || (stride as usize) < row {
                return Err(error(ErrorCode::CorruptDocument, "图片stride无效"));
            }
            let size = (stride as usize)
                .checked_mul(height as usize)
                .filter(|n| *n <= rapiddoc_core::resources::limit(128_000_000))
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "图片缓冲超限"))?;
            let ptr = api.FPDFBitmap_GetBuffer(bitmap.raw);
            if ptr.is_null() {
                return Err(error(ErrorCode::CorruptDocument, "空图片缓冲"));
            }
            let data = std::slice::from_raw_parts(ptr.cast::<u8>(), size);
            let mut rgb = Vec::with_capacity(pixels * channels);
            for y in 0..height as usize {
                let row = &data[y * stride as usize..y * stride as usize + row];
                for p in row.chunks_exact(channels) {
                    if format == 1 {
                        rgb.push(p[0]);
                    } else {
                        rgb.extend_from_slice(&[p[2], p[1], p[0]]);
                        if channels == 4 {
                            rgb.push(p[3]);
                        }
                    }
                }
            }
            let mut png = Vec::new();
            PngEncoder::new(&mut png)
                .write_image(&rgb, width as u32, height as u32, color)
                .map_err(|e| error(ErrorCode::CorruptDocument, &e.to_string()))?;
            self.bytes = self
                .bytes
                .checked_add(png.len())
                .filter(|n| {
                    *n <= rapiddoc_core::resources::limit(128_000_000)
                        && png.len() <= rapiddoc_core::resources::limit(64_000_000)
                })
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "PNG累计超过限制"))?;
            self.pixels += pixels;
            Ok(Some(OriginalImage {
                uuid: format!("pdf-image-{}-{}", self.page_index, self.output.images.len()),
                bbox: [l, self.height - t, r, self.height - b],
                width: width as u32,
                height: height as u32,
                mode: mode.into(),
                png,
            }))
        }
    }
    fn check_pixels(&self, w: f64, h: f64) -> Result<usize, Error> {
        if !w.is_finite()
            || !h.is_finite()
            || w < 1.
            || h < 1.
            || w > rapiddoc_core::resources::limit(16_000_000.)
            || h > rapiddoc_core::resources::limit(16_000_000.)
        {
            return Err(error(ErrorCode::ResourceLimit, "原图尺寸超过限制"));
        }
        let pixels = (w as usize)
            .checked_mul(h as usize)
            .filter(|n| {
                *n <= rapiddoc_core::resources::limit(16_000_000)
                    && *n
                        <= rapiddoc_core::resources::limit(self.config.max_pixels)
                            .saturating_sub(self.pixels)
            })
            .ok_or_else(|| error(ErrorCode::ResourceLimit, "原图像素超过限制"))?;
        Ok(pixels)
    }
}
