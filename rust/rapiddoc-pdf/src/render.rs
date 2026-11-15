//! 受像素预算约束的按页渲染；位图编码在释放 PDFium 锁后执行。

use crate::with_raw;
use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder};
use pdfium_render::prelude::*;
use rapiddoc_core::{Error, ErrorCode, Limits};
use std::path::Path;

// PDFium public C API constants (pdfium-render does not re-export these).
const FPDFBITMAP_BGR: i32 = 2;
const FPDFBITMAP_BGRA: i32 = 4;
const FPDF_ANNOT: i32 = 0x01;
const FPDF_REVERSE_BYTE_ORDER: i32 = 0x10;

/// 完全自有 PNG 资产，调用方无需维护 PDFium 文档或位图句柄。
pub struct RenderedPage {
    pub width: u32,
    pub height: u32,
    pub png: Vec<u8>,
    pub scale: f64,
}

/// Original public PDF API: 200 DPI by default, capped to a 3500 pixel long edge.
pub fn render_model_page(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    dpi: f64,
    max_pixels: usize,
    password: Option<&str>,
    limits: &Limits,
) -> Result<RenderedPage, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    render_mode(
        library,
        bytes,
        page_index,
        dpi,
        max_pixels,
        password,
        limits,
        true,
        Some(3500.),
    )
}

/// Keep the exact pre-encoding RGB buffer for the current page's layout model.
/// The public PNG asset is still encoded once with the original settings.
#[allow(clippy::too_many_arguments)]
pub fn render_model_page_with_pixels(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    dpi: f64,
    max_pixels: usize,
    password: Option<&str>,
    limits: &Limits,
) -> Result<(RenderedPage, Vec<u8>), Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    let (page, pixels) = render_mode_pixels(
        library,
        bytes,
        page_index,
        dpi,
        max_pixels,
        password,
        limits,
        true,
        Some(3500.),
    )?;
    Ok((page, pixels))
}

/// 渲染单页为白底 RGBA PNG。dpi 为每英寸像素数，限制在 0..=1200；max_pixels 为像素总量。
pub fn render_page(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    dpi: f64,
    max_pixels: usize,
    password: Option<&str>,
    limits: &Limits,
) -> Result<RenderedPage, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    render_mode(
        library, bytes, page_index, dpi, max_pixels, password, limits, false, None,
    )
}

/// 与 RGBA 渲染相同的白底像素，按旧 PIL convert("RGB") 丢弃 alpha。
pub fn render_page_rgb(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    dpi: f64,
    max_pixels: usize,
    password: Option<&str>,
    limits: &Limits,
) -> Result<RenderedPage, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    render_mode(
        library, bytes, page_index, dpi, max_pixels, password, limits, true, None,
    )
}

/// 旧原生表内媒体：RGB，scale=2，长边最多3500像素。
pub fn render_table_media_page(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    max_pixels: usize,
    password: Option<&str>,
    limits: &Limits,
) -> Result<RenderedPage, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    render_mode(
        library,
        bytes,
        page_index,
        144.,
        max_pixels,
        password,
        limits,
        true,
        Some(3500.),
    )
}

#[allow(clippy::too_many_arguments)]
fn render_mode(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    dpi: f64,
    max_pixels: usize,
    password: Option<&str>,
    limits: &Limits,
    rgb: bool,
    max_edge: Option<f64>,
) -> Result<RenderedPage, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    render_mode_pixels(
        library, bytes, page_index, dpi, max_pixels, password, limits, rgb, max_edge,
    )
    .map(|(page, _)| page)
}

#[allow(clippy::too_many_arguments)]
fn render_mode_pixels(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    dpi: f64,
    max_pixels: usize,
    password: Option<&str>,
    limits: &Limits,
    rgb: bool,
    max_edge: Option<f64>,
) -> Result<(RenderedPage, Vec<u8>), Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    limits.check_input(bytes)?;
    if !dpi.is_finite() || dpi <= 0. || dpi > 1200. || max_pixels == 0 {
        return Err(error(ErrorCode::InvalidInput, "dpi 或像素预算无效"));
    }
    if password.is_some_and(|value| value.contains('\0')) {
        return Err(error(ErrorCode::InvalidInput, "密码包含NUL"));
    }
    // All handles remain under the existing process-wide PDFium lock. Use the
    // same viewport API as pypdfium2 without initializing a form-fill environment.
    let (width, height, pixels, scale) = with_raw(library, |api| unsafe {
        let mut handles = RenderHandles {
            api,
            document: api.FPDF_LoadMemDocument64(bytes, password),
            page: std::ptr::null_mut(),
            bitmap: std::ptr::null_mut(),
        };
        if handles.document.is_null() {
            let code = if matches!(api.FPDF_GetLastError(), 4 | 5) {
                ErrorCode::Encrypted
            } else {
                ErrorCode::CorruptDocument
            };
            return Err(error(code, "无法打开渲染 PDF"));
        }
        let count = usize::try_from(api.FPDF_GetPageCount(handles.document))
            .map_err(|_| error(ErrorCode::CorruptDocument, "无效页数"))?;
        if count > limits.max_pages {
            return Err(error(ErrorCode::ResourceLimit, "文档页数超过限制"));
        }
        if page_index >= count {
            return Err(error(ErrorCode::InvalidInput, "渲染页号越界"));
        }
        handles.page = api.FPDF_LoadPage(handles.document, page_index as i32);
        if handles.page.is_null() {
            return Err(error(ErrorCode::CorruptDocument, "无法读取渲染页"));
        }
        let page_width = api.FPDF_GetPageWidthF(handles.page) as f64;
        let page_height = api.FPDF_GetPageHeightF(handles.page) as f64;
        let scale = max_edge.map_or(dpi / 72., |edge| {
            (dpi / 72.).min(edge / page_width.max(page_height))
        });
        let (width, height) = if let Some(edge) = max_edge {
            let scale = (dpi / 72.).min(edge / page_width.max(page_height));
            ((page_width * scale).ceil(), (page_height * scale).ceil())
        } else {
            (
                (page_width * dpi / 72.).ceil(),
                (page_height * dpi / 72.).ceil(),
            )
        };
        if !width.is_finite()
            || !height.is_finite()
            || width < 1.
            || height < 1.
            || width > i32::MAX as f64
            || height > i32::MAX as f64
        {
            return Err(error(ErrorCode::ResourceLimit, "渲染尺寸无效或过大"));
        }
        let width = width as usize;
        let height = height as usize;
        let pixels = width
            .checked_mul(height)
            .ok_or_else(|| error(ErrorCode::ResourceLimit, "像素数量溢出"))?;
        let size = pixels
            .checked_mul(4)
            .ok_or_else(|| error(ErrorCode::ResourceLimit, "位图字节数溢出"))?;
        if pixels > max_pixels || size > rapiddoc_core::resources::limit(512 * 1024 * 1024) {
            return Err(error(ErrorCode::ResourceLimit, "渲染超过像素或内存预算"));
        }
        // pypdfium2's opaque default uses BGR. Rendering to BGRA and then
        // dropping alpha can change antialias rounding (and every crop name).
        handles.bitmap = api.FPDFBitmap_CreateEx(
            width as i32,
            height as i32,
            if rgb {
                FPDFBITMAP_BGR as i32
            } else {
                FPDFBITMAP_BGRA as i32
            },
            std::ptr::null_mut(),
            0,
        );
        if handles.bitmap.is_null() {
            return Err(error(ErrorCode::CorruptDocument, "无法创建渲染位图"));
        }
        api.FPDFBitmap_FillRect(
            handles.bitmap,
            0,
            0,
            width as i32,
            height as i32,
            0xffffffff,
        );
        // Matrix rendering resamples differently; enabling pdfium-render's form
        // mode selects the viewport API but also paints widgets absent in Python.
        api.FPDF_RenderPageBitmap(
            handles.bitmap,
            handles.page,
            0,
            0,
            width as i32,
            height as i32,
            0,
            (FPDF_ANNOT | FPDF_REVERSE_BYTE_ORDER) as i32,
        );
        let buffer = api.FPDFBitmap_GetBuffer_as_vec(handles.bitmap);
        let pixels = if rgb {
            // reverse_byte_order already made the BGR bitmap RGB. Retain those
            // exact bytes, removing row padding without expanding to RGBA.
            packed_rgb(buffer, width, height)?
        } else {
            let rgba = buffer;
            if rgba.len() != size {
                return Err(error(ErrorCode::CorruptDocument, "RGBA 位图长度不匹配"));
            }
            rgba
        };
        Ok((width as u32, height as u32, pixels, scale))
    })?;
    let color = if rgb {
        ExtendedColorType::Rgb8
    } else {
        ExtendedColorType::Rgba8
    };
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&pixels, width, height, color)
        .map_err(|value| Error {
            code: ErrorCode::InvalidInput,
            message: value.to_string(),
        })?;
    Ok((
        RenderedPage {
            width,
            height,
            png,
            scale,
        },
        pixels,
    ))
}

struct RenderHandles<'a> {
    api: &'a dyn PdfiumLibraryBindings,
    document: FPDF_DOCUMENT,
    page: FPDF_PAGE,
    bitmap: FPDF_BITMAP,
}

impl Drop for RenderHandles<'_> {
    fn drop(&mut self) {
        // Release in dependency order before the runtime lock is released.
        unsafe {
            if !self.bitmap.is_null() {
                self.api.FPDFBitmap_Destroy(self.bitmap);
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

fn packed_rgb(mut bytes: Vec<u8>, width: usize, height: usize) -> Result<Vec<u8>, Error> {
    let row = width
        .checked_mul(3)
        .ok_or_else(|| error(ErrorCode::CorruptDocument, "RGB 位图长度无效"))?;
    if width == 0 || height == 0 || bytes.len() % height != 0 || bytes.len() / height < row {
        return Err(error(ErrorCode::CorruptDocument, "RGB 位图长度不匹配"));
    }
    let stride = bytes.len() / height;
    if stride != row {
        for y in 1..height {
            bytes.copy_within(y * stride..y * stride + row, y * row);
        }
        bytes.truncate(row * height);
    }
    Ok(bytes)
}

/// 构造渲染错误，资源上限与无效输入保持不同分类。
fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}

#[cfg(test)]
mod api_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    #[test]
    fn rgb_padding_preserves_every_channel_and_row() {
        for width in 1..=9 {
            for height in 1..=4 {
                let expected: Vec<u8> = (0..width * height)
                    .flat_map(|i| [i as u8, (i * 3 + 17) as u8, (255 - i) as u8])
                    .collect();
                for padding in 0..=7 {
                    let mut input = Vec::new();
                    for row in expected.chunks(width * 3) {
                        input.extend_from_slice(row);
                        input.extend(std::iter::repeat_n(0xee, padding));
                    }
                    assert_eq!(packed_rgb(input, width, height).unwrap(), expected);
                }
            }
        }
        assert!(packed_rgb(vec![0; 2], 1, 1).is_err());
        assert!(packed_rgb(vec![0; 7], 1, 2).is_err());
        assert!(packed_rgb(vec![], 1, 0).is_err());
        assert!(packed_rgb(vec![], usize::MAX, 1).is_err());
    }
    #[test]
    fn original_api_long_edge_pixels() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/api-render-caps.json")).unwrap(),
        )
        .unwrap();
        let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
        for c in reference["cases"].as_array().unwrap() {
            let bytes = std::fs::read(root.join(c["source"].as_str().unwrap())).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                c["source_sha256"].as_str().unwrap()
            );
            let rendered = render_model_page(
                &library,
                &bytes,
                c["page_index"].as_u64().unwrap() as usize,
                200.,
                16_000_000,
                None,
                &Limits::default(),
            )
            .unwrap();
            assert_eq!(
                (rendered.width, rendered.height, rendered.scale),
                (
                    c["width"].as_u64().unwrap() as u32,
                    c["height"].as_u64().unwrap() as u32,
                    c["scale"].as_f64().unwrap()
                )
            );
            let pixels = image::load_from_memory(&rendered.png)
                .unwrap()
                .into_rgb8()
                .into_raw();
            assert_eq!(
                format!("{:x}", Sha256::digest(pixels)),
                c["pixels_sha256"].as_str().unwrap(),
                "{}",
                c["source"]
            );
        }
    }
}
