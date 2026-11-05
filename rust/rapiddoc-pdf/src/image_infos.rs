//! 旧PDFImageInfo读取；原始流指纹和独立像素证据不依赖Python/Pillow运行时。
use crate::with_raw_symbols;
use docvortex_pdfium_paths::objects;
use pdfium_render::prelude::*;
use rapiddoc_core::{
    pillow_resize::{resize, Rgb, Work},
    Error, ErrorCode, Limits,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    ffi::{c_ulong, c_void},
    path::Path,
};

#[derive(Debug, Serialize)]
pub struct ImageInfo {
    pub bbox: [f64; 4],
    pub fingerprint: Option<String>,
    pub smooth_background: bool,
    pub blank_top_bbox: Option<[f64; 4]>,
}
#[derive(Debug, Serialize)]
pub struct PageImages {
    pub page_size: [f64; 2],
    pub page_bbox: [f64; 4],
    pub rotation: i32,
    pub image_infos: Vec<ImageInfo>,
}
fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}
fn limit() -> Error {
    error(ErrorCode::ResourceLimit, "PDF图片证据超过资源预算")
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
struct Bitmap<'a> {
    api: &'a dyn PdfiumLibraryBindings,
    raw: FPDF_BITMAP,
}
impl Drop for Bitmap<'_> {
    fn drop(&mut self) {
        unsafe { self.api.FPDFBitmap_Destroy(self.raw) }
    }
}
struct Budget {
    objects: usize,
    segments: usize,
    raw_bytes: usize,
    pixels: usize,
    resize: Work,
}
fn add(v: &mut usize, n: usize, maximum: usize) -> Result<(), Error> {
    *v = v
        .checked_add(n)
        .filter(|v| *v <= maximum)
        .ok_or_else(limit)?;
    Ok(())
}
// Preflight counts objects and all clip segments before the fixed object walker allocates.
unsafe fn preflight(
    api: &dyn PdfiumLibraryBindings,
    handle: usize,
    form: bool,
    depth: usize,
    b: &mut Budget,
    limits: &Limits,
) -> Result<(), Error> {
    if depth >= 15 {
        return Ok(());
    }
    let count = if form {
        api.FPDFFormObj_CountObjects(handle as FPDF_PAGEOBJECT)
    } else {
        api.FPDFPage_CountObjects(handle as FPDF_PAGE)
    };
    add(
        &mut b.objects,
        count.max(0) as usize,
        limits.max_objects_per_page,
    )?;
    for i in 0..count.max(0) {
        let obj = if form {
            api.FPDFFormObj_GetObject(handle as FPDF_PAGEOBJECT, i as c_ulong)
        } else {
            api.FPDFPage_GetObject(handle as FPDF_PAGE, i)
        };
        if obj.is_null() {
            continue;
        }
        let kind = api.FPDFPageObj_GetType(obj);
        {
            let clip = api.FPDFPageObj_GetClipPath(obj);
            if !clip.is_null() {
                let paths = api.FPDFClipPath_CountPaths(clip).max(0);
                add(
                    &mut b.segments,
                    paths as usize,
                    limits.max_path_segments_per_page,
                )?;
                for i in 0..paths {
                    add(
                        &mut b.segments,
                        api.FPDFClipPath_CountPathSegments(clip, i).max(0) as usize,
                        limits.max_path_segments_per_page,
                    )?;
                }
            }
        }
        if kind == 5 {
            preflight(api, obj as usize, true, depth + 1, b, limits)?;
        }
    }
    Ok(())
}
fn visual(x: f64, y: f64, f: [f64; 4], r: i32) -> [f64; 2] {
    match r {
        90 => [y - f[1], x - f[0]],
        180 => [f[2] - x, y - f[1]],
        270 => [f[3] - y, f[2] - x],
        _ => [x - f[0], f[3] - y],
    }
}
fn bounds(points: impl Iterator<Item = [f64; 2]>) -> Option<[f64; 4]> {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in points {
        if p.iter().any(|v| !v.is_finite()) {
            return None;
        }
        b = [
            b[0].min(p[0]),
            b[1].min(p[1]),
            b[2].max(p[0]),
            b[3].max(p[1]),
        ];
    }
    Some(b)
}
fn intersect(a: [f64; 4], b: [f64; 4]) -> Option<[f64; 4]> {
    let b = [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ];
    (b[2] > b[0] && b[3] > b[1]).then_some(b)
}
fn geometry(m: [f64; 6], f: [f64; 4], r: i32, size: [f64; 2]) -> Option<[f64; 4]> {
    let b = bounds(
        [(0., 0.), (0., 1.), (1., 0.), (1., 1.)]
            .into_iter()
            .map(|(x, y)| visual(m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5], f, r)),
    )?;
    intersect(b, [0., 0., size[0], size[1]])
}
fn clipped(b: [f64; 4], clip: Option<[f64; 4]>, f: [f64; 4], r: i32) -> Option<[f64; 4]> {
    let Some(c) = clip else { return Some(b) };
    if c[2] <= c[0] || c[3] <= c[1] {
        return None;
    }
    intersect(
        b,
        bounds(
            [(c[0], c[1]), (c[0], c[3]), (c[2], c[1]), (c[2], c[3])]
                .into_iter()
                .map(|(x, y)| visual(x, y, f, r)),
        )?,
    )
}
unsafe fn fingerprint(
    api: &dyn PdfiumLibraryBindings,
    obj: FPDF_PAGEOBJECT,
    width: u32,
    height: u32,
    budget: &mut Budget,
) -> Result<Option<String>, Error> {
    let n = api.FPDFImageObj_GetImageDataRaw(obj, std::ptr::null_mut(), 0) as usize;
    if width == 0 || height == 0 || n == 0 || n > 16 * 1024 * 1024 {
        return Ok(None);
    }
    add(&mut budget.raw_bytes, n, 64 * 1024 * 1024)?;
    let mut data = vec![0u8; n];
    let read =
        api.FPDFImageObj_GetImageDataRaw(obj, data.as_mut_ptr().cast(), n as c_ulong) as usize;
    if read == 0 || read > n {
        return Ok(None);
    }
    let mut digest = Sha256::new();
    digest.update(format!("{width}:{height}:"));
    digest.update(&data[..read]);
    Ok(Some(format!("{:x}", digest.finalize())))
}
// Independent RGB conversion: smooth evidence drops alpha; blank-top evidence composites onto white.
unsafe fn pixels(
    api: &dyn PdfiumLibraryBindings,
    raw: FPDF_BITMAP,
    budget: &mut Budget,
) -> Result<Option<(Rgb, f64)>, Error> {
    let bitmap = Bitmap { api, raw };
    let (w, h, stride) = (
        api.FPDFBitmap_GetWidth(raw),
        api.FPDFBitmap_GetHeight(raw),
        api.FPDFBitmap_GetStride(raw),
    );
    if w <= 0 || h <= 0 || stride <= 0 {
        return Ok(None);
    }
    let (w, h, stride) = (w as usize, h as usize, stride as usize);
    let format = api.FPDFBitmap_GetFormat(raw);
    let channels = match format {
        1 => 1,
        2 => 3,
        3 | 4 => 4,
        _ => return Ok(None),
    };
    let n = w.checked_mul(h).ok_or_else(limit)?;
    if n > rapiddoc_core::resources::limit(16_000_000)
        || w.checked_mul(channels).is_none_or(|v| v > stride)
    {
        return Err(limit());
    }
    add(&mut budget.pixels, n, 64_000_000)?;
    let bytes = h
        .checked_mul(stride)
        .filter(|v| *v <= rapiddoc_core::resources::limit(128_000_000))
        .ok_or_else(limit)?;
    let ptr = api.FPDFBitmap_GetBuffer(bitmap.raw);
    if ptr.is_null() {
        return Ok(None);
    }
    let buffer = std::slice::from_raw_parts(ptr.cast::<u8>(), bytes);
    let mut rgb = Vec::with_capacity(n * 3);
    let mut first_ink = None;
    for y in 0..h {
        let mut ink = false;
        for x in 0..w {
            let s = y * stride + x * channels;
            let p = if channels == 1 {
                [buffer[s]; 3]
            } else {
                [buffer[s + 2], buffer[s + 1], buffer[s]]
            };
            rgb.extend_from_slice(&p);
            if first_ink.is_none() {
                let a = if format == 4 {
                    buffer[s + 3] as u32
                } else {
                    255
                };
                ink |= p
                    .iter()
                    .any(|v| ((*v as u32 * a + 255 * (255 - a) + 127) / 255) < 250);
            }
        }
        if ink && first_ink.is_none() {
            first_ink = Some(y)
        }
    }
    let fraction = first_ink.unwrap_or(0) as f64 / h as f64;
    let blank = if n <= 4_000_000 && (0.03..=0.12).contains(&fraction) {
        fraction
    } else {
        0.
    };
    Ok(Some((Rgb { data: rgb, w, h }, blank)))
}
fn smooth(rgb: Rgb, budget: &mut Budget) -> Result<bool, Error> {
    let end = (rgb.w as f32, rgb.h as f32);
    let small = resize(rgb, 64, 64, end, true, &mut budget.resize).map_err(|_| limit())?;
    let mut different = 0;
    for y in 0..63 {
        for x in 0..63 {
            for (dx, dy) in [(1, 0), (0, 1)] {
                if (0..3).any(|c| {
                    small.data[(y * 64 + x) * 3 + c]
                        .abs_diff(small.data[((y + dy) * 64 + x + dx) * 3 + c])
                        > 12
                }) {
                    different += 1;
                }
            }
        }
    }
    Ok(different as f64 / (63. * 63. * 2.) <= 0.015)
}
const SYMBOLS: [&[u8]; 11] = [
    b"FPDFPage_CountObjects\0",
    b"FPDFPage_GetObject\0",
    b"FPDFFormObj_CountObjects\0",
    b"FPDFFormObj_GetObject\0",
    b"FPDFPageObj_GetMatrix\0",
    b"FPDFPageObj_GetType\0",
    b"FPDFPageObj_GetClipPath\0",
    b"FPDFClipPath_CountPaths\0",
    b"FPDFClipPath_CountPathSegments\0",
    b"FPDFClipPath_GetPathSegment\0",
    b"FPDFPathSegment_GetPoint\0",
];
pub fn read(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
    limits: &Limits,
) -> Result<PageImages, Error> {
    limits.check_input(bytes)?;
    if password.is_some_and(|v| v.contains('\0')) {
        return Err(error(ErrorCode::InvalidInput, "密码包含NUL"));
    }
    with_raw_symbols(library, |api, symbols| unsafe {
        // SAFETY: Input and all same-library handles remain alive under the sole runtime lock.
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
        let count = api.FPDF_GetPageCount(owner.document);
        if count < 0 {
            return Err(error(ErrorCode::CorruptDocument, "无效页数"));
        }
        if count as usize > limits.max_pages {
            return Err(limit());
        }
        if page_index >= count as usize {
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
            .any(|n| !n.is_finite() || n.abs() > 1e9)
            || frame[2] <= frame[0]
            || frame[3] <= frame[1]
            || !matches!(rotation, 0 | 90 | 180 | 270)
        {
            return Err(error(ErrorCode::CorruptDocument, "PDF页面几何无效"));
        }
        let visual_size = if matches!(rotation, 90 | 270) {
            [frame[3] - frame[1], frame[2] - frame[0]]
        } else {
            [frame[2] - frame[0], frame[3] - frame[1]]
        };
        let mut budget = Budget {
            objects: 0,
            segments: 0,
            raw_bytes: 0,
            pixels: 0,
            resize: Work(0),
        };
        preflight(api, owner.page as usize, false, 0, &mut budget, limits)?;
        let addresses = SYMBOLS
            .iter()
            .map(|n| {
                symbols
                    .get::<*const c_void>(*n)
                    .map(|v| *v as usize)
                    .map_err(|e| Error {
                        code: ErrorCode::RuntimeUnavailable,
                        message: e.to_string(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let objects = objects::read_objects(addresses, owner.page as usize, 3, 15)
            .map_err(|_| error(ErrorCode::CorruptDocument, "PDF图片对象读取失败"))?;
        if objects.len() > rapiddoc_core::resources::limit(5000) {
            return Err(limit());
        }
        let mut infos = Vec::new();
        for (address, matrix, _, _, clip) in objects {
            let Some(full) = geometry(matrix, frame, rotation, visual_size) else {
                continue;
            };
            let Some(bbox) = clipped(full, clip, frame, rotation) else {
                continue;
            };
            let obj = address as FPDF_PAGEOBJECT;
            let mut meta: FPDF_IMAGEOBJ_METADATA = std::mem::zeroed();
            let metadata = api.FPDFImageObj_GetImageMetadata(obj, owner.page, &mut meta) != 0;
            let fingerprint = if metadata {
                fingerprint(api, obj, meta.width, meta.height, &mut budget)?
            } else {
                None
            };
            let area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
            let page_area = (frame[2] - frame[0]) * (frame[3] - frame[1]);
            let npixels = meta.width as u64 * meta.height as u64;
            let check_smooth =
                metadata && npixels > 0 && npixels <= 16_000_000 && area >= 0.95 * page_area;
            let check_blank = metadata
                && npixels > 0
                && npixels <= 4_000_000
                && rotation == 0
                && matrix[0] > 0.
                && matrix[3] > 0.
                && matrix[1].abs() + matrix[2].abs() <= 1e-6
                && (full[1] - bbox[1]).abs() <= 0.5;
            let mut smooth_background = false;
            let mut blank_top_bbox = None;
            if check_smooth || check_blank {
                let bitmap = api.FPDFImageObj_GetBitmap(obj);
                if !bitmap.is_null() {
                    if let Some((rgb, fraction)) = pixels(api, bitmap, &mut budget)? {
                        if check_blank && fraction > 0. {
                            let top = full[1] + fraction * (full[3] - full[1]);
                            if top < bbox[3] {
                                blank_top_bbox = Some([bbox[0], top, bbox[2], bbox[3]]);
                            }
                        }
                        if check_smooth {
                            smooth_background = smooth(rgb, &mut budget)?;
                        }
                    }
                }
            }
            infos.push(ImageInfo {
                bbox,
                fingerprint,
                smooth_background,
                blank_top_bbox,
            });
        }
        infos.sort_by(|a, b| {
            a.bbox[1]
                .total_cmp(&b.bbox[1])
                .then(a.bbox[0].total_cmp(&b.bbox[0]))
                .then(a.bbox[3].total_cmp(&b.bbox[3]))
                .then(a.bbox[2].total_cmp(&b.bbox[2]))
        });
        Ok(PageImages {
            page_size: size,
            page_bbox: frame,
            rotation,
            image_infos: infos,
        })
    })
}
