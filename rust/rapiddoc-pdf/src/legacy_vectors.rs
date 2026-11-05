//! 旧DocVortex Path读取：同库锁内借用ABI，返回独立视觉页面坐标的线/路径。
use crate::with_raw_symbols;
use docvortex_pdfium_paths::path_evidence;
use pdfium_render::prelude::*;
use rapiddoc_core::{Error, ErrorCode, Limits};
use serde::{Deserialize, Serialize};
use std::{
    ffi::{c_ulong, c_void},
    path::Path,
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Line {
    pub start: [f64; 2],
    pub end: [f64; 2],
    pub bbox: [f64; 4],
    pub width: f64,
    pub orientation: String,
}
#[derive(Debug, Serialize)]
pub struct PathInfo {
    pub bbox: [f64; 4],
    pub segment_count: i32,
    pub fill_visible: bool,
    pub stroke_visible: bool,
    pub form_depth: usize,
    pub source_index: usize,
    pub fill_rgba: Option<[u32; 4]>,
    pub rectangle_bboxes: Vec<[f64; 4]>,
}
#[derive(Debug, Serialize)]
pub struct Geometry {
    pub page_size: [f64; 2],
    pub page_bbox: [f64; 4],
    pub rotation: i32,
    pub drawing_lines: Vec<Line>,
    pub path_infos: Vec<PathInfo>,
}
fn error(code: ErrorCode, message: impl Into<String>) -> Error {
    Error {
        code,
        message: message.into(),
    }
}
fn charge(total: &mut usize, n: i32, maximum: usize) -> Result<(), Error> {
    if n < 0 {
        return Ok(());
    }
    *total = total
        .checked_add(n as usize)
        .filter(|v| *v <= maximum)
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "PDF对象或路径段数量超限"))?;
    Ok(())
}
struct PageOwner<'a> {
    api: &'a dyn PdfiumLibraryBindings,
    document: FPDF_DOCUMENT,
    page: FPDF_PAGE,
}
impl Drop for PageOwner<'_> {
    fn drop(&mut self) {
        // SAFETY: 仅由同库创建，始终持全局锁，先关页再关文档，输入bytes仍存活。
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
struct Budget<'a> {
    limits: &'a Limits,
    objects: usize,
    segments: usize,
}
/// 只先读取计数，不解码点。避免把不受限的原生计数交给复用内核分配。
unsafe fn preflight(
    api: &dyn PdfiumLibraryBindings,
    container: usize,
    form: bool,
    depth: usize,
    budget: &mut Budget<'_>,
) -> Result<(), Error> {
    if depth >= 15 {
        return Ok(());
    }
    let n = if form {
        api.FPDFFormObj_CountObjects(container as FPDF_PAGEOBJECT)
    } else {
        api.FPDFPage_CountObjects(container as FPDF_PAGE)
    };
    charge(&mut budget.objects, n, budget.limits.max_objects_per_page)?;
    for index in 0..n.max(0) {
        let obj = if form {
            api.FPDFFormObj_GetObject(container as FPDF_PAGEOBJECT, index as c_ulong)
        } else {
            api.FPDFPage_GetObject(container as FPDF_PAGE, index)
        };
        if obj.is_null() {
            continue;
        }
        let kind = api.FPDFPageObj_GetType(obj);
        if kind != 2 && kind != 5 {
            continue;
        }
        let clip = api.FPDFPageObj_GetClipPath(obj);
        if !clip.is_null() {
            let paths = api.FPDFClipPath_CountPaths(clip);
            // Empty clip paths still cost iteration work; charge their count globally too.
            charge(
                &mut budget.segments,
                paths,
                budget.limits.max_path_segments_per_page,
            )?;
            for p in 0..paths.max(0) {
                charge(
                    &mut budget.segments,
                    api.FPDFClipPath_CountPathSegments(clip, p),
                    budget.limits.max_path_segments_per_page,
                )?;
            }
        }
        if kind == 2 {
            charge(
                &mut budget.segments,
                api.FPDFPath_CountSegments(obj),
                budget.limits.max_path_segments_per_page,
            )?;
        } else {
            preflight(api, obj as usize, true, depth + 1, budget)?;
        }
    }
    Ok(())
}
// CPython3.14 mathmodule.c vector_norm/dl_mul/dl_fast_sum，来源见 licenses/MIGRATION-NOTICES.txt 的 PYTHON-MATH-NOTICE.txt 节，完整许可见 licenses/PYTHON-MATH.txt。
fn power2(e: i32) -> f64 {
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}
pub fn hypot(x: f64, y: f64) -> f64 {
    let (a, b) = (x.abs(), y.abs());
    let maximum = a.max(b);
    if maximum.is_infinite() {
        return maximum;
    }
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if maximum == 0. {
        return maximum;
    }
    let bits = maximum.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let e = if exponent == 0 {
        64 - (bits & ((1u64 << 52) - 1)).leading_zeros() as i32 - 1074
    } else {
        exponent - 1022
    };
    if e < -1023 {
        return f64::MIN_POSITIVE * hypot(a / f64::MIN_POSITIVE, b / f64::MIN_POSITIVE);
    }
    let scale = power2(-e);
    let (mut sum, mut frac1, mut frac2) = (1., 0., 0.);
    for value in [a, b] {
        let x = value * scale;
        let hi = x * x;
        let lo = x.mul_add(x, -hi);
        let next = sum + hi;
        frac1 += lo;
        frac2 += (sum - next) + hi;
        sum = next;
    }
    let mut h = (sum - 1. + (frac1 + frac2)).sqrt();
    let hi = -h * h;
    let lo = (-h).mul_add(h, -hi);
    let next = sum + hi;
    frac1 += lo;
    frac2 += (sum - next) + hi;
    sum = next;
    let correction = sum - 1. + (frac1 + frac2);
    h += correction / (2. * h);
    h / scale
}
fn round3(v: f64) -> f64 {
    format!("{v:.3}").parse().unwrap_or(v)
}
fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut hi, mut lo) = (0f64, 0f64);
    for x in values {
        let t = hi + x;
        lo += if hi.abs() >= x.abs() {
            (hi - t) + x
        } else {
            (x - t) + hi
        };
        hi = t;
    }
    hi + lo
}
fn coordinate(l: &Line) -> f64 {
    if l.orientation == "horizontal" {
        l.start[1]
    } else {
        l.start[0]
    }
}
fn interval(l: &Line) -> (f64, f64) {
    if l.orientation == "horizontal" {
        (l.start[0], l.end[0])
    } else {
        (l.start[1], l.end[1])
    }
}
fn axis(a: [f64; 2], b: [f64; 2], width: f64, page: [f64; 2]) -> Option<Line> {
    if a.iter()
        .chain(&b)
        .chain([width].iter())
        .any(|v| !v.is_finite())
    {
        return None;
    }
    let (dx, dy) = ((b[0] - a[0]).abs(), (b[1] - a[1]).abs());
    if dx >= dy && dy <= 1f64.max(dx * 0.02) {
        let y = (a[1] + b[1]) / 2.;
        if !(0. <= y && y <= page[1]) {
            return None;
        }
        let (x0, x1) = (a[0].min(b[0]).max(0.), a[0].max(b[0]).min(page[0]));
        if x1 - x0 < 1. {
            return None;
        }
        let w = width.max(0.).max(dy);
        Some(Line {
            start: [x0, y],
            end: [x1, y],
            bbox: [x0, (y - w / 2.).max(0.), x1, (y + w / 2.).min(page[1])],
            width: w,
            orientation: "horizontal".into(),
        })
    } else if dy > dx && dx <= 1f64.max(dy * 0.02) {
        let x = (a[0] + b[0]) / 2.;
        if !(0. <= x && x <= page[0]) {
            return None;
        }
        let (y0, y1) = (a[1].min(b[1]).max(0.), a[1].max(b[1]).min(page[1]));
        if y1 - y0 < 1. {
            return None;
        }
        let w = width.max(0.).max(dx);
        Some(Line {
            start: [x, y0],
            end: [x, y1],
            bbox: [(x - w / 2.).max(0.), y0, (x + w / 2.).min(page[0]), y1],
            width: w,
            orientation: "vertical".into(),
        })
    } else {
        None
    }
}
fn combine(lines: &[&Line], page: [f64; 2]) -> Option<Line> {
    let lengths: Vec<_> = lines
        .iter()
        .map(|l| {
            let (a, b) = interval(l);
            (b - a).max(0.01)
        })
        .collect();
    let center = sum(lines.iter().zip(&lengths).map(|(l, n)| coordinate(l) * n))
        / sum(lengths.iter().copied());
    let minimum = lines
        .iter()
        .map(|l| coordinate(l))
        .fold(f64::INFINITY, f64::min);
    let maximum = lines
        .iter()
        .map(|l| coordinate(l))
        .fold(f64::NEG_INFINITY, f64::max);
    let width = lines
        .iter()
        .map(|l| l.width)
        .fold(f64::NEG_INFINITY, f64::max);
    let width = width.max(maximum - minimum + width);
    let first = lines
        .iter()
        .map(|l| interval(l).0)
        .fold(f64::INFINITY, f64::min);
    let last = lines
        .iter()
        .map(|l| interval(l).1)
        .fold(f64::NEG_INFINITY, f64::max);
    if lines[0].orientation == "horizontal" {
        axis([first, center], [last, center], width, page)
    } else {
        axis([center, first], [center, last], width, page)
    }
}
fn merge(lines: &[Line], page: [f64; 2]) -> Vec<Line> {
    let mut output = Vec::new();
    for orientation in ["horizontal", "vertical"] {
        let mut sorted: Vec<_> = lines
            .iter()
            .filter(|l| l.orientation == orientation)
            .collect();
        sorted.sort_by(|a, b| {
            coordinate(a)
                .partial_cmp(&coordinate(b))
                .unwrap()
                .then(interval(a).0.partial_cmp(&interval(b).0).unwrap())
        });
        let mut clusters: Vec<Vec<&Line>> = Vec::new();
        let mut start = 0.;
        for l in sorted {
            let coord = coordinate(l);
            if clusters.is_empty() || coord - start > 2. {
                clusters.push(Vec::new());
                start = coord;
            }
            clusters.last_mut().unwrap().push(l);
        }
        for mut cluster in clusters {
            cluster.sort_by(|a, b| interval(a).0.partial_cmp(&interval(b).0).unwrap());
            let mut group = Vec::new();
            let mut end = f64::NEG_INFINITY;
            for l in cluster {
                let (a, b) = interval(l);
                if !group.is_empty() && a > end + 2. {
                    if let Some(l) = combine(&group, page) {
                        output.push(l);
                    }
                    group.clear();
                    end = f64::NEG_INFINITY;
                }
                group.push(l);
                end = end.max(b);
            }
            if !group.is_empty() {
                if let Some(l) = combine(&group, page) {
                    output.push(l);
                }
            }
        }
    }
    output.sort_by(|a, b| {
        a.bbox[1]
            .partial_cmp(&b.bbox[1])
            .unwrap()
            .then(a.bbox[0].partial_cmp(&b.bbox[0]).unwrap())
            .then(a.orientation.cmp(&b.orientation))
    });
    output
}
const SYMBOLS: [&[u8]; 20] = [
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
    b"FPDFPath_CountSegments\0",
    b"FPDFPath_GetPathSegment\0",
    b"FPDFPathSegment_GetPoint\0",
    b"FPDFPathSegment_GetType\0",
    b"FPDFPathSegment_GetClose\0",
    b"FPDFPath_GetDrawMode\0",
    b"FPDFPageObj_GetStrokeColor\0",
    b"FPDFPageObj_GetStrokeWidth\0",
    b"FPDFPageObj_GetFillColor\0",
];
pub fn read(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
    limits: &Limits,
) -> Result<Geometry, Error> {
    limits.check_input(bytes)?;
    if password.is_some_and(|s| s.contains('\0')) {
        return Err(error(ErrorCode::InvalidInput, "密码包含NUL"));
    }
    with_raw_symbols(library, |api, symbols| {
        // SAFETY: ABI固定为当前PDFium；同库、全局锁、有效字节/文档/页/符号引用持续存活直到自有结果物化。
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
                    "无法打开PDF",
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
                return Err(error(ErrorCode::CorruptDocument, "无法读取PDF页"));
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
                || frame[2] <= frame[0]
                || frame[3] <= frame[1]
                || !matches!(rotation, 0 | 90 | 180 | 270)
            {
                return Err(error(ErrorCode::CorruptDocument, "PDF页面几何无效"));
            }
            let mut budget = Budget {
                limits,
                objects: 0,
                segments: 0,
            };
            preflight(api, owner.page as usize, false, 0, &mut budget)?;
            let addresses = SYMBOLS
                .iter()
                .map(|name| {
                    symbols
                        .get::<*const c_void>(*name)
                        .map(|s| *s as usize)
                        .map_err(|e| error(ErrorCode::RuntimeUnavailable, e.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (lines, paths) = path_evidence::read_path_evidence_with_hypot(
                addresses,
                owner.page as usize,
                frame,
                rotation,
                15,
                true,
                true,
                hypot,
                round3,
            )
            .map_err(|e| {
                error(
                    ErrorCode::CorruptDocument,
                    format!("PDF路径读取失败: {e:?}"),
                )
            })?;
            // Python _clip_object_visual_bbox rejects degenerate boxes even without clipping.
            let drawings: Vec<_> = lines
                .into_iter()
                .filter(|l| l.2 .2 > l.2 .0 && l.2 .3 > l.2 .1)
                .map(|(a, b, c, w, o)| Line {
                    start: [a.0, a.1],
                    end: [b.0, b.1],
                    bbox: [c.0, c.1, c.2, c.3],
                    width: w,
                    orientation: if o == 0 { "horizontal" } else { "vertical" }.into(),
                })
                .collect();
            let page = if [90, 270].contains(&rotation) {
                [frame[3] - frame[1], frame[2] - frame[0]]
            } else {
                [frame[2] - frame[0], frame[3] - frame[1]]
            };
            let drawing_lines = merge(&drawings, page);
            let path_infos = paths
                .into_iter()
                .filter(|p| p.0 .2 > p.0 .0 && p.0 .3 > p.0 .1)
                .map(|(b, n, f, s, d, i, color, rects)| PathInfo {
                    bbox: [b.0, b.1, b.2, b.3],
                    segment_count: n,
                    fill_visible: f,
                    stroke_visible: s,
                    form_depth: d,
                    source_index: i,
                    fill_rgba: color.map(|c| [c.0, c.1, c.2, c.3]),
                    rectangle_bboxes: rects
                        .into_iter()
                        .filter(|b| b.2 > b.0 && b.3 > b.1)
                        .map(|b| [b.0, b.1, b.2, b.3])
                        .collect(),
                })
                .collect();
            Ok(Geometry {
                page_size: size,
                page_bbox: frame,
                rotation,
                drawing_lines,
                path_infos,
            })
        }
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> serde_json::Value {
        serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/pdf-vector-primitives.json"),
            )
            .unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn cpython_hypot_round_and_legacy_line_merging() {
        let f = fixture();
        for c in f["numerics"].as_array().unwrap() {
            assert_eq!(
                format!(
                    "{:016x}",
                    hypot(c["x"].as_f64().unwrap(), c["y"].as_f64().unwrap()).to_bits()
                ),
                c["hypot_bits"].as_str().unwrap()
            );
        }
        for c in f["rounds"].as_array().unwrap() {
            assert_eq!(
                round3(c["value"].as_f64().unwrap()).to_bits(),
                c["expected"].as_f64().unwrap().to_bits()
            );
        }
        for c in f["merges"].as_array().unwrap() {
            let lines: Vec<Line> = serde_json::from_value(c["lines"].clone()).unwrap();
            assert_eq!(
                serde_json::to_value(merge(&lines, [200., 200.])).unwrap(),
                c["expected"]
            );
        }
    }
}
