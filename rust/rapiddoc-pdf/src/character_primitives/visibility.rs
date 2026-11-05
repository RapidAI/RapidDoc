//! Original visible-text clipping and proved opaque rectangle overpainting.
use super::*;
use docvortex_pdfium_paths::objects;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::c_void,
};
mod resources;
type B = [f64; 4];
type V = (bool, Option<B>);
pub(super) unsafe fn restore(
    api: &dyn PdfiumLibraryBindings,
    text: FPDF_TEXTPAGE,
    chars: &[Character],
    count: usize,
) -> Vec<Character> {
    let mut consumed = BTreeSet::new();
    let mut out = vec![];
    for c in chars {
        let i = c.char_idx;
        if consumed.contains(&i) {
            continue;
        }
        let mut c = c.clone();
        if c.char == "\u{fffd}" {
            let pair = if (0xd800..=0xdbff).contains(&c.raw_code) && i + 1 < count {
                let low = api.FPDFText_GetUnicode(text, (i + 1) as i32);
                if (0xdc00..=0xdfff).contains(&low) {
                    consumed.insert(i + 1);
                    Some((c.raw_code, low, i + 1))
                } else {
                    None
                }
            } else if (0xdc00..=0xdfff).contains(&c.raw_code) && i > 0 {
                let high = api.FPDFText_GetUnicode(text, (i - 1) as i32);
                (0xd800..=0xdbff)
                    .contains(&high)
                    .then_some((high, c.raw_code, i - 1))
            } else {
                None
            };
            if let Some((high, low, other)) = pair {
                c.char = char::from_u32(0x10000 + ((high - 0xd800) << 10) + (low - 0xdc00))
                    .unwrap()
                    .to_string();
                c.source_indices.push(other);
                c.source_indices.sort_unstable();
                c.source_indices.dedup();
            }
        }
        out.push(c)
    }
    out
}
const SYMBOLS: &[&[u8]] = &[
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
    b"FPDFTextObj_GetTextRenderMode\0",
    b"FPDFPageObj_GetFillColor\0",
    b"FPDFPageObj_GetStrokeColor\0",
    b"FPDFPageObj_GetBounds\0",
    b"FPDFPath_CountSegments\0",
    b"FPDFPath_GetDrawMode\0",
];
fn intersection(a: B, b: B) -> B {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}
fn valid(b: B) -> bool {
    b[2] > b[0] && b[3] > b[1]
}
pub(super) fn admit(c: &mut Character, visibility: Option<V>) -> bool {
    let Some((visible, clip)) = visibility else {
        return true;
    };
    if !visible {
        return false;
    }
    let Some(clip) = clip else { return true };
    let ink = c.tight_bbox.flatten().unwrap_or(c.bbox);
    if !c.char.is_empty()
        && c.char
            .chars()
            .all(|c| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}'))
    {
        return clip[0] <= (ink[0] + ink[2]) * 0.5
            && (ink[0] + ink[2]) * 0.5 <= clip[2]
            && clip[1] <= (ink[1] + ink[3]) * 0.5
            && (ink[1] + ink[3]) * 0.5 <= clip[3];
    }
    let visible = intersection(ink, clip);
    if !valid(visible) {
        return false;
    }
    if ink != visible {
        let b = intersection(c.bbox, clip);
        c.bbox = if valid(b) { b } else { visible };
        for b in [&mut c.loose_bbox, &mut c.tight_bbox] {
            if let Some(Some(old)) = b {
                let new = intersection(*old, clip);
                *old = if valid(new) { new } else { visible };
            }
        }
    }
    true
}
fn rounded(v: f64) -> f64 {
    (v * 1000.).round_ties_even() / 1000.
}
unsafe fn rectangular_clip(api: &dyn PdfiumLibraryBindings, raw: FPDF_PAGEOBJECT) -> bool {
    let clip = api.FPDFPageObj_GetClipPath(raw);
    let count = if clip.is_null() {
        0
    } else {
        api.FPDFClipPath_CountPaths(clip)
    };
    if count > rapiddoc_core::resources::limit(20000) {
        return false;
    }
    for p in 0..count {
        let n = api.FPDFClipPath_CountPathSegments(clip, p);
        if n > rapiddoc_core::resources::limit(20000) {
            return false;
        }
        let mut points = vec![];
        for j in 0..n {
            let s = api.FPDFClipPath_GetPathSegment(clip, p, j);
            if s.is_null() || !matches!(api.FPDFPathSegment_GetType(s), 0 | 2) {
                return false;
            }
            let (mut x, mut y) = (0., 0.);
            if api.FPDFPathSegment_GetPoint(s, &mut x, &mut y) == 0 {
                return false;
            }
            points.push([rounded(f64::from(x)), rounded(f64::from(y))]);
        }
        if points.is_empty() {
            return false;
        }
        let mut xs = points.iter().map(|p| p[0]).collect::<Vec<_>>();
        let mut ys = points.iter().map(|p| p[1]).collect::<Vec<_>>();
        xs.sort_by(f64::total_cmp);
        ys.sort_by(f64::total_cmp);
        xs.dedup();
        ys.dedup();
        if xs.len() != 2
            || ys.len() != 2
            || xs
                .iter()
                .any(|x| ys.iter().any(|y| !points.contains(&[*x, *y])))
            || points
                .iter()
                .zip(points.iter().cycle().skip(1))
                .any(|(a, b)| a[0] != b[0] && a[1] != b[1])
        {
            return false;
        }
    }
    true
}
fn transform(p: [f64; 2], m: [f64; 6], frame: B, r: i32) -> [f64; 2] {
    let x = m[0] * p[0] + m[2] * p[1] + m[4];
    let y = m[1] * p[0] + m[3] * p[1] + m[5];
    match r {
        90 => [y - frame[1], x - frame[0]],
        180 => [frame[2] - x, y - frame[1]],
        270 => [frame[3] - y, frame[2] - x],
        _ => [x - frame[0], frame[3] - y],
    }
}
fn union(points: &[[f64; 2]]) -> B {
    [
        points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
        points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
        points
            .iter()
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max),
        points
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max),
    ]
}
unsafe fn rectangle(
    api: &dyn PdfiumLibraryBindings,
    obj: &objects::Object,
    frame: B,
    r: i32,
) -> Option<B> {
    let (address, m, _, depth, clip) = *obj;
    let raw = address as FPDF_PAGEOBJECT;
    if depth != 0
        || api.FPDFPageObj_GetType(raw) != 2
        || !(4..=5).contains(&api.FPDFPath_CountSegments(raw))
    {
        return None;
    }
    let (mut fill, mut stroke) = (0, 0);
    if api.FPDFPath_GetDrawMode(raw, &mut fill, &mut stroke) == 0 || fill == 0 {
        return None;
    }
    let (mut red, mut green, mut blue, mut alpha) = (0, 0, 0, 0);
    if api.FPDFPageObj_GetFillColor(raw, &mut red, &mut green, &mut blue, &mut alpha) == 0
        || alpha != 255
        || !rectangular_clip(api, raw)
    {
        return None;
    }
    let mut points = vec![];
    let mut closed = false;
    for i in 0..api.FPDFPath_CountSegments(raw) {
        let s = api.FPDFPath_GetPathSegment(raw, i);
        if s.is_null() {
            return None;
        }
        let t = api.FPDFPathSegment_GetType(s);
        if t == 2 && !points.is_empty() || !matches!(t, 0 | 2) {
            return None;
        }
        let (mut x, mut y) = (0., 0.);
        if api.FPDFPathSegment_GetPoint(s, &mut x, &mut y) == 0 {
            return None;
        }
        points.push(transform([f64::from(x), f64::from(y)], m, frame, r));
        closed |= api.FPDFPathSegment_GetClose(s) != 0;
    }
    if !closed
        || !(4..=5).contains(&points.len())
        || points.iter().flatten().any(|v| !v.is_finite())
    {
        return None;
    }
    let b = union(&points);
    if !valid(b)
        || [b[0], b[2]].iter().any(|x| {
            [b[1], b[3]].iter().any(|y| {
                !points
                    .iter()
                    .any(|p| rounded(p[0]) == rounded(*x) && rounded(p[1]) == rounded(*y))
            })
        })
        || points
            .windows(2)
            .any(|p| (p[0][0] - p[1][0]).abs() > 0.001 && (p[0][1] - p[1][1]).abs() > 0.001)
    {
        return None;
    }
    let b = if let Some(clip) = clip {
        if !valid(clip) {
            return None;
        }
        let ps = [
            [clip[0], clip[1]],
            [clip[0], clip[3]],
            [clip[2], clip[1]],
            [clip[2], clip[3]],
        ]
        .map(|p| transform(p, [1., 0., 0., 1., 0., 0.], frame, r));
        intersection(b, union(&ps))
    } else {
        b
    };
    valid(b).then_some(b)
}
pub(super) unsafe fn read(
    api: &dyn PdfiumLibraryBindings,
    symbols: &libloading::Library,
    page: FPDF_PAGE,
    frame: B,
    rotation: i32,
    bytes: &[u8],
    page_index: usize,
    limits: &Limits,
) -> Result<BTreeMap<usize, V>, Error> {
    // ABI verified against the configured PDFium; addresses are consumed under this page's runtime lock.
    let addresses = SYMBOLS
        .iter()
        .map(|name| {
            symbols
                .get::<*const c_void>(*name)
                .map(|s| *s as usize)
                .map_err(|e| error(ErrorCode::RuntimeUnavailable, &e.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (records, roots, paint) =
        objects::read_text_visibility_with_roots(addresses, page as usize, frame, rotation, 15)
            .map_err(|e| {
                error(
                    ErrorCode::ResourceLimit,
                    &format!("文字可见性读取失败: {e:?}"),
                )
            })?;
    if records.len() > limits.max_objects_per_page {
        return Err(error(ErrorCode::ResourceLimit, "文字对象超过限制"));
    }
    let mut visible = records
        .into_iter()
        .map(|(a, v, c)| (a, (v, c)))
        .collect::<BTreeMap<_, _>>();
    let covers = roots
        .iter()
        .filter_map(|(order, obj)| rectangle(api, obj, frame, rotation).map(|b| (*order, b)))
        .collect::<Vec<_>>();
    let hidden = paint
        .iter()
        .filter(|(order, _, b)| {
            covers.iter().any(|(later, c)| {
                later > order && c[0] <= b[0] && c[1] <= b[1] && c[2] >= b[2] && c[3] >= b[3]
            })
        })
        .map(|(_, address, _)| *address)
        .collect::<BTreeSet<_>>();
    if !hidden.is_empty() && resources::safe(bytes, page_index) {
        for address in hidden {
            if let Some(v) = visible.get_mut(&address) {
                v.0 = false
            }
        }
    }
    Ok(visible)
}
