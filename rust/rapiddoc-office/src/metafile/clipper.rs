//! Clipper 6.4.2 FFI; no replacement geometry algorithm or modified core.
use super::{geometry::FlattenBudget, malformed, model::Point, resource};
use crate::Result;
use std::ffi::c_void;
// Load clipper-sys's bundled native core into this crate's link dependency graph.
extern crate clipper_sys;
#[repr(C)]
#[derive(Clone, Copy)]
struct IntPoint {
    x: clipper_sys::cInt,
    y: clipper_sys::cInt,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct RawPath {
    points: *const IntPoint,
    count: usize,
}
type Visit = unsafe extern "C" fn(*mut c_void, i32, *const IntPoint, usize) -> i32;
extern "C" {
    fn rd_office_union(
        paths: *const RawPath,
        count: usize,
        fill: i32,
        visit: Visit,
        context: *mut c_void,
    ) -> i32;
    fn rd_office_offset(
        path: RawPath,
        join: i32,
        end: i32,
        miter: f64,
        delta: f64,
        visit: Visit,
        context: *mut c_void,
    ) -> i32;
}
#[derive(Debug, serde::Serialize)]
pub struct Contour {
    pub hole: bool,
    pub points: Vec<[i64; 2]>,
}
pub fn quantize(points: &[Point], closed: bool) -> Result<Vec<[i64; 2]>> {
    let mut result = Vec::new();
    for &[x, y] in points {
        if x.abs() > ((1u64 << 61) as f64 / 256.) || y.abs() > ((1u64 << 61) as f64 / 256.) {
            return Err(resource("metafile coordinate exceeds Python Clipper range"));
        }
        let p = [
            (x * 256.).round_ties_even() as i64,
            (y * 256.).round_ties_even() as i64,
        ];
        if result.last() != Some(&p) {
            result.push(p);
        }
    }
    if closed && result.len() > 1 && result.first() == result.last() {
        result.pop();
    }
    Ok(result)
}
struct Output<'a> {
    contours: Vec<Contour>,
    budget: &'a mut FlattenBudget,
    error: Option<crate::Error>,
}
unsafe extern "C" fn collect(
    context: *mut c_void,
    hole: i32,
    points: *const IntPoint,
    count: usize,
) -> i32 {
    // The bridge calls synchronously, never retains the context or input, and
    // supplies count elements valid for the duration of this callback.
    let output = unsafe { &mut *context.cast::<Output<'_>>() };
    if let Err(e) = output.budget.charge_many(count) {
        output.error = Some(e);
        return 0;
    }
    let points = if count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(points, count) }
    };
    output.contours.push(Contour {
        hole: hole != 0,
        points: points.iter().map(|p| [p.x, p.y]).collect(),
    });
    1
}
fn finish(status: i32, output: Output<'_>) -> Result<Vec<Contour>> {
    if let Some(e) = output.error {
        return Err(e);
    }
    if status != 0 {
        return Err(malformed());
    }
    Ok(output.contours)
}
pub fn union(
    paths: &[Vec<[i64; 2]>],
    evenodd: bool,
    budget: &mut FlattenBudget,
) -> Result<Vec<Contour>> {
    if paths.is_empty() {
        return Ok(vec![]);
    }
    let points: Vec<Vec<_>> = paths
        .iter()
        .map(|p| p.iter().map(|&[x, y]| IntPoint { x, y }).collect())
        .collect();
    let raw: Vec<_> = points
        .iter()
        .map(|p| RawPath {
            points: p.as_ptr(),
            count: p.len(),
        })
        .collect();
    let mut output = Output {
        contours: vec![],
        budget,
        error: None,
    };
    // Raw paths and output stay alive and uniquely borrowed for the whole call.
    let status = unsafe {
        rd_office_union(
            raw.as_ptr(),
            raw.len(),
            if evenodd { 0 } else { 1 },
            collect,
            (&mut output as *mut Output<'_>).cast(),
        )
    };
    finish(status, output)
}
pub fn offset(
    points: &[[i64; 2]],
    join: i32,
    end: i32,
    miter: f64,
    width: f64,
    budget: &mut FlattenBudget,
) -> Result<Vec<Contour>> {
    let points: Vec<_> = points.iter().map(|&[x, y]| IntPoint { x, y }).collect();
    let mut output = Output {
        contours: vec![],
        budget,
        error: None,
    };
    let status = unsafe {
        rd_office_offset(
            RawPath {
                points: points.as_ptr(),
                count: points.len(),
            },
            join,
            end,
            miter.max(1.),
            width.max(1.) * 256. / 2.,
            collect,
            (&mut output as *mut Output<'_>).cast(),
        )
    };
    finish(status, output)
}
