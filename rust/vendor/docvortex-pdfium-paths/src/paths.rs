//! PDFium Path 段读取与子路径组装；仅保存点及共享端点索引。

use crate::ReadError;
use std::ffi::{c_float, c_int, c_void};

pub type Point = (f64, f64);
type CountFn = unsafe extern "system" fn(*mut c_void) -> c_int;
type GetFn = unsafe extern "system" fn(*mut c_void, c_int) -> *mut c_void;
type PointFn = unsafe extern "system" fn(*mut c_void, *mut c_float, *mut c_float) -> c_int;
type SegmentFlagFn = unsafe extern "system" fn(*mut c_void) -> c_int;

/// Path 段读取 ABI；同一页所有对象复用函数指针，不做逐对象重建。
pub struct Api {
    pub(crate) count: CountFn,
    pub(crate) get: GetFn,
    pub(crate) point: PointFn,
    pub(crate) kind: SegmentFlagFn,
    pub(crate) closes: SegmentFlagFn,
}

#[derive(Default)]
pub struct Subpath {
    pub points: Vec<Point>,
    pub lines: Vec<(usize, usize)>,
    pub closed: bool,
}

/// 借用同库函数一次读完路径，曲线控制点保留，但不误作直线段。
///
/// # Safety
/// 调用方须验证 ABI，保持对象所属页面存活，并持有 PDFium 全局访问锁。
pub unsafe fn read_subpaths(
    addresses: Vec<usize>,
    handle: usize,
) -> Result<Vec<Subpath>, ReadError> {
    if addresses.len() != 5 || addresses.contains(&0) || handle == 0 {
        return Err(ReadError::InvalidInput(
            "invalid PDFium path bridge arguments",
        ));
    }
    let api = Api {
        count: std::mem::transmute::<usize, CountFn>(addresses[0]),
        get: std::mem::transmute::<usize, GetFn>(addresses[1]),
        point: std::mem::transmute::<usize, PointFn>(addresses[2]),
        kind: std::mem::transmute::<usize, SegmentFlagFn>(addresses[3]),
        closes: std::mem::transmute::<usize, SegmentFlagFn>(addresses[4]),
    };
    read_subpaths_with(&api, handle)
}

/// 使用已核验 ABI 解码一个 Path；段读取失败时保留参考实现的跳过语义。
///
/// # Safety
/// 调用方须保持对象所属页面存活，并持有 PDFium 全局访问锁。
pub unsafe fn read_subpaths_with(api: &Api, handle: usize) -> Result<Vec<Subpath>, ReadError> {
    let mut output = Vec::new();
    let mut current = Subpath::default();
    let mut current_index = 0;
    for index in 0..(api.count)(handle as *mut c_void).max(0) {
        let segment = (api.get)(handle as *mut c_void, index);
        if segment.is_null() {
            continue;
        }
        let (mut x, mut y) = (0.0, 0.0);
        if (api.point)(segment, &mut x, &mut y) == 0 {
            continue;
        }
        let value = (f64::from(x), f64::from(y));
        let segment_type = (api.kind)(segment);
        let segment_closes = (api.closes)(segment) != 0;
        if segment_type == 2 {
            if !current.points.is_empty() {
                output.push(std::mem::take(&mut current));
            }
            current.points.push(value);
            current_index = 0;
        } else if current.points.is_empty() {
            current.points.push(value);
            current_index = 0;
        } else {
            let next = current.points.len();
            current.points.push(value);
            if segment_type == 0 {
                current.lines.push((current_index, next));
            }
            current_index = next;
        }
        if segment_closes {
            // 索引相同即同一 Python 点对象；NaN 也不能凭值比较额外产生闭合边。
            if current_index != 0 && current.points[current_index] != current.points[0] {
                current.lines.push((current_index, 0));
            }
            current.closed = true;
            current_index = 0;
        }
    }
    if !current.points.is_empty() {
        output.push(current);
    }
    Ok(output)
}
