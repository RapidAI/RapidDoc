//! 批量生成 Path 的绘图线与摘要证据，保持 Python 参考实现的对象级隔离语义。

use crate::{objects, paths, ReadError};
use std::ffi::{c_float, c_int, c_uint, c_void};

pub type Line = ((f64, f64), (f64, f64), (f64, f64, f64, f64), f64, i32);
pub type Rgba = (c_uint, c_uint, c_uint, c_uint);
pub type PathInfo = (
    (f64, f64, f64, f64),
    c_int,
    bool,
    bool,
    usize,
    usize,
    Option<Rgba>,
    Vec<(f64, f64, f64, f64)>,
);
pub type Evidence = (Vec<Line>, Vec<PathInfo>);

type ColorFn = unsafe extern "system" fn(
    *mut c_void,
    *mut c_uint,
    *mut c_uint,
    *mut c_uint,
    *mut c_uint,
) -> c_int;
type DrawModeFn = unsafe extern "system" fn(*mut c_void, *mut c_int, *mut c_int) -> c_int;
type StrokeWidthFn = unsafe extern "system" fn(*mut c_void, *mut c_float) -> c_int;

/// 一次页面读取中复用的 Path 状态 ABI。
struct StateApi {
    subpaths: paths::Api,
    draw_mode: DrawModeFn,
    stroke_color: ColorFn,
    stroke_width: StrokeWidthFn,
    fill_color: ColorFn,
}

/// 保存原始子路径及其一次转换后的页面视觉点与直线。
struct PreparedSubpath {
    raw: paths::Subpath,
    points: Vec<(f64, f64)>,
    lines: Vec<((f64, f64), (f64, f64))>,
    closed: bool,
}

/// 保存一次读取后的绘制状态；颜色失败沿用参考实现的 255 alpha。
struct PathState {
    fill_visible: bool,
    stroke_visible: bool,
    fill_rgba: Option<Rgba>,
    raw_stroke_width: f64,
}

/// 保留 Python min 对相同值与 NaN 的首项选择。
fn minimum(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// 保留 Python max 对相同值与 NaN 的首项选择。
fn maximum(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// 按页面旋转角转换 PDF 底左坐标为页面左上坐标。
fn visual_point(point: (f64, f64), frame: [f64; 4], rotation: i32) -> (f64, f64) {
    let (x, y) = point;
    match rotation {
        90 => (y - frame[1], x - frame[0]),
        180 => (frame[2] - x, y - frame[1]),
        270 => (frame[3] - y, frame[2] - x),
        _ => (x - frame[0], frame[3] - y),
    }
}

/// 对象坐标先乘对象/Form 矩阵，再转换为页面视觉坐标。
fn transformed_point(
    point: (f64, f64),
    matrix: [f64; 6],
    frame: [f64; 4],
    rotation: i32,
) -> (f64, f64) {
    let x = matrix[0] * point.0 + matrix[2] * point.1 + matrix[4];
    let y = matrix[1] * point.0 + matrix[3] * point.1 + matrix[5];
    visual_point((x, y), frame, rotation)
}

/// 取页面视觉页尺寸；旋转 90/270 时交换宽高。
fn page_size(frame: [f64; 4], rotation: i32) -> (f64, f64) {
    let size = (frame[2] - frame[0], frame[3] - frame[1]);
    if rotation == 90 || rotation == 270 {
        (size.1, size.0)
    } else {
        size
    }
}

/// 读取一次 Path 绘制状态，颜色失败按不透明处理。
unsafe fn path_state(raw: *mut c_void, api: &StateApi) -> PathState {
    let (mut fill_mode, mut stroke) = (0, 0);
    if (api.draw_mode)(raw, &mut fill_mode, &mut stroke) == 0 {
        return PathState {
            fill_visible: false,
            stroke_visible: false,
            fill_rgba: None,
            raw_stroke_width: 0.0,
        };
    }
    let fill = (fill_mode != 0)
        .then(|| color(raw, api.fill_color))
        .flatten();
    let stroke_rgba = (stroke != 0)
        .then(|| color(raw, api.stroke_color))
        .flatten();
    // 查询失败只为可见性提供不透明 alpha；摘要仍保留颜色未知的 None。
    let fill_visible = fill_mode != 0 && fill.is_none_or(|rgba| rgba.3 > 0);
    let stroke_visible = stroke != 0 && stroke_rgba.is_none_or(|rgba| rgba.3 > 0);
    let mut width: c_float = 0.0;
    let raw_stroke_width = if stroke_visible && (api.stroke_width)(raw, &mut width) != 0 {
        let value = f64::from(width).abs();
        if value.is_finite() {
            value
        } else {
            0.0
        }
    } else {
        0.0
    };
    PathState {
        fill_visible,
        stroke_visible,
        fill_rgba: if fill_visible { fill } else { None },
        raw_stroke_width,
    }
}

/// 调用 PDFium 颜色函数；失败时保留未知颜色，由可见性判断单独回退 alpha。
unsafe fn color(raw: *mut c_void, getter: ColorFn) -> Option<Rgba> {
    let (mut red, mut green, mut blue, mut alpha) = (0, 0, 0, 255);
    if getter(raw, &mut red, &mut green, &mut blue, &mut alpha) == 0 {
        return None;
    }
    Some((red, green, blue, alpha))
}

/// 一次转换整个子路径，避免绘图线和路径摘要重复矩阵乘法。
fn prepare_subpath(
    raw: paths::Subpath,
    matrix: [f64; 6],
    frame: [f64; 4],
    rotation: i32,
) -> PreparedSubpath {
    let points = raw
        .points
        .iter()
        .map(|&point| transformed_point(point, matrix, frame, rotation))
        .collect();
    let lines = raw
        .lines
        .iter()
        .map(|&(first, last)| {
            (
                transformed_point(raw.points[first], matrix, frame, rotation),
                transformed_point(raw.points[last], matrix, frame, rotation),
            )
        })
        .collect();
    let closed = raw.closed;
    PreparedSubpath {
        raw,
        points,
        lines,
        closed,
    }
}

/// 按局部线段法向量换算矩阵后的实际描边宽度。
fn segment_stroke_width_with<H>(
    raw_width: f64,
    start: (f64, f64),
    end: (f64, f64),
    matrix: [f64; 6],
    hypot: &H,
) -> f64
where
    H: Fn(f64, f64) -> f64,
{
    let delta_x = end.0 - start.0;
    let delta_y = end.1 - start.1;
    let length = hypot(delta_x, delta_y);
    let scale = if length <= 0.0 {
        ((matrix[0] * matrix[3] - matrix[1] * matrix[2]).abs()).sqrt()
    } else {
        let normal_x = -delta_y / length;
        let normal_y = delta_x / length;
        let x = matrix[0] * normal_x + matrix[2] * normal_y;
        let y = matrix[1] * normal_x + matrix[3] * normal_y;
        hypot(x, y)
    };
    let width = raw_width * scale;
    if width.is_finite() {
        width
    } else {
        0.0
    }
}

/// 将近水平或近竖直线段吸附为页面内绘图线。
fn axis_line(start: (f64, f64), end: (f64, f64), width: f64, page: (f64, f64)) -> Option<Line> {
    let (x0, y0) = start;
    let (x1, y1) = end;
    if ![x0, y0, x1, y1, width]
        .iter()
        .all(|value| value.is_finite())
    {
        return None;
    }
    let delta_x = (x1 - x0).abs();
    let delta_y = (y1 - y0).abs();
    if delta_x >= delta_y && delta_y <= 1.0_f64.max(delta_x * 0.02) {
        let coordinate = (y0 + y1) / 2.0;
        if !(0.0..=page.1).contains(&coordinate) {
            return None;
        }
        let main_start = 0.0_f64.max(x0.min(x1));
        let main_end = page.0.min(x0.max(x1));
        if main_end - main_start < 1.0 {
            return None;
        }
        let line_width = 0.0_f64.max(width).max(delta_y);
        let half_width = line_width / 2.0;
        return Some((
            (main_start, coordinate),
            (main_end, coordinate),
            (
                main_start,
                0.0_f64.max(coordinate - half_width),
                main_end,
                page.1.min(coordinate + half_width),
            ),
            line_width,
            0,
        ));
    }
    if delta_y > delta_x && delta_x <= 1.0_f64.max(delta_y * 0.02) {
        let coordinate = (x0 + x1) / 2.0;
        if !(0.0..=page.0).contains(&coordinate) {
            return None;
        }
        let main_start = 0.0_f64.max(y0.min(y1));
        let main_end = page.1.min(y0.max(y1));
        if main_end - main_start < 1.0 {
            return None;
        }
        let line_width = 0.0_f64.max(width).max(delta_x);
        let half_width = line_width / 2.0;
        return Some((
            (coordinate, main_start),
            (coordinate, main_end),
            (
                0.0_f64.max(coordinate - half_width),
                main_start,
                page.0.min(coordinate + half_width),
                main_end,
            ),
            line_width,
            1,
        ));
    }
    None
}

/// 把闭合或参考实现认可的开放细长填充子路径折叠为中心线。
fn thin_filled_line<R: Fn(f64) -> f64>(
    subpath: &PreparedSubpath,
    page: (f64, f64),
    round3: &R,
) -> Option<Line> {
    if subpath.points.len() < 4 {
        return None;
    }
    let mut x0 = subpath.points[0].0;
    let mut x1 = subpath.points[0].0;
    let mut y0 = subpath.points[0].1;
    let mut y1 = subpath.points[0].1;
    for &(x, y) in &subpath.points[1..] {
        x0 = minimum(x0, x);
        x1 = maximum(x1, x);
        y0 = minimum(y0, y);
        y1 = maximum(y1, y);
    }
    let (width, height) = (x1 - x0, y1 - y0);
    if !subpath.closed && !open_thin_rectangle(subpath, x0, x1, y0, y1, round3) {
        return None;
    }
    let long_side = width.max(height);
    let short_side = width.min(height);
    if long_side < 1.0 || short_side > 2.0 || long_side < 4.0 * short_side.max(0.01) {
        return None;
    }
    if width >= height {
        axis_line(
            (x0, (y0 + y1) / 2.0),
            (x1, (y0 + y1) / 2.0),
            short_side,
            page,
        )
    } else {
        axis_line(
            ((x0 + x1) / 2.0, y0),
            ((x0 + x1) / 2.0, y1),
            short_side,
            page,
        )
    }
}

/// 仅接纳参考实现允许的开放四点三边轴对齐矩形。
fn open_thin_rectangle<R: Fn(f64) -> f64>(
    subpath: &PreparedSubpath,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    round3: &R,
) -> bool {
    if subpath.points.len() != 4 || subpath.raw.lines.len() != 3 {
        return false;
    }
    let rounded = subpath
        .points
        .iter()
        .map(|&(x, y)| (round3(x), round3(y)))
        .collect::<Vec<_>>();
    let expected = [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
        .into_iter()
        .map(|(x, y)| (round3(x), round3(y)))
        .collect::<Vec<_>>();
    // 与参考实现的集合相等语义保持一致：四个角点必须全部出现，缺失角点的退化开放路径不能折叠。
    if expected.iter().any(|point| !rounded.contains(point))
        || rounded.iter().any(|point| !expected.contains(point))
    {
        return false;
    }
    // 轴对齐判断必须发生在对象/Form 矩阵与页面坐标转换之后，与 Python 参考实现一致。
    subpath
        .lines
        .iter()
        .all(|&(a, b)| !((a.0 - b.0).abs() > 0.001 && (a.1 - b.1).abs() > 0.001))
}

/// 生成单个 Path 的绘图线候选，不应用对象裁剪。
fn object_lines_with<H, R>(
    prepared: &[PreparedSubpath],
    state: &PathState,
    matrix: [f64; 6],
    page: (f64, f64),
    hypot: &H,
    round3: &R,
) -> Vec<Line>
where
    H: Fn(f64, f64) -> f64,
    R: Fn(f64) -> f64,
{
    let mut output = Vec::new();
    if !state.fill_visible && !state.stroke_visible {
        return output;
    }
    for subpath in prepared {
        if state.fill_visible {
            if let Some(line) = thin_filled_line(subpath, page, round3) {
                output.push(line);
                continue;
            }
        }
        if !state.stroke_visible {
            continue;
        }
        for (index, (start, end)) in subpath.lines.iter().enumerate() {
            let raw_start = subpath.raw.points[subpath.raw.lines[index].0];
            let raw_end = subpath.raw.points[subpath.raw.lines[index].1];
            let width = segment_stroke_width_with(
                state.raw_stroke_width,
                raw_start,
                raw_end,
                matrix,
                hypot,
            );
            if let Some(line) = axis_line(*start, *end, width, page) {
                output.push(line);
            }
        }
    }
    output
}

/// 生成单个 Path 的摘要候选；无效几何返回 None。
#[allow(clippy::too_many_arguments)]
unsafe fn object_info_with<H, R>(
    raw: *mut c_void,
    api: &StateApi,
    prepared: &[PreparedSubpath],
    state: &PathState,
    matrix: [f64; 6],
    frame: [f64; 4],
    rotation: i32,
    depth: usize,
    source_index: usize,
    hypot: &H,
    round3: &R,
) -> Option<PathInfo>
where
    H: Fn(f64, f64) -> f64,
    R: Fn(f64) -> f64,
{
    let segment_count = (api.subpaths.count)(raw);
    if segment_count <= 0 || prepared.is_empty() {
        return None;
    }
    let mut points = Vec::new();
    for subpath in prepared {
        points.extend_from_slice(&subpath.points);
    }
    if points
        .iter()
        .any(|&(x, y)| !x.is_finite() || !y.is_finite())
    {
        return None;
    }
    let page = page_size(frame, rotation);
    let mut stroke_margin = 0.0;
    if state.stroke_visible {
        let x = hypot(matrix[0], matrix[1]);
        let y = hypot(matrix[2], matrix[3]);
        stroke_margin = 0.5 * state.raw_stroke_width * maximum(x, y);
    }
    let mut point = points[0];
    let mut bbox = [point.0, point.1, point.0, point.1];
    for current in &points[1..] {
        point = *current;
        bbox[0] = minimum(bbox[0], point.0);
        bbox[1] = minimum(bbox[1], point.1);
        bbox[2] = maximum(bbox[2], point.0);
        bbox[3] = maximum(bbox[3], point.1);
    }
    let bbox = (
        0.0_f64.max(bbox[0] - stroke_margin),
        0.0_f64.max(bbox[1] - stroke_margin),
        page.0.min(bbox[2] + stroke_margin),
        page.1.min(bbox[3] + stroke_margin),
    );
    if bbox.2 > bbox.0 && bbox.3 > bbox.1 {
        Some((
            bbox,
            segment_count,
            state.fill_visible,
            state.stroke_visible,
            depth,
            source_index,
            state.fill_rgba,
            if state.fill_visible {
                prepared
                    .iter()
                    .filter_map(|path| filled_rectangle_bbox(path, round3))
                    .collect()
            } else {
                Vec::new()
            },
        ))
    } else {
        None
    }
}

/// 记录真实四角直边填充矩形，曲线控制点和复杂文字轮廓不能仅凭外框成为柱体。
fn filled_rectangle_bbox<R: Fn(f64) -> f64>(
    path: &PreparedSubpath,
    round3: &R,
) -> Option<(f64, f64, f64, f64)> {
    if !(4..=5).contains(&path.points.len()) || !(3..=4).contains(&path.lines.len()) {
        return None;
    }
    if path
        .points
        .iter()
        .any(|p| !p.0.is_finite() || !p.1.is_finite())
    {
        return None;
    }
    let x0 = path.points.iter().map(|p| p.0).reduce(minimum)?;
    let x1 = path.points.iter().map(|p| p.0).reduce(maximum)?;
    let y0 = path.points.iter().map(|p| p.1).reduce(minimum)?;
    let y1 = path.points.iter().map(|p| p.1).reduce(maximum)?;
    if x1 <= x0
        || y1 <= y0
        || path
            .lines
            .iter()
            .any(|(a, b)| (a.0 - b.0).abs() > 0.001 && (a.1 - b.1).abs() > 0.001)
    {
        return None;
    }
    let mut corners = Vec::new();
    for p in &path.points {
        let point = (round3(p.0), round3(p.1));
        if !corners.contains(&point) {
            corners.push(point);
        }
    }
    if corners.len() != 4
        || [x0, x1].iter().any(|x| {
            [y0, y1]
                .iter()
                .any(|y| !corners.contains(&(round3(*x), round3(*y))))
        })
    {
        return None;
    }
    Some((x0, y0, x1, y1))
}

/// 将父坐标裁剪转换为视觉外框并与对象边界求交。
fn clipped_bbox(
    bbox: (f64, f64, f64, f64),
    clip: Option<[f64; 4]>,
    frame: [f64; 4],
    rotation: i32,
) -> Option<(f64, f64, f64, f64)> {
    let clip = match clip {
        Some(value) => value,
        // 无裁剪时直接保留原 bbox，不能用 Option None 误判为空交集。
        None => return Some(bbox),
    };
    if clip[2] <= clip[0] || clip[3] <= clip[1] {
        return None;
    }
    let corners = [
        visual_point((clip[0], clip[1]), frame, rotation),
        visual_point((clip[0], clip[3]), frame, rotation),
        visual_point((clip[2], clip[1]), frame, rotation),
        visual_point((clip[2], clip[3]), frame, rotation),
    ];
    if corners
        .iter()
        .any(|&(x, y)| !x.is_finite() || !y.is_finite())
    {
        return None;
    }
    let xs = [corners[0].0, corners[1].0, corners[2].0, corners[3].0];
    let ys = [corners[0].1, corners[1].1, corners[2].1, corners[3].1];
    let visual = (
        xs.iter().copied().reduce(minimum).unwrap(),
        ys.iter().copied().reduce(minimum).unwrap(),
        xs.iter().copied().reduce(maximum).unwrap(),
        ys.iter().copied().reduce(maximum).unwrap(),
    );
    let result = (
        maximum(bbox.0, visual.0),
        maximum(bbox.1, visual.1),
        minimum(bbox.2, visual.2),
        minimum(bbox.3, visual.3),
    );
    (result.2 > result.0 && result.3 > result.1).then_some(result)
}

/// 裁剪一条已吸附的轴线，并按方向重建端点。
fn clipped_line(
    line: Line,
    clip: Option<[f64; 4]>,
    frame: [f64; 4],
    rotation: i32,
) -> Option<Line> {
    let clipped = clipped_bbox(line.2, clip, frame, rotation)?;
    if clipped == line.2 {
        return Some(line);
    }
    let (start, end) = if line.4 == 0 {
        let y = minimum(clipped.3, maximum(clipped.1, line.0 .1));
        ((clipped.0, y), (clipped.2, y))
    } else {
        let x = minimum(clipped.2, maximum(clipped.0, line.0 .0));
        ((x, clipped.1), (x, clipped.3))
    };
    Some((start, end, clipped, line.3, line.4))
}

/// 单次遍历全部 Path，并允许绑定层传入宿主解释器的 hypot 语义。
///
/// # Safety
/// 调用方须验证全部 20 个函数 ABI，并持有同一 PDFium 页面、运行库和全局锁。
#[allow(clippy::too_many_arguments)]
pub unsafe fn read_path_evidence_with_hypot<H: Fn(f64, f64) -> f64, R: Fn(f64) -> f64>(
    addresses: Vec<usize>,
    handle: usize,
    frame: [f64; 4],
    rotation: i32,
    max_depth: usize,
    want_lines: bool,
    want_infos: bool,
    hypot: H,
    round3: R,
) -> Result<Evidence, ReadError> {
    if addresses.len() != 20
        || addresses.contains(&0)
        || handle == 0
        || max_depth > 64
        || ![0, 90, 180, 270].contains(&rotation)
    {
        return Err(ReadError::InvalidInput(
            "invalid PDFium path evidence arguments",
        ));
    }
    if !frame.iter().all(|value| value.is_finite()) || frame[2] <= frame[0] || frame[3] <= frame[1]
    {
        return Err(ReadError::InvalidInput(
            "invalid PDFium path evidence geometry",
        ));
    }
    let objects = objects::read_objects(addresses[..11].to_vec(), handle, 2, max_depth)?;
    let state_api = StateApi {
        subpaths: paths::Api {
            count: std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void) -> c_int>(
                addresses[11],
            ),
            get: std::mem::transmute::<
                usize,
                unsafe extern "system" fn(*mut c_void, c_int) -> *mut c_void,
            >(addresses[12]),
            point: std::mem::transmute::<
                usize,
                unsafe extern "system" fn(*mut c_void, *mut c_float, *mut c_float) -> c_int,
            >(addresses[13]),
            kind: std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void) -> c_int>(
                addresses[14],
            ),
            closes: std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void) -> c_int>(
                addresses[15],
            ),
        },
        draw_mode: std::mem::transmute::<usize, DrawModeFn>(addresses[16]),
        stroke_color: std::mem::transmute::<usize, ColorFn>(addresses[17]),
        stroke_width: std::mem::transmute::<usize, StrokeWidthFn>(addresses[18]),
        fill_color: std::mem::transmute::<usize, ColorFn>(addresses[19]),
    };
    let page = page_size(frame, rotation);
    let mut lines = Vec::new();
    let mut infos = Vec::new();
    for (source_index, (address, matrix, _parent, depth, clip)) in objects.into_iter().enumerate() {
        let raw = address as *mut c_void;
        let raw_subpaths = paths::read_subpaths_with(&state_api.subpaths, address)?;
        let state = path_state(raw, &state_api);
        let prepared: Vec<_> = raw_subpaths
            .into_iter()
            .map(|subpath| prepare_subpath(subpath, matrix, frame, rotation))
            .collect();
        if want_lines {
            for line in object_lines_with(&prepared, &state, matrix, page, &hypot, &round3) {
                if let Some(line) = clipped_line(line, clip, frame, rotation) {
                    lines.push(line);
                }
            }
        }
        if want_infos {
            if let Some(info) = object_info_with(
                raw,
                &state_api,
                &prepared,
                &state,
                matrix,
                frame,
                rotation,
                depth,
                source_index,
                &hypot,
                &round3,
            ) {
                if let Some(bbox) = clipped_bbox(info.0, clip, frame, rotation) {
                    let rectangles = info
                        .7
                        .into_iter()
                        .filter_map(|rect| clipped_bbox(rect, clip, frame, rotation))
                        .collect();
                    infos.push((
                        bbox, info.1, info.2, info.3, info.4, info.5, info.6, rectangles,
                    ));
                }
            }
        }
    }
    Ok((lines, infos))
}
