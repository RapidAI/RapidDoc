//! 同库对象树与裁剪遍历；借用地址只允许在原页面作用域内消费。
use crate::ReadError;
use std::ffi::{c_float, c_int, c_uint, c_ulong, c_void};
pub type Matrix = [f64; 6];
pub type Bounds = [f64; 4];
pub type Object = (usize, Matrix, Matrix, usize, Option<Bounds>);
/// 已核验 ABI 的文字外框 getter，地址只在同步页面读取中借用。
type BoundsGetter = unsafe extern "system" fn(
    *mut c_void,
    *mut c_float,
    *mut c_float,
    *mut c_float,
    *mut c_float,
) -> c_int;
/// 颜色读取与既有 ABI 完全相同，不在页面作用域之外保存函数或对象地址。
type ColorGetter = unsafe extern "system" fn(
    *mut c_void,
    *mut c_uint,
    *mut c_uint,
    *mut c_uint,
    *mut c_uint,
) -> c_int;
/// 顶层覆盖的必要条件；完整矩形、裁剪和资源安全裁决仍沿用 Python。
struct CoverFilter {
    count: unsafe extern "system" fn(*mut c_void) -> c_int,
    draw_mode: unsafe extern "system" fn(*mut c_void, *mut c_int, *mut c_int) -> c_int,
    color: ColorGetter,
}
impl CoverFilter {
    /// 只有四至五段的实心不透明路径才可能完全覆盖文字；失败读取保持保守拒绝。
    unsafe fn eligible(&self, raw: *mut c_void) -> bool {
        if !(4..=5).contains(&(self.count)(raw)) {
            return false;
        }
        let (mut fill, mut stroke) = (0, 0);
        if (self.draw_mode)(raw, &mut fill, &mut stroke) == 0 || fill == 0 {
            return false;
        }
        let (mut r, mut g, mut b, mut alpha) = (0, 0, 0, 0);
        (self.color)(raw, &mut r, &mut g, &mut b, &mut alpha) != 0 && alpha == 255
    }
}
#[repr(C)]
#[derive(Default)]
struct RawMatrix {
    values: [c_float; 6],
}
struct Api {
    cover_filter: Option<CoverFilter>,
    page_count: unsafe extern "system" fn(*mut c_void) -> c_int,
    page_get: unsafe extern "system" fn(*mut c_void, c_int) -> *mut c_void,
    form_count: unsafe extern "system" fn(*mut c_void) -> c_int,
    form_get: unsafe extern "system" fn(*mut c_void, c_ulong) -> *mut c_void,
    matrix: unsafe extern "system" fn(*mut c_void, *mut RawMatrix) -> c_int,
    kind: unsafe extern "system" fn(*mut c_void) -> c_int,
    clip: unsafe extern "system" fn(*mut c_void) -> *mut c_void,
    paths: unsafe extern "system" fn(*mut c_void) -> c_int,
    segments: unsafe extern "system" fn(*mut c_void, c_int) -> c_int,
    segment: unsafe extern "system" fn(*mut c_void, c_int, c_int) -> *mut c_void,
    point: unsafe extern "system" fn(*mut c_void, *mut c_float, *mut c_float) -> c_int,
}
/// 保留 Python min 对相同值与有符号零的首项选择。
fn minimum(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}
/// 保留 Python max 对相同值与有符号零的首项选择。
fn maximum(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}
/// 使用与参考实现相同的运算顺序合成对象和父级矩阵。
fn multiply(a: Matrix, b: Matrix) -> Matrix {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}
impl Api {
    /// 裁剪路径在父坐标中变换；损坏的局部路径不抹掉继承裁剪。
    unsafe fn clip_bounds(
        &self,
        raw: *mut c_void,
        parent: Matrix,
        inherited: Option<Bounds>,
    ) -> Option<Bounds> {
        let clip = (self.clip)(raw);
        if clip.is_null() {
            return inherited;
        }
        let mut result = inherited;
        for path in 0..(self.paths)(clip).max(0) {
            let mut bounds: Option<Bounds> = None;
            let mut finite = true;
            for index in 0..(self.segments)(clip, path).max(0) {
                let segment = (self.segment)(clip, path, index);
                let (mut x, mut y) = (0.0, 0.0);
                if (self.point)(segment, &mut x, &mut y) != 0 {
                    let px = parent[0] * x as f64 + parent[2] * y as f64 + parent[4];
                    let py = parent[1] * x as f64 + parent[3] * y as f64 + parent[5];
                    finite &= px.is_finite() && py.is_finite();
                    bounds = Some(match bounds {
                        None => [px, py, px, py],
                        Some(b) => [
                            minimum(b[0], px),
                            minimum(b[1], py),
                            maximum(b[2], px),
                            maximum(b[3], py),
                        ],
                    });
                }
            }
            if finite {
                if let Some(b) = bounds {
                    result = Some(match result {
                        None => b,
                        Some(old) => [
                            maximum(b[0], old[0]),
                            maximum(b[1], old[1]),
                            minimum(b[2], old[2]),
                            minimum(b[3], old[3]),
                        ],
                    });
                }
            }
        }
        result
    }
    /// 按原深度优先顺序遍历叶子，在 Rust 内过滤类型以减少 Python 临时对象。
    #[allow(clippy::too_many_arguments)]
    unsafe fn walk(
        &self,
        container: *mut c_void,
        form: bool,
        parent: Matrix,
        depth: usize,
        inherited: Option<Bounds>,
        max_depth: usize,
        kind: c_int,
        output: &mut Vec<OrderedObject>,
        mut roots: Option<&mut Vec<OrderedObject>>,
        order: &mut usize,
    ) {
        if depth >= max_depth {
            return;
        }
        let count = if form {
            (self.form_count)(container)
        } else {
            (self.page_count)(container)
        };
        for index in 0..count.max(0) {
            let raw = if form {
                (self.form_get)(container, index as c_ulong)
            } else {
                (self.page_get)(container, index)
            };
            if raw.is_null() {
                continue;
            }
            let mut matrix = RawMatrix::default();
            if (self.matrix)(raw, &mut matrix) == 0 {
                continue;
            }
            let combined = multiply(matrix.values.map(f64::from), parent);
            let object_kind = (self.kind)(raw);
            // 类型先于裁剪判断：类型不匹配的叶子（如纯路径页上的 Path）不必读 clip。
            if object_kind == 5 {
                let clip = self.clip_bounds(raw, parent, inherited);
                self.walk(
                    raw,
                    true,
                    combined,
                    depth + 1,
                    clip,
                    max_depth,
                    kind,
                    output,
                    roots.as_deref_mut(),
                    order,
                );
            } else if kind == -1 || object_kind == kind {
                let clip = self.clip_bounds(raw, parent, inherited);
                output.push((*order, (raw as usize, combined, parent, depth, clip)));
            } else if depth == 0
                && object_kind == 2
                // 只在启用遮挡预筛选时提前排除先于全部 TEXT 的路径；旧完整证据模式仍返回原集合。
                && (self.cover_filter.is_none() || !output.is_empty())
                && self
                    .cover_filter
                    .as_ref()
                    .is_none_or(|filter| filter.eligible(raw))
            {
                if let Some(paths) = roots.as_deref_mut() {
                    paths.push((
                        *order,
                        (
                            raw as usize,
                            combined,
                            parent,
                            depth,
                            self.clip_bounds(raw, parent, inherited),
                        ),
                    ));
                }
            }
            if object_kind != 5 {
                *order += 1;
            }
        }
    }
}
/// TEXT 对象地址、最终可见性和页面视觉坐标中的有效裁剪框。
pub type TextVisibility = (usize, bool, Option<Bounds>);
/// 稳定绘制序号及借用对象几何，仅在页面锁的生命周期内消费。
pub type OrderedObject = (usize, Object);
/// 可见文字的绘制序号、对象地址及视觉外框。
pub type TextPaint = (usize, usize, Bounds);
/// 可见性和遮挡证据的单次遍历结果。
pub type PaintVisibility = (Vec<TextVisibility>, Vec<OrderedObject>, Vec<TextPaint>);

/// 按 Python 页面坐标变换裁剪框，四个角点独立取保守外框。
fn visual_bounds(bounds: Bounds, frame: [f64; 4], rotation: i32) -> Bounds {
    let point = |x: f64, y: f64| match rotation {
        90 => (y - frame[1], x - frame[0]),
        180 => (frame[2] - x, y - frame[1]),
        270 => (frame[3] - y, frame[2] - x),
        _ => (x - frame[0], frame[3] - y),
    };
    let points = [
        point(bounds[0], bounds[1]),
        point(bounds[0], bounds[3]),
        point(bounds[2], bounds[1]),
        point(bounds[2], bounds[3]),
    ];
    let xs = [points[0].0, points[1].0, points[2].0, points[3].0];
    let ys = [points[0].1, points[1].1, points[2].1, points[3].1];
    [
        xs.iter().copied().reduce(minimum).unwrap(),
        ys.iter().copied().reduce(minimum).unwrap(),
        xs.iter().copied().reduce(maximum).unwrap(),
        ys.iter().copied().reduce(maximum).unwrap(),
    ]
}

/// 一次 TEXT 对象遍历同时计算绘制状态和有效裁剪；不跨调用保存 PDFium 地址。
///
/// # Safety
/// 调用方必须验证 14 个函数的 ABI，并持有同一 PDFium 页面、运行库和全局锁。
pub unsafe fn read_text_visibility(
    addresses: Vec<usize>,
    handle: usize,
    frame: [f64; 4],
    rotation: i32,
    max_depth: usize,
) -> Result<Vec<TextVisibility>, ReadError> {
    read_paint_visibility(addresses, handle, frame, rotation, max_depth, false).map(|v| v.0)
}

/// 同次对象树读取返回文字状态和顶层 Path，避免无覆盖证据时再次遍历全部文字。
///
/// # Safety
/// 调用方必须核验标准 ABI，并使页面、函数库和全局锁在本次读取中保持存活。
pub unsafe fn read_text_visibility_with_roots(
    addresses: Vec<usize>,
    handle: usize,
    frame: [f64; 4],
    rotation: i32,
    max_depth: usize,
) -> Result<PaintVisibility, ReadError> {
    read_paint_visibility(addresses, handle, frame, rotation, max_depth, true)
}

/// 同步执行既有可见性规则，额外几何仅在遮挡调用方需要时收集。
unsafe fn read_paint_visibility(
    addresses: Vec<usize>,
    handle: usize,
    frame: [f64; 4],
    rotation: i32,
    max_depth: usize,
    collect_roots: bool,
) -> Result<PaintVisibility, ReadError> {
    if !(if collect_roots {
        [15, 17].contains(&addresses.len())
    } else {
        addresses.len() == 14
    }) || addresses.contains(&0)
        || handle == 0
    {
        return Err(ReadError::InvalidInput(
            "invalid PDFium text visibility arguments",
        ));
    }
    if !frame.iter().all(|value| value.is_finite())
        || frame[2] <= frame[0]
        || frame[3] <= frame[1]
        || ![0, 90, 180, 270].contains(&rotation)
    {
        return Err(ReadError::InvalidInput(
            "invalid PDFium text visibility geometry",
        ));
    }
    let filter = if collect_roots && addresses.len() == 17 {
        Some(CoverFilter {
            count: std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void) -> c_int>(
                addresses[15],
            ),
            draw_mode: std::mem::transmute::<
                usize,
                unsafe extern "system" fn(*mut c_void, *mut c_int, *mut c_int) -> c_int,
            >(addresses[16]),
            color: std::mem::transmute::<usize, ColorGetter>(addresses[12]),
        })
    } else {
        None
    };
    let filtered = filter.is_some();
    let (objects, mut roots) = read_objects_and_roots(
        addresses[..11].to_vec(),
        handle,
        1,
        max_depth,
        collect_roots,
        filter,
        None,
    )?;
    if filtered {
        // 早于全部文字绘制的路径不能覆盖任何文字；保持后绘制候选的完整顺序和覆盖。
        let earliest = objects.iter().map(|record| record.0).min();
        roots.retain(|record| earliest.is_some_and(|order| order < record.0));
    }
    let last_cover = roots.iter().map(|record| record.0).max();
    let render_mode: unsafe extern "system" fn(*mut c_void) -> c_int =
        std::mem::transmute(addresses[11]);
    let color: unsafe extern "system" fn(
        *mut c_void,
        *mut c_uint,
        *mut c_uint,
        *mut c_uint,
        *mut c_uint,
    ) -> c_int = std::mem::transmute(addresses[12]);
    let stroke_color: unsafe extern "system" fn(
        *mut c_void,
        *mut c_uint,
        *mut c_uint,
        *mut c_uint,
        *mut c_uint,
    ) -> c_int = std::mem::transmute(addresses[13]);
    let alpha = |raw: *mut c_void,
                 getter: unsafe extern "system" fn(
        *mut c_void,
        *mut c_uint,
        *mut c_uint,
        *mut c_uint,
        *mut c_uint,
    ) -> c_int|
     -> u32 {
        let (mut red, mut green, mut blue, mut alpha) = (0, 0, 0, 255);
        if getter(raw, &mut red, &mut green, &mut blue, &mut alpha) == 0 {
            return 255;
        }
        alpha
    };
    let bounds_getter: Option<
        unsafe extern "system" fn(
            *mut c_void,
            *mut c_float,
            *mut c_float,
            *mut c_float,
            *mut c_float,
        ) -> c_int,
    > = if collect_roots && (!filtered || last_cover.is_some()) {
        Some(std::mem::transmute::<usize, BoundsGetter>(addresses[14]))
    } else {
        None
    };
    let mut paint = Vec::new();
    Ok((
        objects
            .into_iter()
            .map(|(order, (address, _, parent, _, clip))| {
                let raw = address as *mut c_void;
                let mode = render_mode(raw);
                let mut visible = mode != 3 && mode != 7;
                if (0..=2).contains(&mode) || (4..=6).contains(&mode) {
                    visible = ((mode == 0 || mode == 2 || mode == 4 || mode == 6)
                        && alpha(raw, color) > 0)
                        || ((mode == 1 || mode == 2 || mode == 5 || mode == 6)
                            && alpha(raw, stroke_color) > 0);
                }
                let clip = clip.map(|value| {
                    if value[2] <= value[0] || value[3] <= value[1] {
                        visible = false;
                        value
                    } else {
                        visual_bounds(value, frame, rotation)
                    }
                });
                if visible && (!filtered || last_cover.is_some_and(|last| order < last)) {
                    if let Some(getter) = bounds_getter {
                        let mut values = [0.0f32; 4];
                        let ptr = values.as_mut_ptr();
                        if getter(raw, ptr, ptr.add(1), ptr.add(2), ptr.add(3)) != 0 {
                            let b = values.map(f64::from);
                            let points = [(b[0], b[1]), (b[0], b[3]), (b[2], b[1]), (b[2], b[3])]
                                .map(|(x, y)| {
                                    let px = parent[0] * x + parent[2] * y + parent[4];
                                    let py = parent[1] * x + parent[3] * y + parent[5];
                                    match rotation {
                                        90 => (py - frame[1], px - frame[0]),
                                        180 => (frame[2] - px, py - frame[1]),
                                        270 => (frame[3] - py, frame[2] - px),
                                        _ => (px - frame[0], frame[3] - py),
                                    }
                                });
                            paint.push((
                                order,
                                address,
                                [
                                    points.iter().map(|p| p.0).reduce(minimum).unwrap(),
                                    points.iter().map(|p| p.1).reduce(minimum).unwrap(),
                                    points.iter().map(|p| p.0).reduce(maximum).unwrap(),
                                    points.iter().map(|p| p.1).reduce(maximum).unwrap(),
                                ],
                            ));
                        }
                    }
                }
                (address, visible, clip)
            })
            .collect(),
        roots,
        paint,
    ))
}

/// 借用有效页面同步读取指定类型叶子，不跨调用缓存原生地址。
///
/// # Safety
///
/// 调用方必须验证函数 ABI，并持有同一 PDFium 运行库、页面和全局锁。
pub unsafe fn read_objects(
    addresses: Vec<usize>,
    handle: usize,
    kind: c_int,
    max_depth: usize,
) -> Result<Vec<Object>, ReadError> {
    read_objects_and_roots(addresses, handle, kind, max_depth, false, None, None)
        .map(|v| v.0.into_iter().map(|(_, object)| object).collect())
}

/// 从已借用的页面或 Form 根开始遍历，沿用统一裁剪和矩阵计算。
///
/// # Safety
///
/// 调用方须核验同库 ABI，并保证根句柄、父变换及原有页面锁在调用期间有效。
pub unsafe fn read_objects_from(
    addresses: Vec<usize>,
    handle: usize,
    kind: c_int,
    max_depth: usize,
    root: (bool, Matrix, usize, Option<Bounds>),
) -> Result<Vec<Object>, ReadError> {
    read_objects_and_roots(addresses, handle, kind, max_depth, false, None, Some(root))
        .map(|v| v.0.into_iter().map(|(_, object)| object).collect())
}

/// 沿相同遍历顺序收集指定叶子，可选顶层路径不跨页面作用域保存。
unsafe fn read_objects_and_roots(
    addresses: Vec<usize>,
    handle: usize,
    kind: c_int,
    max_depth: usize,
    collect_roots: bool,
    cover_filter: Option<CoverFilter>,
    root: Option<(bool, Matrix, usize, Option<Bounds>)>,
) -> Result<(Vec<OrderedObject>, Vec<OrderedObject>), ReadError> {
    if addresses.len() != 11 || addresses.contains(&0) || handle == 0 || max_depth > 64 {
        return Err(ReadError::InvalidInput(
            "invalid PDFium object bridge arguments",
        ));
    }
    let api = Api {
        cover_filter,
        page_count: std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void) -> c_int>(
            addresses[0],
        ),
        page_get: std::mem::transmute::<
            usize,
            unsafe extern "system" fn(*mut c_void, c_int) -> *mut c_void,
        >(addresses[1]),
        form_count: std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void) -> c_int>(
            addresses[2],
        ),
        form_get: std::mem::transmute::<
            usize,
            unsafe extern "system" fn(*mut c_void, c_ulong) -> *mut c_void,
        >(addresses[3]),
        matrix: std::mem::transmute::<
            usize,
            unsafe extern "system" fn(*mut c_void, *mut RawMatrix) -> c_int,
        >(addresses[4]),
        kind: std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void) -> c_int>(
            addresses[5],
        ),
        clip: std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void) -> *mut c_void>(
            addresses[6],
        ),
        paths: std::mem::transmute::<usize, unsafe extern "system" fn(*mut c_void) -> c_int>(
            addresses[7],
        ),
        segments: std::mem::transmute::<
            usize,
            unsafe extern "system" fn(*mut c_void, c_int) -> c_int,
        >(addresses[8]),
        segment: std::mem::transmute::<
            usize,
            unsafe extern "system" fn(*mut c_void, c_int, c_int) -> *mut c_void,
        >(addresses[9]),
        point: std::mem::transmute::<
            usize,
            unsafe extern "system" fn(*mut c_void, *mut c_float, *mut c_float) -> c_int,
        >(addresses[10]),
    };
    let mut output = Vec::new();
    let mut roots = Vec::new();
    let (is_form, matrix, depth, clip) =
        root.unwrap_or((false, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0, None));
    api.walk(
        handle as *mut c_void,
        is_form,
        matrix,
        depth,
        clip,
        max_depth,
        kind,
        &mut output,
        if collect_roots {
            Some(&mut roots)
        } else {
            None
        },
        &mut 0,
    );
    Ok((output, roots))
}
