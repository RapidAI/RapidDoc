//! 显式、保守的页面块阅读顺序；使用投影留白和通栏屏障，不合并或丢弃区域。
use rapiddoc_core::{Error, ErrorCode, Rect};
use serde::Serialize;

/// 单页块排序的累计工作量硬上限；允许调用方设置更低预算。
pub const MAX_WORK: usize = 2_000_000;

/// 输入始终是页面像素坐标；不支持的方向使整页保留原顺序。
pub struct Region {
    pub bounds: Rect,
    pub horizontal: bool,
}

/// indices 指向原 blocks；拒绝歧义时返回全部原索引，资源不足则返回错误。
#[derive(Serialize)]
pub struct Plan {
    pub indices: Vec<usize>,
    pub reason: &'static str,
    pub work_used: usize,
}

fn charge(work: &mut usize, amount: usize, limit: usize) -> Result<(), Error> {
    let limit = rapiddoc_core::resources::limit(limit);
    *work += amount;
    if *work > limit {
        return Err(Error {
            code: ErrorCode::ResourceLimit,
            message: "页面块阅读顺序工作量超过预算".into(),
        });
    }
    Ok(())
}

/// 区间扫描只在大于最小留白的间隙切开；不逐像素构建投影数组。
fn groups(indices: &[usize], boxes: &[[f64; 4]], axis: usize, gap: f64) -> Vec<Vec<usize>> {
    let mut sorted = indices.to_vec();
    sorted.sort_by(|&a, &b| boxes[a][axis].total_cmp(&boxes[b][axis]).then(a.cmp(&b)));
    let mut output: Vec<Vec<usize>> = Vec::new();
    let mut end = f64::NEG_INFINITY;
    for i in sorted {
        if boxes[i][axis] > end + gap {
            output.push(Vec::new());
        }
        if let Some(group) = output.last_mut() {
            group.push(i);
        }
        end = end.max(boxes[i][axis + 2]);
    }
    output
}

/// 优先稳定并排栏；通栏头尾从中间栏区剥离，避免把整齐双栏按行交错。
fn partition(indices: &[usize], boxes: &[[f64; 4]], gap: f64) -> Option<Vec<Vec<usize>>> {
    let x = groups(indices, boxes, 0, gap);
    if x.len() > 1 {
        let spans: Vec<_> = x
            .iter()
            .map(|g| {
                g.iter().fold([f64::INFINITY, f64::NEG_INFINITY], |b, &i| {
                    [b[0].min(boxes[i][1]), b[1].max(boxes[i][3])]
                })
            })
            .collect();
        if spans
            .windows(2)
            .all(|p| p[0][1].min(p[1][1]) - p[0][0].max(p[1][0]) > gap)
        {
            return Some(x);
        }
    }
    let y = groups(indices, boxes, 1, gap);
    if y.len() < 2 {
        return None;
    }
    let left = indices
        .iter()
        .map(|&i| boxes[i][0])
        .fold(f64::INFINITY, f64::min);
    let right = indices
        .iter()
        .map(|&i| boxes[i][2])
        .fold(f64::NEG_INFINITY, f64::max);
    let wide = |group: &[usize]| {
        group
            .iter()
            .any(|&i| boxes[i][2] - boxes[i][0] >= 0.8 * (right - left))
    };
    let mut first = 0;
    let mut last = y.len();
    while first < last && wide(&y[first]) {
        first += 1;
    }
    while first < last && wide(&y[last - 1]) {
        last -= 1;
    }
    if first == 0 && last == y.len() {
        return y.iter().all(|g| g.len() == 1).then_some(y);
    }
    let mut result = y[..first].to_vec();
    if first < last {
        result.push(y[first..last].iter().flatten().copied().collect());
    }
    result.extend_from_slice(&y[last..]);
    Some(result)
}

/// 在最多 1000 个区域上迭代切分；坐标/方向/重叠/切分歧义整页保留原序。
/// 留白为页面短边的 0.5%，通栏为当前区域总宽的 80%；两者随页面比例变化。
/// 预算包括重叠检查及排序成本估计，超限报错，不伪装成普通保留。
pub fn plan(regions: &[Region], width: usize, height: usize, limit: usize) -> Result<Plan, Error> {
    let limit = rapiddoc_core::resources::limit(limit);
    if width == 0 || height == 0 || limit == 0 || limit > rapiddoc_core::resources::limit(MAX_WORK)
    {
        return Err(Error {
            code: ErrorCode::InvalidInput,
            message: "页面块排序尺寸或预算无效".into(),
        });
    }
    if regions.len() > rapiddoc_core::resources::limit(1000) {
        return Err(Error {
            code: ErrorCode::ResourceLimit,
            message: "页面块排序最多支持 1000 个区域".into(),
        });
    }
    let original: Vec<_> = (0..regions.len()).collect();
    let mut work = 0;
    let preserve = |reason, work_used| Plan {
        indices: original.clone(),
        reason,
        work_used,
    };
    let boxes: Vec<[f64; 4]> = regions.iter().map(|r| r.bounds.into()).collect();
    for (i, r) in regions.iter().enumerate() {
        charge(&mut work, 1, limit)?;
        let b = boxes[i];
        if b[0] < 0.
            || b[1] < 0.
            || b[2] > width as f64
            || b[3] > height as f64
            || b[2] <= b[0]
            || b[3] <= b[1]
        {
            return Ok(preserve("invalid_region_geometry", work));
        }
        if !r.horizontal {
            return Ok(preserve("unsupported_direction", work));
        }
    }
    for a in 0..boxes.len() {
        for b in a + 1..boxes.len() {
            charge(&mut work, 1, limit)?;
            if boxes[a][0].max(boxes[b][0]) < boxes[a][2].min(boxes[b][2])
                && boxes[a][1].max(boxes[b][1]) < boxes[a][3].min(boxes[b][3])
            {
                return Ok(preserve("overlapping_regions", work));
            }
        }
    }
    let gap = width.min(height) as f64 * 0.005;
    let mut stack = vec![original.clone()];
    let mut indices = Vec::with_capacity(regions.len());
    while let Some(group) = stack.pop() {
        if group.len() <= 1 {
            indices.extend(group);
            continue;
        }
        charge(
            &mut work,
            group.len() * (group.len().ilog2() as usize + 2) * 2,
            limit,
        )?;
        let Some(parts) = partition(&group, &boxes, gap) else {
            return Ok(preserve("ambiguous_geometry", work));
        };
        for part in parts.into_iter().rev() {
            stack.push(part);
        }
    }
    let reason = if indices == original {
        "already_ordered"
    } else {
        "geometry"
    };
    Ok(Plan {
        indices,
        reason,
        work_used: work,
    })
}
