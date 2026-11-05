//! 自有表格候选的网格扩展、共享成员与动态合并；只在最终边界导出保留结果。
use crate::geometry::{rotate, Box4, Size};
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

/// 来源 ID 由解析器生成；用整数混合避免对大量行号执行通用字符串哈希。
#[derive(Default)]
pub struct IdHasher(u64);
impl Hasher for IdHasher {
    /// 返回当前混合值，集合仍通过整数相等判定处理哈希碰撞。
    fn finish(&self) -> u64 {
        self.0
    }
    /// 保留 Hasher 的字节入口，实际成员键走专用整数方法。
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = (self.0 ^ u64::from(*b)).wrapping_mul(0x100000001b3);
        }
    }
    /// 混合高低位，避免稀疏行号与负数集中到同一桶。
    fn write_u64(&mut self, value: u64) {
        let mut v = value.wrapping_add(0x9e3779b97f4a7c15);
        v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
        self.0 = v ^ (v >> 31);
    }
    /// 有符号 ID 按位无损转换，不把负来源号当作数组下标。
    fn write_i64(&mut self, value: i64) {
        self.write_u64(value as u64);
    }
}
pub type Ids = HashSet<i64, BuildHasherDefault<IdHasher>>;
pub type Row = (f64, Vec<(i64, Box4)>);

/// 有限普通坐标准入，限制极值以避免面积运算溢出。
pub fn valid(b: &Box4) -> bool {
    b.iter().all(|v| v.is_finite() && v.abs() <= 1e150)
}

/// 按首见坐标合并，保留相等值及负零。
fn union(mut a: Box4, b: Box4) -> Box4 {
    for i in 0..4 {
        if (i < 2 && b[i] < a[i]) || (i >= 2 && b[i] > a[i]) {
            a[i] = b[i];
        }
    }
    a
}

/// 复用原 Python 面积定义，仅网格面积将负宽高截为零。
fn area(b: Box4) -> f64 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

/// 保持较小框面积作为分母，退化框返回零。
fn overlap(a: Box4, b: Box4) -> f64 {
    let intersection =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let denominator = ((a[2] - a[0]) * (a[3] - a[1])).min((b[2] - b[0]) * (b[3] - b[1]));
    if denominator > 0.0 {
        intersection / denominator
    } else {
        0.0
    }
}

/// 从局部坐标回到页面坐标，保持减法顺序及非正交角的原样返回。
fn inverse(b: Box4, size: Size, angle: i32) -> Box4 {
    match angle {
        270 => [b[1], size[1] - b[2], b[3], size[1] - b[0]],
        90 => [size[0] - b[3], b[0], size[0] - b[1], b[2]],
        180 => [
            size[0] - b[2],
            size[1] - b[3],
            size[0] - b[0],
            size[1] - b[1],
        ],
        _ => b,
    }
}

#[derive(Clone)]
pub struct Members {
    base: Arc<Ids>,
    added: Ids,
    removed: Ids,
}
impl Members {
    /// 普通已构造候选没有共享基底，后续可精确转换到首个网格基底。
    pub fn plain(added: Ids) -> Self {
        Self {
            base: Arc::new(Ids::default()),
            added,
            removed: Ids::default(),
        }
    }
    /// 未扩展核心共享不可变成员，不在每个候选上复制完整集合。
    pub fn new(base: Arc<Ids>, excluded: Ids) -> Self {
        let removed = excluded.into_iter().filter(|v| base.contains(v)).collect();
        Self {
            base,
            added: Ids::default(),
            removed,
        }
    }
    /// 查询当前可见成员，包括共享基底上的删改差集。
    fn contains(&self, value: &i64) -> bool {
        self.added.contains(value) || (self.base.contains(value) && !self.removed.contains(value))
    }
    /// 仅在跨基底合并或最终输出时枚举实际成员。
    pub fn values(&self) -> impl Iterator<Item = i64> + '_ {
        self.base
            .iter()
            .filter(|v| !self.removed.contains(v))
            .chain(&self.added)
            .copied()
    }
    /// 网格扩展先并入基底，再排除注释，和共享 Python 集合的构造顺序相同。
    fn expand(&mut self, base: Arc<Ids>, excluded: &Ids) {
        let added = self.values().filter(|v| !base.contains(v)).collect();
        self.base = base;
        self.added = added;
        self.removed = Ids::default();
        for id in excluded {
            self.added.remove(id);
            if self.base.contains(id) {
                self.removed.insert(*id);
            }
        }
    }
    /// 优先统一共享基底，后续同网格候选只合并差集。
    fn merge(&mut self, other: &Self) {
        if !Arc::ptr_eq(&self.base, &other.base) && self.base.is_empty() && !other.base.is_empty() {
            let visible: Ids = self.values().collect();
            self.removed = other.base.difference(&visible).copied().collect();
            self.added = visible.difference(&other.base).copied().collect();
            self.base = other.base.clone();
        }
        if Arc::ptr_eq(&self.base, &other.base) {
            self.removed.retain(|v| other.removed.contains(v));
            for id in &other.added {
                if self.base.contains(id) {
                    self.removed.remove(id);
                } else {
                    self.added.insert(*id);
                }
            }
        } else {
            for id in other.values() {
                if self.base.contains(&id) {
                    self.removed.remove(&id);
                } else {
                    self.added.insert(id);
                }
            }
        }
    }
}

pub struct Annotation {
    pub kind: String,
    pub bbox: Box4,
    pub members: Ids,
    pub rows: Vec<(i64, Box4)>,
}
pub struct Candidate {
    pub bbox: Box4,
    pub local: Box4,
    pub core: Option<Box4>,
    pub angle: i32,
    pub score: f64,
    pub members: Members,
    pub annotations: Vec<Annotation>,
}

pub struct GridContext {
    grids: Vec<Box4>,
    rows: Vec<Row>,
    size: Size,
    angle: i32,
    tolerance: f64,
    cache: HashMap<[u64; 4], Arc<Ids>>,
}
impl GridContext {
    /// 页面只准备一次网格及片段中心数据；拒绝特殊数值而不改变候选覆盖。
    pub fn new(
        grids: Vec<Box4>,
        rows: Vec<Row>,
        size: Size,
        angle: i32,
        height: f64,
    ) -> Option<Self> {
        if grids.iter().any(|b| !valid(b))
            || rows
                .iter()
                .any(|r| !r.0.is_finite() || r.1.iter().any(|(_, b)| !valid(b)))
            || size.iter().any(|v| !v.is_finite() || v.abs() > 1e100)
            || !height.is_finite()
        {
            return None;
        }
        Some(Self {
            grids,
            rows,
            size,
            angle,
            tolerance: 2.0_f64.max(height),
            cache: HashMap::new(),
        })
    }
    /// 扩展后才参与候选合并；首见并列网格及页面旋转往返均保持原顺序。
    pub fn expand(&mut self, candidate: &mut Candidate) {
        let Some(core) = candidate.core else {
            return;
        };
        let core = rotate(core, self.size, self.angle);
        let mut selected = None;
        for grid in &self.grids {
            let shorter = (core[2] - core[0]).min(grid[2] - grid[0]);
            let share = if shorter > 0.0 {
                (core[2].min(grid[2]) - core[0].max(grid[0])).max(0.0) / shorter
            } else {
                0.0
            };
            if share < 0.9
                || core[3] < grid[1] - self.tolerance
                || core[1] > grid[3] + self.tolerance
            {
                continue;
            }
            let score = (core[3].min(grid[3]) - core[1].max(grid[1]), area(*grid));
            if selected
                .as_ref()
                .is_none_or(|(previous, _)| score > *previous)
            {
                selected = Some((score, *grid));
            }
        }
        let Some((_, grid)) = selected else {
            return;
        };
        let expanded = union(core, grid);
        candidate.local = union(candidate.local, grid);
        candidate.core = Some(inverse(expanded, self.size, self.angle));
        candidate.bbox = inverse(candidate.local, self.size, self.angle);
        let key = expanded.map(|v| if v == 0.0 { 0 } else { v.to_bits() });
        let members = self.cache.get(&key).cloned().unwrap_or_else(|| {
            let mut ids = Ids::default();
            for (center, fragments) in &self.rows {
                if *center < expanded[1] || *center > expanded[3] {
                    continue;
                }
                for (id, b) in fragments {
                    let x = (b[0] + b[2]) / 2.0;
                    let y = (b[1] + b[3]) / 2.0;
                    if expanded[0] <= x && x <= expanded[2] && expanded[1] <= y && y <= expanded[3]
                    {
                        ids.insert(*id);
                    }
                }
            }
            let members = Arc::new(ids);
            // 页面级缓存有界，超出后继续精确计算，不减少候选或成员。
            if self.cache.len() < 128 {
                self.cache.insert(key, members.clone());
            }
            members
        });
        let excluded = candidate
            .annotations
            .iter()
            .flat_map(|a| a.members.iter().copied())
            .collect();
        candidate.members.expand(members, &excluded);
    }
}

/// 动态合并注释；表体优先，行框保留首次插入顺序，再计算注释并集。
fn merge_annotations(target: &mut Candidate, annotations: Vec<Annotation>) {
    for annotation in annotations {
        if let Some(existing) = target
            .annotations
            .iter_mut()
            .find(|a| a.kind == annotation.kind)
        {
            existing.bbox = union(existing.bbox, annotation.bbox);
            existing.members.extend(annotation.members);
            for (id, bbox) in annotation.rows {
                if let Some((_, previous)) = existing.rows.iter_mut().find(|(v, _)| *v == id) {
                    *previous = union(*previous, bbox);
                } else {
                    existing.rows.push((id, bbox));
                }
            }
        } else {
            target.annotations.push(annotation);
        }
    }
    for annotation in &mut target.annotations {
        annotation.members.retain(|id| !target.members.contains(id));
        annotation
            .rows
            .retain(|(id, _)| annotation.members.contains(id));
        if let Some((_, first)) = annotation.rows.first() {
            annotation.bbox = annotation
                .rows
                .iter()
                .skip(1)
                .fold(*first, |b, (_, next)| union(b, *next));
        }
    }
    target.annotations.retain(|a| !a.members.is_empty());
}

#[derive(Default)]
pub struct Merger {
    pub candidates: Vec<Candidate>,
}
impl Merger {
    /// 调用方按分数稳定降序追加；始终与已经合并更新后的首个目标比较。
    pub fn push(&mut self, candidate: Candidate) {
        let Some(target) = self
            .candidates
            .iter_mut()
            .find(|c| c.angle == candidate.angle && overlap(candidate.bbox, c.bbox) >= 0.2)
        else {
            self.candidates.push(candidate);
            return;
        };
        target.bbox = union(target.bbox, candidate.bbox);
        target.local = union(target.local, candidate.local);
        target.core = match (target.core, candidate.core) {
            (Some(a), Some(b)) => Some(union(a, b)),
            (None, b) => b,
            (a, None) => a,
        };
        target.members.merge(&candidate.members);
        if candidate.score > target.score {
            target.score = candidate.score;
        }
        merge_annotations(target, candidate.annotations);
    }
    /// 只在最终边界稳定按页面位置排序，不在中间阶段重排目标。
    pub fn finish(mut self) -> Vec<Candidate> {
        self.candidates.sort_by(|a, b| {
            (a.bbox[1], a.bbox[0])
                .partial_cmp(&(b.bbox[1], b.bbox[0]))
                .unwrap()
        });
        self.candidates
    }
}
