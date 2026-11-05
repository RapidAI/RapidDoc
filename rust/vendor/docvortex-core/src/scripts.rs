//! 逐值复现 origin/loose/tight 上下标判定；Unicode 特征由 Python 提供。

use crate::{median, quantile};

pub type Box4 = [f64; 4];
pub type Record = (Box4, Option<Box4>, Option<[f64; 2]>, u32, i64);
const SPACE: u32 = 1;
const BREAK: u32 = 2;
const ALNUM: u32 = 4;
const DECIMAL: u32 = 8;
const LOCAL: u32 = 16;
const SLASH: u32 = 32;
const SEPARATOR: u32 = 64;
const PROTECTED: u32 = 128;
const VALID: u32 = 256;

#[derive(Clone)]
struct Feature {
    loose: Box4,
    tight: Option<Box4>,
    origin: Option<[f64; 2]>,
    flags: u32,
    font: i64,
}
impl Feature {
    /// 检查 Python 预计算的字符类别或保护位。
    fn has(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }
    /// 返回 loose 字形高度。
    fn lh(&self) -> f64 {
        self.loose[3] - self.loose[1]
    }
    /// 返回有效 tight 字形高度。
    fn th(&self) -> f64 {
        self.tight.map_or(0.0, |b| b[3] - b[1])
    }
    /// 返回 loose 中心，保持加法后除二的运算次序。
    fn lc(&self) -> f64 {
        (self.loose[1] + self.loose[3]) / 2.0
    }
    /// 返回 tight 中心，仅在已有有效框的分支使用。
    fn tc(&self) -> f64 {
        self.tight.map_or(0.0, |b| (b[1] + b[3]) / 2.0)
    }
    /// 返回字符 origin 基线。
    fn y(&self) -> f64 {
        self.origin.map_or(0.0, |p| p[1])
    }
}

#[derive(Clone)]
struct Cluster {
    baseline: f64,
    indices: Vec<usize>,
    tight: f64,
    loose: f64,
}

/// 返回上下标角色：零为正文、一为上标、二为下标。
fn role(shift: f64) -> u8 {
    if shift > 0.0 {
        2
    } else {
        1
    }
}

/// 保留双向水平净空的原比较规则。
fn gap(a: Box4, b: Box4) -> f64 {
    0.0_f64.max(a[0] - b[2]).max(b[0] - a[2])
}

/// 按来源顺序切分视觉组件，保留缺失几何字符的位置。
fn components(f: &[Feature]) -> Vec<Vec<usize>> {
    let heights: Vec<_> = f.iter().map(Feature::lh).filter(|v| *v > 0.0).collect();
    let scale = if heights.is_empty() {
        1.0
    } else {
        median(heights)
    };
    let mut result = Vec::new();
    let mut current = Vec::new();
    let mut previous: Option<usize> = None;
    for (i, item) in f.iter().enumerate() {
        if item.has(SPACE | BREAK) {
            if !current.is_empty() {
                result.push(std::mem::take(&mut current));
            }
            previous = None;
            continue;
        }
        if item.has(VALID) {
            if let Some(p) = previous {
                if (f[p].loose[0] - item.loose[0] > scale * 0.5
                    || item.loose[0] - f[p].loose[2] > scale * 1.5)
                    && !current.is_empty()
                {
                    result.push(std::mem::take(&mut current));
                }
            }
        }
        current.push(i);
        if item.has(VALID) {
            previous = Some(i);
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

/// 按 origin 稳定聚类字母数字锚点，并缓存簇的双框高度。
fn clusters(f: &[Feature], indices: &[usize]) -> (Vec<Cluster>, f64) {
    let mut anchors: Vec<_> = indices
        .iter()
        .copied()
        .filter(|i| f[*i].has(VALID) && f[*i].has(ALNUM) && !f[*i].has(PROTECTED))
        .collect();
    if anchors.is_empty() {
        return (Vec::new(), 0.0);
    }
    let tolerance = 0.35_f64.max(median(anchors.iter().map(|i| f[*i].lh()).collect()) * 0.04);
    anchors.sort_by(|a, b| f[*a].y().partial_cmp(&f[*b].y()).unwrap());
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in anchors {
        let append = groups.last().is_some_and(|g| {
            let n = g.len();
            let mid = if n % 2 == 1 {
                f[g[n / 2]].y()
            } else {
                (f[g[n / 2 - 1]].y() + f[g[n / 2]].y()) / 2.0
            };
            (f[i].y() - mid).abs() <= tolerance
        });
        if append {
            groups.last_mut().unwrap().push(i);
        } else {
            groups.push(vec![i]);
        }
    }
    (
        groups.into_iter().map(|g| make_cluster(f, g)).collect(),
        tolerance,
    )
}

/// 从同一簇的成员生成参考统计。
fn make_cluster(f: &[Feature], indices: Vec<usize>) -> Cluster {
    Cluster {
        baseline: median(indices.iter().map(|i| f[*i].y()).collect()),
        tight: quantile(
            indices
                .iter()
                .map(|i| f[*i].th())
                .filter(|v| *v > 0.0)
                .collect(),
            0.9,
        ),
        loose: quantile(
            indices
                .iter()
                .map(|i| f[*i].lh())
                .filter(|v| *v > 0.0)
                .collect(),
            0.9,
        ),
        indices,
    }
}

/// 在可比高度中选最多成员的簇，相同键保留首个候选。
fn body(cs: &[Cluster]) -> Option<usize> {
    let max_height = cs.iter().map(|c| c.tight).fold(0.0_f64, f64::max);
    let mut best: Option<usize> = None;
    for (i, c) in cs.iter().enumerate() {
        if c.tight < max_height * 0.9 {
            continue;
        }
        if best.is_none_or(|j| {
            (c.indices.len(), c.tight, c.loose) > (cs[j].indices.len(), cs[j].tight, cs[j].loose)
        }) {
            best = Some(i);
        }
    }
    best
}

/// 取得同簇字符中心中位数。
fn center(f: &[Feature], c: &Cluster, tight: bool) -> f64 {
    median(
        c.indices
            .iter()
            .map(|i| if tight { f[*i].tc() } else { f[*i].lc() })
            .collect(),
    )
}

/// 判断三种位移证据是否形成足够的同向共识。
fn consistent(f: &[Feature], c: &Cluster, b: &Cluster) -> bool {
    let shift = c.baseline - b.baseline;
    let t = center(f, c, true) - center(f, b, true);
    let l = center(f, c, false) - center(f, b, false);
    let count = [shift, t, l]
        .iter()
        .filter(|v| if shift > 0.0 { **v > 0.05 } else { **v < -0.05 })
        .count();
    count >= 2
        && (shift.abs() / b.tight.max(1e-6) >= 0.22
            || (t.abs() / b.tight.max(1e-6) >= 0.3 && l.abs() / b.loose.max(1e-6) >= 0.15))
}

/// 用连续同字体局部正文撤销弱角标，不使用本轮撤销后的角色作参考。
fn recheck(f: &[Feature], indices: &[usize], cs: &[Cluster], bidx: usize, roles: &mut [u8]) {
    let b = &cs[bidx];
    let initial = roles.to_vec();
    let order: Vec<_> = std::iter::once(bidx)
        .chain((0..cs.len()).filter(|i| *i != bidx))
        .collect();
    for &ci in &order {
        let c = &cs[ci];
        if initial[ci] == 0 || (c.baseline - b.baseline).abs() >= b.tight * 0.3 {
            continue;
        }
        let key = f[c.indices[0]].font;
        if key < 0 || c.indices.iter().any(|i| f[*i].font != key) {
            continue;
        }
        let belongs = |i: usize| {
            i < f.len()
                && indices.contains(&i)
                && f[i].has(VALID)
                && f[i].has(LOCAL)
                && f[i].font == key
        };
        let mut left = *c.indices.iter().min().unwrap();
        let mut right = left + 1;
        while left > 0 && belongs(left - 1) {
            left -= 1;
        }
        while belongs(right) {
            right += 1;
        }
        if c.indices
            .iter()
            .any(|i| *i < left || *i >= right || !belongs(*i))
        {
            continue;
        }
        let mut best: Option<(f64, usize, Cluster)> = None;
        for &ri in &order {
            if initial[ri] != 0 {
                continue;
            }
            let members: Vec<_> = cs[ri]
                .indices
                .iter()
                .copied()
                .filter(|i| *i >= left && *i < right)
                .collect();
            if members.len() < 2 {
                continue;
            }
            let distance = c
                .indices
                .iter()
                .flat_map(|a| {
                    members
                        .iter()
                        .map(|i| gap(f[*a].tight.unwrap(), f[*i].tight.unwrap()))
                })
                .fold(f64::INFINITY, f64::min);
            if distance > 2.5_f64.max(b.tight * 1.2) {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(d, n, _)| distance < *d || (distance == *d && members.len() > *n))
            {
                best = Some((distance, members.len(), make_cluster(f, members)));
            }
        }
        if let Some((_, _, reference)) = best {
            if reference.tight <= 0.0 {
                continue;
            }
            let shift = c.baseline - reference.baseline;
            let ratio = c.tight / reference.tight;
            if shift.abs() < 0.5_f64.max(reference.tight * 0.12)
                || (ratio > 0.9 && !(shift.abs() >= reference.tight * 0.3 && ratio <= 1.1))
            {
                roles[ci] = 0;
            }
        }
    }
}

/// 处理单个组件；集合遍历会影响精确平局时交回 Python 保留其原始顺序。
fn assign(f: &[Feature], indices: &[usize], roles: &mut [u8]) -> Option<()> {
    let (cs, tolerance) = clusters(f, indices);
    let Some(bidx) = body(&cs) else {
        return Some(());
    };
    let b = &cs[bidx];
    if b.tight <= 0.0 || b.loose <= 0.0 {
        return Some(());
    }
    let mut cr = vec![0; cs.len()];
    for (i, c) in cs.iter().enumerate() {
        if i == bidx {
            continue;
        }
        let shift = c.baseline - b.baseline;
        let tr = c.tight / b.tight;
        let lr = c.loose / b.loose;
        if shift.abs() < 0.5_f64.max(b.tight * 0.12)
            || (tr > 0.9 && !(shift.abs() >= b.tight * 0.3 && tr <= 1.1))
            || (lr > 1.35 && !consistent(f, c, b))
        {
            continue;
        }
        cr[i] = role(shift);
    }
    recheck(f, indices, &cs, bidx, &mut cr);
    for &i in indices {
        if f[i].has(PROTECTED | SPACE | BREAK) || f[i].origin.is_none() {
            continue;
        }
        let mut best = 0;
        for j in 1..cs.len() {
            if (f[i].y() - cs[j].baseline).abs() < (f[i].y() - cs[best].baseline).abs() {
                best = j;
            }
        }
        if (f[i].y() - cs[best].baseline).abs() <= tolerance {
            roles[i] = cr[best];
        }
    }
    let seeds: Vec<_> = indices
        .iter()
        .copied()
        .filter(|i| roles[*i] != 0 && f[*i].has(ALNUM) && f[*i].tight.is_some())
        .collect();
    for &i in indices {
        if roles[i] == 0 || f[i].has(ALNUM) || f[i].tight.is_none() {
            continue;
        }
        if !seeds.iter().any(|j| {
            roles[*j] == roles[i]
                && gap(f[i].tight.unwrap(), f[*j].tight.unwrap()) <= 2.0_f64.max(b.tight * 0.5)
                && i.abs_diff(*j) <= 4
        }) {
            roles[i] = 0;
        }
    }
    for &i in indices {
        let item = &f[i];
        if roles[i] != 0 || b.indices.contains(&i) || !item.has(VALID) {
            continue;
        }
        let mut best: Option<(f64, usize, usize)> = None;
        let mut tied = false;
        for &j in &b.indices {
            let key = (gap(item.tight.unwrap(), f[j].tight.unwrap()), i.abs_diff(j));
            match best {
                Some((d, n, _)) if key == (d, n) => {
                    tied = true;
                }
                Some((d, n, _)) if key > (d, n) => {}
                _ => {
                    best = Some((key.0, key.1, j));
                    tied = false;
                }
            }
        }
        let Some((distance, _, j)) = best else {
            continue;
        };
        if tied {
            return None;
        }
        let reference = &f[j];
        if distance > 2.5_f64.max(reference.th() * 1.2)
            || item.th() / reference.th().max(1e-6) > 0.88
        {
            continue;
        }
        let shifts = [
            item.y() - reference.y(),
            item.tc() - reference.tc(),
            item.lc() - reference.lc(),
        ];
        let pos = shifts.iter().filter(|v| **v > 0.05).count();
        let neg = shifts.iter().filter(|v| **v < -0.05).count();
        if pos.max(neg) < 2 {
            continue;
        }
        let r = if pos > neg { 2 } else { 1 };
        if role(shifts[0]) != r {
            continue;
        }
        let or = shifts[0].abs() / reference.th().max(1e-6);
        let tr = shifts[1].abs() / reference.th().max(1e-6);
        let lr = shifts[2].abs() / reference.lh().max(1e-6);
        if !item.has(ALNUM)
            && (!shifts
                .iter()
                .all(|v| if r == 2 { *v > 0.05 } else { *v < -0.05 })
                || !(or >= 0.3 && tr >= 0.3 && lr >= 0.15))
        {
            continue;
        }
        if or >= 0.22 || (tr >= 0.3 && lr >= 0.15) {
            roles[i] = r;
        }
    }
    loop {
        let mut changed = false;
        for &i in indices {
            let item = &f[i];
            if roles[i] != 0
                || item.has(PROTECTED | SPACE)
                || !item.has(VALID)
                || item.th() > b.tight * 1.1
            {
                continue;
            }
            let previous = i
                .checked_sub(1)
                .filter(|j| indices.contains(j))
                .map_or(0, |j| roles[j]);
            let next = if indices.contains(&(i + 1)) {
                roles[i + 1]
            } else {
                0
            };
            if (previous == 0 && next == 0) || (previous != 0 && next != 0 && previous != next) {
                continue;
            }
            let r = previous.max(next);
            let shift = item.y() - b.baseline;
            if shift.abs() < b.tight * 0.08 || role(shift) != r {
                continue;
            }
            roles[i] = r;
            changed = true;
        }
        if !changed {
            break;
        }
    }
    for &i in indices {
        if f[i].has(PROTECTED) {
            roles[i] = 0;
        }
    }
    Some(())
}

/// 独立收集数字引用上标，不让新角色反过来污染后续正文参考。
fn numeric(f: &[Feature], roles: &[u8]) -> Vec<usize> {
    let mut accepted = Vec::new();
    let mut start = 0;
    while start < f.len() {
        if !f[start].has(DECIMAL) {
            start += 1;
            continue;
        }
        let begin = start;
        let mut end = start + 1;
        while end < f.len() && f[end].has(DECIMAL | SEPARATOR) {
            end += 1;
        }
        while end > start && !f[end - 1].has(DECIMAL) {
            end -= 1;
        }
        start = end;
        if (begin > 0 && f[begin - 1].has(SLASH)) || (end < f.len() && f[end].has(SLASH)) {
            continue;
        }
        let mut previous = begin;
        while previous > 0 && f[previous - 1].has(SPACE) {
            if f[previous - 1].has(BREAK) {
                break;
            }
            previous -= 1;
        }
        let reference_end = previous;
        while previous > 0
            && f[previous - 1].has(VALID)
            && roles[previous - 1] == 0
            && !f[previous - 1].has(PROTECTED)
        {
            previous -= 1;
        }
        if previous == reference_end
            || (begin..end).any(|i| !f[i].has(VALID) || f[i].has(PROTECTED) || roles[i] == 2)
        {
            continue;
        }
        let refs: Vec<_> = (previous..reference_end).collect();
        let (cs, _) = clusters(f, &refs);
        let Some(bidx) = body(&cs) else {
            continue;
        };
        let b = &cs[bidx];
        if b.indices.len() < 2 || b.tight <= 0.0 || b.loose <= 0.0 {
            continue;
        }
        let distance = f[begin].tight.unwrap()[0] - f[reference_end - 1].tight.unwrap()[2];
        if distance < -0.15 * b.tight || distance > 2.5_f64.max(1.2 * b.tight) {
            continue;
        }
        let digits: Vec<_> = (begin..end).filter(|i| f[*i].has(DECIMAL)).collect();
        let baseline = median(digits.iter().map(|i| f[*i].y()).collect());
        let shift = b.baseline - baseline;
        if shift < 0.5_f64.max(0.22 * b.tight) || shift > 0.8 * b.tight {
            continue;
        }
        let tolerance = 0.35_f64.max(0.04 * b.loose);
        if (begin..end).any(|i| (f[i].y() - baseline).abs() > tolerance) {
            continue;
        }
        let tr = digits.iter().map(|i| f[*i].th()).fold(0.0_f64, f64::max) / b.tight;
        if tr > 0.88 {
            let lr = digits.iter().map(|i| f[*i].lh()).fold(0.0_f64, f64::max) / b.loose;
            if !(shift >= 0.3 * b.tight && tr <= 1.1 && lr <= 0.88) {
                continue;
            }
        }
        let digit = make_cluster(f, digits);
        if center(f, &digit, true) >= center(f, b, true) - 0.05
            || center(f, &digit, false) >= center(f, b, false) - 0.05
        {
            continue;
        }
        accepted.extend(begin..end);
    }
    accepted
}

/// 分类整批字符；极端浮点或集合顺序平局由参考算法精确处理。
pub fn classify(records: Vec<Record>) -> Option<Vec<u8>> {
    let f: Vec<_> = records
        .into_iter()
        .map(|(loose, tight, origin, flags, font)| Feature {
            loose,
            tight,
            origin,
            flags,
            font,
        })
        .collect();
    if f.iter().any(|v| {
        v.loose
            .iter()
            .chain(v.tight.iter().flatten())
            .chain(v.origin.iter().flatten())
            .any(|x| !x.is_finite() || x.abs() > 1e150)
    }) {
        return None;
    }
    let mut roles = vec![0; f.len()];
    for indices in components(&f) {
        assign(&f, &indices, &mut roles)?;
    }
    for i in numeric(&f, &roles) {
        roles[i] = 1;
    }
    Some(roles)
}
