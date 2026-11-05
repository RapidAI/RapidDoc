//! 在自有文本数据上连续执行 tight-first、标点桥接及空白归属，不依赖 Python/PDFium。
use crate::geometry::Box4;
use std::collections::HashMap;

pub const SPACE: u8 = 1;
pub const PUNCTUATION: u8 = 2;
pub const STOP: u8 = 4;
pub const START: u8 = 8;
pub const BREAK: u8 = 16;

pub struct Character {
    pub bbox: Box4,
    pub tight: Option<Box4>,
    pub flags: u8,
}

/// 与 Python 的有限 bbox 校验一致，空白字符额外允许零面积 advance point。
fn valid(b: Box4, zero: bool) -> bool {
    b.iter().all(|value| value.is_finite())
        && if zero {
            b[2] >= b[0] && b[3] >= b[1]
        } else {
            b[2] > b[0] && b[3] > b[1]
        }
}

/// 保持原有中轴阈值及起止标点边界规则，双重标点优先按行尾规则处理。
fn inside(b: Box4, span: Box4, flags: u8, ratio: f64) -> bool {
    let x = (b[0] + b[2]) / 2.0;
    let y = (b[1] + b[3]) / 2.0;
    let middle = (span[1] + span[3]) / 2.0;
    let height = span[3] - span[1];
    if !(span[1] < y && y < span[3] && (y - middle).abs() < height * ratio) {
        return false;
    }
    if span[0] < x && x < span[2] {
        return true;
    }
    if flags & STOP != 0 {
        span[2] - height < b[0] && b[0] < span[2] && x > span[0]
    } else if flags & START != 0 {
        span[0] < b[2] && b[2] < span[0] + height && x < span[2]
    } else {
        false
    }
}

/// 保留归一化纵/横中心距离及原 span 顺序的稳定排名。
fn rank(b: Box4, span: Box4, index: usize) -> (f64, f64, usize) {
    let width = (span[2] - span[0]).max(1e-6);
    let height = (span[3] - span[1]).max(1e-6);
    (
        (((b[1] + b[3]) / 2.0) - ((span[1] + span[3]) / 2.0)).abs() / height,
        (((b[0] + b[2]) / 2.0) - ((span[0] + span[2]) / 2.0)).abs() / width,
        index,
    )
}

struct Grid {
    size: f64,
    ratio: f64,
    ranges: Vec<(f64, f64)>,
    cells: Option<HashMap<i64, Vec<usize>>>,
}

impl Grid {
    /// 常见范围使用网格；巨大范围改为等价区间扫描，不截断任何候选。
    fn new(spans: &[Box4], size: f64, ratio: f64) -> Self {
        let ranges: Vec<_> = spans
            .iter()
            .map(|b| ((b[1] / size).trunc(), (b[3] / size).trunc()))
            .collect();
        let count: f64 = ranges.iter().map(|(a, b)| (b - a + 1.0).max(0.0)).sum();
        let ordinary = count <= 1_000_000.0
            && ranges
                .iter()
                .all(|(a, b)| a.abs() < i64::MAX as f64 && b.abs() < i64::MAX as f64);
        let cells = ordinary.then(|| {
            let mut cells: HashMap<i64, Vec<usize>> = HashMap::new();
            for (index, (a, b)) in ranges.iter().enumerate() {
                for cell in *a as i64..=*b as i64 {
                    cells.entry(cell).or_default().push(index);
                }
            }
            cells
        });
        Self {
            size,
            ratio,
            ranges,
            cells,
        }
    }

    /// 候选顺序保持 span 输入顺序，空白按首个命中而非最小距离归属。
    fn matching(&self, b: Box4, flags: u8, spans: &[Box4], first: bool) -> Option<usize> {
        let cell = (((b[1] + b[3]) / 2.0) / self.size).trunc();
        let x = (b[0] + b[2]) / 2.0;
        let mut best = None;
        let mut visit = |index: usize| {
            let span = spans[index];
            if flags & (STOP | START) == 0 && !(span[0] < x && x < span[2]) {
                return false;
            }
            if inside(b, span, flags, self.ratio) {
                if first {
                    best = Some(index);
                    return true;
                }
                if best.is_none_or(|old| rank(b, span, index) < rank(b, spans[old], old)) {
                    best = Some(index);
                }
            }
            false
        };
        if let Some(cells) = &self.cells {
            if cell.abs() < i64::MAX as f64 {
                for &index in cells.get(&(cell as i64)).into_iter().flatten() {
                    if visit(index) {
                        break;
                    }
                }
            }
        } else {
            for (index, &(a, b)) in self.ranges.iter().enumerate() {
                if a <= cell && cell <= b && visit(index) {
                    break;
                }
            }
        }
        best
    }
}

/// 一次完成可见字符匹配、断行隔离、标点补归属与空白修复，输出原字符顺序的 span 索引。
pub fn assign(
    chars: &[Character],
    spans: &[Box4],
    median_height: f64,
    ratio: f64,
) -> Vec<Option<usize>> {
    let grid = Grid::new(spans, median_height.max(1.0), ratio);
    let mut assigned = vec![None; chars.len()];
    for (index, ch) in chars.iter().enumerate() {
        if ch.flags & SPACE != 0 {
            continue;
        }
        for bbox in [ch.tight, Some(ch.bbox)].into_iter().flatten() {
            if valid(bbox, false) {
                assigned[index] = grid.matching(bbox, ch.flags, spans, false);
                if assigned[index].is_some() {
                    break;
                }
            }
        }
    }
    let mut previous = vec![None; chars.len()];
    let mut next = vec![None; chars.len()];
    let mut owner = None;
    for (index, ch) in chars.iter().enumerate() {
        if ch.flags & BREAK != 0 {
            owner = None;
            continue;
        }
        previous[index] = owner;
        if ch.flags & SPACE == 0 && assigned[index].is_some() {
            owner = assigned[index];
        }
    }
    owner = None;
    for (index, ch) in chars.iter().enumerate().rev() {
        if ch.flags & BREAK != 0 {
            owner = None;
            continue;
        }
        next[index] = owner;
        if ch.flags & SPACE == 0 && assigned[index].is_some() {
            owner = assigned[index];
        }
    }
    for (index, ch) in chars.iter().enumerate() {
        if assigned[index].is_some() || ch.flags & PUNCTUATION == 0 {
            continue;
        }
        if let Some(owner) = previous[index].filter(|owner| next[index] == Some(*owner)) {
            let span = spans[owner];
            if [ch.tight, Some(ch.bbox)].into_iter().flatten().any(|bbox| {
                let x = (bbox[0] + bbox[2]) / 2.0;
                let y = (bbox[1] + bbox[3]) / 2.0;
                valid(bbox, false) && span[0] <= x && x <= span[2] && span[1] <= y && y <= span[3]
            }) {
                assigned[index] = Some(owner);
            }
        }
    }
    for (index, ch) in chars.iter().enumerate() {
        if ch.flags & SPACE == 0 || !valid(ch.bbox, true) {
            continue;
        }
        let same = previous[index].is_some() && previous[index] == next[index];
        let neighbor = if same {
            previous[index]
        } else {
            match (previous[index], next[index]) {
                (owner, None) | (None, owner) => owner,
                _ => None,
            }
        };
        assigned[index] = if ch.flags & BREAK == 0
            && neighbor.is_some_and(|owner| same || inside(ch.bbox, spans[owner], ch.flags, ratio))
        {
            neighbor
        } else {
            grid.matching(ch.bbox, ch.flags, spans, true)
        };
    }
    assigned
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造不依赖 PDFium 的真实几何输入，测试归属边界而非绑定实现。
    fn character(bbox: Box4, flags: u8) -> Character {
        Character {
            bbox,
            tight: None,
            flags,
        }
    }

    #[test]
    fn tight_precedes_loose_and_ties_keep_span_order() {
        // tight 命中第二框时不得被 loose 的首框覆盖；完全重叠时保留输入顺序。
        let mut chars = vec![character([2.0, 2.0, 4.0, 8.0], 0)];
        chars[0].tight = Some([12.0, 2.0, 14.0, 8.0]);
        let spans = [[0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0]];
        assert_eq!(assign(&chars, &spans, 10.0, 0.33), [Some(1)]);
        assert_eq!(assign(&chars, &[spans[1], spans[1]], 10.0, 0.33), [Some(0)]);
    }

    #[test]
    fn neighbor_spaces_and_punctuation_do_not_cross_breaks() {
        // 标点超出中轴带但中心仍在同框内时桥接；断行阻止邻居归属传播。
        let span = [[0.0, 0.0, 10.0, 10.0]];
        let chars = [
            character([1.0, 2.0, 2.0, 8.0], 0),
            character([3.0, 0.1, 4.0, 0.3], PUNCTUATION),
            character([4.0, 0.0, 4.0, 0.0], SPACE),
            character([5.0, 2.0, 6.0, 8.0], 0),
            character([0.0, -10.0, 0.0, -10.0], SPACE | BREAK),
            character([7.0, 0.0, 7.0, 0.0], SPACE),
        ];
        assert_eq!(
            assign(&chars, &span, 10.0, 0.33),
            [Some(0), Some(0), Some(0), Some(0), None, None]
        );
    }

    #[test]
    fn whitespace_uses_first_match_instead_of_nearest() {
        // 无邻居空白按原候选顺序，而普通字符采用最近中心。
        let spans = [[0.0, 0.0, 10.0, 20.0], [0.0, 5.0, 10.0, 15.0]];
        let spaces = [character([4.0, 7.0, 5.0, 9.0], SPACE)];
        assert_eq!(assign(&spaces, &spans, 10.0, 0.33), [Some(0)]);
    }

    #[test]
    fn sparse_grid_preserves_large_range_candidates() {
        // 巨大网格范围使用等价扫描，仍命中全覆盖 span，不按候选数量裁剪。
        let spans = [[0.0, 0.0, 10.0, 100_000_000.0]];
        let chars = [character([3.0, 49_999_999.0, 4.0, 50_000_001.0], 0)];
        assert_eq!(assign(&chars, &spans, 1.0, 0.33), [Some(0)]);
    }
}
