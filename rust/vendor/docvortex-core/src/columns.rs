//! 按当前 CPython 的求和顺序增量聚类，不重排锚点或首命中簇。

#[derive(Clone)]
pub struct Column {
    pub mean: f64,
    pub hi: f64,
    pub lo: f64,
    pub count: usize,
    pub rows: usize,
    last_row: usize,
}

pub struct Columns {
    pub groups: [Vec<Column>; 3],
    pub row_count: usize,
    compensated: bool,
    valid: bool,
}

impl Columns {
    /// 首个锚点的均值保持原值；累计器模拟 Python 整数零加首个浮点值。
    pub fn new(compensated: bool) -> Self {
        Self {
            groups: Default::default(),
            row_count: 0,
            compensated,
            valid: true,
        }
    }

    /// 单调追加行并保持每簇去重后的行覆盖量，溢出后丢弃整个原生状态。
    pub fn extend(&mut self, rows: &[Vec<(f64, f64)>], tolerance: f64) -> Option<(usize, f64)> {
        self.extend_rows(rows, tolerance)
    }

    /// 自有页面锚点通过借用片段追加，不再为每个区间复制嵌套数值数组。
    pub fn extend_refs(&mut self, rows: &[&[(f64, f64)]], tolerance: f64) -> Option<(usize, f64)> {
        self.extend_rows(rows, tolerance)
    }

    /// 两种传输形式共用完全相同的累计与稳定聚类实现，保留浮点求和顺序。
    fn extend_rows<T: AsRef<[(f64, f64)]>>(
        &mut self,
        rows: &[T],
        tolerance: f64,
    ) -> Option<(usize, f64)> {
        if !self.valid || !tolerance.is_finite() || tolerance < 0.0 {
            return None;
        }
        for alignment in 0..3 {
            let clusters = &mut self.groups[alignment];
            for (offset, fragments) in rows.iter().enumerate() {
                let row = self.row_count + offset;
                for &(left, right) in fragments.as_ref() {
                    let anchor = match alignment {
                        0 => left,
                        1 => (left + right) / 2.0,
                        _ => right,
                    };
                    if !left.is_finite() || !right.is_finite() || !anchor.is_finite() {
                        self.valid = false;
                        return None;
                    }
                    if let Some(cluster) = clusters
                        .iter_mut()
                        .find(|c| (anchor - c.mean).abs() <= tolerance)
                    {
                        let t = cluster.hi + anchor;
                        if self.compensated {
                            cluster.lo += if cluster.hi.abs() >= anchor.abs() {
                                (cluster.hi - t) + anchor
                            } else {
                                (anchor - t) + cluster.hi
                            };
                        }
                        cluster.hi = t;
                        cluster.count += 1;
                        let sum = if self.compensated && cluster.lo != 0.0 && cluster.lo.is_finite()
                        {
                            cluster.hi + cluster.lo
                        } else {
                            cluster.hi
                        };
                        cluster.mean = sum / cluster.count as f64;
                        if !cluster.mean.is_finite()
                            || !cluster.hi.is_finite()
                            || !cluster.lo.is_finite()
                        {
                            self.valid = false;
                            return None;
                        }
                        if cluster.last_row != row {
                            cluster.rows += 1;
                            cluster.last_row = row;
                        }
                    } else {
                        clusters.push(Column {
                            mean: anchor,
                            hi: 0.0 + anchor,
                            lo: 0.0,
                            count: 1,
                            rows: 1,
                            last_row: row,
                        });
                    }
                }
            }
        }
        self.row_count += rows.len();
        let mut best = (0, 0.0);
        if self.row_count == 0 {
            return Some(best);
        }
        for clusters in &self.groups {
            let mut count = 0;
            let mut minimum: f64 = 1.0;
            for cluster in clusters {
                let coverage = cluster.rows as f64 / self.row_count as f64;
                if coverage >= 0.5 {
                    count += 1;
                    minimum = minimum.min(coverage);
                }
            }
            let result = (count, if count == 0 { 0.0 } else { minimum });
            if result > best {
                best = result;
            }
        }
        Some(best)
    }
}
