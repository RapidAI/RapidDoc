//! 保序行框聚合与最大下边界查询，返回来源位置以共享 Python 原坐标。

pub struct RowGeometry {
    boxes: Vec<[f64; 4]>,
    size: usize,
    tree: Vec<[usize; 4]>,
}

impl RowGeometry {
    /// 拒绝非有限几何，平局选较早来源，避免改变负零及原坐标身份。
    pub fn new(boxes: Vec<[f64; 4]>) -> Option<Self> {
        if boxes.iter().flatten().any(|x| !x.is_finite()) {
            return None;
        }
        let size = boxes.len().max(1).next_power_of_two();
        let mut result = Self {
            boxes,
            size,
            tree: vec![[usize::MAX; 4]; size * 2],
        };
        for i in 0..result.boxes.len() {
            result.tree[size + i] = [i; 4];
        }
        for i in (1..size).rev() {
            result.tree[i] = result.combine(result.tree[i * 2], result.tree[i * 2 + 1]);
        }
        Some(result)
    }

    /// 合并保序区间的极值来源，严格比较使相等值始终保留左侧对象。
    fn combine(&self, a: [usize; 4], b: [usize; 4]) -> [usize; 4] {
        let mut output = a;
        for k in 0..4 {
            if a[k] == usize::MAX {
                output[k] = b[k];
            } else if b[k] != usize::MAX {
                let better = if k < 2 {
                    self.boxes[b[k]][k] < self.boxes[a[k]][k]
                } else {
                    self.boxes[b[k]][k] > self.boxes[a[k]][k]
                };
                if better {
                    output[k] = b[k];
                }
            }
        }
        output
    }

    /// 使用左右独立累计器，查询树拆分不能改变原始行的比较顺序。
    pub fn union_indices(&self, start: usize, end: usize) -> Option<[usize; 4]> {
        if start >= end || end > self.boxes.len() {
            return None;
        }
        let (mut l, mut r) = (self.size + start, self.size + end);
        let (mut left, mut right) = ([usize::MAX; 4], [usize::MAX; 4]);
        while l < r {
            if l % 2 == 1 {
                left = self.combine(left, self.tree[l]);
                l += 1;
            }
            if r % 2 == 1 {
                r -= 1;
                right = self.combine(self.tree[r], right);
            }
            l /= 2;
            r /= 2;
        }
        Some(self.combine(left, right))
    }

    /// 找到原顺序中首个下边界严格超过表底的行，不要求边界单调。
    pub fn first_after(&self, bottom: f64) -> usize {
        if self.boxes.is_empty() {
            return 0;
        }
        let mut node = 1;
        let source = self.tree[node][3];
        if self.boxes[source][3] <= bottom {
            return self.boxes.len();
        }
        while node < self.size {
            let left = self.tree[node * 2][3];
            node = if left != usize::MAX && self.boxes[left][3] > bottom {
                node * 2
            } else {
                node * 2 + 1
            };
        }
        (node - self.size).min(self.boxes.len())
    }
}
