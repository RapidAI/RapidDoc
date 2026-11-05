//! 明确坐标系的仿射变换；所有运算拒绝非有限值，空交集保留为空矩形。

use crate::{Error, ErrorCode, Rect};

/// PDF 约定的六参数仿射变换，x'=a*x+c*y+e，y'=b*x+d*y+f。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine([f64; 6]);

impl Affine {
    pub const IDENTITY: Self = Self([1., 0., 0., 1., 0., 0.]);

    /// 返回值拷贝，供事实层记录完整变换，不暴露可变内部状态。
    pub fn values(self) -> [f64; 6] {
        self.0
    }

    /// 允许奇异矩阵以保留退化对象，但不允许 NaN 或无限值。
    pub fn new(values: [f64; 6]) -> Result<Self, Error> {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid("变换矩阵包含非有限值"));
        }
        Ok(Self(values))
    }

    /// 应用变换并检查中间结果溢出。
    pub fn point(self, point: [f64; 2]) -> Result<[f64; 2], Error> {
        let [a, b, c, d, e, f] = self.0;
        let [x, y] = point;
        let result = [a * x + c * y + e, b * x + d * y + f];
        if result.iter().any(|value| !value.is_finite()) {
            return Err(invalid("点变换产生非有限坐标"));
        }
        Ok(result)
    }

    /// 先执行 self 再执行 next，适用于局部对象矩阵后接父级矩阵。
    pub fn then(self, next: Self) -> Result<Self, Error> {
        let [a, b, c, d, e, f] = self.0;
        let [g, h, i, j, k, l] = next.0;
        Self::new([
            g * a + i * b,
            h * a + j * b,
            g * c + i * d,
            h * c + j * d,
            g * e + i * f + k,
            h * e + j * f + l,
        ])
    }

    /// 逆变换仅用于可逆坐标系；零行列式明确报错。
    pub fn inverse(self) -> Result<Self, Error> {
        let [a, b, c, d, e, f] = self.0;
        let determinant = a * d - b * c;
        if determinant == 0. || !determinant.is_finite() {
            return Err(invalid("变换矩阵不可逆"));
        }
        Self::new([
            d / determinant,
            -b / determinant,
            -c / determinant,
            a / determinant,
            (c * f - d * e) / determinant,
            (b * e - a * f) / determinant,
        ])
    }

    /// 变换矩形四角后求外框，不能仅变换对角线而丢失旋转或斜切范围。
    pub fn rect(self, rect: Rect) -> Result<Rect, Error> {
        let [left, bottom, right, top] = rect.into();
        bounds(&[
            self.point([left, bottom])?,
            self.point([left, top])?,
            self.point([right, bottom])?,
            self.point([right, top])?,
        ])
    }
}

/// PDF 原生坐标到旋转后左上原点 point 的映射；bbox 为裁剪后的未旋转页面框。
pub fn page_transform(bbox: Rect, rotation: u16) -> Result<Affine, Error> {
    let [left, bottom, right, top] = bbox.into();
    if right <= left || top <= bottom {
        return Err(invalid("页面框必须有正面积"));
    }
    Affine::new(match rotation {
        0 => [1., 0., 0., -1., -left, top],
        90 => [0., 1., 1., 0., -bottom, -left],
        180 => [-1., 0., 0., 1., right, -bottom],
        270 => [0., -1., -1., 0., top, right],
        _ => return Err(invalid("页面旋转必须为 0/90/180/270")),
    })
}

/// 对有限点集求框；空点集不是有效的几何证据。
pub fn bounds(points: &[[f64; 2]]) -> Result<Rect, Error> {
    let first = points.first().ok_or_else(|| invalid("几何点集为空"))?;
    let mut values = [first[0], first[1], first[0], first[1]];
    for [x, y] in points {
        if !x.is_finite() || !y.is_finite() {
            return Err(invalid("点集包含非有限值"));
        }
        values[0] = values[0].min(*x);
        values[1] = values[1].min(*y);
        values[2] = values[2].max(*x);
        values[3] = values[3].max(*y);
    }
    Rect::try_from(values)
}

/// 空交集保留零面积，不允许重新解释为没有裁剪。
pub fn intersect(first: Rect, second: Rect) -> Result<Rect, Error> {
    let [a, b, c, d] = first.into();
    let [e, f, g, h] = second.into();
    let left = a.max(e);
    let bottom = b.max(f);
    Rect::try_from([left, bottom, c.min(g).max(left), d.min(h).max(bottom)])
}

/// 构造坐标输入错误。
fn invalid(message: &str) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.into(),
    }
}
