//! 有界普通JSON坐标的Python int/float面积算术；坐标差<=2e9，乘积适合i128。
use num_rational::BigRational;
use num_traits::ToPrimitive;
use serde_json::Value;
use std::cmp::Ordering;
#[derive(Clone, Copy, Debug)]
pub(super) enum N {
    I(i128),
    F(f64),
}
impl N {
    pub(super) fn value(v: &Value) -> Self {
        v.as_i64()
            .map_or_else(|| Self::F(v.as_f64().unwrap()), |n| Self::I(n as i128))
    }
    pub(super) fn float(self) -> f64 {
        match self {
            Self::I(n) => n as f64,
            Self::F(n) => n,
        }
    }
    fn rational(self) -> BigRational {
        match self {
            Self::I(n) => BigRational::from_integer(n.into()),
            Self::F(n) => BigRational::from_float(n).unwrap(),
        }
    }
    pub(super) fn cmp(self, b: Self) -> Ordering {
        match (self, b) {
            (Self::I(a), Self::I(b)) => a.cmp(&b),
            (Self::F(a), Self::F(b)) => a.partial_cmp(&b).unwrap(),
            (Self::I(i), Self::F(f)) if i.abs() <= 1i128 << 53 => {
                (i as f64).partial_cmp(&f).unwrap()
            }
            (Self::F(f), Self::I(i)) if i.abs() <= 1i128 << 53 => {
                f.partial_cmp(&(i as f64)).unwrap()
            }
            _ => self.rational().cmp(&b.rational()),
        }
    }
    pub(super) fn sub(self, b: Self) -> Self {
        match (self, b) {
            (Self::I(a), Self::I(b)) => Self::I(a - b),
            _ => Self::F(self.float() - b.float()),
        }
    }
    pub(super) fn mul(self, b: Self) -> Self {
        match (self, b) {
            (Self::I(a), Self::I(b)) => Self::I(a * b),
            _ => Self::F(self.float() * b.float()),
        }
    }
    pub(super) fn div(self, b: Self) -> f64 {
        match (self, b) {
            (Self::I(_), Self::I(_)) => (self.rational() / b.rational()).to_f64().unwrap(),
            _ => self.float() / b.float(),
        }
    }
    pub(super) fn min(self, b: Self) -> Self {
        if b.cmp(self) == Ordering::Less {
            b
        } else {
            self
        }
    }
    pub(super) fn max(self, b: Self) -> Self {
        if b.cmp(self) == Ordering::Greater {
            b
        } else {
            self
        }
    }
    pub(super) fn le(self, b: Self) -> bool {
        self.cmp(b) != Ordering::Greater
    }
}
