//! 有限浮点数的精确二进制比率；范围过宽或累积溢出时完整使用 Python Fraction 路径。

/// 将每个二进制浮点数拆成整数尾数和二次幂，保持精确总和而不做浮点累加。
pub fn exact_sum_parts(values: &[f64]) -> Option<(i128, i32)> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut parts = Vec::with_capacity(values.len());
    let mut minimum = i32::MAX;
    for value in values {
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let mut mantissa = bits & ((1_u64 << 52) - 1);
        let mut power = if exponent == 0 {
            -1074
        } else {
            mantissa |= 1_u64 << 52;
            exponent - 1023 - 52
        };
        if mantissa == 0 {
            continue;
        }
        let shift = mantissa.trailing_zeros();
        mantissa >>= shift;
        power += shift as i32;
        minimum = minimum.min(power);
        let mantissa = if bits >> 63 == 0 {
            mantissa as i128
        } else {
            -(mantissa as i128)
        };
        parts.push((mantissa, power));
    }
    if parts.is_empty() {
        return Some((0, 0));
    }
    let mut total = 0_i128;
    for (mantissa, power) in parts {
        let shift = (power - minimum) as u32;
        if shift > 126 {
            return None;
        }
        total = total.checked_add(mantissa.checked_mul(1_i128 << shift)?)?;
    }
    Some((total, minimum))
}

#[cfg(test)]
mod tests {
    /// 整数比率覆盖抵消、正负零、次正规数、巨大同阶数及不可安全累积时的完整回退。
    #[test]
    fn exact_parts_and_fallback() {
        assert_eq!(super::exact_sum_parts(&[1.0, 2.0, 3.0]), Some((6, 0)));
        assert_eq!(super::exact_sum_parts(&[1.0, -1.0]), Some((0, 0)));
        assert_eq!(super::exact_sum_parts(&[-0.0, 0.0]), Some((0, 0)));
        assert_eq!(
            super::exact_sum_parts(&[f64::from_bits(1), f64::from_bits(1)]),
            Some((2, -1074))
        );
        assert!(super::exact_sum_parts(&[1e308, 1e308]).is_some());
        assert!(super::exact_sum_parts(&[1e308, 1e-308]).is_none());
        assert!(super::exact_sum_parts(&[f64::NAN]).is_none());
        assert!(super::exact_sum_parts(&[]).is_none());
    }
}
