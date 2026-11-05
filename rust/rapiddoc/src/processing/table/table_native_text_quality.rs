//! Upstream 8a3ff6c: reject corrupt native table text, allowing the host OCR fallback.
use super::table_native_text::{angle, local, normalize, normalize_bbox};
use crate::legacy_page::invalid;
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};

pub(super) struct Quality {
    stage: &'static str,
    total: usize,
    abnormal: usize,
}
impl Quality {
    pub(super) fn rejected(&self) -> bool {
        self.abnormal >= 2 && self.abnormal.saturating_mul(20) >= self.total
    }
    pub(super) fn record(&self) -> Value {
        json!({"stage":self.stage,"text_chars":self.total,"abnormal_chars":self.abnormal,
            "abnormal_ratio":if self.total==0 {0.} else {self.abnormal as f64/self.total as f64},
            "rejected":self.rejected()})
    }
}
pub(super) fn measure<'a>(parts: impl Iterator<Item = &'a str>, stage: &'static str) -> Quality {
    let mut result = Quality {
        stage,
        total: 0,
        abnormal: 0,
    };
    for part in parts {
        for c in normalize(part).chars() {
            let n = c as u32;
            let bad = (n < 32 && !matches!(c, '\t' | '\r' | '\n'))
                || (127..=159).contains(&n)
                || n == 0xfffd;
            if bad || !python_space(c) {
                result.total += 1;
                result.abnormal += usize::from(bad);
            }
        }
    }
    result
}
pub(super) fn input(v: &Value) -> Result<Quality, Error> {
    let Some(t) = normalize_bbox(&v["table_bbox"])? else {
        return Ok(measure(std::iter::empty(), "input"));
    };
    let a = angle(&v["angle"])?;
    let mut parts = Vec::new();
    for c in v["chars"].as_array().ok_or_else(invalid)? {
        let Some(b) = normalize_bbox(&c["bbox"])? else {
            continue;
        };
        let (x, y) = ((b[0] + b[2]) / 2., (b[1] + b[3]) / 2.);
        if t[0] <= x
            && x <= t[2]
            && t[1] <= y
            && y <= t[3]
            && local(b, t, a).is_some()
            && !c["is_generated"].as_bool().unwrap_or(false)
        {
            parts.push(c["char"].as_str().unwrap_or(""));
        }
    }
    Ok(measure(parts.into_iter(), "input"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn upstream_corruption_threshold_and_normalization() {
        assert!(!measure(["\u{fffd}"].into_iter(), "input").rejected());
        assert!(measure(["\u{fffd}\u{fffd}"].into_iter(), "input").rejected());
        assert!(!measure(["\u{2}\t汉字\u{f0000}"].into_iter(), "input").rejected());
        assert!(measure(
            [format!("{}\u{fffd}\u{fffd}", "x".repeat(38)).as_str()].into_iter(),
            "input"
        )
        .rejected());
        assert!(!measure(
            [format!("{}\u{fffd}\u{fffd}", "x".repeat(39)).as_str()].into_iter(),
            "input"
        )
        .rejected());
    }
}
