//! Normalize PDFium font-weight metadata before it reaches layout heuristics.
//! PDFium may report values outside the OpenType 1..=1000 range for embedded
//! fonts. Such estimates are not reliable typography evidence.

pub(crate) fn normalize(weight: i32) -> i32 {
    if (1..=1000).contains(&weight) {
        weight
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::normalize;

    #[test]
    fn keeps_valid_open_type_weights() {
        for weight in [1, 400, 700, 1000] {
            assert_eq!(normalize(weight), weight);
        }
    }

    #[test]
    fn rejects_out_of_range_pdfium_estimates() {
        for weight in [-1, 0, 1001, i32::MAX, i32::MIN] {
            assert_eq!(normalize(weight), -1);
        }
    }
}
