//! 原始字符分类统计，与 canonical 去重后的页面字符严格分离。
use std::collections::HashSet;

#[derive(Default)]
pub struct Counts {
    pub null: usize,
    pub replacement: usize,
    pub control: usize,
    pub private: usize,
    pub map_error: usize,
    pub non_generated: usize,
    pub fonts: Vec<(usize, usize, usize)>,
    pub font_first: Vec<[usize; 3]>,
}

/// 保留原判定分支优先级，并按原始字体 ID 累积全部、非生成及非生成 CJK 字符数。
pub fn count(
    records: &[(u32, bool, bool, usize)],
    font_count: usize,
    cjk: &[(u32, u32)],
    allowed: &HashSet<u32>,
    private: (u32, u32),
) -> Counts {
    let mut result = Counts {
        fonts: vec![(0, 0, 0); font_count],
        font_first: vec![[usize::MAX; 3]; font_count],
        ..Counts::default()
    };
    for (position, &(code, generated, error, font)) in records.iter().enumerate() {
        if code == 0 {
            result.null += 1;
        } else if code == 0xfffd {
            result.replacement += 1;
        } else if (code < 32 || (127..=159).contains(&code)) && !allowed.contains(&code) {
            result.control += 1;
        } else if private.0 <= code && code <= private.1 {
            result.private += 1;
        }
        result.map_error += usize::from(error);
        result.fonts[font].0 += 1;
        result.font_first[font][0] = result.font_first[font][0].min(position);
        if !generated {
            result.non_generated += 1;
            result.fonts[font].1 += 1;
            result.font_first[font][1] = result.font_first[font][1].min(position);
            if cjk.iter().any(|&(start, end)| start <= code && code <= end) {
                result.fonts[font].2 += 1;
                result.font_first[font][2] = result.font_first[font][2].min(position);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 控制字符优先级不受重叠 PUA 范围影响，generated 与错误码统计独立保留。
    #[test]
    fn raw_priority_and_generated_counts() {
        let records = [
            (0, false, false, 0),
            (0xfffd, true, true, 0),
            (2, false, false, 0),
            (65, false, true, 1),
            (0xffffffff, false, false, 1),
        ];
        let result = count(&records, 2, &[(65, 90)], &HashSet::new(), (0, 100));
        assert_eq!(
            (
                result.null,
                result.replacement,
                result.control,
                result.private,
                result.map_error,
                result.non_generated
            ),
            (1, 1, 1, 1, 2, 4)
        );
        assert_eq!(result.fonts, vec![(3, 2, 0), (2, 2, 1)]);
    }
}
