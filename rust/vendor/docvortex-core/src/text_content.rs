//! 在自有字符上完成正文片段组装；Unicode 分类和 NFC 组合由绑定层按解释器语义准备。
use crate::{geometry::Box4, text_snapshot::TextSnapshot};
use std::collections::{HashMap, HashSet};

pub struct Content {
    pub text: Option<String>,
    pub private_count: usize,
    pub text_count: usize,
    pub private_run: usize,
}

/// 按原字符顺序收集各 span 的成员，保持 PUA 连续段判定所需的原始顺序。
pub fn groups(assignments: &[Option<usize>], span_count: usize) -> Vec<Vec<usize>> {
    let mut output = vec![Vec::new(); span_count];
    for (index, owner) in assignments.iter().enumerate() {
        if let Some(owner) = owner {
            output[*owner].push(index);
        }
    }
    output
}

/// 与 Python 一样，仅在来源编号乱序时稳定排序，等号不改变成员顺序。
pub fn ordered(data: &TextSnapshot, group: &[usize]) -> Vec<usize> {
    let mut values = group.to_vec();
    if values
        .windows(2)
        .any(|pair| data.chars[pair[0]].index > data.chars[pair[1]].index)
    {
        values.sort_by_key(|&index| data.chars[index].index);
    }
    values
}

/// 计算相对较短边的交集比例，零宽高不组成重叠附加符。
fn overlap(a: Box4, b: Box4, axis: usize) -> f64 {
    let denominator = (a[axis + 2] - a[axis]).min(b[axis + 2] - b[axis]);
    if denominator <= 0.0 {
        return 0.0;
    }
    (a[axis + 2].min(b[axis + 2]) - a[axis].max(b[axis])).max(0.0) / denominator
}

/// 保留当前字符为附加符时的优先分支；不在失败后尝试参考实现未覆盖的反向组合。
pub fn composition_pair(
    data: &TextSnapshot,
    first: usize,
    second: usize,
    modifiers: &HashMap<String, String>,
    threshold: f64,
) -> Option<(usize, usize)> {
    let (base, modifier) = if modifiers.contains_key(&data.chars[first].text) {
        (second, first)
    } else if modifiers.contains_key(&data.chars[second].text) {
        (first, second)
    } else {
        return None;
    };
    let a = &data.chars[base];
    let b = &data.chars[modifier];
    (a.text.chars().count() == 1
        && overlap(a.bbox, b.bbox, 0) >= threshold
        && overlap(a.bbox, b.bbox, 1) >= threshold)
        .then_some((base, modifier))
}

/// 仅替换现有控制字符和七种拉丁连字，不把其他 Unicode 字符擅自做 NFKC 转换。
fn clean(text: &str) -> String {
    let controls = text.replace("\r\n", "").replace('\u{2}', "-");
    let mut output = String::with_capacity(controls.len());
    for ch in controls.chars() {
        match ch {
            '\u{fb01}' => output.push_str("fi"),
            '\u{fb02}' => output.push_str("fl"),
            '\u{fb00}' => output.push_str("ff"),
            '\u{fb03}' => output.push_str("ffi"),
            '\u{fb04}' => output.push_str("ffl"),
            '\u{fb05}' => output.push_str("ft"),
            '\u{fb06}' => output.push_str("st"),
            _ => output.push(ch),
        }
    }
    output
}

/// 连续执行 PUA 信号、重叠附加符合成、词间空格及正文组装；空组保留调用方原文本。
pub struct Rules<'a> {
    pub ordinary: &'a HashSet<char>,
    pub decimals: &'a HashSet<char>,
    pub tight_spacing: &'a [bool],
    pub modifiers: &'a HashMap<String, String>,
    pub threshold: f64,
    pub compositions: &'a HashMap<(String, String), String>,
    pub spaces: &'a HashSet<char>,
    pub breaks: &'a HashSet<String>,
    pub private_ranges: &'a [(u32, u32)],
}

/// 使用本次调用固定的字符规则构造所有片段，避免重新读取可变 Python 配置。
pub fn build(data: &TextSnapshot, groups: &[Vec<usize>], rules: &Rules<'_>) -> Vec<Content> {
    let Rules {
        ordinary,
        decimals,
        tight_spacing,
        modifiers,
        threshold,
        compositions,
        spaces,
        breaks,
        private_ranges,
    } = rules;
    groups
        .iter()
        .enumerate()
        .map(|(group_index, group)| {
            let mut result = Content {
                text: None,
                private_count: 0,
                text_count: 0,
                private_run: 0,
            };
            let mut run = 0;
            for &index in group {
                for ch in data.chars[index].text.chars() {
                    if spaces.contains(&ch) {
                        run = 0;
                        continue;
                    }
                    result.text_count += 1;
                    if private_ranges
                        .iter()
                        .any(|&(start, end)| start <= ch as u32 && ch as u32 <= end)
                    {
                        result.private_count += 1;
                        run += 1;
                        result.private_run = result.private_run.max(run);
                    } else {
                        run = 0;
                    }
                }
            }
            if group.is_empty() {
                return result;
            }
            let members = ordered(data, group);
            let mut chars: Vec<(&str, Box4, usize)> = Vec::with_capacity(members.len());
            let mut cursor = 0;
            while cursor < members.len() {
                if cursor + 1 < members.len() {
                    if let Some((base, modifier)) = composition_pair(
                        data,
                        members[cursor],
                        members[cursor + 1],
                        modifiers,
                        *threshold,
                    ) {
                        let a = &data.chars[base];
                        let b = &data.chars[modifier];
                        if let Some(composed) = compositions.get(&(a.text.clone(), b.text.clone()))
                        {
                            chars.push((composed, a.bbox, base));
                            cursor += 2;
                            continue;
                        }
                    }
                }
                let ch = &data.chars[members[cursor]];
                chars.push((&ch.text, ch.bbox, members[cursor]));
                cursor += 1;
            }
            let mut widths: Vec<_> = chars.iter().map(|(_, b, _)| b[2] - b[0]).collect();
            widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if widths.len() % 2 == 1 {
                widths[widths.len() / 2]
            } else {
                (widths[widths.len() / 2 - 1] + widths[widths.len() / 2]) / 2.0
            };
            let mut text = String::new();
            let spacing_chars: Vec<_> = chars
                .iter()
                .map(|(_, _, source)| &data.chars[*source])
                .collect();
            for (index, &(value, bbox, source)) in chars.iter().enumerate() {
                if breaks.contains(value) {
                    continue;
                }
                if tight_spacing.get(group_index).copied().unwrap_or(false)
                    && index > 0
                    && index + 1 < chars.len()
                    && (crate::text_spacing::generated_cjk_space(
                        &data.chars[chars[index - 1].2],
                        &data.chars[source],
                        &data.chars[chars[index + 1].2],
                    ) || crate::text_spacing::ligature_expansion_space(&spacing_chars, index))
                {
                    continue;
                }
                text.push_str(value);
                if let Some(&(next, next_box, next_source)) = chars.get(index + 1) {
                    if (next_box[0] - bbox[2] > median * 0.25
                        || (tight_spacing.get(group_index).copied().unwrap_or(false)
                            && crate::text_spacing::needs_space(
                                data,
                                source,
                                next_source,
                                ordinary,
                                decimals,
                            )))
                        && value != " "
                        && next != " "
                    {
                        text.push(' ');
                    }
                }
            }
            result.text = Some(clean(&text));
            result
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_snapshot::{Character, Font};

    /// 三个私用区参与同一连续段统计，空白中断连续段，原始文本及边界外字符保持不变。
    #[test]
    fn supplementary_private_ranges_preserve_text() {
        let text = "A\u{e000}\u{f0000} \u{1001b0}B\u{f900}\u{ffffe}\u{20000}";
        let data = TextSnapshot {
            chars: text
                .chars()
                .enumerate()
                .map(|(index, ch)| Character {
                    text: ch.to_string(),
                    bbox: [index as f64 * 5.0, 0.0, index as f64 * 5.0 + 5.0, 10.0],
                    rotation: 0.0,
                    font: 0,
                    index,
                    sources: vec![index],
                    code: ch as u32,
                    generated: Some(false),
                    object: Some(0),
                    mode: Some(0),
                    writing_angle: 0.0,
                    origin: None,
                    loose: None,
                    tight: None,
                    visible: Some(true),
                })
                .collect(),
            fonts: vec![Font {
                name: "TestFont".into(),
                name_id: 0,
                flags: 0,
                size: 10.0,
                weight: 400,
            }],
            extended: true,
            visible_only: false,
        };
        let rules = Rules {
            ordinary: &HashSet::new(),
            decimals: &HashSet::new(),
            tight_spacing: &[false],
            modifiers: &HashMap::new(),
            threshold: 0.5,
            compositions: &HashMap::new(),
            spaces: &HashSet::from([' ']),
            breaks: &HashSet::new(),
            private_ranges: &[(0xe000, 0xf8ff), (0xf0000, 0xffffd), (0x100000, 0x10fffd)],
        };
        let result = build(&data, &[(0..data.chars.len()).collect()], &rules);
        assert_eq!(result[0].text.as_deref(), Some(text));
        assert_eq!(result[0].text_count, 8);
        assert_eq!(result[0].private_count, 3);
        assert_eq!(result[0].private_run, 2);
    }
}
