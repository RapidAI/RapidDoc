//! 与 Python 参考实现相同的相邻非 CJK 字形词界判定。
use crate::{
    geometry::Box4,
    text_snapshot::{Character, TextSnapshot},
};
use std::collections::HashSet;

/// 排除汉字扩展、假名、韩文及兼容字形，不以所在整行的语言决定词界。
pub fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x1100..=0x11ff | 0x2e80..=0xa4cf | 0xa960..=0xa97f | 0xac00..=0xd7ff |
        0xf900..=0xfaff | 0xff66..=0xff9f | 0x1aff0..=0x1b16f | 0x20000..=0x323af)
}

/// 墨迹框只转换坐标轴；相邻间隙与页面原点平移无关。
fn local(b: Box4, angle: i32) -> Box4 {
    match angle {
        90 => [b[1], -b[2], b[3], -b[0]],
        180 => [-b[2], -b[3], -b[0], -b[1]],
        270 => [-b[3], b[0], -b[1], b[2]],
        _ => b,
    }
}

/// 判定单个汉字；与 Python 的中文空格物化范围保持一致。
fn han(text: &str) -> bool {
    let mut chars = text.chars();
    let Some(ch) = chars.next() else { return false };
    chars.next().is_none()
        && matches!(ch as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff | 0x20000..=0x323af)
}

/// 只省略有生成证据且位于同向同基线中文边界的空格，不删除快照成员。
pub fn generated_cjk_space(a: &Character, space: &Character, b: &Character) -> bool {
    const PUNCTUATION: &str = "、。，；：？！“”‘’（）【】《》〈〉「」『』〔〕［］｛｝—…·";
    let punctuation = |text: &str| text.chars().count() == 1 && PUNCTUATION.contains(text);
    if space.text != " "
        || space.generated != Some(true)
        || a.index.checked_add(1) != Some(space.index)
        || space.index.checked_add(1) != Some(b.index)
        || !((han(&a.text) && (han(&b.text) || punctuation(&b.text)))
            || (punctuation(&a.text) && han(&b.text)))
        || !a.writing_angle.is_finite()
        || !b.writing_angle.is_finite()
    {
        return false;
    }
    let degrees = a.writing_angle.to_degrees().rem_euclid(360.0);
    let angle = ((degrees / 90.0).round_ties_even() as i32 * 90) % 360;
    if ((degrees - angle as f64 + 180.0).rem_euclid(360.0) - 180.0).abs() > 0.1
        || ((b.writing_angle.to_degrees() - angle as f64 + 180.0).rem_euclid(360.0) - 180.0).abs()
            > 0.1
    {
        return false;
    }
    let ab = a.tight.unwrap_or(a.bbox);
    let bb = b.tight.unwrap_or(b.bbox);
    if ab.iter().chain(bb.iter()).any(|v| !v.is_finite()) {
        return false;
    }
    let (ab, bb) = (local(ab, angle), local(bb, angle));
    let (ah, bh) = (ab[3] - ab[1], bb[3] - bb[1]);
    if ah.min(bh).min(ab[2] - ab[0]).min(bb[2] - bb[0]) <= 0.0
        || ab[3].min(bb[3]) - ab[1].max(bb[1]) < 0.5 * ah.min(bh)
        || bb[0] - ab[2] > 0.5 * ah.max(bh)
    {
        return false;
    }
    let tolerance = 0.15 * ah.max(bh);
    if let (Some(ao), Some(bo)) = (a.origin, b.origin) {
        let axis = usize::from(angle != 90 && angle != 270);
        ao.iter().chain(bo.iter()).all(|v| v.is_finite())
            && (ao[axis] - bo[axis]).abs() <= tolerance
    } else {
        (ab[3] - bb[3]).abs() <= tolerance
    }
}

/// 仅忽略同一拉丁连字字形内部的空格，源编号、字体、对象与几何证据均需一致。
pub fn ligature_expansion_space(chars: &[&Character], index: usize) -> bool {
    if index == 0 || index + 1 >= chars.len() || chars[index].text != " " {
        return false;
    }
    let right = chars[index + 1];
    if right.text.len() != 1 || !right.text.as_bytes()[0].is_ascii_alphabetic() {
        return false;
    }
    for count in 1..=index.min(3) {
        let glyph = &chars[index - count..index];
        let text: String = glyph.iter().map(|ch| ch.text.as_str()).collect();
        if !matches!(
            text.as_str(),
            "fi" | "fl"
                | "ff"
                | "ffi"
                | "ffl"
                | "ft"
                | "st"
                | "ﬁ"
                | "ﬂ"
                | "ﬀ"
                | "ﬃ"
                | "ﬄ"
                | "ﬅ"
                | "ﬆ"
        ) {
            continue;
        }
        let first = glyph[0];
        let records = &chars[index - count..=index + 1];
        if first.object.is_none()
            || records
                .windows(2)
                .any(|pair| pair[0].index.checked_add(1) != Some(pair[1].index))
            || records
                .iter()
                .any(|ch| ch.object != first.object || ch.font != first.font)
        {
            continue;
        }
        let degrees = first.writing_angle.to_degrees().rem_euclid(360.0);
        let angle = ((degrees / 90.0).round_ties_even() as i32 * 90) % 360;
        if records.iter().any(|ch| {
            !ch.writing_angle.is_finite()
                || ((ch.writing_angle.to_degrees() - angle as f64 + 180.0).rem_euclid(360.0)
                    - 180.0)
                    .abs()
                    > 0.1
                || ch
                    .tight
                    .is_none_or(|bbox| bbox.iter().any(|v| !v.is_finite()))
                || ch
                    .origin
                    .is_none_or(|point| point.iter().any(|v| !v.is_finite()))
        }) {
            continue;
        }
        let boxes: Vec<_> = records
            .iter()
            .map(|ch| local(ch.tight.unwrap(), angle))
            .collect();
        let points: Vec<_> = records
            .iter()
            .map(|ch| {
                let p = ch.origin.unwrap();
                let p = local([p[0], p[1], p[0], p[1]], angle);
                [p[0], p[1]]
            })
            .collect();
        let bbox = boxes[0];
        let height = bbox[3] - bbox[1];
        if height.min(bbox[2] - bbox[0]) <= 0.0
            || (1..count).any(|i| {
                (0..4).any(|j| (bbox[j] - boxes[i][j]).abs() > 0.01 * height)
                    || (0..2).any(|j| (points[0][j] - points[i][j]).abs() > 0.01 * height)
            })
        {
            continue;
        }
        let next = boxes[count + 1];
        let gap = next[0] - bbox[2];
        if bbox[0] <= points[count][0]
            && points[count][0] < bbox[2]
            && points[count..]
                .iter()
                .all(|p| (p[1] - points[0][1]).abs() <= 0.15 * height)
            && gap >= -0.15 * height
            && gap <= 0.15 * height
            && (points[count + 1][0] - bbox[2]).abs() <= 0.15 * height
            && next[3] > next[1]
            && bbox[3].min(next[3]) - bbox[1].max(next[1]) >= 0.5 * height.min(next[3] - next[1])
        {
            return true;
        }
    }
    false
}

/// 只接受解释器确认的普通单字形，避免两端标点或连字展开改变语义。
fn ordinary(text: &str, letters: &HashSet<char>) -> bool {
    let mut chars = text.chars();
    let Some(ch) = chars.next() else { return false };
    chars.next().is_none() && !is_cjk(ch) && (ch.is_ascii_alphanumeric() || letters.contains(&ch))
}

/// 使用真实源编号、方向和基线保护，只为可靠的大墨迹间隙返回一个词界。
pub fn needs_space(
    data: &TextSnapshot,
    first: usize,
    second: usize,
    letters: &HashSet<char>,
    decimals: &HashSet<char>,
) -> bool {
    let a = &data.chars[first];
    let b = &data.chars[second];
    if !ordinary(&a.text, letters)
        || !ordinary(&b.text, letters)
        || a.index.checked_add(1) != Some(b.index)
    {
        return false;
    }
    if a.text
        .chars()
        .all(|c| c.is_ascii_digit() || decimals.contains(&c))
        && b.text
            .chars()
            .all(|c| c.is_ascii_digit() || decimals.contains(&c))
    {
        return false;
    }
    for font in [&data.fonts[a.font], &data.fonts[b.font]] {
        let name = font.name.to_lowercase();
        if font.flags & 1 != 0
            || [
                "mono",
                "courier",
                "consolas",
                "menlo",
                "typewriter",
                "fixed",
            ]
            .iter()
            .any(|hint| name.contains(hint))
        {
            return false;
        }
    }
    let ls = data.fonts[a.font].size;
    let rs = data.fonts[b.font].size;
    let em = ls.max(rs);
    if !ls.is_finite() || !rs.is_finite() || ls.min(rs) <= 0.0 || ls.min(rs) < 0.8 * em {
        return false;
    }
    if !a.writing_angle.is_finite() || !b.writing_angle.is_finite() {
        return false;
    }
    let degrees = a.writing_angle.to_degrees().rem_euclid(360.0);
    let angle = ((degrees / 90.0).round_ties_even() as i32 * 90) % 360;
    if ((degrees - angle as f64 + 180.0).rem_euclid(360.0) - 180.0).abs() > 0.1
        || ((b.writing_angle.to_degrees() - angle as f64 + 180.0).rem_euclid(360.0) - 180.0).abs()
            > 0.1
    {
        return false;
    }
    let (Some(ab), Some(bb)) = (a.tight, b.tight) else {
        return false;
    };
    if ab.iter().chain(bb.iter()).any(|v| !v.is_finite()) {
        return false;
    }
    let ab = local(ab, angle);
    let bb = local(bb, angle);
    if ab[2] <= ab[0]
        || bb[2] <= bb[0]
        || ab[3] <= ab[1]
        || bb[3] <= bb[1]
        || ab[3] - ab[1] > 1.5 * ls
        || bb[3] - bb[1] > 1.5 * rs
        || bb[0] - ab[2] <= 0.25 * em
        || ab[3].min(bb[3]) - ab[1].max(bb[1]) < 0.5 * (ab[3] - ab[1]).min(bb[3] - bb[1])
    {
        return false;
    }
    if let (Some(ao), Some(bo)) = (a.origin, b.origin) {
        let axis = usize::from(angle != 90 && angle != 270);
        if ao.iter().chain(bo.iter()).any(|v| !v.is_finite())
            || (ao[axis] - bo[axis]).abs() > 0.15 * em
        {
            return false;
        }
    } else if (ab[3] - bb[3]).abs() > 0.15 * em {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 建立独立字符记录，以页面字形高度验证生成空格而非源字号。
    fn character(text: &str, index: usize, angle: i32) -> Character {
        let x = index as f64 * 6.0;
        let bbox = local([x, 0.0, x + 10.0, 10.0], (360 - angle) % 360);
        let point = local([x, 10.0, x, 10.0], (360 - angle) % 360);
        Character {
            text: text.into(),
            bbox,
            rotation: 0.0,
            font: 0,
            index,
            sources: vec![index],
            code: text.chars().next().unwrap() as u32,
            generated: Some(text == " "),
            object: None,
            mode: None,
            writing_angle: (angle as f64).to_radians(),
            origin: Some([point[0], point[1]]),
            loose: None,
            tight: Some(bbox),
            visible: None,
        }
    }

    /// 四向连字使用同一墨迹框；真实词界、不同来源或字号缩放不应导致过删。
    #[test]
    fn ligature_internal_spaces_require_complete_evidence() {
        for angle in [0, 90, 180, 270] {
            let mut chars: Vec<_> = ["f", "i", " ", "n"]
                .iter()
                .enumerate()
                .map(|(index, text)| character(text, index, angle))
                .collect();
            for (index, ch) in chars.iter_mut().enumerate() {
                let (bbox, origin) = match index {
                    2 => ([12.0, 19.99, 15.0, 20.0], [12.0, 20.0]),
                    3 => ([15.3, 10.0, 20.0, 20.0], [15.2, 20.0]),
                    _ => ([10.0, 10.0, 15.0, 20.0], [10.0, 20.0]),
                };
                ch.tight = Some(local(bbox, (360 - angle) % 360));
                let p = local(
                    [origin[0], origin[1], origin[0], origin[1]],
                    (360 - angle) % 360,
                );
                ch.origin = Some([p[0], p[1]]);
                ch.object = Some(1);
                ch.generated = Some(false);
            }
            assert!(ligature_expansion_space(
                &chars.iter().collect::<Vec<_>>(),
                2
            ));
            chars[0].object = Some(2);
            assert!(!ligature_expansion_space(
                &chars.iter().collect::<Vec<_>>(),
                2
            ));
            chars[0].object = Some(1);
            chars[3].tight = Some(local([18.0, 10.0, 23.0, 20.0], (360 - angle) % 360));
            assert!(!ligature_expansion_space(
                &chars.iter().collect::<Vec<_>>(),
                2
            ));
        }
    }

    #[test]
    /// 四个方向和中文标点允许过滤，真实或未知空格、英文词界、异基线保持原样。
    fn generated_spaces_require_chinese_and_baseline_evidence() {
        for angle in [0, 90, 180, 270] {
            let a = character("卫", 0, angle);
            let mut space = character(" ", 1, angle);
            let mut b = character("星", 2, angle);
            assert!(generated_cjk_space(&a, &space, &b));
            b.text = "、".into();
            assert!(generated_cjk_space(&a, &space, &b));
            b.text = "A".into();
            assert!(!generated_cjk_space(&a, &space, &b));
            b.text = "星".into();
            for value in [Some(false), None] {
                space.generated = value;
                assert!(!generated_cjk_space(&a, &space, &b));
            }
            space.generated = Some(true);
            b.writing_angle += 0.3;
            assert!(!generated_cjk_space(&a, &space, &b));
        }
        let a = character("卫", 0, 0);
        let space = character(" ", 1, 0);
        let mut b = character("星", 2, 0);
        b.origin = Some([24.0, 30.0]);
        assert!(!generated_cjk_space(&a, &space, &b));
        b.origin = None;
        assert!(generated_cjk_space(&a, &space, &b));
        b.index = 3;
        assert!(!generated_cjk_space(&a, &space, &b));
    }
}
