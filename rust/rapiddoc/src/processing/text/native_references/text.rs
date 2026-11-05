//! Python Unicode whitespace/digits/word-boundary semantics without changing text.
use rapiddoc_native::{
    python_decimal,
    text_boundary::{python_alnum, python_space},
};
use unicode_normalization::UnicodeNormalization;
fn fold(c: char) -> char {
    match c {
        'İ' | 'ı' => 'i',
        'ſ' => 's',
        'K' => 'k',
        _ => c.to_ascii_lowercase(),
    }
}
pub(super) fn heading(t: &str) -> bool {
    let t = t
        .chars()
        .filter(|&c| !python_space(c))
        .map(fold)
        .collect::<String>();
    let t = t
        .trim_end_matches([':', '：'])
        .trim_matches(|c| "[]［］【】()（）".contains(c));
    t == "references"
        || t == "bibliography"
        || t == "参考文献"
        || t.strip_prefix("参考文献").is_some_and(|r| {
            r == "(references)"
                || r == "（references)"
                || r == "(references）"
                || r == "（references）"
        })
}
pub(super) fn number(t: &str) -> Option<(u32, bool)> {
    let cs = t
        .trim_start_matches(python_space)
        .chars()
        .collect::<Vec<_>>();
    let bracket = cs.first().is_some_and(|c| matches!(c, '[' | '［'));
    let start = usize::from(bracket);
    let max = if bracket { 4 } else { 3 };
    let mut n = 0;
    let mut end = start;
    while end < cs.len() && end - start < max {
        let Some(d) = python_decimal(cs[end]) else {
            break;
        };
        n = n * 10 + d as u32;
        end += 1
    }
    if end == start || end >= cs.len() {
        return None;
    }
    if bracket {
        if !matches!(cs[end], ']' | '］') {
            return None;
        }
    } else if cs[end] != '.' {
        return None;
    }
    end += 1;
    if end == cs.len() || python_decimal(cs[end]).is_some() {
        return None;
    }
    Some((n, bracket))
}
pub(super) fn appendix(t: &str) -> bool {
    let s = t
        .trim_start_matches(python_space)
        .chars()
        .map(fold)
        .collect::<String>();
    s.starts_with("附录")
        || s.strip_prefix("appendix")
            .is_some_and(|r| r.chars().next().is_none_or(|c| !word(c)))
}
fn word(c: char) -> bool {
    python_alnum(c) || c == '_'
}
pub(super) fn abstract_line(t: &str) -> bool {
    let t = t.nfkc().map(fold).collect::<String>();
    let t = t.trim_start_matches(python_space);
    t.strip_prefix("abstract")
        .or_else(|| t.strip_prefix("摘要"))
        .is_some_and(|r| r.trim_start_matches(python_space).starts_with([':', '：']))
}
pub(super) fn alpha_count(t: &str) -> usize {
    t.nfkc().filter(|c| c.is_ascii_alphabetic()).count()
}
pub(super) fn author(t: &str) -> bool {
    let mut cs = t.chars();
    if !cs.next().is_some_and(|c| c.is_ascii_uppercase()) {
        return false;
    }
    let remaining = cs.collect::<String>();
    let Some(comma) = remaining.find(',') else {
        return false;
    };
    if comma == 0 {
        return false;
    }
    remaining[comma + 1..]
        .trim_start_matches(python_space)
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_uppercase())
}
pub(super) fn year(t: &str) -> bool {
    let cs = t.chars().collect::<Vec<_>>();
    for i in 0..cs.len() {
        if i > 0 && word(cs[i - 1]) {
            continue;
        }
        if i + 4 >= cs.len() {
            continue;
        }
        if !(cs[i] == '1' && cs[i + 1] == '9' || cs[i] == '2' && cs[i + 1] == '0')
            || python_decimal(cs[i + 2]).is_none()
            || python_decimal(cs[i + 3]).is_none()
        {
            continue;
        }
        let mut j = i + 4;
        if cs[j].is_ascii_lowercase() {
            j += 1
        }
        if cs.get(j) == Some(&'.') {
            return true;
        }
    }
    false
}
pub(super) fn biography(t: &str) -> bool {
    let cs = t.chars().collect::<Vec<_>>();
    let mut birth = false;
    for i in 0..cs.len() {
        if !matches!(cs[i], '(' | '（') {
            continue;
        }
        let mut j = i + 1;
        while cs.get(j).is_some_and(|&c| python_space(c)) {
            j += 1
        }
        for _ in 0..4 {
            if cs.get(j).is_some_and(|&c| python_decimal(c).is_some()) {
                j += 1
            } else {
                j = cs.len();
                break;
            }
        }
        while cs.get(j).is_some_and(|&c| python_space(c)) {
            j += 1
        }
        if cs.get(j).is_some_and(|c| matches!(c, '—' | '–' | '-')) {
            birth = true;
            break;
        }
    }
    let folded = t.chars().map(fold).collect::<String>();
    if birth
        && ["研究", "博士", "本科", "教授", "research", "born"]
            .iter()
            .any(|s| folded.contains(s))
    {
        return true;
    }
    let clean = t
        .trim_matches(python_space)
        .chars()
        .map(fold)
        .collect::<String>();
    let mapped = clean
        .chars()
        .map(|c| if python_space(c) { ' ' } else { c })
        .collect::<String>();
    let re = constant_regex!(
        r"^(?:publisher[’']s +note|author +(?:information|contributions)|competing +interests)",
    );
    if re
        .find(&mapped)
        .is_some_and(|m| mapped[m.end()..].chars().next().is_none_or(|c| !word(c)))
    {
        return true;
    }
    let mut t = t.trim_start_matches(python_space);
    if t.starts_with(['［', '[']) {
        t = &t[t.chars().next().unwrap().len_utf8()..]
    }
    let t = t.trim_start_matches(python_space);
    ["责任编辑", "编者", "编辑说明"].iter().any(|label| {
        t.strip_prefix(label)
            .is_some_and(|r| r.trim_start_matches(python_space).starts_with([':', '：']))
    })
}
pub(super) fn alnum(t: &str) -> bool {
    t.chars().any(python_alnum)
}
