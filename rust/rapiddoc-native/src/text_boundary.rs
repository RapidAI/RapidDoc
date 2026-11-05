//! 移植 foundation/_text.py 的物理行连接规则；仅接受有效 Unicode，保留原始行供上层审计。
use crate::error;
use rapiddoc_core::{Error, ErrorCode};

fn flags(c: char) -> u16 {
    let n = c as u32;
    let values = super::text_unicode_data::RANGES;
    let i = values.partition_point(|range| range.0 <= n);
    if i > 0 && n <= values[i - 1].1 {
        values[i - 1].2
    } else {
        0
    }
}
/// 查询固定 Unicode 16.0.0 的 Python isspace 属性，不依赖宿主 Python。
pub fn python_space(c: char) -> bool {
    flags(c) & 1 != 0
}
/// 查询固定 Unicode 16.0.0 的 Python isdigit 属性，包含非十进制数字。
pub fn python_digit(c: char) -> bool {
    flags(c) & 4 != 0
}
pub fn python_alpha(c: char) -> bool {
    flags(c) & 512 != 0
}
pub fn python_alnum(c: char) -> bool {
    flags(c) & 128 != 0
}
/// 查询固定 Unicode 16.0.0 的单字符 Python isupper 属性。
pub fn python_upper(c: char) -> bool {
    flags(c) & 256 != 0
}
pub fn python_lower(c: char) -> bool {
    flags(c) & 2 != 0
}
/// 逐字母投票移植 uses_unspaced_script，恰好一半 CJK 时也使用无空格阈值。
pub fn uses_unspaced_script(text: &str) -> bool {
    let mut letters = 0;
    let mut cjk = 0;
    for c in text.chars().filter(|c| flags(*c) & 512 != 0) {
        letters += 1;
        cjk += usize::from(unspaced(c));
    }
    letters > 0 && cjk * 2 >= letters
}
fn unspaced(c: char) -> bool {
    "，。！？；：、（）【】《》〈〉「」『』〔〕［］｛｝".contains(c)
        || matches!(c as u32,0x3000..=0x303f|0x3040..=0x30ff|0x31f0..=0x31ff|0x3400..=0x4dbf|
            0x4e00..=0x9fff|0xf900..=0xfaff|0xff61..=0xff9f|0x1aff0..=0x1afff|0x1b000..=0x1b16f|
            0x20000..=0x2ee5f|0x2f800..=0x2fa1f|0x30000..=0x323af)
}
fn fold(c: char) -> char {
    match c {
        'İ' | 'ı' => 'i',
        'ſ' => 's',
        'K' => 'k',
        _ => c.to_ascii_lowercase(),
    }
}
fn visible(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let final_close = text.rfind('>');
    let mut i = 0;
    while i < text.len() {
        let rest = &text[i..];
        if let Some(after) = rest.strip_prefix('<') {
            let after = after.strip_prefix('/').unwrap_or(after);
            let mut name = String::new();
            let mut length = 0;
            for c in after.chars() {
                let c_folded = fold(c);
                if !c_folded.is_ascii_alphabetic() {
                    break;
                }
                name.push(c_folded);
                length += c.len_utf8();
            }
            let word_boundary = after[length..]
                .chars()
                .next()
                .is_none_or(|c| flags(c) & 128 == 0 && c != '_');
            let end = if word_boundary
                && final_close.is_some_and(|end| i <= end)
                && [
                    "sup", "sub", "b", "strong", "i", "em", "u", "s", "del", "strike",
                ]
                .contains(&name.as_str())
            {
                rest.find('>')
            } else {
                None
            };
            if let Some(end) = end {
                i += end + 1;
                continue;
            }
        }
        let c = rest.chars().next().unwrap();
        output.push(c);
        i += c.len_utf8();
    }
    output
}
fn boundary(text: &str, end: bool) -> Option<char> {
    let value = visible(text);
    let mut chars = value
        .trim_matches(python_space)
        .chars()
        .filter(|c| flags(*c) & 16 == 0);
    if end {
        chars.next_back()
    } else {
        chars.next()
    }
}
fn url_prefix(s: &[u8]) -> Option<usize> {
    [b"http://".as_slice(), b"https://", b"ftp://", b"www."]
        .iter()
        .find(|prefix| s.len() >= prefix.len() && s[..prefix.len()].eq_ignore_ascii_case(prefix))
        .map(|prefix| prefix.len())
}
fn url_start(s: &[u8]) -> bool {
    url_prefix(s).is_some()
}
fn url_crosses(previous: &str, next: &str) -> bool {
    let candidate = format!("{previous}{next}");
    let bytes = candidate.as_bytes();
    let boundary = previous.len();
    let mut i = 0;
    while i < bytes.len() {
        if (i == 0 || !bytes[i - 1].is_ascii_alphanumeric()) && url_start(&bytes[i..]) {
            let start = i;
            let suffix = i + url_prefix(&bytes[i..]).unwrap();
            i = suffix;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric()
                    || b"._~:/?#[]@!$&'()*+,;=%-".contains(&bytes[i]))
            {
                i += 1;
            }
            if i > suffix && start < boundary && boundary < i {
                return true;
            }
        } else {
            i += 1;
        }
    }
    false
}
/// 已校验的输出层复用同一旧边界规则；保留断词后的前文和分隔符。
pub fn resolve(previous: &str, next: &str) -> (String, &'static str) {
    let previous = previous.trim_end_matches(python_space);
    let next = next.trim_start_matches(python_space);
    if previous.is_empty() || next.is_empty() {
        return (previous.into(), "");
    }
    if url_crosses(previous, next) {
        return (
            previous.into(),
            if url_start(next.as_bytes()) { " " } else { "" },
        );
    }
    let last = previous.chars().next_back().unwrap();
    if "-\u{ad}\u{2010}\u{2011}\u{2043}".contains(last) {
        let word_end = previous.len() - last.len_utf8();
        let mut start = word_end;
        for (index, c) in previous[..word_end].char_indices().rev() {
            if !c.is_ascii_alphabetic() {
                break;
            }
            start = index;
        }
        if start < word_end {
            let word = &previous[start..word_end];
            let lower = word.to_ascii_lowercase();
            let acronym = word.len() > 1 && word.bytes().all(|b| b.is_ascii_uppercase());
            let compound = last == '-'
                && ([
                    "open", "non", "self", "cross", "anti", "pre", "post", "co", "semi", "multi",
                    "quasi",
                ]
                .contains(&lower.as_str())
                    || (["of", "the", "to", "and", "in", "for", "on", "by", "with"]
                        .contains(&lower.as_str())
                        && previous[..start].ends_with('-')));
            if flags(next.chars().next().unwrap()) & 2 != 0 && !acronym && !compound {
                return (previous[..word_end].into(), "");
            }
            return (previous.into(), "");
        }
    }
    if matches!(last, '-' | '–')
        && previous
            .chars()
            .rev()
            .nth(1)
            .is_some_and(|c| flags(c) & 8 != 0)
        && python_digit(next.chars().next().unwrap())
    {
        return (previous.into(), "");
    }
    let (Some(a), Some(b)) = (boundary(previous, true), boundary(next, false)) else {
        return (previous.into(), "");
    };
    let separator = if unspaced(a)
        || unspaced(b)
        || flags(a) & 32 != 0
        || flags(b) & 64 != 0
        || ",.;:!?%".contains(b)
    {
        ""
    } else {
        " "
    };
    (previous.into(), separator)
}
/// 使用累计上下文连接，保留横跨三行以上的 URL；先限制最坏扫描工作量。
pub fn merge(lines: &[String]) -> Result<String, Error> {
    if lines.len() > rapiddoc_core::resources::limit(1000) {
        return Err(error(ErrorCode::ResourceLimit, "文本连接行数超过预算"));
    }
    let lines: Vec<_> = lines
        .iter()
        .map(|s| s.trim_matches(python_space))
        .filter(|s| !s.is_empty())
        .collect();
    let mut bytes = 0usize;
    let mut work = 0usize;
    for line in &lines {
        bytes += line.len();
        work += bytes;
        if bytes > rapiddoc_core::resources::limit(1_000_000)
            || work > rapiddoc_core::resources::limit(32_000_000)
        {
            return Err(error(
                ErrorCode::ResourceLimit,
                "文本连接字节或累计扫描超过预算",
            ));
        }
    }
    let mut result = String::new();
    for line in lines {
        if result.is_empty() {
            result = line.into();
            continue;
        }
        let (adjusted, separator) = resolve(&result, line);
        result = adjusted;
        result.push_str(separator);
        result.push_str(line);
    }
    Ok(result.trim_matches(python_space).into())
}
