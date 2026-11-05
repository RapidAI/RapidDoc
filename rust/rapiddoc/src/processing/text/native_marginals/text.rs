use super::*;
use rapiddoc_native::{python_decimal, text_boundary::python_alnum};
use regex::Regex;
use std::{collections::BTreeMap, sync::OnceLock};
use unicode_normalization::UnicodeNormalization;
const SPACE: &str = r"[\u{9}-\u{d}\u{1c}-\u{20}\u{85}\u{a0}\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}]";
fn decimals(s: &str) -> String {
    s.chars()
        .map(|c| python_decimal(c).map_or(c, |d| (b'0' + d) as char))
        .collect()
}
fn ignorecase(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'İ' | 'ı' => 'i',
            'ſ' => 's',
            'K' => 'k',
            _ => c.to_ascii_lowercase(),
        })
        .collect()
}
fn canonical(s: &str) -> String {
    let value = s.trim_start_matches('0');
    if value.is_empty() {
        "0".into()
    } else {
        value.into()
    }
}
pub(super) fn increment(s: &str, n: u8) -> String {
    let mut b = s.as_bytes().to_vec();
    let mut carry = n;
    for v in b.iter_mut().rev() {
        let d = *v - b'0' + carry;
        *v = b'0' + d % 10;
        carry = d / 10;
        if carry == 0 {
            break;
        }
    }
    if carry > 0 {
        b.insert(0, b'0' + carry)
    }
    String::from_utf8(b).unwrap()
}
pub(super) fn number(s: &str, w: &mut Work) -> Result<Option<String>, Error> {
    w.charge(s.len() + 1)?;
    static RE: OnceLock<Regex> = OnceLock::new();
    let s: String = s.nfkc().collect();
    // IGNORECASE admission preserves the captured Roman characters: Python's
    // later upper() rejects dotted capital I while accepting dotless small i.
    let s = decimals(
        &s.chars()
            .map(|c| c.to_ascii_lowercase())
            .collect::<String>(),
    );
    let token = r"(?:[0-9]{1,4}|[ivxlcdmİı]+|[〇零一二三四五六七八九十百两]+)";
    let re=RE.get_or_init(||Regex::new(&format!(r"^{SPACE}*(?:page{SPACE}*)?[–—·•-]*{SPACE}*(?:第{SPACE}*)?(?P<value>{token})(?:{SPACE}*(?:/|of|共){SPACE}*{token})?{SPACE}*页?{SPACE}*[–—·•-]*{SPACE}*$")).unwrap());
    let Some(c) = re.captures(&s) else {
        return Ok(None);
    };
    let value = &c["value"];
    if value.bytes().all(|b| b.is_ascii_digit()) {
        return Ok(Some(canonical(value)));
    }
    if value.chars().all(|c| "ivxlcdmİı".contains(c)) {
        if value.contains('İ') {
            return Ok(None);
        }
        let mut total = 0i64;
        let mut previous = 0;
        for c in value.chars().rev() {
            let n = match c {
                'i' | 'ı' => 1,
                'v' => 5,
                'x' => 10,
                'l' => 50,
                'c' => 100,
                'd' => 500,
                _ => rapiddoc_core::resources::limit(1000),
            };
            total += if n < previous { -n } else { n };
            previous = previous.max(n)
        }
        return Ok((1..=4999).contains(&total).then(|| total.to_string()));
    }
    let digit = |c: char| match c {
        '〇' | '零' => Some(0),
        '一' => Some(1),
        '二' | '两' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    };
    if value.chars().all(|c| digit(c).is_some()) {
        if value.chars().count() > 4300 {
            return Ok(None);
        }
        return Ok(Some(canonical(
            &value
                .chars()
                .map(|c| (b'0' + digit(c).unwrap()) as char)
                .collect::<String>(),
        )));
    }
    let mut total = 0usize;
    let mut current = 0;
    for c in value.chars() {
        if let Some(d) = digit(c) {
            current = d as usize
        } else if c == '十' {
            total += current.max(1) * 10;
            current = 0
        } else if c == '百' {
            total += current.max(1) * 100;
            current = 0
        } else {
            return Ok(None);
        }
    }
    Ok((total + current > 0).then(|| (total + current).to_string()))
}
pub(super) fn leader(s: &str, w: &mut Work) -> Result<bool, Error> {
    w.charge(s.len() + 1)?;
    static RE: OnceLock<Regex> = OnceLock::new();
    let s: String = s.nfkc().collect();
    let s = decimals(&s);
    let re = RE.get_or_init(|| {
        Regex::new(&format!(
            r"(?:[.·]{SPACE}*){{3,}}{SPACE}*(?:[0-9]+|[ivxlcdmIVXLCDM]+){SPACE}*$"
        ))
        .unwrap()
    });
    Ok(re.find_iter(&s).any(|m| {
        s[..m.start()]
            .rsplit('\n')
            .next()
            .is_some_and(|s| s.chars().any(|c| python_alnum(c) || c == '_'))
    }))
}
fn normalize(s: &str) -> String {
    let nfkc: String = s.nfkc().collect();
    let folded = super::super::table_text_styles::preparation::fold(&nfkc);
    let mut out = String::new();
    let mut digits = false;
    for c in folded.chars() {
        if python_decimal(c).is_some() {
            if !digits {
                out.push('#')
            }
            digits = true
        } else {
            digits = false;
            if !python_space(c) {
                out.push(c)
            }
        }
    }
    out
}
fn excluded(s: &str) -> bool {
    static CITATION: OnceLock<Regex> = OnceLock::new();
    static NUMBERED: OnceLock<Regex> = OnceLock::new();
    let s = decimals(s.trim_matches(python_space));
    CITATION.get_or_init(||Regex::new(&format!(r"^(?:(?:18|19|20)[0-9]{{2}}{SPACE}*[;,.:]|arxiv{SPACE}*:{SPACE}*[0-9]|[0-9]+{SPACE}*:{SPACE}*[0-9]+{SPACE}*[–—-]{SPACE}*[0-9]+)")).unwrap()).is_match(&ignorecase(&s))
        || NUMBERED.get_or_init(||Regex::new(&format!(r"^[（(﹙]{SPACE}*[A-Za-z]?[0-9]+(?:[.\-][0-9]+)*{SPACE}*[)）﹚]$|^[0-9]+(?:\.[0-9]+)+$")).unwrap()).is_match(&s)
}
fn longest(
    a: &[char],
    b: &[char],
    b2j: &BTreeMap<char, Vec<usize>>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
    w: &mut Work,
) -> Result<(usize, usize, usize), Error> {
    let (mut best_i, mut best_j, mut best) = (alo, blo, 0);
    let mut lengths = BTreeMap::<usize, usize>::new();
    for i in alo..ahi {
        let mut next = BTreeMap::new();
        if let Some(js) = b2j.get(&a[i]) {
            for &j in js {
                w.charge(1)?;
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let n = if j == 0 {
                    1
                } else {
                    lengths.get(&(j - 1)).copied().unwrap_or(0) + 1
                };
                next.insert(j, n);
                if n > best {
                    best_i = i + 1 - n;
                    best_j = j + 1 - n;
                    best = n
                }
            }
        }
        lengths = next
    }
    while best_i > alo && best_j > blo && a[best_i - 1] == b[best_j - 1] {
        w.charge(1)?;
        best_i -= 1;
        best_j -= 1;
        best += 1
    }
    while best_i + best < ahi && best_j + best < bhi && a[best_i + best] == b[best_j + best] {
        w.charge(1)?;
        best += 1
    }
    Ok((best_i, best_j, best))
}
pub(super) fn matches(a: &str, b: &str, w: &mut Work) -> Result<bool, Error> {
    w.charge(a.len() + b.len() + 1)?;
    if excluded(a) || excluded(b) {
        return Ok(false);
    }
    let a = normalize(a);
    let b = normalize(b);
    if a.is_empty() || b.is_empty() {
        return Ok(false);
    }
    if a == b {
        return Ok(true);
    }
    let a = a.chars().collect::<Vec<_>>();
    let b = b.chars().collect::<Vec<_>>();
    if a.len().min(b.len()) < 8 {
        return Ok(false);
    }
    let mut b2j = BTreeMap::<char, Vec<usize>>::new();
    for (i, c) in b.iter().enumerate() {
        b2j.entry(*c).or_default().push(i)
    }
    let mut queue = vec![(0, a.len(), 0, b.len())];
    let mut blocks = Vec::new();
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, n) = longest(&a, &b, &b2j, alo, ahi, blo, bhi, w)?;
        if n > 0 {
            blocks.push((i, j, n));
            if alo < i && blo < j {
                queue.push((alo, i, blo, j))
            }
            if i + n < ahi && j + n < bhi {
                queue.push((i + n, ahi, j + n, bhi))
            }
        }
    }
    let count = blocks.iter().map(|b| b.2).sum::<usize>();
    Ok(2. * count as f64 / (a.len() + b.len()) as f64 >= 0.92)
}
