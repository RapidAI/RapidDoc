//! 旧公式编号、数学符号及 publication/prose 字符串证据。
use rapiddoc_native::text_boundary::{python_alnum, python_space};
use regex::Regex;
use serde_json::Value;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;
fn data() -> &'static Value {
    static D: OnceLock<Value> = OnceLock::new();
    D.get_or_init(|| serde_json::from_str(include_str!("unicode.json")).unwrap())
}
fn digits() -> String {
    data()["decimal"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            format!(
                "\\u{{{:x}}}-\\u{{{:x}}}",
                r[0].as_u64().unwrap(),
                r[1].as_u64().unwrap()
            )
        })
        .collect()
}
fn space() -> &'static str {
    r"[\u{9}-\u{d}\u{1c}-\u{20}\u{85}\u{a0}\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}]"
}
pub(super) fn number(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(&format!(
            r"^[（(﹙]{s}*(?:(?:[A-Za-z][.]?)?[{d}]+(?:[.\-][{d}]+)*){s}*[)）﹚]$",
            s = space(),
            d = digits()
        ))
        .unwrap()
    })
    .is_match(text.trim_matches(python_space))
}
pub(super) fn standalone(text: &str) -> bool {
    let text = text.trim_matches(python_space);
    let c: Vec<_> = text.chars().collect();
    c.len() >= 3
        && "(（﹙".contains(c[0])
        && ")）﹚".contains(*c.last().unwrap())
        && !c[1..c.len() - 1]
            .iter()
            .any(|c| "()（）﹙﹚\r\n".contains(*c))
}
pub(super) fn math(text: &str) -> bool {
    text.chars().any(|c| {
        if "=<>≤≥≠≈∝∈∑∫√±×÷−".contains(c) {
            return true;
        }
        let n = c as u64;
        data()["math"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| n >= r[0].as_u64().unwrap() && n <= r[1].as_u64().unwrap())
    })
}
fn word(c: char) -> bool {
    c == '_' || python_alnum(c)
}
fn boundary(text: &str, start: usize, end: usize) -> bool {
    !text[..start].chars().next_back().is_some_and(word)
        && !text[end..].chars().next().is_some_and(word)
}
pub(super) fn english_words(text: &str) -> usize {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new("[A-Za-z]{2,}").unwrap())
        .find_iter(text)
        .filter(|m| boundary(text, m.start(), m.end()))
        .count()
}
pub(super) fn publication(text: &str) -> bool {
    // Python IGNORECASE additionally folds these four codepoints for ASCII letters.
    let folded: String = text
        .chars()
        .map(|c| match c {
            'İ' | 'ı' => 'i',
            'ſ' => 's',
            'K' => 'k',
            _ => c.to_ascii_lowercase(),
        })
        .collect();
    let bounded = |phrase: &str| {
        folded
            .match_indices(phrase)
            .any(|(i, _)| boundary(&folded, i, i + phrase.len()))
    };
    folded.contains("copyright")
        || folded.contains('©')
        || ["journal", "issn", "doi"].iter().any(|p| bounded(p))
        || folded.contains("出版")
        || folded.contains("版权所有")
        || text.split(python_space).filter(|s| !s.is_empty()).count() <= 4
            && ([
                "review",
                "editorial",
                "letter",
                "research article",
                "original article",
            ]
            .iter()
            .any(|p| bounded(p))
                || folded.contains("综述")
                || folded.contains("研究论文"))
}
pub(super) fn prose(text: &str) -> bool {
    static EXPR: OnceLock<Regex> = OnceLock::new();
    static CALL: OnceLock<Regex> = OnceLock::new();
    static WORDS: OnceLock<Regex> = OnceLock::new();
    let identifier =
        r"[A-Za-z\u{c0}-\u{2af}\u{370}-\u{3ff}][A-Za-z0-9_\u{c0}-\u{2af}\u{370}-\u{3ff}]*";
    let normalized: String = text.nfkc().collect();
    let expression = EXPR.get_or_init(|| {
        let s = space();
        let d = digits();
        let atom =
            format!(r"(?:{identifier}|[{d}]+(?:\.[{d}]+)?)(?:{s}*[(\[{{][^()\[\]{{}}\n]*[)\]}}])*");
        Regex::new(&format!(r"{atom}(?:{s}*[+−\-*/=<>≤≥≠≈∝∈×÷^_]{s}*{atom})+")).unwrap()
    });
    let residue = expression.replace_all(&normalized, " ");
    let call =
        CALL.get_or_init(|| Regex::new(&format!(r"{identifier}{}*\([^()\n]*\)", space())).unwrap());
    let residue = call.replace_all(&residue, " ");
    let folded: String = residue
        .chars()
        .map(|c| match c {
            'İ' | 'ı' => 'i',
            'ſ' => 's',
            'K' => 'k',
            _ => c.to_ascii_lowercase(),
        })
        .collect();
    if [
        "where", "with", "when", "from", "that", "then", "the", "this", "these", "which", "is",
        "are",
    ]
    .iter()
    .any(|p| {
        folded
            .match_indices(p)
            .any(|(i, _)| boundary(&folded, i, i + p.len()))
    }) {
        return true;
    }
    let re = WORDS.get_or_init(|| Regex::new("[A-Za-z]{2,}").unwrap());
    let count = re
        .find_iter(&residue)
        .filter(|m| {
            !residue[..m.start()].chars().next_back().is_some_and(|c| {
                c.is_ascii_alphabetic() || rapiddoc_native::python_decimal(c).is_some()
            }) && !residue[m.end()..].chars().next().is_some_and(|c| {
                c.is_ascii_alphabetic() || rapiddoc_native::python_decimal(c).is_some()
            })
        })
        .count();
    count >= 3
        || residue
            .chars()
            .filter(|c| ('\u{3400}'..='\u{9fff}').contains(c))
            .count()
            >= 3
}
pub(super) fn residue(text: &str) -> String {
    static EXPR: OnceLock<Regex> = OnceLock::new();
    static CALL: OnceLock<Regex> = OnceLock::new();
    let identifier =
        r"[A-Za-z\u{c0}-\u{2af}\u{370}-\u{3ff}][A-Za-z0-9_\u{c0}-\u{2af}\u{370}-\u{3ff}]*";
    let normalized: String = text.nfkc().collect();
    let expression = EXPR.get_or_init(|| {
        let s = space();
        let d = digits();
        let atom =
            format!(r"(?:{identifier}|[{d}]+(?:\.[{d}]+)?)(?:{s}*[(\[{{][^()\[\]{{}}\n]*[)\]}}])*");
        Regex::new(&format!(r"{atom}(?:{s}*[+−\-*/=<>≤≥≠≈∝∈×÷^_]{s}*{atom})+")).unwrap()
    });
    let residue = expression.replace_all(&normalized, " ");
    CALL.get_or_init(|| Regex::new(&format!(r"{identifier}{}*\([^()\n]*\)", space())).unwrap())
        .replace_all(&residue, " ")
        .into_owned()
}
