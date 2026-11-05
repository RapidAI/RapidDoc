//! Original Office block-level Markdown/HTML selection and span wrapper rules.
use super::{html, Block, Span};
use crate::docx::space;
use std::{collections::BTreeSet, sync::OnceLock};
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Delimiters {
    pub inline_left: String,
    pub inline_right: String,
    pub display_left: String,
    pub display_right: String,
}
impl Default for Delimiters {
    fn default() -> Self {
        Self {
            inline_left: "$".into(),
            inline_right: "$".into(),
            display_left: "$$".into(),
            display_right: "$$".into(),
        }
    }
}
pub fn content(s: &Span) -> &str {
    s.content.as_deref().unwrap_or("")
}
fn key(styles: &[String]) -> Option<u8> {
    let mut key = 0;
    for s in styles {
        key |= match s.as_str() {
            "bold" => 1,
            "italic" => 2,
            "strikethrough" => 4,
            "" => 0,
            _ => return None,
        };
    }
    if matches!(key, 0..=4) {
        Some(key)
    } else {
        None
    }
}
fn marker(s: &[String]) -> Option<char> {
    if s.iter().any(|s| s == "underline") {
        Some('_')
    } else if s.iter().any(|s| s == "strikethrough") {
        Some('-')
    } else {
        None
    }
}
fn punct(c: char) -> bool {
    static R: OnceLock<Vec<[u32; 2]>> = OnceLock::new();
    let r = R.get_or_init(|| serde_json::from_str(include_str!("punctuation.json")).unwrap());
    let i = r.partition_point(|r| r[0] <= c as u32);
    i > 0 && r[i - 1][1] >= c as u32
}
fn html_mode(b: &Block) -> bool {
    let mut styles = BTreeSet::new();
    for s in b.spans() {
        let units = if s.kind == "hyperlink" {
            s.children
                .as_deref()
                .filter(|v| !v.is_empty())
                .map(|v| v.iter().collect::<Vec<_>>())
                .unwrap_or_else(|| vec![s])
        } else {
            vec![s]
        };
        let mut linkkeys = BTreeSet::new();
        for unit in units {
            if s.kind == "hyperlink" && unit.kind != "text" && unit.kind != "hyperlink" {
                return true;
            }
            let Some(k) = key(&unit.style) else {
                return true;
            };
            if s.kind == "hyperlink" && !content(unit).trim_matches(space).is_empty() {
                linkkeys.insert(k);
            }
            if !content(unit).is_empty() && k != 0 {
                styles.insert(k);
            }
        }
        if linkkeys.len() > 1 {
            return true;
        }
    }
    styles.len() > 1
}
pub fn escape(s: &str, html_mode: bool) -> String {
    let s = if html_mode {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    } else {
        s.into()
    };
    let mut out = String::new();
    let mut slashes = 0;
    for c in s.chars() {
        if c == '\\' {
            slashes += 1;
            out.push(c);
            continue;
        }
        if matches!(c, '*' | '_' | '`' | '~' | '$') && slashes % 2 == 0 {
            out.push('\\');
        }
        out.push(c);
        slashes = 0;
    }
    out
}
fn stack(styles: &[String]) -> Vec<&'static str> {
    let mut out = vec![];
    for k in [
        "emphasis",
        "strikethrough",
        "italic",
        "bold",
        "underline",
        "superscript",
        "subscript",
    ] {
        if k == "subscript" && styles.iter().any(|s| s == "superscript") {
            continue;
        }
        if styles.iter().any(|s| s == k) {
            out.push(k);
        }
    }
    out
}
fn open(s: &str) -> &'static str {
    match s {
        "emphasis" => "<span style=\"text-emphasis: dot; text-emphasis-position: under;\">",
        "strikethrough" => "<s>",
        "italic" => "<em>",
        "bold" => "<strong>",
        "underline" => "<u>",
        "superscript" => "<sup>",
        "subscript" => "<sub>",
        _ => "",
    }
}
fn close(s: &str) -> &'static str {
    match s {
        "emphasis" => "</span>",
        "strikethrough" => "</s>",
        "italic" => "</em>",
        "bold" => "</strong>",
        "underline" => "</u>",
        "superscript" => "</sup>",
        "subscript" => "</sub>",
        _ => "",
    }
}
fn apply(s: &str, styles: &[String], h: bool) -> String {
    if s.is_empty() {
        return s.into();
    }
    if h {
        let tags = stack(styles);
        format!(
            "{}{}{}",
            tags.iter().map(|s| open(s)).collect::<String>(),
            s,
            tags.iter().rev().map(|s| close(s)).collect::<String>()
        )
    } else {
        let wrapper = match key(styles) {
            Some(1) => "**",
            Some(2) => "*",
            Some(3) => "***",
            Some(4) => "~~",
            _ => "",
        };
        format!("{wrapper}{s}{wrapper}")
    }
}
fn ascii_only(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|c| c == b' ')
}
fn visible(s: &str, m: char, h: bool) -> String {
    let leading = s.len() - s.trim_start_matches(' ').len();
    let trailing = s.len() - s.trim_end_matches(' ').len();
    if ascii_only(s) {
        m.to_string().repeat(s.len())
    } else {
        format!(
            "{}{}{}",
            m.to_string().repeat(leading),
            escape(&s[leading..s.len() - trailing], h),
            m.to_string().repeat(trailing)
        )
    }
}
pub fn styled(s: &str, styles: &[String], h: bool) -> String {
    if let Some(m) = marker(styles) {
        let styles = styles
            .iter()
            .filter(|v| {
                !ascii_only(s)
                    || v.as_str()
                        != if m == '_' {
                            "underline"
                        } else {
                            "strikethrough"
                        }
            })
            .cloned()
            .collect::<Vec<_>>();
        apply(&visible(s, m, h), &styles, h)
    } else {
        apply(&escape(s, h), styles, h)
    }
}
struct Part {
    kind: String,
    text: String,
    raw: String,
    wrapper: bool,
}
impl Part {
    fn plain(kind: &str, text: String) -> Self {
        Self {
            kind: kind.into(),
            text,
            raw: String::new(),
            wrapper: false,
        }
    }
}
fn append_text(parts: &mut Vec<Part>, s: &str, styles: &[String], h: bool) {
    let text = if marker(styles).is_some() && !s.is_empty() {
        styled(s, styles, h)
    } else {
        let escaped = escape(s, h);
        let core = escaped.trim_matches(space);
        if !core.is_empty() {
            let a = escaped.len() - escaped.trim_start_matches(space).len();
            let z = escaped.trim_end_matches(space).len();
            format!(
                "{}{}{}",
                &escaped[..a],
                apply(core, styles, h),
                &escaped[z..]
            )
        } else {
            s.into()
        }
    };
    if !text.is_empty() {
        parts.push(Part {
            kind: "text".into(),
            text,
            raw: s.into(),
            wrapper: !h && matches!(key(styles), Some(1..=4)),
        });
    }
}
fn join(parts: Vec<Part>) -> String {
    let mut out = String::new();
    for (i, p) in parts.iter().enumerate() {
        if p.kind == "inline_equation" {
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(&p.text);
            if i + 1 < parts.len() {
                out.push(' ');
            }
        } else {
            if let Some(prev) = i.checked_sub(1).map(|i| &parts[i]) {
                if prev.wrapper
                    && !matches!(
                        p.kind.as_str(),
                        "hyperlink" | "inline_equation" | "interline_equation"
                    )
                    && !prev.raw.trim_matches(space).is_empty()
                    && !p.raw.trim_matches(space).is_empty()
                    && prev
                        .raw
                        .chars()
                        .last()
                        .is_some_and(|c| !space(c) && punct(c))
                    && p.raw.chars().next().is_some_and(|c| !space(c) && !punct(c))
                {
                    out.push(' ');
                }
            }
            out.push_str(&p.text);
        }
    }
    out
}
fn grouped(parts: &mut Vec<Part>, spans: &[&Span], h: bool) {
    if !h {
        let mut at = 0;
        while at < spans.len() {
            let mut end = at + 1;
            while end < spans.len() && key(&spans[at].style) == key(&spans[end].style) {
                end += 1;
            }
            let text = spans[at..end]
                .iter()
                .map(|s| content(s))
                .collect::<String>();
            append_text(parts, &text, &spans[at].style, false);
            at = end;
        }
        return;
    }
    let a = spans
        .iter()
        .position(|s| !content(s).trim_matches(space).is_empty())
        .unwrap_or(spans.len());
    let z = spans
        .iter()
        .rposition(|s| !content(s).trim_matches(space).is_empty())
        .map(|i| i + 1)
        .unwrap_or(a);
    for s in &spans[..a] {
        append_text(parts, content(s), &s.style, true);
    }
    let mut tokens = vec![];
    for s in &spans[a..z] {
        let raw = content(s);
        if raw.is_empty() {
            continue;
        }
        if let Some(m) = marker(&s.style) {
            let styles = s
                .style
                .iter()
                .filter(|v| {
                    !ascii_only(raw)
                        || v.as_str()
                            != if m == '_' {
                                "underline"
                            } else {
                                "strikethrough"
                            }
                })
                .cloned()
                .collect::<Vec<_>>();
            tokens.push((visible(raw, m, true), styles));
        } else {
            let escaped = escape(raw, true);
            let core = escaped.trim_matches(space);
            if core.is_empty() {
                tokens.push((raw.into(), vec![]));
            } else {
                let l = escaped.len() - escaped.trim_start_matches(space).len();
                let r = escaped.trim_end_matches(space).len();
                if l > 0 {
                    tokens.push((escaped[..l].into(), vec![]));
                }
                tokens.push((core.into(), s.style.clone()));
                if r < escaped.len() {
                    tokens.push((escaped[r..].into(), vec![]));
                }
            }
        }
    }
    let mut out = String::new();
    let mut current: Vec<&str> = vec![];
    for (text, styles) in tokens {
        let next = stack(&styles);
        let shared = current
            .iter()
            .zip(&next)
            .take_while(|(a, b)| a == b)
            .count();
        for s in current[shared..].iter().rev() {
            out.push_str(close(s));
        }
        for s in &next[shared..] {
            out.push_str(open(s));
        }
        out.push_str(&text);
        current = next;
    }
    for s in current.iter().rev() {
        out.push_str(close(s));
    }
    if !out.is_empty() {
        parts.push(Part::plain("text", out));
    }
    for s in &spans[z..] {
        append_text(parts, content(s), &s.style, true);
    }
}
pub fn link(text: &str, url: &str, h: bool) -> String {
    if h {
        format!("<a href=\"{}\">{text}</a>", html::escape(url))
    } else {
        format!("[{text}]({url})")
    }
}
fn render_spans(spans: &[&Span], h: bool, d: &Delimiters) -> String {
    let mut parts = vec![];
    let mut text = vec![];
    for s in spans {
        if s.kind == "text" {
            text.push(*s);
            continue;
        }
        grouped(&mut parts, &text, h);
        text.clear();
        let raw = content(s);
        match s.kind.as_str() {
            "inline_equation" => {
                let eq = format!("{}{raw}{}", d.inline_left, d.inline_right);
                if !eq.trim_matches(space).is_empty() {
                    parts.push(Part::plain(
                        "inline_equation",
                        eq.trim_matches(space).into(),
                    ));
                }
            }
            "interline_equation" => {
                let eq = format!("\n{}\n{raw}\n{}\n", d.display_left, d.display_right);
                parts.push(Part::plain(
                    "interline_equation",
                    eq.trim_matches(space).into(),
                ));
            }
            "hyperlink" => {
                let label = if let Some(children) = s.children.as_ref().filter(|v| !v.is_empty()) {
                    render_spans(&children.iter().collect::<Vec<_>>(), h, d)
                        .trim_matches(space)
                        .into()
                } else {
                    styled(raw.trim_matches(space), &s.style, h)
                };
                if !label.is_empty() {
                    parts.push(Part::plain(
                        "hyperlink",
                        link(&label, s.url.as_deref().unwrap_or(""), h),
                    ));
                }
            }
            "code_inline" => {
                let text = if h {
                    format!("<code>{}</code>", html::escape(raw))
                } else {
                    let mut delimiter = "`".to_owned();
                    while raw.contains(&delimiter) {
                        delimiter.push('`');
                    }
                    let pad = if raw.starts_with('`') || raw.ends_with('`') {
                        " "
                    } else {
                        ""
                    };
                    format!("{delimiter}{pad}{raw}{pad}{delimiter}")
                };
                parts.push(Part::plain("code_inline", text));
            }
            _ => (),
        }
    }
    grouped(&mut parts, &text, h);
    join(parts)
}
pub fn merge(b: &Block, d: &Delimiters, escape_prefix: bool) -> String {
    let mut text = String::new();
    if b.kind == "title" && b.is_numbered_style == Some(true) {
        if let Some(number) = &b.section_number {
            text.push_str(number);
            text.push(' ');
        }
    }
    text.push_str(&render_spans(
        &b.spans().collect::<Vec<_>>(),
        html_mode(b),
        d,
    ));
    if b.kind == "text" {
        text = text.trim_matches(space).into();
        if !text.is_empty() && (text.chars().all(|c| c == '_') || text.chars().all(|c| c == '-')) {
            text.insert(0, '\\');
        }
        if escape_prefix {
            let end = text
                .bytes()
                .take(3)
                .take_while(|c| matches!(c, b' ' | b'\t'))
                .count();
            let tail = &text[end..];
            let n = if tail.starts_with(['+', '-']) {
                1
            } else {
                tail.bytes().take(6).take_while(|c| *c == b'#').count()
            };
            if n > 0
                && tail
                    .as_bytes()
                    .get(n)
                    .is_some_and(|c| matches!(c, b' ' | b'\t'))
            {
                text.insert(end, '\\');
            }
        }
    }
    text
}
pub fn index_leaf(b: &Block, indent: &str, d: &Delimiters) -> Option<String> {
    let mut spans = b.spans().cloned().collect::<Vec<_>>();
    let last = spans
        .iter()
        .rposition(|s| s.kind != "inline_equation" && content(s).contains('\t'));
    let remove = last.is_some_and(|i| {
        let s = content(&spans[i])
            .rsplit('\t')
            .next()
            .unwrap()
            .trim_matches(space);
        !s.is_empty()
            && s.chars().count() <= 12
            && !s.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
            && (s.chars().all(|c| crate::docx::decimal(c).is_some())
                || s.to_lowercase().chars().all(|c| "ivxlcdm".contains(c))
                || s.len() == 1 && s.as_bytes()[0].is_ascii_alphabetic())
    });
    for (i, s) in spans.iter_mut().enumerate() {
        if s.kind != "inline_equation" {
            let text = content(s);
            let text = if Some(i) == last && remove {
                text.rsplit_once('\t').unwrap().0
            } else {
                text
            };
            s.content = Some(text.replace('\t', " "));
        }
    }
    let styles = spans
        .iter()
        .filter(|s| s.kind != "inline_equation" && !content(s).is_empty())
        .map(|s| s.style.clone())
        .collect::<Vec<_>>();
    let uniform = styles
        .first()
        .filter(|s| !s.is_empty() && styles.iter().all(|v| v == *s));
    let h = html_mode(b);
    let text = if let Some(styles) = uniform {
        let raw = spans
            .iter()
            .map(|s| {
                if s.kind == "inline_equation" {
                    format!("{}{}{}", d.inline_left, content(s), d.inline_right)
                } else {
                    escape(content(s), h)
                }
            })
            .collect::<String>();
        apply(raw.trim_matches(space), styles, h)
    } else {
        let mut parts = vec![];
        for s in &spans {
            let raw = content(s);
            if raw.is_empty() {
                continue;
            }
            if s.kind == "inline_equation" {
                parts.push(Part::plain(
                    "inline_equation",
                    format!("{}{raw}{}", d.inline_left, d.inline_right),
                ));
            } else if s.kind == "hyperlink" {
                let core = raw.trim_matches(space);
                if !core.is_empty() {
                    let a = raw.len() - raw.trim_start_matches(space).len();
                    let z = raw.trim_end_matches(space).len();
                    parts.push(Part::plain(
                        "hyperlink",
                        format!("{}{}{}", &raw[..a], styled(core, &s.style, h), &raw[z..]),
                    ));
                }
            } else {
                append_text(&mut parts, raw, &s.style, h);
            }
        }
        join(parts).trim_matches(space).into()
    };
    if text.is_empty() {
        return None;
    }
    let text = if let Some(anchor) = &b.anchor {
        link(&text, &format!("#{anchor}"), h)
    } else {
        text
    };
    Some(format!("{indent}- {text}"))
}
