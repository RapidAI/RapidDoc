//! Frozen pre-optimization cleaner for semantic equivalence tests.
use rapiddoc_office::{
    document_output::html::{escape, unescape},
    Result,
};
use std::sync::OnceLock;
pub fn original_clean(s: &str) -> Result<String> {
    static TAG: OnceLock<regex::Regex> = OnceLock::new();
    static ATTR: OnceLock<regex::Regex> = OnceLock::new();
    let tag = TAG.get_or_init(|| regex::Regex::new(r"<(\w+)(?:\s+[^>]*)?\s*/?>").unwrap());
    let attr = ATTR
        .get_or_init(|| regex::Regex::new(r#"(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+))"#).unwrap());
    Ok(tag
        .replace_all(s, |c: &regex::Captures<'_>| {
            let name = c[1].to_lowercase();
            let full = c.get(0).unwrap().as_str();
            let mut out = format!("<{name}");
            for a in attr.captures_iter(full) {
                let key = a[1].to_lowercase();
                let value = a
                    .get(2)
                    .filter(|m| !m.as_str().is_empty())
                    .or_else(|| a.get(3).filter(|m| !m.as_str().is_empty()))
                    .or_else(|| a.get(4))
                    .map(|m| m.as_str())
                    .unwrap_or("");
                let value = if name == "a" && key == "href" {
                    let value = unescape(value).trim().to_owned();
                    let scheme = value
                        .split_once(':')
                        .filter(|(p, _)| {
                            p.as_bytes().first().is_some_and(u8::is_ascii_alphabetic)
                                && p.bytes().all(|c| {
                                    c.is_ascii_alphanumeric() || matches!(c, b'+' | b'-' | b'.')
                                })
                        })
                        .map(|(s, _)| s.to_ascii_lowercase());
                    if value.is_empty()
                        || scheme.is_some_and(|s| {
                            !matches!(s.as_str(), "http" | "https" | "mailto" | "ftp")
                        })
                    {
                        continue;
                    }
                    escape(&value)
                } else {
                    value.to_owned()
                };
                if matches!(key.as_str(), "colspan" | "rowspan")
                    || name == "img" && matches!(key.as_str(), "src" | "alt" | "width" | "height")
                    || name == "a" && key == "href"
                {
                    out.push_str(&format!(" {key}=\"{value}\""));
                }
            }
            out.push_str(if full.trim_end().ends_with("/>") {
                "/>"
            } else {
                ">"
            });
            out
        })
        .into_owned())
}
