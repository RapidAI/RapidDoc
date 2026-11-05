//! 旧外部URI策略及urllib拆分/主机校验边界；保留原目标字符串。
use std::net::Ipv6Addr;
use unicode_normalization::UnicodeNormalization;
fn space(c: char) -> bool {
    matches!(c as u32,9..=13|28..=32|0x85|0xa0|0x1680|0x2000..=0x200a|0x2028|0x2029|0x202f|0x205f|0x3000)
}
fn bracketed_host(s: &str) -> bool {
    if let Some(s) = s.strip_prefix('v') {
        let Some((version, address)) = s.split_once('.') else {
            return false;
        };
        !version.is_empty() && version.bytes().all(|b| b.is_ascii_hexdigit()) && !address.is_empty()
    } else {
        let address = if let Some((a, scope)) = s.split_once('%') {
            if scope.is_empty() || scope.contains('%') {
                return false;
            }
            a
        } else {
            s
        };
        address.parse::<Ipv6Addr>().is_ok()
    }
}
fn netloc(s: &str) -> bool {
    let open = s.contains('[');
    let close = s.contains(']');
    if open != close {
        return false;
    }
    let host = s.rsplit('@').next().unwrap();
    if open {
        let (hostname, tail) = if let Some((before, after)) = host.split_once('[') {
            if !before.is_empty() {
                return false;
            }
            let Some((hostname, tail)) = after.split_once(']') else {
                return false;
            };
            if !tail.is_empty() && !tail.starts_with(':') {
                return false;
            }
            (hostname, tail)
        } else {
            host.split_once(':').unwrap_or((host, ""))
        };
        let _ = tail;
        if !bracketed_host(hostname) {
            return false;
        }
    }
    if !s.is_ascii() {
        let raw: String = s.chars().filter(|c| !"@:#?".contains(*c)).collect();
        let normalized: String = raw.nfkc().collect();
        if raw != normalized && normalized.chars().any(|c| "/?#@:".contains(c)) {
            return false;
        }
    }
    true
}
pub fn validate(s: &str) -> Option<String> {
    let s = s.trim_matches(space);
    if s.is_empty()
        || s == "."
        || s.starts_with("//")
        || s.chars()
            .any(|c| (c as u32) < 32 || c == '\u{7f}' || "<>\\".contains(c))
    {
        return None;
    }
    let (scheme, rest) = s.split_once(':')?;
    if !scheme.as_bytes().first()?.is_ascii_alphabetic()
        || !scheme
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"+-.".contains(&b))
    {
        return None;
    }
    let scheme = scheme.to_ascii_lowercase();
    if !matches!(scheme.as_str(), "http" | "https" | "mailto" | "tel") {
        return None;
    }
    let (authority, path) = if let Some(rest) = rest.strip_prefix("//") {
        let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let authority = &rest[..end];
        if !netloc(authority) {
            return None;
        }
        (authority, &rest[end..])
    } else {
        ("", rest)
    };
    if matches!(scheme.as_str(), "http" | "https") {
        if authority.is_empty() {
            return None;
        }
        let host = authority.rsplit('@').next().unwrap();
        let hostname = if let Some((_, after)) = host.split_once('[') {
            after.split(']').next().unwrap()
        } else {
            host.split(':').next().unwrap()
        };
        if hostname.is_empty() {
            return None;
        }
    } else if path.split(['?', '#']).next().unwrap_or("").is_empty() {
        return None;
    }
    Some(s.to_owned())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_uri_policy() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/pdf-link-annotations.json");
        let f: serde_json::Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["uris"].as_array().unwrap() {
            assert_eq!(
                serde_json::json!(validate(c["value"].as_str().unwrap())),
                c["expected"],
                "{}",
                c["value"]
            );
        }
    }
}
