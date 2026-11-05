//! Current Python Office hyperlink policy, including urllib netloc checks.
use std::sync::OnceLock;
fn space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}
pub fn sanitize(input: Option<&str>) -> Option<&str> {
    let s = input?.trim_matches(space);
    if s.is_empty()
        || s == "."
        || s.starts_with("//")
        || s.chars()
            .any(|c| c < '\u{20}' || c == '\u{7f}' || matches!(c, '<' | '>' | '\\'))
    {
        return None;
    }
    let scheme = s.split_once(':').filter(|(prefix, _)| {
        prefix
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphabetic)
            && prefix
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'+' | b'-' | b'.'))
    });
    let (scheme, tail) = match scheme {
        Some((name, tail)) => (name.to_ascii_lowercase(), tail),
        None => (String::new(), s),
    };
    let (netloc, path) = if let Some(tail) = tail.strip_prefix("//") {
        let end = tail.find(['/', '?', '#']).unwrap_or(tail.len());
        (&tail[..end], &tail[end..])
    } else {
        ("", tail)
    };
    if !netloc.is_empty() && !valid_netloc(netloc) {
        return None;
    }
    if !scheme.is_empty() {
        match scheme.as_str() {
            "http" | "https" | "ftp" => {
                let host = netloc.rsplit('@').next().unwrap_or("");
                let host = if host.starts_with('[') {
                    host[1..].split(']').next().unwrap_or("")
                } else {
                    host.split(':').next().unwrap_or("")
                };
                if host.is_empty() {
                    return None;
                }
            }
            "mailto" | "tel" => {
                if path.split(['?', '#']).next().unwrap_or("").is_empty() {
                    return None;
                }
            }
            _ => return None,
        }
        return Some(s);
    }
    if !netloc.is_empty() {
        return None;
    }
    if let Some(fragment) = s.strip_prefix('#') {
        return if fragment.trim_matches(space).is_empty() {
            None
        } else {
            Some(s)
        };
    }
    if s.starts_with('/') {
        None
    } else {
        Some(s)
    }
}
fn valid_netloc(netloc: &str) -> bool {
    static FORBIDDEN: OnceLock<String> = OnceLock::new();
    let forbidden = FORBIDDEN.get_or_init(|| {
        serde_json::from_str(include_str!("link_nfkc_forbidden.json")).expect("Unicode link data")
    });
    if netloc
        .chars()
        .any(|c| !c.is_ascii() && forbidden.contains(c))
    {
        return false;
    }
    if netloc.contains('[') != netloc.contains(']') {
        return false;
    }
    if netloc.contains('[') {
        let host_port = netloc.rsplit('@').next().unwrap();
        let host = if let Some((before, after)) = host_port.split_once('[') {
            if !before.is_empty() {
                return false;
            }
            let Some((host, port)) = after.split_once(']') else {
                return false;
            };
            if !port.is_empty() && !port.starts_with(':') {
                return false;
            }
            host
        } else {
            host_port.split(':').next().unwrap_or("")
        };
        if let Some(future) = host.strip_prefix('v') {
            let Some((version, address)) = future.split_once('.') else {
                return false;
            };
            !version.is_empty()
                && version.bytes().all(|c| c.is_ascii_hexdigit())
                && !address.is_empty()
                && !address.contains('\n')
        } else {
            let address = if let Some((ip, zone)) = host.split_once('%') {
                if zone.is_empty() || zone.contains('%') {
                    return false;
                }
                ip
            } else {
                host
            };
            address.parse::<std::net::Ipv6Addr>().is_ok()
        }
    } else {
        true
    }
}
