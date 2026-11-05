//! Original Office HTML cleaning, image persistence and display formatting.
use super::*;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;
#[derive(serde::Deserialize)]
struct Entities {
    entities: BTreeMap<String, String>,
    invalid_map: BTreeMap<u32, String>,
    invalid: BTreeSet<u32>,
}
pub fn unescape(s: &str) -> String {
    static MAP: OnceLock<Entities> = OnceLock::new();
    let map = MAP.get_or_init(|| serde_json::from_str(include_str!("html_entities.json")).unwrap());
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i + 1..];
        if let Some(number) = rest.strip_prefix('#') {
            let (hex, number) = if number.starts_with(['x', 'X']) {
                (true, &number[1..])
            } else {
                (false, number)
            };
            let n = number
                .bytes()
                .take_while(|c| {
                    if hex {
                        c.is_ascii_hexdigit()
                    } else {
                        c.is_ascii_digit()
                    }
                })
                .count();
            if n > 0 {
                let code = u32::from_str_radix(&number[..n], if hex { 16 } else { 10 })
                    .unwrap_or(u32::MAX);
                if let Some(s) = map.invalid_map.get(&code) {
                    out.push_str(s);
                } else if (0xd800..=0xdfff).contains(&code) || code > 0x10ffff {
                    out.push('\u{fffd}');
                } else if !map.invalid.contains(&code) {
                    if let Some(c) = char::from_u32(code) {
                        out.push(c);
                    }
                }
                rest = &number[n..];
                if rest.starts_with(';') {
                    rest = &rest[1..];
                }
                continue;
            }
        }
        let end = rest
            .char_indices()
            .take_while(|(_, c)| !matches!(c, '\t' | '\n' | '\x0c' | ' ' | '<' | '&' | '#' | ';'))
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        let end = if rest[end..].starts_with(';') {
            end + 1
        } else {
            end
        };
        let found = (1..=end)
            .rev()
            .filter(|i| rest.is_char_boundary(*i))
            .find_map(|i| map.entities.get(&rest[..i]).map(|s| (i, s)));
        if let Some((i, s)) = found {
            out.push_str(s);
            rest = &rest[i..];
        } else {
            out.push('&');
        }
    }
    out.push_str(rest);
    out
}
pub fn escape(s: &str) -> String {
    crate::chart::escape(s)
}
pub fn clean(s: &str) -> Result<String> {
    static TAG: OnceLock<regex::Regex> = OnceLock::new();
    static ATTR: OnceLock<regex::Regex> = OnceLock::new();
    static NAME: OnceLock<regex::Regex> = OnceLock::new();
    let tag = TAG.get_or_init(|| regex::Regex::new(r"<(\w+)(?:\s+[^>]*)?\s*/?>").unwrap());
    let attr = ATTR
        .get_or_init(|| regex::Regex::new(r#"(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+))"#).unwrap());
    let name_pattern = NAME.get_or_init(|| regex::Regex::new(r"^\w+").unwrap());
    let mut cleaned = String::with_capacity(s.len());
    let mut previous = 0;
    // Finding the same matches avoids the capture engine's per-byte work on
    // large quoted data URIs. Only short names need a separate word match.
    for matched in tag.find_iter(s) {
        cleaned.push_str(&s[previous..matched.start()]);
        let full = matched.as_str();
        let name = name_pattern
            .find(&full[1..])
            .unwrap()
            .as_str()
            .to_lowercase();
        let mut out = format!("<{name}");
        for a in attr.find_iter(full) {
            let (key, value) = a.as_str().split_once('=').unwrap();
            let key = key.trim_end().to_lowercase();
            let value = value.trim_start();
            let value = if value.len() >= 2
                && (value.starts_with('"') && value.ends_with('"')
                    || value.starts_with('\'') && value.ends_with('\''))
            {
                &value[1..value.len() - 1]
            } else {
                value
            };
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
                    || scheme
                        .is_some_and(|s| !matches!(s.as_str(), "http" | "https" | "mailto" | "ftp"))
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
        cleaned.push_str(&out);
        previous = matched.end();
    }
    cleaned.push_str(&s[previous..]);
    Ok(cleaned)
}
fn save(uri: &str, assets: &mut BTreeMap<String, Arc<[u8]>>) -> Result<Option<String>> {
    let Some(tail) = uri.strip_prefix("data:image/") else {
        return Ok(None);
    };
    let Some((format, data)) = tail.split_once(";base64,") else {
        return Ok(None);
    };
    if data.is_empty() {
        return Ok(None);
    }
    let Ok(bytes) = base64::decode(data) else {
        return Ok(None);
    };
    let mut source = uri.to_owned();
    let mut format = format.to_lowercase();
    let mut bytes: Arc<[u8]> = bytes.into();
    if crate::image::is_vector(&[], None, Some(&format!("image/{format}"))) {
        let Some(rendered) =
            crate::image::serialize(&bytes, None, Some(&format!("image/{format}")))?
        else {
            return Ok(None);
        };
        source = rendered.to_string();
        let Some((mime, data)) = source
            .strip_prefix("data:image/")
            .and_then(|s| s.split_once(";base64,"))
        else {
            return Ok(None);
        };
        format = mime.to_lowercase();
        bytes = base64::decode(data)
            .map_err(crate::Error::malformed)?
            .into();
    }
    let format = format.split('+').next().unwrap_or(&format);
    let format = if format == "jpeg" { "jpg" } else { format };
    let name = format!("{:x}.{format}", Sha256::digest(source.as_bytes()));
    assets.entry(name.clone()).or_insert(bytes);
    Ok(Some(name))
}
pub fn save_images(block: &mut Block, assets: &mut BTreeMap<String, Arc<[u8]>>) -> Result<()> {
    for child in block.blocks.iter_mut().flatten() {
        save_images(child, assets)?;
    }
    for line in block.lines.iter_mut().flatten() {
        for span in &mut line.spans {
            if matches!(span.kind.as_str(), "image" | "chart") {
                if let Some(uri) = span.image_base64.as_deref() {
                    if let Some(path) = save(uri, assets)? {
                        span.image_path = Some(path);
                        span.image_base64 = None;
                    } else {
                        span.image_path = Some(String::new());
                    }
                } else {
                    span.image_path.get_or_insert_default();
                }
            }
            if span.kind == "table" {
                if let Some(markup) = &mut span.html {
                    static SRC: OnceLock<regex::Regex> = OnceLock::new();
                    let re = SRC
                        .get_or_init(|| regex::Regex::new(r#"src="(data:image/[^"]+)""#).unwrap());
                    let mut out = String::new();
                    let mut start = 0;
                    for c in re.captures_iter(markup) {
                        let m = c.get(0).unwrap();
                        out.push_str(&markup[start..m.start()]);
                        if let Some(path) = save(&c[1], assets)? {
                            out.push_str(&format!("src=\"{path}\""));
                        } else {
                            out.push_str(m.as_str());
                        }
                        start = m.end();
                    }
                    out.push_str(&markup[start..]);
                    *markup = out;
                }
            }
        }
    }
    Ok(())
}
pub fn format_embedded(s: &str, prefix: &str, left: &str, right: &str) -> String {
    static SRC: OnceLock<regex::Regex> = OnceLock::new();
    static EQ: OnceLock<regex::Regex> = OnceLock::new();
    let src = SRC.get_or_init(|| regex::Regex::new(r#"src="([^"]+)""#).unwrap());
    let eq = EQ.get_or_init(|| regex::Regex::new(r"(?s)<eq>(.*?)</eq>").unwrap());
    let s = src.replace_all(s, |c: &regex::Captures<'_>| {
        if prefix.is_empty() || c[1].starts_with("data:") {
            c[0].into()
        } else {
            format!("src=\"{prefix}/{}\"", &c[1])
        }
    });
    eq.replace_all(&s, |c: &regex::Captures<'_>| {
        format!(" {left}{}{right} ", &c[1])
    })
    .into_owned()
}
