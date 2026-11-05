//! 旧 HTML data URI 的字符串匹配与原字节写出；不重新编码图片。
use super::{invalid, limit, unsupported};
use crate::legacy_page::assets::Asset;
use rapiddoc_core::Error;
use sha2::{Digest, Sha256};

const EXTENSIONS: &[&str] = &["wmf", "x-wmf", "emf", "x-emf"];

// 对齐本次旧基线 Python 3.14.5 的 binascii(strict_mode=False)：padding
// 不提前终止，末尾才校验；3.10–3.13 的畸形 padding 行为不在此合同内。
fn decode(text: &str) -> Option<Vec<u8>> {
    if !text.is_ascii() {
        return None;
    }
    let mut out = Vec::new();
    let (mut pos, mut pad, mut bits) = (0usize, 0usize, 0u32);
    for b in text.bytes() {
        if b == b'=' {
            pad += 1;
            continue;
        }
        let n = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => continue,
        };
        pad = 0;
        bits = (bits << 6) | n as u32;
        pos += 1;
        if pos >= 2 {
            out.push((bits >> (2 * (4 - pos))) as u8);
        }
        if pos == 4 {
            pos = 0;
            bits = 0;
        }
    }
    (pos == 0 || (pos >= 2 && pos + pad >= 4)).then_some(out)
}

pub fn replace(markup: &str, prefix: &str) -> Result<(String, Vec<Asset>), Error> {
    if markup.len() > rapiddoc_core::resources::limit(2_000_000)
        || prefix.len() > rapiddoc_core::resources::limit(4096)
    {
        return Err(limit());
    }
    let mut result = String::new();
    let mut assets = Vec::new();
    let mut rest = markup;
    while let Some(start) = rest.find("src=\"data:image/") {
        result.push_str(&rest[..start]);
        rest = &rest[start..];
        let Some(end) = rest[5..].find('"').map(|n| n + 5) else {
            break;
        };
        let uri = &rest[5..end];
        let mut saved = None;
        if let Some((fmt, encoded)) = uri[11..].split_once(";base64,") {
            // [^;]+ in the original regex, and (.+) with DOTALL.
            if !fmt.is_empty() && !fmt.contains(';') && !encoded.is_empty() {
                if let Some(bytes) = decode(encoded) {
                    let raw = fmt.to_lowercase();
                    if EXTENSIONS.contains(&raw.trim()) {
                        return Err(unsupported("WMF/EMF 内联转换尚未移植"));
                    }
                    let extension = raw.split('+').next().ok_or_else(invalid)?;
                    let extension = if extension == "jpeg" {
                        "jpg"
                    } else {
                        extension
                    };
                    // 旧代码允许任意扩展名；新文件输出仅接受安全平铺扩展名。
                    if extension.len() > 32
                        || extension.is_empty()
                        || !extension.bytes().all(|b| {
                            b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_'
                        })
                    {
                        return Err(unsupported("不安全的内联图片扩展名"));
                    }
                    let name = format!("{:x}.{extension}", Sha256::digest(uri.as_bytes()));
                    if assets.len() >= rapiddoc_core::resources::limit(1000) {
                        return Err(limit());
                    }
                    assets.push(Asset {
                        name: name.clone(),
                        png: bytes,
                    });
                    saved = Some(name);
                }
            }
        }
        if let Some(name) = saved {
            result.push_str(&format!("src=\"{prefix}{name}\""));
        } else {
            result.push_str(&rest[..=end]);
        }
        rest = &rest[end + 1..];
    }
    result.push_str(rest);
    Ok((result, assets))
}
