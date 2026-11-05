//! One decoded buffer, borrowed read-only tree or streaming events; no JSON tree.
use crate::{Error, Result};
use std::borrow::Cow;

pub fn decode(bytes: &[u8]) -> Result<Cow<'_, str>> {
    if bytes.starts_with(&[0xff, 0xfe])
        || bytes.starts_with(&[0xfe, 0xff])
        || (bytes.starts_with(&[b'<', 0]) && bytes.get(3) == Some(&0))
        || bytes.starts_with(&[0, b'<', 0])
    {
        let little = bytes.starts_with(&[0xff, 0xfe]) || bytes.starts_with(&[b'<', 0]);
        let bytes = if bytes.starts_with(&[0xff, 0xfe]) || bytes.starts_with(&[0xfe, 0xff]) {
            &bytes[2..]
        } else {
            bytes
        };
        if bytes.len() % 2 != 0 {
            return Err(Error::malformed("odd UTF-16 XML byte count"));
        }
        let units = bytes.chunks_exact(2).map(|b| {
            if little {
                u16::from_le_bytes([b[0], b[1]])
            } else {
                u16::from_be_bytes([b[0], b[1]])
            }
        });
        let value = char::decode_utf16(units)
            .collect::<std::result::Result<String, _>>()
            .map_err(Error::malformed)?;
        return Ok(Cow::Owned(value));
    }
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
    // Read an ASCII-compatible declaration before decoding document content.
    if bytes.starts_with(b"<?xml") {
        let end = bytes
            .windows(2)
            .position(|v| v == b"?>")
            .unwrap_or(bytes.len());
        if let Ok(declaration) = std::str::from_utf8(&bytes[..end]) {
            if let Some((_, tail)) = declaration.split_once("encoding") {
                if let Some((_, value)) = tail.split_once('=') {
                    let value = value.trim_start();
                    if let Some(quote @ ('\'' | '"')) = value.chars().next() {
                        if let Some(label) = value[1..].split(quote).next() {
                            let encoding = encoding_rs::Encoding::for_label(label.as_bytes())
                                .ok_or_else(|| Error::malformed("unknown XML encoding"))?;
                            if encoding != encoding_rs::UTF_8 {
                                return encoding
                                    .decode_without_bom_handling_and_without_replacement(bytes)
                                    .ok_or_else(|| {
                                        Error::malformed("invalid XML encoding sequence")
                                    });
                            }
                        }
                    }
                }
            }
        }
    }
    std::str::from_utf8(bytes)
        .map(Cow::Borrowed)
        .map_err(Error::malformed)
}

pub fn parse(text: &str) -> Result<roxmltree::Document<'_>> {
    roxmltree::Document::parse(text).map_err(Error::malformed)
}
/// Overlay edits operate on decoded text. Do not leave a UTF-16 or legacy
/// encoding declaration attached to the resulting UTF-8 bytes.
pub(crate) fn encode_utf8(mut text: String) -> Vec<u8> {
    if text.starts_with("<?xml") {
        if let Some(end) = text.find("?>") {
            if let Some(start) = text[..end].find("encoding") {
                if let Some(eq) = text[start..end].find('=') {
                    let start = start + eq + 1;
                    let tail = text[start..end].trim_start();
                    if let Some(quote @ ('\'' | '"')) = tail.chars().next() {
                        let value = start + (text[start..end].len() - tail.len()) + 1;
                        if let Some(length) = text[value..end].find(quote) {
                            text.replace_range(value..value + length, "UTF-8");
                        }
                    }
                }
            }
        }
    }
    text.into_bytes()
}

/// Validate a large, already decoded part without constructing a full DOM.
pub fn validate_stream(text: &str) -> Result<()> {
    let mut reader = quick_xml::Reader::from_str(text);
    let mut depth = 0usize;
    let mut roots = 0usize;
    loop {
        use quick_xml::events::Event;
        match reader.read_event().map_err(Error::malformed)? {
            Event::Start(_) => {
                if depth == 0 {
                    roots += 1;
                }
                depth += 1;
            }
            Event::Empty(_) => {
                if depth == 0 {
                    roots += 1;
                }
            }
            Event::End(_) => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| Error::malformed("unbalanced XML end"))?
            }
            Event::Text(text) if depth == 0 => {
                if text
                    .as_ref()
                    .chars()
                    .any(|c| !matches!(c, ' ' | '\r' | '\n' | '\t'))
                {
                    return Err(Error::malformed("text outside XML root"));
                }
            }
            Event::CData(_) if depth == 0 => {
                return Err(Error::malformed("CDATA outside XML root"))
            }
            Event::Decl(_) if roots > 0 => {
                return Err(Error::malformed("XML declaration after root"))
            }
            Event::Eof => break,
            _ => {}
        }
    }
    if roots != 1 || depth != 0 {
        return Err(Error::malformed("XML must have one complete root"));
    }
    Ok(())
}
