//! Original code-page, BIFF Unicode/CONTINUE and UTF-16 rich-run semantics.
use super::records::*;
use serde::{Deserialize, Serialize};
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Font {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strike: bool,
    pub superscript: bool,
    pub subscript: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Run {
    pub start: usize,
    pub end: usize,
    pub style: Font,
}
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RichText {
    pub text: String,
    pub runs: Vec<Run>,
}
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Decoded {
    pub text: String,
    pub font_starts: Vec<(u16, u16)>,
}
pub fn clean(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .trim_end_matches('\0')
        .chars()
        .filter(|c| !matches!(*c,'\0'..='\x08'|'\x0b'|'\x0c'|'\x0e'..='\x1f'))
        .collect()
}
pub fn codepage_name(cp: u16) -> &'static str {
    match cp {
        874 => "cp874",
        932 => "shift_jis",
        936 => "gbk",
        949 => "euc_kr",
        950 => "big5",
        1200 => "utf-16le",
        1250 => "cp1250",
        1251 => "cp1251",
        1253 => "cp1253",
        1254 => "cp1254",
        1255 => "cp1255",
        1256 => "cp1256",
        1257 => "cp1257",
        1258 => "cp1258",
        _ => "cp1252",
    }
}
pub fn biff8(r: &mut Segments<'_>, short: bool, rich: bool) -> Option<Decoded> {
    let count = if short {
        r.u8().map(usize::from)
    } else {
        r.u16().map(usize::from)
    };
    let flags = r.u8();
    let count = count?;
    let flags = flags?;
    let mut wide = flags & 1 != 0;
    let runs = if rich && flags & 8 != 0 {
        r.u16()? as usize
    } else {
        0
    };
    let extra = if rich && flags & 4 != 0 {
        r.u32()? as usize
    } else {
        0
    };
    let mut units = Vec::with_capacity(count);
    let mut remaining = count;
    while remaining > 0 {
        if r.remaining() == 0 {
            if !r.next_segment() {
                return None;
            }
            wide = r.u8()? & 1 != 0;
        }
        let size = if wide { 2 } else { 1 };
        let take = (r.remaining() / size).min(remaining);
        if take == 0 {
            return None;
        }
        let b = r.read(take * size)?;
        if wide {
            units.extend(b.chunks_exact(2).map(|b| u16(b, 0).unwrap()));
        } else {
            units.extend(b.iter().map(|b| *b as u16));
        }
        remaining -= take;
    }
    let mut starts = vec![];
    for _ in 0..runs {
        let Some(b) = r.across(4) else { break };
        let a = u16(&b, 0)?;
        if a as usize <= count {
            starts.push((a, u16(&b, 2)?));
        }
    }
    if extra > 0 {
        r.skip(extra);
    }
    Some(Decoded {
        text: clean(&String::from_utf16_lossy(&units)),
        font_starts: starts,
    })
}
pub fn byte_string(r: &mut Segments<'_>, short: bool, codepage: u16) -> Option<Decoded> {
    let n = if short {
        r.u8()? as usize
    } else {
        r.u16()? as usize
    };
    let b = r.across(n)?;
    let text = decode_bytes(&b, codepage);
    Some(Decoded {
        text: clean(&text),
        font_starts: vec![],
    })
}
pub fn decode_bytes(b: &[u8], cp: u16) -> String {
    if cp == 1200 {
        let mut units = b
            .chunks_exact(2)
            .map(|b| u16(b, 0).unwrap())
            .collect::<Vec<_>>();
        let odd = b.len() % 2 != 0;
        // CPython consumes the trailing odd byte with an incomplete high
        // surrogate as one replacement; other odd tails replace separately.
        let high = units.last().is_some_and(|v| (0xd800..=0xdbff).contains(v));
        if odd && !high {
            units.push(0xfffd);
        }
        return String::from_utf16_lossy(&units);
    }
    super::codepages::decode(b, cp)
}
pub fn read(
    r: &mut Segments<'_>,
    biff8_mode: bool,
    cp: u16,
    short: bool,
    rich: bool,
) -> Option<Decoded> {
    if biff8_mode {
        biff8(r, short, rich)
    } else {
        byte_string(r, short, cp)
    }
}
pub fn unit_index(text: &str, offset: usize) -> usize {
    let mut units = 0;
    for (i, c) in text.chars().enumerate() {
        if units >= offset {
            return i;
        }
        units += c.len_utf16();
    }
    text.chars().count()
}
pub fn rich(decoded: &Decoded, fonts: &[Font]) -> RichText {
    let mut result = RichText {
        text: decoded.text.clone(),
        runs: vec![],
    };
    if decoded.font_starts.is_empty() || decoded.text.is_empty() {
        return result;
    }
    let mut starts = decoded.font_starts.clone();
    starts.sort_by_key(|s| s.0);
    for (i, (a, font)) in starts.iter().enumerate() {
        let end = starts.get(i + 1).map_or(0x7fff_ffff, |s| s.0 as usize);
        let start = unit_index(&decoded.text, *a as usize);
        let end = unit_index(&decoded.text, end);
        let font = if *font < 4 { *font } else { *font - 1 };
        let style = fonts.get(font as usize).copied().unwrap_or_default();
        if start < end && style != Font::default() {
            result.runs.push(Run { start, end, style });
        }
    }
    result
}
pub fn txo(base: &[u8], parts: &[&[u8]], fonts: &[Font]) -> Option<RichText> {
    if base.len() < 14 {
        return None;
    }
    let count = u16(base, 10)? as usize;
    let size = u16(base, 12)? as usize;
    if count == 0 {
        return Some(RichText::default());
    }
    let mut left = count;
    let mut consumed = 0;
    let mut units = vec![];
    for part in parts {
        if left == 0 {
            break;
        }
        consumed += 1;
        if part.is_empty() {
            continue;
        }
        let wide = part[0] & 1 != 0;
        let b = &part[1..];
        let take = left.min(b.len() / if wide { 2 } else { 1 });
        if wide {
            units.extend(b[..take * 2].chunks_exact(2).map(|b| u16(b, 0).unwrap()));
        } else {
            units.extend(b[..take].iter().map(|b| *b as u16));
        }
        left -= take;
    }
    let mut formatting = Vec::new();
    for part in &parts[consumed..] {
        formatting.extend_from_slice(part);
    }
    formatting.truncate(size);
    let starts = formatting
        .chunks_exact(8)
        .filter_map(|b| {
            let a = u16(b, 0)?;
            if a as usize <= count {
                Some((a, u16(b, 2)?))
            } else {
                None
            }
        })
        .collect();
    Some(rich(
        &Decoded {
            text: clean(&String::from_utf16_lossy(&units)),
            font_starts: starts,
        },
        fonts,
    ))
}
