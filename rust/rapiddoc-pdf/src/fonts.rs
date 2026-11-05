//! 固定 CJK 替代字体；资源和精确编码表编入运行库，回调不进入 Python。
#[allow(dead_code)] // interface_address is consumed only by the internal ABI test harness.
mod provider;
use crate::{Error, ErrorCode};
use libloading::Library;
pub(crate) use provider::FontProvider;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{Arc, OnceLock},
};
pub const POLICY: &str = "droid-cjk-v1";
pub const SHA256: &str = "8a4dea0899424438af25a6f1f6eb61e5d111d85367eece0a5d30170861ae6b2e";
const FONT: &[u8] = include_bytes!("../resources/fonts/DroidSansFallbackFull.ttf");
#[derive(Deserialize)]
struct Policy {
    aliases: HashMap<String, i32>,
    suffixes: Vec<String>,
    casefold: HashMap<char, String>,
}
fn policy() -> &'static Policy {
    static P: OnceLock<Policy> = OnceLock::new();
    P.get_or_init(|| {
        serde_json::from_str(include_str!("../resources/fonts/policy.json"))
            .expect("pinned font policy")
    })
}
fn ranges() -> &'static [[u32; 3]] {
    static R: OnceLock<Vec<[u32; 3]>> = OnceLock::new();
    R.get_or_init(|| {
        serde_json::from_str(include_str!("../resources/fonts/gb18030-ranges.json"))
            .expect("pinned codec ranges")
    })
}
const CODECS: [&[u8]; 4] = [
    include_bytes!("../resources/fonts/gb18030.bin"),
    include_bytes!("../resources/fonts/big5.bin"),
    include_bytes!("../resources/fonts/cp932.bin"),
    include_bytes!("../resources/fonts/cp949.bin"),
];
fn code(table: &[u8], index: usize) -> Option<char> {
    let offset = index.checked_mul(4)?;
    char::from_u32(u32::from_le_bytes(
        table.get(offset..offset + 4)?.try_into().ok()?,
    ))
}
fn decode(bytes: &[u8], codec: usize) -> Option<String> {
    let table = CODECS[codec];
    let mut i = 0;
    let mut text = String::new();
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(c) = code(table, b as usize) {
            text.push(c);
            i += 1;
            continue;
        }
        if codec == 0 && bytes.get(i + 1).is_some_and(|b| (0x30..=0x39).contains(b)) {
            let part = bytes.get(i..i + 4)?;
            if !(0x81..=0xfe).contains(&part[0])
                || !(0x81..=0xfe).contains(&part[2])
                || !(0x30..=0x39).contains(&part[3])
            {
                return None;
            }
            let index = (((part[0] as u32 - 0x81) * 10 + (part[1] as u32 - 0x30)) * 126
                + (part[2] as u32 - 0x81))
                * 10
                + (part[3] as u32 - 0x30);
            let rs = ranges();
            let n = rs.partition_point(|r| r[0] <= index);
            let r = rs.get(n.checked_sub(1)?)?;
            if index > r[1] {
                return None;
            }
            text.push(char::from_u32(r[2] + index - r[0])?);
            i += 4;
        } else {
            let second = *bytes.get(i + 1)?;
            text.push(code(table, 256 + (b as usize) * 256 + second as usize)?);
            i += 2;
        }
    }
    Some(text)
}
fn normalized(name: &str) -> Option<i32> {
    let p = policy();
    let mut name = name.trim_start_matches('@');
    let b = name.as_bytes();
    if b.len() >= 7 && b[..6].iter().all(u8::is_ascii_uppercase) && b[6] == b'+' {
        name = &name[7..];
    }
    let mut folded = String::new();
    for c in name
        .chars()
        .filter(|c| !c.is_whitespace() && !matches!(c, '\u{1c}'..='\u{1f}' | ',' | '_' | '-'))
    {
        if let Some(s) = p.casefold.get(&c) {
            folded.push_str(s);
        } else {
            folded.push(c);
        }
    }
    for c in std::iter::once(folded.as_str()).chain(
        p.suffixes
            .iter()
            .filter_map(|s| folded.strip_suffix(s.as_str())),
    ) {
        if let Some(&v) = p.aliases.get(c) {
            return Some(v);
        }
        for (marker, value) in [("gb2312", 134), ("gbk", 134), ("big5", 136)] {
            if c.ends_with(marker) {
                return Some(value);
            }
        }
    }
    None
}
/// 旧五编码顺序、严格解码及名称匹配，不做任意子串字体猜测。
pub fn classify(face: &[u8], charset: i32) -> Option<i32> {
    if [128, 129, 134, 136].contains(&charset) {
        return Some(charset);
    }
    if ![0, 1].contains(&charset) {
        return None;
    }
    if let Ok(name) = std::str::from_utf8(face) {
        if let Some(v) = normalized(name) {
            return Some(v);
        }
    }
    for codec in 0..4 {
        if let Some(name) = decode(face, codec) {
            if let Some(v) = normalized(&name) {
                return Some(v);
            }
        }
    }
    None
}
unsafe extern "C" fn legacy_classify(data: *const u8, len: usize, charset: i32) -> i32 {
    // SAFETY: provider lends the face slice for this synchronous call under the runtime lock.
    let bytes = unsafe { std::slice::from_raw_parts(data, len) };
    classify(bytes, charset).unwrap_or(-1)
}
fn unavailable(message: impl ToString) -> Error {
    Error {
        code: ErrorCode::RuntimeUnavailable,
        message: format!("PDFium fixed CJK font: {}", message.to_string()),
    }
}
pub(crate) fn install(symbols: &Library) -> Result<Arc<FontProvider>, Error> {
    if format!("{:x}", Sha256::digest(FONT)) != SHA256 {
        return Err(unavailable("font SHA256 mismatch"));
    }
    if FONT.get(..4) != Some(&[0, 1, 0, 0]) {
        return Err(unavailable("invalid TrueType header"));
    }
    let read16 = |at: usize| -> Result<usize, Error> {
        Ok(u16::from_be_bytes(
            FONT.get(at..at + 2)
                .ok_or_else(|| unavailable("truncated font"))?
                .try_into()
                .unwrap(),
        ) as usize)
    };
    let read32 = |at: usize| -> Result<usize, Error> {
        Ok(u32::from_be_bytes(
            FONT.get(at..at + 4)
                .ok_or_else(|| unavailable("truncated font"))?
                .try_into()
                .unwrap(),
        ) as usize)
    };
    let mut tables = HashMap::new();
    for i in 0..read16(4)? {
        let at = 12 + i * 16;
        let tag = read32(at)? as u32;
        let offset = read32(at + 8)?;
        let len = read32(at + 12)?;
        if offset.checked_add(len).is_none_or(|end| end > FONT.len()) {
            return Err(unavailable("SFNT table bounds"));
        }
        tables.insert(tag, (offset, len));
    }
    // Force immutable policy/range parsing before installing any C callbacks.
    let p = policy();
    let _ = ranges();
    // SAFETY: same canonical DLL as Runtime; symbols and provider live for its process lifetime.
    let fonts = unsafe {
        let get = symbols
            .get::<unsafe extern "system" fn() -> *mut c_void>(b"FPDF_GetDefaultSystemFontInfo\0")
            .map_err(unavailable)?;
        let free = symbols
            .get::<unsafe extern "system" fn(*mut c_void)>(b"FPDF_FreeDefaultSystemFontInfo\0")
            .map_err(unavailable)?;
        let set = symbols
            .get::<unsafe extern "system" fn(*mut c_void)>(b"FPDF_SetSystemFontInfo\0")
            .map_err(unavailable)?;
        FontProvider::new(
            [*get as usize, *free as usize, *set as usize],
            FONT.to_vec(),
            tables,
            format!("DocVortex-{POLICY}-{SHA256}").into_bytes(),
            p.aliases.clone(),
            p.suffixes.clone(),
            legacy_classify as *const () as usize,
        )
        .map_err(unavailable)?
    };
    fonts.install().map_err(unavailable)?;
    Ok(fonts)
}
pub(crate) fn check(fonts: &FontProvider) -> Result<(), Error> {
    fonts.check().map_err(unavailable)
}

#[cfg(test)]
mod tests;
