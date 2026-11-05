//! WMF/GIF MathType comments, following the current Python image decoder.
//! Ordinary image frames are traversed without collecting their pixel payload.
use super::mtef;
use crate::{Error, ErrorKind, Result};
use sha2::{Digest, Sha256};
use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap},
};
const MAX: usize = 128 * 1024 * 1024;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    Wmf,
    Gif,
}
#[derive(Default)]
pub struct Decoder {
    pub total_bytes: usize,
    cache: HashMap<(Format, [u8; 32]), Option<String>>,
    rejected: HashMap<(Format, [u8; 32]), String>,
    pub warnings: Vec<(Format, [u8; 32])>,
}
fn bad() -> Error {
    Error::malformed("malformed or conflicting MathType image comment")
}
fn limit() -> Error {
    Error::new(
        ErrorKind::ResourceLimit,
        "image equation exceeds existing Python candidate/record limit",
    )
}
fn u16(b: &[u8], p: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        b.get(p..p + 2).ok_or_else(bad)?.try_into().unwrap(),
    ))
}
fn u32(b: &[u8], p: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        b.get(p..p + 4).ok_or_else(bad)?.try_into().unwrap(),
    ))
}
fn standard_wmf(b: &[u8]) -> bool {
    b.len() >= 4 && matches!(&b[..2], [1, 0] | [2, 0]) && b[2..4] == [9, 0]
}
fn hint(b: &[u8], part: Option<&str>, mime: Option<&str>) -> Option<Format> {
    if b.starts_with(b"GIF87a") || b.starts_with(b"GIF89a") {
        return Some(Format::Gif);
    }
    if b.starts_with(&[0xd7, 0xcd, 0xc6, 0x9a]) || standard_wmf(b) {
        return Some(Format::Wmf);
    }
    let name = part.unwrap_or("").rsplit('/').next().unwrap_or("");
    let suffix = name
        .rsplit_once('.')
        .filter(|(base, _)| !base.is_empty())
        .map(|(_, s)| crate::ole::casefold(s))
        .unwrap_or_default();
    let mime = crate::ole::casefold(
        mime.unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim_matches(mtef::space),
    );
    if suffix == "gif" || mime == "image/gif" {
        Some(Format::Gif)
    } else if suffix == "wmf"
        || matches!(
            mime.as_str(),
            "image/wmf" | "image/x-wmf" | "application/x-msmetafile"
        )
    {
        Some(Format::Wmf)
    } else {
        None
    }
}
impl Decoder {
    /// Production image boundary: only image-local resource errors become placeholders.
    pub fn probe(
        &mut self,
        bytes: &[u8],
        part: Option<&str>,
        mime: Option<&str>,
    ) -> Result<Option<String>> {
        if self.limit_reason(bytes, part, mime).is_some() {
            return Ok(None);
        }
        match self.decode(bytes, part, mime) {
            Err(error) if error.kind == ErrorKind::ResourceLimit => {
                if let Some(format) = hint(bytes, part, mime) {
                    self.rejected
                        .insert((format, Sha256::digest(bytes).into()), error.to_string());
                }
                Ok(None)
            }
            result => result,
        }
    }
    pub fn limit_reason(
        &self,
        bytes: &[u8],
        part: Option<&str>,
        mime: Option<&str>,
    ) -> Option<&str> {
        let format = hint(bytes, part, mime)?;
        self.rejected
            .get(&(format, Sha256::digest(bytes).into()))
            .map(String::as_str)
    }

    pub fn decode(
        &mut self,
        bytes: &[u8],
        part: Option<&str>,
        mime: Option<&str>,
    ) -> Result<Option<String>> {
        let Some(format) = hint(bytes, part, mime) else {
            return Ok(None);
        };
        if format == Format::Wmf && bytes.len() > MAX {
            return Err(limit());
        }
        let digest: [u8; 32] = Sha256::digest(bytes).into();
        let key = (format, digest);
        if let Some(value) = self.cache.get(&key) {
            return Ok(value.clone());
        }
        let result = match format {
            Format::Wmf => wmf(bytes),
            Format::Gif => gif(bytes, MAX.saturating_sub(self.total_bytes)),
        };
        let (latex, recognized) = match result {
            Ok((candidates, recognized)) => {
                let amount: usize = candidates.iter().map(|b| b.len()).sum();
                if amount > MAX.saturating_sub(self.total_bytes) {
                    return Err(limit());
                }
                self.total_bytes += amount;
                let mut values = BTreeSet::new();
                for candidate in candidates {
                    if let Some(latex) = mtef::decode(&candidate)? {
                        values.insert(latex);
                    }
                }
                (
                    if values.len() == 1 {
                        values.into_iter().next()
                    } else {
                        None
                    },
                    recognized,
                )
            }
            Err(e) if e.kind == ErrorKind::ResourceLimit => return Err(e),
            Err(_) => (
                None,
                bytes
                    .windows(8)
                    .any(|s| matches!(s, b"MathType" | b"AppsMFCC")),
            ),
        };
        self.cache.insert(key, latex.clone());
        if recognized && latex.is_none() {
            self.warnings.push(key);
        }
        Ok(latex)
    }
}
type Candidates<'a> = (Vec<Cow<'a, [u8]>>, bool);
struct Chunk<'a> {
    key: Vec<String>,
    total: usize,
    data: &'a [u8],
}
fn apps(comment: &[u8]) -> Result<Option<Chunk<'_>>> {
    if !comment.starts_with(b"AppsMFCC") {
        return Ok(None);
    }
    if comment.len() > 0x7ffe || comment.len() < 19 || u16(comment, 8)? != 1 {
        return Err(bad());
    }
    let total = u32(comment, 10)? as usize;
    let size = u32(comment, 14)? as usize;
    if total == 0 || size == 0 || size > total || total > MAX {
        return Err(bad());
    }
    let end = comment[18..comment.len().min(4115)]
        .iter()
        .position(|b| *b == 0)
        .ok_or_else(bad)?
        + 18;
    let signature = &comment[18..end];
    if !signature.is_ascii() {
        return Err(bad());
    }
    let mut components = Vec::new();
    let mut current = String::new();
    let mut escaped = false;
    for c in signature.iter().map(|b| char::from(*b)) {
        if escaped {
            current.push(c);
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '/' {
            components.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    if escaped {
        return Err(bad());
    }
    components.push(current);
    if components.iter().any(String::is_empty) {
        return Err(bad());
    }
    let normalized: Vec<_> = components
        .iter()
        .map(|s| crate::ole::casefold(s.trim_matches(mtef::space)))
        .collect();
    if !(normalized.len() >= 2 && normalized[1] == "mtef"
        || normalized.len() == 1
            && matches!(
                normalized[0].as_str(),
                "design science" | "design science, inc." | "wiris"
            ))
    {
        return Ok(None);
    }
    let start = end + 1;
    let end = start
        .checked_add(size)
        .filter(|n| *n <= comment.len())
        .ok_or_else(bad)?;
    if comment[end..].iter().any(|b| *b != 0) {
        return Err(bad());
    }
    Ok(Some(Chunk {
        key: std::iter::once("1".into())
            .chain(normalized.into_iter().take(2))
            .collect(),
        total,
        data: &comment[start..end],
    }))
}
fn pre6(comment: &[u8]) -> Result<Option<&[u8]>> {
    if !comment.starts_with(b"MathType") {
        return Ok(None);
    }
    if comment.len() > 0x7ffe || comment.len() < 12 {
        return Err(bad());
    }
    if u16(comment, 8)? != 0x5555 {
        return Ok(None);
    }
    let end = 12 + usize::from(u16(comment, 10)?);
    let data = comment.get(12..end).ok_or_else(bad)?;
    if comment[end..].iter().any(|b| *b != 0) {
        return Err(bad());
    }
    Ok(Some(data))
}
fn wmf(bytes: &[u8]) -> Result<Candidates<'_>> {
    let start = if bytes.starts_with(&[0xd7, 0xcd, 0xc6, 0x9a]) {
        22
    } else {
        0
    };
    if start + 18 > bytes.len() {
        return Ok((vec![], false));
    }
    if !matches!(u16(bytes, start)?, 1 | 2) || u16(bytes, start + 2)? != 9 {
        return Ok((vec![], false));
    }
    let end = start
        .checked_add(
            (u32(bytes, start + 6)? as usize)
                .checked_mul(2)
                .ok_or_else(bad)?,
        )
        .ok_or_else(bad)?;
    if end < start + 18 || end > bytes.len() {
        return Err(bad());
    }
    let mut cursor = start + 18;
    let mut count = 0;
    let mut eof = false;
    let mut comments = Vec::new();
    while cursor < end {
        count += 1;
        if count > 100_000 {
            return Err(limit());
        }
        if cursor + 6 > end {
            return Err(bad());
        }
        let words = u32(bytes, cursor)? as usize;
        let function = u16(bytes, cursor + 4)?;
        if words < 3 {
            return Err(bad());
        }
        let next = cursor
            .checked_add(words.checked_mul(2).ok_or_else(bad)?)
            .filter(|n| *n > cursor && *n <= end)
            .ok_or_else(bad)?;
        if function == 0 {
            eof = true;
            cursor = next;
            break;
        }
        if function == 0x0626 {
            if cursor + 10 > next {
                return Err(bad());
            }
            let escape = u16(bytes, cursor + 6)?;
            let size = usize::from(u16(bytes, cursor + 8)?);
            let data_end = (cursor + 10)
                .checked_add(size)
                .filter(|n| *n <= next)
                .ok_or_else(bad)?;
            if escape == 15 {
                comments.push(&bytes[cursor + 10..data_end]);
            }
        }
        cursor = next;
    }
    if !eof || cursor != end || bytes[end..].iter().any(|b| *b != 0) {
        return Err(bad());
    }
    let mut candidates = Vec::new();
    let mut recognized = false;
    let mut pending: Option<(Vec<String>, usize, Vec<u8>)> = None;
    for comment in comments {
        if let Some(data) = pre6(comment)? {
            recognized = true;
            candidates.push(Cow::Borrowed(data));
            continue;
        }
        let is_apps = comment.starts_with(b"AppsMFCC");
        let chunk = match apps(comment) {
            Ok(c) => c,
            Err(_) => {
                recognized = true;
                pending = None;
                continue;
            }
        };
        let Some(chunk) = chunk else {
            if is_apps && pending.is_some() {
                recognized = true;
                pending = None;
            }
            continue;
        };
        recognized = true;
        if let Some((key, total, mut data)) = pending.take() {
            if key == chunk.key && total == chunk.total {
                data.extend_from_slice(chunk.data);
                if data.len() == total {
                    candidates.push(Cow::Owned(data));
                } else if data.len() < total {
                    pending = Some((key, total, data));
                }
                continue;
            }
        }
        if chunk.data.len() == chunk.total {
            candidates.push(Cow::Borrowed(chunk.data));
        } else {
            pending = Some((chunk.key, chunk.total, chunk.data.to_vec()));
        }
    }
    if pending.is_some() {
        recognized = true;
    }
    Ok((candidates, recognized))
}
fn subblocks(
    bytes: &[u8],
    cursor: &mut usize,
    count: &mut usize,
    collect: bool,
    remaining: usize,
) -> Result<Option<Vec<u8>>> {
    let mut data = Vec::new();
    loop {
        *count += 1;
        if *count > 16_000_000 {
            return Err(limit());
        }
        let size = usize::from(*bytes.get(*cursor).ok_or_else(bad)?);
        *cursor += 1;
        if size == 0 {
            return Ok(if collect { Some(data) } else { None });
        }
        let end = cursor
            .checked_add(size)
            .filter(|n| *n <= bytes.len())
            .ok_or_else(bad)?;
        if collect {
            if size > MAX.saturating_sub(data.len()) || size > remaining.saturating_sub(data.len())
            {
                return Err(limit());
            }
            data.extend_from_slice(&bytes[*cursor..end]);
        }
        *cursor = end;
    }
}
fn gif(bytes: &[u8], mut remaining: usize) -> Result<Candidates<'_>> {
    if bytes.len() < 13 || !(bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
        return Ok((vec![], false));
    }
    let mut cursor = 13;
    let packed = bytes[10];
    if packed & 128 != 0 {
        cursor += 3 * (1usize << ((packed & 7) + 1));
    }
    if cursor > bytes.len() {
        return Err(bad());
    }
    let mut count = 0;
    let mut blocks = 0;
    let mut eof = false;
    let mut candidates = Vec::new();
    let mut recognized = false;
    while cursor < bytes.len() {
        count += 1;
        if count > 100_000 {
            return Err(limit());
        }
        let marker = bytes[cursor];
        cursor += 1;
        if marker == 0x3b {
            eof = true;
            break;
        }
        if marker == 0x2c {
            if cursor + 9 > bytes.len() {
                return Err(bad());
            }
            let packed = bytes[cursor + 8];
            cursor += 9;
            if packed & 128 != 0 {
                cursor += 3 * (1usize << ((packed & 7) + 1));
            }
            if cursor >= bytes.len() {
                return Err(bad());
            }
            cursor += 1;
            subblocks(bytes, &mut cursor, &mut blocks, false, remaining)?;
            continue;
        }
        if marker != 0x21 {
            return Err(bad());
        }
        let extension = *bytes.get(cursor).ok_or_else(bad)?;
        cursor += 1;
        if extension != 0xff {
            subblocks(bytes, &mut cursor, &mut blocks, false, remaining)?;
            continue;
        }
        let size = usize::from(*bytes.get(cursor).ok_or_else(bad)?);
        cursor += 1;
        let application = bytes.get(cursor..cursor + size).ok_or_else(bad)?;
        cursor += size;
        let mathtype = size == 11 && application == b"MathType001";
        if let Some(payload) = subblocks(bytes, &mut cursor, &mut blocks, mathtype, remaining)? {
            recognized = true;
            remaining -= payload.len();
            candidates.push(Cow::Owned(payload));
        }
    }
    if !eof || bytes[cursor..].iter().any(|b| *b != 0) {
        return Err(bad());
    }
    Ok((candidates, recognized))
}
