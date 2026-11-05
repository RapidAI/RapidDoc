//! Current Python MTEF v3/v5 decoder. Resource limits mirror Python Office.
mod render;
mod v3;
mod v5;
use crate::{ole::OleReader, Error, ErrorKind, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, OnceLock},
};

#[derive(Debug)]
enum Character {
    Code(u32),
    Latex(String),
}
#[derive(Debug)]
struct Char {
    value: Character,
    embellishments: Vec<u8>,
    style: u8,
    face: i32,
    function: bool,
    v5: bool,
}
#[derive(Debug)]
enum Node {
    Sequence(Vec<Node>),
    Character(Char),
    Text(String),
    Function(String),
    Template {
        v5: bool,
        selector: u8,
        variation: u16,
        options: u8,
        slots: Vec<Node>,
    },
    Pile(Vec<Node>),
    Matrix {
        rows: usize,
        cols: usize,
        cells: Vec<Node>,
    },
    Embellishment,
    Metadata,
}
#[derive(Deserialize)]
struct Data {
    characters: BTreeMap<char, String>,
    embellishments: BTreeMap<u8, String>,
    symbol: BTreeMap<u32, u32>,
    extra: BTreeMap<u32, u32>,
    pua: BTreeMap<u32, String>,
    cp1252: BTreeMap<u32, u32>,
}
fn data() -> &'static Data {
    static DATA: OnceLock<Data> = OnceLock::new();
    DATA.get_or_init(|| {
        serde_json::from_str(include_str!("mtef_data.json")).expect("generated MTEF maps")
    })
}
fn bad() -> Error {
    Error::malformed("malformed or unsupported MTEF record")
}
pub(super) fn space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    depth: usize,
    records: usize,
}
impl<'a> Cursor<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self {
            data,
            pos,
            depth: 0,
            records: 0,
        }
    }
    fn bytes(&mut self, size: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(size).ok_or_else(bad)?;
        let bytes = self.data.get(self.pos..end).ok_or_else(bad)?;
        self.pos = end;
        Ok(bytes)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()))
    }
    fn charge(&mut self) -> Result<()> {
        self.records += 1;
        if self.records > 16_000_000 {
            Err(Error::new(
                ErrorKind::ResourceLimit,
                "MTEF exceeds Python max_records",
            ))
        } else {
            Ok(())
        }
    }
    fn enter(&mut self) -> Result<()> {
        self.depth += 1;
        if self.depth > 64 {
            Err(Error::new(
                ErrorKind::ResourceLimit,
                "MTEF exceeds Python max_record_depth",
            ))
        } else {
            Ok(())
        }
    }
    fn nudge(&mut self) -> Result<()> {
        let dx = self.u8()?;
        let dy = self.u8()?;
        if dx == 128 && dy == 128 {
            self.bytes(4)?;
        }
        Ok(())
    }
    fn cstring(&mut self) -> Result<String> {
        let limit = self.data.len().min(self.pos.saturating_add(4097));
        let size = self
            .data
            .get(self.pos..limit)
            .ok_or_else(bad)?
            .iter()
            .position(|b| *b == 0)
            .ok_or_else(bad)?;
        let out = self.bytes(size)?.iter().map(|b| char::from(*b)).collect();
        self.u8()?;
        Ok(out)
    }
    fn unsigned(&mut self) -> Result<u16> {
        let b = self.u8()?;
        if b == 255 {
            self.u16()
        } else {
            Ok(b.into())
        }
    }
    fn signed(&mut self) -> Result<i32> {
        let b = self.u8()?;
        Ok(if b == 255 {
            i32::from(self.u16()?) - 32768
        } else {
            i32::from(b) - 128
        })
    }
}
fn recover(result: Result<String>) -> Result<Option<String>> {
    match result {
        Ok(latex) => {
            let latex = latex.trim_matches(space);
            Ok(if latex.is_empty() {
                None
            } else {
                Some(latex.into())
            })
        }
        Err(e) if e.kind == ErrorKind::ResourceLimit => Err(e),
        Err(_) => Ok(None),
    }
}
pub fn decode(bytes: &[u8]) -> Result<Option<String>> {
    let parsed = match bytes.first() {
        Some(3) => v3::parse(bytes),
        Some(5) => v5::parse(bytes),
        _ => return Ok(None),
    };
    recover(parsed.and_then(|n| render::node(&n)))
}
pub fn equation_native(bytes: &[u8]) -> Result<Option<String>> {
    if bytes.len() < 28 {
        return Ok(None);
    }
    let header = u16::from_le_bytes(bytes[..2].try_into().unwrap()) as usize;
    let version = u32::from_le_bytes(bytes[2..6].try_into().unwrap());
    let size = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if header < 28 || header > bytes.len() || version != 0x0002_0000 {
        return Ok(None);
    }
    let Some(end) = header.checked_add(size).filter(|end| *end <= bytes.len()) else {
        return Ok(None);
    };
    decode(&bytes[header..end])
}
pub fn equation_object(bytes: Arc<[u8]>) -> Result<Option<String>> {
    let result = (|| {
        let mut ole = OleReader::open(bytes)?;
        let native = ole.read_stream("Equation Native", true)?;
        equation_native(&native)
    })();
    match result {
        Err(e) if e.kind != ErrorKind::ResourceLimit => Ok(None),
        other => other,
    }
}
pub fn equation_prog_id(id: Option<&str>) -> bool {
    let id = crate::ole::casefold(id.unwrap_or("").trim_matches(space));
    id == "equation" || id.starts_with("equation.") && id.len() > 9
}
#[derive(Default)]
pub struct OoxmlDecoder {
    pub total_bytes: usize,
    cache: HashMap<[u8; 32], Option<String>>,
}
impl OoxmlDecoder {
    pub fn decode(
        &mut self,
        bytes: Option<Arc<[u8]>>,
        prog_id: Option<&str>,
        icon: bool,
    ) -> Result<Option<String>> {
        if icon || !equation_prog_id(prog_id) {
            return Ok(None);
        }
        let Some(bytes) = bytes else { return Ok(None) };
        if bytes.len() > 128 * 1024 * 1024 {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "OOXML equation exceeds Python max_entry_bytes",
            ));
        }
        if !bytes.starts_with(&[0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1]) {
            return Ok(None);
        }
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        if let Some(value) = self.cache.get(&digest) {
            return Ok(value.clone());
        }
        if bytes.len() > (128usize * 1024 * 1024).saturating_sub(self.total_bytes) {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "OOXML equations exceed Python max_asset_total_bytes",
            ));
        }
        self.total_bytes += bytes.len();
        let value = equation_object(bytes)?;
        self.cache.insert(digest, value.clone());
        Ok(value)
    }
}
pub fn object_pool(ole: &mut OleReader) -> Result<BTreeMap<String, String>> {
    let names: Vec<_> = ole
        .stream_names(Some("ObjectPool/"))
        .into_iter()
        .map(str::to_owned)
        .collect();
    let mut out = BTreeMap::new();
    static POOL: OnceLock<regex::Regex> = OnceLock::new();
    let pattern = POOL.get_or_init(|| {
        regex::Regex::new("(?i)^ObjectPool/_([0-9]+)/Equat[iİı]on Nat[iİı]ve\\n?$")
            .expect("ObjectPool pattern")
    });
    for name in names {
        let Some(capture) = pattern.captures(&name) else {
            continue;
        };
        let id = &capture[1];
        let key = id.trim_start_matches('0');
        let key = if key.is_empty() { "0" } else { key };
        let bytes = ole.read_stream(&name, true)?;
        if let Some(latex) = equation_native(&bytes)? {
            out.entry(key.into()).or_insert(latex);
        }
    }
    Ok(out)
}
