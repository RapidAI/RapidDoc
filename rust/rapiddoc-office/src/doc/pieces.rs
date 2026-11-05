//! DOC CLX/Pcdt piece table and text extraction.
use super::fib::Pair;
use crate::{Error, Result};
#[derive(Clone, Debug)]
pub struct Piece {
    pub cp_start: u32,
    pub cp_end: u32,
    pub fc: u32,
    pub compressed: bool,
    pub prm: Vec<u8>,
}
fn u16at(d: &[u8], p: usize) -> Option<u16> {
    d.get(p..p + 2).map(|x| u16::from_le_bytes([x[0], x[1]]))
}
fn u32at(d: &[u8], p: usize) -> Option<u32> {
    d.get(p..p + 4)
        .and_then(|x| Some(u32::from_le_bytes(x.try_into().ok()?)))
}
pub fn parse_clx(table: &[u8], pair: Pair) -> Result<Vec<Piece>> {
    let end = pair
        .fc
        .checked_add(pair.lcb)
        .ok_or_else(|| Error::malformed("DOC CLX range overflow"))? as usize;
    let d = table
        .get(pair.fc as usize..end)
        .ok_or_else(|| Error::malformed("DOC CLX range is out of bounds"))?;
    let mut p = 0;
    let mut prcs: Vec<Vec<u8>> = Vec::new();
    while p < d.len() {
        match d[p] {
            1 => {
                let n = u16at(d, p + 1).ok_or_else(|| Error::malformed("DOC CLX Prc truncated"))?
                    as usize;
                let payload = d
                    .get(p + 3..p + 3 + n)
                    .ok_or_else(|| Error::malformed("DOC CLX Prc exceeds its range"))?;
                prcs.push(payload.to_vec());
                p = p
                    .checked_add(3 + n)
                    .ok_or_else(|| Error::malformed("DOC CLX overflow"))?;
                if p > d.len() {
                    return Err(Error::malformed("DOC CLX Prc exceeds its range"));
                }
            }
            2 => {
                let n = u32at(d, p + 1).ok_or_else(|| Error::malformed("DOC CLX Pcdt truncated"))?
                    as usize;
                let plc = d
                    .get(p + 5..p + 5 + n)
                    .ok_or_else(|| Error::malformed("DOC Pcdt exceeds CLX"))?;
                if plc.len() < 16 || (plc.len() - 4) % 12 != 0 {
                    return Err(Error::malformed("DOC piece table is malformed"));
                }
                let count = (plc.len() - 4) / 12;
                let base = (count + 1) * 4;
                let mut out = Vec::with_capacity(count);
                let mut previous = 0;
                for i in 0..count {
                    let a = u32at(plc, i * 4).unwrap();
                    let b = u32at(plc, (i + 1) * 4).unwrap();
                    let raw = u32at(plc, base + i * 8 + 2).unwrap();
                    let prm = u16at(plc, base + i * 8 + 6).unwrap();
                    let properties = if prm & 1 != 0 {
                        prcs.get((prm >> 1) as usize).cloned().unwrap_or_default()
                    } else {
                        let opcode: Option<u16> = match (prm >> 1) & 0x7f {
                            0x0c => Some(0x260a),
                            0x18 => Some(0x2416),
                            0x19 => Some(0x2417),
                            0x55 => Some(0x0835),
                            0x56 => Some(0x0836),
                            0x57 => Some(0x0837),
                            0x78 => Some(0x2640),
                            _ => None,
                        };
                        opcode
                            .map(|op| {
                                let mut v = op.to_le_bytes().to_vec();
                                v.push((prm >> 8) as u8);
                                v
                            })
                            .unwrap_or_default()
                    };
                    if a < previous || b < a {
                        return Err(Error::malformed("DOC piece CP values are not ordered"));
                    }
                    previous = b;
                    out.push(Piece {
                        cp_start: a,
                        cp_end: b,
                        fc: if raw & 0x4000_0000 != 0 {
                            (raw & 0x3fff_ffff) / 2
                        } else {
                            raw & 0x3fff_ffff
                        },
                        compressed: raw & 0x4000_0000 != 0,
                        prm: properties,
                    });
                }
                return Ok(out);
            }
            _ => return Err(Error::malformed("DOC CLX contains an unknown record")),
        }
    }
    Err(Error::malformed("DOC CLX does not contain a Pcdt"))
}
/// Original LID selection for compressed pieces.
fn codec(lid: u16) -> u16 {
    match lid & 0x3ff {
        0x11 => 932,
        0x12 => 949,
        4 => {
            if matches!(lid, 0x0404 | 0x0c04 | 0x1404 | 0x7c04) {
                950
            } else {
                936
            }
        }
        1 | 0x20 | 0x29 => 1256,
        2 | 0x19 | 0x22 | 0x23 => 1251,
        5 | 0x0e | 0x15 | 0x18 | 0x1a | 0x1b | 0x24 => 1250,
        8 => 1253,
        0x0d => 1255,
        0x1e => 874,
        0x1f | 0x2c => 1254,
        0x25 | 0x26 | 0x27 => 1257,
        0x2a => 1258,
        _ => 1252,
    }
}
#[derive(Debug, Default)]
pub struct TextStream {
    pub chars: Vec<char>,
    pub cps: Vec<u32>,
    pub fcs: Vec<u32>,
    pub piece_indexes: Vec<u32>,
}
impl TextStream {
    pub fn text_between(&self, start: u32, end: u32) -> String {
        self.chars[self.cps.partition_point(|n| *n < start)..self.cps.partition_point(|n| *n < end)]
            .iter()
            .collect()
    }
}
pub fn extract_mapped(word: &[u8], pieces: &[Piece], total_cp: u32, lid: u16) -> TextStream {
    // Reserve only recoverable code units. This avoids four power-of-two
    // reallocations while retaining ordinary growth for unusual overlapping input.
    let capacity = pieces
        .iter()
        .take_while(|p| p.cp_start < total_cp)
        .filter_map(|p| {
            let n = p.cp_end.min(total_cp).saturating_sub(p.cp_start) as usize;
            let size = n.checked_mul(if p.compressed { 1 } else { 2 })?;
            word.get(p.fc as usize..(p.fc as usize).checked_add(size)?)
                .map(|_| n)
        })
        .fold(0usize, usize::saturating_add)
        .min(word.len());
    let mut out = TextStream {
        chars: Vec::with_capacity(capacity),
        cps: Vec::with_capacity(capacity),
        fcs: Vec::with_capacity(capacity),
        piece_indexes: Vec::with_capacity(capacity),
    };
    let encoding = codec(lid);
    for (index, x) in pieces.iter().enumerate() {
        if x.cp_start >= total_cp {
            break;
        }
        let n = x.cp_end.min(total_cp).saturating_sub(x.cp_start) as usize;
        let Some(bytes) = n.checked_mul(if x.compressed { 1 } else { 2 }) else {
            continue;
        };
        let Some(end) = (x.fc as usize).checked_add(bytes) else {
            continue;
        };
        let Some(d) = word.get(x.fc as usize..end) else {
            continue;
        };
        let mut cursor = 0;
        let mut cp = x.cp_start;
        while cursor < d.len() {
            let mut push = |c| {
                out.chars.push(c);
                out.cps.push(cp);
                out.fcs.push(x.fc + cursor as u32);
                out.piece_indexes.push(index as u32);
            };
            let width;
            if x.compressed {
                let lead = match encoding {
                    932 => (0x81..=0x9f).contains(&d[cursor]) || (0xe0..=0xfc).contains(&d[cursor]),
                    936 | 949 | 950 => (0x81..=0xfe).contains(&d[cursor]),
                    _ => false,
                };
                width = if lead && cursor + 1 < d.len() { 2 } else { 1 };
                if d[cursor] < 0x80 {
                    push(d[cursor] as char);
                } else {
                    for c in super::codecs::decode(&d[cursor..cursor + width], encoding).chars() {
                        push(c);
                    }
                }
            } else {
                let first = u16at(d, cursor).unwrap();
                if (0xd800..=0xdbff).contains(&first)
                    && u16at(d, cursor + 2).is_some_and(|v| (0xdc00..=0xdfff).contains(&v))
                {
                    width = 4;
                    let second = u16at(d, cursor + 2).unwrap();
                    push(
                        char::from_u32(
                            0x10000 + ((first as u32 - 0xd800) << 10) + (second as u32 - 0xdc00),
                        )
                        .unwrap(),
                    );
                } else {
                    width = 2;
                    push(char::from_u32(first as u32).unwrap_or('\u{fffd}'));
                }
            }
            cp += (width / if x.compressed { 1 } else { 2 }) as u32;
            cursor += width;
        }
    }
    out
}
pub fn extract(word: &[u8], pieces: &[Piece], total_cp: u32, lid: u16) -> String {
    extract_mapped(word, pieces, total_cp, lid)
        .chars
        .into_iter()
        .collect()
}
