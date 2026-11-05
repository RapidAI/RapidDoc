//! Variable Word 97+ File Information Block parsing.
use crate::{Error, ErrorKind, Result};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Pair {
    pub fc: u32,
    pub lcb: u32,
}
#[derive(Clone, Debug, Serialize)]
pub struct Base {
    pub n_fib: u16,
    pub lid: u16,
    pub flags: u16,
    pub fc_min: u32,
    pub fc_mac: u32,
}
impl Base {
    pub fn encrypted(&self) -> bool {
        self.flags & 0x0100 != 0
    }
    pub fn obfuscated(&self) -> bool {
        self.flags & 0x8000 != 0
    }
    pub fn uses_1table(&self) -> bool {
        self.flags & 0x0200 != 0
    }
    pub fn complex(&self) -> bool {
        self.flags & 4 != 0
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Fib {
    pub base: Base,
    pub rgw: Vec<u16>,
    pub rglw: Vec<u32>,
    pub pairs: Vec<Pair>,
    pub csw_new: Vec<u16>,
    pub size: usize,
}
impl Fib {
    pub fn pair(&self, index: usize) -> Pair {
        self.pairs.get(index).copied().unwrap_or_default()
    }
    pub fn ccp_text(&self) -> u32 {
        self.rglw.get(3).copied().unwrap_or(0)
    }
    pub fn clx(&self) -> Pair {
        self.pair(33)
    }
    pub fn total_story_cp(&self) -> u32 {
        (3..=10)
            .map(|i| self.rglw.get(i).copied().unwrap_or(0))
            .fold(0, u32::saturating_add)
    }
    pub fn story_base(&self, story: usize) -> u32 {
        (3..story.min(11))
            .map(|i| self.rglw.get(i).copied().unwrap_or(0))
            .fold(0, u32::saturating_add)
    }
    pub fn text_lid(&self, word: &[u8]) -> u16 {
        if self.base.flags & 0x4000 != 0 {
            let fixed = word
                .get(0x3c..0x3e)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .unwrap_or(0);
            if fixed != 0 {
                return fixed;
            }
            if let Some(lid) = self.rgw.get(13).copied().filter(|v| *v != 0) {
                return lid;
            }
        }
        self.base.lid
    }
}
fn u16at(d: &[u8], p: &mut usize) -> Result<u16> {
    let e = p
        .checked_add(2)
        .ok_or_else(|| Error::malformed("FIB offset overflow"))?;
    if e > d.len() {
        return Err(Error::malformed("truncated FIB"));
    };
    let v = u16::from_le_bytes([d[*p], d[*p + 1]]);
    *p = e;
    Ok(v)
}
fn u32at(d: &[u8], p: &mut usize) -> Result<u32> {
    let e = p
        .checked_add(4)
        .ok_or_else(|| Error::malformed("FIB offset overflow"))?;
    if e > d.len() {
        return Err(Error::malformed("truncated FIB"));
    };
    let v = u32::from_le_bytes(d[*p..e].try_into().unwrap());
    *p = e;
    Ok(v)
}
pub fn parse(d: &[u8]) -> Result<Fib> {
    if d.len() < 34 {
        return Err(Error::malformed("WordDocument FIB is truncated"));
    }
    let ident = u16::from_le_bytes([d[0], d[1]]);
    let n = u16::from_le_bytes([d[2], d[3]]);
    if ident != 0xa5ec {
        return Err(Error::malformed("WordDocument FIB magic is invalid"));
    }
    if n < 0x00c1 {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "Word 95 or earlier DOC is unsupported",
        ));
    }
    let mut p = 32;
    let csw = u16at(d, &mut p)? as usize;
    let mut rgw = Vec::with_capacity(csw);
    for _ in 0..csw {
        rgw.push(u16at(d, &mut p)?);
    }
    let cslw = u16at(d, &mut p)? as usize;
    let mut rglw = Vec::with_capacity(cslw);
    for _ in 0..cslw {
        rglw.push(u32at(d, &mut p)?);
    }
    let count = u16at(d, &mut p)? as usize;
    let mut pairs = Vec::with_capacity(count);
    for _ in 0..count {
        pairs.push(Pair {
            fc: u32at(d, &mut p)?,
            lcb: u32at(d, &mut p)?,
        });
    }
    let mut csw_new = Vec::new();
    if p + 2 <= d.len() {
        let count = u16at(d, &mut p)? as usize;
        for _ in 0..count {
            csw_new.push(u16at(d, &mut p)?);
        }
    }
    // Original parser recovery for minimal Word97 fixtures with fixed slots.
    if rglw.len() <= 3 {
        if d.len() < 0x6c {
            return Err(Error::malformed("FIB does not contain ccpText"));
        }
        rglw = (0..11)
            .map(|i| {
                let start = 0x40 + i * 4;
                u32::from_le_bytes(d[start..start + 4].try_into().unwrap())
            })
            .collect();
        if pairs.is_empty() && d.len() >= 0x382 {
            pairs = (0..93)
                .map(|i| {
                    let start = 0x9a + i * 8;
                    Pair {
                        fc: u32::from_le_bytes(d[start..start + 4].try_into().unwrap()),
                        lcb: u32::from_le_bytes(d[start + 4..start + 8].try_into().unwrap()),
                    }
                })
                .collect();
        }
        p = p.max(0x382);
    }
    Ok(Fib {
        base: Base {
            n_fib: n,
            lid: u16::from_le_bytes([d[6], d[7]]),
            flags: u16::from_le_bytes([d[10], d[11]]),
            fc_min: u32::from_le_bytes(d[24..28].try_into().unwrap()),
            fc_mac: u32::from_le_bytes(d[28..32].try_into().unwrap()),
        },
        rgw,
        rglw,
        pairs,
        csw_new,
        size: p,
    })
}
