//! Borrowed BIFF records and CONTINUE fields, matching the original reader.
use crate::{Error, ErrorKind, Result};
pub const BOF: u16 = 0x0809;
pub const EOF: u16 = 0x000a;
pub const CONTINUE: u16 = 0x003c;
pub fn u16(b: &[u8], p: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        b.get(p..p.checked_add(2)?)?.try_into().ok()?,
    ))
}
pub fn u32(b: &[u8], p: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        b.get(p..p.checked_add(4)?)?.try_into().ok()?,
    ))
}
pub fn f64(b: &[u8], p: usize) -> Option<f64> {
    Some(f64::from_le_bytes(
        b.get(p..p.checked_add(8)?)?.try_into().ok()?,
    ))
}
#[derive(Clone, Copy)]
pub struct Record<'a> {
    pub offset: usize,
    pub kind: u16,
    pub payload: &'a [u8],
    pub next: usize,
}
#[derive(Default)]
pub struct Budget {
    pub count: usize,
}
impl Budget {
    pub fn charge(&mut self) -> Result<()> {
        self.count += 1;
        if self.count > 16_000_000 {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "workbook stream exceeds original max_records",
            ));
        }
        Ok(())
    }
}
pub fn at(data: &[u8], offset: usize) -> Option<Record<'_>> {
    let kind = u16(data, offset)?;
    let length = u16(data, offset.checked_add(2)?)? as usize;
    let start = offset.checked_add(4)?;
    let next = start.checked_add(length)?;
    Some(Record {
        offset,
        kind,
        payload: data.get(start..next)?,
        next,
    })
}
pub fn charged<'a>(
    data: &'a [u8],
    offset: usize,
    budget: &mut Budget,
) -> Result<Option<Record<'a>>> {
    let r = at(data, offset);
    if r.is_some() {
        budget.charge()?
    }
    Ok(r)
}
pub fn continues<'a>(
    data: &'a [u8],
    base: Record<'a>,
    budget: &mut Budget,
) -> Result<(Vec<&'a [u8]>, usize)> {
    let mut segments = vec![base.payload];
    let mut cursor = base.next;
    while let Some(r) = at(data, cursor).filter(|r| r.kind == CONTINUE) {
        budget.charge()?;
        segments.push(r.payload);
        cursor = r.next;
    }
    Ok((segments, cursor))
}
pub struct Segments<'a> {
    pub parts: Vec<&'a [u8]>,
    pub index: usize,
    pub offset: usize,
}
impl<'a> Segments<'a> {
    pub fn new(parts: Vec<&'a [u8]>) -> Self {
        Self {
            parts,
            index: 0,
            offset: 0,
        }
    }
    pub fn remaining(&self) -> usize {
        self.parts
            .get(self.index)
            .map_or(0, |b| b.len() - self.offset)
    }
    pub fn normalize(&mut self) {
        while self
            .parts
            .get(self.index)
            .is_some_and(|b| self.offset >= b.len())
        {
            self.index += 1;
            self.offset = 0;
        }
    }
    pub fn next_segment(&mut self) -> bool {
        if self.index + 1 >= self.parts.len() {
            return false;
        }
        self.index += 1;
        self.offset = 0;
        true
    }
    pub fn read(&mut self, size: usize) -> Option<&'a [u8]> {
        self.normalize();
        let end = self.offset.checked_add(size)?;
        let out = self.parts.get(self.index)?.get(self.offset..end)?;
        self.offset = end;
        Some(out)
    }
    pub fn across(&mut self, size: usize) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        let mut left = size;
        while left > 0 {
            self.normalize();
            let take = self.remaining().min(left);
            if take == 0 {
                return None;
            }
            out.extend_from_slice(self.read(take)?);
            left -= take;
        }
        Some(out)
    }
    pub fn skip(&mut self, size: usize) -> bool {
        let mut left = size;
        while left > 0 {
            self.normalize();
            let take = self.remaining().min(left);
            if take == 0 {
                return false;
            }
            self.offset += take;
            left -= take;
        }
        true
    }
    pub fn u8(&mut self) -> Option<u8> {
        Some(self.read(1)?[0])
    }
    pub fn u16(&mut self) -> Option<u16> {
        u16(self.read(2)?, 0)
    }
    pub fn u32(&mut self) -> Option<u32> {
        u32(self.read(4)?, 0)
    }
}
