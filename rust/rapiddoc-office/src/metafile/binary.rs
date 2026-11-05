// Copyright (c) 2026 Xiaomeng Zhao (myhloli). SPDX-License-Identifier: MIT
use super::{header::Source, malformed, model::*, resource};
use crate::Result;
#[derive(Debug, Clone, Copy)]
pub struct Reader<'a> {
    pub data: &'a [u8],
    pub base: usize,
}
impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, base: 0 }
    }
    pub fn bytes(self, offset: usize, size: usize) -> Result<&'a [u8]> {
        self.data
            .get(offset..offset.checked_add(size).ok_or_else(malformed)?)
            .ok_or_else(malformed)
    }
    pub fn sub(self, offset: usize, size: usize) -> Result<Self> {
        Ok(Self {
            data: self.bytes(offset, size)?,
            base: self.base.checked_add(offset).ok_or_else(malformed)?,
        })
    }
    pub fn remaining(self, offset: usize) -> Result<usize> {
        self.bytes(offset, 0)?;
        Ok(self.data.len() - offset)
    }
    pub fn u8(self, p: usize) -> Result<u8> {
        Ok(self.bytes(p, 1)?[0])
    }
    pub fn u16(self, p: usize) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes(p, 2)?.try_into().unwrap()))
    }
    pub fn i16(self, p: usize) -> Result<i16> {
        Ok(i16::from_le_bytes(self.bytes(p, 2)?.try_into().unwrap()))
    }
    pub fn u32(self, p: usize) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(p, 4)?.try_into().unwrap()))
    }
    pub fn i32(self, p: usize) -> Result<i32> {
        Ok(i32::from_le_bytes(self.bytes(p, 4)?.try_into().unwrap()))
    }
    pub fn f32(self, p: usize) -> Result<f64> {
        Ok(f32::from_le_bytes(self.bytes(p, 4)?.try_into().unwrap()) as f64)
    }
    pub fn rect32(self, p: usize) -> Result<Rect> {
        Ok(Rect {
            left: self.i32(p)? as f64,
            top: self.i32(p + 4)? as f64,
            right: self.i32(p + 8)? as f64,
            bottom: self.i32(p + 12)? as f64,
        })
    }
    pub fn matrix(self, p: usize) -> Result<Matrix> {
        let v = (0..6)
            .map(|i| self.f32(p + i * 4))
            .collect::<Result<Vec<_>>>()?;
        if v.iter().any(|v| !v.is_finite() || v.abs() > 1e12) {
            return Err(malformed());
        }
        Ok(Matrix {
            a: v[0],
            b: v[1],
            c: v[2],
            d: v[3],
            e: v[4],
            f: v[5],
        })
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Record<'a> {
    pub kind: u32,
    pub index: usize,
    pub offset: usize,
    pub data: Reader<'a>,
}
pub struct Records<'a> {
    reader: Reader<'a>,
    offset: usize,
    end: usize,
    index: usize,
    source: Source,
    finished: bool,
}
impl<'a> Records<'a> {
    pub fn new(data: &'a [u8], start: usize, end: usize, source: Source) -> Result<Self> {
        Ok(Self {
            reader: Reader::new(data).sub(0, end)?,
            offset: start,
            end,
            index: 0,
            source,
            finished: false,
        })
    }
}
impl<'a> Iterator for Records<'a> {
    type Item = Result<Record<'a>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let result = (|| {
            if self.offset >= self.end {
                return Err(malformed());
            }
            let emf = self.source == Source::Emf;
            let minimum = if emf { 8 } else { 6 };
            let head = self.reader.sub(self.offset, minimum)?;
            let kind = if emf {
                head.u32(0)?
            } else {
                u32::from(head.u16(4)?)
            };
            let size = if emf {
                head.u32(4)? as usize
            } else {
                (head.u32(0)? as usize)
                    .checked_mul(2)
                    .ok_or_else(malformed)?
            };
            self.index += 1;
            if self.index > 1_000_000 {
                return Err(resource("metafile exceeds Python max_records"));
            }
            if size < minimum || size > self.end - self.offset || emf && size % 4 != 0 {
                return Err(malformed());
            }
            let record = Record {
                kind,
                index: self.index,
                offset: self.offset,
                data: self.reader.sub(self.offset, size)?,
            };
            self.offset += size;
            if kind == if emf { 14 } else { 0 } {
                self.finished = true;
            }
            Ok(record)
        })();
        if result.is_err() {
            self.finished = true;
        }
        Some(result)
    }
}
#[derive(Debug)]
pub struct PlusRecord<'a> {
    pub kind: u16,
    pub flags: u16,
    pub data: Reader<'a>,
    pub offset: usize,
}
pub struct PlusRecords<'a> {
    comment: Reader<'a>,
    offset: usize,
    finished: bool,
}
pub fn plus_records(record: Reader<'_>) -> Result<PlusRecords<'_>> {
    let comment = record.sub(12, record.u32(8)? as usize)?;
    Ok(PlusRecords {
        comment,
        offset: 4,
        finished: comment.data.len() < 4 || comment.u32(0)? != 0x2b464d45,
    })
}
impl<'a> Iterator for PlusRecords<'a> {
    type Item = Result<PlusRecord<'a>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished || self.offset >= self.comment.data.len() {
            return None;
        }
        let result = (|| {
            let head = self.comment.sub(self.offset, 12)?;
            let kind = head.u16(0)?;
            let flags = head.u16(2)?;
            let size = head.u32(4)? as usize;
            let data_size = head.u32(8)? as usize;
            if size < 12 || size % 4 != 0 || data_size > size - 12 || size - 12 - data_size > 3 {
                return Err(malformed());
            }
            let entry = self.comment.sub(self.offset, size)?;
            self.offset += size;
            Ok(PlusRecord {
                kind,
                flags,
                data: entry.sub(12, data_size)?,
                offset: entry.base,
            })
        })();
        if result.is_err() {
            self.finished = true;
        }
        Some(result)
    }
}
