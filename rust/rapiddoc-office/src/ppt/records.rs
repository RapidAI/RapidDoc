//! MS-PPT record boundaries and original cumulative traversal budgets.
pub use crate::xls::records::{u16, u32};
use crate::{Error, ErrorKind, Result};
pub fn i16(b: &[u8], p: usize) -> Option<i16> {
    u16(b, p).map(|v| v as i16)
}
#[derive(Default, Debug)]
pub struct Budget {
    pub count: usize,
}
impl Budget {
    pub fn charge(&mut self) -> Result<()> {
        self.count += 1;
        if self.count > 16_000_000 {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "record stream exceeds original max_records=16000000",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Record<'a> {
    pub offset: usize,
    pub version: u8,
    pub instance: u16,
    pub kind: u16,
    pub payload: &'a [u8],
}
pub fn at<'a>(
    data: &'a [u8],
    offset: usize,
    end: Option<usize>,
    strict: bool,
    budget: &mut Budget,
) -> Result<Option<Record<'a>>> {
    let limit = end.unwrap_or(data.len()).min(data.len());
    let Some(start) = offset.checked_add(8).filter(|s| *s <= limit) else {
        return if strict {
            Err(Error::malformed("PowerPoint record header is truncated"))
        } else {
            Ok(None)
        };
    };
    let Some(stop) = start
        .checked_add(u32(data, offset + 4).unwrap() as usize)
        .filter(|s| *s <= limit)
    else {
        return if strict {
            Err(Error::malformed(
                "PowerPoint record extends beyond its container",
            ))
        } else {
            Ok(None)
        };
    };
    budget.charge()?;
    let vi = u16(data, offset).unwrap();
    Ok(Some(Record {
        offset,
        version: (vi & 15) as u8,
        instance: vi >> 4,
        kind: u16(data, offset + 2).unwrap(),
        payload: &data[start..stop],
    }))
}
pub struct Records<'a, 'b> {
    data: &'a [u8],
    cursor: usize,
    end: usize,
    strict: bool,
    budget: &'b mut Budget,
    done: bool,
}
pub fn iter<'a, 'b>(
    data: &'a [u8],
    start: usize,
    end: Option<usize>,
    strict_first: bool,
    budget: &'b mut Budget,
) -> Records<'a, 'b> {
    Records {
        data,
        cursor: start,
        end: end.unwrap_or(data.len()).min(data.len()),
        strict: strict_first,
        budget,
        done: false,
    }
}
impl<'a> Iterator for Records<'a, '_> {
    type Item = Result<Record<'a>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.cursor >= self.end {
            return None;
        }
        let r = at(
            self.data,
            self.cursor,
            Some(self.end),
            self.strict,
            self.budget,
        );
        self.strict = false;
        match r {
            Ok(Some(r)) => {
                self.cursor += 8 + r.payload.len();
                Some(Ok(r))
            }
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}
pub struct Descendants<'a, 'b> {
    stack: Vec<(&'a [u8], usize)>,
    budget: &'b mut Budget,
    done: bool,
}
pub fn descendants<'a, 'b>(record: Record<'a>, budget: &'b mut Budget) -> Descendants<'a, 'b> {
    Descendants {
        stack: if record.version == 15 {
            vec![(record.payload, 0)]
        } else {
            vec![]
        },
        budget,
        done: false,
    }
}
impl<'a> Iterator for Descendants<'a, '_> {
    type Item = Result<Record<'a>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        while !self.stack.is_empty() {
            if self.stack.len() > 64 {
                self.done = true;
                return Some(Err(Error::new(
                    ErrorKind::ResourceLimit,
                    "record nesting exceeds original max_record_depth=64",
                )));
            }
            let (data, cursor) = self.stack.last_mut().unwrap();
            match at(data, *cursor, None, false, self.budget) {
                Ok(Some(r)) => {
                    *cursor += 8 + r.payload.len();
                    if r.version == 15 && !(1053..=1064).contains(&r.kind) {
                        self.stack.push((r.payload, 0));
                    }
                    return Some(Ok(r));
                }
                Ok(None) => {
                    self.stack.pop();
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }
        None
    }
}
pub fn children<'a>(r: Record<'a>, budget: &mut Budget) -> Result<Vec<Record<'a>>> {
    if r.version != 15 {
        Ok(vec![])
    } else {
        iter(r.payload, 0, None, false, budget).collect()
    }
}
pub fn utf16(data: &[u8]) -> String {
    String::from_utf16_lossy(
        &data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect::<Vec<_>>(),
    )
    .trim_end_matches('\0')
    .into()
}
