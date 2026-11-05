use super::super::{binary::Reader, malformed, model::*, resource};
use crate::Result;
pub fn finite(v: f64) -> Result<f64> {
    if !v.is_finite() || v.abs() > 1e12 {
        Err(malformed())
    } else {
        Ok(v)
    }
}
pub fn argb(v: u32) -> Color {
    Color {
        red: (v >> 16) as u8,
        green: (v >> 8) as u8,
        blue: v as u8,
        alpha: (v >> 24) as u8,
    }
}
pub struct Cursor<'a> {
    pub r: Reader<'a>,
    pub offset: usize,
}
impl<'a> Cursor<'a> {
    pub fn new(r: Reader<'a>) -> Self {
        Self { r, offset: 0 }
    }
    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let b = self.r.bytes(self.offset, n)?;
        self.offset += n;
        Ok(b)
    }
    pub fn remaining(&self) -> Result<usize> {
        self.r.remaining(self.offset)
    }
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn f32(&mut self) -> Result<f64> {
        finite(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64)
    }
    pub fn count(&mut self) -> Result<usize> {
        let n = self.u32()? as usize;
        if n > 1_000_000 {
            Err(resource("EMF+ count exceeds Python point budget"))
        } else {
            Ok(n)
        }
    }
    pub fn relative(&mut self) -> Result<i32> {
        let a = self.u8()?;
        if a & 128 != 0 {
            let v = ((a as i32 & 127) << 8) | self.u8()? as i32;
            Ok(if v & 16384 != 0 { v - 32768 } else { v })
        } else {
            Ok(if a & 64 != 0 {
                a as i32 - 128
            } else {
                a as i32
            })
        }
    }
    pub fn point(&mut self, compressed: bool) -> Result<Point> {
        if compressed {
            let b = self.take(4)?;
            Ok([
                i16::from_le_bytes(b[..2].try_into().unwrap()) as f64,
                i16::from_le_bytes(b[2..].try_into().unwrap()) as f64,
            ])
        } else {
            Ok([self.f32()?, self.f32()?])
        }
    }
    pub fn points(&mut self, n: usize, flags: u16) -> Result<Vec<Point>> {
        if n > 1_000_000 {
            return Err(resource("EMF+ points exceed Python point budget"));
        }
        if flags & 0x800 == 0 {
            self.r
                .bytes(self.offset, n * if flags & 0x4000 != 0 { 4 } else { 8 })?;
            (0..n).map(|_| self.point(flags & 0x4000 != 0)).collect()
        } else {
            let mut p = [0., 0.];
            (0..n)
                .map(|_| {
                    p = [
                        finite(p[0] + self.relative()? as f64)?,
                        finite(p[1] + self.relative()? as f64)?,
                    ];
                    Ok(p)
                })
                .collect()
        }
    }
    pub fn rect(&mut self, compressed: bool) -> Result<Rect> {
        let [x, y] = self.point(compressed)?;
        let [w, h] = self.point(compressed)?;
        Ok(Rect {
            left: x,
            top: y,
            right: finite(x + w)?,
            bottom: finite(y + h)?,
        })
    }
    pub fn matrix(&mut self) -> Result<Matrix> {
        Ok(Matrix {
            a: self.f32()?,
            b: self.f32()?,
            c: self.f32()?,
            d: self.f32()?,
            e: self.f32()?,
            f: self.f32()?,
        })
    }
    pub fn version(&mut self) -> Result<()> {
        if self.u32()? >> 12 != 0xdbc01 {
            Err(malformed())
        } else {
            Ok(())
        }
    }
    pub fn text(&mut self, n: usize) -> Result<String> {
        if n > 1_000_000 {
            return Err(resource("EMF+ text exceeds Python character budget"));
        }
        let data: Vec<_> = self
            .take(n * 2)?
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        String::from_utf16(&data).map_err(|_| malformed())
    }
    pub fn finish(&self) -> Result<()> {
        if self.remaining()? > 3 {
            Err(malformed())
        } else {
            Ok(())
        }
    }
}
