//! Shared legacy OfficeArt records, shape properties and BLIP/BStore decoding.
//! Mirrors the Python decoder; image-local failures do not consume later BSEs.
use crate::{
    xls::records::{u16, u32, Budget},
    Error, ErrorKind, Result,
};
use std::{collections::BTreeMap, sync::Arc};
#[derive(Clone, Copy, Debug)]
pub struct Record<'a> {
    pub offset: usize,
    pub version: u8,
    pub instance: u16,
    pub kind: u16,
    pub payload: &'a [u8],
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct Shape {
    pub shape_id: Option<u32>,
    pub anchor: Option<[u32; 4]>,
    pub pib: Option<u32>,
    pub hidden: bool,
}
#[derive(Debug, Clone)]
pub struct Payload {
    pub data: Arc<[u8]>,
    pub extension: &'static str,
    pub content_type: &'static str,
    pub render_size_emu: Option<[i64; 2]>,
}
#[derive(Debug, Clone)]
pub struct ImageLimit(pub &'static str);
impl std::fmt::Display for ImageLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
const MAX_ENTRY: usize = 128 * 1024 * 1024;
pub fn at<'a>(
    data: &'a [u8],
    offset: usize,
    end: usize,
    budget: &mut Budget,
) -> Result<Option<Record<'a>>> {
    let end = end.min(data.len());
    let Some(header) = offset.checked_add(8).filter(|e| *e <= end) else {
        return Ok(None);
    };
    let length = u32(data, offset + 4).unwrap() as usize;
    let Some(limit) = header.checked_add(length).filter(|e| *e <= end) else {
        return Ok(None);
    };
    budget.charge()?;
    let vi = u16(data, offset).unwrap();
    Ok(Some(Record {
        offset,
        version: (vi & 15) as u8,
        instance: vi >> 4,
        kind: u16(data, offset + 2).unwrap(),
        payload: &data[header..limit],
    }))
}
pub fn visit<'a>(
    data: &'a [u8],
    budget: &mut Budget,
    mut visitor: impl FnMut(Record<'a>, &mut Budget) -> Result<()>,
) -> Result<()> {
    let mut stack = vec![(data, 0usize)];
    while !stack.is_empty() {
        if stack.len() > 64 {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "record nesting exceeds original max_record_depth=64",
            ));
        }
        let (data, cursor) = stack.last_mut().unwrap();
        let Some(r) = at(data, *cursor, data.len(), budget)? else {
            stack.pop();
            continue;
        };
        *cursor += 8 + r.payload.len();
        visitor(r, budget)?;
        if r.version == 15 {
            stack.push((r.payload, 0));
        }
    }
    Ok(())
}
pub fn excel_shapes(data: &[u8], budget: &mut Budget) -> Result<Vec<Shape>> {
    let mut out = vec![];
    visit(data, budget, |r, budget| {
        if r.kind != 0xf004 {
            return Ok(());
        }
        let mut properties = BTreeMap::new();
        let mut id = None;
        let mut anchor = None;
        let mut cursor = 0;
        while let Some(child) = at(r.payload, cursor, r.payload.len(), budget)? {
            cursor += 8 + child.payload.len();
            match child.kind {
                0xf00a if child.payload.len() >= 4 => id = u32(child.payload, 0),
                0xf00b | 0xf122 => {
                    for i in 0..child.instance as usize {
                        let p = i * 6;
                        let (Some(op), Some(value)) =
                            (u16(child.payload, p), u32(child.payload, p + 2))
                        else {
                            break;
                        };
                        properties.insert(op & 0x3fff, value);
                    }
                }
                0xf010 => {
                    anchor = if child.payload.len() >= 18 {
                        Some([
                            u16(child.payload, 6).unwrap() as u32,
                            u16(child.payload, 2).unwrap() as u32,
                            u16(child.payload, 14).unwrap() as u32,
                            u16(child.payload, 10).unwrap() as u32,
                        ])
                    } else {
                        None
                    }
                }
                _ => {}
            }
        }
        let pib = properties.get(&0x0104).copied();
        let flags = properties.get(&0x03bf).copied().unwrap_or(0);
        if anchor.is_some() || pib.is_some() {
            out.push(Shape {
                shape_id: id,
                anchor,
                pib,
                hidden: flags & 0x0002_0000 != 0 && flags & 2 != 0,
            });
        }
        Ok(())
    })?;
    Ok(out)
}
pub fn bmp_header(data: &[u8]) -> Vec<u8> {
    if data.starts_with(b"BM") || data.len() < 4 {
        return data.to_vec();
    }
    let header = u32(data, 0).unwrap() as usize;
    if header < 12 || header > data.len() {
        return data.to_vec();
    }
    let mut palette = 0u64;
    if header >= 40 && data.len() >= 36 {
        let bits = u16(data, 14).unwrap();
        let used = u32(data, 32).unwrap();
        let colors = if used != 0 {
            u64::from(used)
        } else if bits <= 8 {
            1u64 << bits
        } else {
            0
        };
        palette = colors * 4;
    }
    let offset = (14 + header as u64 + palette).min(14 + data.len() as u64) as u32;
    let mut out = Vec::with_capacity(data.len() + 14);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&((14 + data.len()) as u32).to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&offset.to_le_bytes());
    out.extend_from_slice(data);
    out
}
fn inflate(payload: &[u8], zlib: bool) -> std::result::Result<Option<Vec<u8>>, ImageLimit> {
    let mut decoder = flate2::Decompress::new(zlib);
    let mut out = vec![];
    let mut position = 0;
    let mut buffer = [0u8; 65536];
    loop {
        let before_in = decoder.total_in();
        let before_out = decoder.total_out();
        let space = buffer.len().min(MAX_ENTRY + 1 - out.len());
        let status = match decoder.decompress(
            &payload[position..],
            &mut buffer[..space],
            flate2::FlushDecompress::None,
        ) {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        let count = (decoder.total_out() - before_out) as usize;
        position += (decoder.total_in() - before_in) as usize;
        out.extend_from_slice(&buffer[..count]);
        if out.len() > MAX_ENTRY {
            return Err(ImageLimit("metafile BLIP decompression exceeded its limit"));
        }
        if status == flate2::Status::StreamEnd {
            return Ok(Some(out));
        }
        if decoder.total_in() == before_in && count == 0 {
            return Ok(None);
        }
    }
}
pub fn blip(r: Record<'_>) -> std::result::Result<Option<Payload>, ImageLimit> {
    let body = r.payload;
    if matches!(r.kind, 0xf01d | 0xf01e | 0xf01f | 0xf029) {
        let start = if matches!(r.instance, 0x46b | 0x6e3 | 0x6e1 | 0x7a9) {
            33
        } else {
            17
        };
        if start >= body.len() {
            return Ok(None);
        }
        let data = &body[start..];
        if data.len() > MAX_ENTRY {
            return Err(ImageLimit("bitmap BLIP exceeds max_entry_bytes"));
        }
        let (data, extension, content_type): (Arc<[u8]>, &'static str, &'static str) = match r.kind
        {
            0xf01d => (Arc::from(data), "jpg", "image/jpeg"),
            0xf01e => (Arc::from(data), "png", "image/png"),
            0xf029 => (Arc::from(data), "tiff", "image/tiff"),
            _ => (Arc::from(bmp_header(data)), "bmp", "image/bmp"),
        };
        if data.len() > MAX_ENTRY {
            return Err(ImageLimit("bitmap BLIP exceeds max_entry_bytes"));
        }
        return Ok(Some(Payload {
            data,
            extension,
            content_type,
            render_size_emu: None,
        }));
    }
    if !matches!(r.kind, 0xf01a | 0xf01b) {
        return Ok(None);
    }
    let doubled = r.instance == if r.kind == 0xf01a { 0x3d5 } else { 0x217 };
    let h = if doubled { 32 } else { 16 };
    if body.len() < h + 34 {
        return Ok(None);
    }
    let declared = u32(body, h).unwrap() as usize;
    let size = u32(body, h + 28).unwrap() as usize;
    let compression = body[h + 32];
    let payload = &body[h + 34..(h + 34).saturating_add(size).min(body.len())];
    if declared > MAX_ENTRY {
        return Err(ImageLimit("metafile BLIP exceeds max_entry_bytes"));
    }
    let data: Arc<[u8]> = if compression == 0 {
        let mut data = None;
        for wrapped in [false, true] {
            if let Some(bytes) = inflate(payload, wrapped)? {
                data = Some(bytes);
                break;
            }
        }
        let Some(data) = data else { return Ok(None) };
        data.into()
    } else {
        if payload.len() > MAX_ENTRY {
            return Err(ImageLimit("metafile BLIP exceeds max_entry_bytes"));
        }
        payload.into()
    };
    let width = i32::from_le_bytes(body[h + 20..h + 24].try_into().unwrap());
    let height = i32::from_le_bytes(body[h + 24..h + 28].try_into().unwrap());
    Ok(Some(Payload {
        data,
        extension: if r.kind == 0xf01a { "emf" } else { "wmf" },
        content_type: if r.kind == 0xf01a {
            "image/emf"
        } else {
            "image/wmf"
        },
        render_size_emu: if width > 0 && height > 0 {
            Some([width as i64, height as i64])
        } else {
            None
        },
    }))
}
pub fn bstore(
    data: &[u8],
    delay: Option<&[u8]>,
    budget: &mut Budget,
    diagnostics: &mut Vec<String>,
) -> Result<BTreeMap<u32, Payload>> {
    let mut records = vec![];
    visit(data, budget, |r, _| {
        if r.kind == 0xf007 {
            records.push(r)
        }
        Ok(())
    })?;
    let mut out = BTreeMap::new();
    let mut total = 0usize;
    let mut accepted = std::collections::BTreeSet::new();
    for (index, r) in records.iter().take(100_000).enumerate() {
        let body = r.payload;
        if body.len() < 36 {
            continue;
        }
        let start = 36 + body[33] as usize;
        let mut inner = if start < body.len() {
            at(body, start, body.len(), budget)?
        } else {
            None
        };
        if inner.is_none() {
            if let Some(delay) = delay {
                let size = u32(body, 20).unwrap() as usize;
                let count = u32(body, 24).unwrap();
                let offset = u32(body, 28).unwrap();
                if count != 0 && offset != 0xffff_ffff {
                    if let Some(end) = (offset as usize)
                        .checked_add(size)
                        .filter(|e| *e <= delay.len())
                    {
                        inner = at(delay, offset as usize, end, budget)?;
                    }
                }
            }
        }
        let Some(inner) = inner else { continue };
        let decoded = match blip(inner) {
            Ok(Some(p)) => p,
            Ok(None) => continue,
            Err(e) => {
                diagnostics.push(format!(
                    "OFFICE_IMAGE_LIMIT: placeholder for BSE {}: {e}",
                    index + 1
                ));
                out.insert(index as u32 + 1, Payload::placeholder()?);
                continue;
            }
        };
        use sha2::Digest;
        let digest: [u8; 32] = sha2::Sha256::digest(&decoded.data).into();
        if !accepted.contains(&digest)
            && decoded.data.len() > (128 * 1024 * 1024usize).saturating_sub(total)
        {
            diagnostics.push(format!(
                "OFFICE_IMAGE_LIMIT: BSE {} exceeds aggregate image budget; using placeholder",
                index + 1
            ));
            out.insert(index as u32 + 1, Payload::placeholder()?);
            continue;
        }
        if accepted.insert(digest) {
            total += decoded.data.len();
        }
        out.insert(index as u32 + 1, decoded);
    }
    if records.len() > 100_000 {
        diagnostics.push("LEGACY_OFFICE_PICTURE_LIMIT: ignored BSE records after 100000".into());
    }
    Ok(out)
}
impl Payload {
    pub(crate) fn placeholder() -> Result<Self> {
        let source = crate::image::image_limit_placeholder()?;
        let bytes = base64::decode(source.split_once(',').unwrap().1).map_err(Error::malformed)?;
        Ok(Self {
            data: bytes.into(),
            extension: "jpg",
            content_type: "image/jpeg",
            render_size_emu: None,
        })
    }

    pub fn serialize(&self) -> Result<Option<Arc<str>>> {
        self.serialize_with_decoder(&mut crate::equation::image_mtef::Decoder::default())
    }
    pub fn serialize_with_decoder(
        &self,
        decoder: &mut crate::equation::image_mtef::Decoder,
    ) -> Result<Option<Arc<str>>> {
        if crate::image::is_vector(
            &self.data,
            Some(&format!("picture.{}", self.extension)),
            Some(self.content_type),
        ) {
            Ok(Some(
                crate::image::vector_with_decoder(&self.data, self.render_size_emu, decoder)?
                    .source,
            ))
        } else {
            crate::image::serialize_with_decoder(
                &self.data,
                Some(&format!("picture.{}", self.extension)),
                Some(self.content_type),
                decoder,
            )
        }
    }
}
