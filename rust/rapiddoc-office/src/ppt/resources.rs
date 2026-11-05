//! PPT pictures and embedded Equation/OLE resources. Resource ids are bound
//! once per presentation and reused by every slide.
use super::{
    document::Layout,
    records::{self, u32, Budget, Record},
    shapes,
};
use crate::{
    equation::mtef,
    officeart::{self, Payload},
    ole::OleReader,
    Error, ErrorKind, Result,
};
use flate2::{Decompress, FlushDecompress};
use std::{collections::BTreeMap, sync::Arc};
const OLE_GRAPH: u32 = 4;
const OLE_EQUATION: u32 = 6;
const OLE_CHART: u32 = 14;
const MAX: usize = 128 * 1024 * 1024;
const MAGIC: &[u8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1";
#[derive(Default)]
pub struct Catalog {
    pub images: BTreeMap<u32, Payload>,
    pub equations: BTreeMap<u32, String>,
    pub charts: BTreeMap<u32, String>,
    pub diagnostics: Vec<String>,
    asset_total: usize,
}
fn storage(record: Record<'_>) -> Result<Option<Arc<[u8]>>> {
    if record.kind != 0x1011 {
        return Ok(None);
    }
    let data = if record.instance == 0 {
        record.payload.to_vec()
    } else if record.instance == 1 && record.payload.len() >= 4 {
        let declared = u32(record.payload, 0).unwrap_or(0) as usize;
        if declared == 0 || declared > MAX {
            return Ok(None);
        }
        let mut decoder = Decompress::new(true);
        let mut out = vec![0u8; declared + 1];
        let Ok(status) =
            decoder.decompress(&record.payload[4..], &mut out, FlushDecompress::Finish)
        else {
            return Ok(None);
        };
        // Original Python accepts a complete sync-flushed stream without an
        // end marker, but rejects unused input, truncation and size mismatch.
        let complete =
            status == flate2::Status::StreamEnd || record.payload[4..].ends_with(&[0, 0, 255, 255]);
        if !complete
            || decoder.total_in() as usize != record.payload.len() - 4
            || decoder.total_out() as usize != declared
        {
            return Ok(None);
        }
        out.truncate(declared);
        out
    } else {
        return Ok(None);
    };
    if data.len() > MAX || !data.starts_with(MAGIC) {
        return Ok(None);
    }
    match OleReader::open(data.clone().into()) {
        Ok(_) => Ok(Some(data.into())),
        Err(_) => Ok(None),
    }
}
impl Catalog {
    pub fn load(
        data: &[u8],
        pictures: &[u8],
        layout: &Layout<'_>,
        budget: &mut Budget,
    ) -> Result<Self> {
        let mut out = Catalog::default();
        let mut diagnostics = vec![];
        out.images = officeart::bstore(
            data,
            Some(pictures),
            &mut crate::xls::records::Budget::default(),
            &mut diagnostics,
        )
        .map_err(|e| e)?;
        let mut refs = BTreeMap::<u32, (u32, u32)>::new();
        for r in records::descendants(layout.document, budget) {
            let r = r?;
            if r.kind == 0xfc3 && r.payload.len() >= 24 {
                let object_type = u32(r.payload, 4).unwrap_or(1);
                let object_id = u32(r.payload, 8).unwrap_or(0);
                let subtype = u32(r.payload, 12).unwrap_or(0);
                let persist = u32(r.payload, 16).unwrap_or(0);
                if object_type == 0
                    && object_id != 0
                    && persist != 0
                    && matches!(subtype, OLE_GRAPH | OLE_EQUATION | OLE_CHART)
                {
                    refs.entry(object_id).or_insert((subtype, persist));
                }
            }
        }
        for (id, (subtype, persist)) in refs {
            let Some(offset) = layout.persist.get(&persist) else {
                continue;
            };
            let Some(record) = records::at(data, *offset as usize, None, false, budget)? else {
                continue;
            };
            let Some(storage) = storage(record)? else {
                out.diagnostics
                    .push(format!("PPT_OLE_FALLBACK: exObjId={id}"));
                continue;
            };
            out.asset_total = out.asset_total.saturating_add(storage.len());
            if out.asset_total > MAX {
                return Err(Error::new(
                    ErrorKind::ResourceLimit,
                    "embedded assets exceed original max_asset_total_bytes",
                ));
            }
            match subtype {
                OLE_EQUATION => match mtef::equation_object(storage) {
                    Ok(Some(latex)) => {
                        out.equations.insert(id, latex);
                    }
                    Ok(None) | Err(_) => out
                        .diagnostics
                        .push(format!("PPT_MTEF_FALLBACK: exObjId={id}")),
                },
                OLE_GRAPH | OLE_CHART => {
                    let mut chart_reader = OleReader::open(Arc::clone(&storage));
                    let html = if let Ok(ref mut reader) = chart_reader {
                        let workbook = {
                            let value = reader.read_stream("Workbook", false)?;
                            if value.is_empty() {
                                reader.read_stream("Book", false)?
                            } else {
                                value
                            }
                        };
                        crate::xls::embedded_chart::extract(workbook.into())?.unwrap_or_default()
                    } else {
                        String::new()
                    };
                    if html.is_empty() {
                        out.diagnostics
                            .push(format!("PPT_CHART_RESOURCE_FALLBACK: exObjId={id}"));
                    } else {
                        out.charts.insert(id, html);
                    }
                }
                _ => {}
            }
        }
        out.diagnostics.extend(diagnostics);
        Ok(out)
    }
    pub fn image(
        &self,
        shape: &shapes::Shape<'_>,
        budget: &mut Budget,
        decoder: &mut crate::equation::image_mtef::Decoder,
    ) -> Result<Option<Arc<str>>> {
        let Some(pib) = shapes::shape_properties(shape.record, budget)?
            .get(&0x104)
            .copied()
        else {
            return Ok(None);
        };
        let Some(payload) = self.images.get(&pib) else {
            return Ok(None);
        };
        payload.serialize_with_decoder(decoder)
    }
    pub fn image_equation(
        &self,
        shape: &shapes::Shape<'_>,
        decoder: &mut crate::equation::image_mtef::Decoder,
        budget: &mut Budget,
    ) -> Result<Option<String>> {
        let Some(pib) = shapes::shape_properties(shape.record, budget)?
            .get(&0x104)
            .copied()
        else {
            return Ok(None);
        };
        let Some(payload) = self.images.get(&pib) else {
            return Ok(None);
        };
        decoder.probe(
            &payload.data,
            Some(&format!("picture.{}", payload.extension)),
            Some(payload.content_type),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};
    #[test]
    fn ole_zlib_accepts_finished_and_sync_flushed_streams_but_not_truncation_or_tail() {
        let mut ole = cfb::CompoundFile::create(Cursor::new(vec![])).unwrap();
        ole.create_stream("Workbook")
            .unwrap()
            .write_all(b"test workbook")
            .unwrap();
        let data = ole.into_inner().into_inner();
        for flush in [flate2::FlushCompress::Finish, flate2::FlushCompress::Sync] {
            let mut encoder = flate2::Compress::new(flate2::Compression::default(), true);
            let mut compressed = Vec::with_capacity(data.len() * 2 + 128);
            encoder.compress_vec(&data, &mut compressed, flush).unwrap();
            let mut payload = (data.len() as u32).to_le_bytes().to_vec();
            payload.extend_from_slice(&compressed);
            let read = |bytes: &[u8]| {
                storage(Record {
                    offset: 0,
                    version: 0,
                    instance: 1,
                    kind: 0x1011,
                    payload: bytes,
                })
                .unwrap()
            };
            assert_eq!(&*read(&payload).unwrap(), data);
            let mut tail = payload.clone();
            tail.push(0);
            assert!(read(&tail).is_none());
            let mut wrong_size = payload.clone();
            wrong_size[..4].copy_from_slice(&((data.len() + 1) as u32).to_le_bytes());
            assert!(read(&wrong_size).is_none());
            assert!(read(&payload[..payload.len() - 1]).is_none());
        }
    }
}
