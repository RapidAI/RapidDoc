//! Read-only CFB wrapper matching Python's BoundedOleReader accounting.
use crate::{Error, ErrorKind, Format, Result};
use std::{
    collections::{BTreeMap, HashMap},
    io::{Cursor, Read},
    path::PathBuf,
    sync::{Arc, OnceLock},
};

pub const MAX_ENTRY_BYTES: u64 = 128 * 1024 * 1024;
pub const MAX_TOTAL_BYTES: u64 = 512 * 1024 * 1024;

pub struct OleReader {
    container: cfb::CompoundFile<Cursor<Arc<[u8]>>>,
    names: BTreeMap<String, (String, PathBuf)>,
    total_read: u64,
    max_entry: u64,
    max_total: u64,
    normalized_directory: bool,
}
pub fn casefold(text: &str) -> String {
    static MAP: OnceLock<HashMap<char, String>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        serde_json::from_str(include_str!("unicode_casefold.json"))
            .expect("generated casefold table")
    });
    let mut output = String::with_capacity(text.len());
    for c in text.chars() {
        if let Some(value) = map.get(&c) {
            output.push_str(value);
        } else {
            output.push(c);
        }
    }
    output
}
impl OleReader {
    pub fn open(bytes: Arc<[u8]>) -> Result<Self> {
        Self::open_with_limits(bytes, MAX_ENTRY_BYTES, MAX_TOTAL_BYTES)
    }
    fn open_with_limits(bytes: Arc<[u8]>, max_entry: u64, max_total: u64) -> Result<Self> {
        if bytes.len() as u64 > max_total {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "OLE input exceeds max_total_bytes",
            ));
        }
        let (container, normalized_directory) =
            match cfb::CompoundFile::open(Cursor::new(Arc::clone(&bytes))) {
                Ok(container) => (container, false),
                Err(original) => {
                    let repaired = crate::ole_directory::normalize(&bytes)
                        .ok_or_else(|| Error::malformed(&original))?;
                    (
                        cfb::CompoundFile::open(Cursor::new(repaired))
                            .map_err(|_| Error::malformed(original))?,
                        true,
                    )
                }
            };
        let mut names = BTreeMap::new();
        for entry in container.walk().filter(|entry| entry.is_stream()) {
            let name = entry
                .path()
                .components()
                .filter_map(|c| match c {
                    std::path::Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("/");
            names.insert(casefold(&name), (name, entry.path().to_path_buf()));
        }
        Ok(Self {
            container,
            names,
            total_read: 0,
            max_entry,
            max_total,
            normalized_directory,
        })
    }
    pub fn has_stream(&self, name: &str) -> bool {
        self.names.contains_key(&casefold(name))
    }
    pub fn stream_names(&self, prefix: Option<&str>) -> Vec<&str> {
        let prefix = prefix.map(casefold);
        self.names
            .iter()
            .filter(|(key, _)| prefix.as_ref().is_none_or(|p| key.starts_with(p)))
            .map(|(_, (name, _))| name.as_str())
            .collect()
    }
    pub fn total_read(&self) -> u64 {
        self.total_read
    }
    pub fn normalized_directory_order(&self) -> bool {
        self.normalized_directory
    }
    pub fn read_stream(&mut self, name: &str, required: bool) -> Result<Arc<[u8]>> {
        let Some((_, path)) = self.names.get(&casefold(name)) else {
            return if required {
                Err(Error::missing(name))
            } else {
                Ok(Arc::from([]))
            };
        };
        let size = self.container.entry(path).map_err(Error::malformed)?.len();
        let next = self
            .total_read
            .checked_add(size)
            .ok_or_else(|| Error::new(ErrorKind::ResourceLimit, "OLE read accounting overflow"))?;
        if size > self.max_entry || next > self.max_total {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "OLE stream exceeds original Python read budget",
            ));
        }
        let stream = self.container.open_stream(path).map_err(Error::malformed)?;
        let mut bytes = Vec::new();
        stream
            .take(self.max_entry.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(Error::malformed)?;
        if bytes.len() as u64 > self.max_entry {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "OLE stream exceeds max_entry_bytes",
            ));
        }
        self.total_read += bytes.len() as u64;
        Ok(bytes.into())
    }
    pub fn format(&self) -> Result<Format> {
        if self.has_stream("EncryptedPackage") || self.has_stream("EncryptionInfo") {
            return Err(Error::new(
                ErrorKind::Encrypted,
                "encrypted Office container",
            ));
        }
        if self.has_stream("WordDocument") {
            Ok(Format::Doc)
        } else if self.has_stream("Workbook") || self.has_stream("Book") {
            Ok(Format::Xls)
        } else if self.has_stream("PowerPoint Document") {
            Ok(Format::Ppt)
        } else {
            Err(Error::new(
                ErrorKind::Unsupported,
                "unrecognized Office compound file",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    fn compound() -> Arc<[u8]> {
        let mut compound = cfb::CompoundFile::create(Cursor::new(Vec::new())).unwrap();
        compound
            .create_stream("/Straße")
            .unwrap()
            .write_all(b"abcd")
            .unwrap();
        compound.into_inner().into_inner().into()
    }
    #[test]
    fn original_casefold_and_budget_accounting() {
        let bytes = compound();
        let mut reader = OleReader::open_with_limits(bytes.clone(), 4, bytes.len() as u64).unwrap();
        assert!(reader.has_stream("STRASSE"));
        assert_eq!(&*reader.read_stream("strasse", true).unwrap(), b"abcd");
        assert_eq!(reader.total_read(), 4);
        reader.max_total = 6;
        assert_eq!(
            reader.read_stream("strasse", true).unwrap_err().kind,
            ErrorKind::ResourceLimit
        );
        assert_eq!(reader.total_read(), 4);
        assert!(reader.read_stream("absent", false).unwrap().is_empty());
        assert_eq!(reader.total_read(), 4);
        assert_eq!(
            reader.read_stream("absent", true).unwrap_err().kind,
            ErrorKind::MissingPart
        );
        let mut reader = OleReader::open_with_limits(bytes.clone(), 3, bytes.len() as u64).unwrap();
        assert_eq!(
            reader.read_stream("strasse", true).unwrap_err().kind,
            ErrorKind::ResourceLimit
        );
        assert_eq!(reader.total_read(), 0);
    }
}
