//! Lazy OPC package access. No archive extraction and no eager media copies.
use crate::{xml, Error, ErrorKind, Format, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Cursor, Read},
    sync::Arc,
};

pub const ROOT_RELS: &str = "_rels/.rels";
pub const CONTENT_TYPES: &str = "[Content_Types].xml";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relationship {
    pub id: String,
    pub kind: String,
    pub target: String,
    pub external: bool,
    pub resolved: Option<String>,
}
impl Relationship {
    pub fn kind_tail(&self) -> &str {
        self.kind.rsplit('/').next().unwrap_or("")
    }
}

#[derive(Debug, Default)]
pub struct ContentTypes {
    pub defaults: BTreeMap<String, String>,
    pub overrides: BTreeMap<String, String>,
}
impl ContentTypes {
    pub fn get(&self, part: &str) -> Option<&str> {
        self.overrides
            .get(part.trim_start_matches('/'))
            .or_else(|| {
                part.rsplit_once('.')
                    .and_then(|(_, extension)| self.defaults.get(extension))
            })
            .map(String::as_str)
    }
}

pub struct Package {
    archive: zip::ZipArchive<Cursor<Arc<[u8]>>>,
    names: Vec<String>,
    patches: BTreeMap<String, Arc<[u8]>>,
    omitted: BTreeSet<String>,
}
fn zip_error(error: zip::result::ZipError) -> Error {
    use zip::result::ZipError;
    let kind = match &error {
        ZipError::InvalidPassword | ZipError::UnsupportedArchive(ZipError::PASSWORD_REQUIRED) => {
            ErrorKind::Encrypted
        }
        ZipError::CompressionMethodNotSupported(_) | ZipError::UnsupportedArchive(_) => {
            ErrorKind::Unsupported
        }
        _ => ErrorKind::Malformed,
    };
    Error::new(kind, error.to_string())
}
const MAX_DIRECTORY_BYTES: u64 = 32 * 1024 * 1024;
const MAX_MEMBERS: u64 = 1_000_000;
const MAX_ENTRY_BYTES: u64 = 256 * 1024 * 1024;
const MAX_CONTROL_XML_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 4 * 1024 * 1024 * 1024;

fn package_limit(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::ResourceLimit, message)
}

fn check_directory(bytes: &[u8]) -> Result<()> {
    let invalid = || Error::malformed("invalid OOXML ZIP central directory");
    let u16_at = |p| u16::from_le_bytes(bytes[p..p + 2].try_into().unwrap()) as u64;
    let u32_at = |p| u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as u64;
    let u64_at = |p| u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
    let end = (bytes.len().saturating_sub(65_557)..bytes.len().saturating_sub(21))
        .rev()
        .find(|&i| {
            bytes[i..].starts_with(b"PK\x05\x06") && i + 22 + u16_at(i + 20) as usize <= bytes.len()
        })
        .ok_or_else(invalid)?;
    if u16_at(end + 4) != 0 || u16_at(end + 6) != 0 {
        return Err(invalid());
    }
    let (count, size, directory_end) = if let Some(locator) = end
        .checked_sub(20)
        .filter(|&i| bytes[i..].starts_with(b"PK\x06\x07"))
    {
        if u32_at(locator + 4) != 0 || u32_at(locator + 16) != 1 {
            return Err(invalid());
        }
        let valid = |i: usize| {
            i <= locator.saturating_sub(56)
                && bytes[i..].starts_with(b"PK\x06\x06")
                && u64_at(i + 4) == (locator - i - 12) as u64
        };
        let record = usize::try_from(u64_at(locator + 8))
            .ok()
            .filter(|&i| locator >= 56 && valid(i))
            .or_else(|| {
                let start = locator.saturating_sub(MAX_DIRECTORY_BYTES as usize + 12);
                let end = locator.checked_sub(56)?;
                (start..=end).rev().find(|&i| valid(i))
            })
            .ok_or_else(invalid)?;
        if u64_at(record + 4) > MAX_DIRECTORY_BYTES {
            return Err(package_limit(
                "OOXML ZIP64 metadata exceeds the directory budget",
            ));
        }
        if u32_at(record + 16) != 0
            || u32_at(record + 20) != 0
            || u64_at(record + 24) != u64_at(record + 32)
        {
            return Err(invalid());
        }
        (u64_at(record + 32), u64_at(record + 40), record)
    } else {
        if u16_at(end + 8) != u16_at(end + 10) {
            return Err(invalid());
        }
        (u16_at(end + 10), u32_at(end + 12), end)
    };
    if count > MAX_MEMBERS || size > MAX_DIRECTORY_BYTES {
        return Err(package_limit(
            "OOXML ZIP central directory exceeds the budget",
        ));
    }
    let mut pos = directory_end
        .checked_sub(size as usize)
        .ok_or_else(invalid)?;
    // 按实际目录记录计数，不能只信 ZIP 结束记录声明的条目数。
    let mut actual = 0;
    while pos < directory_end {
        if directory_end - pos < 46 || !bytes[pos..].starts_with(b"PK\x01\x02") {
            return Err(invalid());
        }
        pos += 46 + (u16_at(pos + 28) + u16_at(pos + 30) + u16_at(pos + 32)) as usize;
        actual += 1;
        if actual > MAX_MEMBERS {
            return Err(package_limit("OOXML ZIP has too many members"));
        }
    }
    if pos != directory_end || actual != count {
        return Err(invalid());
    }
    Ok(())
}

fn member_limit(name: &str) -> u64 {
    if name == CONTENT_TYPES || name.ends_with(".rels") {
        MAX_CONTROL_XML_BYTES
    } else {
        MAX_ENTRY_BYTES
    }
}

fn compressed_limit(expanded: u64) -> u64 {
    expanded + expanded / 100 + 64 * 1024
}

fn validate_members(archive: &mut zip::ZipArchive<Cursor<Arc<[u8]>>>) -> Result<()> {
    use zip::CompressionMethod::{Deflated, Stored};

    let mut declared = 0;
    let mut compressed = 0;
    for index in 0..archive.len() {
        let entry = archive.by_index_data(index).map_err(zip_error)?;
        let name = entry.name().map_err(zip_error)?;
        let cap = member_limit(&name);
        if entry.size() > cap || entry.compressed_size() > compressed_limit(cap) {
            return Err(package_limit(format!(
                "OOXML member exceeds the ZIP budget: {name}"
            )));
        }
        declared += entry.size();
        compressed += entry.compressed_size();
        if declared > MAX_TOTAL_BYTES || compressed > compressed_limit(MAX_TOTAL_BYTES) {
            return Err(package_limit("OOXML package exceeds the ZIP budget"));
        }
        if !matches!(entry.compression(), Stored | Deflated) {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "OOXML requires stored or deflated members",
            ));
        }
        if entry.compression() == Stored
            && !entry.encrypted()
            && entry.size() != entry.compressed_size()
        {
            return Err(Error::malformed(format!(
                "OOXML ZIP member length mismatch: {name}"
            )));
        }
    }

    let mut actual = 0;
    let mut buffer = [0; 64 * 1024];
    for index in 0..archive.len() {
        let Ok(entry) = archive.by_index_raw(index) else {
            continue; // 由各格式现有路径处理损坏的可选成员。
        };
        if entry.encrypted() {
            continue;
        }
        let name = entry.name().map_err(zip_error)?.into_owned();
        let expected = entry.size();
        if entry.compression() == Stored {
            actual += expected;
            if actual > MAX_TOTAL_BYTES {
                return Err(package_limit(
                    "OOXML actual decompression exceeds the package budget",
                ));
            }
            continue;
        }
        // 原始 Deflate 流的计数独立于 ZIP 中央目录声明的展开长度。
        let limit = member_limit(&name).min(MAX_TOTAL_BYTES - actual);
        let mut reader = flate2::read::DeflateDecoder::new(entry);
        let mut count = 0;
        loop {
            let chunk = (limit - count + 1).min(buffer.len() as u64) as usize;
            match reader.read(&mut buffer[..chunk]) {
                Ok(0) => {
                    if count != expected && reader.total_in() == reader.get_ref().compressed_size()
                    {
                        return Err(Error::malformed(format!(
                            "OOXML ZIP member length mismatch: {name}"
                        )));
                    }
                    break;
                }
                Ok(size) => {
                    count += size as u64;
                    if count > limit {
                        return Err(package_limit(format!(
                            "OOXML actual decompression exceeds the budget: {name}"
                        )));
                    }
                }
                Err(_) => break, // 保留可选损坏部件的格式专用修复逻辑。
            }
        }
        actual += count;
    }
    Ok(())
}

impl Package {
    pub fn open(bytes: Arc<[u8]>) -> Result<Self> {
        check_directory(&bytes)?;
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(zip_error)?;
        if archive.has_duplicate_names() {
            return Err(Error::malformed("duplicate OOXML ZIP member names"));
        }
        validate_members(&mut archive)?;
        let names = archive
            .file_names()
            .map(|name| name.map(|v| v.into_owned()).map_err(zip_error))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            archive,
            names,
            patches: BTreeMap::new(),
            omitted: BTreeSet::new(),
        })
    }
    pub fn names(&self) -> &[String] {
        &self.names
    }
    pub fn contains(&self, part: &str) -> bool {
        !self.omitted.contains(part)
            && (self.patches.contains_key(part) || self.archive.index_for_name(part).is_some())
    }
    /// Validate compression/CRC using a bounded streaming buffer. DOCX's
    /// original normalizer checks every member, including unreferenced parts.
    pub(crate) fn validate_member(&mut self, part: &str) -> Result<()> {
        let mut entry = self.archive.by_name(part).map_err(zip_error)?;
        std::io::copy(&mut entry, &mut std::io::sink()).map_err(Error::malformed)?;
        Ok(())
    }
    pub(crate) fn omit(&mut self, part: &str) {
        self.patches.remove(part);
        self.names.retain(|p| p != part);
        self.omitted.insert(part.into());
    }
    pub fn patch(&mut self, part: String, bytes: Arc<[u8]>) {
        let restored = self.omitted.remove(&part);
        if restored || !self.contains(&part) {
            self.names.push(part.clone());
        }
        self.patches.insert(part, bytes);
    }
    pub fn read(&mut self, part: &str) -> Result<Option<Arc<[u8]>>> {
        if self.omitted.contains(part) {
            return Ok(None);
        }
        if let Some(bytes) = self.patches.get(part) {
            return Ok(Some(Arc::clone(bytes)));
        }
        let entry = match self.archive.by_name(part) {
            Ok(entry) => entry,
            Err(zip::result::ZipError::FileNotFound) => return Ok(None),
            Err(error) => return Err(zip_error(error)),
        };
        let size = usize::try_from(entry.size()).map_err(|_| {
            Error::new(
                ErrorKind::ResourceLimit,
                "ZIP entry size cannot be represented",
            )
        })?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| Error::new(ErrorKind::ResourceLimit, "cannot allocate ZIP entry"))?;
        entry
            .take(size as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(Error::malformed)?;
        if bytes.len() != size {
            return Err(Error::malformed(format!(
                "OOXML ZIP member length mismatch: {part}"
            )));
        }
        Ok(Some(bytes.into()))
    }
    pub fn required(&mut self, part: &str) -> Result<Arc<[u8]>> {
        self.read(part)?.ok_or_else(|| Error::missing(part))
    }
    pub fn read_xml(&mut self, part: &str) -> Result<Option<String>> {
        self.read(part)?
            .map(|bytes| xml::decode(&bytes).map(|s| s.into_owned()))
            .transpose()
    }
    pub fn relationships(&mut self, source_part: Option<&str>) -> Result<Vec<Relationship>> {
        let name = source_part
            .map(part_relationships_member)
            .unwrap_or_else(|| Some(ROOT_RELS.to_owned()))
            .ok_or_else(|| Error::missing("relationship source part"))?;
        let Some(text) = self.read_xml(&name)? else {
            return Ok(Vec::new());
        };
        let document = xml::parse(&text)?;
        let mut relationships = Vec::new();
        for node in document
            .root_element()
            .children()
            .filter(|n| n.is_element() && n.tag_name().name() == "Relationship")
        {
            let target = node.attribute("Target").unwrap_or("").to_owned();
            let external = node
                .attribute("TargetMode")
                .is_some_and(|v| v.eq_ignore_ascii_case("external"));
            let resolved = if external {
                None
            } else if let Some(source) = source_part {
                resolve_part_target(source, &target)
            } else {
                resolve_relationship_target(&name, Some(&target))
            };
            relationships.push(Relationship {
                id: node.attribute("Id").unwrap_or("").to_owned(),
                kind: node.attribute("Type").unwrap_or("").to_owned(),
                target,
                external,
                resolved,
            });
        }
        Ok(relationships)
    }
    pub fn main_part(&mut self) -> Result<String> {
        self.relationships(None)?
            .into_iter()
            .find(|rel| !rel.external && rel.kind_tail() == "officeDocument")
            .and_then(|rel| rel.resolved)
            .filter(|part| self.contains(part))
            .ok_or_else(|| Error::missing("officeDocument"))
    }
    pub fn content_types(&mut self) -> Result<ContentTypes> {
        let text = self
            .read_xml(CONTENT_TYPES)?
            .ok_or_else(|| Error::missing(CONTENT_TYPES))?;
        let document = xml::parse(&text)?;
        let mut types = ContentTypes::default();
        for node in document
            .root_element()
            .children()
            .filter(|node| node.is_element())
        {
            let content_type = node.attribute("ContentType").unwrap_or("");
            match node.tag_name().name() {
                "Default" => {
                    if let Some(extension) = node.attribute("Extension") {
                        types
                            .defaults
                            .insert(extension.to_owned(), content_type.to_owned());
                    }
                }
                "Override" => {
                    if let Some(name) = node.attribute("PartName") {
                        let name = name.replace('\\', "/").trim_start_matches('/').to_owned();
                        if !name.is_empty() {
                            types.overrides.insert(name, content_type.to_owned());
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(types)
    }
    pub fn format(&mut self) -> Result<Format> {
        let main = self.main_part()?;
        let text = self.read_xml(&main)?.ok_or_else(|| Error::missing(&main))?;
        let mut reader = quick_xml::NsReader::from_str(&text);
        loop {
            let (namespace, event) = reader.read_resolved_event().map_err(Error::malformed)?;
            use quick_xml::{events::Event, name::ResolveResult};
            match event {
                Event::Start(node) | Event::Empty(node) => {
                    let namespace = match namespace {
                        ResolveResult::Bound(ns) => Some(ns.into_inner()),
                        _ => None,
                    };
                    return match node.local_name().into_inner() {
                        "document" if word_namespace(namespace) => Ok(Format::Docx),
                        "workbook" if namespace_family(namespace, "spreadsheetml/main") => {
                            Ok(Format::Xlsx)
                        }
                        "presentation" if namespace_family(namespace, "presentationml/main") => {
                            Ok(Format::Pptx)
                        }
                        _ => Err(Error::new(
                            ErrorKind::Unsupported,
                            "unsupported officeDocument root",
                        )),
                    };
                }
                Event::Eof => return Err(Error::malformed("missing officeDocument root")),
                _ => {}
            }
        }
    }
}

pub fn namespace_family(namespace: Option<&str>, tail: &str) -> bool {
    let Some((family, name)) = tail.split_once('/') else {
        return false;
    };
    namespace.is_some_and(|ns| {
        ns == format!("http://schemas.openxmlformats.org/{}/2006/{}", family, name)
            || ns == format!("http://purl.oclc.org/ooxml/{tail}")
    })
}
pub fn word_namespace(namespace: Option<&str>) -> bool {
    namespace_family(namespace, "wordprocessingml/main")
}

pub fn part_relationships_member(part: &str) -> Option<String> {
    let normalized = part.replace('\\', "/");
    if normalized.is_empty() || normalized == "." {
        return None;
    }
    let (directory, basename) = split(&normalized);
    if basename.is_empty() {
        return None;
    }
    Some(if directory.is_empty() {
        format!("_rels/{basename}.rels")
    } else {
        format!("{directory}/_rels/{basename}.rels")
    })
}
pub fn relationship_source_base_dir(name: &str) -> Option<String> {
    if name == ROOT_RELS {
        return Some(String::new());
    }
    let (prefix, basename) = name.rsplit_once("/_rels/")?;
    let basename = basename.strip_suffix(".rels")?;
    let source = normalize(&join(prefix, basename));
    Some(dirname(&source).to_owned())
}
pub fn resolve_relationship_target(rels: &str, target: Option<&str>) -> Option<String> {
    let target = target.filter(|v| !v.is_empty())?.replace('\\', "/");
    let resolved = if target.starts_with('/') {
        normalize(target.trim_start_matches('/'))
    } else {
        normalize(&join(
            &relationship_source_base_dir(&rels.replace('\\', "/"))?,
            &target,
        ))
    };
    // Match opc.py, including its distinct treatment of exactly "..".
    if resolved.is_empty() || resolved == "." || resolved.starts_with("../") {
        None
    } else {
        Some(resolved)
    }
}
/// Actual part readers know their source part, including a main part at ZIP root.
/// Keep the separate normalizer helper's historical path behavior unchanged.
pub(crate) fn resolve_part_target(source: &str, target: &str) -> Option<String> {
    if target.is_empty() {
        return None;
    }
    let target = target.replace('\\', "/");
    let source = source.replace('\\', "/");
    let resolved = if target.starts_with('/') {
        normalize(target.trim_start_matches('/'))
    } else {
        normalize(&join(dirname(&source), &target))
    };
    if matches!(resolved.as_str(), "" | ".") || resolved.starts_with("../") {
        None
    } else {
        Some(resolved)
    }
}
/// Embedded OLE targets deliberately have stricter rules than the normalizer.
pub fn resolve_internal_target(source: &str, target: &str) -> Option<String> {
    let target = target.trim_matches(python_space);
    if target.is_empty() || target.contains('\\') {
        return None;
    }
    let resolved = if target.starts_with('/') {
        normalize(target.trim_start_matches('/'))
    } else {
        normalize(&join(dirname(source), target))
    };
    if matches!(resolved.as_str(), "" | "." | "..") || resolved.starts_with("../") {
        None
    } else {
        Some(resolved)
    }
}
fn python_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{001c}'..='\u{001f}')
}
fn split(path: &str) -> (&str, &str) {
    let index = path.rfind('/').map_or(0, |i| i + 1);
    let head = &path[..index];
    let head = if head.chars().any(|c| c != '/') {
        head.trim_end_matches('/')
    } else {
        head
    };
    (head, &path[index..])
}
fn dirname(path: &str) -> &str {
    split(path).0
}
fn join(directory: &str, name: &str) -> String {
    if name.starts_with('/') || directory.is_empty() {
        name.to_owned()
    } else if name.is_empty() {
        format!("{directory}/")
    } else {
        format!("{directory}/{name}")
    }
}
fn normalize(path: &str) -> String {
    let absolute = path.starts_with('/');
    let double = path.starts_with("//") && !path.starts_with("///");
    let mut parts = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." if parts.last().is_some_and(|v| *v != "..") => {
                parts.pop();
            }
            ".." if !absolute => parts.push(part),
            ".." => {}
            _ => parts.push(part),
        }
    }
    let body = parts.join("/");
    if absolute {
        format!("{}{body}", if double { "//" } else { "/" })
    } else if body.is_empty() {
        ".".to_owned()
    } else {
        body
    }
}
