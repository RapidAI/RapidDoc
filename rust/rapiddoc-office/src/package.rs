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
impl Package {
    pub fn open(bytes: Arc<[u8]>) -> Result<Self> {
        let archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(zip_error)?;
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
        let mut entry = match self.archive.by_name(part) {
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
        entry.read_to_end(&mut bytes).map_err(Error::malformed)?;
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
