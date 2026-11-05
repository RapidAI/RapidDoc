//! 完整 lopdf 内容解析，沿实际 Do 调用恢复 q/Q/cm 与资源/循环边界。
use super::*;
use lopdf::{content::Content, Dictionary, Document, Object, ObjectId};
fn fail() -> Error {
    error(ErrorCode::CorruptDocument, "PDF Form声明无法验证")
}
fn resolved<'a>(d: &'a Document, o: &'a Object) -> Result<&'a Object, Error> {
    d.dereference(o).map(|(_, o)| o).map_err(|_| fail())
}
fn number(o: &Object) -> Result<f64, Error> {
    match o {
        Object::Integer(n) => Ok(*n as f64),
        Object::Real(n) => Ok(*n as f64),
        _ => Err(fail()),
    }
}
fn values<const N: usize>(d: &Document, o: &Object) -> Result<[f64; N], Error> {
    let a = resolved(d, o)?.as_array().map_err(|_| fail())?;
    if a.len() != N {
        return Err(fail());
    }
    let mut out = [0.; N];
    for (i, o) in a.iter().enumerate() {
        out[i] = number(resolved(d, o)?)?;
        if !out[i].is_finite() {
            return Err(fail());
        }
    }
    Ok(out)
}
fn inherited<'a>(d: &'a Document, page: ObjectId, key: &[u8]) -> Result<&'a Object, Error> {
    let mut id = page;
    let mut seen = BTreeSet::new();
    loop {
        if !seen.insert(id) {
            return Err(fail());
        }
        let dict = d.get_dictionary(id).map_err(|_| fail())?;
        if let Ok(v) = dict.get(key) {
            return resolved(d, v);
        }
        id = dict
            .get(b"Parent")
            .and_then(Object::as_reference)
            .map_err(|_| fail())?;
    }
}
struct Walker<'a> {
    doc: &'a Document,
    out: BTreeMap<Vec<usize>, (B, M)>,
    operations: usize,
    bytes: usize,
    limit: usize,
    cache: HashMap<ObjectId, std::sync::Arc<Content<Vec<lopdf::content::Operation>>>>,
}
impl Walker<'_> {
    fn walk(
        &mut self,
        data: Vec<u8>,
        resources: &Dictionary,
        parent: M,
        occurrence: Vec<usize>,
        active: BTreeSet<ObjectId>,
        depth: usize,
        key: Option<ObjectId>,
    ) -> Result<(), Error> {
        if depth >= 15 {
            return Ok(());
        }
        self.bytes = self.bytes.checked_add(data.len()).ok_or_else(limited)?;
        if self.bytes > self.limit {
            return Err(limited());
        }
        let content = if let Some(c) = key.and_then(|id| self.cache.get(&id)) {
            c.clone()
        } else {
            let c = std::sync::Arc::new(Content::decode(&data).map_err(|_| fail())?);
            if let Some(id) = key {
                self.cache.insert(id, c.clone());
            }
            c
        };
        let mut matrix = parent;
        let mut stack = Vec::new();
        let mut index = 0;
        for op in &content.operations {
            self.operations += 1;
            if self.operations > rapiddoc_core::resources::limit(1_000_000) {
                return Err(limited());
            }
            match op.operator.as_str() {
                "q" => stack.push(matrix),
                "Q" => matrix = stack.pop().ok_or_else(fail)?,
                "cm" => {
                    if op.operands.len() != 6 {
                        return Err(fail());
                    }
                    let mut m = [0.; 6];
                    for (i, o) in op.operands.iter().enumerate() {
                        m[i] = number(resolved(self.doc, o)?)?;
                    }
                    matrix = multiply(m, matrix);
                }
                "Do" => {
                    let name = op
                        .operands
                        .first()
                        .and_then(|o| o.as_name().ok())
                        .ok_or_else(fail)?;
                    let dict = resolved(self.doc, resources.get(b"XObject").map_err(|_| fail())?)?
                        .as_dict()
                        .map_err(|_| fail())?;
                    let original = dict.get(name).map_err(|_| fail())?;
                    let id = original.as_reference().ok();
                    let object = resolved(self.doc, original)?;
                    let stream = object.as_stream().map_err(|_| fail())?;
                    if stream.dict.get(b"Subtype").and_then(Object::as_name).ok()
                        != Some(&b"Form"[..])
                    {
                        continue;
                    }
                    let mut path = occurrence.clone();
                    path.push(index);
                    index += 1;
                    if id.is_some_and(|id| active.contains(&id)) {
                        return Err(fail());
                    }
                    let m = stream
                        .dict
                        .get(b"Matrix")
                        .map_or(Ok(ID), |o| values(self.doc, o))?;
                    let combined = multiply(m, matrix);
                    let b = values(self.doc, stream.dict.get(b"BBox").map_err(|_| fail())?)?;
                    if b.iter().chain(&combined).any(|n| !n.is_finite()) {
                        return Err(fail());
                    }
                    if self.out.len() >= rapiddoc_core::resources::limit(5000) {
                        return Err(limited());
                    }
                    self.out.insert(path.clone(), (b, combined));
                    let resources = stream.dict.get(b"Resources").map_or(Ok(resources), |o| {
                        resolved(self.doc, o)?.as_dict().map_err(|_| fail())
                    })?;
                    let mut active = active.clone();
                    if let Some(id) = id {
                        active.insert(id);
                    }
                    let data = stream
                        .decompressed_content()
                        .unwrap_or_else(|_| stream.content.clone());
                    self.walk(data, resources, combined, path, active, depth + 1, id)?;
                }
                _ => {}
            }
        }
        Ok(())
    }
}
pub(super) fn read(
    bytes: &[u8],
    page: usize,
    password: Option<&str>,
    limits: &Limits,
) -> Result<(BTreeMap<Vec<usize>, (B, M)>, B), Error> {
    let shared = crate::parsed_document::load(bytes).map_err(|_| fail())?;
    let mut decrypted;
    let doc = if shared.is_encrypted() {
        decrypted = Document::load_mem(bytes).map_err(|_| fail())?;
        decrypted
            .decrypt(password.unwrap_or(""))
            .map_err(|_| fail())?;
        &decrypted
    } else {
        shared.as_ref()
    };
    let pages = doc.get_pages();
    let id = *pages.get(&((page + 1) as u32)).ok_or_else(fail)?;
    let resources = inherited(&doc, id, b"Resources")?
        .as_dict()
        .map_err(|_| fail())?;
    let media = values(&doc, inherited(&doc, id, b"MediaBox")?)?;
    let data = doc.get_page_content(id).map_err(|_| fail())?;
    let mut walker = Walker {
        doc: &doc,
        out: BTreeMap::new(),
        operations: 0,
        bytes: 0,
        limit: limits.max_input_bytes,
        cache: HashMap::new(),
    };
    walker.walk(data, resources, ID, Vec::new(), BTreeSet::new(), 0, None)?;
    Ok((walker.out, media))
}
