//! Original _safe_paint_resources: reject blending, masks, transparency groups and overprint.
use lopdf::{Document, Object, ObjectId};
use std::collections::BTreeSet;
fn resolve<'a>(d: &'a Document, v: &'a Object) -> Option<&'a Object> {
    d.dereference(v).ok().map(|(_, v)| v)
}
fn truth(v: &Object) -> bool {
    match v {
        Object::Boolean(v) => *v,
        Object::Integer(v) => *v != 0,
        Object::Real(v) => *v != 0.,
        Object::Dictionary(v) => !v.is_empty(),
        Object::Array(v) => !v.is_empty(),
        _ => true,
    }
}
fn check(
    d: &Document,
    value: &Object,
    seen: &mut BTreeSet<ObjectId>,
    depth: usize,
) -> Option<bool> {
    if depth > rapiddoc_core::resources::limit(64) {
        return None;
    }
    if let Object::Reference(id) = value {
        if !seen.insert(*id) {
            return Some(true);
        }
    }
    let node = match resolve(d, value)? {
        Object::Stream(s) => &s.dict,
        Object::Dictionary(d) => d,
        _ => return None,
    };
    if node.get(b"Group").is_ok_and(truth) {
        return Some(false);
    }
    let res = match node.get(b"Resources") {
        Ok(v) => Some(resolve(d, v)?.as_dict().ok()?),
        Err(_) => None,
    };
    if let Some(res) = res {
        if let Ok(states) = res.get(b"ExtGState") {
            let states = resolve(d, states)?.as_dict().ok()?;
            for (_, state) in states.iter() {
                let state = resolve(d, state)?.as_dict().ok()?;
                for (key, default) in [
                    (b"BM".as_slice(), b"Normal".as_slice()),
                    (b"SMask", b"None"),
                ] {
                    if let Ok(v) = state.get(key) {
                        if resolve(d, v)?.as_name().ok() != Some(default) {
                            return Some(false);
                        }
                    }
                }
                for key in [b"ca", b"CA"] {
                    if let Ok(v) = state.get(key) {
                        let n = match resolve(d, v)? {
                            Object::Integer(n) => *n as f64,
                            Object::Real(n) => *n,
                            _ => return Some(false),
                        };
                        if n != 1. {
                            return Some(false);
                        }
                    }
                }
                if state.get(b"OP").is_ok_and(truth) || state.get(b"op").is_ok_and(truth) {
                    return Some(false);
                }
            }
        }
        if let Ok(xs) = res.get(b"XObject") {
            for (_, v) in resolve(d, xs)?.as_dict().ok()?.iter() {
                let obj = resolve(d, v)?;
                let dict = match obj {
                    Object::Stream(s) => &s.dict,
                    Object::Dictionary(d) => d,
                    _ => return None,
                };
                if dict.get(b"Subtype").ok().and_then(|v| v.as_name().ok())
                    == Some(b"Form".as_slice())
                    && !check(d, v, seen, depth + 1)?
                {
                    return Some(false);
                }
            }
        }
    }
    Some(true)
}
pub(super) fn safe(bytes: &[u8], index: usize) -> bool {
    let Ok(d) = crate::parsed_document::load(bytes) else {
        return false;
    };
    if d.is_encrypted() {
        return false;
    }
    let Some(id) = d.get_pages().get(&((index + 1) as u32)).copied() else {
        return false;
    };
    let Ok(page) = d.get_dictionary(id) else {
        return false;
    };
    let mut page = page.clone();
    // PdfReader flattens inherited page Resources before the original safety check.
    if page.get(b"Resources").is_err() {
        let mut parent = page.get(b"Parent").ok().and_then(|v| v.as_reference().ok());
        let mut seen = BTreeSet::new();
        while let Some(id) = parent {
            if !seen.insert(id) || seen.len() > 128 {
                return false;
            }
            let Ok(dict) = d.get_dictionary(id) else {
                return false;
            };
            if let Ok(resources) = dict.get(b"Resources") {
                page.set("Resources", resources.clone());
                break;
            }
            parent = dict.get(b"Parent").ok().and_then(|v| v.as_reference().ok());
        }
    }
    check(&d, &Object::Dictionary(page), &mut BTreeSet::from([id]), 0) == Some(true)
}
