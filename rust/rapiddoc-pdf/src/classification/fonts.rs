//! Font resource checks matching the original pypdf classifier, without Python.
use super::rules::{cjk, normalize, space};
use lopdf::{Dictionary, Document, Object, ObjectId};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

fn resolve<'a>(doc: &'a Document, object: &'a Object) -> Option<&'a Object> {
    doc.dereference(object).ok().map(|(_, o)| o)
}
fn dictionary<'a>(doc: &'a Document, object: &'a Object) -> Option<&'a Dictionary> {
    resolve(doc, object)?.as_dict().ok()
}
fn text(o: &Object) -> String {
    match o {
        Object::Name(b) => String::from_utf8_lossy(b).into_owned(),
        Object::String(_, _) => lopdf::decode_text_string(o).unwrap_or_default(),
        Object::Null => "None".into(),
        _ => format!("{o:?}"),
    }
}
fn name(d: &Dictionary, key: &[u8]) -> Option<String> {
    d.get(key).ok().map(text)
}
fn latin(doc: &Document, font: &Dictionary) -> bool {
    if name(font, b"Subtype").as_deref() != Some("Type1") {
        return false;
    }
    let Some(descriptor) = font
        .get(b"FontDescriptor")
        .ok()
        .and_then(|o| dictionary(doc, o))
    else {
        return false;
    };
    if font
        .get(b"ToUnicode")
        .ok()
        .and_then(|o| resolve(doc, o))
        .is_none_or(|o| matches!(o, Object::Null))
    {
        return false;
    }
    let Some(charset) = descriptor.get(b"CharSet").ok().map(text) else {
        return false;
    };
    let glyphs = charset
        .split('/')
        .skip(1)
        .filter_map(|s| s.split(space).next().filter(|s| !s.is_empty()))
        .collect::<BTreeSet<_>>();
    let mut count = 0;
    let mut cjks = 0;
    for g in &glyphs {
        if g.len() == 1 && g.as_bytes()[0].is_ascii_alphabetic() {
            count += 1
        }
        let hex = g.strip_prefix("uni").or_else(|| g.strip_prefix('u'));
        if let Some(hex) =
            hex.filter(|s| (4..=6).contains(&s.len()) && s.bytes().all(|c| c.is_ascii_hexdigit()))
        {
            if u32::from_str_radix(hex, 16).is_ok_and(cjk) {
                cjks += 1
            }
        }
    }
    count >= 10 && (count as f64 / glyphs.len().max(1) as f64) >= 0.5 && cjks == 0
}
fn resources<'a>(doc: &'a Document, mut id: ObjectId) -> Option<&'a Dictionary> {
    let mut seen = BTreeSet::new();
    loop {
        if !seen.insert(id) {
            return None;
        }
        let node = doc.get_dictionary(id).ok()?;
        if let Ok(value) = node.get(b"Resources") {
            return dictionary(doc, value);
        }
        id = node.get(b"Parent").ok()?.as_reference().ok()?;
    }
}
pub fn signals(bytes: &[u8], selected: &[(usize, usize)]) -> Result<Value, lopdf::Error> {
    let doc = Document::load_mem(bytes)?;
    let pages = doc.get_pages();
    let mut cid = Map::new();
    let mut latin_fonts = Map::new();
    for &(local, actual) in selected {
        let Some(&id) = pages.get(&((actual + 1) as u32)) else {
            continue;
        };
        let Some(fonts) = resources(&doc, id)
            .and_then(|r| r.get(b"Font").ok())
            .and_then(|o| dictionary(&doc, o))
        else {
            continue;
        };
        let mut cid_names = BTreeSet::new();
        let mut states = BTreeMap::<String, BTreeMap<usize, bool>>::new();
        for (key, value) in fonts.iter() {
            let Some(font) = dictionary(&doc, value) else {
                continue;
            };
            let font_name = normalize(
                &name(font, b"BaseFont")
                    .filter(|s| !s.is_empty() && s != "None")
                    .unwrap_or_else(|| String::from_utf8_lossy(key).into_owned()),
            );
            if font_name.is_empty() {
                continue;
            }
            let subtype = name(font, b"Subtype");
            let encoding = name(font, b"Encoding");
            if subtype.as_deref() == Some("Type0")
                && matches!(encoding.as_deref(), Some("Identity-H" | "Identity-V"))
                && font.has(b"DescendantFonts")
                && !font.has(b"ToUnicode")
            {
                cid_names.insert(font_name.clone());
            }
            states
                .entry(font_name)
                .or_default()
                .insert(font as *const Dictionary as usize, latin(&doc, font));
        }
        let latin_names = states
            .into_iter()
            .filter_map(|(name, s)| (s.len() == 1 && s.values().all(|&v| v)).then_some(name))
            .collect::<Vec<_>>();
        if !cid_names.is_empty() {
            cid.insert(local.to_string(), json!(cid_names));
        }
        if !latin_names.is_empty() {
            latin_fonts.insert(local.to_string(), json!(latin_names));
        }
    }
    Ok(
        json!({"cid_without_to_unicode":{"triggered":!cid.is_empty(),"page_fonts":cid},"latin_charset_with_to_unicode":{"triggered":!latin_fonts.is_empty(),"page_fonts":latin_fonts}}),
    )
}
