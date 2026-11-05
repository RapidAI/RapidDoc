//! DocVortex 0.6.11: explicit NUL CMap recovery from the font's own glyph names.
use lopdf::{Dictionary, Document, Object, Stream};
use pdfium_render::prelude::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};
fn tokens(data: &[u8]) -> Option<Vec<Vec<u8>>> {
    if data.len() > 1024 * 1024 {
        return None;
    }
    static RE: OnceLock<regex::bytes::Regex> = OnceLock::new();
    let re=RE.get_or_init(||regex::bytes::Regex::new(r"%[^\r\n]*|/[^\s<>{}\[\]()/%]+|\((?:\\[\s\S]|[^\\()])*\)|<<|>>|<[0-9A-Fa-f\s]*>|[^\s<>{}\[\]()/%]+|[^\s]").unwrap());
    Some(
        re.find_iter(data)
            .filter(|m| !m.as_bytes().starts_with(b"%"))
            .map(|m| m.as_bytes().to_vec())
            .collect(),
    )
}
fn hex(s: &[u8]) -> Option<Vec<u8>> {
    if s.first() != Some(&b'<') || s.last() != Some(&b'>') || s.starts_with(b"<<") {
        return None;
    }
    let digits: Vec<_> = s[1..s.len() - 1]
        .iter()
        .copied()
        .filter(|c| !c.is_ascii_whitespace())
        .collect();
    if digits.len() % 2 != 0 {
        return None;
    }
    digits
        .chunks(2)
        .map(|p| Some((p[0] as char).to_digit(16)? as u8 * 16 + (p[1] as char).to_digit(16)? as u8))
        .collect()
}
fn integer(s: &[u8]) -> Option<usize> {
    std::str::from_utf8(s).ok()?.parse().ok()
}
fn cmap(data: &[u8]) -> Option<(BTreeMap<u8, Vec<u8>>, Vec<(u8, u8)>)> {
    let t = tokens(data)?;
    if t.iter().filter(|s| s.as_slice() == b"begincmap").count() != 1
        || t.iter().filter(|s| s.as_slice() == b"endcmap").count() != 1
        || t.iter().any(|s| {
            [
                b"usecmap".as_slice(),
                b"begincidchar",
                b"begincidrange",
                b"(",
                b")",
            ]
            .contains(&s.as_slice())
        })
    {
        return None;
    }
    let mut i = t.iter().position(|s| s == b"begincmap")? + 1;
    let end = t.iter().position(|s| s == b"endcmap")?;
    let mut map = BTreeMap::new();
    let mut spaces = vec![];
    while i < end {
        let op = t.get(i)?.as_slice();
        let close = match op {
            b"begincodespacerange" => b"endcodespacerange".as_slice(),
            b"beginbfchar" => b"endbfchar",
            b"beginbfrange" => b"endbfrange",
            _ => {
                i += 1;
                continue;
            }
        };
        let count = integer(t.get(i.checked_sub(1)?)?)?;
        if count > 256 {
            return None;
        }
        i += 1;
        for _ in 0..count {
            let first = hex(t.get(i)?)?;
            let second = hex(t.get(i + 1)?)?;
            i += 2;
            if first.len() != 1 {
                return None;
            }
            let start = first[0];
            let pairs = if op == b"beginbfchar" {
                vec![(start, second)]
            } else {
                if second.len() != 1 || second[0] < start {
                    return None;
                }
                let stop = second[0];
                if op == b"begincodespacerange" {
                    spaces.push((start, stop));
                    continue;
                }
                let dest = t.get(i)?;
                i += 1;
                let mut pairs = vec![];
                if dest == b"[" {
                    for code in start..=stop {
                        pairs.push((code, hex(t.get(i)?)?));
                        i += 1;
                    }
                    if t.get(i)? != b"]" {
                        return None;
                    }
                    i += 1;
                } else {
                    let base = hex(dest)?;
                    for code in start..=stop {
                        let mut value = base.clone();
                        let mut carry = (code - start) as u16;
                        for b in value.iter_mut().rev() {
                            carry += *b as u16;
                            *b = carry as u8;
                            carry >>= 8;
                        }
                        if carry != 0 {
                            return None;
                        }
                        pairs.push((code, value));
                    }
                }
                pairs
            };
            for (code, value) in pairs {
                if value.is_empty()
                    || value.len() % 2 != 0
                    || value.len() > 512
                    || map.insert(code, value).is_some()
                {
                    return None;
                }
            }
        }
        if t.get(i)?.as_slice() != close {
            return None;
        }
        i += 1;
    }
    if spaces.is_empty()
        || map
            .keys()
            .any(|code| !spaces.iter().any(|(a, b)| a <= code && code <= b))
    {
        return None;
    }
    Some((map, spaces))
}
fn glyph(name: &[u8]) -> Option<char> {
    static GLYPHS: OnceLock<BTreeMap<String, Vec<u32>>> = OnceLock::new();
    let names = GLYPHS.get_or_init(|| {
        serde_json::from_str(include_str!("unicode_fallback/glyphs.json")).unwrap()
    });
    let base = name.split(|c| *c == b'.').next()?;
    let mut values = vec![];
    for component in base.split(|c| *c == b'_') {
        if let Some(mapped) = std::str::from_utf8(component)
            .ok()
            .and_then(|n| names.get(n))
        {
            values.extend(mapped);
            continue;
        }
        let digits = if component.starts_with(b"uni") {
            &component[3..]
        } else if component.starts_with(b"u") {
            &component[1..]
        } else {
            continue;
        };
        if !digits
            .iter()
            .all(|c| c.is_ascii_digit() || (b'A'..=b'F').contains(c))
        {
            continue;
        }
        if component.starts_with(b"uni") && !digits.is_empty() && digits.len() % 4 == 0 {
            let sequence: Option<Vec<_>> = digits
                .chunks(4)
                .map(|s| {
                    u32::from_str_radix(std::str::from_utf8(s).ok()?, 16)
                        .ok()
                        .and_then(char::from_u32)
                        .map(u32::from)
                })
                .collect();
            if let Some(sequence) = sequence {
                values.extend(sequence)
            }
        } else if !component.starts_with(b"uni") && (4..=6).contains(&digits.len()) {
            if let Some(c) = u32::from_str_radix(std::str::from_utf8(digits).ok()?, 16)
                .ok()
                .and_then(char::from_u32)
            {
                values.push(c as u32)
            }
        }
    }
    if values.len() != 1 {
        return None;
    }
    let code = values[0];
    let c = char::from_u32(code)?;
    (c != '\u{fffd}' && allowed_scalar(code)).then_some(c)
}
fn allowed_scalar(code: u32) -> bool {
    static RANGES: OnceLock<Vec<[u32; 2]>> = OnceLock::new();
    let ranges = RANGES.get_or_init(|| {
        serde_json::from_str(include_str!("unicode_fallback/category_c.json")).unwrap()
    });
    !ranges
        .binary_search_by(|[a, b]| {
            if code < *a {
                std::cmp::Ordering::Greater
            } else if code > *b {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}
fn resolved<'a>(d: &'a Document, o: &'a Object) -> Option<&'a Object> {
    Some(d.dereference(o).ok()?.1)
}
fn repair_font(d: &Document, font: &Dictionary) -> Option<(Vec<u8>, BTreeSet<u32>)> {
    if ![b"Type1".as_slice(), b"TrueType"].contains(&font.get(b"Subtype").ok()?.as_name().ok()?) {
        return None;
    }
    let encoding = resolved(d, font.get(b"Encoding").ok()?)?.as_dict().ok()?;
    let differences = encoding.get(b"Differences").ok()?.as_array().ok()?;
    let stream = resolved(d, font.get(b"ToUnicode").ok()?)?
        .as_stream()
        .ok()?;
    let data = stream
        .decompressed_content()
        .unwrap_or_else(|_| stream.content.clone());
    let (mut map, spaces) = cmap(&data)?;
    let mut codes = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut code = -1i64;
    for value in differences {
        if let Object::Integer(n) = value {
            code = *n;
            continue;
        }
        if !(0..=255).contains(&code) || !seen.insert(code) {
            return None;
        }
        let name = value.as_name().ok()?;
        if map.get(&(code as u8)).is_some_and(|s| s == &[0, 0]) {
            if let Some(c) = glyph(name) {
                let mut b = [0; 2];
                map.insert(
                    code as u8,
                    c.encode_utf16(&mut b)
                        .iter()
                        .flat_map(|v| v.to_be_bytes())
                        .collect(),
                );
                codes.insert(c as u32);
            }
        }
        code += 1;
    }
    if codes.is_empty() {
        return None;
    }
    let mut out=format!("/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n/CMapName /DocVortexGlyphFallback def\n/CMapType 2 def\n{} begincodespacerange\n",spaces.len());
    for (a, b) in spaces {
        out += &format!("<{a:02x}><{b:02x}>\n")
    }
    out += "endcodespacerange\n";
    let entries: Vec<_> = map.into_iter().collect();
    for batch in entries.chunks(100) {
        out += &format!("{} beginbfchar\n", batch.len());
        for (code, value) in batch {
            out += &format!(
                "<{code:02x}><{}>\n",
                value.iter().map(|b| format!("{b:02x}")).collect::<String>()
            );
        }
        out += "endbfchar\n";
    }
    out += "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend";
    Some((out.into_bytes(), codes))
}
fn visit(
    d: &mut Document,
    resources: Object,
    seen: &mut BTreeSet<lopdf::ObjectId>,
    depth: usize,
    recovered: &mut BTreeSet<u32>,
) -> Option<()> {
    if depth >= 32 {
        return Some(());
    }
    if let Object::Reference(id) = resources {
        if !seen.insert(id) {
            return Some(());
        }
    }
    let mut dict = resolved(d, &resources)?.as_dict().ok()?.clone();
    if let Ok(fonts) = dict.get(b"Font") {
        if let Some(fonts) = resolved(d, fonts).and_then(|o| o.as_dict().ok()).cloned() {
            let mut updated = fonts;
            for (_, value) in updated.iter_mut() {
                let Some(mut font) = resolved(d, value).and_then(|o| o.as_dict().ok()).cloned()
                else {
                    continue;
                };
                if let Some((data, codes)) = repair_font(d, &font) {
                    let id = d.add_object(Stream::new(Dictionary::new(), data));
                    font.set("ToUnicode", Object::Reference(id));
                    let id = d.add_object(font);
                    *value = Object::Reference(id);
                    recovered.extend(codes);
                }
            }
            dict.set("Font", updated);
        }
    }
    if let Ok(objects) = dict.get(b"XObject") {
        if let Some(objects) = resolved(d, objects).and_then(|o| o.as_dict().ok()).cloned() {
            for (_, value) in objects.iter() {
                let Some(mut form) = resolved(d, value).and_then(|o| o.as_stream().ok()).cloned()
                else {
                    continue;
                };
                if form.dict.get(b"Subtype").and_then(Object::as_name).ok()
                    != Some(b"Form".as_slice())
                {
                    continue;
                }
                if let Ok(child) = form.dict.get(b"Resources") {
                    let id = d.add_object(child.clone());
                    visit(d, Object::Reference(id), seen, depth + 1, recovered);
                    form.dict.set("Resources", Object::Reference(id));
                }
                if let Object::Reference(id) = value {
                    d.objects.insert(*id, Object::Stream(form));
                }
            }
        }
    }
    if let Object::Reference(id) = resources {
        d.objects.insert(id, Object::Dictionary(dict));
    }
    Some(())
}
pub(crate) fn recover(
    api: &dyn PdfiumLibraryBindings,
    text: FPDF_TEXTPAGE,
    bytes: &[u8],
    page_index: usize,
) -> BTreeMap<usize, u32> {
    attempt(api, text, bytes, page_index).unwrap_or_default()
}
fn attempt(
    api: &dyn PdfiumLibraryBindings,
    text: FPDF_TEXTPAGE,
    bytes: &[u8],
    page_index: usize,
) -> Option<BTreeMap<usize, u32>> {
    // SAFETY: caller owns the textpage and holds this PDFium library global lock.
    unsafe {
        let count = api.FPDFText_CountChars(text);
        if count <= 0 {
            return None;
        }
        let original: Vec<_> = (0..count)
            .map(|i| api.FPDFText_GetUnicode(text, i))
            .collect();
        if !original.contains(&0) {
            return None;
        }
        let mut d = (*crate::parsed_document::load(bytes).ok()?).clone();
        let pages = d.get_pages();
        let page = *pages.get(&(page_index as u32 + 1))?;
        // Clone only on anomalous fonts; no change to the source or shared parsed-document cache.
        let (inline, ids) = d.get_page_resources(page).ok()?;
        let resources = inline.cloned().or_else(|| {
            ids.first()
                .and_then(|id| d.get_dictionary(*id).ok())
                .cloned()
        })?;
        let resource_id = d.add_object(resources);
        let mut codes = BTreeSet::new();
        visit(
            &mut d,
            Object::Reference(resource_id),
            &mut BTreeSet::new(),
            0,
            &mut codes,
        )?;
        if codes.is_empty() {
            return None;
        }
        {
            let p = d.get_object_mut(page).ok()?.as_dict_mut().ok()?;
            p.set("Resources", Object::Reference(resource_id));
            p.remove(b"Annots");
            p.remove(b"B");
        }
        let remove: Vec<_> = pages
            .keys()
            .copied()
            .filter(|n| *n != page_index as u32 + 1)
            .collect();
        d.delete_pages(&remove);
        d.prune_objects();
        let mut output = vec![];
        d.save_to(&mut output).ok()?;
        let document = api.FPDF_LoadMemDocument64(&output, None);
        if document.is_null() {
            return None;
        }
        struct Guard<'a> {
            api: &'a dyn PdfiumLibraryBindings,
            doc: FPDF_DOCUMENT,
            page: FPDF_PAGE,
            text: FPDF_TEXTPAGE,
        }
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                unsafe {
                    if !self.text.is_null() {
                        self.api.FPDFText_ClosePage(self.text)
                    }
                    if !self.page.is_null() {
                        self.api.FPDF_ClosePage(self.page)
                    }
                    self.api.FPDF_CloseDocument(self.doc)
                }
            }
        }
        let mut owner = Guard {
            api,
            doc: document,
            page: std::ptr::null_mut(),
            text: std::ptr::null_mut(),
        };
        owner.page = api.FPDF_LoadPage(document, 0);
        if owner.page.is_null() {
            return None;
        }
        owner.text = api.FPDFText_LoadPage(owner.page);
        if owner.text.is_null() || api.FPDFText_CountChars(owner.text) != count {
            return None;
        }
        let candidate: Vec<_> = (0..count)
            .map(|i| api.FPDFText_GetUnicode(owner.text, i))
            .collect();
        if original
            .iter()
            .zip(&candidate)
            .any(|(a, b)| a != b && (*a != 0 || !codes.contains(b)))
        {
            return None;
        }
        let bbox = |handle, index| {
            let mut loose = FS_RECTF {
                left: 0.,
                right: 0.,
                top: 0.,
                bottom: 0.,
            };
            let mut a = 0.;
            let mut b = 0.;
            let mut c = 0.;
            let mut e = 0.;
            if api.FPDFText_GetLooseCharBox(handle, index, &mut loose) == 0
                || api.FPDFText_GetCharBox(handle, index, &mut a, &mut b, &mut c, &mut e) == 0
            {
                return None;
            }
            Some((
                [loose.left, loose.bottom, loose.right, loose.top],
                [a, b, c, e],
                api.FPDFText_GetCharAngle(handle, index),
            ))
        };
        Some(
            original
                .iter()
                .zip(&candidate)
                .enumerate()
                .filter(|(i, (a, b))| {
                    **a == 0
                        && codes.contains(b)
                        && api.FPDFText_IsGenerated(text, *i as i32) == 0
                        && api.FPDFText_IsGenerated(owner.text, *i as i32) == 0
                        && bbox(text, *i as i32).is_some()
                        && bbox(text, *i as i32) == bbox(owner.text, *i as i32)
                })
                .map(|(i, (_, b))| (i, *b))
                .collect(),
        )
    }
}
