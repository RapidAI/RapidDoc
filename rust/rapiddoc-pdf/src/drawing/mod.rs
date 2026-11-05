//! Shared vector PDF debug overlays. No Python, rasterization or font downloads.
use lopdf::{dictionary, Dictionary, Document, Object, ObjectId, Stream};
use rapiddoc_core::{Error, ErrorCode};
use serde_json::Value;
use std::{collections::BTreeSet, fmt::Write};
mod plan;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Layout,
    Span,
    LineSort,
}
impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Layout => "layout",
            Self::Span => "span",
            Self::LineSort => "line_sort",
        }
    }
}

fn invalid(message: impl std::fmt::Display) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: format!("PDF debug drawing: {message}"),
    }
}
fn inherited(doc: &Document, mut page: ObjectId, key: &[u8]) -> Result<Option<Object>, Error> {
    let mut seen = BTreeSet::new();
    loop {
        if !seen.insert(page) {
            return Err(invalid("cyclic page inheritance"));
        }
        let dict = doc.get_dictionary(page).map_err(invalid)?;
        if let Ok(value) = dict.get(key) {
            return Ok(Some(doc.dereference(value).map_err(invalid)?.1.clone()));
        }
        match dict.get(b"Parent").and_then(Object::as_reference) {
            Ok(parent) => page = parent,
            Err(_) => return Ok(None),
        }
    }
}

fn pdf_number(doc: &Document, value: &Object) -> Result<f64, Error> {
    match doc.dereference(value).map_err(invalid)?.1 {
        Object::Integer(n) => Ok(*n as f64),
        Object::Real(n) if n.is_finite() => Ok(*n),
        _ => Err(invalid("invalid numeric page coordinate")),
    }
}

#[derive(Clone, Copy)]
struct Geometry {
    width: f64,
    height: f64,
    rotation: i64,
}
impl Geometry {
    fn load(doc: &Document, page: ObjectId) -> Result<Self, Error> {
        let crop = inherited(doc, page, b"CropBox")?
            .or(inherited(doc, page, b"MediaBox")?)
            .ok_or_else(|| invalid("missing page box"))?;
        let coords = crop.as_array().map_err(invalid)?;
        if coords.len() != 4 {
            return Err(invalid("invalid page box"));
        }
        // Match the original drawing canvas: upper-right coordinates, including
        // its behavior for nonzero crop origins. Do not silently normalize them.
        let width = pdf_number(doc, &coords[2])?;
        let height = pdf_number(doc, &coords[3])?;
        let rotation = match inherited(doc, page, b"Rotate")? {
            Some(Object::Integer(n)) => n.rem_euclid(360),
            Some(Object::Real(n)) if n.is_finite() => (n as i64).rem_euclid(360),
            _ => 0,
        };
        Ok(Self {
            width,
            height,
            rotation,
        })
    }
    fn point(self, [x, y]: [f64; 2]) -> [f64; 2] {
        match self.rotation {
            90 => [y, x],
            180 => [self.width - x, y],
            270 => [self.width - y, self.height - x],
            _ => [x, self.height - y],
        }
    }
    fn rect(self, [x0, y0, x1, y1]: [f64; 4]) -> [f64; 4] {
        let (w, h) = ((x1 - x0).abs(), (y1 - y0).abs());
        match self.rotation {
            90 => [y0, x0, h, w],
            180 => [self.width - x1, y0, w, h],
            270 => [self.width - y1, self.height - x1, h, w],
            _ => [x0, self.height - y1, w, h],
        }
    }
}
fn numbers<const N: usize>(value: &Value) -> Result<[f64; N], Error> {
    let values = value
        .as_array()
        .filter(|a| a.len() == N)
        .ok_or_else(|| invalid("invalid drawing coordinates"))?;
    let mut out = [0.; N];
    for (slot, value) in out.iter_mut().zip(values) {
        *slot = value
            .as_f64()
            .filter(|n| n.is_finite())
            .ok_or_else(|| invalid("non-finite drawing coordinate"))?;
    }
    Ok(out)
}
fn commands(page: &Value, kind: Kind, g: Geometry) -> Result<Vec<u8>, Error> {
    let mut out = String::from("1 w\n");
    for draw in plan::plan(page, kind) {
        let bbox = if draw.item.is_object() {
            draw.item.get("bbox")
        } else {
            Some(draw.item)
        };
        let bbox = bbox
            .filter(|v| !v.is_null())
            .map(numbers::<4>)
            .transpose()?;
        if draw.number.is_some() && bbox.is_none() {
            continue;
        }
        let [r, green, b] = draw.color.map(|n| f64::from(n) / 255.);
        if draw.shape {
            writeln!(
                out,
                "{r} {green} {b} {}",
                if draw.fill { "rg /RD_Fill gs" } else { "RG" }
            )
            .unwrap();
            let poly = draw
                .item
                .get("polygon_points")
                .and_then(Value::as_array)
                .filter(|p| p.len() >= 3);
            if let Some(poly) = poly {
                for (i, p) in poly.iter().enumerate() {
                    let mut point = numbers::<2>(p)?;
                    if draw.number.is_some() {
                        point = point.map(|x| x / (200. / 72.));
                    }
                    let [x, y] = g.point(point);
                    writeln!(out, "{x} {y} {}", if i == 0 { "m" } else { "l" }).unwrap();
                }
                out.push_str("h\n");
            } else if let Some(bbox) = bbox {
                let [x, y, w, h] = g.rect(bbox);
                writeln!(out, "{x} {y} {w} {h} re").unwrap();
            } else {
                continue;
            }
            out.push_str(if draw.fill { "f*\n" } else { "S\n" });
        }
        if let (Some(number), Some(bbox)) = (draw.number, bbox) {
            let [x, y, w, h] = g.rect(bbox);
            let (tx, ty) = match g.rotation {
                0 => (x + w + 2., y + h - 10.),
                90 => (x + 10., y + h + 2.),
                180 => (x - 2., y + 10.),
                270 => (x + w - 10., y - 2.),
                _ => (0., 0.),
            };
            let (sin, cos) = (g.rotation as f64).to_radians().sin_cos();
            let clean = |n: f64| if n.abs() < 1e-12 { 0. } else { n };
            writeln!(out,"{r} {green} {b} rg /RD_Opaque gs\nq {} {} {} {} {tx} {ty} cm\nBT /RD_Font 10 Tf 1 0 0 1 0 0 Tm ({number}) Tj ET\nQ",clean(cos),clean(sin),clean(-sin),clean(cos)).unwrap();
        }
    }
    Ok(out.into_bytes())
}

fn overlay(doc: &mut Document, page: ObjectId, facts: &Value, kind: Kind) -> Result<(), Error> {
    let g = Geometry::load(doc, page)?;
    let content = commands(facts, kind, g)?;
    let form = Stream::new(
        dictionary! {
            "Type"=>"XObject","Subtype"=>"Form","FormType"=>1,
            "BBox"=>vec![Object::from(0),Object::from(0),Object::Real(g.width),Object::Real(g.height)],
            "Resources"=>dictionary! {
                "Font"=>dictionary! {"RD_Font"=>dictionary! {"Type"=>"Font","Subtype"=>"Type1","BaseFont"=>"Helvetica","Encoding"=>"WinAnsiEncoding"}},
                "ExtGState"=>dictionary! {"RD_Fill"=>dictionary! {"Type"=>"ExtGState","ca"=>Object::Real(0.3)},"RD_Opaque"=>dictionary! {"Type"=>"ExtGState","ca"=>1}},
            },
        },
        content,
    );
    let form_id = doc.add_object(form);
    let mut resources = inherited(doc, page, b"Resources")?
        .map(|o| o.as_dict().cloned().map_err(invalid))
        .transpose()?
        .unwrap_or_default();
    let mut xobjects = match resources.get(b"XObject") {
        Ok(obj) => doc
            .dereference(obj)
            .map_err(invalid)?
            .1
            .as_dict()
            .map_err(invalid)?
            .clone(),
        Err(_) => Dictionary::new(),
    };
    let mut index = 0;
    let name = loop {
        let name = format!("RD_Overlay{index}");
        if !xobjects.has(name.as_bytes()) {
            break name;
        }
        index += 1;
    };
    xobjects.set(name.as_bytes(), form_id);
    resources.set("XObject", xobjects);
    // Isolate original content state, as pypdf.merge_page does. Existing font,
    // XObject and graphics-state names cannot collide with the Form resources.
    let before = doc.add_object(Stream::new(Dictionary::new(), b"q\n".to_vec()));
    let mut contents = vec![Object::Reference(before)];
    contents.extend(
        doc.get_page_contents(page)
            .into_iter()
            .map(Object::Reference),
    );
    let after = doc.add_object(Stream::new(
        Dictionary::new(),
        format!("\nQ\nq /{name} Do Q\n").into_bytes(),
    ));
    contents.push(Object::Reference(after));
    let page = doc
        .get_object_mut(page)
        .map_err(invalid)?
        .as_dict_mut()
        .map_err(invalid)?;
    page.set("Resources", resources);
    page.set("Contents", contents);
    Ok(())
}

/// Draw the selected source pages. Page facts are in selection order, matching
/// the production PDF pipeline; no original page contents are rasterized.
pub fn draw(
    data: &[u8],
    pages: &[Value],
    start: usize,
    end: Option<usize>,
    kinds: &[Kind],
) -> Result<Vec<(Kind, Vec<u8>)>, Error> {
    if kinds.is_empty() {
        return Ok(vec![]);
    }
    let mut source = Document::load_mem(data).map_err(invalid)?;
    if source.is_encrypted() {
        source.decrypt("").map_err(invalid)?;
    }
    let ids = source.get_pages();
    let end = end.unwrap_or(ids.len()).min(ids.len());
    if start > end || pages.len() != end - start {
        return Err(invalid("page facts do not match selected PDF range"));
    }
    let remove: Vec<_> = ids
        .keys()
        .copied()
        .filter(|n| (*n as usize) <= start || (*n as usize) > end)
        .collect();
    source.delete_pages(&remove);
    let mut outputs = vec![];
    for &kind in kinds {
        let mut doc = source.clone();
        for ((_, id), facts) in doc.get_pages().into_iter().zip(pages) {
            overlay(&mut doc, id, facts, kind)?;
        }
        doc.prune_objects();
        let mut bytes = vec![];
        doc.save_to(&mut bytes).map_err(invalid)?;
        outputs.push((kind, bytes));
    }
    Ok(outputs)
}
