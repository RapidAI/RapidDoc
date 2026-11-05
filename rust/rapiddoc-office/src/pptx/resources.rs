//! Existing PPTX tables, charts, OLE/image equations and image priority rules.
use super::{
    context::Inheritance,
    geometry::{self, child, Shape, A, P},
    styles, text,
    titles::Entry,
};
use crate::{
    chart,
    docx::body::{Block, Content, Kind},
    equation::{image_mtef, mtef},
    image,
    package::{ContentTypes, Package, Relationship},
    xml, Error, Result,
};
use num_bigint::BigInt;
use roxmltree::Node;
use std::{collections::BTreeMap, sync::Arc};
const R: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
const C: &str = "http://schemas.openxmlformats.org/drawingml/2006/chart";
const M: &str = "http://schemas.openxmlformats.org/officeDocument/2006/math";
const SVG: &str = "http://schemas.microsoft.com/office/drawing/2016/SVG/main";
const URI: &str = "http://schemas.openxmlformats.org/drawingml/2006/";

fn descendants<'a, 'i>(n: Node<'a, 'i>, ns: &str, name: &str) -> Option<Node<'a, 'i>> {
    n.descendants().skip(1).find(|n| n.has_tag_name((ns, name)))
}
fn graphic_data<'a, 'i>(shape: Node<'a, 'i>) -> Option<Node<'a, 'i>> {
    child(child(shape, A, "graphic")?, A, "graphicData")
}
fn has_uri(shape: Node<'_, '_>, tail: &str) -> bool {
    graphic_data(shape)
        .and_then(|n| n.attribute("uri"))
        .is_some_and(|s| {
            if tail == "ole" {
                s == "http://schemas.openxmlformats.org/presentationml/2006/ole"
            } else {
                s == format!("{URI}{tail}")
            }
        })
}
fn embedded_id<'a, 'i>(shape: Node<'a, 'i>, svg: bool) -> Option<&'a str> {
    let lookup = |ns, name| {
        shape
            .descendants()
            .skip(1)
            .filter(|n| n.has_tag_name((ns, name)))
            .find_map(|n| n.attribute((R, "embed")).filter(|s| !s.is_empty()))
    };
    (if svg { lookup(SVG, "svgBlip") } else { None }).or_else(|| lookup(A, "blip"))
}
fn empty_blips(shape: Node<'_, '_>) -> bool {
    let mut seen = false;
    for n in shape
        .descendants()
        .skip(1)
        .filter(|n| n.has_tag_name((A, "blip")) || n.has_tag_name((SVG, "svgBlip")))
    {
        seen = true;
        if n.attribute((R, "embed")).is_some_and(|s| !s.is_empty())
            || n.attribute((R, "link")).is_some_and(|s| !s.is_empty())
        {
            return false;
        }
    }
    seen
}
struct ImageData {
    bytes: Arc<[u8]>,
    mime: Option<String>,
}
pub struct Resources {
    types: ContentTypes,
    media: BTreeMap<String, Arc<[u8]>>,
    image_equations: image_mtef::Decoder,
    ole_equations: mtef::OoxmlDecoder,
    pub diagnostics: Vec<String>,
}
impl Resources {
    pub fn new(package: &mut Package) -> Result<Self> {
        Ok(Self {
            types: package.content_types()?,
            media: BTreeMap::new(),
            image_equations: Default::default(),
            ole_equations: Default::default(),
            diagnostics: vec![],
        })
    }
    fn related(
        &mut self,
        package: &mut Package,
        rels: &[Relationship],
        rid: &str,
    ) -> Option<(Arc<[u8]>, String)> {
        let mut read = || -> Result<_> {
            let target = rels
                .iter()
                .find(|r| r.id == rid && !r.external)
                .and_then(|r| r.resolved.as_ref())
                .ok_or_else(|| Error::missing(rid))?;
            let bytes = if let Some(bytes) = self.media.get(target) {
                Arc::clone(bytes)
            } else {
                package.required(target)?
            };
            Ok((bytes, target.clone()))
        };
        match read() {
            Ok((bytes, target)) => {
                self.media.insert(target.clone(), Arc::clone(&bytes));
                Some((bytes, target))
            }
            Err(e) => {
                self.diagnostics
                    .push(format!("PPTX relationship {rid}: {e}"));
                None
            }
        }
    }
    fn image_data(
        &mut self,
        package: &mut Package,
        shape: Node<'_, '_>,
        rels: &[Relationship],
        svg: bool,
    ) -> Option<ImageData> {
        if let Some(rid) = embedded_id(shape, svg) {
            if let Some((bytes, target)) = self.related(package, rels, rid) {
                return Some(ImageData {
                    bytes,
                    mime: self.types.get(&target).map(str::to_owned),
                });
            }
        }
        if empty_blips(shape) {
            return None;
        }
        // shape.image exists only on Picture, and its public fallback has no
        // content_type hint. Retain that difference from related_part lookup.
        if shape.has_tag_name((P, "pic")) {
            let rid = descendants(shape, A, "blip")?.attribute((R, "embed"))?;
            let (bytes, _) = self.related(package, rels, rid)?;
            return Some(ImageData { bytes, mime: None });
        }
        None
    }
    pub fn image_equation(
        &mut self,
        package: &mut Package,
        shape: Node<'_, '_>,
        rels: &[Relationship],
    ) -> Result<Option<String>> {
        let Some(data) = self.image_data(package, shape, rels, true) else {
            return Ok(None);
        };
        self.image_equations
            .probe(&data.bytes, None, data.mime.as_deref())
    }
    pub fn picture(
        &mut self,
        package: &mut Package,
        shape: Node<'_, '_>,
        rels: &[Relationship],
    ) -> Result<Option<Block>> {
        let Some(data) = self.image_data(package, shape, rels, true) else {
            return Ok(None);
        };
        if let Some(latex) = self
            .image_equations
            .probe(&data.bytes, None, data.mime.as_deref())?
        {
            return Ok(Some(Block::new(Kind::Equation, Content::Text(latex))));
        }
        let mut serialized = image::serialize_with_decoder(
            &data.bytes,
            None,
            data.mime.as_deref(),
            &mut self.image_equations,
        )?;
        if serialized.is_none() && data.mime.as_deref() == Some("image/svg+xml") {
            if let Some(fallback) = self
                .image_data(package, shape, rels, false)
                .filter(|d| d.mime.as_deref() != Some("image/svg+xml"))
            {
                serialized = image::serialize_with_decoder(
                    &fallback.bytes,
                    None,
                    fallback.mime.as_deref(),
                    &mut self.image_equations,
                )?;
            }
        }
        let Some(image) = serialized else {
            return Ok(None);
        };
        let alt = shape
            .descendants()
            .filter(|n| n.is_element() && n.tag_name().name() == "cNvPr")
            .filter_map(|n| n.attribute("descr"))
            .map(str::trim)
            .find(|s| !s.is_empty());
        let mut block = Block::new(
            Kind::Image,
            alt.map(|s| Content::Text(s.into()))
                .unwrap_or(Content::None),
        );
        block.image_base64 = Some(image);
        Ok(Some(block))
    }
    pub fn ole_math(
        &mut self,
        package: &mut Package,
        shape: Node<'_, '_>,
        rels: &[Relationship],
    ) -> Result<Vec<Block>> {
        let Some(ole) = descendants(shape, P, "oleObj") else {
            return Ok(vec![]);
        };
        if !mtef::equation_prog_id(ole.attribute("progId")) {
            return Ok(vec![]);
        }
        let mut equations = vec![];
        for n in shape
            .descendants()
            .skip(1)
            .filter(|n| n.has_tag_name((M, "oMath")))
        {
            if let Some(latex) = text::math(n)? {
                equations.push(Block::new(Kind::Equation, Content::Text(latex)));
            }
        }
        if !equations.is_empty() {
            return Ok(equations);
        }
        if child(ole, P, "embed").is_some()
            && !ole
                .attribute("showAsIcon")
                .is_some_and(|s| matches!(s, "1" | "true"))
        {
            let bytes = ole
                .attribute((R, "id"))
                .and_then(|id| self.related(package, rels, id))
                .map(|(b, _)| b);
            if let Some(latex) = self
                .ole_equations
                .decode(bytes, ole.attribute("progId"), false)?
            {
                return Ok(vec![Block::new(Kind::Equation, Content::Text(latex))]);
            }
            self.diagnostics
                .push("PPTX_MTEF_FALLBACK: invalid or unsupported equation OLE".into());
        }
        Ok(vec![])
    }
    pub fn ole(
        &mut self,
        package: &mut Package,
        shape: Node<'_, '_>,
        rels: &[Relationship],
    ) -> Result<Vec<Block>> {
        if !is_equation_ole(shape) {
            return Ok(vec![]);
        }
        let equations = self.ole_math(package, shape, rels)?;
        if !equations.is_empty() {
            return Ok(equations);
        }
        Ok(self.picture(package, shape, rels)?.into_iter().collect())
    }
    fn chart(
        &mut self,
        package: &mut Package,
        shape: Node<'_, '_>,
        rels: &[Relationship],
    ) -> Option<Block> {
        let mut run = |this: &mut Self| -> Result<Option<Block>> {
            let Some(id) = descendants(shape, C, "chart").and_then(|n| n.attribute((R, "id")))
            else {
                return Ok(None);
            };
            let Some((bytes, target)) = this.related(package, rels, id) else {
                return Ok(None);
            };
            let workbook = (|| -> Result<Option<Arc<[u8]>>> {
                let chart_rels = package.relationships(Some(&target))?;
                let xml = xml::decode(&bytes)?;
                let doc = xml::parse(&xml)?;
                let id = descendants(doc.root_element(), C, "externalData")
                    .and_then(|n| n.attribute((R, "id")));
                Ok(id
                    .and_then(|id| this.related(package, &chart_rels, id))
                    .map(|(b, _)| b))
            })();
            let workbook = match workbook {
                Ok(b) => b,
                Err(e) => {
                    this.diagnostics.push(format!("PPTX chart workbook: {e}"));
                    None
                }
            };
            let html = chart::extract(&bytes, workbook)?;
            Ok((!html.is_empty()).then(|| Block::new(Kind::Chart, Content::Text(html))))
        };
        match run(self) {
            Ok(b) => b,
            Err(e) => {
                self.diagnostics.push(format!("PPTX chart HTML: {e}"));
                None
            }
        }
    }
    /// Match _collect_shape_blocks before geometry sorting: a shape can emit
    /// multiple blocks, but OLE and recognized image equations return early.
    pub fn collect(
        &mut self,
        package: &mut Package,
        shapes: &[Shape<'_, '_>],
        i: usize,
        inheritance: Inheritance<'_, '_>,
        rels: &[Relationship],
        size: [i64; 2],
    ) -> Result<Vec<Entry>> {
        let s = &shapes[i];
        let shape = s.node;
        if shape.has_tag_name((P, "contentPart")) {
            return Err(crate::Error::new(
                crate::ErrorKind::Unsupported,
                "PPTX contentPart requires fallback normalization",
            ));
        }
        if has_uri(shape, "ole") && super::context::placeholder(shape).is_none() {
            return self
                .ole(package, shape, rels)
                .map(|blocks| blocks.into_iter().map(text::entry).collect());
        }
        let mut blocks = vec![];
        if has_uri(shape, "table") {
            if let Some(html) = table(shape, inheritance, rels)? {
                blocks.push(text::entry(Block::new(Kind::Table, Content::Text(html))));
            }
        }
        if has_uri(shape, "chart") {
            if let Some(block) = self.chart(package, shape, rels) {
                blocks.push(text::entry(block));
            }
        }
        if s.picture {
            if let Some(latex) = self.image_equation(package, shape, rels)? {
                blocks.push(text::entry(Block::new(
                    Kind::Equation,
                    Content::Text(latex),
                )));
                return Ok(blocks);
            }
            let skip = geometry::small_picture(s.bbox, size[0] as f64, size[1] as f64)
                || geometry::background_picture(
                    s.bbox,
                    shapes[i + 1..].iter().map(|s| (s.bbox, s.has_text)),
                );
            if !skip {
                if let Some(block) = self.picture(package, shape, rels)? {
                    blocks.push(text::entry(block));
                }
            }
        }
        blocks.extend(text::shape(shape, inheritance, rels)?);
        Ok(blocks)
    }
}
pub fn is_ole(shape: Node<'_, '_>) -> bool {
    has_uri(shape, "ole") && super::context::placeholder(shape).is_none()
}
pub fn is_equation_ole(shape: Node<'_, '_>) -> bool {
    descendants(shape, P, "oleObj").is_some_and(|n| mtef::equation_prog_id(n.attribute("progId")))
}
fn text_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
pub fn table_paragraph(
    p: Node<'_, '_>,
    shape: Node<'_, '_>,
    inheritance: Inheritance<'_, '_>,
    rels: &[Relationship],
) -> Result<String> {
    let sources = styles::font_sources(shape, p, inheritance);
    let mut html = String::new();
    for n in p.children().filter(|n| text::content_node(*n)) {
        if n.has_tag_name((A, "br")) {
            html.push_str("<br>");
            continue;
        }
        if n.has_tag_name(("http://schemas.microsoft.com/office/drawing/2010/main", "m")) {
            if let Some(latex) = text::math(n)? {
                html.push_str(&format!("<eq>{latex}</eq>"));
            }
            continue;
        }
        let raw = child(n, A, "t").and_then(|n| n.text()).unwrap_or("");
        if raw.is_empty() {
            continue;
        }
        let mut value = text_escape(raw).replace("\r\n", "\n").replace('\r', "\n");
        let style = styles::run_style(n, &sources);
        for (s, tag) in [
            ("underline", "u"),
            ("bold", "strong"),
            ("italic", "em"),
            ("strikethrough", "s"),
        ] {
            if style.styles.contains(&s) {
                value = format!("<{tag}>{value}</{tag}>");
            }
        }
        if let Some(url) = text::hyperlink(n, rels)
            .as_deref()
            .and_then(|url| crate::links::sanitize(Some(url)))
        {
            value = format!("<a href=\"{}\">{value}</a>", chart::escape(url));
        }
        html.push_str(&value);
    }
    Ok(html)
}
fn span(node: Node<'_, '_>, attr: &str) -> Result<BigInt> {
    let raw = node
        .attribute(attr)
        .filter(|s| !s.is_empty())
        .unwrap_or("1");
    let value = crate::docx::integer_text(Some(raw))
        .ok_or_else(|| Error::malformed("invalid PPTX cell span"))?;
    value.parse().map_err(Error::malformed)
}
pub fn table(
    shape: Node<'_, '_>,
    inheritance: Inheritance<'_, '_>,
    rels: &[Relationship],
) -> Result<Option<String>> {
    let Some(table) = descendants(shape, A, "tbl") else {
        return Ok(None);
    };
    let mut html = vec!["<table border=\"1\">".into()];
    // The original marks all covered positions, even outside the table. Keep
    // equivalent rectangles instead of allocating absent rows/columns.
    let mut occupied: Vec<(usize, usize, BigInt, BigInt)> = vec![];
    for (r, row) in table
        .children()
        .filter(|n| n.has_tag_name((A, "tr")))
        .enumerate()
    {
        html.push("  <tr>".into());
        for (c, cell) in row
            .children()
            .filter(|n| n.has_tag_name((A, "tc")))
            .enumerate()
        {
            if occupied.iter().any(|(y, x, h, w)| {
                r >= *y && c >= *x && BigInt::from(r - y) < *h && BigInt::from(c - x) < *w
            }) {
                continue;
            }
            let (h, w) = (span(cell, "rowSpan")?, span(cell, "gridSpan")?);
            let mut attrs = String::new();
            if h > BigInt::from(1) {
                attrs.push_str(&format!(" rowspan=\"{h}\""));
            }
            if w > BigInt::from(1) {
                attrs.push_str(&format!(" colspan=\"{w}\""));
            }
            if h > BigInt::from(0) && w > BigInt::from(0) {
                occupied.push((r, c, h, w));
            }
            let paragraphs = child(cell, A, "txBody")
                .into_iter()
                .flat_map(|n| n.children())
                .filter(|n| n.has_tag_name((A, "p")))
                .map(|p| table_paragraph(p, shape, inheritance, rels))
                .collect::<Result<Vec<_>>>()?;
            let text = paragraphs.join("<br>");
            let tag = if r == 0 { "th" } else { "td" };
            html.push(format!("    <{tag}{attrs}>{}</{tag}>", text.trim()));
        }
        html.push("  </tr>".into());
    }
    html.push("</table>".into());
    Ok(Some(html.join("\n")))
}
