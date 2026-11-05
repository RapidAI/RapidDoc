//! Original XLSX resource arbitration. Drawings remain native OPC parts;
//! equations share the Office decoders and images share the Office codec path.
use super::*;
use crate::{
    equation::{image_mtef, mtef, omml},
    spreadsheet::{table, Position},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
const XDR: &str = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing";
const DRAW: &str = "http://schemas.openxmlformats.org/drawingml/2006/main";
const CHART: &str = "http://schemas.openxmlformats.org/drawingml/2006/chart";
const MATH: &str = "http://schemas.openxmlformats.org/officeDocument/2006/math";
// Original converter's sort sentinel, not a document size limit.
const UNKNOWN: Position = (1_000_000_000, 1_000_000_000);
type Visual = (Position, usize, Value);
#[derive(Default)]
pub(super) struct Context {
    image_equations: image_mtef::Decoder,
    ole_equations: mtef::OoxmlDecoder,
    cell_image_cache: BTreeMap<String, String>,
}
struct Image {
    position: Option<Position>,
    source: Option<Arc<str>>,
    latex: Option<String>,
    order: usize,
}
struct RawImage {
    position: Option<Position>,
    bytes: Arc<[u8]>,
    part: Option<String>,
    order: usize,
}
struct Ole {
    position: Option<Position>,
    preview: Option<Arc<str>>,
    latex: Option<String>,
    shape: Option<String>,
    order: usize,
}
fn named(n: Node<'_, '_>, name: &str) -> bool {
    n.is_element() && n.tag_name().name() == name
}
fn direct<'a, 'd>(n: Node<'a, 'd>, name: &str) -> Option<Node<'a, 'd>> {
    n.children().find(|n| named(*n, name))
}
fn descendant<'a, 'd>(n: Node<'a, 'd>, name: &str) -> Option<Node<'a, 'd>> {
    n.descendants().find(|n| named(*n, name))
}
fn in_ns(n: Node<'_, '_>, ns: &str, name: &str) -> bool {
    named(n, name) && n.tag_name().namespace() == Some(ns)
}
fn rel_attr<'a>(n: Node<'a, '_>, name: &str) -> Option<&'a str> {
    n.attributes()
        .find(|a| {
            a.name() == name
                && a.namespace()
                    .is_some_and(|s| s.trim_end_matches('/').ends_with("/relationships"))
        })
        .map(|a| a.value())
}
fn target<'a>(rels: &'a [Relationship], id: Option<&str>) -> Option<&'a str> {
    rels.iter()
        .rev()
        .find(|r| Some(r.id.as_str()) == id && !r.external)
        .and_then(|r| r.resolved.as_deref())
}
fn number(n: Node<'_, '_>, name: &str) -> Option<u32> {
    descendant(n, name)
        .and_then(|n| n.text())
        .and_then(|s| integer(s).ok())
        .and_then(|i| i.try_into().ok())
}
fn anchor(n: Node<'_, '_>) -> Option<Position> {
    let from = direct(n, "from")?;
    number(from, "row").zip(number(from, "col"))
}
fn is_anchor(n: Node<'_, '_>) -> bool {
    n.is_element()
        && matches!(
            n.tag_name().name(),
            "oneCellAnchor" | "twoCellAnchor" | "absoluteAnchor"
        )
}
fn optional_xml(package: &mut Package, part: &str) -> Result<Option<String>> {
    match package.read_xml(part) {
        Ok(Some(text)) if xml::parse(&text).is_ok() => Ok(Some(text)),
        Err(e) if matches!(e.kind, ErrorKind::ResourceLimit | ErrorKind::Encrypted) => Err(e),
        _ => Ok(None),
    }
}
fn optional_rels(package: &mut Package, part: &str) -> Result<Vec<Relationship>> {
    match package.relationships(Some(part)) {
        Ok(r) => Ok(r),
        Err(e) if e.kind == ErrorKind::ResourceLimit => Err(e),
        _ => Ok(vec![]),
    }
}
fn shape_matches(raw: Option<&str>, shape: &str) -> bool {
    let Some(raw) = raw else { return false };
    raw == shape
        || raw
            .rsplit(['_', 's'])
            .next()
            .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) && s == shape)
}
fn add_cell(sheet: &mut Worksheet, p: Position) -> &mut SourceCell {
    let default = &sheet.cells.default_cell;
    sheet.cells.cells.entry(p).or_insert_with(|| SourceCell {
        row: p.0,
        col: p.1,
        has_value: false,
        semantic: false,
        cell: Cell {
            row_span: 1,
            col_span: 1,
            source_row: Some(p.0),
            source_col: Some(p.1),
            ..default.clone()
        },
    })
}
fn image_html(source: &str) -> String {
    format!("<img src=\"{source}\" />")
}
fn image_block(source: &str) -> Value {
    json!({"type":"image","image_base64":source})
}
fn equation(latex: &str) -> Value {
    json!({"type":"equation","content":latex})
}
impl Context {
    fn image_payload(
        &mut self,
        bytes: &[u8],
        part: Option<&str>,
        position: Option<Position>,
        order: usize,
    ) -> Result<Option<Image>> {
        let latex = self.image_equations.probe(bytes, part, None)?;
        let source =
            crate::image::serialize_with_decoder(bytes, part, None, &mut self.image_equations)?;
        Ok(if latex.is_some() || source.is_some() {
            Some(Image {
                position,
                source,
                latex,
                order,
            })
        } else {
            None
        })
    }
    fn drawings(
        &mut self,
        package: &mut Package,
        sheet: &Worksheet,
    ) -> Result<(
        Vec<RawImage>,
        Vec<(Position, String, usize)>,
        BTreeSet<String>,
        Vec<(Option<Position>, String)>,
    )> {
        let mut images = vec![];
        let mut omml = vec![];
        let mut shapes = BTreeSet::new();
        let mut charts = vec![];
        let mut seen = BTreeSet::new();
        let mut order = 0;
        let mut first = true;
        for rel in &sheet.relationships {
            if rel.external
                || !rel
                    .kind
                    .trim_end_matches('/')
                    .to_ascii_lowercase()
                    .ends_with("/drawing")
            {
                continue;
            }
            let Some(part) = rel.resolved.as_deref() else {
                continue;
            };
            let Some(text) = optional_xml(package, part)? else {
                first = false;
                continue;
            };
            let tree = xml::parse(&text)?;
            let root = tree.root_element();
            let rels = optional_rels(package, part)?;
            for a in root.children().filter(|n| is_anchor(*n)) {
                let p = anchor(a);
                for blip in a.descendants().filter(|n| named(*n, "blip")) {
                    let Some(path) = target(&rels, rel_attr(blip, "embed")) else {
                        continue;
                    };
                    let Some(bytes) = package.read(path)? else {
                        continue;
                    };
                    let current = order;
                    order += 1;
                    let digest: [u8; 32] = Sha256::digest(&bytes).into();
                    if !seen.insert((p, digest)) {
                        continue;
                    }
                    images.push(RawImage {
                        bytes,
                        part: Some(path.into()),
                        position: p,
                        order: current,
                    });
                }
            }
            // openpyxl enumerates charts by anchor class, not XML document order.
            for kind in ["absoluteAnchor", "oneCellAnchor", "twoCellAnchor"] {
                for a in root.children().filter(|n| in_ns(*n, XDR, kind)) {
                    let Some(frame) = a.children().find(|n| in_ns(*n, XDR, "graphicFrame")) else {
                        continue;
                    };
                    if let Some(c) = frame.descendants().find(|n| in_ns(*n, CHART, "chart")) {
                        if let Some(path) = target(&rels, rel_attr(c, "id")) {
                            charts.push((anchor(a), path.into()))
                        }
                    }
                }
            }
            // The original OMML scanner examines only the first drawing and
            // two-cell anchors before one-cell anchors. Absolute anchors excluded.
            if first {
                for kind in ["twoCellAnchor", "oneCellAnchor"] {
                    for a in root.descendants().filter(|n| in_ns(*n, XDR, kind)) {
                        let Some(p) = anchor(a) else { continue };
                        let mut has = false;
                        for math in a.descendants().filter(|n| in_ns(*n, MATH, "oMath")) {
                            if let Ok(latex) = omml::convert(math) {
                                let latex = latex.trim_matches(crate::docx::space);
                                if !latex.is_empty() {
                                    omml.push((p, latex.into(), omml.len()));
                                    has = true
                                }
                            }
                        }
                        if has {
                            shapes.extend(
                                a.descendants()
                                    .filter(|n| in_ns(*n, XDR, "cNvPr"))
                                    .filter_map(|n| n.attribute("id"))
                                    .map(str::to_owned),
                            );
                        }
                    }
                }
            }
            first = false;
        }
        Ok((images, omml, shapes, charts))
    }
    fn shape_info(
        &mut self,
        package: &mut Package,
        sheet: &Worksheet,
        shape: &str,
    ) -> Result<(Option<Position>, Option<String>)> {
        for kind in ["drawing", "vmldrawing"] {
            for rel in &sheet.relationships {
                if rel.external
                    || !rel
                        .kind
                        .trim_end_matches('/')
                        .to_ascii_lowercase()
                        .ends_with(&format!("/{kind}"))
                {
                    continue;
                }
                let Some(part) = rel.resolved.as_deref() else {
                    continue;
                };
                let Some(text) = optional_xml(package, part)? else {
                    continue;
                };
                let tree = xml::parse(&text)?;
                let rels = optional_rels(package, part)?;
                if kind == "drawing" {
                    for a in tree.root_element().children().filter(|n| is_anchor(*n)) {
                        if !a
                            .descendants()
                            .any(|n| named(n, "cNvPr") && shape_matches(n.attribute("id"), shape))
                        {
                            continue;
                        }
                        let p = anchor(a);
                        let preview = a
                            .descendants()
                            .filter(|n| named(*n, "blip"))
                            .find_map(|n| target(&rels, rel_attr(n, "embed")))
                            .map(str::to_owned);
                        if p.is_some() || preview.is_some() {
                            return Ok((p, preview));
                        }
                        break;
                    }
                } else {
                    for a in tree
                        .descendants()
                        .filter(|n| named(*n, "shape") && shape_matches(n.attribute("id"), shape))
                    {
                        let p = descendant(a, "ClientData").and_then(|d| {
                            let values = direct(d, "Anchor").and_then(|n| n.text()).and_then(|s| {
                                s.split(',').map(integer).collect::<Result<Vec<_>>>().ok()
                            });
                            values
                                .and_then(|v| {
                                    if v.len() >= 4 {
                                        u32::try_from(v[2]).ok().zip(u32::try_from(v[0]).ok())
                                    } else {
                                        None
                                    }
                                })
                                .or_else(|| number(d, "Row").zip(number(d, "Column")))
                        });
                        let preview = a
                            .descendants()
                            .filter(|n| named(*n, "imagedata"))
                            .find_map(|n| target(&rels, rel_attr(n, "id")))
                            .map(str::to_owned);
                        if p.is_some() || preview.is_some() {
                            return Ok((p, preview));
                        }
                        break;
                    }
                }
            }
        }
        Ok((None, None))
    }
    fn ole(&mut self, package: &mut Package, sheet: &Worksheet) -> Result<Vec<Ole>> {
        let Some(text) = optional_xml(package, &sheet.part)? else {
            return Ok(vec![]);
        };
        let tree = xml::parse(&text)?;
        let mut output = vec![];
        for (order, obj) in tree
            .descendants()
            .filter(|n| named(*n, "oleObject"))
            .enumerate()
        {
            let prog = obj.attribute("progId").or_else(|| obj.attribute("ProgID"));
            if !mtef::equation_prog_id(prog) {
                continue;
            }
            let rel = sheet
                .relationships
                .iter()
                .rev()
                .find(|r| Some(r.id.as_str()) == rel_attr(obj, "id"));
            let linked = obj.attribute("link").is_some_and(|s| !s.is_empty())
                || rel.is_some_and(|r| r.external);
            let icon = obj
                .attribute("dvAspect")
                .unwrap_or("")
                .to_lowercase()
                .contains("icon");
            let mut latex = None;
            if !linked {
                if let Some(path) = rel
                    .filter(|r| {
                        r.kind
                            .trim_end_matches('/')
                            .to_ascii_lowercase()
                            .ends_with("/oleobject")
                    })
                    .and_then(|r| r.resolved.as_deref())
                {
                    latex = self.ole_equations.decode(package.read(path)?, prog, icon)?;
                }
            }
            let properties = direct(obj, "objectPr");
            let p = properties
                .and_then(|p| direct(p, "anchor"))
                .and_then(anchor);
            let preview = properties
                .and_then(|n| target(&sheet.relationships, rel_attr(n, "id")))
                .map(str::to_owned);
            let (shape_p, shape_preview) =
                self.shape_info(package, sheet, obj.attribute("shapeId").unwrap_or(""))?;
            let preview = preview.or(shape_preview);
            let mut source = None;
            if let Some(part) = preview.as_deref() {
                if let Some(bytes) = package.read(part)? {
                    if latex.is_none() && !icon {
                        latex = self.image_equations.probe(&bytes, Some(part), None)?
                    }
                    source = crate::image::serialize_with_decoder(
                        &bytes,
                        Some(part),
                        None,
                        &mut self.image_equations,
                    )?;
                }
            }
            if latex.is_some() || source.is_some() {
                output.push(Ole {
                    position: p.or(shape_p),
                    preview: source,
                    latex,
                    shape: obj.attribute("shapeId").map(str::to_owned),
                    order,
                })
            }
        }
        Ok(output)
    }
    fn cell_image_paths(&mut self, package: &mut Package) -> Result<BTreeMap<String, String>> {
        let Some(text) = optional_xml(package, "xl/cellimages.xml")? else {
            return Ok(BTreeMap::new());
        };
        let tree = xml::parse(&text)?;
        let rels = optional_rels(package, "xl/cellimages.xml")?;
        let mut names = BTreeMap::new();
        for n in tree.descendants().filter(|n| {
            in_ns(
                *n,
                "http://www.wps.cn/officeDocument/2017/etCustomData",
                "cellImage",
            )
        }) {
            let name = n
                .descendants()
                .find(|n| in_ns(*n, XDR, "cNvPr"))
                .and_then(|n| n.attribute("name"));
            let id = n
                .descendants()
                .find(|n| in_ns(*n, DRAW, "blip"))
                .and_then(|n| rel_attr(n, "embed"));
            if let Some((name, id)) = name.zip(id).filter(|(n, i)| !n.is_empty() && !i.is_empty()) {
                names.insert(id.to_owned(), name.to_owned());
            }
        }
        let mut out = BTreeMap::new();
        for r in rels {
            if let Some(name) = names.get(&r.id) {
                // DISPIMG's original code joins Target with xl, including
                // absolute targets. It does not apply OPC relationship rebasing.
                let path = if r.target.starts_with('/') {
                    r.target.clone()
                } else {
                    normalize_path(&format!("xl/{}", r.target))
                };
                out.insert(name.clone(), path);
            }
        }
        Ok(out)
    }
    fn disp_images(&mut self, package: &mut Package, sheet: &mut Worksheet) -> Result<()> {
        if sheet.cell_images.is_empty() {
            return Ok(());
        }
        let paths = self.cell_image_paths(package)?;
        for (p, raw) in std::mem::take(&mut sheet.cell_images) {
            let Some(id) = raw
                .split_once('"')
                .and_then(|(_, s)| s.split_once('"'))
                .map(|(id, _)| id)
                .filter(|s| !s.is_empty())
            else {
                continue;
            };
            let Some(path) = paths.get(id) else { continue };
            let result = if let Some(s) = self.cell_image_cache.get(path) {
                Some(s.clone())
            } else {
                // This original path catches all image/formula failures and
                // treats the cell image as absent (unlike drawing resources).
                self.resolve_cell_image(package, path).ok().flatten()
            };
            if let Some(s) = result {
                let cell = add_cell(sheet, p);
                cell.cell.media.push(s);
                cell.semantic = true;
            }
        }
        Ok(())
    }
    fn resolve_cell_image(&mut self, package: &mut Package, path: &str) -> Result<Option<String>> {
        let Some(bytes) = package.read(path)? else {
            return Ok(None);
        };
        // Unlike drawing images, a recovered cell equation does not serialize
        // its preview. Keep this short circuit even for damaged vector previews.
        let (result, cacheable) = if let Some(latex) = self
            .image_equations
            .probe(&bytes, Some(path), None)?
            .filter(|s| !s.is_empty())
        {
            (Some(format!("<eq>{latex}</eq>")), true)
        } else {
            (
                crate::image::serialize_with_decoder(
                    &bytes,
                    Some(path),
                    None,
                    &mut self.image_equations,
                )?
                .as_deref()
                .map(image_html),
                cacheable_bitmap(path, &bytes),
            )
        };
        if cacheable {
            if let Some(s) = &result {
                self.cell_image_cache.insert(path.into(), s.clone());
            }
        }
        Ok(result)
    }
    pub fn convert(
        &mut self,
        package: &mut Package,
        sheet: &mut Worksheet,
        gap: Option<u32>,
        singleton: bool,
    ) -> Result<Vec<Value>> {
        let (raw_images, math, shapes, charts) = self.drawings(package, sheet)?;
        let mut ole = self.ole(package, sheet)?;
        // Keep the original decoder order: OMML, OLE (including image previews),
        // then ordinary drawing images, then cell images during projection.
        let mut images = Vec::new();
        for raw in raw_images {
            if let Some(image) =
                self.image_payload(&raw.bytes, raw.part.as_deref(), raw.position, raw.order)?
            {
                images.push(image);
            }
        }
        let previews = ole
            .iter()
            .filter_map(|o| Some((o.position?, o.preview.clone()?)))
            .collect::<BTreeSet<_>>();
        ole.retain(|o| !o.shape.as_ref().is_some_and(|s| shapes.contains(s)));
        images.retain(|i| {
            !i.position
                .zip(i.source.clone())
                .is_some_and(|p| previews.contains(&p))
        });
        self.disp_images(package, sheet)?;
        for (p, latex, _) in &math {
            let c = add_cell(sheet, *p);
            c.cell.equations.push(latex.clone());
            c.semantic = true;
        }
        for o in &ole {
            if let Some(p) = o.position {
                let c = add_cell(sheet, p);
                if let Some(latex) = &o.latex {
                    c.cell.equations.push(latex.clone());
                    c.semantic = true;
                }
            }
        }
        for image in &images {
            if let Some(p) = image.position {
                let media = image
                    .latex
                    .as_ref()
                    .filter(|s| !s.is_empty())
                    .map(|s| format!("<eq>{s}</eq>"))
                    .or_else(|| image.source.as_deref().map(image_html));
                if let Some(media) = media {
                    let c = add_cell(sheet, p);
                    c.cell.media.push(media);
                    c.semantic = true;
                }
            }
        }
        for o in &ole {
            if o.latex.is_none() {
                if let Some((p, source)) = o.position.zip(o.preview.as_deref()) {
                    let c = add_cell(sheet, p);
                    c.cell.media.push(image_html(source));
                    c.semantic = true;
                }
            }
        }
        let (_, _, tables) = sheet.cells.project(gap);
        let mut used = BTreeSet::new();
        let mut visuals: Vec<Visual> = vec![];
        for (i, t) in tables.iter().enumerate() {
            used.extend(t.data.iter().filter_map(|c| c.source_row.zip(c.source_col)));
            visuals.push(((t.anchor.1, t.anchor.0), i, table::block(t, singleton)));
        }
        for (i, (p, path)) in charts.iter().enumerate() {
            if let Some((rows, cols)) = chart_axes(package, path, &sheet.name)? {
                let t = sheet.cells.synthetic(&rows, &cols)?;
                visuals.push((
                    p.unwrap_or(UNKNOWN),
                    10000 + i,
                    json!({"type":"chart","content":table::html(&t)}),
                ));
            }
        }
        for (p, latex, order) in math {
            if !used.contains(&p) {
                visuals.push((p, 15000 + order, equation(&latex)));
            }
        }
        for o in ole {
            if o.position.is_some_and(|p| used.contains(&p)) {
                continue;
            }
            let block = o
                .latex
                .as_deref()
                .map(equation)
                .or_else(|| o.preview.as_deref().map(image_block));
            if let Some(b) = block {
                visuals.push((o.position.unwrap_or(UNKNOWN), 20000 + o.order, b));
            }
        }
        for i in &images {
            if i.position.is_some_and(|p| used.contains(&p)) {
                continue;
            }
            if let Some(latex) = i.latex.as_deref().filter(|s| !s.is_empty()) {
                visuals.push((
                    i.position.unwrap_or(UNKNOWN),
                    25000 + i.order,
                    equation(latex),
                ));
            }
        }
        visuals.sort_by_key(|(p, i, _)| (*p, *i));
        let mut page = visuals.into_iter().map(|(_, _, b)| b).collect::<Vec<_>>();
        for image in images {
            if image.position.is_some_and(|p| used.contains(&p))
                || image.latex.as_ref().is_some_and(|s| !s.is_empty())
            {
                continue;
            }
            if let Some(source) = image.source {
                page.push(image_block(&source));
            }
        }
        Ok(page)
    }
}
fn normalize_path(path: &str) -> String {
    let mut parts = vec![];
    for p in path.split('/') {
        match p {
            "" | "." => {}
            ".." => {
                if parts.last().is_some_and(|s| *s != "..") {
                    parts.pop();
                } else {
                    parts.push(p)
                }
            }
            _ => parts.push(p),
        }
    }
    parts.join("/")
}
fn cacheable_bitmap(path: &str, bytes: &[u8]) -> bool {
    path.rsplit_once('.').is_some_and(|(_, e)| {
        matches!(
            e.to_ascii_lowercase().as_str(),
            "png" | "jpg" | "jpeg" | "gif" | "bmp" | "tif" | "tiff" | "webp"
        )
    }) && ([
        b"\x89PNG\r\n\x1a\n".as_slice(),
        b"\xff\xd8\xff",
        b"GIF87a",
        b"GIF89a",
        b"BM",
        b"II*\0",
        b"MM\0*",
    ]
    .iter()
    .any(|p| bytes.starts_with(p))
        || bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP"))
}
fn chart_axes(
    package: &mut Package,
    part: &str,
    sheet: &str,
) -> Result<Option<(Vec<u32>, Vec<u32>)>> {
    let Some(text) = package.read_xml(part)? else {
        return Ok(None);
    };
    let tree = xml::parse(&text)?;
    let Some(area) = tree.descendants().find(|n| in_ns(*n, CHART, "plotArea")) else {
        return Ok(None);
    };
    let kinds = [
        "areaChart",
        "area3DChart",
        "lineChart",
        "line3DChart",
        "stockChart",
        "radarChart",
        "scatterChart",
        "pieChart",
        "pie3DChart",
        "doughnutChart",
        "barChart",
        "bar3DChart",
        "ofPieChart",
        "surfaceChart",
        "surface3DChart",
        "bubbleChart",
    ];
    let Some(plot) = area.children().find(|n| {
        n.is_element()
            && n.tag_name().namespace() == Some(CHART)
            && kinds.contains(&n.tag_name().name())
    }) else {
        return Ok(None);
    };
    let mut formulas = vec![];
    for ser in plot.children().filter(|n| in_ns(*n, CHART, "ser")) {
        for name in ["cat", "val", "xVal", "yVal", "bubbleSize"] {
            if let Some(source) = direct(ser, name) {
                for kind in ["numRef", "strRef", "multiLvlStrRef"] {
                    if let Some(f) = direct(source, kind)
                        .and_then(|r| direct(r, "f"))
                        .and_then(|n| n.text())
                        .filter(|s| !s.is_empty())
                    {
                        formulas.push(f);
                        break;
                    }
                }
            }
        }
        if let Some(f) = direct(ser, "tx")
            .and_then(|n| direct(n, "strRef"))
            .and_then(|n| direct(n, "f"))
            .and_then(|n| n.text())
            .filter(|s| !s.is_empty())
        {
            formulas.push(f);
        }
    }
    let mut rows = BTreeSet::new();
    let mut cols = BTreeSet::new();
    for formula in formulas {
        let Some((r, c, rr, cc)) = chart_reference(formula, sheet) else {
            return Ok(None);
        };
        rows.extend(r..=rr);
        cols.extend(c..=cc);
    }
    Ok(if rows.is_empty() || cols.is_empty() {
        None
    } else {
        Some((rows.into_iter().collect(), cols.into_iter().collect()))
    })
}
fn chart_reference(formula: &str, sheet: &str) -> Option<(u32, u32, u32, u32)> {
    // openpyxl.range_to_tuple keeps doubled apostrophes and rejects '=' prefixes.
    static RANGE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RANGE.get_or_init(|| regex::Regex::new(r"^(?:'((?:[^']|'')*)'|([^'^ !]*))!((?:\$?[A-Za-z]{1,3})?(?:\$?[0-9]+)?(?::(?:\$?[A-Za-z]{1,3})?(?:\$?[0-9]+)?)?)").unwrap());
    let m = re.captures(formula)?;
    let title = m.get(1).or_else(|| m.get(2))?.as_str();
    let range = m.get(3)?.as_str();
    if title != sheet {
        return None;
    }
    let (c, r, cc, rr) = workbook::bounds(range)?;
    Some((
        r.checked_sub(1)?,
        c.checked_sub(1)?,
        rr.checked_sub(1)?,
        cc.checked_sub(1)?,
    ))
}
