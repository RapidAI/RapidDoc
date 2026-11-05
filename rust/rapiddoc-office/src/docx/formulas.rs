//! Formula carrier arbitration and the original Word-formatted cell rebuild.
use super::{
    table_html::{Html, Reader, Resources},
    table_markup, *,
};
use crate::{
    equation::{equation_xml, omml},
    links, ErrorKind,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Omml,
    EquationXml,
    Mtef,
    ImageMtef,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Text(String),
    Formula(Kind, String),
}
pub trait Embedded {
    fn decode(&mut self, node: Node<'_, '_>) -> Result<Option<(Kind, String)>>;
}
/// Equation XML is native. Binary/image MTEF remains a required pending handler.
#[derive(Default)]
pub struct NativeEmbedded {
    pub equation_xml: equation_xml::Decoder,
}
impl Embedded for NativeEmbedded {
    fn decode(&mut self, node: Node<'_, '_>) -> Result<Option<(Kind, String)>> {
        if node.tag_name().namespace() == Some("urn:schemas-microsoft-com:vml")
            && node.tag_name().name() == "shape"
        {
            return self
                .equation_xml
                .decode(node.attribute("equationxml"))
                .map(|value| value.map(|s| (Kind::EquationXml, s)));
        }
        if node.tag_name().name() == "OLEObject" {
            let id = node
                .attribute("ProgID")
                .or_else(|| node.attribute("ProgId"))
                .unwrap_or("")
                .trim_matches(space);
            let id = crate::ole::casefold(id);
            if !(id == "equation" || id.starts_with("equation.") && id.len() > 9)
                || !node
                    .attribute("Type")
                    .unwrap_or("Embed")
                    .trim_matches(space)
                    .eq_ignore_ascii_case("embed")
                || node
                    .attribute("DrawAspect")
                    .unwrap_or("Content")
                    .trim_matches(space)
                    .eq_ignore_ascii_case("icon")
            {
                return Ok(None);
            }
        }
        Err(Error::new(
            ErrorKind::Unsupported,
            format!(
                "DOCX embedded/image MTEF decoder not connected: {}",
                node.tag_name().name()
            ),
        ))
    }
}
pub struct Tokens<E> {
    pub embedded: E,
    pub warnings: Vec<String>,
}
impl<E: Embedded> Tokens<E> {
    pub fn new(embedded: E) -> Self {
        Self {
            embedded,
            warnings: Vec::new(),
        }
    }
    pub fn extract(&mut self, node: Node<'_, '_>) -> Result<Vec<Token>> {
        if !node.is_element() {
            return Ok(vec![]);
        }
        let tag = node.tag_name().name();
        if tag == "AlternateContent" || is_word(node, "object") {
            let mut groups = Vec::new();
            for n in node.children().filter(|n| {
                n.is_element()
                    && (tag != "AlternateContent"
                        || matches!(n.tag_name().name(), "Choice" | "Fallback"))
            }) {
                groups.push(self.extract(n)?);
            }
            for kind in [Kind::Omml, Kind::EquationXml, Kind::Mtef, Kind::ImageMtef] {
                if let Some(index) = groups.iter().position(|tokens| {
                    tokens
                        .iter()
                        .any(|t| matches!(t,Token::Formula(k,_) if *k==kind))
                }) {
                    return Ok(groups.swap_remove(index));
                }
            }
            return Ok(groups.into_iter().next().unwrap_or_default());
        }
        if tag == "txbxContent" {
            return Ok(vec![]);
        }
        if tag == "oMath" && omml::math_namespace(node.tag_name().namespace()) {
            return match omml::convert(node) {
                Ok(value) => {
                    let value = value.trim_matches(space);
                    Ok(if value.is_empty() {
                        vec![]
                    } else {
                        vec![Token::Formula(Kind::Omml, value.into())]
                    })
                }
                Err(error) => {
                    self.warnings.push(error.to_string());
                    Ok(vec![])
                } // Original Python skips malformed OMML.
            };
        }
        let carrier = tag == "OLEObject"
            || node.tag_name().namespace() == Some("urn:schemas-microsoft-com:vml")
                && matches!(tag, "shape" | "imagedata")
            || tag == "blip"
                && crate::package::namespace_family(node.tag_name().namespace(), "drawingml/main");
        if carrier {
            if let Some((kind, value)) = self.embedded.decode(node)? {
                return Ok(vec![Token::Formula(kind, value)]);
            }
        }
        if tag == "t" && !omml::math_namespace(node.tag_name().namespace()) {
            return Ok(node
                .text()
                .map(|s| vec![Token::Text(s.into())])
                .unwrap_or_default());
        }
        let text = if is_word(node, "tab") || is_word(node, "ptab") {
            Some("\t")
        } else if is_word(node, "cr")
            || is_word(node, "br") && matches!(attr(node, "type"), None | Some("textWrapping"))
        {
            Some("\n")
        } else if is_word(node, "noBreakHyphen") {
            Some("-")
        } else {
            None
        };
        if let Some(text) = text {
            return Ok(vec![Token::Text(text.into())]);
        }
        let mut out = Vec::new();
        for n in node.children().filter(|n| n.is_element()) {
            out.extend(self.extract(n)?);
        }
        Ok(out)
    }
}
/// The Mammoth stage ignores math; the following injection pass restores it.
/// This wrapper is only valid together with injection, never a complete parser.
pub struct BeforeInjection<R>(pub R);
impl<R: Resources> Resources for BeforeInjection<R> {
    fn render(&mut self, node: Node<'_, '_>) -> Result<Vec<Html>> {
        if omml::math_namespace(node.tag_name().namespace()) {
            Ok(vec![])
        } else {
            self.0.render(node)
        }
    }
}
fn styled(text: &str, style: Option<&Formatting>, href: Option<&str>) -> Vec<Html> {
    if text.is_empty() {
        return vec![];
    }
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut nodes = Vec::new();
    for (index, line) in text.split('\n').enumerate() {
        if index > 0 {
            nodes.push(Html::element("br", false, vec![]));
        }
        if !line.is_empty() {
            nodes.push(Html::Text(line.into()));
        }
    }
    if let Some(s) = style {
        use super::styles::Script;
        let script = match s.script {
            Script::Super => Some("sup"),
            Script::Sub => Some("sub"),
            _ => None,
        };
        if let Some(tag) = script {
            nodes = vec![Html::element(tag, false, nodes)];
        }
        for (active, tag) in [
            (s.underline, "u"),
            (s.bold, "strong"),
            (s.italic, "em"),
            (s.strikethrough, "s"),
        ] {
            if active {
                nodes = vec![Html::element(tag, false, nodes)];
            }
        }
    }
    if let Some(href) = links::sanitize(href) {
        nodes = vec![Html::element("a", false, nodes).attribute("href", href)];
    }
    nodes
}
fn render_node<E: Embedded>(
    context: &Context<'_>,
    node: Node<'_, '_>,
    paragraph: Node<'_, '_>,
    target: Option<&str>,
    tokens: &mut Tokens<E>,
) -> Result<Vec<Html>> {
    let name = node.tag_name().name();
    if name == "hyperlink" {
        let id = node
            .attributes()
            .find(|a| {
                a.name() == "id" && a.namespace().is_some_and(|n| n.ends_with("/relationships"))
            })
            .map(|a| a.value());
        let address = id
            .and_then(|id| context.relationships.get(id))
            .map(|r| r.target.as_str())
            .unwrap_or("");
        let link = attr(node, "anchor")
            .filter(|s| !s.is_empty())
            .map(|a| format!("{address}#{a}"))
            .unwrap_or_else(|| address.into());
        let mut out = Vec::new();
        for child in node.children().filter(|n| n.is_element()) {
            out.extend(render_node(
                context,
                child,
                paragraph,
                if link.is_empty() { None } else { Some(&link) },
                tokens,
            )?);
        }
        return Ok(out);
    }
    if matches!(name, "p" | "sdt" | "sdtContent" | "smartTag" | "customXml") {
        let mut out = Vec::new();
        for child in node.children().filter(|n| n.is_element()) {
            out.extend(render_node(context, child, paragraph, target, tokens)?);
        }
        return Ok(out);
    }
    let style = if name == "r" {
        Some(context.styles.format(node, paragraph)?)
    } else {
        None
    };
    let mut out = Vec::new();
    for token in tokens.extract(node)? {
        match token {
            Token::Text(text) => out.extend(styled(&text, style.as_ref(), target)),
            Token::Formula(_, value) => {
                out.push(Html::element("eq", false, vec![Html::Text(value)]))
            }
        }
    }
    Ok(out)
}
pub fn paragraph<E: Embedded>(
    context: &Context<'_>,
    node: Node<'_, '_>,
    tokens: &mut Tokens<E>,
) -> Result<Option<Html>> {
    let children = render_node(context, node, node, None, tokens)?;
    Ok(if children.is_empty() {
        None
    } else {
        Some(Html::element("p", false, children))
    })
}
fn contains(tokens: &[Token]) -> bool {
    tokens.iter().any(|t| matches!(t, Token::Formula(..)))
}
/// Preserve the existing row/cell count guards, including nested descendants.
pub fn inject<E: Embedded>(
    context: &Context<'_>,
    html: &mut Html,
    xml: Node<'_, '_>,
    tokens: &mut Tokens<E>,
) -> Result<()> {
    if !contains(&tokens.extract(xml)?) {
        return Ok(());
    }
    let rows = table_markup::matching_paths(std::slice::from_ref(html), &["tr"]);
    let xml_rows: Vec<_> = xml.children().filter(|n| is_word(*n, "tr")).collect();
    if rows.len() != xml_rows.len() {
        return Ok(());
    }
    for (row, xml_row) in rows.into_iter().zip(xml_rows) {
        let row_node = table_markup::node_at(std::slice::from_ref(html), &row);
        let cells =
            table_markup::matching_paths(table_markup::child_nodes(row_node), &["td", "th"]);
        let xml_cells: Vec<_> = xml_row.children().filter(|n| is_word(*n, "tc")).collect();
        if cells.len() != xml_cells.len() {
            continue;
        }
        for (cell, xml_cell) in cells.into_iter().zip(xml_cells) {
            if !contains(&tokens.extract(xml_cell)?) {
                continue;
            }
            let mut content = Vec::new();
            // The current Python rebuild only direct paragraphs. Nested tables
            // remain governed by its outer injection/count guards.
            for p in xml_cell
                .children()
                .filter(|n| n.is_element() && n.tag_name().name() == "p")
            {
                if let Some(p) = paragraph(context, p, tokens)? {
                    content.push(p);
                }
            }
            if !content.is_empty() {
                html_whitespace(&mut content, false);
                let mut path = row.clone();
                path.extend(cell);
                if let Html::Element { children, .. } =
                    table_markup::node_at_mut(std::slice::from_mut(html), &path)
                {
                    *children = content;
                }
            }
        }
    }
    Ok(())
}
/// Primary preparse ordering: convert/collapse, inject equations, then colspan.
/// The document prepass supplies aligned nodes with styles and references;
/// the isolated-table API below retains full-context fallback semantics.
pub fn table<R: Resources, E: Embedded>(
    context: &Context<'_>,
    reader: &mut Reader<BeforeInjection<R>>,
    xml: Node<'_, '_>,
    tokens: &mut Tokens<E>,
) -> Result<String> {
    table_from_prepared(context, reader.prepared(xml)?, xml, tokens)
}
pub(super) fn table_from_prepared<E: Embedded>(
    context: &Context<'_>,
    mut nodes: Vec<Html>,
    xml: Node<'_, '_>,
    tokens: &mut Tokens<E>,
) -> Result<String> {
    html_whitespace(&mut nodes, false);
    if let Some(html) = nodes
        .iter_mut()
        .find(|n| matches!(n,Html::Element {names,..} if names[0]=="table"))
    {
        inject(context, html, xml, tokens)?;
    }
    table_markup::normalize_colspans(&mut nodes);
    normalize_links(&mut nodes);
    let mut output = String::new();
    table_markup::write_table(&nodes, &mut output);
    Ok(output)
}
pub fn normalize_links(nodes: &mut Vec<Html>) {
    let mut output = Vec::with_capacity(nodes.len());
    for mut node in std::mem::take(nodes) {
        if let Html::Element {
            names,
            attributes,
            children,
            ..
        } = &mut node
        {
            normalize_links(children);
            if names[0] == "a"
                && attributes
                    .get("href")
                    .is_some_and(|href| links::sanitize(Some(href)).is_none())
            {
                output.append(children);
                continue;
            }
        }
        output.push(node);
    }
    *nodes = output;
}
fn html_whitespace(nodes: &mut Vec<Html>, preserve: bool) {
    // BeautifulSoup's existing table boundary normalizes ASCII-whitespace-only
    // data. Do it directly on the tree, after joining adjacent text fragments.
    let mut output = Vec::with_capacity(nodes.len());
    for node in std::mem::take(nodes) {
        match (output.last_mut(), node) {
            (Some(Html::Text(previous)), Html::Text(next)) => previous.push_str(&next),
            (_, node) => output.push(node),
        }
    }
    for node in &mut output {
        match node {
            Html::Text(text)
                if !preserve
                    && !text.is_empty()
                    && text
                        .chars()
                        .all(|c| matches!(c, ' ' | '\n' | '\t' | '\x0c' | '\r')) =>
            {
                *text = if text.contains('\n') { "\n" } else { " " }.into();
            }
            Html::Element {
                names, children, ..
            } => html_whitespace(
                children,
                preserve || matches!(names[0].as_str(), "pre" | "textarea"),
            ),
            _ => (),
        }
    }
    *nodes = output;
}
