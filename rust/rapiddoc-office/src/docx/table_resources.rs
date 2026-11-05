//! Whole-document Mammoth table prepass, note/comment resources and alignment.
//! Only matched table elements enter DOCX pages; referent bodies retain the
//! original validation/conversion behavior without adding a new footnote export.
use super::{
    embedded::Shared,
    formulas::{self, BeforeInjection, Embedded, Tokens},
    table_html::{Html, Reader},
    table_markup, *,
};

struct Referent {
    xml: String,
    relationships: Arc<Vec<Relationship>>,
    initials: String,
}
#[derive(Default)]
struct Referents {
    notes: BTreeMap<(String, String), Referent>,
    comments: BTreeMap<String, Referent>,
}
struct ReadOnlyResources;
impl super::table_html::Resources for ReadOnlyResources {
    fn render(&mut self, _: Node<'_, '_>) -> Result<Vec<Html>> {
        Ok(vec![])
    }
}
impl Referents {
    fn load(
        shared: &Shared<'_>,
        rels: &[Relationship],
        styles: Option<&str>,
        numbering: Option<&str>,
    ) -> Result<Self> {
        let mut result = Self::default();
        for (kind, partkind) in [
            ("footnote", "footnotes"),
            ("endnote", "endnotes"),
            ("comment", "comments"),
        ] {
            let mut paths=rels.iter().filter(|r|!r.external&&r.kind==format!("http://schemas.openxmlformats.org/officeDocument/2006/relationships/{partkind}")).filter_map(|r|r.resolved.clone()).collect::<Vec<_>>();
            paths.push(format!("word/{partkind}.xml"));
            let mut selected = None;
            for path in paths {
                if let Some(text) = shared.read_xml(&path)? {
                    selected = Some((path, text));
                    break;
                }
            }
            let Some((path, text)) = selected else {
                continue;
            };
            let tree = xml::parse(&text)?;
            let relationships = Arc::new(shared.part_relationships(&path)?);
            // Mammoth reads every non-separator note before converting the
            // filtered document. Even an unused malformed note selects fallback.
            // This pass reads XML only: unused note images are not rendered.
            let mut validator = Reader::new(styles, numbering, &relationships, ReadOnlyResources)?;
            for n in tree.root_element().children().filter(|n| is_word(*n, kind)) {
                if kind != "comment"
                    && matches!(attr(n, "type"), Some("separator" | "continuationSeparator"))
                {
                    continue;
                }
                let id =
                    attr(n, "id").ok_or_else(|| Error::malformed("note/comment without id"))?;
                validator.read(n)?;
                let referent = Referent {
                    xml: super::table_html::fragment(n),
                    relationships: relationships.clone(),
                    initials: attr(n, "initials").unwrap_or("").trim_matches(space).into(),
                };
                if kind == "comment" {
                    result.comments.insert(id.into(), referent);
                } else {
                    result.notes.insert((kind.into(), id.into()), referent);
                }
            }
        }
        Ok(result)
    }
    fn initials(&self) -> Arc<BTreeMap<String, String>> {
        Arc::new(
            self.comments
                .iter()
                .map(|(id, r)| (id.clone(), r.initials.clone()))
                .collect(),
        )
    }
}

pub(super) struct Tables {
    style_map: String,
    prepared: BTreeMap<usize, String>,
    setup_error: Option<Error>,
}
impl Tables {
    pub fn failed(error: Error) -> Self {
        Self {
            style_map: String::new(),
            prepared: BTreeMap::new(),
            setup_error: Some(error),
        }
    }
    pub fn prepare<E: Embedded>(
        context: &Context<'_>,
        body: Node<'_, '_>,
        styles: Option<&str>,
        numbering: Option<&str>,
        rels: &[Relationship],
        shared: &Shared<'_>,
        tokens: &mut Tokens<E>,
        diagnostics: &mut Vec<String>,
    ) -> Result<Self> {
        let mut result = Self {
            style_map: String::new(),
            prepared: BTreeMap::new(),
            setup_error: None,
        };
        let xml_tables = body
            .children()
            .filter(|n| is_word(*n, "tbl"))
            .collect::<Vec<_>>();
        if xml_tables.is_empty() {
            return Ok(result);
        }
        if let Some(bytes) = shared.read_part("mammoth/style-map")? {
            result.style_map = std::str::from_utf8(&bytes)
                .map_err(Error::malformed)?
                .into();
        }
        let prepass = (|| -> Result<Vec<Html>> {
            let referents = Referents::load(shared, rels, styles, numbering)?;
            let mut reader =
                Reader::new(styles, numbering, rels, BeforeInjection(shared.images()))?;
            diagnostics.extend(reader.set_style_map(&result.style_map));
            let initials = referents.initials();
            reader.comments = Arc::clone(&initials);
            let mut nodes = vec![];
            for n in &xml_tables {
                nodes.extend(reader.read(*n)?);
            }
            // Resolve the initial complete reference list before visiting note
            // bodies; references inside a note do not grow the notes output list.
            let notes = reader
                .note_references
                .iter()
                .map(|key| {
                    referents
                        .notes
                        .get(key)
                        .ok_or_else(|| Error::missing(&format!("{} {}", key.0, key.1)))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut note_nodes = vec![];
            for note in notes {
                shared.switch_relationships(&note.relationships);
                let tree = xml::parse(&note.xml)?;
                let mut local = Reader::new(
                    styles,
                    numbering,
                    &note.relationships,
                    BeforeInjection(shared.images()),
                )?;
                local.set_style_map(&result.style_map);
                local.comments = Arc::clone(&initials);
                local.inherit_references(&mut reader);
                let source = tree
                    .root_element()
                    .children()
                    .find(|n| n.is_element())
                    .unwrap();
                let kind = source.tag_name().name();
                let id = attr(source, "id").unwrap();
                let mut body = local.read(source)?;
                body.push(backlink(kind, id));
                note_nodes
                    .push(Html::element("li", false, body).attribute("id", format!("{kind}-{id}")));
                reader.inherit_references(&mut local);
            }
            nodes.push(Html::element("ol", false, note_nodes));
            let mut comment_nodes = vec![];
            let mut index = 0;
            while index < reader.comment_references.len() {
                let id = reader.comment_references[index].clone();
                index += 1;
                let comment = referents
                    .comments
                    .get(&id)
                    .ok_or_else(|| Error::missing(&id))?;
                shared.switch_relationships(&comment.relationships);
                let tree = xml::parse(&comment.xml)?;
                let mut local = Reader::new(
                    styles,
                    numbering,
                    &comment.relationships,
                    BeforeInjection(shared.images()),
                )?;
                local.set_style_map(&result.style_map);
                local.comments = Arc::clone(&initials);
                local.inherit_references(&mut reader);
                let source = tree
                    .root_element()
                    .children()
                    .find(|n| n.is_element())
                    .unwrap();
                let mut body = local.read(source)?;
                body.push(backlink("comment", &id));
                comment_nodes.push(
                    Html::element(
                        "dt",
                        false,
                        vec![Html::Text(format!("Comment [{}{index}]", comment.initials))],
                    )
                    .attribute("id", format!("comment-{id}")),
                );
                comment_nodes.push(Html::element("dd", false, body));
                reader.inherit_references(&mut local);
            }
            nodes.push(Html::element("dl", false, comment_nodes));
            Ok(table_markup::collapse(table_markup::strip_empty(nodes)))
        })();
        shared.switch_relationships(rels);
        let nodes = match prepass {
            Ok(nodes) => nodes,
            Err(error) => {
                diagnostics.push(format!("DOCX table prepass unavailable; using per-table full-context fallback: {error}"));
                return Ok(result);
            }
        };
        let mut candidates = vec![];
        top_tables(nodes, &mut candidates);
        let signatures = candidates.iter().map(html_signature).collect::<Vec<_>>();
        let mut position = 0;
        for table in xml_tables {
            let signature = super::tables::signature(table);
            if let Some(index) = (position..candidates.len())
                .find(|i| super::tables::signatures_match(&signature, &signatures[*i]))
            {
                position = index + 1;
                let node = std::mem::replace(&mut candidates[index], Html::ForceWrite);
                match formulas::table_from_prepared(context, vec![node], table, tokens) {
                    Ok(html) => {
                        result.prepared.insert(table.range().start, html);
                    }
                    Err(error) => {
                        diagnostics.push(format!("DOCX table alignment/injection failed; using per-table fallback: {error}"));
                        result.prepared.clear();
                        break;
                    }
                }
            }
        }
        Ok(result)
    }
    pub fn convert<E: Embedded>(
        &mut self,
        context: &Context<'_>,
        table: Node<'_, '_>,
        styles: Option<&str>,
        numbering: Option<&str>,
        rels: &[Relationship],
        shared: &Shared<'_>,
        tokens: &mut Tokens<E>,
    ) -> Result<String> {
        if let Some(error) = &self.setup_error {
            return Err(error.clone());
        }
        if let Some(html) = self.prepared.remove(&table.range().start) {
            return Ok(html);
        }
        // The original isolated-table fallback has a fresh conversion/reference
        // counter and no Document-level comments/notes resolution.
        let mut reader = Reader::new(styles, numbering, rels, BeforeInjection(shared.images()))?;
        reader.set_style_map(&self.style_map);
        formulas::table(context, &mut reader, table, tokens)
    }
}
fn top_tables(nodes: Vec<Html>, out: &mut Vec<Html>) {
    for n in nodes {
        match n {
            Html::Element { ref names, .. } if names[0] == "table" => out.push(n),
            Html::Element { children, .. } => top_tables(children, out),
            _ => (),
        }
    }
}
fn html_signature(node: &Html) -> super::tables::Signature {
    let mut result = super::tables::Signature {
        row_count: 0,
        cell_count: 0,
        image_count: 0,
        text: String::new(),
    };
    fn walk(n: &Html, parent: &str, skip: bool, out: &mut super::tables::Signature) {
        match n {
            Html::Text(s) if !skip => out.text.extend(s.chars().filter(|c| !space(*c))),
            Html::Element {
                names,
                attributes,
                children,
                ..
            } => {
                let tag = names[0].as_str();
                match tag {
                    "tr" => out.row_count += 1,
                    "td" | "th" => out.cell_count += 1,
                    "img" => out.image_count += 1,
                    _ => (),
                }
                let marker = if tag == "a" && parent == "sup" {
                    attributes
                        .get("id")
                        .and_then(|s| s.split_once("-ref-"))
                        .is_some_and(|(kind, id)| {
                            matches!(kind, "footnote" | "endnote")
                                && !id.is_empty()
                                && id.chars().all(|c| super::decimal(c).is_some())
                                && attributes.get("href") == Some(&format!("#{kind}-{id}"))
                        })
                } else {
                    false
                };
                for c in children {
                    walk(c, tag, skip || marker, out);
                }
            }
            _ => (),
        }
    }
    walk(node, "", false, &mut result);
    result
}

fn backlink(kind: &str, id: &str) -> Html {
    Html::element(
        "p",
        true,
        vec![
            Html::Text(" ".into()),
            Html::element("a", false, vec![Html::Text("↑".into())])
                .attribute("href", format!("#{kind}-ref-{id}")),
        ],
    )
}
