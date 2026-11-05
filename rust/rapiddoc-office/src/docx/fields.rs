use super::*;
use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub(super) enum Content<'a, 'input> {
    Run(Node<'a, 'input>),
    Hyperlink(Node<'a, 'input>),
}
pub(super) fn contents<'a, 'input>(paragraph: Node<'a, 'input>) -> Vec<Content<'a, 'input>> {
    let mut pending: Vec<_> = paragraph.children().rev().collect();
    let mut output = Vec::new();
    while let Some(node) = pending.pop() {
        if is_word(node, "r") {
            output.push(Content::Run(node));
        } else if is_word(node, "hyperlink") {
            output.push(Content::Hyperlink(node));
        } else if is_word(node, "sdt") {
            if let Some(content) = child(node, "sdtContent") {
                pending.extend(content.children().rev());
            }
        } else if [
            "bdo",
            "customXml",
            "dir",
            "fldSimple",
            "ins",
            "moveTo",
            "smartTag",
        ]
        .iter()
        .any(|name| is_word(node, name))
        {
            pending.extend(node.children().rev());
        }
    }
    output
}
pub(super) fn run_text(run: Node<'_, '_>) -> String {
    let mut output = String::new();
    for node in run.children().filter(|n| n.is_element()) {
        if is_word(node, "t") {
            output.push_str(&text(node));
        } else if is_word(node, "tab") || is_word(node, "ptab") {
            output.push('\t');
        } else if is_word(node, "cr")
            || (is_word(node, "br")
                && attr(node, "type").unwrap_or("textWrapping") == "textWrapping")
        {
            output.push('\n');
        } else if is_word(node, "noBreakHyphen") {
            output.push('-');
        }
    }
    output
}
pub(super) fn hidden(run: Node<'_, '_>) -> bool {
    child(run, "rPr")
        .is_some_and(|p| child(p, "webHidden").is_some() || child(p, "vanish").is_some())
}
pub(super) fn plain_text(contents: &[Content<'_, '_>]) -> String {
    let mut output = String::new();
    for content in contents {
        match content {
            Content::Run(run) => output.push_str(&run_text(*run)),
            Content::Hyperlink(link) => {
                for run in link.children().filter(|n| is_word(*n, "r")) {
                    output.push_str(&run_text(run));
                }
            }
        }
    }
    output
}
#[derive(Default)]
struct Frame {
    instruction: String,
    result: Vec<Element>,
    result_phase: bool,
}
pub(super) fn hyperlink_instruction(instruction: &str) -> (Option<String>, bool) {
    static EXTERNAL: OnceLock<regex::Regex> = OnceLock::new();
    static BOOKMARK: OnceLock<regex::Regex> = OnceLock::new();
    let external =
        EXTERNAL.get_or_init(|| regex::Regex::new(r#"(?i)\bHYPERL[iİı]NK\s+"([^"]+)""#).unwrap());
    let bookmark = BOOKMARK.get_or_init(|| regex::Regex::new(r#"(?i)\\l\s+"([^"]+)""#).unwrap());
    let address = external
        .captures(instruction)
        .map(|c| c[1].trim_matches(space).to_owned())
        .unwrap_or_default();
    let fragment = bookmark
        .captures(instruction)
        .map(|c| c[1].trim_matches(space).to_owned())
        .unwrap_or_default();
    if !address.is_empty() {
        (
            Some(if fragment.is_empty() {
                address
            } else {
                format!("{address}#{fragment}")
            }),
            false,
        )
    } else if !fragment.is_empty() {
        (Some(format!("#{fragment}")), true)
    } else {
        (None, false)
    }
}
fn resolved(mut frame: Frame, suppress: bool) -> Vec<Element> {
    let (target, internal) = hyperlink_instruction(&frame.instruction);
    if target.is_some() && !(internal && suppress) {
        for element in &mut frame.result {
            if element.hyperlink.as_ref().is_none_or(|s| s.is_empty()) {
                element.hyperlink = target.clone();
            }
        }
    }
    frame.result
}
fn flush(result: Vec<Element>, stack: &mut [Frame], output: &mut Vec<Element>) {
    if let Some(frame) = stack.last_mut().filter(|f| f.result_phase) {
        frame.result.extend(result);
    } else {
        output.extend(result);
    }
}
fn adjacent(
    contents: &[Content<'_, '_>],
    index: usize,
    step: isize,
    paragraph: Node<'_, '_>,
    styles: &Styles,
) -> Result<Option<Formatting>> {
    let mut index = index as isize + step;
    while index >= 0 && index < contents.len() as isize {
        match contents[index as usize] {
            Content::Hyperlink(_) => return Ok(None),
            Content::Run(run) if !hidden(run) && !run_text(run).trim_matches(space).is_empty() => {
                return styles.format(run, paragraph).map(Some)
            }
            _ => {}
        }
        index += step;
    }
    Ok(None)
}
fn direct_link(link: Node<'_, '_>, relationships: &BTreeMap<String, Relationship>) -> String {
    let id = link
        .attributes()
        .find(|a| {
            a.name() == "id"
                && a.namespace()
                    .is_some_and(|ns| ns.ends_with("/relationships"))
        })
        .map(|a| a.value());
    let address = id
        .and_then(|id| relationships.get(id))
        .map(|r| r.target.as_str())
        .unwrap_or("");
    let fragment = attr(link, "anchor").unwrap_or("");
    if !address.is_empty() && !fragment.is_empty() {
        format!("{address}#{fragment}")
    } else if !address.is_empty() && address.contains("://") {
        address.to_owned()
    } else if !address.is_empty() {
        std::path::Path::new(address)
            .components()
            .collect::<std::path::PathBuf>()
            .display()
            .to_string()
    } else if !fragment.is_empty() {
        format!("#{fragment}")
    } else {
        ".".to_owned()
    }
}
fn flatten(
    contents: &[Content<'_, '_>],
    paragraph: Node<'_, '_>,
    styles: &Styles,
    relationships: &BTreeMap<String, Relationship>,
    suppress: bool,
) -> Result<Vec<Element>> {
    let mut stack: Vec<Frame> = Vec::new();
    let mut output = Vec::new();
    for (index, content) in contents.iter().enumerate() {
        let run = match content {
            Content::Hyperlink(link) => {
                let target = direct_link(*link, relationships);
                let target = if suppress && target.starts_with('#') {
                    None
                } else {
                    Some(target)
                };
                let mut elements = Vec::new();
                for run in link.children().filter(|n| is_word(*n, "r") && !hidden(*n)) {
                    let text = run_text(run);
                    let format = styles.format(run, paragraph)?.for_text(&text, true);
                    if !text.is_empty() || format.as_ref().is_some_and(Formatting::visible) {
                        elements.push(Element {
                            text,
                            format,
                            hyperlink: target.clone(),
                        });
                    }
                }
                flush(elements, &mut stack, &mut output);
                continue;
            }
            Content::Run(run) => *run,
        };
        // The Python implementation filters hidden hyperlink runs but keeps
        // ordinary runs here. Preserve that existing distinction.
        if let Some(field) = child(run, "fldChar") {
            match attr(field, "fldCharType") {
                Some("begin") => stack.push(Frame::default()),
                Some("separate") => {
                    if let Some(frame) = stack.last_mut() {
                        frame.result_phase = true;
                    }
                }
                Some("end") => {
                    if let Some(frame) = stack.pop() {
                        flush(resolved(frame, suppress), &mut stack, &mut output);
                    }
                }
                _ => {}
            }
            continue;
        }
        if let Some(instruction) = child(run, "instrText") {
            if let Some(frame) = stack.last_mut().filter(|f| !f.result_phase) {
                frame.instruction.push_str(&text(instruction));
                continue;
            }
        }
        let text = run_text(run);
        let format = styles.format(run, paragraph)?;
        let preserve = !text.is_empty()
            && text.trim_matches(space).is_empty()
            && format.nonvisible()
            && (adjacent(contents, index, -1, paragraph, styles)?.as_ref() == Some(&format)
                || adjacent(contents, index, 1, paragraph, styles)?.as_ref() == Some(&format));
        let format = format.for_text(&text, preserve);
        let element = Element {
            text,
            format,
            hyperlink: None,
        };
        if let Some(frame) = stack.last_mut() {
            if frame.result_phase {
                frame.result.push(element);
            }
        } else {
            output.push(element);
        }
    }
    while let Some(frame) = stack.pop() {
        flush(resolved(frame, suppress), &mut stack, &mut output);
    }
    Ok(output)
}
fn visible(format: Option<&Formatting>) -> bool {
    format.is_some_and(Formatting::visible)
}
fn keep(text: &str, format: Option<&Formatting>, plain_blank: bool) -> bool {
    !text.is_empty() && (!text.trim_matches(space).is_empty() || visible(format) || plain_blank)
}
pub(super) fn elements(
    contents: &[Content<'_, '_>],
    paragraph: Node<'_, '_>,
    styles: &Styles,
    relationships: &BTreeMap<String, Relationship>,
    suppress: bool,
) -> Result<Vec<Element>> {
    if plain_text(contents).trim_matches(space).is_empty() {
        let mut found = false;
        for content in contents {
            if let Content::Run(run) = content {
                if !run_text(*run).is_empty() && styles.format(*run, paragraph)?.visible() {
                    found = true;
                    break;
                }
            }
        }
        if !found {
            return Ok(vec![Element {
                text: String::new(),
                format: None,
                hyperlink: None,
            }]);
        }
    }
    let flattened = flatten(contents, paragraph, styles, relationships, suppress)?;
    let mut output: Vec<Element> = Vec::new();
    let mut group = String::new();
    let mut previous: Option<Formatting> = None;
    for mut element in flattened {
        let content_visible =
            !element.text.trim_matches(space).is_empty() || visible(element.format.as_ref());
        let blank = !element.text.is_empty() && element.text.trim_matches(space).is_empty();
        let changed = previous != element.format;
        let boundary = visible(previous.as_ref()) || visible(element.format.as_ref());
        if (content_visible && changed)
            || (blank && !group.is_empty() && changed && boundary)
            || element.hyperlink.is_some()
        {
            let preserve = !group.is_empty() && group.trim_matches(space).is_empty() && boundary;
            if keep(&group, previous.as_ref(), preserve) {
                output.push(Element {
                    text: std::mem::take(&mut group),
                    format: previous.clone(),
                    hyperlink: None,
                });
            }
            group.clear();
            if element.hyperlink.is_some() {
                if let Some(last) = output.last_mut().filter(|last| {
                    last.hyperlink.is_some()
                        && last.hyperlink == element.hyperlink
                        && last.format == element.format
                }) {
                    last.text.push_str(&element.text);
                } else {
                    output.push(element.clone());
                }
                element.text.clear();
            } else {
                previous = element.format.clone();
            }
        }
        group.push_str(&element.text);
    }
    if keep(&group, previous.as_ref(), false) {
        output.push(Element {
            text: group,
            format: previous,
            hyperlink: None,
        });
    }
    Ok(link_boundaries(output))
}
fn link_boundaries(elements: Vec<Element>) -> Vec<Element> {
    let mut input = elements.into_iter().peekable();
    let mut output = Vec::new();
    while let Some(element) = input.next() {
        let Some(target) = element.hyperlink.as_ref() else {
            output.push(element);
            continue;
        };
        let mut group = vec![element.clone()];
        while input
            .peek()
            .is_some_and(|e| e.hyperlink.as_ref() == Some(target))
        {
            group.push(input.next().unwrap());
        }
        let first_format = group[0].format.clone();
        let last_format = group.last().unwrap().format.clone();
        let (leading, trailing) =
            if group.len() == 1 && group[0].text.trim_matches(space).is_empty() {
                (String::new(), std::mem::take(&mut group[0].text))
            } else {
                let leading = group[0].text
                    [..group[0].text.len() - group[0].text.trim_start_matches(space).len()]
                    .to_owned();
                group[0].text = group[0].text.trim_start_matches(space).to_owned();
                let last = group.last_mut().unwrap();
                let trailing = last.text[last.text.trim_end_matches(space).len()..].to_owned();
                last.text.truncate(last.text.trim_end_matches(space).len());
                (leading, trailing)
            };
        if !leading.is_empty() && !output.is_empty() {
            output.push(Element {
                text: leading,
                format: first_format,
                hyperlink: None,
            });
        }
        output.extend(group.into_iter().filter(|e| !e.text.is_empty()));
        if !trailing.is_empty() && input.peek().is_some() {
            output.push(Element {
                text: trailing,
                format: last_format,
                hyperlink: None,
            });
        }
    }
    output
}
pub(super) fn instructions(paragraph: Node<'_, '_>) -> Vec<String> {
    let mut stack: Vec<(String, bool)> = Vec::new();
    let mut output = Vec::new();
    let append = |text: &str, output: &mut Vec<String>| {
        let value = text.trim_matches(space);
        if !value.is_empty() {
            output.push(value.to_owned());
        }
    };
    for node in paragraph.descendants() {
        if is_word(node, "fldChar") {
            match attr(node, "fldCharType") {
                Some("begin") => stack.push((String::new(), false)),
                Some("separate") => {
                    if let Some((text, result)) = stack.last_mut() {
                        if !*result {
                            append(text, &mut output);
                            *result = true;
                        }
                    }
                }
                Some("end") => {
                    if let Some((text, result)) = stack.pop() {
                        if !result {
                            append(&text, &mut output);
                        }
                    }
                }
                _ => {}
            }
        } else if is_word(node, "instrText") {
            if let Some((value, _)) = stack.last_mut().filter(|(_, result)| !*result) {
                value.push_str(&text(node));
            } else {
                append(&text(node), &mut output);
            }
        }
    }
    for (text, result) in stack {
        if !result {
            append(&text, &mut output);
        }
    }
    output
}
