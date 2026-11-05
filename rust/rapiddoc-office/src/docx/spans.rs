//! Original rich_text.py / content/spans.py typed boundary, including Unicode
//! character slicing for formula insertion (not byte offsets).
use super::{formulas::Token, space, Element, Formatting};
use serde::Serialize;
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Span {
    #[serde(rename = "text")]
    Text {
        content: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        styles: Vec<&'static str>,
    },
    #[serde(rename = "equation_inline")]
    Equation { content: String },
    #[serde(rename = "hyperlink")]
    Link {
        url: String,
        #[serde(serialize_with = "serialize_link_children")]
        content: Vec<Span>,
    },
}
fn serialize_link_children<S: serde::Serializer>(
    children: &[Span],
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    // Pydantic's HyperlinkSpan retains default styles=[] on nested TextSpan;
    // top-level normalized text deliberately omits that empty field.
    struct Child<'a>(&'a Span);
    impl Serialize for Child<'_> {
        fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
            if let Span::Text { content, styles } = self.0 {
                use serde::ser::SerializeMap;
                let mut m = s.serialize_map(Some(3))?;
                m.serialize_entry("type", "text")?;
                m.serialize_entry("content", content)?;
                m.serialize_entry("styles", styles)?;
                m.end()
            } else {
                self.0.serialize(s)
            }
        }
    }
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(children.len()))?;
    for child in children {
        seq.serialize_element(&Child(child))?;
    }
    seq.end()
}
fn styles(f: Option<&Formatting>) -> Vec<&'static str> {
    let Some(f) = f else {
        return vec![];
    };
    let mut s = vec![];
    for (on, name) in [
        (f.bold, "bold"),
        (f.italic, "italic"),
        (f.underline, "underline"),
        (f.emphasis, "emphasis"),
        (f.strikethrough, "strikethrough"),
        (
            matches!(f.script, super::styles::Script::Super),
            "superscript",
        ),
        (matches!(f.script, super::styles::Script::Sub), "subscript"),
    ] {
        if on {
            s.push(name);
        }
    }
    s
}
pub fn append(out: &mut Vec<Span>, next: Span) {
    if let Span::Text { content, styles } = &next {
        if content.is_empty() {
            return;
        }
        if let Some(Span::Text {
            content: previous,
            styles: old,
        }) = out.last_mut()
        {
            if old == styles {
                previous.push_str(content);
                return;
            }
        }
    }
    out.push(next);
}
pub fn text(out: &mut Vec<Span>, value: String) {
    append(
        out,
        Span::Text {
            content: value,
            styles: vec![],
        },
    );
}
pub fn visible(spans: &[Span]) -> String {
    let mut s = String::new();
    for span in spans {
        match span {
            Span::Text { content, .. } | Span::Equation { content } => s.push_str(content),
            Span::Link { content, .. } => s.push_str(&visible(content)),
        }
    }
    s
}
pub fn slice(spans: &[Span], start: usize, end: usize) -> Vec<Span> {
    let mut out = vec![];
    let mut cursor = 0;
    for span in spans {
        let len = match span {
            Span::Text { content, .. } | Span::Equation { content } => content.chars().count(),
            Span::Link { content, .. } => visible(content).chars().count(),
        };
        let a = start.max(cursor);
        let b = end.min(cursor + len);
        if a < b {
            let a = a - cursor;
            let b = b - cursor;
            match span {
                Span::Text { content, styles } => append(
                    &mut out,
                    Span::Text {
                        content: content.chars().skip(a).take(b - a).collect(),
                        styles: styles.clone(),
                    },
                ),
                Span::Equation { content } => {
                    let content: String = content.chars().skip(a).take(b - a).collect();
                    if !content.trim_matches(space).is_empty() {
                        out.push(Span::Equation { content });
                    }
                }
                Span::Link { url, content } => {
                    let content = slice(content, a, b);
                    if !content.is_empty() {
                        out.push(Span::Link {
                            url: url.clone(),
                            content,
                        });
                    }
                }
            }
        }
        cursor += len;
        if cursor >= end {
            break;
        }
    }
    out
}
pub fn strip(spans: Vec<Span>) -> Vec<Span> {
    let text = visible(&spans);
    let start = text.chars().take_while(|c| space(*c)).count();
    let end = text.chars().count() - text.chars().rev().take_while(|c| space(*c)).count();
    slice(&spans, start, end.max(start))
}
pub fn from_elements(elements: &[Element]) -> Vec<Span> {
    let mut out = vec![];
    let mut group: Option<(String, Vec<Span>)> = None;
    for e in elements.iter().filter(|e| !e.text.is_empty()) {
        let href = crate::links::sanitize(e.hyperlink.as_deref());
        if group.as_ref().map(|(url, _)| url.as_str()) != href.as_deref() {
            if let Some((url, content)) = group.take() {
                out.push(Span::Link { url, content });
            }
        }
        if let Some(url) = href {
            let (_, children) = group.get_or_insert_with(|| (url.into(), vec![]));
            append(
                children,
                Span::Text {
                    content: e.text.clone(),
                    styles: styles(e.format.as_ref()),
                },
            );
        } else {
            append(
                &mut out,
                Span::Text {
                    content: e.text.replace("\r\n", "\n").replace('\r', "\n"),
                    styles: styles(e.format.as_ref()),
                },
            );
        }
    }
    if let Some((url, content)) = group {
        out.push(Span::Link { url, content });
    }
    out
}
/// Both original guards are preserved: formula token text must reconstruct the
/// raw paragraph and then the styled visible text, otherwise use styled text.
pub fn with_formulas(elements: &[Element], raw: &str, tokens: &[Token]) -> (Vec<Span>, bool) {
    let styled = from_elements(elements);
    let has = tokens.iter().any(|t| matches!(t, Token::Formula(..)));
    let plain = tokens
        .iter()
        .filter_map(|t| {
            if let Token::Text(s) = t {
                Some(s.as_str())
            } else {
                None
            }
        })
        .collect::<String>();
    if !has || plain != raw {
        return (styled, false);
    }
    if plain != visible(&styled) {
        return (styled, true);
    }
    let mut out = vec![];
    let mut cursor = 0;
    for token in tokens {
        match token {
            Token::Text(s) => {
                let end = cursor + s.chars().count();
                for span in slice(&styled, cursor, end) {
                    append(&mut out, span);
                }
                cursor = end;
            }
            Token::Formula(_, s) => {
                let s = s.trim_matches(space);
                if !s.is_empty() {
                    out.push(Span::Equation { content: s.into() });
                }
            }
        }
    }
    (strip(out), true)
}
