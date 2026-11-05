//! Mammoth embedded style-map grammar and first-matching-rule semantics.
//! The default mappings remain in table_html; embedded rules precede them.
use super::{space, table_markup::Html};
use num_bigint::BigInt;
use std::{collections::BTreeMap, sync::OnceLock};

#[derive(Clone, Debug, Default)]
pub struct Path {
    pub ignore: bool,
    pub elements: Vec<Html>,
}
impl Path {
    pub fn wrap(&self, mut nodes: Vec<Html>) -> Vec<Html> {
        if self.ignore {
            return vec![];
        }
        for element in self.elements.iter().rev() {
            let mut element = element.clone();
            if let Html::Element { children, .. } = &mut element {
                *children = nodes;
            }
            nodes = vec![element];
        }
        nodes
    }
}
#[derive(Debug, Clone)]
struct Matcher {
    kind: String,
    id: Option<String>,
    name: Option<(bool, String)>,
    numbering: Option<(String, bool)>,
    qualifier: Option<String>,
}
#[derive(Debug, Clone)]
struct Rule {
    matcher: Matcher,
    path: Path,
}
#[derive(Debug, Default, Clone)]
pub struct Rules {
    rules: Vec<Rule>,
    pub warnings: Vec<String>,
}
impl Rules {
    pub fn parse(text: &str) -> Self {
        let mut result = Self::default();
        for line in text
            .split('\n')
            .map(|s| s.trim_matches(space))
            .filter(|s| !s.is_empty() && !s.starts_with('#'))
        {
            if let Some(rule) = parse(line) {
                result.rules.push(rule);
            } else {
                result.warnings.push(format!(
                    "Did not understand this style mapping, so ignored it: {line}"
                ));
            }
        }
        result
    }
    pub fn find(
        &self,
        kind: &str,
        id: Option<&str>,
        name: Option<&str>,
        numbering: Option<(&str, bool)>,
        qualifier: Option<&str>,
    ) -> Option<&Path> {
        self.rules
            .iter()
            .find(|r| {
                let m = &r.matcher;
                m.kind == kind
                    && m.id.as_deref().is_none_or(|v| Some(v) == id)
                    && m.name.as_ref().is_none_or(|(prefix, value)| {
                        name.is_some_and(|s| {
                            if *prefix {
                                s.to_uppercase().starts_with(&value.to_uppercase())
                            } else {
                                s.to_uppercase() == value.to_uppercase()
                            }
                        })
                    })
                    && m.numbering.as_ref().is_none_or(|(level, ordered)| {
                        numbering == Some((level.as_str(), *ordered))
                    })
                    && m.qualifier.as_deref().is_none_or(|s| Some(s) == qualifier)
            })
            .map(|r| &r.path)
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Identifier,
    Symbol,
    Space,
    String,
    Integer,
    Unknown,
    End,
}
struct Token {
    kind: Kind,
    text: String,
}
struct Tokens {
    tokens: Vec<Token>,
    index: usize,
}
impl Tokens {
    fn take(&mut self, kind: Kind, value: Option<&str>) -> Option<String> {
        let t = self.tokens.get(self.index)?;
        if t.kind != kind || value.is_some_and(|v| v != t.text) {
            return None;
        }
        self.index += 1;
        Some(t.text.clone())
    }
    fn sym(&mut self, value: &str) -> bool {
        self.take(Kind::Symbol, Some(value)).is_some()
    }
    fn ident(&mut self, value: &str) -> bool {
        self.take(Kind::Identifier, Some(value)).is_some()
    }
    fn name(&mut self) -> Option<String> {
        Some(unescape(&self.take(Kind::Identifier, None)?))
    }
    fn string(&mut self) -> Option<String> {
        let s = self.take(Kind::String, None)?;
        Some(unescape(&s[1..s.len() - 1]))
    }
    fn space(&mut self) -> bool {
        self.take(Kind::Space, None).is_some()
    }
}
fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' && chars.peek().is_some_and(|c| *c != '\n') {
            out.push(match chars.next().unwrap() {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                c => c,
            });
        } else {
            out.push(c);
        }
    }
    out
}
fn tokens(s: &str) -> Tokens {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re=RE.get_or_init(||regex::Regex::new(r"(?s)\A(?:(?P<id>(?:[a-zA-Z\-_]|\\[^\n])(?:[a-zA-Z\-_0-9]|\\[^\n])*)|(?P<symbol>=>|\^=|[:>=()\[\]|!.])|(?P<space>[\s\x1c-\x1f]+)|(?P<string>'(?:\\[^\n]|[^'])*')|(?P<integer>[0-9]+)|(?P<unknown>.))").unwrap());
    let mut rest = s;
    let mut result = vec![];
    while !rest.is_empty() {
        let c = re.captures(rest).unwrap();
        let text = c.get(0).unwrap().as_str();
        let kind = if c.name("id").is_some() {
            Kind::Identifier
        } else if c.name("symbol").is_some() {
            Kind::Symbol
        } else if c.name("space").is_some() {
            Kind::Space
        } else if c.name("string").is_some() {
            Kind::String
        } else if c.name("integer").is_some() {
            Kind::Integer
        } else {
            Kind::Unknown
        };
        result.push(Token {
            kind,
            text: text.into(),
        });
        rest = &rest[text.len()..];
    }
    result.push(Token {
        kind: Kind::End,
        text: String::new(),
    });
    Tokens {
        tokens: result,
        index: 0,
    }
}
fn parse(s: &str) -> Option<Rule> {
    let mut t = tokens(s);
    let kind = t.take(Kind::Identifier, None)?;
    let mut matcher = Matcher {
        kind: kind.clone(),
        id: None,
        name: None,
        numbering: None,
        qualifier: None,
    };
    match kind.as_str() {
        "p" | "r" | "table" => {
            if t.sym(".") {
                matcher.id = Some(t.name()?);
            }
            if t.sym("[") {
                if !t.ident("style-name") {
                    return None;
                }
                let prefix = if t.sym("=") {
                    false
                } else if t.sym("^=") {
                    true
                } else {
                    return None;
                };
                matcher.name = Some((prefix, t.string()?));
                if !t.sym("]") {
                    return None;
                }
            }
            if kind == "p" && t.sym(":") {
                let ordered = if t.ident("ordered-list") {
                    true
                } else if t.ident("unordered-list") {
                    false
                } else {
                    return None;
                };
                if !t.sym("(") {
                    return None;
                }
                let index = t.take(Kind::Integer, None)?.parse::<BigInt>().ok()? - BigInt::from(1);
                if !t.sym(")") {
                    return None;
                }
                matcher.numbering = Some((index.to_string(), ordered));
            }
        }
        "b" | "i" | "u" | "strike" | "all-caps" | "small-caps" | "comment-reference" => (),
        "highlight" => {
            if t.sym("[") {
                if !t.ident("color") || !t.sym("=") {
                    return None;
                }
                matcher.qualifier = Some(t.string()?);
                if !t.sym("]") {
                    return None;
                }
            }
        }
        "br" => {
            if !t.sym("[") || !t.ident("type") || !t.sym("=") {
                return None;
            }
            let value = t.string()?;
            if !matches!(value.as_str(), "line" | "page" | "column") || !t.sym("]") {
                return None;
            }
            matcher.qualifier = Some(value);
        }
        _ => return None,
    }
    if !t.space() || !t.sym("=>") {
        return None;
    }
    t.space();
    let mut path = Path::default();
    if t.sym("!") {
        path.ignore = true;
    } else if t.tokens.get(t.index)?.kind == Kind::Identifier {
        loop {
            let mut names = vec![t.name()?];
            while t.sym("|") {
                names.push(t.name()?);
            }
            let mut attributes = BTreeMap::<String, String>::new();
            loop {
                if t.sym(".") {
                    let class = t.name()?;
                    attributes
                        .entry("class".into())
                        .and_modify(|s| {
                            if !s.is_empty() {
                                s.push(' ');
                            }
                            s.push_str(&class);
                        })
                        .or_insert(class);
                } else if t.sym("[") {
                    let key = t.name()?;
                    if !t.sym("=") {
                        return None;
                    }
                    let value = t.string()?;
                    if !t.sym("]") {
                        return None;
                    }
                    attributes.insert(key, value);
                } else {
                    break;
                }
            }
            let mut fresh = false;
            let mut separator = None;
            let saved = t.index;
            if t.sym(":") && t.ident("fresh") {
                fresh = true;
            } else {
                t.index = saved;
            }
            let saved = t.index;
            if t.sym(":") && t.ident("separator") {
                if !t.sym("(") {
                    return None;
                }
                separator = Some(t.string()?);
                if !t.sym(")") {
                    return None;
                }
            } else {
                t.index = saved;
            }
            path.elements.push(Html::Element {
                names,
                attributes,
                collapsible: !fresh,
                separator,
                children: vec![],
            });
            let saved = t.index;
            if t.space() && t.sym(">") {
                if !t.space() {
                    return None;
                }
            } else {
                t.index = saved;
                break;
            }
        }
    }
    t.take(Kind::End, None)?;
    Some(Rule { matcher, path })
}
