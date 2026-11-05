//! Strict merge HTML and a separate BeautifulSoup-compatible normalization mode.
use super::*;
use std::{collections::BTreeMap, sync::OnceLock};

#[derive(Clone, Default)]
pub(crate) struct Node {
    pub tag: String,
    pub attrs: BTreeMap<String, String>,
    pub children: Vec<usize>,
    pub parent: Option<usize>,
    pub text: String,
}
#[derive(Clone)]
pub(crate) struct Dom {
    pub nodes: Vec<Node>,
}
fn void(tag: &str) -> bool {
    matches!(
        tag,
        "area"
            | "base"
            | "br"
            | "col"
            | "embed"
            | "hr"
            | "img"
            | "input"
            | "link"
            | "meta"
            | "param"
            | "source"
            | "track"
            | "wbr"
    )
}
fn entities(s: &str) -> Result<String, Error> {
    static ENTITIES: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    let map = ENTITIES.get_or_init(|| serde_json::from_str(include_str!("entities.json")).unwrap());
    let mut result = String::new();
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        result.push_str(&rest[..i]);
        rest = &rest[i + 1..];
        let Some(end) = rest.find(';').filter(|n| *n <= 32) else {
            return Err(unsupported("表格实体必须以分号闭合"));
        };
        let key = &rest[..=end];
        if let Some(number) = key.strip_prefix('#') {
            let number = &number[..number.len() - 1];
            let code = if let Some(hex) = number
                .strip_prefix('x')
                .or_else(|| number.strip_prefix('X'))
            {
                u32::from_str_radix(hex, 16)
            } else {
                number.parse::<u32>()
            };
            let n = code.map_err(|_| unsupported("无效 HTML 数值实体"))?;
            // 控制字符替换/非法 Unicode 的 HTML 修复暂不声称兼容。
            if n == 0
                || n < 32 && !matches!(n, 9 | 10 | 13)
                || (0x7f..=0x9f).contains(&n)
                || (0xfdd0..=0xfdef).contains(&n)
                || n & 0xffff >= 0xfffe
            {
                return Err(unsupported("表格数值实体需要 HTML 容错修复"));
            }
            result.push(char::from_u32(n).ok_or_else(|| unsupported("非法 Unicode 实体"))?);
        } else {
            result.push_str(map.get(key).ok_or_else(|| unsupported("未知 HTML 实体"))?);
        }
        rest = &rest[end + 1..];
    }
    result.push_str(rest);
    Ok(result)
}
fn escaped(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
impl Dom {
    pub fn copy_within(&mut self, id: usize, parent: usize) -> usize {
        let mut node = self.nodes[id].clone();
        let children = std::mem::take(&mut node.children);
        let copy = self.push(parent, node);
        for child in children {
            self.copy_within(child, copy);
        }
        copy
    }
    pub fn parse(html: &str, work: &mut Work) -> Result<Self, Error> {
        Self::parse_mode(html, work, false)
    }
    pub(crate) fn parse_normalization(html: &str, work: &mut Work) -> Result<Self, Error> {
        Self::parse_mode(html, work, true)
    }
    fn parse_mode(html: &str, work: &mut Work, bs: bool) -> Result<Self, Error> {
        work.charge(html.len())?;
        if html.len() > rapiddoc_core::resources::limit(2_000_000) {
            return Err(limit());
        }
        let mut dom = Self {
            nodes: vec![Node::default()],
        };
        let mut stack = vec![0usize];
        let mut rest = html;
        let mut tables = 0;
        while !rest.is_empty() {
            let (text, tail) = rest.split_once('<').unwrap_or((rest, ""));
            if !text.is_empty() {
                let mut text = if bs {
                    super::super::legacy_model::table_normalize::decode(text, false)?
                } else {
                    entities(text)?
                };
                // BeautifulSoup 将仅含 ASCII HTML 空白的文本节点压成一个换行或空格。
                if (!bs
                    || !stack
                        .iter()
                        .any(|i| matches!(dom.nodes[*i].tag.as_str(), "pre" | "textarea")))
                    && text
                        .chars()
                        .all(|c| matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{000c}'))
                {
                    text = if text.contains('\n') {
                        "\n".into()
                    } else {
                        " ".into()
                    };
                }
                dom.push(
                    *stack.last().unwrap(),
                    Node {
                        text,
                        ..Node::default()
                    },
                );
            }
            if tail.is_empty() {
                if rest.ends_with('<') {
                    if bs {
                        dom.push(
                            *stack.last().unwrap(),
                            Node {
                                text: "<".into(),
                                ..Node::default()
                            },
                        );
                    } else {
                        return Err(unsupported("HTML 标签未闭合"));
                    }
                }
                break;
            }
            // HTMLParser treats '<' not followed by a tag opener as text.
            // Reprocess the suffix so a later real end tag is not consumed.
            if bs
                && !tail.as_bytes()[0].is_ascii_alphabetic()
                && !matches!(tail.as_bytes()[0], b'/' | b'!' | b'?')
            {
                dom.push(
                    *stack.last().unwrap(),
                    Node {
                        text: "<".into(),
                        ..Node::default()
                    },
                );
                rest = tail;
                continue;
            }
            if let Some(comment) = tail.strip_prefix("!--") {
                let (text, after) = comment
                    .split_once("-->")
                    .ok_or_else(|| unsupported("HTML 注释未闭合"))?;
                dom.push(
                    *stack.last().unwrap(),
                    Node {
                        tag: "!comment".into(),
                        text: text.into(),
                        ..Node::default()
                    },
                );
                rest = after;
                continue;
            }
            let mut quote = None;
            let mut end = None;
            for (i, c) in tail.char_indices() {
                match c {
                    '\'' | '"' if quote == Some(c) => quote = None,
                    '\'' | '"' if quote.is_none() => quote = Some(c),
                    '>' if quote.is_none() => {
                        end = Some(i);
                        break;
                    }
                    _ => {}
                }
            }
            let Some(end) = end else {
                if bs {
                    // BeautifulSoup's HTMLParser builder does not flush an
                    // unfinished start-tag buffer when closing the document.
                    break;
                }
                return Err(unsupported("HTML 标签未闭合"));
            };
            let raw = tail[..end].trim();
            rest = &tail[end + 1..];
            if bs && raw.to_ascii_lowercase().starts_with("!doctype") {
                dom.push(
                    *stack.last().unwrap(),
                    Node {
                        tag: "!doctype".into(),
                        text: raw[8..].trim_start().into(),
                        ..Node::default()
                    },
                );
                continue;
            }
            if bs && raw.starts_with('?') {
                dom.push(
                    *stack.last().unwrap(),
                    Node {
                        tag: "!pi".into(),
                        text: raw[1..].into(),
                        ..Node::default()
                    },
                );
                continue;
            }
            if bs && raw.starts_with("![CDATA[") && raw.ends_with("]]") {
                dom.push(
                    *stack.last().unwrap(),
                    Node {
                        tag: "!cdata".into(),
                        text: raw[8..raw.len() - 2].into(),
                        ..Node::default()
                    },
                );
                continue;
            }
            if let Some(name) = raw.strip_prefix('/') {
                let name = name.trim().to_ascii_lowercase();
                if bs {
                    let name = name
                        .split(|c: char| {
                            !c.is_ascii_alphanumeric() && !matches!(c, '-' | '.' | ':' | '_')
                        })
                        .next()
                        .unwrap_or("");
                    if let Some(index) = stack.iter().rposition(|i| dom.nodes[*i].tag == name) {
                        if index > 0 {
                            stack.truncate(index);
                        }
                    }
                    continue;
                }
                if stack.len() == 1 || dom.nodes[*stack.last().unwrap()].tag != name {
                    return Err(unsupported("表格 HTML 必须显式正确闭合"));
                }
                stack.pop();
                continue;
            }
            let bare_slash = bs
                && raw.ends_with('/')
                && raw.rsplit_once('=').is_some_and(|(_, v)| {
                    let v = v.trim_start();
                    !v.starts_with(['\'', '"']) && !v.chars().any(char::is_whitespace)
                })
                && !raw[..raw.len().saturating_sub(1)].ends_with(['\'', '"']);
            let closing = raw.ends_with('/') && !bare_slash;
            let raw = if closing {
                raw[..raw.len() - 1].trim_end()
            } else {
                raw
            };
            let end = if bs {
                raw.find(|c: char| matches!(c, '\t' | '\n' | '\r' | '\u{000c}' | ' ' | '/' | '>'))
                    .unwrap_or(raw.len())
            } else {
                raw.find(char::is_whitespace).unwrap_or(raw.len())
            };
            let tag = raw[..end].to_ascii_lowercase();
            if tag.is_empty()
                || !tag.as_bytes()[0].is_ascii_alphabetic()
                || !bs
                    && !tag
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b':')
                || !bs && matches!(tag.as_str(), "script" | "style" | "template")
            {
                return Err(unsupported("表格含未支持的 HTML 标签"));
            }
            if tag == "table" {
                tables += 1;
                if tables > 1 && !bs {
                    return Err(unsupported("嵌套或多表 HTML 尚未移植"));
                }
            }
            if matches!(tag.as_str(), "td" | "th")
                && !bs
                && stack
                    .iter()
                    .any(|&i| matches!(dom.nodes[i].tag.as_str(), "td" | "th"))
            {
                return Err(unsupported("嵌套单元格需要 HTML 修复"));
            }
            let mut attrs = BTreeMap::new();
            let mut a = raw[end..].trim();
            while !a.is_empty() {
                if bs {
                    a = a.trim_start_matches('/').trim_start();
                    if a.is_empty() {
                        break;
                    }
                }
                let end = a
                    .find(|c: char| c.is_whitespace() || c == '=')
                    .unwrap_or(a.len());
                if end == 0 {
                    return Err(unsupported("HTML 属性无效"));
                }
                let key = a[..end].to_ascii_lowercase();
                if !bs
                    && !key.bytes().all(|b| {
                        b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b':' | b'.')
                    })
                {
                    return Err(unsupported("HTML 属性名需要容错修复"));
                }
                a = a[end..].trim_start();
                let mut value = String::new();
                if let Some(after) = a.strip_prefix('=') {
                    a = after.trim_start();
                    if let Some(q) = a.chars().next().filter(|c| *c == '\'' || *c == '"') {
                        let end = a[1..]
                            .find(q)
                            .ok_or_else(|| unsupported("HTML 属性引号未闭合"))?
                            + 1;
                        value = if bs {
                            super::super::legacy_model::table_normalize::decode(&a[1..end], true)?
                        } else {
                            entities(&a[1..end])?
                        };
                        a = &a[end + 1..];
                    } else {
                        let end = a.find(char::is_whitespace).unwrap_or(a.len());
                        value = if bs {
                            super::super::legacy_model::table_normalize::decode(&a[..end], true)?
                        } else {
                            entities(&a[..end])?
                        };
                        a = &a[end..];
                    }
                }
                if bs && super::super::legacy_model::table_normalize::list_attribute(&tag, &key) {
                    value = value
                        .split(python_space)
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                        .join(" ");
                } else if !bs && matches!(key.as_str(), "class" | "rel" | "headers") {
                    value = value.split_whitespace().collect::<Vec<_>>().join(" ");
                }
                attrs.insert(key, value);
                a = a.trim_start();
            }
            let id = dom.push(
                *stack.last().unwrap(),
                Node {
                    tag: tag.clone(),
                    attrs,
                    ..Node::default()
                },
            );
            if bs && matches!(tag.as_str(), "script" | "style") && !closing {
                let close = format!("</{tag}>");
                let lower = rest.to_ascii_lowercase();
                let (text, after) = if let Some(i) = lower.find(&close) {
                    (&rest[..i], &rest[i + close.len()..])
                } else {
                    (rest, "")
                };
                if !text.is_empty() {
                    dom.push(
                        id,
                        Node {
                            tag: "!raw".into(),
                            text: text.into(),
                            ..Node::default()
                        },
                    );
                }
                rest = after;
                continue;
            }
            if !(void(&tag) || bs && super::super::legacy_model::table_normalize::is_void(&tag)) {
                if closing && !bs {
                    return Err(unsupported("非 void 标签必须显式闭合"));
                }
                if !closing {
                    stack.push(id);
                }
            }
            if dom.nodes.len() > rapiddoc_core::resources::limit(20_000)
                || stack.len() > rapiddoc_core::resources::limit(64)
            {
                return Err(limit());
            }
        }
        if stack.len() != 1 && !bs {
            return Err(unsupported("表格 HTML 未完整闭合"));
        }
        if dom.nodes.len() > rapiddoc_core::resources::limit(20_000) {
            return Err(limit());
        }
        Ok(dom)
    }
    fn push(&mut self, parent: usize, mut node: Node) -> usize {
        let id = self.nodes.len();
        node.parent = Some(parent);
        self.nodes.push(node);
        self.nodes[parent].children.push(id);
        id
    }
    pub fn find(&self, root: usize, tags: &[&str]) -> Vec<usize> {
        let mut out = Vec::new();
        for &id in &self.nodes[root].children {
            if tags.contains(&self.nodes[id].tag.as_str()) {
                out.push(id);
            }
            out.extend(self.find(id, tags));
        }
        out
    }
    pub fn text(&self, id: usize) -> String {
        let n = &self.nodes[id];
        if n.tag.is_empty() {
            return n.text.clone();
        }
        if n.tag == "!comment" {
            return String::new();
        }
        n.children.iter().map(|&id| self.text(id)).collect()
    }
    pub fn semantic(&self, id: usize) -> bool {
        !self.text(id).trim_matches(python_space).is_empty()
            || !self
                .find(
                    id,
                    &[
                        "img", "svg", "math", "eq", "table", "figure", "object", "embed", "canvas",
                    ],
                )
                .is_empty()
    }
    pub fn serialize(&self, id: usize) -> String {
        self.serialize_mode(id, false)
    }
    fn serialize_mode(&self, id: usize, bs: bool) -> String {
        let n = &self.nodes[id];
        if n.tag.is_empty() {
            return escaped(&n.text);
        }
        if n.tag == "!comment" {
            return format!("<!--{}-->", n.text);
        }
        if bs && n.tag == "!doctype" {
            return format!("<!DOCTYPE {}>\n", n.text);
        }
        if bs && n.tag == "!pi" {
            return format!("<?{}>", n.text);
        }
        if bs && n.tag == "!cdata" {
            return format!("<![CDATA[{}]]>", n.text);
        }
        if bs && n.tag == "!raw" {
            return n.text.clone();
        }
        let mut s = format!("<{}", n.tag);
        for (key, value) in &n.attrs {
            let value = escaped(value);
            let (quote, value) = if value.contains('"') && !value.contains('\'') {
                ('\'', value)
            } else {
                ('"', value.replace('"', "&quot;"))
            };
            s.push_str(&format!(" {key}={quote}{value}{quote}"));
        }
        if void(&n.tag) || bs && super::super::legacy_model::table_normalize::is_void(&n.tag) {
            s.push_str("/>");
            return s;
        }
        s.push('>');
        for &c in &n.children {
            s.push_str(&self.serialize_mode(c, bs));
        }
        s.push_str(&format!("</{}>", n.tag));
        s
    }
    pub fn all_html(&self) -> String {
        self.nodes[0]
            .children
            .iter()
            .map(|&i| self.serialize(i))
            .collect()
    }
    pub(crate) fn normalized_html(&self) -> String {
        self.nodes[0]
            .children
            .iter()
            .map(|i| self.serialize_mode(*i, true))
            .collect()
    }
    pub fn detach(&mut self, id: usize) {
        if let Some(p) = self.nodes[id].parent.take() {
            self.nodes[p].children.retain(|&c| c != id);
        }
    }
    pub fn copy_from(&mut self, other: &Self, id: usize, parent: usize) -> usize {
        let mut node = other.nodes[id].clone();
        node.children.clear();
        let new = self.push(parent, node);
        for &c in &other.nodes[id].children {
            self.copy_from(other, c, new);
        }
        new
    }
}
