//! Word character SPRMs and iterative STD inheritance, following styles.py.
use super::{fib::Pair, formatting::sprms};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Character {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub emphasis: bool,
    pub strike: bool,
    pub superscript: bool,
    pub subscript: bool,
    pub hidden: bool,
    pub deleted: bool,
}
impl Character {
    pub fn apply(&self, bytes: &[u8], base: &Self) -> Self {
        let mut s = self.clone();
        for (op, v) in sprms(bytes) {
            let target = match op {
                0x0800 => Some((&mut s.deleted, base.deleted)),
                0x0802 | 0x083c => Some((&mut s.hidden, base.hidden)),
                0x0835 => Some((&mut s.bold, base.bold)),
                0x0836 => Some((&mut s.italic, base.italic)),
                0x0837 => Some((&mut s.strike, base.strike)),
                _ => None,
            };
            if let Some((slot, b)) = target {
                if let Some(value) = match v[0] {
                    0 => Some(false),
                    1 => Some(true),
                    0x80 => Some(b),
                    0x81 => Some(!b),
                    _ => None,
                } {
                    *slot = value;
                }
                continue;
            }
            match op {
                0x2a3e => s.underline = !matches!(v[0], 0 | 5),
                0x2a48 => {
                    s.superscript = v[0] == 1;
                    s.subscript = v[0] == 2
                }
                0x2a53 => s.strike = v[0] != 0,
                0x2a54 => s.emphasis = v[0] != 0,
                _ => {}
            }
        }
        s
    }
    pub fn names(&self) -> Vec<&'static str> {
        let mut out = vec![];
        for (v, name) in [
            (self.bold, "bold"),
            (self.italic, "italic"),
            (self.underline, "underline"),
            (self.emphasis, "emphasis"),
            (self.strike, "strikethrough"),
            (self.superscript, "superscript"),
            (self.subscript && !self.superscript, "subscript"),
        ] {
            if v {
                out.push(name)
            }
        }
        out
    }
}
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Style {
    pub name: String,
    pub character: Character,
    pub heading: Option<u16>,
    pub title: bool,
    pub toc: Option<u16>,
    pub code: bool,
    pub paragraph: Vec<Vec<u8>>,
}
struct Raw {
    sti: u16,
    name: String,
    base: u16,
    chpx: Vec<u8>,
    papx: Vec<u8>,
}
fn u16at(d: &[u8], p: usize) -> Option<u16> {
    Some(u16::from_le_bytes(d.get(p..p + 2)?.try_into().ok()?))
}
fn raw(d: &[u8], base: usize) -> Option<Raw> {
    if d.len() < base.max(10) {
        return None;
    }
    let sti = u16at(d, 0)? & 0xfff;
    let second = u16at(d, 2)?;
    let count = u16at(d, 4)? & 15;
    let p = base.max(10);
    let n = u16at(d, p)? as usize;
    let bytes = d.get(p + 2..p + 2 + n * 2)?;
    let name = String::from_utf16_lossy(
        &bytes
            .chunks_exact(2)
            .map(|v| u16::from_le_bytes([v[0], v[1]]))
            .collect::<Vec<_>>(),
    );
    let mut p = p + 2 + n * 2 + 2;
    let mut upx = vec![];
    for _ in 0..count {
        p += p % 2;
        let Some(n) = u16at(d, p) else { break };
        let Some(v) = d.get(p + 2..p + 2 + n as usize) else {
            break;
        };
        upx.push(v.to_vec());
        p += 2 + n as usize;
    }
    let (papx, chpx) = match second & 15 {
        1 => (
            upx.first().and_then(|v| v.get(2..)).unwrap_or(&[]).to_vec(),
            upx.get(1).cloned().unwrap_or_default(),
        ),
        2 => (vec![], upx.first().cloned().unwrap_or_default()),
        _ => (vec![], vec![]),
    };
    Some(Raw {
        sti,
        name,
        base: second >> 4,
        chpx,
        papx,
    })
}
#[derive(Default)]
pub struct Stylesheet {
    pub styles: BTreeMap<u16, Style>,
    default: Style,
}
impl Stylesheet {
    pub fn get(&self, id: u16) -> &Style {
        self.styles.get(&id).unwrap_or(&self.default)
    }
    pub fn load(table: &[u8], pair: Pair) -> Self {
        let mut out = Self::default();
        let Some(end) = pair.fc.checked_add(pair.lcb) else {
            return out;
        };
        let Some(d) = table.get(pair.fc as usize..end as usize) else {
            return out;
        };
        if d.len() < 8 {
            return out;
        }
        let mut p = 2 + u16at(d, 0).unwrap() as usize;
        let count = u16at(d, 2).unwrap();
        let base = u16at(d, 4).unwrap() as usize;
        let mut records = BTreeMap::new();
        for id in 0..count {
            let Some(n) = u16at(d, p) else { break };
            p += 2;
            if n == 0 {
                continue;
            }
            let Some(v) = d.get(p..p + n as usize) else {
                break;
            };
            p += n as usize;
            if let Some(r) = raw(v, base) {
                records.insert(id, r);
            }
        }
        for &id in records.keys() {
            if out.styles.contains_key(&id) {
                continue;
            }
            let mut chain = vec![];
            let mut seen = BTreeSet::new();
            let mut cursor = id;
            let mut value = Style::default();
            loop {
                if let Some(s) = out.styles.get(&cursor) {
                    value = s.clone();
                    break;
                }
                if !seen.insert(cursor) {
                    break;
                }
                let Some(r) = records.get(&cursor) else { break };
                chain.push(cursor);
                if r.base == 0xfff || r.base == cursor {
                    break;
                }
                cursor = r.base;
            }
            for id in chain.into_iter().rev() {
                let r = &records[&id];
                value.character = value.character.apply(&r.chpx, &value.character);
                if !r.papx.is_empty() {
                    value.paragraph.push(r.papx.clone())
                }
                let name = r
                    .name
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .to_lowercase();
                let numbered = |prefix: &str| {
                    name.strip_prefix(prefix)
                        .and_then(|s| s.trim().parse::<u16>().ok())
                        .filter(|n| (1..=9).contains(n))
                };
                value.heading = (1..=9)
                    .contains(&r.sti)
                    .then_some(r.sti)
                    .or_else(|| numbered("heading"))
                    .or_else(|| numbered("标题"))
                    .or(value.heading);
                value.title |=
                    r.sti == 15 || matches!(name.as_str(), "title" | "标题" | "文档标题");
                value.toc = numbered("toc")
                    .or_else(|| numbered("目录"))
                    .map(|n| n - 1)
                    .or(value.toc);
                value.code |= ["code", "preformatted", "source code", "代码"]
                    .iter()
                    .any(|s| name.contains(s));
                if !r.name.is_empty() {
                    value.name = r.name.clone()
                }
                out.styles.insert(id, value.clone());
            }
        }
        out
    }
}
