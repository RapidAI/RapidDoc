//! Word PlfLst/PlfLfo definitions and numbering state, ported from lists.py.
use super::fib::Pair;
use std::collections::{BTreeMap, BTreeSet};
const N: usize = 9;
#[derive(Clone)]
struct Level {
    marker: u8,
    start: u32,
    restart: Option<u8>,
    tokens: Vec<Token>,
}
impl Default for Level {
    fn default() -> Self {
        Self {
            marker: 23,
            start: 1,
            restart: None,
            tokens: vec![],
        }
    }
}
#[derive(Clone)]
enum Token {
    Level(usize),
    Text(String),
}
#[derive(Clone)]
struct Def {
    lsid: u32,
    levels: Vec<Level>,
    overrides: [Option<u32>; N],
}
#[derive(Default)]
struct Counter {
    values: [u32; N],
    started: [bool; N],
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct Info {
    pub identity: u32,
    pub ordered: bool,
    pub level: u8,
    pub start: u32,
    pub label: Option<String>,
}
#[derive(Default)]
pub struct ListTables {
    defs: BTreeMap<u32, Def>,
    counters: BTreeMap<u32, Counter>,
    used: BTreeSet<(u32, usize)>,
}
fn u16at(d: &[u8], p: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        d.get(p..p.checked_add(2)?)?.try_into().ok()?,
    ))
}
fn u32at(d: &[u8], p: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        d.get(p..p.checked_add(4)?)?.try_into().ok()?,
    ))
}
fn level(d: &[u8], p: usize) -> Option<(Level, usize)> {
    let h = d.get(p..p.checked_add(28)?)?;
    let start = u32at(h, 0)?;
    let marker = h[4];
    let cursor = p + 28 + h[24] as usize + h[25] as usize;
    let n = u16at(d, cursor)? as usize;
    let raw = d.get(cursor + 2..cursor + 2 + n * 2)?;
    let mut tokens = vec![];
    if !matches!(marker, 23 | 255) {
        for (i, v) in raw.chunks_exact(2).enumerate() {
            let unit = u16::from_le_bytes([v[0], v[1]]);
            if unit <= 8 && h[6..15].iter().any(|n| *n != 0 && *n as usize == i + 1) {
                tokens.push(Token::Level(unit as usize));
            } else if unit >= 32 {
                let c = char::from_u32(unit as u32).unwrap_or('\u{fffd}');
                if let Some(Token::Text(s)) = tokens.last_mut() {
                    s.push(c)
                } else {
                    tokens.push(Token::Text(c.to_string()))
                }
            }
        }
    }
    Some((
        Level {
            marker,
            start,
            restart: if h[5] & 8 != 0 { Some(h[26]) } else { None },
            tokens,
        },
        cursor + 2 + n * 2,
    ))
}
impl ListTables {
    pub fn load(table: &[u8], list: Pair, over: Pair) -> Self {
        let mut by_lsid = BTreeMap::new();
        // lcbPlfLst covers descriptors only; LVLs follow in the same Table stream.
        if let Some(d) = table.get(list.fc as usize..) {
            if let Some(n) = u16at(d, 0) {
                if list.lcb as usize >= 2 + n as usize * 28 {
                    let mut desc = vec![];
                    for i in 0..n as usize {
                        let p = 2 + i * 28;
                        let Some(h) = d.get(p..p + 28) else { break };
                        desc.push((u32at(h, 0).unwrap(), h[26] & 1 != 0));
                    }
                    let mut p = list.lcb as usize;
                    'defs: for (id, simple) in desc {
                        let mut levels = vec![];
                        for _ in 0..if simple { 1 } else { N } {
                            let Some((l, end)) = level(d, p) else {
                                break 'defs;
                            };
                            levels.push(l);
                            p = end;
                        }
                        if simple {
                            levels = vec![levels[0].clone(); N]
                        }
                        by_lsid.insert(id, levels);
                    }
                }
            }
        }
        let mut out = Self::default();
        let Some(end) = over.fc.checked_add(over.lcb) else {
            return out;
        };
        let Some(d) = table.get(over.fc as usize..end as usize) else {
            return out;
        };
        let Some(n) = u32at(d, 0) else { return out };
        if n as u64 * 16 + 4 > d.len() as u64 {
            return out;
        }
        let desc: Vec<_> = (0..n as usize)
            .map(|i| {
                let p = 4 + i * 16;
                (u32at(d, p).unwrap(), d[p + 12])
            })
            .collect();
        let mut p = 4 + n as usize * 16;
        for (index, (lsid, count)) in desc.into_iter().enumerate() {
            let mut def = Def {
                lsid,
                levels: by_lsid
                    .get(&lsid)
                    .cloned()
                    .unwrap_or_else(|| vec![Level::default(); N]),
                overrides: [None; N],
            };
            for _ in 0..count {
                let Some(h) = d.get(p..p + 8) else { break };
                let start = u32at(h, 0).unwrap();
                let bits = h[4];
                let idx = (bits & 15) as usize;
                p += 8;
                if bits & 32 != 0 {
                    let Some((l, end)) = level(d, p) else { break };
                    p = end;
                    if idx < N {
                        if bits & 16 != 0 {
                            def.overrides[idx] = Some(l.start)
                        }
                        def.levels[idx] = l;
                    }
                } else if bits & 16 != 0 && idx < N {
                    def.levels[idx].start = start;
                    def.overrides[idx] = Some(start);
                }
            }
            out.defs.insert(index as u32 + 1, def);
        }
        out
    }
    pub fn next(&mut self, id: u32, level: u8) -> Option<Info> {
        if matches!(id, 0 | 0xf801) {
            return None;
        }
        let fallback = Def {
            lsid: u32::MAX ^ id,
            levels: vec![Level::default(); N],
            overrides: [None; N],
        };
        let def = self.defs.get(&id).unwrap_or(&fallback);
        let i = (level as usize).min(8);
        let l = &def.levels[i];
        if l.marker == 255 {
            return None;
        }
        let mut info = Info {
            identity: def.lsid,
            ordered: l.marker != 23,
            level: i as u8,
            start: 1,
            label: None,
        };
        if !info.ordered {
            return Some(info);
        }
        let c = self.counters.entry(def.lsid).or_default();
        let first = self.used.insert((id, i));
        let value = if first && def.overrides[i].is_some() {
            def.overrides[i].unwrap()
        } else if c.started[i] {
            c.values[i].saturating_add(1)
        } else {
            l.start
        };
        c.values[i] = value;
        c.started[i] = true;
        for j in i + 1..N {
            if def.levels[j].restart.is_none_or(|r| i < (r as usize)) {
                c.started[j] = false
            }
        }
        info.start = value;
        info.label = Some(if l.tokens.is_empty() {
            format!("{}.", marker(l.marker, value))
        } else {
            l.tokens
                .iter()
                .map(|t| match t {
                    Token::Text(s) => s.clone(),
                    Token::Level(j) => marker(
                        def.levels[*j].marker,
                        if c.started[*j] {
                            c.values[*j]
                        } else {
                            def.levels[*j].start
                        },
                    ),
                })
                .collect::<String>()
                .trim()
                .to_owned()
        });
        Some(info)
    }
    pub fn info(&mut self, id: Option<u16>, level: Option<u8>) -> Option<(u32, bool, u8, u32)> {
        let v = self.next(id? as u32, level.unwrap_or(0))?;
        Some((v.identity, v.ordered, v.level, v.start))
    }
    pub fn paragraph(&mut self, id: Option<u16>, level: Option<u8>) -> Option<Info> {
        self.next(id? as u32, level.unwrap_or(0))
    }
}
fn marker(kind: u8, n: u32) -> String {
    if matches!(kind, 3 | 4) && n > 0 {
        let mut n = n;
        let mut s = vec![];
        while n > 0 {
            n -= 1;
            s.push((b'A' + (n % 26) as u8) as char);
            n /= 26
        }
        let s: String = s.into_iter().rev().collect();
        return if kind == 4 { s.to_lowercase() } else { s };
    }
    if matches!(kind, 1 | 2) && n > 0 {
        let mut n = n;
        let mut s = String::new();
        for (v, t) in [
            (1000, "M"),
            (900, "CM"),
            (500, "D"),
            (400, "CD"),
            (100, "C"),
            (90, "XC"),
            (50, "L"),
            (40, "XL"),
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I"),
        ] {
            while n >= v {
                s.push_str(t);
                n -= v
            }
        }
        return if kind == 2 { s.to_lowercase() } else { s };
    }
    n.to_string()
}
