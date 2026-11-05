//! OfficeArt geometry, group membership and shape-local PPT text binding.
use super::{
    records::{self, i16, u16, u32, Budget, Record},
    styles::MasterLevel,
    text::{self, Content},
};
use crate::Result;
use serde::Serialize;
use std::collections::BTreeMap;
pub type Bbox = [f64; 4];
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Space {
    pub coord_left: i32,
    pub coord_top: i32,
    pub coord_right: i32,
    pub coord_bottom: i32,
    pub abs_left: f64,
    pub abs_top: f64,
    pub abs_right: f64,
    pub abs_bottom: f64,
}
#[derive(Clone, Debug)]
pub struct Shape<'a> {
    pub key: Vec<usize>,
    pub record: Record<'a>,
    pub space: Option<Space>,
    pub bbox: Bbox,
    pub order: usize,
    pub group_key: Option<Vec<usize>>,
}
pub struct Group<'a> {
    pub key: Vec<usize>,
    pub record: Record<'a>,
    pub members: Vec<usize>,
    pub order: usize,
    pub authoritative: bool,
}
pub struct Collection<'a> {
    pub shapes: Vec<Shape<'a>>,
    pub groups: Vec<Group<'a>>,
}
pub fn properties(r: Record<'_>) -> BTreeMap<u16, u32> {
    (0..r.instance as usize)
        .map_while(|i| Some((u16(r.payload, i * 6)? & 0x3fff, u32(r.payload, i * 6 + 2)?)))
        .collect()
}
pub fn complex(r: Record<'_>) -> BTreeMap<u16, &[u8]> {
    let mut out = BTreeMap::new();
    let mut cursor = r.instance as usize * 6;
    for i in 0..r.instance as usize {
        let (Some(op), Some(size)) = (u16(r.payload, i * 6), u32(r.payload, i * 6 + 2)) else {
            break;
        };
        if op & 0x8000 == 0 {
            continue;
        }
        let Some(end) = cursor
            .checked_add(size as usize)
            .filter(|e| *e <= r.payload.len())
        else {
            break;
        };
        out.insert(op & 0x3fff, &r.payload[cursor..end]);
        cursor = end;
    }
    out
}
pub fn shape_properties(r: Record<'_>, budget: &mut Budget) -> Result<BTreeMap<u16, u32>> {
    let mut out = BTreeMap::new();
    for c in records::children(r, budget)? {
        if matches!(c.kind, 0xf00b | 0xf122) {
            out.extend(properties(c));
        }
    }
    Ok(out)
}
pub fn rect(payload: &[u8], child: bool) -> Option<Bbox> {
    if child {
        Some([
            u32(payload, 0)? as i32 as f64,
            u32(payload, 4)? as i32 as f64,
            u32(payload, 8)? as i32 as f64,
            u32(payload, 12)? as i32 as f64,
        ])
    } else {
        Some([
            i16(payload, 2)? as f64,
            i16(payload, 0)? as f64,
            i16(payload, 4)? as f64,
            i16(payload, 6)? as f64,
        ])
    }
}
pub fn map(b: Bbox, s: Space) -> Bbox {
    let x =
        (s.abs_right - s.abs_left) / ((s.coord_right as i64 - s.coord_left as i64) as f64).max(1.);
    let y =
        (s.abs_bottom - s.abs_top) / ((s.coord_bottom as i64 - s.coord_top as i64) as f64).max(1.);
    [
        s.abs_left + (b[0] - s.coord_left as f64) * x,
        s.abs_top + (b[1] - s.coord_top as f64) * y,
        s.abs_left + (b[2] - s.coord_left as f64) * x,
        s.abs_top + (b[3] - s.coord_top as f64) * y,
    ]
}
pub fn bbox(r: Record<'_>, space: Option<Space>, budget: &mut Budget) -> Result<Option<Bbox>> {
    let children = records::children(r, budget)?;
    for c in children.iter().filter(|c| c.kind == 0xf010) {
        if let Some(b) = rect(c.payload, false) {
            return Ok(Some(b));
        }
    }
    for c in children.iter().filter(|c| c.kind == 0xf00f) {
        if let Some(b) = rect(c.payload, true) {
            return Ok(Some(if let Some(s) = space {
                map(b, s)
            } else if b.iter().any(|n| n.abs() > 100000.) {
                b.map(|n| n * 576. / 914400.)
            } else {
                b
            }));
        }
    }
    Ok(None)
}
pub fn group_space(
    r: Record<'_>,
    parent: Option<Space>,
    budget: &mut Budget,
) -> Result<Option<Space>> {
    let children = records::children(r, budget)?;
    let Some(f) = children
        .iter()
        .find(|r| r.kind == 0xf009 && r.payload.len() >= 16)
    else {
        return Ok(parent);
    };
    let b = rect(f.payload, true).unwrap();
    if b[2] <= b[0] || b[3] <= b[1] {
        return Ok(parent);
    }
    let raw = children
        .iter()
        .find_map(|c| {
            if c.kind == 0xf010 {
                Some(rect(c.payload, false))
            } else if c.kind == 0xf00f {
                Some(rect(c.payload, true))
            } else {
                None
            }
        })
        .flatten();
    let Some(raw) = raw else {
        return Ok(parent);
    };
    let abs = parent.map(|p| map(raw, p)).unwrap_or(raw);
    Ok(Some(Space {
        coord_left: b[0] as i32,
        coord_top: b[1] as i32,
        coord_right: b[2] as i32,
        coord_bottom: b[3] as i32,
        abs_left: abs[0],
        abs_top: abs[1],
        abs_right: abs[2],
        abs_bottom: abs[3],
    }))
}
pub fn collect<'a>(slide: Record<'a>, budget: &mut Budget) -> Result<Collection<'a>> {
    let mut out = Collection {
        shapes: vec![],
        groups: vec![],
    };
    let mut work = vec![(slide, None, vec![], None::<usize>)];
    while let Some((r, space, path, active)) = work.pop() {
        if r.kind == 0xf003 && r.version == 15 {
            let children = records::children(r, budget)?;
            let Some(&first) = children.first() else {
                continue;
            };
            let mut key = path.clone();
            key.push(0);
            let nested = group_space(first, space, budget)?;
            let authoritative = shape_properties(first, budget)?.get(&0x39f).unwrap_or(&0) & 1 != 0;
            let group = out.groups.len();
            out.groups.push(Group {
                key,
                record: first,
                members: vec![],
                order: out.shapes.len(),
                authoritative,
            });
            for (i, c) in children.into_iter().enumerate().skip(1).rev() {
                let mut p = path.clone();
                p.push(i);
                work.push((c, nested, p, Some(group)));
            }
            continue;
        }
        if r.kind == 0xf004 {
            let Some(b) = bbox(r, space, budget)?.filter(|b| b[2] > b[0] && b[3] > b[1]) else {
                continue;
            };
            let index = out.shapes.len();
            out.shapes.push(Shape {
                key: path,
                record: r,
                space,
                bbox: b,
                order: index,
                group_key: active.map(|i| out.groups[i].key.clone()),
            });
            if let Some(i) = active {
                out.groups[i].members.push(index);
            }
            continue;
        }
        if r.version == 15 {
            for (i, c) in records::children(r, budget)?.into_iter().enumerate().rev() {
                let mut p = path.clone();
                p.push(i);
                work.push((c, space, p, active));
            }
        }
    }
    Ok(out)
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Numbering {
    pub enabled: Option<bool>,
    pub start: Option<u16>,
}
pub fn numbering(payload: &[u8]) -> BTreeMap<u8, Numbering> {
    let mut out = BTreeMap::new();
    let (mut p, mut slot) = (0, 0u8);
    while p + 12 <= payload.len() && slot < 16 {
        let Some(mask) = u32(payload, p).filter(|m| m & !0x03800000 == 0) else {
            break;
        };
        p += 4;
        if mask & 0x800000 != 0 {
            p += 2;
        }
        let enabled = if mask & 0x2000000 != 0 {
            let Some(v) = i16(payload, p) else {
                break;
            };
            p += 2;
            Some(v == 1)
        } else {
            None
        };
        let start = if mask & 0x1000000 != 0 {
            let (Some(_), Some(v)) = (u16(payload, p), u16(payload, p + 2)) else {
                break;
            };
            p += 4;
            Some(v)
        } else {
            None
        };
        if u32(payload, p) != Some(0) {
            break;
        }
        p += 4;
        let Some(mask) = u32(payload, p) else {
            break;
        };
        p += 4;
        for (bit, size) in [(0, 2), (1, 2), (2, 2), (5, 4), (6, 2)] {
            if mask & (1 << bit) != 0 {
                p += size;
            }
        }
        if mask & (1 << 9) != 0 || p > payload.len() {
            break;
        }
        out.insert(slot, Numbering { enabled, start });
        slot += 1;
    }
    out
}
pub fn shape_numbering(shape: Record<'_>, budget: &mut Budget) -> Result<BTreeMap<u8, Numbering>> {
    // Visit one descendant at a time; source returns on the first valid PPT9
    // tag, so later records must not be charged or rejected prematurely.
    let mut stack = vec![(shape.payload, 0usize)];
    if shape.version != 15 {
        return Ok(BTreeMap::new());
    }
    while !stack.is_empty() {
        if stack.len() > 64 {
            return Err(crate::Error::new(
                crate::ErrorKind::ResourceLimit,
                "record nesting exceeds original max_record_depth=64",
            ));
        }
        let (data, cursor) = stack.last_mut().unwrap();
        let Some(r) = records::at(data, *cursor, None, false, budget)? else {
            stack.pop();
            continue;
        };
        *cursor += 8 + r.payload.len();
        if r.kind == 0x138a && r.version == 15 {
            let children =
                records::iter(r.payload, 0, None, false, budget).collect::<Result<Vec<_>>>()?;
            if children
                .iter()
                .any(|c| c.kind == 0xfba && records::utf16(c.payload) == "___PPT9")
            {
                if let Some(blob) = children.iter().find(|c| c.kind == 0x138b) {
                    if let Some(atom) = records::at(blob.payload, 0, None, false, budget)?
                        .filter(|r| r.kind == 0xfac)
                    {
                        return Ok(numbering(atom.payload));
                    }
                }
            }
        }
        if r.version == 15 && !(1053..=1064).contains(&r.kind) {
            stack.push((r.payload, 0));
        }
    }
    Ok(BTreeMap::new())
}
pub fn apply_numbering(
    mut content: Content,
    shape: Record<'_>,
    budget: &mut Budget,
) -> Result<Content> {
    let numbering = shape_numbering(shape, budget)?;
    for p in &mut content.paragraphs {
        if let Some(style) = numbering.get(&p.pp9rt) {
            if let Some(enabled) = style.enabled {
                p.list_kind = enabled.then(|| "ordered".into());
                p.start = if enabled {
                    style.start.map(i64::from)
                } else {
                    None
                };
            }
        }
    }
    Ok(content)
}
pub fn shape_text(
    shape: &Shape<'_>,
    external: &[Content],
    masters: &BTreeMap<u16, Vec<MasterLevel>>,
    links: &BTreeMap<u32, String>,
    budget: &mut Budget,
) -> Result<Option<Content>> {
    let Some(textbox) = records::children(shape.record, budget)?
        .into_iter()
        .find(|r| r.kind == 0xf00d)
    else {
        return Ok(None);
    };
    let records = records::descendants(textbox, budget).collect::<Result<Vec<_>>>()?;
    let contents = text::parse(&records, masters, links, budget, text::is_control)?;
    if let Some(content) = contents.into_iter().next() {
        return apply_numbering(content, shape.record, budget).map(Some);
    }
    let Some(reference) = records
        .iter()
        .find(|r| r.kind == 0xf9e && r.payload.len() >= 4)
    else {
        return Ok(None);
    };
    let index = u32(reference.payload, 0).unwrap() as usize;
    for index in std::iter::once(index).chain(index.checked_sub(1)) {
        if let Some(content) = external.get(index) {
            return apply_numbering(content.clone(), shape.record, budget).map(Some);
        }
    }
    Ok(None)
}

pub fn object_id(shape: &Shape<'_>, budget: &mut Budget) -> Result<Option<u32>> {
    for child in records::children(shape.record, budget)? {
        if child.kind != 0xf011 {
            continue;
        }
        let candidates = if child.version == 15 {
            records::descendants(child, budget).collect::<Result<Vec<_>>>()?
        } else {
            records::iter(child.payload, 0, None, false, budget).collect::<Result<Vec<_>>>()?
        };
        for r in candidates {
            if r.kind == 0x0bc1 {
                return Ok(u32(r.payload, 0).filter(|n| *n != 0));
            }
        }
    }
    Ok(None)
}
