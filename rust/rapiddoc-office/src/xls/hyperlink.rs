//! BIFF HLink monikers with the existing Office link policy.
use super::{
    records::{u16, u32},
    strings,
};
use std::collections::BTreeMap;
fn unicode(p: &[u8], cursor: &mut usize) -> Option<String> {
    let Some(n) = u32(p, *cursor) else {
        *cursor = p.len();
        return None;
    };
    *cursor += 4;
    let Some(end) = (*cursor)
        .checked_add(n as usize * 2)
        .filter(|e| *e <= p.len())
    else {
        *cursor = p.len();
        return None;
    };
    let text = strings::decode_bytes(&p[*cursor..end], 1200);
    *cursor = end;
    Some(strings::clean(text.trim_end_matches('\0')))
}
pub fn target(p: &[u8]) -> Option<String> {
    if p.len() < 32 {
        return None;
    }
    let version = u32(p, 24)?;
    let flags = u32(p, 28)?;
    if version != 2 {
        return None;
    }
    let mut cursor = 32;
    if flags & 0x10 != 0 {
        unicode(p, &mut cursor);
    }
    if flags & 0x80 != 0 {
        unicode(p, &mut cursor);
    }
    let mut target = None;
    let mut blocked = false;
    if flags & 1 != 0 {
        if flags & 0x100 != 0 {
            target = unicode(p, &mut cursor);
        } else if let Some(guid) = p.get(cursor..cursor + 16) {
            cursor += 16;
            if guid
                == [
                    0xe0, 0xc9, 0xea, 0x79, 0xf9, 0xba, 0xce, 0x11, 0x8c, 0x82, 0x00, 0xaa, 0x00,
                    0x4b, 0xa9, 0x0b,
                ]
            {
                if let Some(n) = u32(p, cursor) {
                    cursor += 4;
                    if let Some(end) = cursor.checked_add(n as usize).filter(|e| *e <= p.len()) {
                        let raw = &p[cursor..end];
                        target = Some(strings::clean(
                            strings::decode_bytes(&raw[..raw.len() / 2 * 2], 1200)
                                .split('\0')
                                .next()
                                .unwrap_or(""),
                        ));
                        cursor = end;
                    } else {
                        cursor = p.len()
                    }
                } else {
                    cursor = p.len()
                }
            } else if guid
                == [
                    0x03, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x46,
                ]
            {
                blocked = true;
            }
        }
    }
    let location = if flags & 8 != 0 {
        unicode(p, &mut cursor)
    } else {
        None
    };
    if let Some(s) = location.filter(|s| !s.is_empty()) {
        target = Some(match target.filter(|s| !s.is_empty()) {
            Some(t) => format!("{t}#{s}"),
            None => format!("#{s}"),
        });
    }
    if blocked {
        return None;
    }
    crate::links::sanitize(target.as_deref()).map(str::to_owned)
}
pub fn apply(p: &[u8], links: &mut BTreeMap<(u32, u32), String>) {
    if p.len() < 8 {
        return;
    }
    let Some(target) = target(p) else { return };
    let (a, b, c, d) = (
        u16(p, 0).unwrap() as u32,
        u16(p, 2).unwrap() as u32,
        u16(p, 4).unwrap() as u32,
        u16(p, 6).unwrap() as u32,
    );
    for row in a.min(b)..=a.max(b).min(65535) {
        for col in c.min(d)..=c.max(d).min(255) {
            links.insert((row, col), target.clone());
        }
    }
}
