//! Read-only compatibility for acyclic CFB directories not sorted in CFB order.
//! Only sibling links/colors change in a temporary copy; streams/FAT/names do not.
use std::{cmp::Ordering, collections::HashSet, sync::Arc};

const NONE: u32 = 0xffff_ffff;
const END: u32 = 0xffff_fffe;
fn word(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?,
    ))
}
fn sector(bytes: &[u8], size: usize, id: u32) -> Option<usize> {
    let start = (id as usize).checked_add(1)?.checked_mul(size)?;
    bytes.get(start..start.checked_add(size)?)?;
    Some(start)
}
struct Entry {
    offset: usize,
    name: String,
    kind: u8,
    left: u32,
    right: u32,
    child: u32,
}
fn compare(a: &str, b: &str) -> Ordering {
    // Match cfb 0.15.0's documented simple-uppercase order. The exceptional
    // uppercase data is from that MIT-licensed crate (licenses/CFB.txt).
    const SPECIAL: &[(char, char)] = &include!("cfb_uppercase.rs");
    let upper = |c: char| {
        SPECIAL
            .iter()
            .find(|(key, _)| *key == c)
            .map(|(_, v)| *v)
            .unwrap_or_else(|| c.to_uppercase().next().unwrap_or(c))
    };
    a.encode_utf16()
        .count()
        .cmp(&b.encode_utf16().count())
        .then_with(|| a.chars().map(upper).cmp(b.chars().map(upper)))
}
fn store(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn tree(ids: &[u32], entries: &[Entry], bytes: &mut [u8], depth: usize, red: usize) -> u32 {
    if ids.is_empty() {
        return NONE;
    }
    let middle = ids.len() / 2;
    let id = ids[middle];
    let offset = entries[id as usize].offset;
    let left = tree(&ids[..middle], entries, bytes, depth + 1, red);
    let right = tree(&ids[middle + 1..], entries, bytes, depth + 1, red);
    store(bytes, offset + 68, left);
    store(bytes, offset + 72, right);
    bytes[offset + 67] = u8::from(depth != red);
    id
}

/// Return no repair for cycles, aliased trees, bad chains or duplicate names.
/// Reopening the result through cfb still performs its normal container checks.
pub(super) fn normalize(bytes: &[u8]) -> Option<Arc<[u8]>> {
    if bytes.get(..8)? != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" {
        return None;
    }
    let shift = u16::from_le_bytes(bytes.get(30..32)?.try_into().ok()?);
    if !matches!(shift, 9 | 12) {
        return None;
    }
    let size = 1usize << shift;
    let sectors = bytes.len().checked_div(size)?.checked_sub(1)?;
    let fat_count = word(bytes, 44)? as usize;
    if fat_count > sectors {
        return None;
    }
    let mut fat_ids: Vec<_> = (0..109)
        .filter_map(|i| word(bytes, 76 + 4 * i))
        .filter(|id| *id != NONE)
        .collect();
    let mut difat = word(bytes, 68)?;
    let difat_count = word(bytes, 72)? as usize;
    if difat_count > sectors {
        return None;
    }
    let mut visited = HashSet::new();
    for _ in 0..difat_count {
        if !visited.insert(difat) {
            return None;
        }
        let start = sector(bytes, size, difat)?;
        for i in 0..size / 4 - 1 {
            let id = word(bytes, start + 4 * i)?;
            if id != NONE {
                fat_ids.push(id);
            }
        }
        difat = word(bytes, start + size - 4)?;
    }
    if fat_ids.len() != fat_count {
        return None;
    }
    let next = |id: u32| -> Option<u32> {
        let index = id as usize;
        let fat = *fat_ids.get(index / (size / 4))?;
        word(bytes, sector(bytes, size, fat)? + 4 * (index % (size / 4)))
    };
    let mut directory = word(bytes, 48)?;
    let mut offsets = Vec::new();
    visited.clear();
    while directory != END {
        if !visited.insert(directory) || visited.len() > sectors {
            return None;
        }
        let start = sector(bytes, size, directory)?;
        offsets.extend((0..size / 128).map(|i| start + 128 * i));
        directory = next(directory)?;
    }
    let mut entries = Vec::new();
    for offset in offsets {
        let kind = *bytes.get(offset + 66)?;
        let length =
            u16::from_le_bytes(bytes.get(offset + 64..offset + 66)?.try_into().ok()?) as usize;
        let name = if kind == 0 {
            String::new()
        } else {
            if !(2..=64).contains(&length) || length % 2 != 0 {
                return None;
            }
            let units = bytes
                .get(offset..offset + length - 2)?
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect::<Vec<_>>();
            String::from_utf16(&units).ok()?
        };
        entries.push(Entry {
            offset,
            name,
            kind,
            left: word(bytes, offset + 68)?,
            right: word(bytes, offset + 72)?,
            child: word(bytes, offset + 76)?,
        });
    }
    let root = entries.first()?;
    if root.kind != 5 || root.left != NONE || root.right != NONE {
        return None;
    }
    let mut pending = vec![0u32];
    let mut used = HashSet::from([0u32]);
    let mut groups = Vec::new();
    let mut changed = false;
    while let Some(storage) = pending.pop() {
        let mut stack = vec![entries[storage as usize].child];
        let mut siblings = Vec::new();
        while let Some(id) = stack.pop() {
            if id == NONE {
                continue;
            }
            if !used.insert(id) {
                return None;
            }
            let entry = entries.get(id as usize)?;
            match entry.kind {
                1 => pending.push(id),
                2 if entry.child == NONE => {}
                _ => return None,
            }
            if entry.left != NONE {
                let left = entries.get(entry.left as usize)?;
                changed |= compare(&left.name, &entry.name) != Ordering::Less;
            }
            if entry.right != NONE {
                let right = entries.get(entry.right as usize)?;
                changed |= compare(&right.name, &entry.name) != Ordering::Greater;
            }
            siblings.push(id);
            stack.push(entry.left);
            stack.push(entry.right);
        }
        siblings.sort_by(|a, b| compare(&entries[*a as usize].name, &entries[*b as usize].name));
        if siblings.windows(2).any(|pair| {
            compare(
                &entries[pair[0] as usize].name,
                &entries[pair[1] as usize].name,
            ) == Ordering::Equal
        }) {
            return None;
        }
        groups.push((storage, siblings));
    }
    if !changed {
        return None;
    }
    let mut output = bytes.to_vec();
    for (storage, siblings) in groups {
        let red = (usize::BITS - (siblings.len() + 1).leading_zeros() - 1) as usize;
        let child = tree(&siblings, &entries, &mut output, 0, red);
        store(&mut output, entries[storage as usize].offset + 76, child);
    }
    Some(output.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Read, Write};
    fn unordered() -> (Vec<u8>, usize) {
        let mut file = cfb::CompoundFile::create(Cursor::new(Vec::new())).unwrap();
        file.create_stream("/Longer")
            .unwrap()
            .write_all(b"long-stream")
            .unwrap();
        file.create_stream("/A")
            .unwrap()
            .write_all(b"short-stream")
            .unwrap();
        let mut bytes = file.into_inner().into_inner();
        let size = 1usize << u16::from_le_bytes(bytes[30..32].try_into().unwrap());
        let start = sector(&bytes, size, word(&bytes, 48).unwrap()).unwrap();
        store(&mut bytes, start + 76, 1);
        store(&mut bytes, start + 128 + 68, NONE);
        store(&mut bytes, start + 128 + 72, 2);
        store(&mut bytes, start + 256 + 68, NONE);
        store(&mut bytes, start + 256 + 72, NONE);
        (bytes, start)
    }
    #[test]
    fn repair_only_reorders_directory_and_preserves_streams() {
        let (bytes, start) = unordered();
        assert!(cfb::CompoundFile::open(Cursor::new(&bytes)).is_err());
        let fixed = normalize(&bytes).unwrap();
        for (index, (a, b)) in bytes.iter().zip(fixed.iter()).enumerate() {
            if a != b {
                assert!(index >= start && index < start + 384);
                assert!((67..80).contains(&((index - start) % 128)));
            }
        }
        let mut file = cfb::CompoundFile::open(Cursor::new(fixed)).unwrap();
        for (path, expected) in [("/Longer", "long-stream"), ("/A", "short-stream")] {
            let mut actual = String::new();
            file.open_stream(path)
                .unwrap()
                .read_to_string(&mut actual)
                .unwrap();
            assert_eq!(actual, expected);
        }
    }
    #[test]
    fn cycles_aliases_and_duplicate_names_are_not_repaired() {
        let (mut bytes, start) = unordered();
        store(&mut bytes, start + 256 + 72, 1);
        assert!(normalize(&bytes).is_none());
        let (mut bytes, start) = unordered();
        store(&mut bytes, start + 128 + 68, 2);
        assert!(normalize(&bytes).is_none());
        let (mut bytes, start) = unordered();
        let name = bytes[start + 128..start + 128 + 66].to_vec();
        bytes[start + 256..start + 256 + 66].copy_from_slice(&name);
        assert!(normalize(&bytes).is_none());
    }
}
