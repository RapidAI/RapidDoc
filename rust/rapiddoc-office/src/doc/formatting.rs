//! Word CHPX/PAPX FKP run indexing, preserving raw SPRMs for semantic decoding.
use super::fib::Pair;
#[derive(Debug, Clone)]
pub struct Run {
    pub start: u32,
    pub end: u32,
    pub style_id: u16,
    pub grpprl: Vec<u8>,
}
fn u32at(d: &[u8], p: usize) -> Option<u32> {
    Some(u32::from_le_bytes(d.get(p..p + 4)?.try_into().ok()?))
}
pub fn bte_pages(table: &[u8], pair: Pair) -> Vec<u32> {
    let Some(end) = pair.fc.checked_add(pair.lcb) else {
        return vec![];
    };
    let Some(d) = table.get(pair.fc as usize..end as usize) else {
        return vec![];
    };
    if d.len() < 8 || (d.len() - 4) % 8 != 0 {
        return vec![];
    }
    let count = (d.len() - 4) / 8;
    (0..count)
        .filter_map(|i| u32at(d, (count + 1) * 4 + i * 4).map(|n| n & 0x003f_ffff))
        .collect()
}
pub fn page(d: &[u8], paragraph: bool) -> Vec<Run> {
    if d.len() != 512 {
        return vec![];
    }
    let count = d[511] as usize;
    let base = (count + 1) * 4;
    if count == 0 || base + count * if paragraph { 13 } else { 1 } > 511 {
        return vec![];
    }
    let mut out = vec![];
    for i in 0..count {
        let start = u32at(d, i * 4).unwrap();
        let end = u32at(d, (i + 1) * 4).unwrap();
        if end <= start {
            continue;
        }
        let offset = d[base + i * if paragraph { 13 } else { 1 }] as usize * 2;
        let mut style_id = 0;
        let mut grpprl = vec![];
        if offset != 0 && offset < 511 {
            if paragraph {
                let first = d[offset] as usize;
                let (p, n) = if first != 0 {
                    (offset + 1, first * 2 - 1)
                } else if offset + 1 < 511 {
                    (offset + 2, d[offset + 1] as usize * 2)
                } else {
                    (511, 0)
                };
                let content = &d[p..(p + n).min(511)];
                if content.len() >= 2 {
                    style_id = u16::from_le_bytes([content[0], content[1]]);
                    grpprl = content[2..].to_vec();
                }
            } else {
                let n = d[offset] as usize;
                grpprl = d[offset + 1..(offset + 1 + n).min(512)].to_vec();
            }
        }
        out.push(Run {
            start,
            end,
            style_id,
            grpprl,
        });
    }
    out
}
#[derive(Default)]
pub struct Formatting {
    pub characters: Vec<Run>,
    pub paragraphs: Vec<Run>,
    pub diagnostics: Vec<String>,
}
impl Formatting {
    pub fn load(word: &[u8], table: &[u8], chpx: Pair, papx: Pair) -> Self {
        let mut out = Self::default();
        for (paragraph, pair) in [(false, chpx), (true, papx)] {
            for n in bte_pages(table, pair) {
                let start = n as usize * 512;
                let Some(d) = word.get(start..start + 512) else {
                    out.diagnostics.push(format!(
                        "DOC {}Fkp page is truncated: {n}",
                        if paragraph { "Papx" } else { "Chpx" }
                    ));
                    continue;
                };
                let runs = page(d, paragraph);
                if paragraph {
                    out.paragraphs.extend(runs)
                } else {
                    out.characters.extend(runs)
                }
            }
        }
        out.characters.sort_by_key(|r| r.start);
        out.paragraphs.sort_by_key(|r| r.start);
        out
    }
    pub fn at(runs: &[Run], fc: u32) -> Option<&Run> {
        let i = runs.partition_point(|r| r.start <= fc).checked_sub(1)?;
        runs.get(i).filter(|r| fc < r.end)
    }
}
/// Truncated SPRM tails are recoverable, matching the Python walker.
pub fn sprms(d: &[u8]) -> Vec<(u16, &[u8])> {
    let mut out = vec![];
    let mut p = 0;
    while p + 2 <= d.len() {
        let op = u16::from_le_bytes([d[p], d[p + 1]]);
        p += 2;
        let n = match op >> 13 {
            0 | 1 => 1,
            2 | 4 | 5 => 2,
            3 => 4,
            7 => 3,
            _ => {
                if op == 0xd608 {
                    if p + 2 > d.len() {
                        break;
                    }
                    let n = u16::from_le_bytes([d[p], d[p + 1]]) as usize;
                    if n == 0 {
                        break;
                    }
                    n + 1
                } else {
                    if p == d.len() {
                        break;
                    }
                    d[p] as usize + 1
                }
            }
        };
        if p + n > d.len() {
            break;
        }
        out.push((op, &d[p..p + n]));
        p += n;
    }
    out
}
