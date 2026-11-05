//! StyleTextPropAtom and TextMasterStyleAtom, including tri-state overrides.
use super::records::{i16, u16, u32};
use serde::{Deserialize, Serialize};
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct ParagraphRun {
    pub count: u32,
    pub depth: u16,
    pub bullet: Option<bool>,
    pub ordered: Option<bool>,
    pub start: Option<i64>,
}
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct CharacterRun {
    pub count: u32,
    pub bold: Option<bool>,
    pub italic: Option<bool>,
    pub underline: Option<bool>,
    pub strike: Option<bool>,
    pub baseline: Option<i16>,
    pub pp9rt: u8,
}
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct MasterLevel {
    pub bullet: Option<bool>,
    pub bold: Option<bool>,
    pub italic: Option<bool>,
    pub underline: Option<bool>,
    pub baseline: Option<i16>,
}
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct StyleRuns {
    pub paragraphs: Vec<ParagraphRun>,
    pub characters: Vec<CharacterRun>,
}
pub fn paragraph_exception(b: &[u8], mut p: usize) -> Option<(Option<bool>, usize)> {
    let mask = u32(b, p)?;
    p += 4;
    let mut bullet = None;
    if mask & 15 != 0 {
        let flags = u16(b, p)?;
        if mask & 1 != 0 {
            bullet = Some(flags & 1 != 0);
        }
        p += 2;
    }
    for (mask_bit, size) in [
        (0x80, 2),
        (0x10, 2),
        (0x40, 2),
        (0x20, 4),
        (0x800, 2),
        (0x1000, 2),
        (0x2000, 2),
        (0x4000, 2),
        (0x100, 2),
        (0x400, 2),
        (0x8000, 2),
    ] {
        if mask & mask_bit != 0 {
            p += size;
        }
    }
    if mask & 0x100000 != 0 {
        let count = u16(b, p)? as usize;
        p += 2 + count * 4;
    }
    if mask & 0x10000 != 0 {
        p += 2;
    }
    if mask & 0xe0000 != 0 {
        p += 2;
    }
    if mask & 0x200000 != 0 {
        p += 2;
    }
    (p <= b.len()).then_some((bullet, p))
}
pub fn character_exception(b: &[u8], mut p: usize) -> Option<(CharacterRun, usize)> {
    let mask = u32(b, p)?;
    p += 4;
    let mut style = CharacterRun::default();
    if mask & 0xffff != 0 {
        let flags = u16(b, p)?;
        p += 2;
        for (bit, field) in [
            (1, &mut style.bold),
            (2, &mut style.italic),
            (4, &mut style.underline),
            (0x100, &mut style.strike),
        ] {
            if mask & bit != 0 {
                *field = Some(flags & bit as u16 != 0);
            }
        }
        style.pp9rt = ((flags >> 10) & 15) as u8;
    }
    for bit in [0x10000, 0x200000, 0x400000, 0x800000] {
        if mask & bit != 0 {
            p += 2;
        }
    }
    if mask & 0x20000 != 0 {
        p += 2;
    }
    if mask & 0x40000 != 0 {
        p += 4;
    }
    if mask & 0x80000 != 0 {
        style.baseline = Some(i16(b, p)?);
        p += 2;
    }
    (p <= b.len()).then_some((style, p))
}
pub fn parse(b: &[u8], text_utf16_length: usize) -> StyleRuns {
    let mut runs = StyleRuns::default();
    let (mut p, mut covered) = (0, 0u64);
    while covered <= text_utf16_length as u64 {
        let (Some(count), Some(depth)) = (u32(b, p), u16(b, p + 4)) else {
            break;
        };
        p += 6;
        let Some((bullet, next)) = paragraph_exception(b, p) else {
            return runs;
        };
        p = next;
        runs.paragraphs.push(ParagraphRun {
            count,
            depth: depth.min(8),
            bullet,
            ..Default::default()
        });
        covered += count as u64;
        if count == 0 {
            break;
        }
    }
    covered = 0;
    while covered <= text_utf16_length as u64 {
        let Some(count) = u32(b, p) else {
            break;
        };
        p += 4;
        let Some((mut style, next)) = character_exception(b, p) else {
            break;
        };
        p = next;
        style.count = count;
        runs.characters.push(style);
        covered += count as u64;
        if count == 0 {
            break;
        }
    }
    runs
}
pub fn master(b: &[u8], instance: u16) -> Vec<MasterLevel> {
    let Some(count) = u16(b, 0) else {
        return vec![];
    };
    let mut p = 2;
    let mut out = vec![];
    for _ in 0..count.min(10) {
        if instance >= 5 {
            p += 2;
        }
        let Some((bullet, next)) = paragraph_exception(b, p) else {
            break;
        };
        p = next;
        let Some((style, next)) = character_exception(b, p) else {
            break;
        };
        p = next;
        out.push(MasterLevel {
            bullet,
            bold: style.bold,
            italic: style.italic,
            underline: style.underline,
            baseline: style.baseline,
        });
    }
    out
}
