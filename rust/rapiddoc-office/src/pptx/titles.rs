//! Slide-local title promotion. Keep metadata out of shared Office output.
use crate::docx::body::{Block, Kind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Center,
    Title,
    Subtitle,
}
pub struct Entry {
    pub block: Block,
    pub font_size_pt: Option<f64>,
    pub all_bold: bool,
    pub candidate: bool,
    pub role: Option<Role>,
}
fn plain(b: &Entry) -> bool {
    matches!(b.block.kind, Kind::Text) && !b.candidate
}
pub fn finalize(blocks: &mut [Entry], first_visible_slide: bool) {
    let mut counts: Vec<(f64, usize)> = vec![];
    for b in blocks.iter().filter(|b| plain(b) && !b.all_bold) {
        if let Some(size) = b.font_size_pt {
            if let Some((_, n)) = counts.iter_mut().find(|(s, _)| *s == size) {
                *n += 1;
            } else {
                counts.push((size, 1));
            }
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.total_cmp(&b.0)));
    let body = counts.first().map(|(s, _)| *s);
    let mut bold_sizes = blocks
        .iter()
        .filter(|b| plain(b) && b.all_bold)
        .filter_map(|b| b.font_size_pt)
        .collect::<Vec<_>>();
    bold_sizes.sort_by(|a, b| b.total_cmp(a));
    bold_sizes.dedup();
    if let Some(&largest) = bold_sizes.first() {
        let candidates = blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| plain(b) && b.all_bold && b.font_size_pt == Some(largest))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if candidates.len() == 1
            && body.is_none_or(|body| largest >= body + 4.)
            && bold_sizes
                .get(1)
                .is_none_or(|second| largest >= second + 2.)
        {
            let b = &mut blocks[candidates[0]];
            b.candidate = true;
            b.block.level = Some(2);
        }
    }
    if let Some(body) = body {
        let level2 = blocks
            .iter()
            .filter(|b| b.candidate && b.block.level == Some(2))
            .filter_map(|b| b.font_size_pt)
            .max_by(f64::total_cmp);
        if let Some(level2) = level2 {
            let level3 = blocks
                .iter()
                .filter(|b| plain(b) && b.all_bold)
                .filter_map(|b| b.font_size_pt)
                .filter(|s| *s < level2)
                .max_by(f64::total_cmp);
            if let Some(level3) = level3.filter(|s| *s >= body + 2. && level2 >= *s + 2.) {
                for b in blocks
                    .iter_mut()
                    .filter(|b| plain(b) && b.all_bold && b.font_size_pt == Some(level3))
                {
                    b.candidate = true;
                    b.block.level = Some(3);
                }
            }
        }
    }
    for b in blocks {
        if !b.candidate {
            continue;
        }
        match b.role {
            Some(Role::Subtitle) => {
                b.block.kind = Kind::Text;
                b.block.level = None;
                b.block.is_numbered_style = None;
            }
            Some(Role::Center) if first_visible_slide => {
                b.block.kind = Kind::DocTitle;
                b.block.level = Some(1);
                b.block.is_numbered_style = None;
            }
            _ => b.block.kind = Kind::ParagraphTitle,
        }
        b.candidate = false;
        b.role = None;
    }
}
