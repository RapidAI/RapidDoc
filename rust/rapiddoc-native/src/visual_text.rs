//! 固定vendor字符分行与上下标配对内核的有界调用入口。
use crate::{
    python_decimal, python_printable,
    text_boundary::{python_alnum, python_digit, python_space},
};
pub use docvortex_core::text_pipeline::{TextChar, VisualTextRun};
use std::sync::OnceLock;
pub fn prepare(
    chars: &[TextChar],
    size: [f64; 2],
    rotation: i32,
    supported: &[f64],
) -> Vec<VisualTextRun> {
    static MATH: OnceLock<Vec<[u32; 2]>> = OnceLock::new();
    let ranges =
        MATH.get_or_init(|| serde_json::from_str(include_str!("visual_math.json")).unwrap());
    let mut unicode = docvortex_core::text_pipeline::UnicodeProperties::default();
    for c in chars
        .iter()
        .flat_map(|c| c.text.chars())
        .filter(|c| !c.is_ascii())
    {
        unicode.0.entry(c).or_insert_with(|| {
            let n = c as u32;
            let i = ranges.partition_point(|r| r[0] <= n);
            let math = i > 0 && n <= ranges[i - 1][1];
            u8::from(python_alnum(c))
                | (u8::from(python_digit(c)) << 1)
                | (u8::from(math) << 2)
                | (u8::from(python_space(c)) << 3)
                | (u8::from(python_printable(c)) << 4)
                | (u8::from(python_decimal(c).is_some()) << 5)
        });
    }
    docvortex_core::text_pipeline::prepare_visual_lines(
        chars,
        size,
        rotation,
        supported,
        &unicode,
        &[],
    )
}
pub type PairRecord = ([f64; 4], f64, f64, usize, bool, usize, usize);
pub fn matches(records: Vec<PairRecord>) -> Option<Vec<(usize, usize, bool, bool)>> {
    docvortex_core::inline_pairs::matches(records)
}
/// Original geometry kernel, also used after canonical character repair.
pub fn split_runs(
    raw: Vec<Option<[f64; 4]>>,
    overrides: Vec<Option<[f64; 4]>>,
    flags: Vec<u8>,
    size: [f64; 2],
    angle: i32,
) -> Option<Vec<(usize, usize, Option<[f64; 4]>)>> {
    if raw.len() != flags.len()
        || overrides.len() != flags.len()
        || raw.len() > rapiddoc_core::resources::limit(20000)
    {
        return None;
    }
    Some(docvortex_core::geometry::visual_runs(
        raw, overrides, flags, size, angle,
    ))
}
