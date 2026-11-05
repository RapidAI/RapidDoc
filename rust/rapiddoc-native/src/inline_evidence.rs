//! 原样固定vendor样式内核；JSON适配层先校验索引与工作上限。
pub use docvortex_core::inline_styles::{Character, Drawing, Line, Payload};
pub fn styles(chars: &[Character], lines: Vec<Line>, drawings: &[Drawing]) -> Option<Vec<Payload>> {
    docvortex_core::inline_styles::detect(
        chars,
        lines,
        drawings,
        [1.8, 0.2, 0.2, 0.2, 0.55, 0.5, 0.15, 0.8],
        2,
    )
}
