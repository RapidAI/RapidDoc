//! Word 97-2003 binary document primitives. Conversion is added after the
//! variable FIB and piece-table contracts are frozen.
pub mod body;
mod codecs;
pub mod fib;
pub mod fields;
pub mod formatting;
pub mod lists;
pub mod navigation;
pub mod paragraph;
pub mod pieces;
pub mod resources;
pub mod story;
pub mod stories;
pub mod styles;
pub mod tables;

pub mod list_tree;

pub mod auxiliary;
