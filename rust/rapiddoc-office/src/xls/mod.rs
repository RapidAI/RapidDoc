//! Native BIFF5/8 workbooks, OfficeArt resources and spreadsheet page projection.
pub mod chart;
pub(crate) mod codepages;
pub mod embedded_chart;
mod hyperlink;
pub mod number_format;
mod parser;
mod projection;
pub mod records;
mod resources;
pub mod strings;
pub use parser::{Cell, ChartSheet, Descriptor, Workbook, Worksheet};
pub use resources::{Chart, Equation, Image};
