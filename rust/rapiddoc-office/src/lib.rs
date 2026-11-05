//! Office containers and typed semantic contracts. No Python/PDFium dependency.
//! Format converters are added separately; opening a container is not conversion.
pub mod chart;
pub mod convert;
pub mod doc;
pub mod docx;
pub mod equation;
pub mod error;
pub mod image;
pub mod links;
pub mod metafile;
pub mod model;
pub mod officeart;
pub mod ole;
mod ole_directory;
pub mod document_output;
/// Compatibility alias for Rust callers using the preview module name.
pub use document_output as output;
pub mod package;
pub mod ppt;
pub mod pptx;
pub mod spreadsheet;
pub mod xls;
pub mod xlsx;
pub mod xml;

pub use error::{Error, ErrorKind, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Doc,
    Docx,
    Xls,
    Xlsx,
    Ppt,
    Pptx,
}

impl Format {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Doc => "doc",
            Self::Docx => "docx",
            Self::Xls => "xls",
            Self::Xlsx => "xlsx",
            Self::Ppt => "ppt",
            Self::Pptx => "pptx",
        }
    }
}
