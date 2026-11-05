//! Fixed DocVortex Path ABI subset; no runtime or Python initialization.
pub mod objects;
pub mod path_evidence;
pub mod paths;
#[derive(Debug)]
pub enum ReadError {
    InvalidInput(&'static str),
    Pdfium(&'static str),
    Allocation(&'static str),
}
