use std::fmt;

/// Keep the Python Office error categories, including missing_part, at this layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Malformed,
    MissingPart,
    Encrypted,
    ResourceLimit,
    Unsupported,
}

impl ErrorKind {
    pub fn code(self) -> &'static str {
        match self {
            Self::Malformed => "malformed",
            Self::MissingPart => "missing_part",
            Self::Encrypted => "encrypted",
            Self::ResourceLimit => "resource_limit",
            Self::Unsupported => "unsupported",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
}
impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
    pub(crate) fn malformed(message: impl fmt::Display) -> Self {
        Self::new(ErrorKind::Malformed, message.to_string())
    }
    pub(crate) fn missing(name: &str) -> Self {
        Self::new(
            ErrorKind::MissingPart,
            format!("missing required Office part: {name}"),
        )
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.code(), self.message)
    }
}
impl std::error::Error for Error {}
impl From<Error> for rapiddoc_core::Error {
    fn from(error: Error) -> Self {
        use rapiddoc_core::ErrorCode;
        Self {
            code: match error.kind {
                ErrorKind::Malformed | ErrorKind::MissingPart => ErrorCode::CorruptDocument,
                ErrorKind::Encrypted => ErrorCode::Encrypted,
                ErrorKind::ResourceLimit => ErrorCode::ResourceLimit,
                ErrorKind::Unsupported => ErrorCode::Unsupported,
            },
            message: error.to_string(),
        }
    }
}
pub type Result<T> = std::result::Result<T, Error>;
