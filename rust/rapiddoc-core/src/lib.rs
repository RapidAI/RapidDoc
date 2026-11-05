//! 不依赖 Python、PDFium 或推理运行库的基础契约。

use serde::{Deserialize, Serialize};
pub mod document;
pub mod geometry;
pub mod ocr;
pub mod pillow_resize;
pub mod progress;
pub mod resources;

/// 解析错误的稳定分类，调用方不应依靠错误文案分支。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidInput,
    Unsupported,
    Encrypted,
    CorruptDocument,
    ResourceLimit,
    ModelMissing,
    DeviceUnavailable,
    RuntimeUnavailable,
}

impl ErrorCode {
    /// 跨语言稳定的错误码，与 serde 名称一致。
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::InvalidInput => "invalid_input",
            Self::Unsupported => "unsupported",
            Self::Encrypted => "encrypted",
            Self::CorruptDocument => "corrupt_document",
            Self::ResourceLimit => "resource_limit",
            Self::ModelMissing => "model_missing",
            Self::DeviceUnavailable => "device_unavailable",
            Self::RuntimeUnavailable => "runtime_unavailable",
        }
    }
}

/// 不包含裸 FFI 句柄的错误。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}: {}", self.code, self.message)
    }
}
impl std::error::Error for Error {}

/// 有限、非反向的矩形；单位由所在页面坐标系声明。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "[f64; 4]", into = "[f64; 4]")]
pub struct Rect([f64; 4]);

impl TryFrom<[f64; 4]> for Rect {
    type Error = Error;

    /// 校验反向边界及 NaN/Infinity；允许零面积框表示空几何。
    fn try_from(value: [f64; 4]) -> Result<Self, Error> {
        if value.iter().any(|x| !x.is_finite()) || value[2] < value[0] || value[3] < value[1] {
            return Err(Error {
                code: ErrorCode::InvalidInput,
                message: "矩形坐标必须有限且有序".into(),
            });
        }
        Ok(Self(value))
    }
}

impl From<Rect> for [f64; 4] {
    fn from(value: Rect) -> Self {
        value.0
    }
}

/// 页范围使用从零开始的原始页码，结束位置不包含在结果中。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PageRange {
    pub start: usize,
    pub end: Option<usize>,
}

impl PageRange {
    /// 校验范围，不静默截断超界请求或重排页码。
    pub fn resolve(&self, page_count: usize) -> Result<std::ops::Range<usize>, Error> {
        let end = self.end.unwrap_or(page_count);
        if self.start > end || end > page_count {
            return Err(Error {
                code: ErrorCode::InvalidInput,
                message: "页范围越界".into(),
            });
        }
        Ok(self.start..end)
    }
}

fn default_native_chars() -> usize {
    20_000
}
/// 输入资源限制必须在打开原生库之前检查。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Limits {
    pub max_input_bytes: usize,
    pub max_pages: usize,
    pub max_chars_per_page: usize,
    #[serde(default = "default_native_chars")]
    pub max_native_chars_per_page: usize,
    pub max_total_chars: usize,
    pub max_objects_per_page: usize,
    pub max_path_segments_per_page: usize,
    pub max_form_depth: usize,
    pub max_links_per_page: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_input_bytes: 256 * 1024 * 1024,
            max_pages: 10_000,
            max_chars_per_page: 1_000_000,
            max_native_chars_per_page: 20_000,
            max_total_chars: 10_000_000,
            max_objects_per_page: 200_000,
            max_path_segments_per_page: 2_000_000,
            max_form_depth: 32,
            max_links_per_page: 100_000,
        }
    }
}

impl Limits {
    /// Match Python's document API: no invented input/count/work cutoffs.
    /// Explicit limits remain available through `Limits::default()` and fields.
    pub fn pdf_host() -> Self {
        Self {
            max_input_bytes: usize::MAX,
            max_pages: usize::MAX,
            max_chars_per_page: usize::MAX,
            max_native_chars_per_page: usize::MAX,
            max_total_chars: usize::MAX,
            max_objects_per_page: usize::MAX,
            max_path_segments_per_page: usize::MAX,
            max_form_depth: usize::MAX,
            max_links_per_page: usize::MAX,
        }
    }
    /// 空输入和超限输入明确返回错误，避免误认为成功的空文档。
    pub fn check_input(&self, bytes: &[u8]) -> Result<(), Error> {
        if bytes.is_empty() {
            return Err(Error {
                code: ErrorCode::InvalidInput,
                message: "输入为空".into(),
            });
        }
        if bytes.len() > self.max_input_bytes {
            return Err(Error {
                code: ErrorCode::ResourceLimit,
                message: "输入超过字节限制".into(),
            });
        }
        Ok(())
    }
}
