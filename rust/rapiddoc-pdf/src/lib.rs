//! PDFium 独立运行库边界：全进程单个所有者，全次调用持锁，返回值不含句柄。

use pdfium_render::prelude::*;
use rapiddoc_core::document::{Character, Diagnostic, Document, LinkFact, Page};
use rapiddoc_core::geometry::{page_transform, Affine};
use rapiddoc_core::{Error, ErrorCode, Limits, PageRange, Rect};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

struct Runtime {
    path: PathBuf,
    pdfium: Pdfium,
    // 同一 DLL 的额外符号表，仅调用原始 API，不再次初始化/销毁 PDFium。
    raw: Box<dyn PdfiumLibraryBindings>,
    fonts: std::sync::Arc<fonts::FontProvider>,
    _font_symbols: libloading::Library,
}
// Pdfium 的绑定本身是进程级 OnceCell，因此资源所有者也使用进程寿命。
// 这里的锁覆盖初始化、读取和所有局部句柄销毁，不在外部共享句柄。
static RUNTIME: Mutex<Option<Runtime>> = Mutex::new(None);
pub mod character_primitives;
pub mod classification;
pub mod drawing;
pub mod fonts;
pub mod form_structure;
pub mod image_infos;
pub mod legacy_chars;
mod unicode_fallback;
pub mod legacy_vectors;
pub mod link_annotations;
mod font_weight;
mod objects;
pub mod original_images;
pub mod parsed_document;
pub mod render;
pub mod signatures;
pub mod text_paint;
pub fn page_count(
    library: &Path,
    bytes: &[u8],
    password: Option<&str>,
    limits: &Limits,
) -> Result<usize, Error> {
    limits.check_input(bytes)?;
    if password.is_some_and(|p| p.contains('\0')) {
        return Err(Error {
            code: ErrorCode::InvalidInput,
            message: "密码包含NUL".into(),
        });
    }
    with_runtime(library, |pdfium| {
        let d = pdfium
            .load_pdf_from_byte_slice(bytes, password)
            .map_err(map_error)?;
        let n = usize::try_from(d.pages().len()).map_err(|_| Error {
            code: ErrorCode::CorruptDocument,
            message: "无效页数".into(),
        })?;
        if n > limits.max_pages {
            Err(Error {
                code: ErrorCode::ResourceLimit,
                message: "文档页数超过限制".into(),
            })
        } else {
            Ok(n)
        }
    })
}

/// 获得当前同库固定字体身份与真实回调计数；不暴露原生接口地址。
pub fn font_runtime_info(library: &Path) -> Result<serde_json::Value, Error> {
    with_state(library, |runtime| {
        let (released, requests, copies) = runtime.fonts.stats();
        Ok(
            serde_json::json!({"font_policy":fonts::POLICY,"font_name":"Droid Sans Fallback Full","font_sha256":fonts::SHA256,"released":released,"bundled_requests":requests,"full_font_copies":copies}),
        )
    })
}

/// 将 PDFium 错误映射到稳定分类，不把加密输入当作普通损坏文件。
pub(crate) fn map_error(error: PdfiumError) -> Error {
    let code = match &error {
        PdfiumError::PdfiumLibraryInternalError(
            PdfiumInternalError::PasswordError | PdfiumInternalError::SecurityError,
        ) => ErrorCode::Encrypted,
        _ => ErrorCode::CorruptDocument,
    };
    Error {
        code,
        message: error.to_string(),
    }
}

/// 在打开 DLL 前检查字节数；选择页也保持原始页号，密码仅在本次调用使用。
pub fn extract(
    library: &Path,
    bytes: &[u8],
    password: Option<&str>,
    range: &PageRange,
    limits: &Limits,
) -> Result<Document, Error> {
    limits.check_input(bytes)?;
    with_runtime(library, |pdfium| {
        extract_document(pdfium, bytes, password, range, limits)
    })
}

/// 在整次操作中持有唯一运行库锁；操作闭包返回前必须物化全部借用数据。
pub(crate) fn with_runtime<T>(
    library: &Path,
    operation: impl FnOnce(&Pdfium) -> Result<T, Error>,
) -> Result<T, Error> {
    with_state(library, |runtime| operation(&runtime.pdfium))
}

pub(crate) fn with_raw<T>(
    library: &Path,
    operation: impl FnOnce(&dyn PdfiumLibraryBindings) -> Result<T, Error>,
) -> Result<T, Error> {
    with_state(library, |runtime| operation(runtime.raw.as_ref()))
}

/// 同一规范路径的额外符号引用；不调用InitLibrary，不向外暴露符号或页面地址。
pub(crate) fn with_raw_symbols<T>(
    library: &Path,
    operation: impl FnOnce(&dyn PdfiumLibraryBindings, &libloading::Library) -> Result<T, Error>,
) -> Result<T, Error> {
    with_state(library, |runtime| {
        // SAFETY: 规范路径已由唯一Runtime验证；所有符号及调用都限制在持锁同步闭包内。
        let symbols = unsafe { libloading::Library::new(&runtime.path) }.map_err(|e| Error {
            code: ErrorCode::RuntimeUnavailable,
            message: e.to_string(),
        })?;
        operation(runtime.raw.as_ref(), &symbols)
    })
}

fn with_state<T>(
    library: &Path,
    operation: impl FnOnce(&Runtime) -> Result<T, Error>,
) -> Result<T, Error> {
    let path = library.canonicalize().map_err(|error| Error {
        code: ErrorCode::RuntimeUnavailable,
        message: error.to_string(),
    })?;
    let mut state = RUNTIME.lock().map_err(|_| Error {
        code: ErrorCode::RuntimeUnavailable,
        message: "PDFium 运行库锁已损坏".into(),
    })?;
    if let Some(runtime) = state.as_ref() {
        if runtime.path != path {
            return Err(Error {
                code: ErrorCode::RuntimeUnavailable,
                message: "同一进程不允许切换 PDFium 运行库，请使用新进程".into(),
            });
        }
    } else {
        let bindings = Pdfium::bind_to_library(&path).map_err(|error| Error {
            code: ErrorCode::RuntimeUnavailable,
            message: error.to_string(),
        })?;
        let raw = Pdfium::bind_to_library(&path).map_err(|error| Error {
            code: ErrorCode::RuntimeUnavailable,
            message: error.to_string(),
        })?;
        let pdfium = Pdfium::new(bindings);
        // SAFETY: same canonical library, retained alongside all callback owners.
        let symbols = unsafe { libloading::Library::new(&path) }.map_err(|e| Error {
            code: ErrorCode::RuntimeUnavailable,
            message: e.to_string(),
        })?;
        let fonts = fonts::install(&symbols)?;
        *state = Some(Runtime {
            path,
            pdfium,
            raw,
            fonts,
            _font_symbols: symbols,
        });
    }
    let runtime = state.as_ref().ok_or_else(|| Error {
        code: ErrorCode::RuntimeUnavailable,
        message: "PDFium 未初始化".into(),
    })?;
    fonts::check(&runtime.fonts)?;
    let result = operation(runtime);
    fonts::check(&runtime.fonts)?;
    result
}

/// 在运行库锁内提取整份自有事实，所有 document/page/textpage 按逆序析构。
fn extract_document(
    pdfium: &Pdfium,
    bytes: &[u8],
    password: Option<&str>,
    range: &PageRange,
    limits: &Limits,
) -> Result<Document, Error> {
    let document = pdfium
        .load_pdf_from_byte_slice(bytes, password)
        .map_err(map_error)?;
    let count = usize::try_from(document.pages().len()).map_err(|_| Error {
        code: ErrorCode::CorruptDocument,
        message: "无效页数".into(),
    })?;
    if count > limits.max_pages {
        return Err(Error {
            code: ErrorCode::ResourceLimit,
            message: "文档页数超过限制".into(),
        });
    }
    let mut result = Document {
        schema_version: 1,
        coordinate_system: "pdfium_native_bottom_left_points".into(),
        source_page_count: count,
        pages: Vec::new(),
        diagnostics: Vec::new(),
    };
    let mut total_chars = 0usize;
    for index in range.resolve(count)? {
        let page = document.pages().get(index as i32).map_err(map_error)?;
        let page_bounds = objects::rect(page.boundaries().bounding().map_err(map_error)?.bounds)?;
        let rotation = page.rotation().map_err(map_error)?.as_degrees() as u16;
        let visual = page_transform(page_bounds, rotation)?;
        let visual_rect: [f64; 4] = visual.rect(page_bounds)?.into();
        let text = page.text().map_err(map_error)?;
        let chars = text.chars();
        let char_count = chars.len();
        total_chars = total_chars.checked_add(char_count).ok_or_else(|| Error {
            code: ErrorCode::ResourceLimit,
            message: "字符累计数量溢出".into(),
        })?;
        if char_count > limits.max_chars_per_page || total_chars > limits.max_total_chars {
            return Err(Error {
                code: ErrorCode::ResourceLimit,
                message: "字符数量超过限制".into(),
            });
        }
        let mut characters = Vec::with_capacity(char_count);
        for (char_index, character) in chars.iter().enumerate() {
            let bounds = character.tight_bounds().map_err(map_error)?;
            let (x, y) = character.origin().map_err(map_error)?;
            let font_size = character.unscaled_font_size().value as f64;
            let angle = character.angle_degrees().map_err(map_error)? as f64;
            if [x.value as f64, y.value as f64, font_size, angle]
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(Error {
                    code: ErrorCode::CorruptDocument,
                    message: "字符几何包含非有限数值".into(),
                });
            }
            characters.push(Character {
                index: char_index,
                unicode: character.unicode_value(),
                is_generated: character.is_generated().map_err(map_error)?,
                text: character.unicode_string(),
                font_name: character.font_name(),
                font_size_pt: font_size,
                angle_degrees: angle,
                origin: [x.value as f64, y.value as f64],
                tight_bounds: Rect::try_from([
                    bounds.left().value as f64,
                    bounds.bottom().value as f64,
                    bounds.right().value as f64,
                    bounds.top().value as f64,
                ])?,
                loose_bounds: objects::rect(character.loose_bounds().map_err(map_error)?)?,
                visual_loose_bounds: visual
                    .rect(objects::rect(character.loose_bounds().map_err(map_error)?)?)?,
                visual_tight_bounds: visual.rect(objects::rect(bounds)?)?,
                visual_origin: visual.point([x.value as f64, y.value as f64])?,
            });
        }
        // text.all() 使用视觉页框查询文字，在 CropBox + 旋转组合下会漏掉
        // 部分原生字符。事实层使用完整字符索引顺序，后续布局层再决定阅读顺序。
        let value: String = characters
            .iter()
            .map(|character| character.text.as_deref().unwrap_or("\u{fffd}"))
            .collect();
        if characters.iter().any(|character| character.text.is_none()) {
            result.diagnostics.push(Diagnostic {
                code: "unmapped_unicode".into(),
                page_index: index,
                message: "存在无法映射为 Unicode 的原始字符，文字使用替代符；raw unicode 仍保留"
                    .into(),
            });
        }
        let page_objects = page.objects();
        if page_objects.len() > limits.max_objects_per_page {
            return Err(Error {
                code: ErrorCode::ResourceLimit,
                message: "页面对象超过限制".into(),
            });
        }
        let mut facts = objects::Objects::default();
        let context = objects::Context {
            visual,
            page_clip: visual.rect(page_bounds)?,
            limits,
        };
        for object_index in page_objects.as_range() {
            let object = page_objects.get(object_index).map_err(map_error)?;
            objects::walk(
                &object,
                &mut vec![object_index],
                Affine::IDENTITY,
                None,
                &mut facts,
                &context,
            )?;
        }
        let page_links = page.links();
        if page_links.len() > limits.max_links_per_page {
            return Err(Error {
                code: ErrorCode::ResourceLimit,
                message: "页面链接超过限制".into(),
            });
        }
        let mut links = Vec::with_capacity(page_links.len());
        for link_index in 0..page_links.len() {
            let link = page_links.get(link_index).map_err(map_error)?;
            let bounds = objects::rect(link.rect().map_err(map_error)?)?;
            let uri = match link.action() {
                Some(action) => match action.as_uri_action() {
                    Some(uri) => Some(uri.uri().map_err(map_error)?),
                    None => None,
                },
                None => None,
            };
            links.push(LinkFact {
                index: link_index,
                bounds,
                visual_bounds: visual.rect(bounds)?,
                uri,
            });
        }
        let object_count = facts.object_count;
        if value.trim().is_empty() && object_count > 0 {
            result.diagnostics.push(Diagnostic {
                code: "ocr_required".into(),
                page_index: index,
                message: "页面有绘制内容但没有可提取文字，需 OCR 或图形处理".into(),
            });
        }
        result.pages.push(Page {
            page_index: index,
            width_pt: page.width().value as f64,
            height_pt: page.height().value as f64,
            text: value,
            object_count,
            characters,
            page_bounds,
            rotation,
            visual_width_pt: visual_rect[2],
            visual_height_pt: visual_rect[3],
            paths: facts.paths,
            images: facts.images,
            links,
            form_count: facts.form_count,
            drawing_lines: Vec::new(),
        });
    }
    // result 已物化；局部 text/page/document 会在锁释放前按逆序析构。
    Ok(result)
}
