//! Runtime discovery shared by Rust and language hosts. Hosts provide package
//! roots rather than reimplementing platform-specific library selection.
use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Resources {
    pub bundle_dirs: Vec<PathBuf>,
    pub ort_library: Option<PathBuf>,
    pub pdfium_library: Option<PathBuf>,
    pub geos_library: Option<PathBuf>,
    pub operator_library: Option<PathBuf>,
    pub model_directory: Option<PathBuf>,
    pub allow_download: bool,
}
impl Default for Resources {
    fn default() -> Self {
        let path = |keys: &[&str]| {
            keys.iter().find_map(|k| {
                std::env::var_os(k)
                    .filter(|v| !v.is_empty())
                    .map(PathBuf::from)
            })
        };
        let mut bundle_dirs = vec![];
        if let Some(root) = path(&["RAPIDDOC_RUNTIME_DIR"]) {
            bundle_dirs.push(root);
        }
        if let Some(parent) = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf))
        {
            bundle_dirs.push(parent);
        }
        // Optional Cargo delivery is a fallback; explicit paths and deployed
        // application runtimes above always take precedence.
        if let Some(root) = option_env!("RAPIDDOC_BUNDLED_ORT_DIR") {
            bundle_dirs.push(PathBuf::from(root));
        }
        Self {
            bundle_dirs,
            ort_library: path(&["RAPIDDOC_ONNXRUNTIME_LIBRARY"]),
            pdfium_library: path(&["RAPIDDOC_PDFIUM_LIBRARY"]),
            geos_library: path(&["RAPIDDOC_GEOS_LIBRARY"]),
            operator_library: path(&["RAPIDDOC_OPERATOR_LIBRARY", "ONNXRUNTIME_OPS_LIBRARY"]),
            model_directory: path(&["RAPID_MODELS_DIR", "RAPIDDOC_MODEL_DIR"]),
            allow_download: true,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub enum Resource {
    Ort,
    Pdfium,
    Geos,
    Operators,
}
impl Resources {
    pub fn locate(&self, kind: Resource) -> Result<PathBuf, Error> {
        let explicit = match kind {
            Resource::Ort => &self.ort_library,
            Resource::Pdfium => &self.pdfium_library,
            Resource::Geos => &self.geos_library,
            Resource::Operators => &self.operator_library,
        };
        if let Some(path) = explicit {
            if !path.is_file() {
                return Err(runtime_error(format!(
                    "{kind:?} library not found: {}",
                    path.display()
                )));
            }
            return std::path::absolute(path).map_err(|e| runtime_error(e.to_string()));
        }
        let mut roots = vec![];
        for root in &self.bundle_dirs {
            for suffix in [
                "RapidDoc.Runtime.PrivateCRT",
                "runtime/RapidDoc.Runtime.PrivateCRT",
                "",
                "runtime",
                "capi",
                "geos",
                "onnx_ops",
                "runtime/geos",
                "runtime/onnx_ops",
            ] {
                let p = root.join(suffix);
                if !roots.contains(&p) {
                    roots.push(p);
                }
            }
        }
        for root in roots {
            let mut files = std::fs::read_dir(root)
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.is_file())
                .collect::<Vec<_>>();
            files.sort();
            for path in files {
                if path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|name| matches_library(kind, name))
                {
                    return Ok(path);
                }
            }
        }
        Err(runtime_error(format!(
            "{kind:?} runtime not found; supply Resources library paths or bundle_dirs"
        )))
    }
}
fn matches_library(kind: Resource, name: &str) -> bool {
    if cfg!(windows) {
        match kind {
            Resource::Ort => name == "onnxruntime.dll",
            Resource::Pdfium => name == "rapiddoc_pdfium.dll",
            Resource::Geos => name.starts_with("geos_c") && name.ends_with(".dll"),
            Resource::Operators => name == "onnxruntime_ops.dll",
        }
    } else if cfg!(target_os = "macos") {
        match kind {
            Resource::Ort => name.starts_with("libonnxruntime") && name.ends_with(".dylib"),
            Resource::Pdfium => name == "librapiddoc_pdfium.dylib",
            Resource::Geos => name.starts_with("libgeos_c") && name.ends_with(".dylib"),
            Resource::Operators => name == "libonnxruntime_ops.dylib",
        }
    } else {
        match kind {
            Resource::Ort => name == "libonnxruntime.so" || name.starts_with("libonnxruntime.so."),
            Resource::Pdfium => name == "librapiddoc_pdfium.so",
            Resource::Geos => name.starts_with("libgeos_c") && name.contains(".so"),
            Resource::Operators => name == "libonnxruntime_ops.so",
        }
    }
}
fn runtime_error(message: String) -> Error {
    Error {
        code: ErrorCode::RuntimeUnavailable,
        message,
    }
}
pub(super) fn store_error(error: rapiddoc_inference::model_store::StoreError) -> Error {
    use rapiddoc_inference::model_store::StoreErrorKind as K;
    Error {
        code: match error.kind {
            K::InvalidInput | K::Integrity => ErrorCode::InvalidInput,
            K::NotFound => ErrorCode::ModelMissing,
            _ => ErrorCode::RuntimeUnavailable,
        },
        message: error.message,
    }
}
