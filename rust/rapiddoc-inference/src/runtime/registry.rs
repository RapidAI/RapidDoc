//! Language-independent lazy model ownership and shared ONNX sessions.
use crate::{Device, Model};
use rapiddoc_core::{Error, ErrorCode};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{Arc, LazyLock, Mutex, Weak},
};
fn threads() -> usize {
    1
}
fn elements() -> usize {
    isize::MAX as usize / std::mem::size_of::<f32>()
}
fn cpu() -> String {
    "cpu".into()
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {
    pub runtime: PathBuf,
    pub model: PathBuf,
    pub sha256: String,
    #[serde(default = "cpu")]
    pub device: String,
    #[serde(default = "threads")]
    pub threads: usize,
    #[serde(default = "elements")]
    pub max_elements: usize,
    #[serde(default)]
    pub cpu_spinning: Option<bool>,
    #[serde(default)]
    pub cpu_dynamic_block_base: Option<usize>,
    #[serde(default)]
    pub cpu_force_spinning_stop: Option<bool>,
    #[serde(default)]
    pub cpu_shared_arena: Option<bool>,
    /// Libraries required by in-memory optimized models. Registration stays in Rust,
    /// so other language bindings can use the same graph and descriptor.
    #[serde(default)]
    pub operator_libraries: Vec<PathBuf>,
    #[serde(default)]
    pub optimization: Option<crate::onnx::Optimization>,
}
type Key = (
    PathBuf,
    PathBuf,
    String,
    String,
    usize,
    usize,
    Option<bool>,
    Option<usize>,
    Option<bool>,
    Option<bool>,
    Vec<PathBuf>,
    Option<crate::onnx::Optimization>,
);
pub type Shared = Arc<Mutex<Model>>;
static SESSIONS: LazyLock<Mutex<BTreeMap<Key, Weak<Mutex<Model>>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));
fn error(code: ErrorCode, message: impl ToString) -> Error {
    Error {
        code,
        message: message.to_string(),
    }
}
pub fn cached(descriptor: &Descriptor) -> Result<Shared, Error> {
    let runtime = descriptor
        .runtime
        .canonicalize()
        .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?;
    let model = descriptor
        .model
        .canonicalize()
        .map_err(|e| error(ErrorCode::ModelMissing, e))?;
    let device = match descriptor.device.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda,
        _ => return Err(error(ErrorCode::InvalidInput, "device must be cpu or cuda")),
    };
    let libraries = descriptor
        .operator_libraries
        .iter()
        .map(|path| {
            path.canonicalize()
                .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let key = (
        runtime.clone(),
        model.clone(),
        descriptor.sha256.to_ascii_lowercase(),
        descriptor.device.clone(),
        descriptor.threads,
        descriptor.max_elements,
        descriptor.cpu_spinning,
        descriptor.cpu_dynamic_block_base,
        descriptor.cpu_force_spinning_stop,
        descriptor.cpu_shared_arena,
        libraries.clone(),
        descriptor.optimization.clone(),
    );
    let mut cache = SESSIONS
        .lock()
        .map_err(|_| error(ErrorCode::RuntimeUnavailable, "ONNX registry lock poisoned"))?;
    if let Some(session) = cache.get(&key).and_then(Weak::upgrade) {
        crate::verify_model(&model, &descriptor.sha256)?;
        return Ok(session);
    }
    let loaded = Model::load_with_operator_libraries(
        &runtime,
        &model,
        &descriptor.sha256,
        device,
        descriptor.threads,
        descriptor.max_elements,
        descriptor.cpu_spinning,
        descriptor.cpu_dynamic_block_base,
        descriptor.cpu_force_spinning_stop,
        descriptor.cpu_shared_arena,
        &libraries,
        descriptor.optimization.as_ref(),
    )?;
    let session = Arc::new(Mutex::new(loaded));
    cache.retain(|_, v| v.strong_count() > 0);
    cache.insert(key, Arc::downgrade(&session));
    Ok(session)
}
#[derive(Clone, Default)]
struct ModelCache {
    descriptors: BTreeMap<String, Descriptor>,
    sessions: BTreeMap<String, Shared>,
}
/// Clones share weights and role discovery, but keep request timing separate.
pub struct Models {
    initialization: std::time::Duration,
    cache: Arc<Mutex<ModelCache>>,
}
impl Clone for Models {
    fn clone(&self) -> Self {
        Self {
            initialization: std::time::Duration::ZERO,
            cache: self.cache.clone(),
        }
    }
}
impl Models {
    pub fn retain_roles(&mut self, roles: &[&str]) {
        // Configuration changes detach from in-flight requests' cache.
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner()).clone();
        cache.descriptors.retain(|k, _| roles.contains(&k.as_str()));
        cache.sessions.retain(|k, _| roles.contains(&k.as_str()));
        self.cache = Arc::new(Mutex::new(cache));
    }
    pub fn loaded_roles(&self) -> Vec<String> {
        self.cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .sessions
            .keys()
            .cloned()
            .collect()
    }
    pub fn new(descriptors: BTreeMap<String, Descriptor>) -> Self {
        Self {
            initialization: std::time::Duration::ZERO,
            cache: Arc::new(Mutex::new(ModelCache {
                descriptors,
                sessions: BTreeMap::new(),
            })),
        }
    }
    pub fn with<T>(
        &mut self,
        role: &str,
        operation: impl FnOnce(&mut Model) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let session = self.session(role)?;
        let mut model = session
            .lock()
            .map_err(|_| error(ErrorCode::RuntimeUnavailable, "ONNX session lock poisoned"))?;
        operation(&mut model)
    }
    fn session(&mut self, role: &str) -> Result<Shared, Error> {
        let (session, descriptor) = {
            let cache = self
                .cache
                .lock()
                .map_err(|_| error(ErrorCode::RuntimeUnavailable, "Model cache lock poisoned"))?;
            (
                cache.sessions.get(role).cloned(),
                cache.descriptors.get(role).cloned(),
            )
        };
        let session = if let Some(session) = session {
            session
        } else {
            let descriptor = descriptor.ok_or_else(|| {
                error(
                    ErrorCode::ModelMissing,
                    format!("Required model missing: {role}"),
                )
            })?;
            let started = std::time::Instant::now();
            // The global registry deduplicates simultaneous first loads. Never
            // hold the role cache while loading a model or running host code.
            let loaded = cached(&descriptor);
            self.initialization += started.elapsed();
            let loaded = loaded?;
            let mut cache = self
                .cache
                .lock()
                .map_err(|_| error(ErrorCode::RuntimeUnavailable, "Model cache lock poisoned"))?;
            cache.sessions.entry(role.into()).or_insert(loaded).clone()
        };
        Ok(session)
    }
    pub fn recognize_images_with_progress(
        &mut self,
        role: &str,
        resolve: &mut dyn FnMut(&str) -> Result<Descriptor, Error>,
        images: &[crate::recognition_batch::Image],
        batch_size: usize,
        progress: &mut dyn FnMut(usize, usize) -> Result<(), Error>,
    ) -> Result<Vec<crate::ctc::Recognition>, Error> {
        self.with_lazy(role, resolve, |_| Ok(()))?;
        let session = self.session(role)?;
        let max_elements = session
            .lock()
            .map_err(|_| error(ErrorCode::RuntimeUnavailable, "ONNX session lock poisoned"))?
            .max_elements;
        crate::recognition_runtime::recognize(
            images,
            batch_size,
            max_elements,
            progress,
            &mut |shape, values| {
                let pending = session
                    .lock()
                    .map_err(|_| {
                        error(ErrorCode::RuntimeUnavailable, "ONNX session lock poisoned")
                    })?
                    .forward_ctc(shape, values)?;
                Ok(pending)
            },
        )
    }
    pub fn warmup(&mut self, roles: &[&str]) -> Result<(), Error> {
        for role in roles {
            self.with(role, |_| Ok(()))?;
        }
        Ok(())
    }
    pub fn initialization_time(&self) -> std::time::Duration {
        self.initialization
    }
    pub fn with_lazy<T>(
        &mut self,
        role: &str,
        resolve: &mut dyn FnMut(&str) -> Result<Descriptor, Error>,
        operation: impl FnOnce(&mut Model) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let missing = !self
            .cache
            .lock()
            .map_err(|_| error(ErrorCode::RuntimeUnavailable, "Model cache lock poisoned"))?
            .descriptors
            .contains_key(role);
        if missing {
            let started = std::time::Instant::now();
            let descriptor = resolve(role);
            self.initialization += started.elapsed();
            self.cache
                .lock()
                .map_err(|_| error(ErrorCode::RuntimeUnavailable, "Model cache lock poisoned"))?
                .descriptors
                .entry(role.into())
                .or_insert(descriptor?);
        }
        self.with(role, operation)
    }
}
