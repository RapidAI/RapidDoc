//! Local original-model discovery and verified streaming downloads shared by
//! all bindings. No Python installation, project path, or optimized disk cache.
mod catalog;
pub use catalog::{Catalog, Model, CATALOG_JSON};

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    env, fmt, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreErrorKind {
    InvalidInput,
    NotFound,
    Integrity,
    Io,
    Http,
}

#[derive(Debug)]
pub struct StoreError {
    pub kind: StoreErrorKind,
    pub message: String,
}
impl StoreError {
    pub fn new(kind: StoreErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
    fn invalid(message: impl Into<String>) -> Self {
        Self::new(StoreErrorKind::InvalidInput, message)
    }
}
impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for StoreError {}
impl From<io::Error> for StoreError {
    fn from(e: io::Error) -> Self {
        Self::new(StoreErrorKind::Io, e.to_string())
    }
}

#[derive(Debug, Serialize)]
pub struct DownloadEvent {
    pub stage: &'static str,
    pub model_id: String,
    pub url: String,
    pub path: PathBuf,
    pub completed: u64,
    pub total: Option<u64>,
    pub elapsed_seconds: f64,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ResolvedModel {
    pub model_id: String,
    pub path: PathBuf,
    pub sha256: String,
    pub downloaded: bool,
}

fn home_directory() -> Result<PathBuf, StoreError> {
    let keys = if cfg!(windows) {
        ["USERPROFILE", "HOME"]
    } else {
        ["HOME", "USERPROFILE"]
    };
    keys.iter()
        .find_map(|key| {
            env::var_os(key)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        })
        .ok_or_else(|| StoreError::invalid("Cannot locate home directory; set RAPID_MODELS_DIR"))
}

fn absolute(path: &Path) -> Result<PathBuf, StoreError> {
    let mut parts = path.components();
    let expanded = if parts.next().is_some_and(|p| p.as_os_str() == "~") {
        home_directory()?.join(parts.as_path())
    } else {
        path.to_path_buf()
    };
    Ok(std::path::absolute(expanded)?)
}

pub fn model_directory(explicit: Option<&Path>) -> Result<PathBuf, StoreError> {
    if let Some(path) = explicit {
        return absolute(path);
    }
    for key in ["RAPID_MODELS_DIR", "RAPIDDOC_MODEL_DIR"] {
        if let Some(value) = env::var_os(key).filter(|v| !v.is_empty()) {
            return absolute(Path::new(&value));
        }
    }
    absolute(&home_directory()?.join(".cache/rapid_doc/models"))
}

pub fn digest(path: &Path) -> Result<String, StoreError> {
    let mut source = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let size = source.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        hash.update(&buffer[..size]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn verify(path: &Path, model: &Model) -> Result<String, StoreError> {
    let sha = digest(path)?;
    if !sha.eq_ignore_ascii_case(&model.sha256) {
        return Err(StoreError::new(
            StoreErrorKind::Integrity,
            format!(
                "Model hash mismatch: {}; use an explicit path for custom weights",
                path.display()
            ),
        ));
    }
    Ok(sha)
}

fn find_local(directory: &Path, model: &Model) -> Result<Option<PathBuf>, StoreError> {
    let direct = directory.join(&model.filename);
    if direct.is_file() {
        verify(&direct, model)?;
        return Ok(Some(direct));
    }
    if !directory.exists() {
        return Ok(None);
    }
    // Search only the configured/shared model directory. Never follow links
    // into a source tree or walk an installed Python package.
    let mut pending = vec![directory.to_path_buf()];
    let mut found = Vec::new();
    while let Some(folder) = pending.pop() {
        for entry in fs::read_dir(folder)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_dir() && !kind.is_symlink() {
                pending.push(entry.path());
            } else if kind.is_file() && entry.file_name() == model.filename.as_str() {
                found.push(entry.path());
            }
        }
    }
    found.sort();
    if found.is_empty() {
        return Ok(None);
    }
    for path in &found {
        if digest(path)?.eq_ignore_ascii_case(&model.sha256) {
            return Ok(Some(path.clone()));
        }
    }
    Err(StoreError::new(
        StoreErrorKind::Integrity,
        format!(
            "Model hash mismatch in {} for {}",
            directory.display(),
            model.filename
        ),
    ))
}

static TEMP_ID: AtomicU64 = AtomicU64::new(0);
struct PartialFile(PathBuf);
impl Drop for PartialFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn partial(directory: &Path, filename: &str) -> Result<(PartialFile, fs::File), StoreError> {
    loop {
        let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(".{filename}.{}.{id}.part", std::process::id()));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => return Ok((PartialFile(path), file)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
}

impl Catalog {
    /// Resolve a role or concrete model ID. Custom paths deliberately compute
    /// their own hash and do not trigger default-model graph optimizations.
    pub fn resolve(
        &self,
        role_or_id: &str,
        directory: Option<&Path>,
        custom: Option<&Path>,
        allow_download: bool,
    ) -> Result<ResolvedModel, StoreError> {
        self.resolve_progress(role_or_id, directory, custom, allow_download, &mut |_| {
            Ok(())
        })
    }

    /// Streaming download events; cached/custom models produce no download events.
    /// Callback errors cancel the transfer and retain partial-file cleanup.
    pub fn resolve_progress(
        &self,
        role_or_id: &str,
        directory: Option<&Path>,
        custom: Option<&Path>,
        allow_download: bool,
        progress: &mut dyn FnMut(&DownloadEvent) -> Result<(), StoreError>,
    ) -> Result<ResolvedModel, StoreError> {
        let (id, model) = self.model(role_or_id)?;
        let mut event = DownloadEvent {
            stage: "download_started",
            model_id: id.into(),
            url: model.url.clone(),
            path: model_directory(directory)?.join(&model.filename),
            completed: 0,
            total: None,
            elapsed_seconds: 0.0,
            error: None,
        };
        let start = Instant::now();
        let mut downloading = false;
        let mut cancelled = false;
        let mut notify = |event: &DownloadEvent| {
            progress(event).map_err(|error| {
                cancelled = true;
                error
            })
        };
        let result = self.resolve_with(
            role_or_id,
            directory,
            custom,
            allow_download,
            |url, output| {
                downloading = true;
                notify(&event)?;
                let agent = ureq::Agent::config_builder()
                    .timeout_connect(Some(Duration::from_secs(60)))
                    .timeout_recv_response(Some(Duration::from_secs(60)))
                    .timeout_per_read(Some(Duration::from_secs(60)))
                    .build()
                    .new_agent();
                let mut response = agent
                    .get(url)
                    .call()
                    .map_err(|e| StoreError::new(StoreErrorKind::Http, e.to_string()))?;
                event.total = response
                    .headers()
                    .get("content-length")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse().ok());
                event.stage = "download_progress";
                event.elapsed_seconds = start.elapsed().as_secs_f64();
                notify(&event)?;
                let mut last = Instant::now();
                let mut reader = response.body_mut().as_reader();
                let mut buffer = [0u8; 65536];
                loop {
                    let count = reader.read(&mut buffer)?;
                    if count == 0 {
                        break;
                    }
                    output.write_all(&buffer[..count])?;
                    event.completed += count as u64;
                    if last.elapsed() >= Duration::from_secs(1) {
                        event.elapsed_seconds = start.elapsed().as_secs_f64();
                        notify(&event)?;
                        last = Instant::now();
                    }
                }
                event.stage = "download_verifying";
                event.elapsed_seconds = start.elapsed().as_secs_f64();
                notify(&event)?;
                Ok(())
            },
        );
        drop(notify);
        if downloading && !cancelled {
            event.elapsed_seconds = start.elapsed().as_secs_f64();
            match &result {
                Ok(model) => {
                    event.stage = "download_completed";
                    event.path = model.path.clone();
                }
                Err(error) => {
                    event.stage = "download_failed";
                    event.error = Some(error.to_string());
                }
            }
            progress(&event)?;
        }
        result
    }

    fn resolve_with<F>(
        &self,
        role_or_id: &str,
        directory: Option<&Path>,
        custom: Option<&Path>,
        allow_download: bool,
        fetch: F,
    ) -> Result<ResolvedModel, StoreError>
    where
        F: FnOnce(&str, &mut dyn Write) -> Result<(), StoreError>,
    {
        let (id, model) = self.model(role_or_id)?;
        if let Some(custom) = custom {
            let path = absolute(custom)?;
            if !path.is_file() {
                return Err(StoreError::new(
                    StoreErrorKind::NotFound,
                    format!("Model not found: {}", path.display()),
                ));
            }
            return Ok(ResolvedModel {
                model_id: id.into(),
                sha256: digest(&path)?,
                path,
                downloaded: false,
            });
        }
        let directory = model_directory(directory)?;
        if let Some(path) = find_local(&directory, model)? {
            return Ok(ResolvedModel {
                model_id: id.into(),
                path,
                sha256: model.sha256.to_ascii_lowercase(),
                downloaded: false,
            });
        }
        if !allow_download {
            return Err(StoreError::new(
                StoreErrorKind::NotFound,
                format!("Model {} not found in {}", id, directory.display()),
            ));
        }
        fs::create_dir_all(&directory)?;
        let (temporary, mut file) = partial(&directory, &model.filename)?;
        fetch(&model.url, &mut file)?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        verify(&temporary.0, model)?;
        let path = directory.join(&model.filename);
        // A concurrent process may have already published the same file.
        if path.is_file() {
            verify(&path, model)?;
        } else if let Err(e) = fs::rename(&temporary.0, &path) {
            if path.is_file() {
                verify(&path, model)?;
            } else {
                return Err(e.into());
            }
        }
        Ok(ResolvedModel {
            model_id: id.into(),
            path,
            sha256: model.sha256.to_ascii_lowercase(),
            downloaded: true,
        })
    }
}

#[cfg(test)]
mod tests;
