//! Public seven-format native SDK. Language adapters do not schedule algorithms.
use crate::pipeline;
use rapiddoc_core::{Error, ErrorCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};
mod config;
mod concurrency;
pub mod input;
mod models;
mod resources;
pub use config::{Config, ParseOptions};
pub use models::{ModelPlan, ModelSelection};
use resources::store_error;
pub use resources::{Resource, Resources};
fn invalid(message: impl Into<String>) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.into(),
    }
}

/// Preserve typed Office output until serialization, rather than building a
/// second complete JSON tree just to share the public document envelope.
#[derive(Serialize)]
#[serde(untagged)]
pub enum Middle {
    Pdf(Value),
    Office(rapiddoc_office::document_output::Middle),
}
impl Middle {
    pub fn page_count(&self) -> usize {
        match self {
            Self::Pdf(v) => v["pdf_info"].as_array().map_or(0, Vec::len),
            Self::Office(v) => v.pdf_info.len(),
        }
    }
}
/// Keep existing ownership: PDF PNGs are Vecs, Office assets are already shared.
pub enum Image {
    Owned(Vec<u8>),
    Shared(Arc<[u8]>),
}
impl AsRef<[u8]> for Image {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v,
            Self::Shared(v) => v,
        }
    }
}
impl std::ops::Deref for Image {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_ref()
    }
}

#[derive(Serialize)]
pub struct Document {
    pub markdown: Value,
    pub middle_json: Middle,
    pub content_list_json: Value,
    pub source_page_count: usize,
    pub diagnostics: Vec<String>,
    pub metadata: Value,
    #[serde(skip)]
    pub images: BTreeMap<String, Image>,
    #[serde(skip)]
    pub drawings: BTreeMap<String, Vec<u8>>,
}
pub type Progress<'a> = pipeline::Progress<'a>;
/// Optional host hook, e.g. a Python environment's NVIDIA preload helper. The
/// native Rust API works without it when OS runtime dependencies are available.
pub type DeviceHook<'a> = dyn FnMut(&str) -> Result<(), Error> + 'a;
#[derive(Clone)]
pub struct RapidDoc {
    config: Config,
    resources: Resources,
    plan: OnceLock<ModelPlan>,
    pdf: pipeline::Engine,
    identity: Arc<()>,
}
impl RapidDoc {
    pub fn new(config: Config, resources: Resources) -> Self {
        Self {
            config,
            resources,
            plan: OnceLock::new(),
            pdf: pipeline::Engine::new(BTreeMap::new()),
            identity: Arc::new(()),
        }
    }
    pub fn configure(&mut self, config: Config) -> Result<(), Error> {
        if self.config == config {
            return Ok(());
        }
        let new = config.model_plan()?;
        let retained = self
            .plan
            .get()
            .map(|old| {
                old.models
                    .iter()
                    .filter(|(name, model)| new.models.get(*name) == Some(*model))
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        self.pdf.retain_model_roles(&retained);
        self.plan = OnceLock::from(new);
        self.config = config;
        Ok(())
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn resources_mut(&mut self) -> &mut Resources {
        self.pdf = pipeline::Engine::new(BTreeMap::new());
        &mut self.resources
    }
    pub fn set_resources(&mut self, resources: Resources) {
        if self.resources != resources {
            self.pdf = pipeline::Engine::new(BTreeMap::new());
            self.resources = resources;
        }
    }
    pub fn loaded_model_roles(&self) -> Vec<String> {
        self.pdf.loaded_model_roles()
    }
    fn plan(&self) -> Result<&ModelPlan, Error> {
        if self.plan.get().is_none() {
            let _ = self.plan.set(self.config.model_plan()?);
        }
        Ok(self.plan.get().unwrap())
    }
    pub fn warmup(&self, options: &ParseOptions) -> Result<Vec<String>, Error> {
        self.warmup_with_host(options, &mut |_| Ok(()), &mut |_| Ok(()))
    }
    pub fn warmup_with_host(
        &self,
        options: &ParseOptions,
        progress: &mut dyn FnMut(Progress<'_>) -> Result<(), Error>,
        device: &mut DeviceHook<'_>,
    ) -> Result<Vec<String>, Error> {
        let _request = concurrency::Request::enter(&self.identity)?;
        let plan = self.plan()?.clone();
        let request = self.config.request(options, &plan)?;
        let roles = plan.warmup_roles(&request);
        let resources = &self.resources;
        self.pdf.clone().warmup_models(&roles, &mut |role| {
            resolve(&plan, resources, role, progress, device)
        })?;
        Ok(self.loaded_model_roles())
    }
    pub fn parse(&self, path: impl AsRef<Path>, options: &ParseOptions) -> Result<Document, Error> {
        self.parse_with_progress(path, options, &mut |_| Ok(()))
    }
    /// Read a file natively while reporting the same events as byte input.
    pub fn parse_with_progress(
        &self,
        path: impl AsRef<Path>,
        options: &ParseOptions,
        progress: &mut dyn FnMut(Progress<'_>) -> Result<(), Error>,
    ) -> Result<Document, Error> {
        let _request = concurrency::Request::enter(&self.identity)?;
        let path = path.as_ref();
        let remote = path.to_str().filter(|source| input::is_url(source));
        let (name, data) = if let Some(source) = remote {
            input::read_url(source, progress)?
        } else {
            (
                input::source_name(&path.to_string_lossy()),
                std::fs::read(path).map_err(|e| invalid(e.to_string()))?,
            )
        };
        let mut options = options.clone();
        if data.is_empty() && options.empty_format.is_none() {
            options.empty_format = Path::new(&name)
                .extension()
                .and_then(|s| s.to_str())
                .map(str::to_lowercase);
        }
        let mut document = self.parse_inner(&data, &options, progress, &mut |_| Ok(()))?;
        if remote.is_some() {
            document.metadata["source_name"] = name.into();
        }
        Ok(document)
    }
    pub fn parse_bytes(
        &self,
        data: &[u8],
        options: &ParseOptions,
        progress: &mut dyn FnMut(Progress<'_>) -> Result<(), Error>,
    ) -> Result<Document, Error> {
        self.parse_with_host(data, options, progress, &mut |_| Ok(()))
    }
    pub fn parse_with_host(
        &self,
        data: &[u8],
        options: &ParseOptions,
        progress: &mut dyn FnMut(Progress<'_>) -> Result<(), Error>,
        device: &mut DeviceHook<'_>,
    ) -> Result<Document, Error> {
        let _request = concurrency::Request::enter(&self.identity)?;
        self.parse_inner(data, options, progress, device)
    }
    fn parse_inner(
        &self,
        data: &[u8],
        options: &ParseOptions,
        progress: &mut dyn FnMut(Progress<'_>) -> Result<(), Error>,
        device: &mut DeviceHook<'_>,
    ) -> Result<Document, Error> {
        let prefix = options.image_dir_name.as_deref().unwrap_or("images");
        if prefix.is_empty() || prefix == "." || prefix == ".." || prefix.contains(['/', '\\']) {
            return Err(invalid("image_dir_name must be a directory name"));
        }
        if data.is_empty()
            || data.starts_with(b"PK\x03\x04")
            || data.starts_with(b"PK\x05\x06")
            || data.starts_with(&[0xd0, 0xcf, 0x11, 0xe0])
        {
            let delimiters = options.latex_delimiters.clone().unwrap_or_default();
            let doc = rapiddoc_office::convert::parse_document(
                Arc::from(data),
                &self.config.make_md_mode,
                prefix,
                &delimiters,
                false,
                options.empty_format.as_deref(),
            )?;
            return Ok(Document {
                markdown: doc.markdown,
                middle_json: Middle::Office(doc.middle_json),
                content_list_json: doc.content_list_json,
                source_page_count: doc.source_page_count,
                diagnostics: doc.diagnostics,
                images: doc
                    .assets
                    .into_iter()
                    .map(|(name, bytes)| (name, Image::Shared(bytes)))
                    .collect(),
                drawings: BTreeMap::new(),
                metadata: json!({"format":"office"}),
            });
        }
        let plan = self.plan()?.clone();
        let mut request = self.config.request(options, &plan)?;
        request.geos_library = Some(self.resources.locate(Resource::Geos)?);
        let library = self.resources.locate(Resource::Pdfium)?;
        // Events and resource discovery share one synchronous callback without
        // holding language-runtime objects or invoking host algorithms.
        let progress = std::cell::RefCell::new(progress);
        let mut output = self.pdf.clone().parse(
            &library,
            data,
            &request,
            &mut |role| {
                resolve(
                    &plan,
                    &self.resources,
                    role,
                    &mut **progress.borrow_mut(),
                    device,
                )
            },
            &mut |event| (**progress.borrow_mut())(event),
        )?;
        output.render_document(
            &self.config.make_md_mode,
            prefix,
            self.resources.locate(Resource::Ort).ok().as_deref(),
        )?;
        let kinds = [
            (options.draw_layout, rapiddoc_pdf::drawing::Kind::Layout),
            (options.draw_span, rapiddoc_pdf::drawing::Kind::Span),
            (
                options.draw_line_sort,
                rapiddoc_pdf::drawing::Kind::LineSort,
            ),
        ]
        .into_iter()
        .filter_map(|(enabled, kind)| enabled.then_some(kind))
        .collect::<Vec<_>>();
        let start = output.metadata["page_range"][0].as_u64().unwrap() as usize;
        let end = output.metadata["page_range"][1].as_u64().unwrap() as usize;
        let drawings = rapiddoc_pdf::drawing::draw(
            data,
            output.metadata["pages"].as_array().unwrap(),
            start,
            Some(end),
            &kinds,
        )?
        .into_iter()
        .map(|(kind, bytes)| (kind.name().into(), bytes))
        .collect();
        let fields = output.metadata.as_object_mut().unwrap();
        let pages = fields.remove("pages").unwrap();
        let markdown = fields.remove("markdown").unwrap();
        let content_list_json = fields.remove("content_list_json").unwrap();
        let mut middle_json = json!({"_backend":"pipeline","_version_name":"0.9.11"});
        middle_json["pdf_info"] = pages;
        Ok(Document {
            markdown,
            middle_json: Middle::Pdf(middle_json),
            content_list_json,
            source_page_count: output.metadata["source_page_count"].as_u64().unwrap() as usize,
            diagnostics: vec![],
            metadata: output.metadata,
            images: output
                .assets
                .into_iter()
                .map(|a| {
                    (
                        a.name.strip_prefix("images/").unwrap_or(&a.name).to_owned(),
                        Image::Owned(a.png),
                    )
                })
                .collect(),
            drawings,
        })
    }
}
impl Default for RapidDoc {
    fn default() -> Self {
        Self::new(Config::default(), Resources::default())
    }
}
fn resolve(
    plan: &ModelPlan,
    resources: &Resources,
    role: &str,
    progress: &mut dyn FnMut(Progress<'_>) -> Result<(), Error>,
    device: &mut DeviceHook<'_>,
) -> Result<rapiddoc_inference::registry::Descriptor, Error> {
    let selection = plan
        .models
        .get(role)
        .ok_or_else(|| invalid(format!("Unknown model role: {role}")))?;
    if selection.device == "cuda" {
        device("cuda")?;
    }
    let (descriptor, downloaded) = selection.resolve_progress(resources, &mut |event| {
        progress(Progress {
            stage: event.stage,
            detail: Some(serde_json::to_string(event).map_err(|e| invalid(e.to_string()))?),
            elapsed_seconds: Some(event.elapsed_seconds),
            completed: usize::try_from(event.completed).ok(),
            total: event.total.and_then(|n| usize::try_from(n).ok()),
            page_index: None,
        })
    })?;
    let detail=json!({"role":role,"device":descriptor.device,"model":descriptor.model,"optimization":descriptor.optimization,"threads":descriptor.threads,"downloaded":downloaded}).to_string();
    progress(Progress {
        stage: "model_resolved",
        detail: Some(detail),
        elapsed_seconds: None,
        completed: None,
        total: None,
        page_index: None,
    })?;
    Ok(descriptor)
}
