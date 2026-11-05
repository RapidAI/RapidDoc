//! Synchronous worker-local N-API adapter. Node's public API runs this in a
//! dedicated worker isolate; no model/page scheduling is implemented here.
use napi::bindgen_prelude::*;
use napi_derive::napi;
use rapiddoc::{ParseOptions, RapidDoc, RapidDocConfig, RuntimeResources};
use rapiddoc_core::{Error as CoreError, ErrorCode};
use serde_json::json;
use std::{
    collections::HashMap,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Mutex,
};

// Worker isolates may unload and reload this addon on different OS threads.
// Use the host allocator: mimalloc's process-global heap initialization can
// retain TLS owned by the first (already terminated) worker on Linux.

fn invalid(error: impl ToString) -> CoreError {
    CoreError {
        code: ErrorCode::InvalidInput,
        message: error.to_string(),
    }
}
fn event_json(event: rapiddoc::sdk::Progress<'_>) -> String {
    json!({
        "stage": event.stage,
        "page_index": event.page_index,
        "completed": event.completed,
        "total": event.total,
        "elapsed_seconds": event.elapsed_seconds,
        "detail": event.detail,
    })
    .to_string()
}
fn boundary<T>(call: impl FnOnce() -> std::result::Result<T, CoreError>) -> Result<T> {
    match catch_unwind(AssertUnwindSafe(call)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(Error::from_reason(
            json!({"code":error.code.as_str(),"message":error.message}).to_string(),
        )),
        Err(_) => Err(Error::from_reason(
            json!({"code":"native_panic","message":"Rust panic at N-API boundary"}).to_string(),
        )),
    }
}

#[napi(object)]
pub struct NativeResult {
    pub json: String,
    pub images: HashMap<String, Buffer>,
    pub drawings: HashMap<String, Buffer>,
}

#[napi]
pub struct NativeEngine {
    engine: Mutex<Option<RapidDoc>>,
}

#[napi]
impl NativeEngine {
    #[napi(constructor)]
    pub fn new(config: String, resources: String) -> Result<Self> {
        // Rendering/inference dependencies own process-global native threads.
        // Keep their code mapped after an individual worker isolate closes;
        // close() still releases the engine, model sessions and document data.
        napi::bindgen_prelude::retain_current_module_for_unload_safety();
        boundary(|| {
            Ok(Self {
                engine: Mutex::new(Some(RapidDoc::new(
                    serde_json::from_str::<RapidDocConfig>(&config).map_err(invalid)?,
                    serde_json::from_str::<RuntimeResources>(&resources).map_err(invalid)?,
                ))),
            })
        })
    }

    #[napi]
    pub fn parse(
        &self,
        path: Option<String>,
        data: Option<Buffer>,
        options: String,
        callback: Function<'_, String, bool>,
    ) -> Result<NativeResult> {
        boundary(|| {
            let mut guard = self.engine.try_lock().map_err(|_| invalid("Engine busy"))?;
            let engine = guard.as_mut().ok_or_else(|| invalid("Engine closed"))?;
            let options: ParseOptions = serde_json::from_str(&options).map_err(invalid)?;
            let mut progress = |event: rapiddoc::sdk::Progress<'_>| {
                if !callback.call(event_json(event)).map_err(invalid)? {
                    return Err(invalid("Parsing cancelled by host"));
                }
                Ok(())
            };
            let doc = if let Some(path) = path {
                engine.parse_with_progress(path, &options, &mut progress)?
            } else {
                engine.parse_bytes(
                    data.as_deref()
                        .ok_or_else(|| invalid("Missing bytes input"))?,
                    &options,
                    &mut progress,
                )?
            };
            let metadata = serde_json::to_string(&doc).map_err(invalid)?;
            Ok(NativeResult {
                json: metadata,
                images: doc
                    .images
                    .into_iter()
                    .map(|(name, image)| {
                        let data = match image {
                            rapiddoc::sdk::Image::Owned(data) => data,
                            rapiddoc::sdk::Image::Shared(data) => data.to_vec(),
                        };
                        (name, Buffer::from(data))
                    })
                    .collect(),
                drawings: doc
                    .drawings
                    .into_iter()
                    .map(|(name, data)| (name, Buffer::from(data)))
                    .collect(),
            })
        })
    }

    #[napi]
    pub fn warmup(
        &self,
        options: String,
        callback: Function<'_, String, bool>,
    ) -> Result<Vec<String>> {
        boundary(|| {
            let mut guard = self.engine.try_lock().map_err(|_| invalid("Engine busy"))?;
            let engine = guard.as_mut().ok_or_else(|| invalid("Engine closed"))?;
            let options: ParseOptions = serde_json::from_str(&options).map_err(invalid)?;
            engine.warmup_with_host(
                &options,
                &mut |event| {
                    if callback.call(event_json(event)).map_err(invalid)? {
                        Ok(())
                    } else {
                        Err(invalid("Warmup cancelled by host"))
                    }
                },
                &mut |_| Ok(()),
            )
        })
    }

    #[napi]
    pub fn loaded_models(&self) -> Result<Vec<String>> {
        boundary(|| {
            let guard = self.engine.try_lock().map_err(|_| invalid("Engine busy"))?;
            Ok(guard
                .as_ref()
                .ok_or_else(|| invalid("Engine closed"))?
                .loaded_model_roles())
        })
    }
    #[napi]
    pub fn close(&self) -> Result<()> {
        boundary(|| {
            self.engine
                .try_lock()
                .map_err(|_| invalid("Engine busy"))?
                .take();
            Ok(())
        })
    }
}
