//! Ownership and panic boundary only; document processing stays in the shared SDK.
use rapiddoc::{RapidDoc, RapidDocConfig, RuntimeResources, ParseOptions, RapidDocDocument};
use rapiddoc_core::{Error, ErrorCode};
use serde_json::json;
#[cfg(feature = "mimalloc-allocator")]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;
use std::{
    collections::HashMap,
    ffi::c_void,
    sync::{
        Arc, Mutex, LazyLock,
        atomic::{AtomicU64, Ordering},
    },
    panic::{catch_unwind, AssertUnwindSafe},
};

#[repr(C)]
pub struct Bytes {
    data: *const u8,
    len: usize,
}
impl Bytes {
    fn of(data: &[u8]) -> Self {
        Self {
            data: data.as_ptr(),
            len: data.len(),
        }
    }
}
#[repr(C)]
pub struct Asset {
    name: Bytes,
    data: Bytes,
    kind: u32,
}
#[repr(C)]
pub struct Response {
    status: i32,
    json: Bytes,
    assets: *const Asset,
    asset_count: usize,
    owner: *mut c_void,
}
struct Payload {
    json: Vec<u8>,
    names: Vec<String>,
    assets: Vec<Asset>,
    document: Option<RapidDocDocument>,
}
type Callback = Option<unsafe extern "C" fn(*const u8, usize, *mut c_void) -> i32>;
static ENGINES: LazyLock<Mutex<HashMap<u64, Arc<RapidDoc>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT: AtomicU64 = AtomicU64::new(1);
fn invalid(message: impl ToString) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.to_string(),
    }
}

fn response(status: i32, data: Vec<u8>, doc: Option<RapidDocDocument>) -> *mut Response {
    let mut payload = Box::new(Payload {
        json: data,
        names: vec![],
        assets: vec![],
        document: doc,
    });
    if let Some(doc) = payload.document.as_ref() {
        payload.names.extend(doc.images.keys().cloned());
        payload.names.extend(doc.drawings.keys().cloned());
        for (index, bytes) in doc
            .images
            .values()
            .map(|v| (0, v.as_ref()))
            .chain(doc.drawings.values().map(|v| (1, v.as_slice())))
            .enumerate()
        {
            payload.assets.push(Asset {
                name: Bytes::of(payload.names[index].as_bytes()),
                data: Bytes::of(bytes.1),
                kind: bytes.0,
            });
        }
    }
    let result = Box::new(Response {
        status,
        json: Bytes::of(&payload.json),
        assets: payload.assets.as_ptr(),
        asset_count: payload.assets.len(),
        owner: Box::into_raw(payload).cast(),
    });
    Box::into_raw(result)
}
unsafe fn bytes<'a>(data: *const u8, len: usize) -> Result<&'a [u8], Error> {
    if len == 0 {
        return Ok(&[]);
    }
    if data.is_null() || len > isize::MAX as usize {
        return Err(invalid("Invalid input buffer"));
    }
    Ok(std::slice::from_raw_parts(data, len))
}
fn text(data: &[u8]) -> Result<&str, Error> {
    std::str::from_utf8(data).map_err(invalid)
}
fn options(data: &[u8]) -> &[u8] {
    if data.is_empty() {
        b"{}"
    } else {
        data
    }
}
#[no_mangle]
pub extern "C" fn rd_abi_version() -> u32 {
    1
}

#[no_mangle]
pub unsafe extern "C" fn rd_invoke(
    handle: u64,
    operation: u32,
    input: *const u8,
    input_len: usize,
    option: *const u8,
    option_len: usize,
    callback: Callback,
    context: *mut c_void,
) -> *mut Response {
    let call = catch_unwind(AssertUnwindSafe(|| -> Result<*mut Response, Error> {
        let input = bytes(input, input_len)?;
        let option = bytes(option, option_len)?;
        if operation == 0 {
            let config: RapidDocConfig = serde_json::from_slice(options(input)).map_err(invalid)?;
            let resources: RuntimeResources =
                serde_json::from_slice(options(option)).map_err(invalid)?;
            let id = NEXT
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .map_err(|_| invalid("Engine handle space exhausted"))?;
            ENGINES
                .lock()
                .map_err(invalid)?
                .insert(id, Arc::new(RapidDoc::new(config, resources)));
            return Ok(response(
                0,
                json!({"handle":id}).to_string().into_bytes(),
                None,
            ));
        }
        if operation == 4 {
            ENGINES.lock().map_err(invalid)?.remove(&handle);
            return Ok(response(0, b"{}".to_vec(), None));
        }
        let engine = ENGINES
            .lock()
            .map_err(invalid)?
            .get(&handle)
            .cloned()
            .ok_or_else(|| invalid("Engine closed or invalid handle"))?;
        let mut progress = |event: rapiddoc::sdk::Progress<'_>| {
            if let Some(callback) = callback {
                let data = json!({"stage":event.stage,"page_index":event.page_index,"completed":event.completed,"total":event.total,"elapsed_seconds":event.elapsed_seconds,"detail":event.detail}).to_string();
                if callback(data.as_ptr(), data.len(), context) != 0 {
                    return Err(invalid("Progress callback cancelled parsing"));
                }
            }
            Ok(())
        };
        if operation == 5 {
            return Ok(response(
                0,
                serde_json::to_vec(&engine.loaded_model_roles()).map_err(invalid)?,
                None,
            ));
        }
        let opts: ParseOptions = serde_json::from_slice(options(option)).map_err(invalid)?;
        if operation == 3 {
            let roles = engine.warmup_with_host(&opts, &mut progress, &mut |_| Ok(()))?;
            return Ok(response(
                0,
                serde_json::to_vec(&roles).map_err(invalid)?,
                None,
            ));
        }
        let doc = match operation {
            1 => engine.parse_with_progress(text(input)?, &opts, &mut progress)?,
            2 => engine.parse_bytes(input, &opts, &mut progress)?,
            _ => return Err(invalid("Unknown C ABI operation")),
        };
        Ok(response(
            0,
            serde_json::to_vec(&doc).map_err(invalid)?,
            Some(doc),
        ))
    }));
    match call {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => response(
            1,
            json!({"code":error.code.as_str(),"message":error.message})
                .to_string()
                .into_bytes(),
            None,
        ),
        Err(_) => response(
            1,
            json!({"code":"native_panic","message":"Rust panic at C ABI boundary"})
                .to_string()
                .into_bytes(),
            None,
        ),
    }
}
#[no_mangle]
pub unsafe extern "C" fn rd_response_free(response: *mut Response) {
    if !response.is_null() {
        let response = Box::from_raw(response);
        drop(Box::from_raw(response.owner as *mut Payload));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    unsafe fn read(value: *mut Response) -> (i32, serde_json::Value) {
        let status = (*value).status;
        let data =
            serde_json::from_slice(bytes((*value).json.data, (*value).json.len).unwrap()).unwrap();
        rd_response_free(value);
        (status, data)
    }
    #[test]
    fn header_layout_and_invalid_buffers() {
        assert_eq!(std::mem::size_of::<Bytes>(), 16);
        assert_eq!(std::mem::size_of::<Asset>(), 40);
        assert_eq!(std::mem::size_of::<Response>(), 48);
        unsafe {
            let (status, error) = read(rd_invoke(
                0,
                0,
                std::ptr::null(),
                1,
                std::ptr::null(),
                0,
                None,
                std::ptr::null_mut(),
            ));
            assert_eq!(status, 1);
            assert_eq!(error["code"], "invalid_input");
            rd_response_free(std::ptr::null_mut());
        }
    }
    #[test]
    fn create_close_and_stale_handle_return_owned_errors() {
        unsafe {
            let (status, value) = read(rd_invoke(
                0,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                None,
                std::ptr::null_mut(),
            ));
            assert_eq!(status, 0);
            let id = value["handle"].as_u64().unwrap();
            for _ in 0..2 {
                assert_eq!(
                    read(rd_invoke(
                        id,
                        4,
                        std::ptr::null(),
                        0,
                        std::ptr::null(),
                        0,
                        None,
                        std::ptr::null_mut()
                    ))
                    .0,
                    0
                );
            }
            let (status, error) = read(rd_invoke(
                id,
                5,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                None,
                std::ptr::null_mut(),
            ));
            assert_eq!(status, 1);
            assert_eq!(error["code"], "invalid_input");
        }
    }
}
