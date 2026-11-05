//! JNI owns handles and JVM values only; all parsing belongs to the Rust SDK.
use jni::{
    objects::{JByteArray, JClass, JObject, JString, JThrowable, JValue},
    sys::{jlong, jobject},
    JNIEnv,
};
use rapiddoc::{sdk::Progress, ParseOptions, RapidDoc, RapidDocConfig, RuntimeResources};
use rapiddoc_core::{Error, ErrorCode};
use std::{
    collections::HashMap,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicI64, Ordering},
        Arc, LazyLock, Mutex,
    },
};

#[cfg(feature = "mimalloc-allocator")]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

type Engine = Arc<RapidDoc>;
static ENGINES: LazyLock<Mutex<HashMap<i64, Engine>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT: AtomicI64 = AtomicI64::new(1);
const PACKAGE: &str = "io/github/rapidai/rapiddoc";

fn invalid(message: impl ToString) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.to_string(),
    }
}
fn string(env: &mut JNIEnv<'_>, value: JString<'_>) -> Result<String, Error> {
    env.get_string(&value).map(Into::into).map_err(invalid)
}
fn engine(handle: jlong) -> Result<Engine, Error> {
    ENGINES
        .lock()
        .map_err(invalid)?
        .get(&handle)
        .cloned()
        .ok_or_else(|| invalid("RapidDoc is closed or its handle is invalid"))
}
fn with_engine<T>(
    handle: jlong,
    f: impl FnOnce(&RapidDoc) -> Result<T, Error>,
) -> Result<T, Error> {
    let engine = engine(handle)?;
    f(&engine)
}
fn throw(env: &mut JNIEnv<'_>, code: &str, message: &str) {
    // Preserve an exception thrown by the user's progress callback.
    if env.exception_check().unwrap_or(true) {
        return;
    }
    let result = (|| -> jni::errors::Result<()> {
        let code = env.new_string(code)?;
        let message = env.new_string(message)?;
        let value = env.new_object(
            format!("{PACKAGE}/RapidDocException"),
            "(Ljava/lang/String;Ljava/lang/String;)V",
            &[
                JValue::Object(code.as_ref()),
                JValue::Object(message.as_ref()),
            ],
        )?;
        env.throw(JThrowable::from(value))
    })();
    if result.is_err() && !env.exception_check().unwrap_or(true) {
        let _ = env.throw_new("java/lang/IllegalStateException", message);
    }
}
fn boundary<T: Default>(
    env: &mut JNIEnv<'_>,
    call: impl FnOnce(&mut JNIEnv<'_>) -> Result<T, Error>,
) -> T {
    match catch_unwind(AssertUnwindSafe(|| call(env))) {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => {
            throw(env, error.code.as_str(), &error.message);
            T::default()
        }
        Err(panic) => {
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("Rust panic");
            throw(env, "native_panic", message);
            T::default()
        }
    }
}

fn notify(env: &mut JNIEnv<'_>, listener: &JObject<'_>, event: Progress<'_>) -> Result<(), Error> {
    if listener.is_null() {
        return Ok(());
    }
    // Progress can run for thousands of pages without accumulating local refs.
    env.with_local_frame(8, |env| -> jni::errors::Result<()> {
        let stage = env.new_string(event.stage)?;
        let detail = match event.detail {
            Some(value) => JObject::from(env.new_string(value)?),
            None => JObject::null(),
        };
        let number = |v: Option<usize>| v.map_or(-1, |n| n as jlong);
        let value = env.new_object(
            format!("{PACKAGE}/ProgressEvent"),
            "(Ljava/lang/String;JJJDLjava/lang/String;)V",
            &[
                JValue::Object(stage.as_ref()),
                JValue::Long(number(event.page_index)),
                JValue::Long(number(event.completed)),
                JValue::Long(number(event.total)),
                JValue::Double(event.elapsed_seconds.unwrap_or(-1.0)),
                JValue::Object(&detail),
            ],
        )?;
        env.call_method(
            listener,
            "onProgress",
            "(Lio/github/rapidai/rapiddoc/ProgressEvent;)V",
            &[JValue::Object(&value)],
        )?;
        Ok(())
    })
    .map_err(invalid)
}
fn asset_map<'a, 'b>(
    env: &mut JNIEnv<'a>,
    items: impl Iterator<Item = (&'b str, &'b [u8])>,
) -> Result<JObject<'a>, Error> {
    let map = env
        .new_object("java/util/LinkedHashMap", "()V", &[])
        .map_err(invalid)?;
    for (name, data) in items {
        if data.len() > i32::MAX as usize {
            return Err(invalid("Asset exceeds JVM byte array size"));
        }
        env.with_local_frame(8, |env| -> jni::errors::Result<()> {
            let name = env.new_string(name)?;
            let bytes = env.byte_array_from_slice(data)?;
            env.call_method(
                &map,
                "put",
                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                &[
                    JValue::Object(name.as_ref()),
                    JValue::Object(bytes.as_ref()),
                ],
            )?;
            Ok(())
        })
        .map_err(invalid)?;
    }
    Ok(map)
}
fn result(env: &mut JNIEnv<'_>, doc: rapiddoc::RapidDocDocument) -> Result<jobject, Error> {
    let metadata = serde_json::to_string(&doc).map_err(invalid)?;
    let metadata = env.new_string(metadata).map_err(invalid)?;
    let images = asset_map(
        env,
        doc.images.iter().map(|(k, v)| (k.as_str(), v.as_ref())),
    )?;
    let drawings = asset_map(
        env,
        doc.drawings.iter().map(|(k, v)| (k.as_str(), v.as_slice())),
    )?;
    env.new_object(
        format!("{PACKAGE}/NativeBindings$Result"),
        "(Ljava/lang/String;Ljava/util/Map;Ljava/util/Map;)V",
        &[
            JValue::Object(metadata.as_ref()),
            JValue::Object(&images),
            JValue::Object(&drawings),
        ],
    )
    .map(JObject::into_raw)
    .map_err(invalid)
}

#[no_mangle]
pub extern "system" fn Java_io_github_rapidai_rapiddoc_NativeBindings_create(
    mut env: JNIEnv,
    _: JClass,
    config: JString,
    resources: JString,
) -> jlong {
    boundary(&mut env, |env| {
        let config: RapidDocConfig =
            serde_json::from_str(&string(env, config)?).map_err(invalid)?;
        let resources: RuntimeResources =
            serde_json::from_str(&string(env, resources)?).map_err(invalid)?;
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        if id <= 0 {
            return Err(invalid("Native handle space exhausted"));
        }
        ENGINES
            .lock()
            .map_err(invalid)?
            .insert(id, Arc::new(RapidDoc::new(config, resources)));
        Ok(id)
    })
}
#[no_mangle]
pub extern "system" fn Java_io_github_rapidai_rapiddoc_NativeBindings_close(
    mut env: JNIEnv,
    _: JClass,
    handle: jlong,
) {
    boundary(&mut env, |_| {
        ENGINES.lock().map_err(invalid)?.remove(&handle);
        Ok(())
    });
}
#[no_mangle]
pub extern "system" fn Java_io_github_rapidai_rapiddoc_NativeBindings_parsePath(
    mut env: JNIEnv,
    _: JClass,
    handle: jlong,
    path: JString,
    options: JString,
    listener: JObject,
) -> jobject {
    boundary(&mut env, |env| {
        let path = string(env, path)?;
        let options: ParseOptions =
            serde_json::from_str(&string(env, options)?).map_err(invalid)?;
        let doc = with_engine(handle, |engine| {
            engine.parse_with_progress(path, &options, &mut |event| notify(env, &listener, event))
        })?;
        result(env, doc)
    })
}
#[no_mangle]
pub extern "system" fn Java_io_github_rapidai_rapiddoc_NativeBindings_parseBytes(
    mut env: JNIEnv,
    _: JClass,
    handle: jlong,
    data: JByteArray,
    options: JString,
    listener: JObject,
) -> jobject {
    boundary(&mut env, |env| {
        let data = env.convert_byte_array(data).map_err(invalid)?;
        let options: ParseOptions =
            serde_json::from_str(&string(env, options)?).map_err(invalid)?;
        let doc = with_engine(handle, |engine| {
            engine.parse_bytes(&data, &options, &mut |event| notify(env, &listener, event))
        })?;
        result(env, doc)
    })
}
#[no_mangle]
pub extern "system" fn Java_io_github_rapidai_rapiddoc_NativeBindings_warmup(
    mut env: JNIEnv,
    _: JClass,
    handle: jlong,
    options: JString,
    listener: JObject,
) {
    boundary(&mut env, |env| {
        let options: ParseOptions =
            serde_json::from_str(&string(env, options)?).map_err(invalid)?;
        with_engine(handle, |engine| {
            engine.warmup_with_host(
                &options,
                &mut |event| notify(env, &listener, event),
                &mut |_| Ok(()),
            )
        })?;
        Ok(())
    });
}
#[no_mangle]
pub extern "system" fn Java_io_github_rapidai_rapiddoc_NativeBindings_loadedModels(
    mut env: JNIEnv,
    _: JClass,
    handle: jlong,
) -> jobject {
    boundary(&mut env, |env| {
        let roles = with_engine(handle, |engine| Ok(engine.loaded_model_roles()))?;
        env.new_string(serde_json::to_string(&roles).map_err(invalid)?)
            .map(|s| s.into_raw())
            .map_err(invalid)
    })
}
