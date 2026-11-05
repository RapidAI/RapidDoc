//! Windows-only, single-thread diagnostic prototype. Not a release binding.
use std::{
    ffi::c_void,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};
#[repr(C)]
struct ActCtx {
    size: u32,
    flags: u32,
    source: *const u16,
    arch: u16,
    lang: u16,
    directory: *const u16,
    resource: *const u16,
    app: *const u16,
    module: *mut c_void,
}
#[link(name = "kernel32")]
extern "system" {
    fn CreateActCtxW(ctx: *const ActCtx) -> *mut c_void;
    fn ActivateActCtx(handle: *mut c_void, cookie: *mut usize) -> i32;
    fn DeactivateActCtx(flags: u32, cookie: usize) -> i32;
    fn ReleaseActCtx(handle: *mut c_void);
    fn GetLastError() -> u32;
    fn LoadLibraryW(path: *const u16) -> *mut c_void;

}
static HANDLE: AtomicUsize = AtomicUsize::new(0);
#[no_mangle]
pub extern "system" fn Java_OrtSxsProbe_activate(_: *mut c_void, _: *mut c_void) -> i64 {
    let manifest = std::env::var("RAPIDDOC_SXS_TEST_MANIFEST").unwrap();
    let path: Vec<u16> = manifest.encode_utf16().chain(Some(0)).collect();
    let ctx = ActCtx {
        size: std::mem::size_of::<ActCtx>() as u32,
        flags: 0,
        source: path.as_ptr(),
        arch: 0,
        lang: 0,
        directory: ptr::null(),
        resource: ptr::null(),
        app: ptr::null(),
        module: ptr::null_mut(),
    };
    unsafe {
        let handle = CreateActCtxW(&ctx);
        if handle as isize == -1 {
            return -(GetLastError() as i64);
        }
        let mut cookie = 0;
        if ActivateActCtx(handle, &mut cookie) == 0 {
            let error = GetLastError();
            ReleaseActCtx(handle);
            return -(error as i64);
        }
        HANDLE.store(handle as usize, Ordering::SeqCst);
        cookie as i64
    }
}
#[no_mangle]
pub extern "system" fn Java_OrtSxsProbe_deactivate(_: *mut c_void, _: *mut c_void, cookie: i64) {
    unsafe {
        DeactivateActCtx(0, cookie as usize);
        ReleaseActCtx(HANDLE.swap(0, Ordering::SeqCst) as *mut c_void);
    }
}

#[no_mangle]
pub extern "system" fn Java_OrtSxsProbe_loadOrt(env: *mut c_void, class: *mut c_void) -> i64 {
    let cookie = Java_OrtSxsProbe_activate(env, class);
    if cookie <= 0 {
        return cookie;
    }
    let name: Vec<u16> = "onnxruntime.dll".encode_utf16().chain(Some(0)).collect();
    let error = unsafe {
        let module = LoadLibraryW(name.as_ptr());
        if module.is_null() {
            GetLastError() as i64
        } else {
            0
        }
    };
    Java_OrtSxsProbe_deactivate(env, class, cookie);
    error
}
