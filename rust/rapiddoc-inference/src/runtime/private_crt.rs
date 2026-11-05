//! Optional package-private Windows CRT. No downloads or host installation changes.
//! The guard is deliberately thread-bound and never spans host progress callbacks.
use rapiddoc_core::Error;
use std::path::Path;

#[cfg(not(windows))]
pub(crate) fn activate(_: &Path) -> Result<(), Error> {
    Ok(())
}

#[cfg(windows)]
pub(crate) use windows::activate;

#[cfg(windows)]
mod windows {
    use super::*;
    use crate::error;
    use rapiddoc_core::ErrorCode;
    use std::{ffi::c_void, marker::PhantomData, os::windows::ffi::OsStrExt, ptr, rc::Rc};

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
    }

    pub(crate) struct Guard {
        handle: *mut c_void,
        cookie: usize,
        // Windows activation cookies must be popped on the activating thread.
        _thread: PhantomData<Rc<()>>,
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            unsafe {
                DeactivateActCtx(0, self.cookie);
                ReleaseActCtx(self.handle);
            }
        }
    }

    pub(crate) fn activate(runtime: &Path) -> Result<Option<Guard>, Error> {
        // Only an explicitly prepared private bundle opts in. External ORT paths
        // and ordinary pip installations retain their own dependency resolution.
        let Some(directory) = runtime.parent().filter(|p| {
            p.file_name()
                .is_some_and(|n| n == "RapidDoc.Runtime.PrivateCRT")
        }) else {
            return Ok(None);
        };
        let manifest = directory.join("rapiddoc-ort.manifest");
        let path: Vec<u16> = manifest.as_os_str().encode_wide().chain(Some(0)).collect();
        let context = ActCtx {
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
            let handle = CreateActCtxW(&context);
            if handle as isize == -1 {
                return Err(error(
                    ErrorCode::RuntimeUnavailable,
                    format!(
                        "Cannot create private ORT CRT context ({}): {}",
                        manifest.display(),
                        std::io::Error::last_os_error()
                    ),
                ));
            }
            let mut cookie = 0;
            if ActivateActCtx(handle, &mut cookie) == 0 {
                let cause = std::io::Error::last_os_error();
                ReleaseActCtx(handle);
                return Err(error(ErrorCode::RuntimeUnavailable, cause));
            }
            Ok(Some(Guard {
                handle,
                cookie,
                _thread: PhantomData,
            }))
        }
    }
}
