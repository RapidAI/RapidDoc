//! One CPU arena per environment; sessions retain their own weights and state.
use crate::error;
use ort::{
    environment::Environment,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    AsPointer,
};
use rapiddoc_core::{Error, ErrorCode};
use std::sync::atomic::{AtomicUsize, Ordering};

static USERS: AtomicUsize = AtomicUsize::new(0);

pub(crate) struct Lease;
impl Lease {
    // The environment lock serializes registration and acquisition.
    pub(crate) fn acquire(environment: &Environment, registered: &mut bool) -> Result<Self, Error> {
        if !*registered {
            register(environment)?;
            *registered = true;
        }
        USERS.fetch_add(1, Ordering::SeqCst);
        Ok(Self)
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        if USERS.fetch_sub(1, Ordering::SeqCst) != 1 {
            return;
        }
        let Ok(mut slot) = crate::ENVIRONMENT.lock() else {
            return;
        };
        // A new model may have acquired a lease before this lock was obtained.
        if USERS.load(Ordering::SeqCst) != 0 {
            return;
        }
        let Some((_, environment, registered)) = slot.as_mut() else {
            return;
        };
        if !*registered {
            return;
        }
        if let Ok(memory) = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Arena,
            MemoryType::Default,
        ) {
            // SAFETY: all sessions using this allocator have already dropped.
            let result = ort::ortsys![@ort: unsafe UnregisterAllocator(environment.ptr().cast_mut(),memory.ptr()) as Result];
            if result.is_ok() {
                *registered = false;
            }
        }
    }
}

/// Called under the environment lock, once. ORT owns the registered allocator;
/// MemoryInfo and ArenaCfg are copied by CreateAndRegisterAllocator.
pub(crate) fn register(environment: &Environment) -> Result<(), Error> {
    let memory = MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Arena,
        MemoryType::Default,
    )
    .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?;
    let mut config = std::ptr::null_mut();
    // SAFETY: the C API creates this configuration; the pointer is used only
    // until registration returns, then released exactly once on either result.
    ort::ortsys![@ort: unsafe CreateArenaCfg(0, 0, -1, -1, &mut config) as Result]
        .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?;
    let result = ort::ortsys![@ort: unsafe CreateAndRegisterAllocator(
        environment.ptr().cast_mut(), memory.ptr(), config
    ) as Result];
    unsafe {
        (ort::api().ReleaseArenaCfg)(config);
    }
    result.map_err(|e| error(ErrorCode::RuntimeUnavailable, e))
}
