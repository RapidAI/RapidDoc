//! Resource policy shared by the production pipeline and low-level diagnostics.
//!
//! Python-compatible document calls do not impose invented document-size or
//! cumulative-work cutoffs. Explicit diagnostic limits and arithmetic/shape
//! validation remain separate. The scope is thread-local and restored on unwind.
use std::{cell::Cell, marker::PhantomData, rc::Rc};

thread_local! { static COMPATIBLE: Cell<bool> = const { Cell::new(false) }; }

pub struct Scope {
    previous: bool,
    _thread: PhantomData<Rc<()>>,
}

pub fn python_compatible() -> Scope {
    Scope {
        previous: COMPATIBLE.with(|v| v.replace(true)),
        _thread: PhantomData,
    }
}

#[inline]
pub fn is_python_compatible() -> bool {
    COMPATIBLE.with(Cell::get)
}

impl Drop for Scope {
    fn drop(&mut self) {
        COMPATIBLE.with(|v| v.set(self.previous));
    }
}

pub trait Bound: Copy + Default + PartialEq {
    const REPRESENTABLE_MAX: Self;
}
macro_rules! bounds {
    ($($t:ty),*) => {$ (impl Bound for $t { const REPRESENTABLE_MAX: Self = <$t>::MAX; })*};
}
bounds!(usize, u32, u64, i32, i64, f32, f64);

/// Apply an artificial diagnostic cap only outside production document calls.
/// This never replaces checked arithmetic, valid tensor shapes, or model rules.
#[inline]
pub fn limit<T: Bound>(diagnostic: T) -> T {
    if COMPATIBLE.with(Cell::get) && diagnostic != T::default() {
        T::REPRESENTABLE_MAX
    } else {
        diagnostic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatible_scope_restores_nested_panicking_and_other_thread_calls() {
        assert_eq!(limit(256_000_000usize), 256_000_000);
        {
            let _outer = python_compatible();
            assert_eq!(limit(256_000_000usize), usize::MAX);
            assert_eq!(limit(0usize), 0); // A zero capacity never becomes an allocation request.
            {
                let _inner = python_compatible();
            }
            assert_eq!(limit(1000usize), usize::MAX);
            assert_eq!(
                std::thread::spawn(|| limit(1000usize)).join().unwrap(),
                1000
            );
        }
        let _ = std::panic::catch_unwind(|| {
            let _scope = python_compatible();
            panic!("deliberate scope unwind");
        });
        assert_eq!(limit(1000usize), 1000);
    }
}
