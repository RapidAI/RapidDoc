//! Lexically scoped reuse of immutable lopdf objects; never retains input bytes
//! or PDFium handles after the current native collection window.
use lopdf::Document;
use std::{cell::RefCell, marker::PhantomData, rc::Rc, sync::Arc};

struct State {
    address: usize,
    length: usize,
    document: Option<Result<Arc<Document>, ()>>,
}
thread_local! { static CURRENT: RefCell<Option<State>> = const { RefCell::new(None) }; }

/// Keeps the input immutably borrowed and stays on its creating thread. Nested
/// scopes restore their predecessor, including on error or panic unwinding.
struct Scope<'a> {
    previous: Option<State>,
    _input: PhantomData<&'a [u8]>,
    _thread: PhantomData<Rc<()>>,
}
impl Drop for Scope<'_> {
    fn drop(&mut self) {
        CURRENT.with(|current| *current.borrow_mut() = self.previous.take());
    }
}
fn scope(bytes: &[u8]) -> Scope<'_> {
    let previous = CURRENT.with(|current| {
        current.replace(Some(State {
            address: bytes.as_ptr() as usize,
            length: bytes.len(),
            document: None,
        }))
    });
    Scope {
        previous,
        _input: PhantomData,
        _thread: PhantomData,
    }
}
pub fn with_scope<T>(bytes: &[u8], call: impl FnOnce() -> T) -> T {
    let _scope = scope(bytes);
    call()
}
pub(crate) fn load(bytes: &[u8]) -> Result<Arc<Document>, ()> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if let Some(state) = current
            .as_mut()
            .filter(|state| state.address == bytes.as_ptr() as usize && state.length == bytes.len())
        {
            return state
                .document
                .get_or_insert_with(|| Document::load_mem(bytes).map(Arc::new).map_err(|_| ()))
                .clone();
        }
        Document::load_mem(bytes).map(Arc::new).map_err(|_| ())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source() -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }
    #[test]
    fn scoped_reuse_identity_nesting_and_release() {
        let bytes = source();
        let other = bytes.clone();
        let outer = scope(&bytes);
        let original = load(&bytes).unwrap();
        assert!(Arc::ptr_eq(&original, &load(&bytes).unwrap()));
        assert!(!Arc::ptr_eq(&original, &load(&other).unwrap()));
        {
            let _inner = scope(&other);
            let inner = load(&other).unwrap();
            assert!(Arc::ptr_eq(&inner, &load(&other).unwrap()));
            assert!(!Arc::ptr_eq(&original, &load(&bytes).unwrap()));
        }
        assert!(Arc::ptr_eq(&original, &load(&bytes).unwrap()));
        let weak = Arc::downgrade(&original);
        drop(original);
        drop(outer);
        assert!(weak.upgrade().is_none());
        assert!(CURRENT.with(|current| current.borrow().is_none()));
    }
    #[test]
    fn failure_and_unwind_do_not_keep_document_state() {
        with_scope(b"invalid PDF", || {
            assert!(load(b"invalid PDF").is_err());
            assert!(load(b"invalid PDF").is_err());
        });
        assert!(CURRENT.with(|current| current.borrow().is_none()));
        let bytes = source();
        let mut weak = None;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_scope(&bytes, || {
                weak = Some(Arc::downgrade(&load(&bytes).unwrap()));
                panic!("intentional scope unwind");
            })
        }));
        assert!(result.is_err());
        assert!(weak.unwrap().upgrade().is_none());
        assert!(CURRENT.with(|current| current.borrow().is_none()));
    }
}
