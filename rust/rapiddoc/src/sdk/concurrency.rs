//! Reject synchronous callback reentry without serializing independent threads.
use super::{invalid, Error};
use std::{cell::RefCell, collections::BTreeSet, marker::PhantomData, rc::Rc, sync::Arc};

thread_local! {
    static ACTIVE: RefCell<BTreeSet<usize>> = const { RefCell::new(BTreeSet::new()) };
}

pub(super) struct Request(usize, PhantomData<Rc<()>>);
impl Request {
    pub(super) fn enter(identity: &Arc<()>) -> Result<Self, Error> {
        let key = Arc::as_ptr(identity) as usize;
        if !ACTIVE.with(|active| active.borrow_mut().insert(key)) {
            return Err(invalid(
                "RapidDoc is already in use on this thread; callback reentry is not supported",
            ));
        }
        Ok(Self(key, PhantomData))
    }
}
impl Drop for Request {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.borrow_mut().remove(&self.0));
    }
}
