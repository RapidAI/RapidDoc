//! Development-only allocator attribution; never linked into the Python wheel.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
struct Counter;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
fn add(n: usize) {
    let live = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(live, Relaxed);
    ALLOCATED.fetch_add(n, Relaxed);
}
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            add(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            add(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() {
            LIVE.fetch_sub(l.size(), Relaxed);
            add(n);
        }
        q
    }
}
#[global_allocator]
static A: Counter = Counter;
fn measure<T>(phase: &str, f: impl FnOnce() -> T) -> T {
    let baseline = LIVE.load(Relaxed);
    PEAK.store(baseline, Relaxed);
    ALLOCATED.store(0, Relaxed);
    let result = f();
    let live = LIVE.load(Relaxed);
    let peak = PEAK.load(Relaxed);
    let allocated = ALLOCATED.load(Relaxed);
    eprintln!(
        "{}",
        serde_json::json!({"phase":phase,"baseline":baseline,"live":live,"peak":peak,"allocated":allocated})
    );
    result
}
fn main() {
    let path = std::env::args().nth(1).expect("DOC path");
    let bytes = std::fs::read(path).unwrap();
    let mut ole = rapiddoc_office::ole::OleReader::open(bytes.into()).unwrap();
    let parsed = measure("body", || {
        rapiddoc_office::doc::body::convert(&mut ole).unwrap()
    });
    drop(ole);
    let output = measure("middle", || {
        rapiddoc_office::document_output::middle(parsed.pages, "0.9.11").unwrap()
    });
    let old = std::env::args().any(|v| v == "--value");
    measure("binding_metadata", || {
        let markdown = rapiddoc_office::document_output::render::export(
            &output.middle,
            "mm_markdown",
            "images",
            &Default::default(),
        );
        let content = rapiddoc_office::document_output::render::export(
            &output.middle,
            "content_list",
            "images",
            &Default::default(),
        );
        if old {
            serde_json::to_string(&serde_json::json!({"middle_json":output.middle,"markdown":markdown,"content_list_json":content,"source_page_count":output.middle.pdf_info.len(),"diagnostics":parsed.diagnostics})).unwrap()
        } else {
            #[derive(serde::Serialize)]
            struct Metadata<'a> {
                middle_json: &'a rapiddoc_office::document_output::Middle,
                markdown: &'a serde_json::Value,
                content_list_json: &'a serde_json::Value,
                source_page_count: usize,
                diagnostics: &'a [String],
            }
            serde_json::to_string(&Metadata {
                middle_json: &output.middle,
                markdown: &markdown,
                content_list_json: &content,
                source_page_count: output.middle.pdf_info.len(),
                diagnostics: &parsed.diagnostics,
            })
            .unwrap()
        }
    });
}
