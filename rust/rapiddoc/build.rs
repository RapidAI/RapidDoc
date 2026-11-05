// Optional build-time runtime delivery; never downloads from a parsing thread.
#[cfg(feature = "download-runtime")]
#[path = "build/runtime.rs"]
mod runtime;

fn main() {
    println!("cargo:rerun-if-changed=build/runtime.rs");
    println!("cargo:rerun-if-env-changed=RAPIDDOC_ORT_ARCHIVE");
    #[cfg(feature = "download-runtime")]
    runtime::install().unwrap_or_else(|e| panic!("ORT runtime delivery failed: {e}"));
}
