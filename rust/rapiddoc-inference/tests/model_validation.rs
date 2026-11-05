use rapiddoc_core::ErrorCode;
use rapiddoc_inference::{verify_model, Device, Model};
use std::path::Path;

#[test]
fn requested_operator_library_is_not_silently_ignored() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let descriptor: rapiddoc_inference::registry::Descriptor =
        serde_json::from_value(serde_json::json!({
            "runtime": root.join("Cargo.toml"), "model": root.join("Cargo.toml"),
            "sha256": "0".repeat(64),
            "operator_libraries": [root.join("missing-operators-library")]
        }))
        .unwrap();
    let error = rapiddoc_inference::registry::cached(&descriptor)
        .err()
        .expect("missing library must fail");
    assert_eq!(error.code, ErrorCode::RuntimeUnavailable);
}

#[test]
fn invalid_hash_is_rejected_before_io() {
    assert_eq!(
        verify_model(Path::new("missing.onnx"), "invalid")
            .expect_err("bad hash")
            .code,
        ErrorCode::InvalidInput
    );
    assert_eq!(
        verify_model(Path::new("missing.onnx"), &"0".repeat(64))
            .expect_err("missing model")
            .code,
        ErrorCode::ModelMissing
    );
}

#[test]
fn invalid_session_limits_fail_without_loading_runtime() {
    let result = Model::load(
        Path::new("missing.dll"),
        Path::new("missing.onnx"),
        &"0".repeat(64),
        Device::Cpu,
        0,
        0,
    );
    assert_eq!(
        result.err().expect("zero tensor capacity").code,
        ErrorCode::InvalidInput
    );
}

#[test]
fn zero_threads_preserves_automatic_thread_selection() {
    let result = Model::load(
        Path::new("missing.dll"),
        Path::new("missing.onnx"),
        &"0".repeat(64),
        Device::Cpu,
        0,
        10,
    );
    assert_eq!(
        result
            .err()
            .expect("model IO reached with automatic threads")
            .code,
        ErrorCode::ModelMissing
    );
}

#[test]
#[cfg(not(feature = "cuda"))]
fn cpu_build_does_not_silently_accept_cuda() {
    let result = Model::load(
        Path::new("missing.dll"),
        Path::new("missing.onnx"),
        &"0".repeat(64),
        Device::Cuda,
        1,
        10,
    );
    assert_eq!(
        result.err().expect("unsupported device").code,
        ErrorCode::DeviceUnavailable
    );
}
