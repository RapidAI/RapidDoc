//! Original Magika standard_v3_3 code-language path, using the shared ORT ABI.
//! Feature extraction and high-confidence selection follow Google's Apache-2.0
//! Python implementation; source/model notices are in assets/magika/.
use crate::{error, ENVIRONMENT};
use ort::{environment::Environment, session::Session, value::Tensor};
use rapiddoc_core::{Error, ErrorCode};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{LazyLock, Mutex},
};

#[derive(Deserialize)]
struct Config {
    beg_size: usize,
    end_size: usize,
    block_size: usize,
    padding_token: u16,
    min_file_size_for_dl: usize,
    medium_confidence_threshold: f64,
    target_labels_space: Vec<String>,
    thresholds: BTreeMap<String, f64>,
    overwrite_map: BTreeMap<String, String>,
}
static CONFIG: LazyLock<Config> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../assets/magika/config.min.json"))
        .expect("embedded original Magika configuration")
});
static SESSION: Mutex<Option<Session>> = Mutex::new(None);

fn environment(runtime: Option<&Path>) -> Result<Environment, Error> {
    let selected = runtime
        .map(Path::canonicalize)
        .transpose()
        .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?;
    let mut state = ENVIRONMENT.lock().map_err(|_| {
        error(
            ErrorCode::RuntimeUnavailable,
            "ORT environment lock poisoned",
        )
    })?;
    if let Some((path, environment, _)) = state.as_ref() {
        if selected.as_ref().is_some_and(|requested| requested != path) {
            return Err(error(
                ErrorCode::RuntimeUnavailable,
                "Cannot switch ORT library in one process",
            ));
        }
        return Ok(environment.clone());
    }
    let path = selected.ok_or_else(|| {
        error(
            ErrorCode::RuntimeUnavailable,
            "ORT runtime is not initialized",
        )
    })?;
    let _crt = crate::private_crt::activate(&path)?;
    let environment = ort::init_from(&path)
        .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?
        .with_telemetry(false)
        .build()
        .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?;
    *state = Some((path, environment.clone(), false));
    Ok(environment)
}

fn byte_space(value: u8) -> bool {
    matches!(value, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// Fixed prefix/suffix features; do not Unicode-trim or scan the whole input.
pub fn features(content: &[u8]) -> Vec<i32> {
    let config = &*CONFIG;
    let count = content.len().min(config.block_size);
    let prefix = &content[..count];
    let begin = prefix
        .iter()
        .position(|&c| !byte_space(c))
        .unwrap_or(prefix.len());
    let prefix = &prefix[begin..];
    let suffix = &content[content.len() - count..];
    let end = suffix
        .iter()
        .rposition(|&c| !byte_space(c))
        .map_or(0, |index| index + 1);
    let suffix = &suffix[..end];
    let mut values = vec![i32::from(config.padding_token); config.beg_size + config.end_size];
    for (index, &value) in prefix.iter().take(config.beg_size).enumerate() {
        values[index] = i32::from(value);
    }
    let suffix = &suffix[suffix.len().saturating_sub(config.end_size)..];
    let offset = values.len() - suffix.len();
    for (index, &value) in suffix.iter().enumerate() {
        values[offset + index] = i32::from(value);
    }
    values
}

/// Match guess_language_by_text, including unknown/low-confidence -> txt.
/// Runtime errors are returned so diagnostics can distinguish actual inference
/// from the original public wrapper's txt fallback.
pub fn language(code: &str, runtime: Option<&Path>) -> Result<String, Error> {
    let config = &*CONFIG;
    if code.is_empty() || code.len() < config.min_file_size_for_dl {
        return Ok("txt".into());
    }
    let values = features(code.as_bytes());
    if values[config.min_file_size_for_dl - 1] == i32::from(config.padding_token) {
        return Ok("txt".into());
    }
    let env = environment(runtime)?;
    // Session/provider initialization is also scoped, including when another
    // engine initialized the environment on a different thread.
    let runtime_path = ENVIRONMENT
        .lock()
        .map_err(|_| {
            error(
                ErrorCode::RuntimeUnavailable,
                "ORT environment lock poisoned",
            )
        })?
        .as_ref()
        .map(|(path, _, _)| path.clone());
    let _crt = runtime_path
        .as_deref()
        .map(crate::private_crt::activate)
        .transpose()?;
    let mut slot = SESSION.lock().map_err(|_| {
        error(
            ErrorCode::RuntimeUnavailable,
            "Magika session lock poisoned",
        )
    })?;
    if slot.is_none() {
        let session = Session::builder(&env)
            .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?
            .with_execution_providers([ort::ep::CPU::default().build().error_on_failure()])
            .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?
            .commit_from_memory(include_bytes!("../assets/magika/model.onnx"))
            .map_err(|e| error(ErrorCode::RuntimeUnavailable, e))?;
        *slot = Some(session);
    }
    let tensor = Tensor::from_array((vec![1usize, values.len()], values))
        .map_err(|e| error(ErrorCode::InvalidInput, e))?;
    let outputs = slot
        .as_mut()
        .unwrap()
        .run(ort::inputs!["bytes" => tensor])
        .map_err(|e| error(ErrorCode::InvalidInput, e))?;
    let (_, scores) = outputs["target_label"]
        .try_extract_tensor::<f32>()
        .map_err(|e| error(ErrorCode::InvalidInput, e))?;
    if scores.len() != config.target_labels_space.len() || scores.iter().any(|v| !v.is_finite()) {
        return Err(error(ErrorCode::InvalidInput, "Unexpected Magika output"));
    }
    // Python max(range(...), key=...) keeps the first equal maximum.
    let best = (1..scores.len()).fold(0, |best, index| {
        if scores[index] > scores[best] {
            index
        } else {
            best
        }
    });
    let label = &config.target_labels_space[best];
    let threshold = config
        .thresholds
        .get(label)
        .copied()
        .unwrap_or(config.medium_confidence_threshold);
    if f64::from(scores[best]) < threshold {
        return Ok("txt".into());
    }
    let output = config.overwrite_map.get(label).unwrap_or(label);
    Ok(if output == "unknown" {
        "txt".into()
    } else {
        output.clone()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_byte_whitespace_padding_and_end_window() {
        let values = features(b" \tABC\r\n");
        assert_eq!(&values[..5], &[65, 66, 67, 13, 10]);
        assert_eq!(&values[2045..], &[65, 66, 67]);
        assert_eq!(values[1023], 256);
        assert_eq!(values[1024], 256);
        assert_eq!(features(&vec![b' '; 5000]), vec![256; 2048]);
        assert_eq!(language("", None).unwrap(), "txt");
        assert_eq!(language("short", None).unwrap(), "txt");
    }
}
