//! SLANet's original CUDA encoder / CPU recurrent decoder, owned by Rust.
use crate::{error, Model, TensorOutput};
use ort::session::SessionInputValue;
use rapiddoc_core::{Error, ErrorCode};
use std::collections::BTreeMap;

impl Model {
    /// Raw table output for differential probes; unpartitioned models retain run().
    pub fn run_table(
        &mut self,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<BTreeMap<String, TensorOutput>, Error> {
        let Some(decoder) = self.table_decoder.as_mut() else {
            return self.run(shape, values);
        };
        let names: Vec<String> = decoder
            .inputs()
            .iter()
            .map(|v| v.name().to_owned())
            .collect();
        let maximum = self.max_elements;
        Self::run_f32(&mut self.session, maximum, shape, values, |encoded| {
            // Borrow the encoder's CPU OrtValues directly. No Python callback,
            // JSON, float-vector copy or second GPU transfer at the boundary.
            let inputs = names
                .iter()
                .map(|name| {
                    encoded
                        .get(name)
                        .map(|v| (name.clone(), SessionInputValue::from(v)))
                        .ok_or_else(|| {
                            error(ErrorCode::InvalidInput, "Missing SLANet encoded tensor")
                        })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let outputs = decoder
                .run(inputs)
                .map_err(|e| error(ErrorCode::InvalidInput, e))?;
            let mut result = BTreeMap::new();
            let mut total = 0usize;
            for (name, value) in outputs.iter() {
                let (shape, values) = value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| error(ErrorCode::Unsupported, e))?;
                total = total
                    .checked_add(values.len())
                    .filter(|n| *n <= maximum)
                    .ok_or_else(|| {
                        error(
                            ErrorCode::ResourceLimit,
                            "SLANet decoder output exceeds budget",
                        )
                    })?;
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(error(
                        ErrorCode::InvalidInput,
                        "Nonfinite SLANet decoder output",
                    ));
                }
                result.insert(
                    name.into(),
                    TensorOutput {
                        shape: shape.iter().copied().collect(),
                        values: values.to_vec(),
                    },
                );
            }
            Ok(result)
        })
    }
}
