//! 多输入 FP32 会话，保留整型输出，供版面/表格模型使用。
use crate::{error, Model};
use ort::{
    session::SessionInputValue,
    value::{Tensor, TensorElementType, ValueType},
};
use rapiddoc_core::{Error, ErrorCode};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "dtype")]
pub enum Output {
    #[serde(rename = "uint8")]
    UInt8 { shape: Vec<i64>, values: Vec<u8> },
    #[serde(rename = "float32")]
    Float32 { shape: Vec<i64>, values: Vec<f32> },
    #[serde(rename = "int32")]
    Int32 { shape: Vec<i64>, values: Vec<i32> },
    #[serde(rename = "int64")]
    Int64 { shape: Vec<i64>, values: Vec<i64> },
}

// Views never outlive the ORT run; only public raw APIs materialize owned arrays.
#[derive(Clone, Copy)]
pub(crate) enum OutputRef<'a> {
    UInt8 { shape: &'a [i64], values: &'a [u8] },
    Float32 { shape: &'a [i64], values: &'a [f32] },
    Int32 { shape: &'a [i64], values: &'a [i32] },
    Int64 { shape: &'a [i64], values: &'a [i64] },
}
impl Output {
    pub(crate) fn as_ref(&self) -> OutputRef<'_> {
        match self {
            Self::UInt8 { shape, values } => OutputRef::UInt8 { shape, values },
            Self::Float32 { shape, values } => OutputRef::Float32 { shape, values },
            Self::Int32 { shape, values } => OutputRef::Int32 { shape, values },
            Self::Int64 { shape, values } => OutputRef::Int64 { shape, values },
        }
    }
}
impl OutputRef<'_> {
    pub(crate) fn into_owned(self) -> Output {
        match self {
            Self::UInt8 { shape, values } => Output::UInt8 {
                shape: shape.to_vec(),
                values: values.to_vec(),
            },
            Self::Float32 { shape, values } => Output::Float32 {
                shape: shape.to_vec(),
                values: values.to_vec(),
            },
            Self::Int32 { shape, values } => Output::Int32 {
                shape: shape.to_vec(),
                values: values.to_vec(),
            },
            Self::Int64 { shape, values } => Output::Int64 {
                shape: shape.to_vec(),
                values: values.to_vec(),
            },
        }
    }
}

impl Model {
    /// 输入名必须与模型完全一致；所有输入合计受预算约束，输出复制前检查类型和总量。
    pub fn run_named(
        &mut self,
        inputs: BTreeMap<String, Input>,
    ) -> Result<BTreeMap<String, Output>, Error> {
        self.run_named_with(inputs, |outputs| {
            Ok(outputs
                .into_iter()
                .map(|(name, value)| (name, value.into_owned()))
                .collect())
        })
    }

    pub(crate) fn run_named_with<T>(
        &mut self,
        inputs: BTreeMap<String, Input>,
        consume: impl FnOnce(BTreeMap<String, OutputRef<'_>>) -> Result<T, Error>,
    ) -> Result<T, Error> {
        if inputs.len() != self.session.inputs().len()
            || self
                .session
                .inputs()
                .iter()
                .any(|input| !inputs.contains_key(input.name()))
        {
            return Err(error(ErrorCode::InvalidInput, "命名输入集合与模型不一致"));
        }
        let mut total = 0usize;
        let mut tensors: Vec<(String, SessionInputValue<'static>)> =
            Vec::with_capacity(inputs.len());
        for (name, input) in inputs {
            let count = input
                .shape
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "命名输入形状溢出"))?;
            total = total
                .checked_add(count)
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "命名输入数量溢出"))?;
            if total > self.max_elements {
                return Err(error(ErrorCode::ResourceLimit, "命名输入合计超过预算"));
            }
            if input.shape.is_empty()
                || count == 0
                || count != input.values.len()
                || input.values.iter().any(|v| !v.is_finite())
            {
                return Err(error(ErrorCode::InvalidInput, "命名输入形状或数值无效"));
            }
            let tensor = Tensor::from_array((input.shape, input.values))
                .map_err(|e| error(ErrorCode::InvalidInput, e))?;
            tensors.push((name, tensor.into()));
        }
        let outputs = self
            .session
            .run(tensors)
            .map_err(|e| error(ErrorCode::InvalidInput, e))?;
        let mut result = BTreeMap::new();
        let mut total = 0usize;
        for (name, _) in outputs.iter() {
            let value = outputs
                .get(name)
                .expect("output name from the same session");
            let ValueType::Tensor { ty, shape, .. } = value.dtype() else {
                return Err(error(ErrorCode::Unsupported, "命名输出当前仅支持张量"));
            };
            let count = shape
                .iter()
                .try_fold(1usize, |n, d| {
                    usize::try_from(*d).ok().and_then(|d| n.checked_mul(d))
                })
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "命名输出形状无效或溢出"))?;
            total = total
                .checked_add(count)
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "命名输出数量溢出"))?;
            if total > self.max_elements {
                return Err(error(ErrorCode::ResourceLimit, "命名输出合计超过预算"));
            }
            let shape = &shape[..];
            let output = match ty {
                TensorElementType::Uint8 => {
                    let (_, v) = value
                        .try_extract_tensor::<u8>()
                        .map_err(|e| error(ErrorCode::InvalidInput, e))?;
                    OutputRef::UInt8 { shape, values: v }
                }
                TensorElementType::Float32 => {
                    let (_, v) = value
                        .try_extract_tensor::<f32>()
                        .map_err(|e| error(ErrorCode::InvalidInput, e))?;
                    if v.iter().any(|v| !v.is_finite()) {
                        return Err(error(ErrorCode::InvalidInput, "命名输出含非有限值"));
                    }
                    OutputRef::Float32 { shape, values: v }
                }
                TensorElementType::Int32 => {
                    let (_, v) = value
                        .try_extract_tensor::<i32>()
                        .map_err(|e| error(ErrorCode::InvalidInput, e))?;
                    OutputRef::Int32 { shape, values: v }
                }
                TensorElementType::Int64 => {
                    let (_, v) = value
                        .try_extract_tensor::<i64>()
                        .map_err(|e| error(ErrorCode::InvalidInput, e))?;
                    OutputRef::Int64 { shape, values: v }
                }
                _ => {
                    return Err(error(
                        ErrorCode::Unsupported,
                        "输出类型目前仅支持 float32/int32/int64/uint8",
                    ))
                }
            };
            result.insert(name.into(), output);
        }
        consume(result)
    }
}
