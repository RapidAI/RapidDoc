//! CTC 贪心解码；blank 隔开重复字符，概率相同时保留首个 token。
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;

/// 识别结果保留 token 和原时间步，便于定位字符差异。
#[derive(Debug, Serialize)]
pub struct Recognition {
    pub text: String,
    pub confidence: f64,
    pub token_indices: Vec<usize>,
    pub timesteps: Vec<usize>,
}

/// 解码 batch/time/classes 张量，alphabet 的索引 0 必须为 blank。
pub fn decode(
    batch: usize,
    steps: usize,
    classes: usize,
    values: &[f32],
    alphabet: &[String],
) -> Result<Vec<Recognition>, Error> {
    decode_impl(batch, steps, classes, values, alphabet, false)
}

// Only the native model runner may skip the finite scan: it has checked every output.
pub(crate) fn decode_finite(
    batch: usize,
    steps: usize,
    classes: usize,
    values: &[f32],
    alphabet: &[String],
) -> Result<Vec<Recognition>, Error> {
    decode_impl(batch, steps, classes, values, alphabet, true)
}

fn decode_impl(
    batch: usize,
    steps: usize,
    classes: usize,
    values: &[f32],
    alphabet: &[String],
    finite: bool,
) -> Result<Vec<Recognition>, Error> {
    let count = batch
        .checked_mul(steps)
        .and_then(|v| v.checked_mul(classes))
        .ok_or_else(|| invalid("CTC 尺寸溢出"))?;
    if batch == 0
        || classes == 0
        || alphabet.len() != classes
        || count != values.len()
        || (!finite && values.iter().any(|v| !v.is_finite()))
    {
        return Err(invalid("CTC 形状、字典或数值不匹配"));
    }
    decode_selected(batch, steps, alphabet, |item, step| {
        let start = (item * steps + step) * classes;
        let row = &values[start..start + classes];
        let mut best = 0;
        for index in 1..classes {
            if row[index] > row[best] {
                best = index;
            }
        }
        (best, row[best])
    })
}

/// Decode GPU-reduced ArgMax/ReduceMax output using the same CTC and confidence
/// reduction as the full probability tensor. No threshold or precision changes.
pub fn decode_compact(
    batch: usize,
    steps: usize,
    indices: &[i64],
    scores: &[f32],
    alphabet: &[String],
) -> Result<Vec<Recognition>, Error> {
    let count = batch
        .checked_mul(steps)
        .ok_or_else(|| invalid("CTC dimensions overflow"))?;
    if batch == 0
        || alphabet.is_empty()
        || indices.len() != count
        || scores.len() != count
        || indices
            .iter()
            .any(|i| *i < 0 || *i as usize >= alphabet.len())
        || scores.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("Invalid compact CTC outputs"));
    }
    decode_selected(batch, steps, alphabet, |item, step| {
        let i = item * steps + step;
        (indices[i] as usize, scores[i])
    })
}

fn decode_selected(
    batch: usize,
    steps: usize,
    alphabet: &[String],
    selected: impl Fn(usize, usize) -> (usize, f32),
) -> Result<Vec<Recognition>, Error> {
    let mut results = Vec::with_capacity(batch);
    for item in 0..batch {
        let mut text = String::new();
        let mut previous = None;
        let mut scores = Vec::new();
        let mut tokens = Vec::new();
        let mut timesteps = Vec::new();
        for step in 0..steps {
            let (best, probability) = selected(item, step);
            if best != 0 && previous != Some(best) {
                text.push_str(&alphabet[best]);
                tokens.push(best);
                timesteps.push(step);
                scores.push(round5(probability as f64));
            }
            previous = Some(best);
        }
        let confidence = if scores.is_empty() {
            0.
        } else {
            round5(pairwise(&scores) / scores.len() as f64)
        };
        results.push(Recognition {
            text,
            confidence,
            token_indices: tokens,
            timesteps,
        });
    }
    Ok(results)
}

// NumPy 2.4 的连续 FP64 pairwise reduction 次序，许可见 licenses/NUMPY-PAIRWISE.txt。
fn pairwise(values: &[f64]) -> f64 {
    if values.len() < 8 {
        return values.iter().fold(-0., |a, b| a + b);
    }
    if values.len() > 128 {
        let middle = (values.len() / 2) / 8 * 8;
        return pairwise(&values[..middle]) + pairwise(&values[middle..]);
    }
    let mut sums: [f64; 8] = values[..8].try_into().unwrap();
    let full = values.len() / 8 * 8;
    for chunk in values[8..full].as_chunks::<8>().0 {
        for i in 0..8 {
            sums[i] += chunk[i];
        }
    }
    let sum =
        ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
    values[full..].iter().fold(sum, |a, b| a + b)
}
/// 当前 Python 后处理使用五位小数置信度，文字与 token 另做精确比较。
fn round5(value: f64) -> f64 {
    (value * 100_000.).round_ties_even() / 100_000.
}

/// 字典或 shape 错误不能伪装为空白预测。
fn invalid(message: &str) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.into(),
    }
}

#[cfg(test)]
mod borrowed_tests {
    use super::*;

    #[test]
    fn numpy_confidence_reduction_preserves_decimal_boundary() {
        // Original RapidOCR probabilities: sequential reduction rounds to .99983,
        // NumPy's pairwise reduction rounds to .99982 on this same input.
        let probabilities = [
            0.9998462200164795f32,
            0.9998338222503662f32,
            0.999889612197876f32,
            0.9999862909317017f32,
            0.9999665021896362f32,
            0.9985263347625732f32,
            0.9999970197677612f32,
            0.9998878240585327f32,
            0.9999964237213135f32,
            0.9999696016311646f32,
            0.9999738931655884f32,
            0.9999395608901978f32,
            0.9999945163726807f32,
            0.9999825954437256f32,
            0.999998927116394f32,
            0.9999791383743286f32,
            0.9999536275863647f32,
            0.9999880790710449f32,
            0.9985923171043396f32,
            0.9999809265136719f32,
            0.9997721314430237f32,
            0.9999016523361206f32,
            0.9999978542327881f32,
            0.9999547004699707f32,
            0.9992824196815491f32,
            0.9999496936798096f32,
            0.9999945163726807f32,
            0.9999656677246094f32,
        ];
        let mut values = Vec::new();
        for p in probabilities {
            values.extend([0., p, 1., 0.]);
        }
        let got = decode(
            1,
            probabilities.len() * 2,
            2,
            &values,
            &["blank".into(), "x".into()],
        )
        .unwrap();
        assert_eq!(got[0].confidence, 0.99982);
        assert_eq!(got[0].text, "x".repeat(probabilities.len()));
    }

    #[test]
    fn borrowed_decode_preserves_output_and_removes_copy_scan() {
        let (batch, steps, classes) = (6, 160, 8192);
        let alphabet = (0..classes).map(|i| i.to_string()).collect::<Vec<_>>();
        let values = (0..batch * steps * classes)
            .map(|i| ((i * 13 % 1009) as f32) / 1009.)
            .collect::<Vec<_>>();
        let run = |owned: bool| {
            let started = std::time::Instant::now();
            // Model runner validation is retained in both cases.
            assert!(values.iter().all(|v| v.is_finite()));
            let result = if owned {
                let copy = values.clone();
                decode(batch, steps, classes, &copy, &alphabet).unwrap()
            } else {
                decode_finite(batch, steps, classes, &values, &alphabet).unwrap()
            };
            (
                started.elapsed().as_secs_f64(),
                serde_json::to_value(result).unwrap(),
            )
        };
        let expected = run(true).1;
        let (mut old, mut new) = (vec![], vec![]);
        for i in 0..6 {
            for owned in if i % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            } {
                let (seconds, result) = run(owned);
                assert_eq!(result, expected);
                if owned {
                    old.push(seconds);
                } else {
                    new.push(seconds);
                }
            }
        }
        old.sort_by(f64::total_cmp);
        new.sort_by(f64::total_cmp);
        println!("CTC transfer/decode only, {} floats: owned median {:.6}s; borrowed median {:.6}s; no ORT execution", values.len(), (old[2]+old[3])/2., (new[2]+new[3])/2.);
        assert!(decode_finite(1, 1, 3, &[0., 1.], &alphabet).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(decode(1, 1, 1, &[value], &["blank".into()]).is_err());
        }
    }
}
