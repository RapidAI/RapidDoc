//! NumPy 2.4.6 ordering, including equal-key permutations that affect OCR padding.
//! Original Intel BSD-3-Clause backend is compiled into the native library.
use rapiddoc_core::{Error, ErrorCode};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe extern "C" {
    fn rd_numpy_argsort_avx2(values: *mut f64, indices: *mut usize, count: usize) -> i32;
    fn rd_numpy_argsort_avx512(values: *mut f64, indices: *mut usize, count: usize) -> i32;
}
pub fn indices(values: &[f64]) -> Result<Vec<usize>, Error> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error {
            code: ErrorCode::InvalidInput,
            message: "Nonfinite recognition aspect ratio".into(),
        });
    }
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let mut values = values.to_vec();
        let call = if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512dq")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512cd")
        {
            Some(
                rd_numpy_argsort_avx512 as unsafe extern "C" fn(*mut f64, *mut usize, usize) -> i32,
            )
        } else if std::is_x86_feature_detected!("avx2") {
            Some(rd_numpy_argsort_avx2 as unsafe extern "C" fn(*mut f64, *mut usize, usize) -> i32)
        } else {
            None
        };
        if let Some(call) = call {
            // Owned equal-length contiguous buffers; dispatched ISA is checked.
            if unsafe { call(values.as_mut_ptr(), indices.as_mut_ptr(), values.len()) } != 0 {
                return Err(Error {
                    code: ErrorCode::ResourceLimit,
                    message: "Native recognition sorting failed".into(),
                });
            }
            return Ok(indices);
        }
    }
    scalar(values, &mut indices);
    Ok(indices)
}
/// NumPy's original non-SIMD aquicksort/introsort; no Rust unstable-sort substitute.
fn scalar(v: &[f64], a: &mut [usize]) {
    if a.len() < 2 {
        return;
    }
    let mut stack = vec![(0usize, a.len() - 1, (a.len().ilog2() * 2) as i32)];
    while let Some((mut l, mut r, mut depth)) = stack.pop() {
        if depth < 0 {
            heapsort(v, &mut a[l..=r]);
            continue;
        }
        while r - l > 15 {
            let m = l + (r - l) / 2;
            if v[a[m]] < v[a[l]] {
                a.swap(m, l)
            }
            if v[a[r]] < v[a[m]] {
                a.swap(r, m)
            }
            if v[a[m]] < v[a[l]] {
                a.swap(m, l)
            }
            let pivot = v[a[m]];
            let mut i = l;
            let mut j = r - 1;
            a.swap(m, j);
            loop {
                i += 1;
                while v[a[i]] < pivot {
                    i += 1
                }
                j -= 1;
                while pivot < v[a[j]] {
                    j -= 1
                }
                if i >= j {
                    break;
                }
                a.swap(i, j);
            }
            a.swap(i, r - 1);
            depth -= 1;
            if i - l < r - i {
                stack.push((i + 1, r, depth));
                r = i - 1
            } else {
                stack.push((l, i - 1, depth));
                l = i + 1
            }
        }
        for i in l + 1..=r {
            let item = a[i];
            let mut j = i;
            while j > l && v[item] < v[a[j - 1]] {
                a[j] = a[j - 1];
                j -= 1
            }
            a[j] = item;
        }
    }
}
fn heapsort(v: &[f64], a: &mut [usize]) {
    let n = a.len();
    if n < 2 {
        return;
    }
    let mut l = n / 2 + 1;
    let mut r = n;
    loop {
        let item = if l > 1 {
            l -= 1;
            a[l - 1]
        } else {
            let item = a[r - 1];
            a[r - 1] = a[0];
            r -= 1;
            if r == 1 {
                a[0] = item;
                break;
            }
            item
        };
        let mut i = l;
        let mut j = l * 2;
        while j <= r {
            if j < r && v[a[j - 1]] < v[a[j]] {
                j += 1
            }
            if v[item] < v[a[j - 1]] {
                a[i - 1] = a[j - 1];
                i = j;
                j *= 2
            } else {
                break;
            }
        }
        a[i - 1] = item;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_numpy_order_and_padding() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let data: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/recognition-tie-order.json")).unwrap(),
        )
        .unwrap();
        let scalar_data: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                root.join("build/migration/recognition-scalar-order-x86v3-disabled.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for (c, scalar_expected) in data["cases"]
            .as_array()
            .unwrap()
            .iter()
            .zip(scalar_data["indices"].as_array().unwrap())
        {
            let values: Vec<f64> = serde_json::from_value(c["values"].clone()).unwrap();
            let expected: Vec<usize> = serde_json::from_value(c["indices"].clone()).unwrap();
            assert_eq!(indices(&values).unwrap(), expected);
            let mut fallback = (0..values.len()).collect::<Vec<_>>();
            scalar(&values, &mut fallback);
            assert_eq!(&serde_json::json!(fallback), scalar_expected);
            let sizes: Vec<[usize; 2]> = serde_json::from_value(c["sizes"].clone()).unwrap();
            let batches = crate::recognition_batch::plan(&sizes, 6, 32_000_000).unwrap();
            assert_eq!(serde_json::to_value(batches).unwrap(), c["batches"]);
        }
    }
}
