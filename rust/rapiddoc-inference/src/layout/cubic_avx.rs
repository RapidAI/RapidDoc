//! Exact SIMD evaluation of the existing FP32 cubic kernel, not a new sampler.
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,fma,sse4.1")]
unsafe fn rgb(bytes: &[u8], offset: usize) -> __m128 {
    // Four bytes are available except for the final RGB pixel. The spare lane
    // never contributes to a channel and is overwritten by the next pixel.
    let packed = if offset + 4 <= bytes.len() {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    } else {
        u32::from(bytes[offset])
            | (u32::from(bytes[offset + 1]) << 8)
            | (u32::from(bytes[offset + 2]) << 16)
    };
    _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(packed as i32)))
}

#[target_feature(enable = "avx2,fma,sse4.1")]
pub(super) unsafe fn horizontal(
    bytes: &[u8],
    base: usize,
    coefficients: &[(isize, [f32; 4])],
    positions: &[[usize; 4]],
    row: &mut [f32],
    edge: bool,
) {
    for (dx, ((_, a), xs)) in coefficients.iter().zip(positions).enumerate() {
        // Coefficient positions are clamped to the validated source dimensions.
        let [p0, p1, p2, p3] = xs.map(|x| unsafe { rgb(bytes, base + x) });
        let [a0, a1, a2, a3] = a.map(|v| _mm_set1_ps(v));
        let p3 = _mm_mul_ps(p3, a3);
        let value = if edge {
            _mm_fmadd_ps(p0, a0, _mm_fmadd_ps(p1, a1, _mm_fmadd_ps(p2, a2, p3)))
        } else {
            _mm_add_ps(
                _mm_fmadd_ps(p0, a0, p3),
                _mm_fmadd_ps(p1, a1, _mm_mul_ps(p2, a2)),
            )
        };
        let index = dx * 3;
        if index + 4 <= row.len() {
            // The extra lane is inside this row and overwritten by the next RGB.
            unsafe {
                _mm_storeu_ps(row.as_mut_ptr().add(index), value);
            }
        } else {
            let mut tail = [0f32; 4];
            unsafe {
                _mm_storeu_ps(tail.as_mut_ptr(), value);
            }
            row[index..index + 3].copy_from_slice(&tail[..3]);
        }
    }
}

#[target_feature(enable = "avx2,fma,sse4.1")]
pub(super) unsafe fn vertical(rows: [&[f32]; 4], b: [f32; 4], mode: u8, output: &mut [u8]) {
    let [b0, b1, b2, b3] = b.map(|v| _mm256_set1_ps(v));
    let first = _mm256_set1_ps((b[0] + b[1]) + b[2]);
    let last = _mm256_set1_ps((b[1] + b[2]) + b[3]);
    let end = output.len() / 8 * 8;
    for i in (0..end).step_by(8) {
        // Each input row has exactly output.len() entries; loads stay in bounds.
        let [r0, r1, r2, r3] = rows.map(|r| unsafe { _mm256_loadu_ps(r.as_ptr().add(i)) });
        let value = match mode {
            0 => _mm256_fmadd_ps(r0, first, _mm256_mul_ps(r3, b3)),
            1 => _mm256_fmadd_ps(r0, b0, _mm256_mul_ps(r1, last)),
            _ => _mm256_add_ps(
                _mm256_fmadd_ps(r0, b0, _mm256_mul_ps(r1, b1)),
                _mm256_fmadd_ps(r2, b2, _mm256_mul_ps(r3, b3)),
            ),
        };
        let clamped = _mm256_min_ps(
            _mm256_max_ps(value, _mm256_setzero_ps()),
            _mm256_set1_ps(255.),
        );
        let rounded = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(clamped);
        let ints = _mm256_cvttps_epi32(rounded);
        let words = _mm_packs_epi32(
            _mm256_castsi256_si128(ints),
            _mm256_extracti128_si256::<1>(ints),
        );
        let bytes = _mm_packus_epi16(words, _mm_setzero_si128());
        // Eight output bytes fit inside the current output block.
        unsafe {
            _mm_storel_epi64(output.as_mut_ptr().add(i).cast(), bytes);
        }
    }
    for i in end..output.len() {
        let v = match mode {
            0 => rows[0][i].mul_add((b[0] + b[1]) + b[2], rows[3][i] * b[3]),
            1 => rows[0][i].mul_add(b[0], rows[1][i] * ((b[1] + b[2]) + b[3])),
            _ => {
                rows[0][i].mul_add(b[0], rows[1][i] * b[1])
                    + rows[2][i].mul_add(b[2], rows[3][i] * b[3])
            }
        };
        output[i] = v.round_ties_even().clamp(0., 255.) as u8;
    }
}
