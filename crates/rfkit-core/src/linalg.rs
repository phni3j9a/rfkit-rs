//! Small, operation-independent linear-algebra primitives for the core.
//!
//! The numerical kernels in this crate deliberately use a local solver rather
//! than an operation-specific inverse.  Keeping the solver here makes its
//! exact-singularity and non-finite-value policy explicit and lets conversions
//! such as power waves and impedance/admittance share one implementation.

use num_complex::Complex64;
use thiserror::Error;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Failure modes reported by the internal multiple-right-hand-side solver.
///
/// This error intentionally carries no operation or frequency context.  The
/// caller owns that context and maps the result to its private operation error
/// after solving one frequency slice.  A pivot is singular only when the
/// selected complex value is exactly zero; no tolerance or regularization is
/// part of this primitive.
#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum SolveError {
    #[error("linear solve storage must contain two {n}x{n} matrices, got A={a_len}, B={b_len}")]
    InvalidStorage {
        n: usize,
        a_len: usize,
        b_len: usize,
    },

    #[error("linear solve selected an exactly-zero pivot at index {pivot}")]
    Singular { pivot: usize },

    #[error("linear solve produced a non-finite value at row {row}, column {column}")]
    NonFinite { row: usize, column: usize },
}

/// Solve `A X = B` in place for a square `A` and multiple right-hand sides.
///
/// `A` and `B` are row-major `n`-by-`n` matrices.  Gaussian elimination uses
/// partial pivoting, ranking candidates by the scale-safe magnitude
/// `max(abs(real), abs(imag))`.  This avoids both squaring underflow for tiny
/// non-zero values and squaring overflow for finite values near `f64::MAX`.
/// Exact complex-zero pivot detection remains the sole singularity criterion.
/// The routine checks arithmetic as it proceeds so a finite input that
/// overflows or otherwise becomes non-finite is reported to the operation
/// rather than returned as silent garbage.
pub(crate) fn solve_multiple_rhs(
    a: &mut [Complex64],
    b: &mut [Complex64],
    n: usize,
) -> Result<(), SolveError> {
    let expected_len = n.checked_mul(n).ok_or(SolveError::InvalidStorage {
        n,
        a_len: a.len(),
        b_len: b.len(),
    })?;
    if a.len() != expected_len || b.len() != expected_len {
        return Err(SolveError::InvalidStorage {
            n,
            a_len: a.len(),
            b_len: b.len(),
        });
    }

    for pivot_index in 0..n {
        let mut pivot_row = pivot_index;
        let pivot_candidate = a[pivot_index * n + pivot_index];
        if !is_finite(pivot_candidate) {
            return Err(SolveError::NonFinite {
                row: pivot_index,
                column: pivot_index,
            });
        }
        let mut pivot_magnitude = pivot_score(pivot_candidate);

        for row in (pivot_index + 1)..n {
            let candidate = a[row * n + pivot_index];
            if !is_finite(candidate) {
                return Err(SolveError::NonFinite {
                    row,
                    column: pivot_index,
                });
            }
            let candidate_magnitude = pivot_score(candidate);
            if candidate_magnitude > pivot_magnitude {
                pivot_row = row;
                pivot_magnitude = candidate_magnitude;
            }
        }

        let pivot = a[pivot_row * n + pivot_index];
        if !is_finite(pivot) {
            return Err(SolveError::NonFinite {
                row: pivot_row,
                column: pivot_index,
            });
        }
        if pivot == ZERO {
            return Err(SolveError::Singular { pivot: pivot_index });
        }

        if pivot_row != pivot_index {
            for column in 0..n {
                a.swap(pivot_index * n + column, pivot_row * n + column);
                b.swap(pivot_index * n + column, pivot_row * n + column);
            }
        }

        let pivot_value = a[pivot_index * n + pivot_index];
        for row in (pivot_index + 1)..n {
            let factor = divide_complex(a[row * n + pivot_index], pivot_value);
            if !is_finite(factor) {
                return Err(SolveError::NonFinite {
                    row,
                    column: pivot_index,
                });
            }

            a[row * n + pivot_index] = ZERO;
            for column in (pivot_index + 1)..n {
                let index = row * n + column;
                a[index] -= factor * a[pivot_index * n + column];
                if !is_finite(a[index]) {
                    return Err(SolveError::NonFinite { row, column });
                }
            }
            for rhs_column in 0..n {
                let index = row * n + rhs_column;
                b[index] -= factor * b[pivot_index * n + rhs_column];
                if !is_finite(b[index]) {
                    return Err(SolveError::NonFinite {
                        row,
                        column: rhs_column,
                    });
                }
            }
        }
    }

    for row in (0..n).rev() {
        let pivot = a[row * n + row];
        if !is_finite(pivot) {
            return Err(SolveError::NonFinite { row, column: row });
        }
        if pivot == ZERO {
            return Err(SolveError::Singular { pivot: row });
        }

        for rhs_column in 0..n {
            let mut value = b[row * n + rhs_column];
            for column in (row + 1)..n {
                value -= a[row * n + column] * b[column * n + rhs_column];
                if !is_finite(value) {
                    return Err(SolveError::NonFinite {
                        row,
                        column: rhs_column,
                    });
                }
            }
            let solution = divide_complex(value, pivot);
            if !is_finite(solution) {
                return Err(SolveError::NonFinite {
                    row,
                    column: rhs_column,
                });
            }
            b[row * n + rhs_column] = solution;
        }
    }

    Ok(())
}

/// Return whether both components of a complex value are finite.
pub(crate) fn is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

/// Return a finite, scale-safe pivot ranking for a finite complex value.
///
/// Unlike `Complex::norm_sqr`, this comparison does not square either
/// component, so non-zero subnormal values retain a positive score and finite
/// values with `f64::MAX` components retain a finite score.
fn pivot_score(value: Complex64) -> f64 {
    value.re.abs().max(value.im.abs())
}

/// Divide finite complex values with component scaling.
///
/// `num_complex::Complex64` division uses `re*re + im*im` as its denominator,
/// which can underflow for a non-zero tiny pivot (or overflow for a large
/// finite pivot) even though the quotient is representable.  Scaling both
/// operands by their largest component keeps the denominator in a small finite
/// range and preserves the solver's exact-zero singularity policy.  Callers
/// still perform the finite-result check, so a quotient that is genuinely
/// outside the finite `f64` range is reported as a non-finite computation.
fn divide_complex(left: Complex64, right: Complex64) -> Complex64 {
    let left_scale = left.re.abs().max(left.im.abs());
    if left_scale == 0.0 {
        return ZERO;
    }
    let right_scale = right.re.abs().max(right.im.abs());
    debug_assert!(
        right_scale > 0.0,
        "divide_complex requires a non-zero divisor"
    );
    let left_re = left.re / left_scale;
    let left_im = left.im / left_scale;
    let right_re = right.re / right_scale;
    let right_im = right.im / right_scale;
    let denominator = right_re * right_re + right_im * right_im;
    let quotient_re = (left_re * right_re + left_im * right_im) / denominator;
    let quotient_im = (left_im * right_re - left_re * right_im) / denominator;
    Complex64::new(
        scale_quotient(quotient_re, left_scale, right_scale),
        scale_quotient(quotient_im, left_scale, right_scale),
    )
}

/// Apply `value * left_scale / right_scale` without rounding the scale ratio.
///
/// Each positive factor is represented as a mantissa in `[0.5, 1)` and an
/// integer power of two.  The bounded mantissas are combined before the
/// exponent is applied, so neither a ratio overflow nor a ratio underflow can
/// erase a finite result.  Subnormal results are rounded by one final IEEE
/// multiplication with the corresponding power of two.
fn scale_quotient(value: f64, left_scale: f64, right_scale: f64) -> f64 {
    if value == 0.0 {
        return value;
    }

    let negative = value.is_sign_negative();
    let (value_mantissa, value_exponent) = decompose_positive(value.abs());
    let (left_mantissa, left_exponent) = decompose_positive(left_scale);
    let (right_mantissa, right_exponent) = decompose_positive(right_scale);

    let mut mantissa = value_mantissa * left_mantissa / right_mantissa;
    let mut exponent = value_exponent + left_exponent - right_exponent;
    while mantissa >= 1.0 {
        mantissa *= 0.5;
        exponent += 1;
    }
    while mantissa < 0.5 {
        mantissa *= 2.0;
        exponent -= 1;
    }

    let result = scale_by_power_of_two(mantissa, exponent);
    if negative { -result } else { result }
}

/// Decompose a finite positive number as `mantissa * 2^exponent`.
///
/// The mantissa is in `[0.5, 1)`.  Subnormal inputs are first shifted by the
/// exact power `2^54`, which makes the recursive decomposition normal without
/// losing any source bits.
fn decompose_positive(value: f64) -> (f64, i32) {
    debug_assert!(value.is_finite() && value > 0.0);
    let bits = value.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    if exponent_bits == 0 {
        let (mantissa, exponent) = decompose_positive(value * 18_014_398_509_481_984.0);
        return (mantissa, exponent - 54);
    }

    let fraction = bits & ((1_u64 << 52) - 1);
    let mantissa = f64::from_bits((1022_u64 << 52) | fraction);
    (mantissa, exponent_bits - 1022)
}

/// Scale a normalized mantissa by a power of two, preserving subnormal
/// round-to-nearest behavior without constructing an unrepresentable factor.
fn scale_by_power_of_two(mantissa: f64, exponent: i32) -> f64 {
    const MAX_POWER_OF_TWO: f64 = f64::from_bits(0x7fe0_0000_0000_0000);

    if exponent < -1074 {
        return 0.0;
    }
    if exponent <= 1023 {
        let factor = if exponent >= -1022 {
            f64::from_bits(((exponent + 1023) as u64) << 52)
        } else {
            f64::from_bits(1_u64 << (exponent + 1074) as u32)
        };
        return mantissa * factor;
    }
    if exponent == 1024 {
        return mantissa * MAX_POWER_OF_TWO * 2.0;
    }
    f64::INFINITY
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONE: Complex64 = Complex64::new(1.0, 0.0);

    fn assert_close(actual: Complex64, expected: Complex64) {
        for (actual_component, expected_component, label) in [
            (actual.re, expected.re, "real"),
            (actual.im, expected.im, "imaginary"),
        ] {
            let difference = (actual_component - expected_component).abs();
            let bound = 1.0e-12 * expected_component.abs().max(1.0);
            assert!(
                difference <= bound,
                "{label}: actual={actual:?}, expected={expected:?}, difference={difference:e}, bound={bound:e}"
            );
        }
    }

    #[test]
    fn solves_tiny_nonzero_pivot_without_norm_square_underflow() {
        // The first-column candidates are zero and 1e-200 respectively.  A
        // norm_sqr-based ranker underflows the non-zero candidate's score to
        // zero and falsely reports the leading zero as singular; the
        // scale-safe ranker swaps the non-zero row into the pivot position.
        let mut a = [
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(-1.0e-200, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let original_a = a;
        let mut b = [ONE, ZERO, ZERO, ONE];

        solve_multiple_rhs(&mut a, &mut b, 2).expect("tiny-pivot system is invertible");

        assert_close(b[0], Complex64::new(-1.0e200, 0.0));
        assert_close(b[1], Complex64::new(-1.0e200, 0.0));
        assert_close(b[2], Complex64::new(-1.0, 0.0));
        assert_close(b[3], ZERO);

        for row in 0..2 {
            for column in 0..2 {
                let reconstructed =
                    original_a[row * 2] * b[column] + original_a[row * 2 + 1] * b[2 + column];
                let expected = if row == column { ONE } else { ZERO };
                assert_close(reconstructed, expected);
            }
        }
    }

    #[test]
    fn finite_max_component_does_not_overflow_pivot_score() {
        let finite_max = Complex64::new(f64::MAX, f64::MAX);
        let mut a = [finite_max];
        let mut b = [finite_max];

        solve_multiple_rhs(&mut a, &mut b, 1).expect("finite maximum pivot is valid");

        assert_close(b[0], ONE);

        // The scale ratio itself can overflow while the component-wise
        // quotient remains finite at f64::MAX.  Keep this boundary case
        // covered so scale-safe division does not introduce a false error.
        let mut a = [Complex64::new(0.5, 0.5)];
        let mut b = [Complex64::new(f64::MAX, 0.0)];
        solve_multiple_rhs(&mut a, &mut b, 1).expect("representable maximum quotient");
        assert_close(b[0], Complex64::new(f64::MAX, -f64::MAX));
    }

    #[test]
    fn divides_subnormal_without_scale_ratio_underflow() {
        let minimum_subnormal = f64::from_bits(1);

        assert_eq!(
            divide_complex(
                Complex64::new(minimum_subnormal, minimum_subnormal),
                Complex64::new(2.0, 1.0),
            ),
            Complex64::new(minimum_subnormal, 0.0)
        );
    }

    #[test]
    fn exact_zero_pivot_remains_the_only_singularity_classification() {
        let mut a = [ZERO, ZERO, ZERO, ONE];
        let mut b = [ONE, ZERO, ZERO, ONE];

        assert_eq!(
            solve_multiple_rhs(&mut a, &mut b, 2),
            Err(SolveError::Singular { pivot: 0 })
        );
    }
}
