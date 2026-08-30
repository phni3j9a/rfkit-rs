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
/// partial pivoting, with exact complex-zero pivot detection as its sole
/// singularity criterion.  The routine checks arithmetic as it proceeds so a
/// finite input that overflows or otherwise becomes non-finite is reported to
/// the operation rather than returned as silent garbage.
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
        let mut pivot_magnitude = a[pivot_index * n + pivot_index].norm_sqr();
        if !pivot_magnitude.is_finite() {
            return Err(SolveError::NonFinite {
                row: pivot_index,
                column: pivot_index,
            });
        }

        for row in (pivot_index + 1)..n {
            let candidate = a[row * n + pivot_index];
            if !is_finite(candidate) {
                return Err(SolveError::NonFinite {
                    row,
                    column: pivot_index,
                });
            }
            let candidate_magnitude = candidate.norm_sqr();
            if !candidate_magnitude.is_finite() {
                return Err(SolveError::NonFinite {
                    row,
                    column: pivot_index,
                });
            }
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
            let factor = a[row * n + pivot_index] / pivot_value;
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
            let solution = value / pivot;
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
