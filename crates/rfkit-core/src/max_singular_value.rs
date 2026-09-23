//! Sampled maximum singular values of power-wave scattering matrices.
//!
//! For each frequency this module evaluates the full dense complex SVD of the
//! stored S matrix and returns its largest singular value.  The implementation
//! deliberately does not form `SᴴS`: that Gram-matrix route squares scaling and
//! conditioning, and a column norm cannot represent coherent simultaneous
//! excitation.  nalgebra's SVD implementation scales the dense input before
//! bidiagonal reduction; the returned singular values are rescaled by that
//! implementation and are checked before crossing the public boundary.

use nalgebra::{DMatrix, SVD};
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::MaxSingularValuePowerArithmetic;

/// Binary64 tolerance passed to nalgebra's bidiagonal SVD convergence test.
///
/// This controls decomposition convergence only.  It is intentionally not an
/// RF pass/fail tolerance and is not exposed as a public configuration option.
pub(crate) const SVD_TOLERANCE: f64 = 5.0 * f64::EPSILON;

/// Finite total iteration budget supplied to nalgebra for each sample.
pub(crate) const SVD_MAX_ITERATIONS: usize = 10_000;

/// Failure modes for the sampled maximum singular-value kernel.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum MaxSingularValueError {
    #[error("frequency axis must not be empty")]
    EmptyFrequency,

    #[error(
        "frequency axis length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    FrequencyLengthMismatch { expected: usize, actual: usize },

    #[error("S-parameter shape must be square with a positive port count, got {shape:?}")]
    InvalidSShape { shape: (usize, usize, usize) },

    #[error("reference-impedance shape must be (nfreq, nport), got {shape:?}")]
    InvalidZ0Shape { shape: (usize, usize) },

    #[error("frequency is non-finite at index {index}: {value:?}")]
    NonFiniteFrequency { index: usize, value: f64 },

    #[error("S-parameter is non-finite at frequency {frequency}, row {row}, column {column}")]
    NonFiniteS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("reference impedance is non-finite at frequency {frequency}, port {port}")]
    NonFiniteZ0 { frequency: usize, port: usize },

    #[error(
        "reference impedance must have a strictly positive real part at frequency {frequency}, port {port}: {value:?}"
    )]
    NonPositiveRealZ0 {
        frequency: usize,
        port: usize,
        value: Complex64,
    },

    #[error(
        "dense complex SVD did not converge at frequency {frequency} within {max_iterations} iterations (tolerance {tolerance:?})"
    )]
    NonConvergence {
        frequency: usize,
        max_iterations: usize,
        tolerance: f64,
    },

    #[error("non-finite arithmetic at frequency {frequency} while evaluating {stage}")]
    Arithmetic {
        frequency: usize,
        stage: MaxSingularValuePowerArithmetic,
    },
}

#[derive(Debug, Clone, Copy)]
struct InputShape {
    nfreq: usize,
    nport: usize,
}

/// Evaluate one largest singular value for every source-frequency sample.
pub(crate) fn max_singular_value_power(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<Vec<f64>, MaxSingularValueError> {
    let shape = validate_input(frequency, s, z0)?;
    let matrix_len =
        shape
            .nport
            .checked_mul(shape.nport)
            .ok_or(MaxSingularValueError::InvalidSShape {
                shape: (shape.nfreq, shape.nport, shape.nport),
            })?;
    let mut output = Vec::with_capacity(shape.nfreq);

    for frequency_index in 0..shape.nfreq {
        let mut values = Vec::with_capacity(matrix_len);
        for row in 0..shape.nport {
            for column in 0..shape.nport {
                values.push(s[[frequency_index, row, column]]);
            }
        }

        // Validation above guarantees a nonempty square matrix.  Keeping the
        // construction explicit makes ndarray layout irrelevant and prevents
        // malformed serde-created dimensions from being indexed prematurely.
        let matrix = DMatrix::from_row_slice(shape.nport, shape.nport, &values);
        output.push(compute_sigma_max(
            matrix,
            frequency_index,
            SVD_MAX_ITERATIONS,
        )?);
    }

    Ok(output)
}

fn compute_sigma_max(
    matrix: DMatrix<Complex64>,
    frequency: usize,
    max_iterations: usize,
) -> Result<f64, MaxSingularValueError> {
    if matrix.iter().any(|value| !is_finite(*value)) {
        return Err(MaxSingularValueError::Arithmetic {
            frequency,
            stage: MaxSingularValuePowerArithmetic::Matrix,
        });
    }

    let svd = SVD::try_new_unordered(matrix, false, false, SVD_TOLERANCE, max_iterations).ok_or(
        MaxSingularValueError::NonConvergence {
            frequency,
            max_iterations,
            tolerance: SVD_TOLERANCE,
        },
    )?;

    if svd
        .singular_values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(MaxSingularValueError::Arithmetic {
            frequency,
            stage: MaxSingularValuePowerArithmetic::SingularValues,
        });
    }

    // The unordered solver does not promise an order.  The operation needs
    // only the largest value, so scanning all returned singular values avoids
    // an extra sort and keeps the bounded adapter independent of ordering.
    let sigma_max = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    if !sigma_max.is_finite() {
        return Err(MaxSingularValueError::Arithmetic {
            frequency,
            stage: MaxSingularValuePowerArithmetic::Output,
        });
    }
    Ok(sigma_max)
}

fn validate_input(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<InputShape, MaxSingularValueError> {
    if frequency.is_empty() {
        return Err(MaxSingularValueError::EmptyFrequency);
    }

    let s_shape = s.dim();
    if s_shape.0 != frequency.len() {
        return Err(MaxSingularValueError::FrequencyLengthMismatch {
            expected: frequency.len(),
            actual: s_shape.0,
        });
    }
    if s_shape.1 == 0 || s_shape.1 != s_shape.2 {
        return Err(MaxSingularValueError::InvalidSShape { shape: s_shape });
    }

    let z0_shape = z0.dim();
    if z0_shape != (frequency.len(), s_shape.1) {
        return Err(MaxSingularValueError::InvalidZ0Shape { shape: z0_shape });
    }

    for (index, &value) in frequency.iter().enumerate() {
        if !value.is_finite() {
            return Err(MaxSingularValueError::NonFiniteFrequency { index, value });
        }
    }
    for (index, &value) in s.indexed_iter() {
        if !is_finite(value) {
            return Err(MaxSingularValueError::NonFiniteS {
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in z0.indexed_iter() {
        if !is_finite(value) {
            return Err(MaxSingularValueError::NonFiniteZ0 {
                frequency: index.0,
                port: index.1,
            });
        }
        if value.re <= 0.0 {
            return Err(MaxSingularValueError::NonPositiveRealZ0 {
                frequency: index.0,
                port: index.1,
                value,
            });
        }
    }

    Ok(InputShape {
        nfreq: frequency.len(),
        nport: s_shape.1,
    })
}

fn is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn z0(nfreq: usize, nport: usize) -> Array2<Complex64> {
        Array2::from_elem((nfreq, nport), c(50.0, 0.0))
    }

    #[test]
    fn analytic_examples_use_full_svd_not_column_norms() {
        let coherent = Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.6, 0.0), c(0.6, 0.0), c(0.6, 0.0), c(0.6, 0.0)],
        )
        .unwrap();
        let nonnormal = Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.0, 0.0), c(2.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        )
        .unwrap();

        let coherent_result = max_singular_value_power(&[1.0], &coherent, &z0(1, 2)).unwrap();
        let nonnormal_result = max_singular_value_power(&[1.0], &nonnormal, &z0(1, 2)).unwrap();
        assert!((coherent_result[0] - 1.2).abs() < 1.0e-14);
        assert!((nonnormal_result[0] - 2.0).abs() < 1.0e-14);
    }

    #[test]
    fn zero_singular_and_unitary_inputs_are_valid() {
        let zero = Array3::<Complex64>::zeros((1, 3, 3));
        assert_eq!(
            max_singular_value_power(&[1.0], &zero, &z0(1, 3)).unwrap(),
            vec![0.0]
        );

        let unitary = Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        )
        .unwrap();
        assert!(
            (max_singular_value_power(&[1.0], &unitary, &z0(1, 2)).unwrap()[0] - 1.0).abs()
                < 1.0e-14
        );
    }

    #[test]
    fn shape_and_reference_validation_precede_indexing() {
        let frequency = [1.0, 2.0];
        let s = Array3::<Complex64>::zeros((1, 2, 2));
        let z0_data = Array2::<Complex64>::zeros((1, 2));
        assert_eq!(
            max_singular_value_power(&frequency, &s, &z0_data),
            Err(MaxSingularValueError::FrequencyLengthMismatch {
                expected: 2,
                actual: 1,
            })
        );

        let s = Array3::<Complex64>::zeros((1, 0, 0));
        let z0_data = Array2::<Complex64>::zeros((1, 0));
        assert!(matches!(
            max_singular_value_power(&[1.0], &s, &z0_data),
            Err(MaxSingularValueError::InvalidSShape { .. })
        ));

        let s = Array3::<Complex64>::zeros((1, 1, 1));
        let z0_data = Array2::from_elem((1, 1), c(0.0, 1.0));
        assert!(matches!(
            max_singular_value_power(&[1.0], &s, &z0_data),
            Err(MaxSingularValueError::NonPositiveRealZ0 { .. })
        ));
    }

    #[test]
    fn non_finite_values_and_frequency_labels_are_explicit() {
        let s = Array3::<Complex64>::zeros((1, 1, 1));
        let bad_s = Array3::from_elem((1, 1, 1), c(f64::NAN, 0.0));
        assert!(matches!(
            max_singular_value_power(&[-0.0], &bad_s, &z0(1, 1)),
            Err(MaxSingularValueError::NonFiniteS { .. })
        ));
        assert!(matches!(
            max_singular_value_power(&[f64::INFINITY], &s, &z0(1, 1)),
            Err(MaxSingularValueError::NonFiniteFrequency { .. })
        ));
        let result = max_singular_value_power(&[-0.0], &s, &z0(1, 1)).unwrap();
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    fn finite_iteration_budget_reports_nonconvergence() {
        let matrix = DMatrix::from_row_slice(
            6,
            6,
            &[
                c(1.0, 0.0),
                c(0.2, 0.1),
                c(-0.3, 0.2),
                c(0.4, -0.2),
                c(0.5, 0.3),
                c(-0.6, 0.1),
                c(-0.7, 0.4),
                c(0.8, -0.3),
                c(0.9, 0.2),
                c(-1.0, 0.1),
                c(1.1, -0.4),
                c(1.2, 0.5),
                c(1.3, -0.2),
                c(-1.4, 0.3),
                c(1.5, 0.1),
                c(1.6, -0.5),
                c(-1.7, 0.2),
                c(1.8, 0.4),
                c(1.9, -0.1),
                c(2.0, 0.2),
                c(-2.1, 0.3),
                c(2.2, -0.2),
                c(2.3, 0.1),
                c(-2.4, 0.4),
                c(2.5, -0.3),
                c(2.6, 0.2),
                c(-2.7, 0.1),
                c(2.8, -0.4),
                c(2.9, 0.5),
                c(-3.0, 0.2),
                c(3.1, -0.1),
                c(3.2, 0.3),
                c(-3.3, 0.4),
                c(3.4, -0.2),
                c(3.5, 0.1),
                c(-3.6, 0.5),
            ],
        );
        assert_eq!(
            compute_sigma_max(matrix, 4, 1),
            Err(MaxSingularValueError::NonConvergence {
                frequency: 4,
                max_iterations: 1,
                tolerance: SVD_TOLERANCE,
            })
        );
    }
}
