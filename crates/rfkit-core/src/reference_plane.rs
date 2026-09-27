//! Sampled lossless power-wave reference-plane shifts.
//!
//! The operation implemented here is the direct diagonal propagation
//! transformation
//!
//! ```text
//! d[f,p] = exp(-j * phase[f,p])
//! S_out[f,i,j] = d[f,i] * S_in[f,i,j] * d[f,j].
//! ```
//!
//! The two endpoint factors are evaluated separately and multiplied before
//! they are applied to the S-parameter value.  The combined factor is then
//! normalized to unit magnitude with binary64 `hypot` and division, correcting
//! only the small norm error introduced by the two independent trigonometric
//! evaluations.  This preserves the operation's one-way phase semantics and
//! avoids treating a finite phase sum as a prerequisite for an otherwise
//! representable product.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::ReferencePlaneShiftArithmetic;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Failure modes for the sampled lossless reference-plane shift kernel.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum ReferencePlaneShiftError {
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

    #[error("one-way phase shape must be (nfreq, nport), got {shape:?}")]
    InvalidPhaseShape { shape: (usize, usize) },

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
        "reference impedance must be finite, real, and strictly positive at frequency {frequency}, port {port}: {value:?}"
    )]
    InvalidZ0 {
        frequency: usize,
        port: usize,
        value: Complex64,
    },

    #[error("one-way phase is non-finite at frequency {frequency}, port {port}")]
    NonFinitePhase { frequency: usize, port: usize },

    #[error(
        "reference-plane shift arithmetic became non-finite or unrepresentable at frequency {frequency}, port {port} while evaluating {stage}"
    )]
    NonFinitePhaseArithmetic {
        frequency: usize,
        port: usize,
        stage: ReferencePlaneShiftArithmetic,
    },

    #[error(
        "reference-plane shift arithmetic became non-finite or unrepresentable at frequency {frequency}, row {row}, column {column} while evaluating {stage}"
    )]
    NonFiniteComputation {
        frequency: usize,
        row: usize,
        column: usize,
        stage: ReferencePlaneShiftArithmetic,
    },
}

#[derive(Debug, Clone, Copy)]
struct InputShape {
    nfreq: usize,
    nport: usize,
}

/// Evaluate a sampled N-port lossless power-wave reference-plane shift.
///
/// Validation is completed before any indexed access so values created by
/// deserialization cannot cause an ndarray panic.  The phase table is a
/// borrowed real-valued `(nfreq, nport)` array; it is never broadcast,
/// converted from another unit, or combined across endpoints before the
/// individual complex factors have been evaluated.  After endpoint
/// multiplication, the combined factor is normalized with binary64 `hypot`
/// and one division before it is applied to S.
pub(crate) fn shift_reference_planes_lossless_power(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    one_way_phase_rad: &Array2<f64>,
) -> Result<Array3<Complex64>, ReferencePlaneShiftError> {
    let shape = validate_input(frequency, s, z0, one_way_phase_rad)?;

    let mut output = Array3::from_elem((shape.nfreq, shape.nport, shape.nport), ZERO);
    for frequency_index in 0..shape.nfreq {
        let mut factors = vec![ZERO; shape.nport];
        for port in 0..shape.nport {
            let phase = one_way_phase_rad[[frequency_index, port]];
            let factor = Complex64::new(phase.cos(), -phase.sin());
            if !is_finite(factor) {
                return Err(ReferencePlaneShiftError::NonFinitePhaseArithmetic {
                    frequency: frequency_index,
                    port,
                    stage: ReferencePlaneShiftArithmetic::PhaseFactor,
                });
            }
            factors[port] = factor;
        }

        for row in 0..shape.nport {
            for column in 0..shape.nport {
                let endpoint_factor = factors[row] * factors[column];
                if !is_finite(endpoint_factor) {
                    return Err(ReferencePlaneShiftError::NonFiniteComputation {
                        frequency: frequency_index,
                        row,
                        column,
                        stage: ReferencePlaneShiftArithmetic::EndpointFactor,
                    });
                }
                let endpoint_magnitude = endpoint_factor.re.hypot(endpoint_factor.im);
                if !endpoint_magnitude.is_finite() || endpoint_magnitude == 0.0 {
                    return Err(ReferencePlaneShiftError::NonFiniteComputation {
                        frequency: frequency_index,
                        row,
                        column,
                        stage: ReferencePlaneShiftArithmetic::EndpointFactor,
                    });
                }
                let endpoint_factor = endpoint_factor / endpoint_magnitude;
                if !is_finite(endpoint_factor) {
                    return Err(ReferencePlaneShiftError::NonFiniteComputation {
                        frequency: frequency_index,
                        row,
                        column,
                        stage: ReferencePlaneShiftArithmetic::EndpointFactor,
                    });
                }

                let value = endpoint_factor * s[[frequency_index, row, column]];
                if !is_finite(value) {
                    return Err(ReferencePlaneShiftError::NonFiniteComputation {
                        frequency: frequency_index,
                        row,
                        column,
                        stage: ReferencePlaneShiftArithmetic::Output,
                    });
                }
                output[[frequency_index, row, column]] = value;
            }
        }
    }

    Ok(output)
}

fn validate_input(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    one_way_phase_rad: &Array2<f64>,
) -> Result<InputShape, ReferencePlaneShiftError> {
    if frequency.is_empty() {
        return Err(ReferencePlaneShiftError::EmptyFrequency);
    }

    let s_shape = s.dim();
    if s_shape.0 != frequency.len() {
        return Err(ReferencePlaneShiftError::FrequencyLengthMismatch {
            expected: frequency.len(),
            actual: s_shape.0,
        });
    }
    if s_shape.1 == 0 || s_shape.1 != s_shape.2 {
        return Err(ReferencePlaneShiftError::InvalidSShape { shape: s_shape });
    }

    let expected_shape = (frequency.len(), s_shape.1);
    let z0_shape = z0.dim();
    if z0_shape != expected_shape {
        return Err(ReferencePlaneShiftError::InvalidZ0Shape { shape: z0_shape });
    }
    let phase_shape = one_way_phase_rad.dim();
    if phase_shape != expected_shape {
        return Err(ReferencePlaneShiftError::InvalidPhaseShape { shape: phase_shape });
    }

    for (index, &value) in frequency.iter().enumerate() {
        if !value.is_finite() {
            return Err(ReferencePlaneShiftError::NonFiniteFrequency { index, value });
        }
    }
    for (index, &value) in s.indexed_iter() {
        if !is_finite(value) {
            return Err(ReferencePlaneShiftError::NonFiniteS {
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in z0.indexed_iter() {
        if !is_finite(value) {
            return Err(ReferencePlaneShiftError::NonFiniteZ0 {
                frequency: index.0,
                port: index.1,
            });
        }
        if value.im != 0.0 || value.re <= 0.0 {
            return Err(ReferencePlaneShiftError::InvalidZ0 {
                frequency: index.0,
                port: index.1,
                value,
            });
        }
    }
    for (index, &value) in one_way_phase_rad.indexed_iter() {
        if !value.is_finite() {
            return Err(ReferencePlaneShiftError::NonFinitePhase {
                frequency: index.0,
                port: index.1,
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

    fn c(real: f64, imag: f64) -> Complex64 {
        Complex64::new(real, imag)
    }

    fn z0(nfreq: usize, nport: usize) -> Array2<Complex64> {
        Array2::from_elem((nfreq, nport), c(50.0, 0.0))
    }

    #[test]
    fn applies_individual_endpoint_factors_without_phase_sum() {
        let s = Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.5, -0.25), c(-0.2, 0.4), c(0.7, 0.1), c(-0.3, -0.6)],
        )
        .unwrap();
        let phase = Array2::from_shape_vec((1, 2), vec![0.4, -0.7]).unwrap();
        let output = shift_reference_planes_lossless_power(&[1.0], &s, &z0(1, 2), &phase).unwrap();

        for row in 0..2 {
            for column in 0..2 {
                let expected = Complex64::from_polar(1.0, -phase[[0, row]])
                    * s[[0, row, column]]
                    * Complex64::from_polar(1.0, -phase[[0, column]]);
                assert!((output[[0, row, column]] - expected).norm() < 2.0e-16);
            }
        }
    }

    #[test]
    fn rejects_complex_or_nonpositive_references() {
        let s = Array3::zeros((1, 1, 1));
        let phase = Array2::zeros((1, 1));
        for reference in [c(50.0, 1.0), c(0.0, 0.0), c(-50.0, 0.0)] {
            assert!(matches!(
                shift_reference_planes_lossless_power(
                    &[1.0],
                    &s,
                    &Array2::from_elem((1, 1), reference),
                    &phase,
                ),
                Err(ReferencePlaneShiftError::InvalidZ0 { .. })
            ));
        }
    }

    #[test]
    fn combines_huge_finite_endpoint_factors_before_s() {
        let s = Array3::from_elem((1, 1, 1), c(f64::MAX, 0.0));
        let phase = Array2::from_elem((1, 1), 1.0e308);
        let output = shift_reference_planes_lossless_power(&[1.0], &s, &z0(1, 1), &phase).unwrap();
        assert!(output[[0, 0, 0]].re.is_finite());
        assert!(output[[0, 0, 0]].im.is_finite());
    }
}
