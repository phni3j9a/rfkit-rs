//! Adjacent-interval secant group delay of a selected power-wave S entry.
//!
//! This module implements the deliberately local definition used by
//! [`crate::Network::group_delay_secant_power`].  It works directly on the
//! selected complex coordinate: phase is extracted with `atan2`, adjacent
//! principal-phase differences are reduced to the shortest branch, and the
//! result is assigned to the interval between the two source-frequency
//! samples.  No magnitude, S/Z/Y conversion, interpolation, or cumulative
//! unwrap is involved.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::GroupDelayArithmetic;

const TWO_PI: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

/// Failure modes for the adjacent-interval group-delay kernel.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum GroupDelayError {
    #[error("frequency axis must contain at least two samples, got {actual}")]
    TooFewFrequencySamples { actual: usize },

    #[error(
        "frequency axis length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    FrequencyLengthMismatch { expected: usize, actual: usize },

    #[error("S-parameter shape must be square with a positive port count, got {shape:?}")]
    InvalidSShape { shape: (usize, usize, usize) },

    #[error("reference-impedance shape must be (nfreq, nport), got {shape:?}")]
    InvalidZ0Shape { shape: (usize, usize) },

    #[error("output port {port} is out of range for {nports} ports")]
    InvalidOutputPort { port: usize, nports: usize },

    #[error("input port {port} is out of range for {nports} ports")]
    InvalidInputPort { port: usize, nports: usize },

    #[error("frequency is non-finite at index {index}: {value:?}")]
    NonFiniteFrequency { index: usize, value: f64 },

    #[error("frequency is negative at index {index}: {value:?}")]
    NegativeFrequency { index: usize, value: f64 },

    #[error(
        "frequency axis is not strictly increasing at index {index}: previous={previous:?}, current={current:?}"
    )]
    FrequencyNotStrictlyIncreasing {
        index: usize,
        previous: f64,
        current: f64,
    },

    #[error("S-parameter is non-finite at frequency {frequency}, row {row}, column {column}")]
    NonFiniteS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("reference impedance is non-finite at frequency {frequency}, port {port}")]
    NonFiniteZ0 { frequency: usize, port: usize },

    #[error("reference impedance has a zero real part at frequency {frequency}, port {port}")]
    ZeroRealZ0 { frequency: usize, port: usize },

    #[error("selected S[{port_out},{port_in}] has undefined exact-zero phase at sample {sample}")]
    UndefinedPhase {
        sample: usize,
        port_out: usize,
        port_in: usize,
    },

    #[error(
        "selected S[{port_out},{port_in}] has an ambiguous exact half-turn on interval {interval}"
    )]
    AmbiguousHalfTurn {
        interval: usize,
        port_out: usize,
        port_in: usize,
    },

    #[error(
        "group-delay arithmetic became non-finite or unrepresentable on interval {interval} for S[{port_out},{port_in}] while evaluating {stage}"
    )]
    Arithmetic {
        interval: usize,
        port_out: usize,
        port_in: usize,
        stage: GroupDelayArithmetic,
    },
}

#[derive(Debug, Clone, Copy)]
struct InputShape {
    nfreq: usize,
}

/// Evaluate adjacent-interval secant group delay for one S entry.
pub(crate) fn group_delay_secant_power(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    port_out: usize,
    port_in: usize,
) -> Result<Vec<f64>, GroupDelayError> {
    let shape = validate_input(frequency, s, z0, port_out, port_in)?;

    // Validation above deliberately checks every S entry and every reference
    // before this first selected indexing.  This keeps malformed serde-created
    // arrays on structured error paths rather than allowing ndarray to panic.
    for sample in 0..shape.nfreq {
        if is_exact_zero(s[[sample, port_out, port_in]]) {
            return Err(GroupDelayError::UndefinedPhase {
                sample,
                port_out,
                port_in,
            });
        }
    }

    let mut output = Vec::with_capacity(shape.nfreq - 1);
    for interval in 0..(shape.nfreq - 1) {
        let phase_left = phase(s[[interval, port_out, port_in]]);
        let phase_right = phase(s[[interval + 1, port_out, port_in]]);
        // Both phases are finite and principal by construction.  The
        // subtraction is bounded by 2*pi and therefore cannot overflow.
        let mut increment = phase_right - phase_left;
        if increment > PI {
            increment -= TWO_PI;
        } else if increment < -PI {
            increment += TWO_PI;
        }

        // Exact evaluated half-turns are intentionally undefined.  This is a
        // local ambiguity rule; no tolerance band is applied around pi.
        if increment.abs() == PI {
            return Err(GroupDelayError::AmbiguousHalfTurn {
                interval,
                port_out,
                port_in,
            });
        }

        let delta_frequency = frequency[interval + 1] - frequency[interval];
        let delay = scaled_delay(increment, delta_frequency).map_err(|stage| {
            GroupDelayError::Arithmetic {
                interval,
                port_out,
                port_in,
                stage,
            }
        })?;
        output.push(delay);
    }

    Ok(output)
}

fn validate_input(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    port_out: usize,
    port_in: usize,
) -> Result<InputShape, GroupDelayError> {
    if frequency.len() < 2 {
        return Err(GroupDelayError::TooFewFrequencySamples {
            actual: frequency.len(),
        });
    }

    let s_shape = s.dim();
    if s_shape.0 != frequency.len() {
        return Err(GroupDelayError::FrequencyLengthMismatch {
            expected: frequency.len(),
            actual: s_shape.0,
        });
    }
    if s_shape.1 == 0 || s_shape.1 != s_shape.2 {
        return Err(GroupDelayError::InvalidSShape { shape: s_shape });
    }

    let z0_shape = z0.dim();
    if z0_shape != (frequency.len(), s_shape.1) {
        return Err(GroupDelayError::InvalidZ0Shape { shape: z0_shape });
    }

    // Port validation precedes any selected-coordinate access.
    if port_out >= s_shape.1 {
        return Err(GroupDelayError::InvalidOutputPort {
            port: port_out,
            nports: s_shape.1,
        });
    }
    if port_in >= s_shape.1 {
        return Err(GroupDelayError::InvalidInputPort {
            port: port_in,
            nports: s_shape.1,
        });
    }

    for (index, &value) in frequency.iter().enumerate() {
        if !value.is_finite() {
            return Err(GroupDelayError::NonFiniteFrequency { index, value });
        }
        if value < 0.0 {
            return Err(GroupDelayError::NegativeFrequency { index, value });
        }
        if index > 0 && value <= frequency[index - 1] {
            return Err(GroupDelayError::FrequencyNotStrictlyIncreasing {
                index,
                previous: frequency[index - 1],
                current: value,
            });
        }
    }

    for (index, &value) in s.indexed_iter() {
        if !is_finite(value) {
            return Err(GroupDelayError::NonFiniteS {
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in z0.indexed_iter() {
        if !is_finite(value) {
            return Err(GroupDelayError::NonFiniteZ0 {
                frequency: index.0,
                port: index.1,
            });
        }
        // The operation does not use the normalization, but the stored
        // power-wave coordinate is still required to have a nonzero real
        // reference.  Negative real parts remain an explicitly accepted
        // algebraic extension of the existing power-wave domain.
        if value.re == 0.0 {
            return Err(GroupDelayError::ZeroRealZ0 {
                frequency: index.0,
                port: index.1,
            });
        }
    }

    Ok(InputShape {
        nfreq: frequency.len(),
    })
}

fn phase(value: Complex64) -> f64 {
    // `arg` is atan2(im, re), not a magnitude/product/ratio path.  That is
    // important for finite huge and subnormal nonzero components.
    value.im.atan2(value.re)
}

fn scaled_delay(increment: f64, delta_frequency: f64) -> Result<f64, GroupDelayArithmetic> {
    if !delta_frequency.is_finite() || delta_frequency <= 0.0 {
        return Err(GroupDelayArithmetic::FrequencyDifference);
    }

    // The common expression `increment / (TAU * df)` can overflow in the
    // denominator for a finite large aperture, spuriously yielding zero after
    // a later division.  Form the product only when it is representable.  In
    // the overflow branch divide the bounded increment by TAU first, then by
    // the large aperture.  This leaves the only possible subnormal rounding
    // at the final division; dividing by the aperture first and then by TAU
    // can double-round a representable minimum-subnormal result to zero.  In
    // the ordinary branch one direct division preserves tiny increments better
    // than dividing the numerator by TAU first.
    let denominator_limit = f64::MAX / TWO_PI;
    let value = if delta_frequency > denominator_limit {
        -(increment / TWO_PI) / delta_frequency
    } else {
        let denominator = TWO_PI * delta_frequency;
        if denominator == 0.0 || !denominator.is_finite() {
            // The comparison above is intentionally conservative around the
            // rounded f64 boundary.  A product that overflows here still has
            // a safe quotient formulation; do not turn it into a false zero
            // or an avoidable arithmetic error.
            -(increment / TWO_PI) / delta_frequency
        } else {
            -increment / denominator
        }
    };
    value
        .is_finite()
        .then_some(value)
        .ok_or(GroupDelayArithmetic::DelayOutput)
}

fn is_exact_zero(value: Complex64) -> bool {
    value.re == 0.0 && value.im == 0.0
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

    fn refs(nfreq: usize, nport: usize) -> Array2<Complex64> {
        Array2::from_elem((nfreq, nport), c(50.0, 0.0))
    }

    #[test]
    fn wraps_both_branch_directions_and_rejects_exact_half_turn() {
        let s = Array3::from_shape_vec(
            (3, 1, 1),
            vec![
                c((-2.9_f64).cos(), (-2.9_f64).sin()),
                c(2.9_f64.cos(), 2.9_f64.sin()),
                c((-2.9_f64).cos(), (-2.9_f64).sin()),
            ],
        )
        .unwrap();
        let delay = group_delay_secant_power(&[0.0, 1.0, 2.0], &s, &refs(3, 1), 0, 0).unwrap();
        assert!(delay[0] > 0.0);
        assert!(delay[1] < 0.0);

        let half = Array3::from_shape_vec((2, 1, 1), vec![c(1.0, 0.0), c(-1.0, 0.0)]).unwrap();
        assert!(matches!(
            group_delay_secant_power(&[0.0, 1.0], &half, &refs(2, 1), 0, 0),
            Err(GroupDelayError::AmbiguousHalfTurn { interval: 0, .. })
        ));
    }

    #[test]
    fn scale_conscious_large_and_tiny_apertures() {
        let s = Array3::from_shape_vec((2, 1, 1), vec![c(1.0, 0.0), c(1.0, 1.0)]).unwrap();
        let output = group_delay_secant_power(&[0.0, f64::MAX], &s, &refs(2, 1), 0, 0).unwrap();
        assert!(output[0].is_finite());
        assert_ne!(output[0], 0.0);

        // Keep the overflow-avoiding path honest at the final subnormal
        // boundary: this phase/aperture pair rounds to a representable
        // minimum-subnormal delay rather than zero.
        let boundary_phase: f64 = 2.9e-15;
        let boundary = Array3::from_shape_vec(
            (2, 1, 1),
            vec![c(1.0, 0.0), c(boundary_phase.cos(), boundary_phase.sin())],
        )
        .unwrap();
        let boundary_output =
            group_delay_secant_power(&[0.0, f64::MAX], &boundary, &refs(2, 1), 0, 0).unwrap();
        assert_ne!(boundary_output[0], 0.0);

        assert!(matches!(
            group_delay_secant_power(&[0.0, f64::from_bits(1)], &s, &refs(2, 1), 0, 0),
            Err(GroupDelayError::Arithmetic { interval: 0, .. })
        ));
    }
}
