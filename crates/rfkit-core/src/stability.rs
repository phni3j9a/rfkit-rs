//! Sampled two-port Kurokawa power-wave stability metrics.
//!
//! The kernel is intentionally independent of S/Z/Y conversion. For each
//! frequency sample it evaluates
//!
//! ```text
//! delta = S11*S22 - S12*S21
//! K     = (1 - |S11|^2 - |S22|^2 + |delta|^2) / (2*|S12|*|S21|)
//! ```
//!
//! An exactly zero `S12` or `S21` makes K undefined and is represented by the
//! public `Option` result. Any finite nonzero transmission whose magnitude
//! product is not representable is an arithmetic error instead.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::{TwoPortStability, TwoPortStabilityArithmetic};

/// Failure modes for the private sampled two-port stability kernel.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum TwoPortStabilityError {
    #[error("frequency axis must not be empty")]
    EmptyFrequency,

    #[error(
        "frequency axis length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    FrequencyLengthMismatch { expected: usize, actual: usize },

    #[error("S-parameter shape must be (nfreq, 2, 2), got {shape:?}")]
    InvalidSShape { shape: (usize, usize, usize) },

    #[error("reference-impedance shape must be (nfreq, 2), got {shape:?}")]
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
        "non-finite or unrepresentable arithmetic at frequency {frequency} while evaluating {stage}"
    )]
    Arithmetic {
        frequency: usize,
        stage: TwoPortStabilityArithmetic,
    },
}

#[derive(Debug, Clone, Copy)]
struct InputShape {
    nfreq: usize,
}

/// Evaluate sampled two-port stability metrics from raw network arrays.
pub(crate) fn two_port_stability_power(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<Vec<TwoPortStability>, TwoPortStabilityError> {
    let shape = validate_input(frequency, s, z0)?;
    let mut output = Vec::with_capacity(shape.nfreq);

    for frequency_index in 0..shape.nfreq {
        let s11 = s[[frequency_index, 0, 0]];
        let s12 = s[[frequency_index, 0, 1]];
        let s21 = s[[frequency_index, 1, 0]];
        let s22 = s[[frequency_index, 1, 1]];

        // Evaluate delta before checking whether K is undefined. This keeps
        // the determinant observable for unilateral/isolated samples and
        // ensures determinant arithmetic cannot be hidden by `None`.
        let delta_left = checked_complex_mul(
            s11,
            s22,
            frequency_index,
            TwoPortStabilityArithmetic::DeltaS11S22Product,
        )?;
        let delta_right = checked_complex_mul(
            s12,
            s21,
            frequency_index,
            TwoPortStabilityArithmetic::DeltaS12S21Product,
        )?;
        let delta = checked_complex_sub(
            delta_left,
            delta_right,
            frequency_index,
            TwoPortStabilityArithmetic::DeltaSubtraction,
        )?;

        // Delta is the only required metric for an exactly unilateral or
        // isolated sample. Return immediately after its finite arithmetic has
        // been checked: the K-only reflection magnitudes and numerator are
        // deliberately not evaluated in this undefined-denominator branch.
        // This keeps finite delta + None available even when those unused
        // terms would overflow.
        if is_exact_zero(s12) || is_exact_zero(s21) {
            output.push(TwoPortStability {
                delta,
                rollet_k: None,
            });
            continue;
        }

        let s11_magnitude = checked_norm(
            s11,
            frequency_index,
            TwoPortStabilityArithmetic::S11Magnitude,
        )?;
        let s22_magnitude = checked_norm(
            s22,
            frequency_index,
            TwoPortStabilityArithmetic::S22Magnitude,
        )?;
        let delta_magnitude = checked_norm(
            delta,
            frequency_index,
            TwoPortStabilityArithmetic::DeltaMagnitude,
        )?;

        let s11_squared = checked_mul_f64(
            s11_magnitude,
            s11_magnitude,
            frequency_index,
            TwoPortStabilityArithmetic::S11MagnitudeSquared,
        )?;
        let s22_squared = checked_mul_f64(
            s22_magnitude,
            s22_magnitude,
            frequency_index,
            TwoPortStabilityArithmetic::S22MagnitudeSquared,
        )?;
        let delta_squared = checked_mul_f64(
            delta_magnitude,
            delta_magnitude,
            frequency_index,
            TwoPortStabilityArithmetic::DeltaMagnitudeSquared,
        )?;

        let numerator_after_s11 = checked_sub_f64(
            1.0,
            s11_squared,
            frequency_index,
            TwoPortStabilityArithmetic::NumeratorS11Subtraction,
        )?;
        let numerator_after_s22 = checked_sub_f64(
            numerator_after_s11,
            s22_squared,
            frequency_index,
            TwoPortStabilityArithmetic::NumeratorS22Subtraction,
        )?;
        let numerator = checked_add_f64(
            numerator_after_s22,
            delta_squared,
            frequency_index,
            TwoPortStabilityArithmetic::NumeratorDeltaAddition,
        )?;

        let s12_magnitude = checked_norm(
            s12,
            frequency_index,
            TwoPortStabilityArithmetic::S12Magnitude,
        )?;
        let s21_magnitude = checked_norm(
            s21,
            frequency_index,
            TwoPortStabilityArithmetic::S21Magnitude,
        )?;
        let transmission_product = checked_mul_f64(
            s12_magnitude,
            s21_magnitude,
            frequency_index,
            TwoPortStabilityArithmetic::TransmissionMagnitudeProduct,
        )?;
        if transmission_product == 0.0 {
            return Err(TwoPortStabilityError::Arithmetic {
                frequency: frequency_index,
                stage: TwoPortStabilityArithmetic::TransmissionMagnitudeProduct,
            });
        }
        let denominator = checked_mul_f64(
            2.0,
            transmission_product,
            frequency_index,
            TwoPortStabilityArithmetic::TransmissionDenominatorScaling,
        )?;
        if denominator == 0.0 {
            return Err(TwoPortStabilityError::Arithmetic {
                frequency: frequency_index,
                stage: TwoPortStabilityArithmetic::TransmissionDenominatorScaling,
            });
        }
        let rollet_k = checked_div_f64(
            numerator,
            denominator,
            frequency_index,
            TwoPortStabilityArithmetic::RolletKDivision,
        )?;
        output.push(TwoPortStability {
            delta,
            rollet_k: Some(rollet_k),
        });
    }

    Ok(output)
}

fn validate_input(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<InputShape, TwoPortStabilityError> {
    if frequency.is_empty() {
        return Err(TwoPortStabilityError::EmptyFrequency);
    }

    let s_shape = s.dim();
    if s_shape.0 != frequency.len() {
        return Err(TwoPortStabilityError::FrequencyLengthMismatch {
            expected: frequency.len(),
            actual: s_shape.0,
        });
    }
    if s_shape != (frequency.len(), 2, 2) {
        return Err(TwoPortStabilityError::InvalidSShape { shape: s_shape });
    }

    let z0_shape = z0.dim();
    if z0_shape != (frequency.len(), 2) {
        return Err(TwoPortStabilityError::InvalidZ0Shape { shape: z0_shape });
    }

    for (index, &value) in frequency.iter().enumerate() {
        if !value.is_finite() {
            return Err(TwoPortStabilityError::NonFiniteFrequency { index, value });
        }
    }
    for (index, &value) in s.indexed_iter() {
        if !is_finite(value) {
            return Err(TwoPortStabilityError::NonFiniteS {
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in z0.indexed_iter() {
        if !is_finite(value) {
            return Err(TwoPortStabilityError::NonFiniteZ0 {
                frequency: index.0,
                port: index.1,
            });
        }
        if value.re <= 0.0 {
            return Err(TwoPortStabilityError::NonPositiveRealZ0 {
                frequency: index.0,
                port: index.1,
                value,
            });
        }
    }

    Ok(InputShape {
        nfreq: frequency.len(),
    })
}

fn is_exact_zero(value: Complex64) -> bool {
    value.re == 0.0 && value.im == 0.0
}

fn is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn checked_complex_mul(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    stage: TwoPortStabilityArithmetic,
) -> Result<Complex64, TwoPortStabilityError> {
    let value = left * right;
    if !is_finite(value) {
        return Err(TwoPortStabilityError::Arithmetic { frequency, stage });
    }
    Ok(value)
}

fn checked_complex_sub(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    stage: TwoPortStabilityArithmetic,
) -> Result<Complex64, TwoPortStabilityError> {
    let value = left - right;
    if !is_finite(value) {
        return Err(TwoPortStabilityError::Arithmetic { frequency, stage });
    }
    Ok(value)
}

fn checked_norm(
    value: Complex64,
    frequency: usize,
    stage: TwoPortStabilityArithmetic,
) -> Result<f64, TwoPortStabilityError> {
    let norm = value.norm();
    if !norm.is_finite() {
        return Err(TwoPortStabilityError::Arithmetic { frequency, stage });
    }
    Ok(norm)
}

fn checked_mul_f64(
    left: f64,
    right: f64,
    frequency: usize,
    stage: TwoPortStabilityArithmetic,
) -> Result<f64, TwoPortStabilityError> {
    let value = left * right;
    if !value.is_finite() {
        return Err(TwoPortStabilityError::Arithmetic { frequency, stage });
    }
    Ok(value)
}

fn checked_sub_f64(
    left: f64,
    right: f64,
    frequency: usize,
    stage: TwoPortStabilityArithmetic,
) -> Result<f64, TwoPortStabilityError> {
    let value = left - right;
    if !value.is_finite() {
        return Err(TwoPortStabilityError::Arithmetic { frequency, stage });
    }
    Ok(value)
}

fn checked_add_f64(
    left: f64,
    right: f64,
    frequency: usize,
    stage: TwoPortStabilityArithmetic,
) -> Result<f64, TwoPortStabilityError> {
    let value = left + right;
    if !value.is_finite() {
        return Err(TwoPortStabilityError::Arithmetic { frequency, stage });
    }
    Ok(value)
}

fn checked_div_f64(
    numerator: f64,
    denominator: f64,
    frequency: usize,
    stage: TwoPortStabilityArithmetic,
) -> Result<f64, TwoPortStabilityError> {
    let value = numerator / denominator;
    if !value.is_finite() {
        return Err(TwoPortStabilityError::Arithmetic { frequency, stage });
    }
    Ok(value)
}
