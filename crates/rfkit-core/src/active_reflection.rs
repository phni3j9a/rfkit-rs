//! Per-port active reflection for an explicitly supplied incident drive.
//!
//! The kernel evaluates the stored scattering relation
//! `b[f, i] = sum_j S[f, i, j] * incident[f, j]` and then divides that
//! coherent response by the corresponding incident coordinate. The
//! accumulation is deterministic in ascending input-port order and is kept
//! in scale-separated binary64 form until the one final division.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::linalg;

/// Failure modes for the sampled active-reflection kernel.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum ActiveReflectionPowerError {
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

    #[error("incident-wave shape must be (nfreq, nport), got {shape:?}")]
    InvalidIncidentShape { shape: (usize, usize) },

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

    #[error("incident wave is non-finite at frequency {frequency}, port {port}")]
    NonFiniteIncident { frequency: usize, port: usize },

    #[error(
        "active-reflection arithmetic became non-finite or unrepresentable at frequency {frequency}, output port {output_port}"
    )]
    Arithmetic {
        frequency: usize,
        output_port: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct InputShape {
    nfreq: usize,
    nport: usize,
}

/// Evaluate active reflection for each sampled output port.
pub(crate) fn active_reflection_power(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    incident: &Array2<Complex64>,
) -> Result<Array2<Option<Complex64>>, ActiveReflectionPowerError> {
    let shape = validate_input(frequency, s, z0, incident)?;
    let mut output = Array2::from_elem((shape.nfreq, shape.nport), None);

    for frequency_index in 0..shape.nfreq {
        for output_port in 0..shape.nport {
            let denominator = incident[[frequency_index, output_port]];
            if denominator == Complex64::new(0.0, 0.0) {
                continue;
            }

            let mut response = linalg::ScaledComplex::zero();
            for input_port in 0..shape.nport {
                let term = linalg::ScaledComplex::multiply(
                    s[[frequency_index, output_port, input_port]],
                    incident[[frequency_index, input_port]],
                );
                response = response.add(term);
            }

            let ratio = response.divide_by(denominator).map_err(|()| {
                ActiveReflectionPowerError::Arithmetic {
                    frequency: frequency_index,
                    output_port,
                }
            })?;
            output[[frequency_index, output_port]] = Some(ratio);
        }
    }

    Ok(output)
}

fn validate_input(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    incident: &Array2<Complex64>,
) -> Result<InputShape, ActiveReflectionPowerError> {
    if frequency.is_empty() {
        return Err(ActiveReflectionPowerError::EmptyFrequency);
    }

    let s_shape = s.dim();
    if s_shape.0 != frequency.len() {
        return Err(ActiveReflectionPowerError::FrequencyLengthMismatch {
            expected: frequency.len(),
            actual: s_shape.0,
        });
    }
    if s_shape.1 == 0 || s_shape.1 != s_shape.2 {
        return Err(ActiveReflectionPowerError::InvalidSShape { shape: s_shape });
    }

    let z0_shape = z0.dim();
    if z0_shape != (frequency.len(), s_shape.1) {
        return Err(ActiveReflectionPowerError::InvalidZ0Shape { shape: z0_shape });
    }

    let incident_shape = incident.dim();
    if incident_shape != (frequency.len(), s_shape.1) {
        return Err(ActiveReflectionPowerError::InvalidIncidentShape {
            shape: incident_shape,
        });
    }

    for (index, &value) in frequency.iter().enumerate() {
        if !value.is_finite() {
            return Err(ActiveReflectionPowerError::NonFiniteFrequency { index, value });
        }
    }
    for (index, &value) in s.indexed_iter() {
        if !linalg::is_finite(value) {
            return Err(ActiveReflectionPowerError::NonFiniteS {
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in z0.indexed_iter() {
        if !linalg::is_finite(value) {
            return Err(ActiveReflectionPowerError::NonFiniteZ0 {
                frequency: index.0,
                port: index.1,
            });
        }
        if value.re <= 0.0 {
            return Err(ActiveReflectionPowerError::NonPositiveRealZ0 {
                frequency: index.0,
                port: index.1,
                value,
            });
        }
    }
    for (index, &value) in incident.indexed_iter() {
        if !linalg::is_finite(value) {
            return Err(ActiveReflectionPowerError::NonFiniteIncident {
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
