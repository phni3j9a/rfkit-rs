//! Direct physical-load termination for frequency-major S-parameter data.
//!
//! The selected port uses currents directed into the source network and the
//! repository's Kurokawa power-wave convention.  For a finite physical load
//! `ZL` and source reference `z_k`, the boundary equation is
//!
//! ```text
//! c       = ZL - z_k
//! d       = ZL + conj(z_k)
//! den     = d - c*S[k,k]
//! S_out   = S[E,E] + S[E,k]*(c/den)*S[k,E]
//! ```
//!
//! The finite-load `den` form is intentional: `d` may be exactly zero while
//! the termination remains valid, so this kernel never forms `c/d`.  An ideal
//! open is selected as a separate physical boundary and uses `den = 1-Skk`.
//! Only an exactly zero evaluated denominator is singular; finite non-zero
//! near-singular values remain in domain.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::{PortLoad, linalg};

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// Failure modes for the private direct physical-load termination kernel.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum TerminationError {
    #[error("source S-parameter shape must be (nfreq, nport, nport), got {shape:?}")]
    InvalidSShape { shape: (usize, usize, usize) },

    #[error("source frequency axis must not be empty")]
    EmptyFrequency,

    #[error(
        "source frequency axis length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    FrequencyShape { expected: usize, actual: usize },

    #[error("source reference-impedance shape must be (nfreq, nport), got {shape:?}")]
    InvalidZ0Shape { shape: (usize, usize) },

    #[error(
        "load length does not match the source frequency dimension: expected {expected}, got {actual}"
    )]
    LoadLengthMismatch { expected: usize, actual: usize },

    #[error("termination requires at least two source ports, got {nports}")]
    NoExternalPorts { nports: usize },

    #[error("termination port {port} is out of range for {nports} source ports")]
    InvalidPort { port: usize, nports: usize },

    #[error(
        "source S-parameter is non-finite at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("source reference impedance is non-finite at frequency {frequency}, port {port}")]
    NonFiniteZ0 { frequency: usize, port: usize },

    #[error("source reference impedance has zero real part at frequency {frequency}, port {port}")]
    ZeroRealReferenceImpedance { frequency: usize, port: usize },

    #[error("termination load is non-finite at frequency {frequency}")]
    NonFiniteLoad { frequency: usize },

    #[error("termination denominator is exactly singular at frequency {frequency}, port {port}")]
    Singular { frequency: usize, port: usize },

    #[error(
        "non-finite value while evaluating termination at frequency {frequency}, selected port {port}, row {row}, column {column}"
    )]
    NonFiniteComputation {
        frequency: usize,
        port: usize,
        row: usize,
        column: usize,
    },
}

/// Result of a direct physical-load termination.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TerminatedNetwork {
    pub(crate) frequency_hz: Vec<f64>,
    pub(crate) s: Array3<Complex64>,
    pub(crate) z0: Array2<Complex64>,
}

/// Apply one physical load per source sample to one source port and remove
/// that port.
///
/// This is a raw-array kernel so malformed serde-created `Network` values can
/// be rejected before any indexing.  The output survivor order is the input
/// order with `port` omitted, and surviving references are copied exactly.
pub(crate) fn terminate_port_power(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    port: usize,
    loads: &[PortLoad],
) -> Result<TerminatedNetwork, TerminationError> {
    let shape = validate_input(frequency, s, z0, loads)?;

    if port >= shape.nports {
        return Err(TerminationError::InvalidPort {
            port,
            nports: shape.nports,
        });
    }
    if shape.nports < 2 {
        return Err(TerminationError::NoExternalPorts {
            nports: shape.nports,
        });
    }

    // Validate every source reference before selecting the requested port or
    // evaluating arithmetic.  Complex and negative-real references are valid;
    // only a zero real part is outside the Kurokawa normalization domain.
    for frequency_index in 0..shape.nfreq {
        for source_port in 0..shape.nports {
            let reference = z0[[frequency_index, source_port]];
            if reference.re == 0.0 {
                return Err(TerminationError::ZeroRealReferenceImpedance {
                    frequency: frequency_index,
                    port: source_port,
                });
            }
        }
    }

    let survivors = survivors(shape.nports, port);
    let n_external = shape.nports - 1;
    let mut output_s = Array3::from_elem((shape.nfreq, n_external, n_external), ZERO);
    let mut output_z0 = Array2::from_elem((shape.nfreq, n_external), ZERO);

    for frequency_index in 0..shape.nfreq {
        for (output_port, &source_port) in survivors.iter().enumerate() {
            output_z0[[frequency_index, output_port]] = z0[[frequency_index, source_port]];
        }

        let source_reference = z0[[frequency_index, port]];
        let feedback = match loads[frequency_index] {
            PortLoad::ImpedanceOhm(load) => {
                // Keep the historical finite-impedance arithmetic sequence
                // unchanged.  In particular, d == 0 is valid when the
                // evaluated denominator is nonzero; do not form c/d.
                let c = checked_sub(load, source_reference, frequency_index, port, port, port)?;
                let d = checked_add(
                    load,
                    source_reference.conj(),
                    frequency_index,
                    port,
                    port,
                    port,
                )?;
                let c_times_skk = checked_mul(
                    c,
                    s[[frequency_index, port, port]],
                    frequency_index,
                    port,
                    port,
                    port,
                )?;
                let denominator = checked_sub(d, c_times_skk, frequency_index, port, port, port)?;
                if denominator == ZERO {
                    return Err(TerminationError::Singular {
                        frequency: frequency_index,
                        port,
                    });
                }

                checked_div(c, denominator, frequency_index, port, port, port)?
            }
            PortLoad::Open => {
                // For I_k = q_k(a_k-b_k) = 0, a_k = b_k and the external
                // elimination factor is exactly 1/(1-S[k,k]).  Selecting the
                // boundary here avoids any infinity sentinel or finite-load
                // fallback, including when external coupling is zero.
                let denominator = checked_sub(
                    ONE,
                    s[[frequency_index, port, port]],
                    frequency_index,
                    port,
                    port,
                    port,
                )?;
                if denominator == ZERO {
                    return Err(TerminationError::Singular {
                        frequency: frequency_index,
                        port,
                    });
                }
                checked_div(ONE, denominator, frequency_index, port, port, port)?
            }
        };

        for (output_row, &row) in survivors.iter().enumerate() {
            for (output_column, &column) in survivors.iter().enumerate() {
                let mut correction = checked_mul(
                    s[[frequency_index, row, port]],
                    feedback,
                    frequency_index,
                    port,
                    output_row,
                    output_column,
                )?;
                correction = checked_mul(
                    correction,
                    s[[frequency_index, port, column]],
                    frequency_index,
                    port,
                    output_row,
                    output_column,
                )?;
                output_s[[frequency_index, output_row, output_column]] = checked_add(
                    s[[frequency_index, row, column]],
                    correction,
                    frequency_index,
                    port,
                    output_row,
                    output_column,
                )?;
            }
        }
    }

    // Every output element is assigned above, but retain a result-wide guard
    // so a future edit cannot silently return non-finite data.
    for (index, &value) in output_s.indexed_iter() {
        if !is_finite(value) {
            return Err(TerminationError::NonFiniteComputation {
                frequency: index.0,
                port,
                row: index.1,
                column: index.2,
            });
        }
    }

    Ok(TerminatedNetwork {
        frequency_hz: frequency.to_vec(),
        s: output_s,
        z0: output_z0,
    })
}

#[derive(Debug, Clone, Copy)]
struct InputShape {
    nfreq: usize,
    nports: usize,
}

fn validate_input(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    loads: &[PortLoad],
) -> Result<InputShape, TerminationError> {
    let shape = s.dim();
    if shape.1 == 0 || shape.1 != shape.2 {
        return Err(TerminationError::InvalidSShape { shape });
    }
    if frequency.is_empty() || shape.0 == 0 {
        return Err(TerminationError::EmptyFrequency);
    }
    if frequency.len() != shape.0 {
        return Err(TerminationError::FrequencyShape {
            expected: shape.0,
            actual: frequency.len(),
        });
    }
    let z0_shape = z0.dim();
    if z0_shape != (shape.0, shape.1) {
        return Err(TerminationError::InvalidZ0Shape { shape: z0_shape });
    }
    if loads.len() != shape.0 {
        return Err(TerminationError::LoadLengthMismatch {
            expected: shape.0,
            actual: loads.len(),
        });
    }

    for (index, &value) in s.indexed_iter() {
        if !is_finite(value) {
            return Err(TerminationError::NonFiniteS {
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in z0.indexed_iter() {
        if !is_finite(value) {
            return Err(TerminationError::NonFiniteZ0 {
                frequency: index.0,
                port: index.1,
            });
        }
    }
    for (frequency, load) in loads.iter().enumerate() {
        if let PortLoad::ImpedanceOhm(value) = load {
            if !is_finite(*value) {
                return Err(TerminationError::NonFiniteLoad { frequency });
            }
        }
    }

    Ok(InputShape {
        nfreq: shape.0,
        nports: shape.1,
    })
}

fn survivors(nports: usize, selected: usize) -> Vec<usize> {
    (0..nports).filter(|&port| port != selected).collect()
}

fn checked_add(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    port: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, TerminationError> {
    checked_value(left + right, frequency, port, row, column)
}

fn checked_sub(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    port: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, TerminationError> {
    checked_value(left - right, frequency, port, row, column)
}

fn checked_mul(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    port: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, TerminationError> {
    checked_value(left * right, frequency, port, row, column)
}

fn checked_div(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    port: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, TerminationError> {
    checked_value(
        linalg::divide_complex(left, right),
        frequency,
        port,
        row,
        column,
    )
}

fn checked_value(
    value: Complex64,
    frequency: usize,
    port: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, TerminationError> {
    if is_finite(value) {
        Ok(value)
    } else {
        Err(TerminationError::NonFiniteComputation {
            frequency,
            port,
            row,
            column,
        })
    }
}

fn is_finite(value: Complex64) -> bool {
    linalg::is_finite(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(real: f64, imag: f64) -> Complex64 {
        Complex64::new(real, imag)
    }

    #[test]
    fn direct_kernel_accepts_zero_d() {
        let frequency = [1.0];
        let s = Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.2, 0.1), c(0.3, -0.2), c(-0.15, 0.25), c(0.1, -0.05)],
        )
        .unwrap();
        let z0 = Array2::from_shape_vec((1, 2), vec![c(50.0, 10.0), c(70.0, -4.0)]).unwrap();
        let load = [PortLoad::ImpedanceOhm(c(-50.0, 10.0))]; // d = ZL + conj(z_k) = 0
        let result = terminate_port_power(&frequency, &s, &z0, 0, &load)
            .expect("finite non-zero den remains valid when d is zero");
        assert_eq!(result.s.dim(), (1, 1, 1));
        assert!(is_finite(result.s[[0, 0, 0]]));
    }

    #[test]
    fn direct_kernel_reports_exact_singularity_but_not_near_singularity() {
        let frequency = [1.0];
        let z0 = Array2::from_elem((1, 2), c(50.0, 0.0));
        let mut s = Array3::zeros((1, 2, 2));
        s[[0, 0, 0]] = c(-1.0, 0.0);
        assert_eq!(
            terminate_port_power(
                &frequency,
                &s,
                &z0,
                0,
                &[PortLoad::ImpedanceOhm(c(0.0, 0.0))],
            ),
            Err(TerminationError::Singular {
                frequency: 0,
                port: 0,
            })
        );

        s[[0, 0, 0]] = c(-1.0 + 1.0e-15, 0.0);
        let result = terminate_port_power(
            &frequency,
            &s,
            &z0,
            0,
            &[PortLoad::ImpedanceOhm(c(0.0, 0.0))],
        )
        .expect("finite near-singular denominator must stay in domain");
        assert!(is_finite(result.s[[0, 0, 0]]));
    }

    #[test]
    fn direct_kernel_rejects_non_finite_inputs_before_indexing() {
        let frequency = [1.0];
        let s = Array3::zeros((1, 2, 2));
        let mut z0 = Array2::from_elem((1, 2), c(50.0, 0.0));
        z0[[0, 1]] = c(f64::NAN, 0.0);
        assert_eq!(
            terminate_port_power(
                &frequency,
                &s,
                &z0,
                0,
                &[PortLoad::ImpedanceOhm(c(1.0, 0.0))],
            ),
            Err(TerminationError::NonFiniteZ0 {
                frequency: 0,
                port: 1,
            })
        );
    }
}
