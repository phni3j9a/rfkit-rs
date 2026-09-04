//! Internal matched-junction connection for frequency-major S-parameter stacks.
//!
//! This module implements two narrowly scoped operations: connect one port of
//! an A network to one port of a B network, or connect two ports of one network,
//! through a matched real junction.  The operations are intentionally private
//! while the crate's public network model is still being established.  Inputs
//! use the Kurokawa power-wave convention.  The connected reference impedances
//! must be finite, real, strictly positive, and exactly equal at each
//! frequency; external reference impedances may be any finite complex values.
//!
//! If `k` and `l` are the connected A and B ports, respectively, and `EA` and
//! `EB` are the surviving ports in their original order, the eliminated
//! internal wave variables give
//!
//! ```text
//! D     = 1 - S_A[k,k] S_B[l,l]
//! C_AA = S_A[EA,EA] + S_A[EA,k] S_B[l,l] S_A[k,EA] / D
//! C_AB = S_A[EA,k] S_B[l,EB] / D
//! C_BA = S_B[EB,l] S_A[k,EA] / D
//! C_BB = S_B[EB,EB] + S_B[EB,l] S_A[k,k] S_B[l,EB] / D.
//! ```
//!
//! For same-network inner connection, with internal ports `I = [k, l]`, the
//! matched junction exchange matrix is `P = [[0, 1], [1, 0]]` and elimination
//! uses `S_out = S_EE + S_EI (I - P S_II)^-1 P S_IE`.
//!
//! A denominator is singular only when the computed complex value is exactly
//! zero.  No near-singular cutoff, regularization, or mismatch renormalization
//! is part of this kernel.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::linalg;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// Identifies which input network supplied a failing value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NetworkSide {
    A,
    B,
}

/// Failure modes for the private matched-junction connection kernel.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum ConnectionError {
    #[error("{network:?} S-parameter shape must be (nfreq, nport, nport), got {shape:?}")]
    InvalidSShape {
        network: NetworkSide,
        shape: (usize, usize, usize),
    },

    #[error("{network:?} reference-impedance shape must be (nfreq, nport), got {shape:?}")]
    InvalidZ0Shape {
        network: NetworkSide,
        shape: (usize, usize),
    },

    #[error("{network:?} frequency axis must not be empty")]
    EmptyFrequency { network: NetworkSide },

    #[error(
        "{network:?} frequency axis length does not match S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    FrequencyShape {
        network: NetworkSide,
        expected: usize,
        actual: usize,
    },

    #[error("A and B frequency axes have different lengths: A={a}, B={b}")]
    FrequencyLengthMismatch { a: usize, b: usize },

    #[error("frequency axes differ at index {index}: A={a:?}, B={b:?}")]
    FrequencyMismatch { index: usize, a: f64, b: f64 },

    #[error("{network:?} frequency is non-finite at index {index}: {value:?}")]
    NonFiniteFrequency {
        network: NetworkSide,
        index: usize,
        value: f64,
    },

    #[error(
        "{network:?} S-parameter is non-finite at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteS {
        network: NetworkSide,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("{network:?} reference impedance is non-finite at frequency {frequency}, port {port}")]
    NonFiniteZ0 {
        network: NetworkSide,
        frequency: usize,
        port: usize,
    },

    #[error("{network:?} connection port {port} is out of range for {nports} ports")]
    InvalidPort {
        network: NetworkSide,
        port: usize,
        nports: usize,
    },

    #[error("inner-connect port {port} is out of range for {nports} ports")]
    InvalidInnerPort { port: usize, nports: usize },

    #[error("inner-connect requires two distinct ports, got {port_a} and {port_b}")]
    IdenticalInnerPorts { port_a: usize, port_b: usize },

    #[error(
        "{network:?} connected reference impedance is not finite, real, and strictly positive at frequency {frequency}, port {port}: {value:?}"
    )]
    InvalidJunctionZ0 {
        network: NetworkSide,
        frequency: usize,
        port: usize,
        value: Complex64,
    },

    #[error("connected reference impedances differ at frequency {frequency}: A={a:?}, B={b:?}")]
    MismatchedJunctionZ0 {
        frequency: usize,
        a: Complex64,
        b: Complex64,
    },

    #[error("connecting the selected ports leaves no external ports")]
    NoExternalPorts,

    #[error("matched-junction connection is exactly singular at frequency {frequency}")]
    Singular { frequency: usize },

    #[error(
        "non-finite value while evaluating matched-junction connection at frequency {frequency}, output row {row}, column {column}"
    )]
    NonFiniteComputation {
        frequency: usize,
        row: usize,
        column: usize,
    },
}

/// Result of a private matched-junction connection.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConnectedNetwork {
    /// The common input frequency axis, copied without interpolation.
    pub(crate) frequency_hz: Vec<f64>,
    /// Connected frequency-major S-parameter stack.
    pub(crate) s: Array3<Complex64>,
    /// Connected frequency-major reference impedances in output-port order.
    pub(crate) z0: Array2<Complex64>,
}

/// Connect two distinct ports of one network through a matched real junction.
///
/// The result contains every port except `port_k` and `port_l`, in the input's
/// original order.  The two selected ports are ordered internally as
/// `I = [port_k, port_l]`; the matched junction exchanges their incident waves
/// with `P = [[0, 1], [1, 0]]`.  Eliminating those internal waves therefore
/// gives
///
/// ```text
/// S_out = S_EE + S_EI (I - P S_II)^-1 P S_IE.
/// ```
///
/// This is intentionally an internal raw-array operation.  It uses the same
/// Kurokawa power-wave and matched-junction contract as [`connect_matched`]:
/// selected reference impedances must be finite, real, strictly positive, and
/// exactly equal at every frequency.  No renormalization or near-singular
/// classification is performed.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn inner_connect_matched(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    port_k: usize,
    port_l: usize,
) -> Result<ConnectedNetwork, ConnectionError> {
    let shape = validate_input(NetworkSide::A, frequency, s, z0)?;

    // Validate both indices before applying the distinct-port rule.  This
    // keeps an out-of-range request deterministic even for a one-port input.
    if port_k >= shape.nports {
        return Err(ConnectionError::InvalidInnerPort {
            port: port_k,
            nports: shape.nports,
        });
    }
    if port_l >= shape.nports {
        return Err(ConnectionError::InvalidInnerPort {
            port: port_l,
            nports: shape.nports,
        });
    }
    if port_k == port_l {
        return Err(ConnectionError::IdenticalInnerPorts {
            port_a: port_k,
            port_b: port_l,
        });
    }

    let external = inner_survivors(shape.nports, port_k, port_l);
    let n_external = shape
        .nports
        .checked_sub(2)
        .ok_or(ConnectionError::NoExternalPorts)?;
    if n_external == 0 {
        return Err(ConnectionError::NoExternalPorts);
    }

    // Validate the junction contract before allocating or evaluating any
    // arithmetic.  validate_input has already rejected non-finite z0 values.
    for frequency_index in 0..shape.nfreq {
        let junction_k = z0[[frequency_index, port_k]];
        let junction_l = z0[[frequency_index, port_l]];
        if junction_k.re <= 0.0 || junction_k.im != 0.0 {
            return Err(ConnectionError::InvalidJunctionZ0 {
                network: NetworkSide::A,
                frequency: frequency_index,
                port: port_k,
                value: junction_k,
            });
        }
        if junction_l.re <= 0.0 || junction_l.im != 0.0 {
            return Err(ConnectionError::InvalidJunctionZ0 {
                network: NetworkSide::A,
                frequency: frequency_index,
                port: port_l,
                value: junction_l,
            });
        }
        if junction_k != junction_l {
            return Err(ConnectionError::MismatchedJunctionZ0 {
                frequency: frequency_index,
                a: junction_k,
                b: junction_l,
            });
        }
    }

    let mut output_s = Array3::from_elem((shape.nfreq, n_external, n_external), ZERO);
    let mut output_z0 = Array2::from_elem((shape.nfreq, n_external), ZERO);

    for frequency_index in 0..shape.nfreq {
        for (output_port, &input_port) in external.iter().enumerate() {
            output_z0[[frequency_index, output_port]] = z0[[frequency_index, input_port]];
        }

        // Construct A = I - P S_II and B = P, then solve A X = B.  The
        // row-major layout is the same layout required by the shared N-port
        // solver.  S_II uses internal order [port_k, port_l].
        let s_kk = s[[frequency_index, port_k, port_k]];
        let s_kl = s[[frequency_index, port_k, port_l]];
        let s_lk = s[[frequency_index, port_l, port_k]];
        let s_ll = s[[frequency_index, port_l, port_l]];
        let mut a = vec![ZERO; 4];
        a[0] = checked_sub(ONE, s_lk, frequency_index, 0, 0)?;
        a[1] = checked_neg(s_ll, frequency_index, 0, 1)?;
        a[2] = checked_neg(s_kk, frequency_index, 1, 0)?;
        a[3] = checked_sub(ONE, s_kl, frequency_index, 1, 1)?;
        let mut x = vec![ZERO, ONE, ONE, ZERO];
        match linalg::solve_multiple_rhs(&mut a, &mut x, 2) {
            Ok(()) => {}
            Err(linalg::SolveError::InvalidStorage { .. }) => {
                unreachable!("inner matched-connect solver storage is fixed 2x2")
            }
            Err(linalg::SolveError::Singular { .. }) => {
                return Err(ConnectionError::Singular {
                    frequency: frequency_index,
                });
            }
            Err(linalg::SolveError::NonFinite { row, column }) => {
                return Err(ConnectionError::NonFiniteComputation {
                    frequency: frequency_index,
                    row,
                    column,
                });
            }
        }

        for (output_row, &row) in external.iter().enumerate() {
            for (output_column, &column) in external.iter().enumerate() {
                let mut correction = ZERO;
                for internal_row in 0..2 {
                    let left_port = if internal_row == 0 { port_k } else { port_l };
                    let left = s[[frequency_index, row, left_port]];
                    for internal_column in 0..2 {
                        let right_port = if internal_column == 0 { port_k } else { port_l };
                        let right = s[[frequency_index, right_port, column]];
                        let term = checked_mul(
                            left,
                            x[internal_row * 2 + internal_column],
                            frequency_index,
                            output_row,
                            output_column,
                        )?;
                        let term =
                            checked_mul(term, right, frequency_index, output_row, output_column)?;
                        correction = checked_add(
                            correction,
                            term,
                            frequency_index,
                            output_row,
                            output_column,
                        )?;
                    }
                }
                output_s[[frequency_index, output_row, output_column]] = checked_add(
                    s[[frequency_index, row, column]],
                    correction,
                    frequency_index,
                    output_row,
                    output_column,
                )?;
            }
        }
    }

    for (index, &value) in output_s.indexed_iter() {
        if !is_finite(value) {
            return Err(ConnectionError::NonFiniteComputation {
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in output_z0.indexed_iter() {
        if !is_finite(value) {
            return Err(ConnectionError::NonFiniteComputation {
                frequency: index.0,
                row: index.1,
                column: index.1,
            });
        }
    }

    Ok(ConnectedNetwork {
        frequency_hz: frequency.to_vec(),
        s: output_s,
        z0: output_z0,
    })
}

/// Connect A port `port_a` to B port `port_b` through a matched real junction.
///
/// The result contains A's unconnected ports in original order followed by
/// B's unconnected ports in original order.  The returned reference
/// impedances use the same ordering.  Both raw frequency axes must be finite
/// and exactly identical; no interpolation, subset, or resampling is done.
///
/// The function accepts raw axes and arrays so malformed shapes can be tested
/// directly.  It deliberately does not construct or expose a public
/// [`crate::Network`] operation.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn connect_matched(
    frequency_a: &[f64],
    s_a: &Array3<Complex64>,
    z0_a: &Array2<Complex64>,
    port_a: usize,
    frequency_b: &[f64],
    s_b: &Array3<Complex64>,
    z0_b: &Array2<Complex64>,
    port_b: usize,
) -> Result<ConnectedNetwork, ConnectionError> {
    let shape_a = validate_input(NetworkSide::A, frequency_a, s_a, z0_a)?;
    let shape_b = validate_input(NetworkSide::B, frequency_b, s_b, z0_b)?;

    if frequency_a.len() != frequency_b.len() {
        return Err(ConnectionError::FrequencyLengthMismatch {
            a: frequency_a.len(),
            b: frequency_b.len(),
        });
    }
    for (index, (&a, &b)) in frequency_a.iter().zip(frequency_b).enumerate() {
        if a != b {
            return Err(ConnectionError::FrequencyMismatch { index, a, b });
        }
    }

    if port_a >= shape_a.nports {
        return Err(ConnectionError::InvalidPort {
            network: NetworkSide::A,
            port: port_a,
            nports: shape_a.nports,
        });
    }
    if port_b >= shape_b.nports {
        return Err(ConnectionError::InvalidPort {
            network: NetworkSide::B,
            port: port_b,
            nports: shape_b.nports,
        });
    }

    let external_a = survivors(shape_a.nports, port_a);
    let external_b = survivors(shape_b.nports, port_b);
    let n_external = shape_a
        .nports
        .checked_add(shape_b.nports)
        .and_then(|sum| sum.checked_sub(2))
        .ok_or(ConnectionError::NoExternalPorts)?;
    if n_external == 0 {
        return Err(ConnectionError::NoExternalPorts);
    }

    for frequency in 0..shape_a.nfreq {
        let junction_a = z0_a[[frequency, port_a]];
        let junction_b = z0_b[[frequency, port_b]];
        if junction_a.re <= 0.0 || junction_a.im != 0.0 {
            return Err(ConnectionError::InvalidJunctionZ0 {
                network: NetworkSide::A,
                frequency,
                port: port_a,
                value: junction_a,
            });
        }
        if junction_b.re <= 0.0 || junction_b.im != 0.0 {
            return Err(ConnectionError::InvalidJunctionZ0 {
                network: NetworkSide::B,
                frequency,
                port: port_b,
                value: junction_b,
            });
        }
        if junction_a != junction_b {
            return Err(ConnectionError::MismatchedJunctionZ0 {
                frequency,
                a: junction_a,
                b: junction_b,
            });
        }
    }

    let mut output_s = Array3::from_elem((shape_a.nfreq, n_external, n_external), ZERO);
    let mut output_z0 = Array2::from_elem((shape_a.nfreq, n_external), ZERO);
    for frequency in 0..shape_a.nfreq {
        for (output_port, &input_port) in external_a.iter().enumerate() {
            let value = z0_a[[frequency, input_port]];
            if !is_finite(value) {
                return Err(ConnectionError::NonFiniteComputation {
                    frequency,
                    row: output_port,
                    column: output_port,
                });
            }
            output_z0[[frequency, output_port]] = value;
        }
        for (offset, &input_port) in external_b.iter().enumerate() {
            let output_port = external_a.len() + offset;
            let value = z0_b[[frequency, input_port]];
            if !is_finite(value) {
                return Err(ConnectionError::NonFiniteComputation {
                    frequency,
                    row: output_port,
                    column: output_port,
                });
            }
            output_z0[[frequency, output_port]] = value;
        }

        let a_kk = s_a[[frequency, port_a, port_a]];
        let b_ll = s_b[[frequency, port_b, port_b]];
        let product = checked_mul(a_kk, b_ll, frequency, 0, 0)?;
        let denominator = checked_sub(ONE, product, frequency, 0, 0)?;
        if denominator == ZERO {
            return Err(ConnectionError::Singular { frequency });
        }

        // A-survivor by A-survivor block.
        for (output_row, &row) in external_a.iter().enumerate() {
            for (output_column, &column) in external_a.iter().enumerate() {
                let mut correction = checked_mul(
                    s_a[[frequency, row, port_a]],
                    b_ll,
                    frequency,
                    output_row,
                    output_column,
                )?;
                correction = checked_mul(
                    correction,
                    s_a[[frequency, port_a, column]],
                    frequency,
                    output_row,
                    output_column,
                )?;
                correction = checked_div(
                    correction,
                    denominator,
                    frequency,
                    output_row,
                    output_column,
                )?;
                let value = checked_add(
                    s_a[[frequency, row, column]],
                    correction,
                    frequency,
                    output_row,
                    output_column,
                )?;
                output_s[[frequency, output_row, output_column]] = value;
            }
        }

        // A-survivor by B-survivor block.
        for (output_row, &row) in external_a.iter().enumerate() {
            for (b_offset, &column) in external_b.iter().enumerate() {
                let output_column = external_a.len() + b_offset;
                let numerator = checked_mul(
                    s_a[[frequency, row, port_a]],
                    s_b[[frequency, port_b, column]],
                    frequency,
                    output_row,
                    output_column,
                )?;
                let value =
                    checked_div(numerator, denominator, frequency, output_row, output_column)?;
                output_s[[frequency, output_row, output_column]] = value;
            }
        }

        // B-survivor by A-survivor block.
        for (b_offset, &row) in external_b.iter().enumerate() {
            let output_row = external_a.len() + b_offset;
            for (output_column, &column) in external_a.iter().enumerate() {
                let numerator = checked_mul(
                    s_b[[frequency, row, port_b]],
                    s_a[[frequency, port_a, column]],
                    frequency,
                    output_row,
                    output_column,
                )?;
                let value =
                    checked_div(numerator, denominator, frequency, output_row, output_column)?;
                output_s[[frequency, output_row, output_column]] = value;
            }
        }

        // B-survivor by B-survivor block.
        for (b_offset_row, &row) in external_b.iter().enumerate() {
            let output_row = external_a.len() + b_offset_row;
            for (b_offset_column, &column) in external_b.iter().enumerate() {
                let output_column = external_a.len() + b_offset_column;
                let mut correction = checked_mul(
                    s_b[[frequency, row, port_b]],
                    a_kk,
                    frequency,
                    output_row,
                    output_column,
                )?;
                correction = checked_mul(
                    correction,
                    s_b[[frequency, port_b, column]],
                    frequency,
                    output_row,
                    output_column,
                )?;
                correction = checked_div(
                    correction,
                    denominator,
                    frequency,
                    output_row,
                    output_column,
                )?;
                let value = checked_add(
                    s_b[[frequency, row, column]],
                    correction,
                    frequency,
                    output_row,
                    output_column,
                )?;
                output_s[[frequency, output_row, output_column]] = value;
            }
        }
    }

    // Every output element is assigned by one of the four non-empty blocks,
    // but keep a final result-wide guard close to the return boundary so a
    // future edit cannot silently leak a non-finite value.
    for (index, &value) in output_s.indexed_iter() {
        if !is_finite(value) {
            return Err(ConnectionError::NonFiniteComputation {
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in output_z0.indexed_iter() {
        if !is_finite(value) {
            return Err(ConnectionError::NonFiniteComputation {
                frequency: index.0,
                row: index.1,
                column: index.1,
            });
        }
    }

    Ok(ConnectedNetwork {
        frequency_hz: frequency_a.to_vec(),
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
    network: NetworkSide,
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<InputShape, ConnectionError> {
    let shape = s.dim();
    if shape.1 == 0 || shape.1 != shape.2 {
        return Err(ConnectionError::InvalidSShape { network, shape });
    }
    if frequency.is_empty() || shape.0 == 0 {
        return Err(ConnectionError::EmptyFrequency { network });
    }
    if frequency.len() != shape.0 {
        return Err(ConnectionError::FrequencyShape {
            network,
            expected: shape.0,
            actual: frequency.len(),
        });
    }
    let z0_shape = z0.dim();
    if z0_shape != (shape.0, shape.1) {
        return Err(ConnectionError::InvalidZ0Shape {
            network,
            shape: z0_shape,
        });
    }

    for (index, &value) in frequency.iter().enumerate() {
        if !value.is_finite() {
            return Err(ConnectionError::NonFiniteFrequency {
                network,
                index,
                value,
            });
        }
    }
    for (index, &value) in s.indexed_iter() {
        if !is_finite(value) {
            return Err(ConnectionError::NonFiniteS {
                network,
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in z0.indexed_iter() {
        if !is_finite(value) {
            return Err(ConnectionError::NonFiniteZ0 {
                network,
                frequency: index.0,
                port: index.1,
            });
        }
    }

    Ok(InputShape {
        nfreq: shape.0,
        nports: shape.1,
    })
}

fn survivors(nports: usize, connected: usize) -> Vec<usize> {
    (0..nports).filter(|&port| port != connected).collect()
}

fn inner_survivors(nports: usize, port_k: usize, port_l: usize) -> Vec<usize> {
    (0..nports)
        .filter(|&port| port != port_k && port != port_l)
        .collect()
}

fn checked_mul(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, ConnectionError> {
    checked_value(left * right, frequency, row, column)
}

fn checked_add(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, ConnectionError> {
    checked_value(left + right, frequency, row, column)
}

fn checked_sub(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, ConnectionError> {
    checked_value(left - right, frequency, row, column)
}

fn checked_neg(
    value: Complex64,
    frequency: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, ConnectionError> {
    checked_value(-value, frequency, row, column)
}

fn checked_div(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, ConnectionError> {
    checked_value(linalg::divide_complex(left, right), frequency, row, column)
}

fn checked_value(
    value: Complex64,
    frequency: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, ConnectionError> {
    if is_finite(value) {
        Ok(value)
    } else {
        Err(ConnectionError::NonFiniteComputation {
            frequency,
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
    use ndarray::{Array2, Array3};
    use serde::Deserialize;

    const CONNECT_FIXTURE_JSON: &str = include_str!(
        "../../../tools/oracle/fixtures/power_wave_connect_matched_three_to_four_port_real_frequency_dependent_z0.json"
    );
    const INNER_CONNECT_FIXTURE_JSON: &str = include_str!(
        "../../../tools/oracle/fixtures/power_wave_inner_connect_matched_five_port_real_frequency_dependent_z0.json"
    );

    fn complex(real: f64, imag: f64) -> Complex64 {
        Complex64::new(real, imag)
    }

    fn ideal_tee(nfreq: usize) -> Array3<Complex64> {
        Array3::from_shape_fn((nfreq, 3, 3), |(_, row, column)| {
            if row == column {
                complex(-1.0 / 3.0, 0.0)
            } else {
                complex(2.0 / 3.0, 0.0)
            }
        })
    }

    fn valid_z0(nfreq: usize, nports: usize) -> Array2<Complex64> {
        Array2::from_elem((nfreq, nports), complex(61.25, 0.0))
    }

    type Inputs = (
        Vec<f64>,
        Array3<Complex64>,
        Array2<Complex64>,
        Vec<f64>,
        Array3<Complex64>,
        Array2<Complex64>,
    );

    fn valid_inputs() -> Inputs {
        let frequency = vec![1.0e9, 1.7e9];
        (
            frequency.clone(),
            ideal_tee(frequency.len()),
            valid_z0(frequency.len(), 3),
            frequency,
            ideal_tee(2),
            valid_z0(2, 3),
        )
    }

    fn two_ideal_tees(nfreq: usize) -> Array3<Complex64> {
        Array3::from_shape_fn((nfreq, 6, 6), |(_, row, column)| {
            if row / 3 != column / 3 {
                ZERO
            } else if row == column {
                complex(-1.0 / 3.0, 0.0)
            } else {
                complex(2.0 / 3.0, 0.0)
            }
        })
    }

    #[test]
    fn inner_connects_block_diagonal_ideal_tees_to_known_four_port() {
        let frequency = vec![1.0e9, 1.7e9];
        let s = two_ideal_tees(frequency.len());
        let z0 = Array2::from_shape_vec(
            (2, 6),
            vec![
                complex(41.0, 1.0),
                complex(73.5, 0.0),
                complex(83.0, -2.0),
                complex(101.0, 3.0),
                complex(73.5, 0.0),
                complex(107.0, -1.0),
                complex(44.0, 1.5),
                complex(86.25, 0.0),
                complex(97.0, -2.5),
                complex(111.0, 3.5),
                complex(86.25, 0.0),
                complex(117.0, -1.5),
            ],
        )
        .expect("fixed inner-connect z0 shape must be valid");

        // Connect one port from each isolated ideal tee.  The surviving order
        // is [0, 2, 3, 5], and the analytically known result is an ideal
        // four-port with -1/2 on the diagonal and +1/2 off-diagonal.
        let result = inner_connect_matched(&frequency, &s, &z0, 1, 4)
            .expect("ideal tee inner connection must succeed");

        assert_eq!(result.frequency_hz, frequency);
        assert_eq!(result.s.dim(), (2, 4, 4));
        for frequency_index in 0..2 {
            for row in 0..4 {
                for column in 0..4 {
                    let expected = if row == column { -0.5 } else { 0.5 };
                    assert!((result.s[[frequency_index, row, column]].re - expected).abs() < 1e-13);
                    assert_eq!(result.s[[frequency_index, row, column]].im, 0.0);
                }
            }
        }
        assert_eq!(
            result.z0,
            Array2::from_shape_vec(
                (2, 4),
                vec![
                    complex(41.0, 1.0),
                    complex(83.0, -2.0),
                    complex(101.0, 3.0),
                    complex(107.0, -1.0),
                    complex(44.0, 1.5),
                    complex(97.0, -2.5),
                    complex(111.0, 3.5),
                    complex(117.0, -1.5),
                ],
            )
            .expect("fixed output z0 shape must be valid")
        );
    }

    #[test]
    fn inner_connect_supports_a_single_survivor() {
        let frequency = [2.4e9];
        let mut s = Array3::zeros((1, 3, 3));
        s[[0, 1, 1]] = complex(0.1, -0.02);
        s[[0, 1, 0]] = complex(0.2, 0.0);
        s[[0, 2, 1]] = complex(0.3, 0.0);
        let z0 = valid_z0(1, 3);

        let result = inner_connect_matched(&frequency, &s, &z0, 0, 2)
            .expect("three-port inner connection must leave one survivor");

        assert_eq!(result.s.dim(), (1, 1, 1));
        // With S_II=0, (I-P S_II)^-1 P=P.  Only the k->survivor and
        // survivor->l paths are populated, so the correction is 0.2*0.3.
        assert_eq!(result.s[[0, 0, 0]], complex(0.16, -0.02));
        assert_eq!(result.z0, Array2::from_elem((1, 1), complex(61.25, 0.0)));
    }

    #[test]
    fn inner_connect_rejects_invalid_port_selection_and_zero_survivors() {
        let frequency = [1.0e9];
        let s = Array3::zeros((1, 3, 3));
        let z0 = valid_z0(1, 3);

        assert_eq!(
            inner_connect_matched(&frequency, &s, &z0, 1, 1).unwrap_err(),
            ConnectionError::IdenticalInnerPorts {
                port_a: 1,
                port_b: 1
            }
        );
        assert_eq!(
            inner_connect_matched(&frequency, &s, &z0, 3, 1).unwrap_err(),
            ConnectionError::InvalidInnerPort { port: 3, nports: 3 }
        );
        assert_eq!(
            inner_connect_matched(&frequency, &s, &z0, 1, 3).unwrap_err(),
            ConnectionError::InvalidInnerPort { port: 3, nports: 3 }
        );

        let two_port_s = Array3::zeros((1, 2, 2));
        let two_port_z0 = valid_z0(1, 2);
        assert_eq!(
            inner_connect_matched(&frequency, &two_port_s, &two_port_z0, 0, 1).unwrap_err(),
            ConnectionError::NoExternalPorts
        );
    }

    #[test]
    fn inner_connect_enforces_matched_real_positive_junction_contract() {
        let frequency = [1.0e9, 1.7e9];
        let s = Array3::zeros((2, 3, 3));
        let z0 = valid_z0(2, 3);

        let mut imaginary = z0.clone();
        imaginary[[0, 0]] = complex(61.25, 1.0);
        assert!(matches!(
            inner_connect_matched(&frequency, &s, &imaginary, 0, 2),
            Err(ConnectionError::InvalidJunctionZ0 {
                network: NetworkSide::A,
                frequency: 0,
                port: 0,
                ..
            })
        ));

        for invalid in [0.0, -1.0] {
            let mut invalid_z0 = z0.clone();
            invalid_z0[[0, 0]] = complex(invalid, 0.0);
            assert!(matches!(
                inner_connect_matched(&frequency, &s, &invalid_z0, 0, 2),
                Err(ConnectionError::InvalidJunctionZ0 {
                    network: NetworkSide::A,
                    frequency: 0,
                    port: 0,
                    ..
                })
            ));
        }

        let mut mismatched = z0.clone();
        mismatched[[1, 2]] = complex(62.0, 0.0);
        assert_eq!(
            inner_connect_matched(&frequency, &s, &mismatched, 0, 2).unwrap_err(),
            ConnectionError::MismatchedJunctionZ0 {
                frequency: 1,
                a: complex(61.25, 0.0),
                b: complex(62.0, 0.0),
            }
        );
    }

    #[test]
    fn inner_connect_rejects_nonfinite_values_and_reports_nonfinite_computation() {
        let frequency = [1.0e9];
        let s = Array3::zeros((1, 3, 3));
        let z0 = valid_z0(1, 3);

        let mut nonfinite_frequency = frequency;
        nonfinite_frequency[0] = f64::NAN;
        assert!(matches!(
            inner_connect_matched(&nonfinite_frequency, &s, &z0, 0, 2),
            Err(ConnectionError::NonFiniteFrequency {
                network: NetworkSide::A,
                index: 0,
                value,
            }) if value.is_nan()
        ));

        let mut nonfinite_s = s.clone();
        nonfinite_s[[0, 1, 0]] = complex(f64::INFINITY, 0.0);
        assert_eq!(
            inner_connect_matched(&frequency, &nonfinite_s, &z0, 0, 2).unwrap_err(),
            ConnectionError::NonFiniteS {
                network: NetworkSide::A,
                frequency: 0,
                row: 1,
                column: 0,
            }
        );

        let mut nonfinite_z0 = z0.clone();
        nonfinite_z0[[0, 1]] = complex(50.0, f64::NAN);
        assert_eq!(
            inner_connect_matched(&frequency, &s, &nonfinite_z0, 0, 2).unwrap_err(),
            ConnectionError::NonFiniteZ0 {
                network: NetworkSide::A,
                frequency: 0,
                port: 1,
            }
        );

        // Keep the internal system safe (`S_II=0`) and trigger overflow in the
        // external correction with finite operands instead.  This verifies
        // the operation's checked multiplication rather than solver pivot
        // ranking behavior.
        let mut overflowing = s.clone();
        overflowing[[0, 1, 0]] = complex(f64::MAX, 0.0);
        overflowing[[0, 2, 1]] = complex(2.0, 0.0);
        assert_eq!(
            inner_connect_matched(&frequency, &overflowing, &z0, 0, 2).unwrap_err(),
            ConnectionError::NonFiniteComputation {
                frequency: 0,
                row: 0,
                column: 0,
            }
        );
    }

    #[test]
    fn inner_connect_rejects_exact_singularity_but_accepts_nonzero_near_singularity() {
        let frequency = [1.0e9];
        let mut s = Array3::zeros((1, 3, 3));
        let z0 = valid_z0(1, 3);
        s[[0, 2, 0]] = complex(1.0, 0.0);
        assert_eq!(
            inner_connect_matched(&frequency, &s, &z0, 0, 2).unwrap_err(),
            ConnectionError::Singular { frequency: 0 }
        );

        s[[0, 2, 0]] = complex(1.0 - 2.0_f64.powi(-40), 0.0);
        s[[0, 1, 0]] = complex(1.0e-6, 0.0);
        s[[0, 2, 1]] = complex(1.0e-6, 0.0);
        let result = inner_connect_matched(&frequency, &s, &z0, 0, 2)
            .expect("non-zero near-singular internal system must not be cut off");
        assert!(result.s.iter().all(|value| is_finite(*value)));
        assert_ne!(result.s[[0, 0, 0]], ZERO);
    }

    #[test]
    fn connects_two_ideal_tees_and_preserves_documented_output_order() {
        let frequency = vec![1.0e9, 1.7e9];
        let s_a = ideal_tee(frequency.len());
        let s_b = ideal_tee(frequency.len());
        let z0_a = Array2::from_shape_vec(
            (2, 3),
            vec![
                complex(41.0, 1.0),
                complex(61.25, 0.0),
                complex(83.0, -2.0),
                complex(44.0, 1.5),
                complex(77.5, 0.0),
                complex(97.0, -2.5),
            ],
        )
        .expect("fixed A z0 shape must be valid");
        let z0_b = Array2::from_shape_vec(
            (2, 3),
            vec![
                complex(101.0, 3.0),
                complex(107.0, -1.0),
                complex(61.25, 0.0),
                complex(111.0, 3.5),
                complex(117.0, -1.5),
                complex(77.5, 0.0),
            ],
        )
        .expect("fixed B z0 shape must be valid");

        let result = connect_matched(&frequency, &s_a, &z0_a, 1, &frequency, &s_b, &z0_b, 2)
            .expect("ideal tee connection must succeed");

        assert_eq!(result.frequency_hz, frequency);
        assert_eq!(result.s.dim(), (2, 4, 4));
        for frequency_index in 0..2 {
            for row in 0..4 {
                assert_eq!(result.s[[frequency_index, row, row]], complex(-0.5, 0.0));
                for column in 0..4 {
                    if row != column {
                        assert_eq!(result.s[[frequency_index, row, column]], complex(0.5, 0.0));
                    }
                }
            }
        }
        assert_eq!(
            result.z0,
            Array2::from_shape_vec(
                (2, 4),
                vec![
                    complex(41.0, 1.0),
                    complex(83.0, -2.0),
                    complex(101.0, 3.0),
                    complex(107.0, -1.0),
                    complex(44.0, 1.5),
                    complex(97.0, -2.5),
                    complex(111.0, 3.5),
                    complex(117.0, -1.5),
                ],
            )
            .expect("fixed output z0 shape must be valid")
        );
    }

    #[test]
    fn rejects_invalid_ports_and_zero_external_port_result() {
        let (frequency_a, s_a, z0_a, frequency_b, s_b, z0_b) = valid_inputs();
        assert_eq!(
            connect_matched(&frequency_a, &s_a, &z0_a, 3, &frequency_b, &s_b, &z0_b, 0,)
                .unwrap_err(),
            ConnectionError::InvalidPort {
                network: NetworkSide::A,
                port: 3,
                nports: 3,
            }
        );
        assert_eq!(
            connect_matched(&frequency_a, &s_a, &z0_a, 0, &frequency_b, &s_b, &z0_b, 3,)
                .unwrap_err(),
            ConnectionError::InvalidPort {
                network: NetworkSide::B,
                port: 3,
                nports: 3,
            }
        );

        let one_port_s = Array3::from_elem((1, 1, 1), ZERO);
        let one_port_z0 = valid_z0(1, 1);
        assert_eq!(
            connect_matched(
                &[1.0],
                &one_port_s,
                &one_port_z0,
                0,
                &[1.0],
                &one_port_s,
                &one_port_z0,
                0,
            )
            .unwrap_err(),
            ConnectionError::NoExternalPorts
        );
    }

    #[test]
    fn rejects_frequency_and_array_shape_mismatches() {
        let (frequency_a, s_a, z0_a, frequency_b, s_b, z0_b) = valid_inputs();

        let mut mismatched_frequency = frequency_b.clone();
        mismatched_frequency[1] += 1.0;
        assert!(matches!(
            connect_matched(
                &frequency_a,
                &s_a,
                &z0_a,
                0,
                &mismatched_frequency,
                &s_b,
                &z0_b,
                0,
            ),
            Err(ConnectionError::FrequencyMismatch { index: 1, .. })
        ));

        let frequency_b_three = vec![1.0e9, 1.7e9, 2.4e9];
        let s_b_three = ideal_tee(frequency_b_three.len());
        let z0_b_three = valid_z0(frequency_b_three.len(), 3);
        assert_eq!(
            connect_matched(
                &frequency_a,
                &s_a,
                &z0_a,
                0,
                &frequency_b_three,
                &s_b_three,
                &z0_b_three,
                0,
            )
            .unwrap_err(),
            ConnectionError::FrequencyLengthMismatch { a: 2, b: 3 }
        );

        let short_frequency = vec![1.0e9];
        assert_eq!(
            connect_matched(
                &frequency_a,
                &s_a,
                &z0_a,
                0,
                &short_frequency,
                &s_b,
                &z0_b,
                0,
            )
            .unwrap_err(),
            ConnectionError::FrequencyShape {
                network: NetworkSide::B,
                expected: 2,
                actual: 1,
            }
        );

        // A non-square S shape is rejected before any arithmetic.
        let bad_s = Array3::zeros((2, 3, 2));
        assert_eq!(
            connect_matched(&frequency_a, &bad_s, &z0_a, 0, &frequency_b, &s_b, &z0_b, 0,)
                .unwrap_err(),
            ConnectionError::InvalidSShape {
                network: NetworkSide::A,
                shape: (2, 3, 2),
            }
        );

        let bad_z0 = Array2::zeros((2, 2));
        assert_eq!(
            connect_matched(&frequency_a, &s_a, &bad_z0, 0, &frequency_b, &s_b, &z0_b, 0,)
                .unwrap_err(),
            ConnectionError::InvalidZ0Shape {
                network: NetworkSide::A,
                shape: (2, 2),
            }
        );

        let empty_s = Array3::zeros((0, 3, 3));
        let empty_z0 = Array2::zeros((0, 3));
        assert_eq!(
            connect_matched(&[], &empty_s, &empty_z0, 0, &[], &empty_s, &empty_z0, 0,).unwrap_err(),
            ConnectionError::EmptyFrequency {
                network: NetworkSide::A
            }
        );
    }

    #[test]
    fn rejects_nonfinite_frequency_s_and_z0_values() {
        let (frequency_a, s_a, z0_a, frequency_b, s_b, z0_b) = valid_inputs();

        let mut bad_frequency = frequency_a.clone();
        bad_frequency[0] = f64::NAN;
        assert!(matches!(
            connect_matched(
                &bad_frequency,
                &s_a,
                &z0_a,
                0,
                &frequency_b,
                &s_b,
                &z0_b,
                0,
            ),
            Err(ConnectionError::NonFiniteFrequency {
                network: NetworkSide::A,
                index: 0,
                value,
            }) if value.is_nan()
        ));

        let mut bad_s = s_a.clone();
        bad_s[[1, 2, 1]] = complex(f64::INFINITY, 0.0);
        assert_eq!(
            connect_matched(&frequency_a, &bad_s, &z0_a, 0, &frequency_b, &s_b, &z0_b, 0,)
                .unwrap_err(),
            ConnectionError::NonFiniteS {
                network: NetworkSide::A,
                frequency: 1,
                row: 2,
                column: 1,
            }
        );

        let mut bad_z0 = z0_b.clone();
        bad_z0[[0, 2]] = complex(50.0, f64::NAN);
        assert_eq!(
            connect_matched(&frequency_a, &s_a, &z0_a, 0, &frequency_b, &s_b, &bad_z0, 0,)
                .unwrap_err(),
            ConnectionError::NonFiniteZ0 {
                network: NetworkSide::B,
                frequency: 0,
                port: 2,
            }
        );

        let mut infinite_z0 = z0_a.clone();
        infinite_z0[[1, 1]] = complex(f64::INFINITY, 0.0);
        assert_eq!(
            connect_matched(
                &frequency_a,
                &s_a,
                &infinite_z0,
                0,
                &frequency_b,
                &s_b,
                &z0_b,
                0,
            )
            .unwrap_err(),
            ConnectionError::NonFiniteZ0 {
                network: NetworkSide::A,
                frequency: 1,
                port: 1,
            }
        );
    }

    #[test]
    fn enforces_real_positive_and_exactly_equal_junction_impedances() {
        let (frequency_a, s_a, z0_a, frequency_b, s_b, z0_b) = valid_inputs();

        let mut imaginary_a = z0_a.clone();
        imaginary_a[[0, 0]] = complex(61.25, 1.0);
        assert!(matches!(
            connect_matched(
                &frequency_a,
                &s_a,
                &imaginary_a,
                0,
                &frequency_b,
                &s_b,
                &z0_b,
                0,
            ),
            Err(ConnectionError::InvalidJunctionZ0 {
                network: NetworkSide::A,
                frequency: 0,
                port: 0,
                ..
            })
        ));

        for invalid in [0.0, -1.0] {
            let mut invalid_a = z0_a.clone();
            invalid_a[[0, 0]] = complex(invalid, 0.0);
            assert!(matches!(
                connect_matched(
                    &frequency_a,
                    &s_a,
                    &invalid_a,
                    0,
                    &frequency_b,
                    &s_b,
                    &z0_b,
                    0,
                ),
                Err(ConnectionError::InvalidJunctionZ0 {
                    network: NetworkSide::A,
                    frequency: 0,
                    port: 0,
                    ..
                })
            ));
        }

        let mut invalid_b = z0_b.clone();
        invalid_b[[1, 0]] = complex(61.25, -0.5);
        assert!(matches!(
            connect_matched(
                &frequency_a,
                &s_a,
                &z0_a,
                0,
                &frequency_b,
                &s_b,
                &invalid_b,
                0,
            ),
            Err(ConnectionError::InvalidJunctionZ0 {
                network: NetworkSide::B,
                frequency: 1,
                port: 0,
                ..
            })
        ));

        let mut unequal_b = z0_b.clone();
        unequal_b[[1, 0]] = complex(62.0, 0.0);
        assert!(matches!(
            connect_matched(
                &frequency_a,
                &s_a,
                &z0_a,
                0,
                &frequency_b,
                &s_b,
                &unequal_b,
                0,
            ),
            Err(ConnectionError::MismatchedJunctionZ0 { frequency: 1, .. })
        ));
    }

    #[test]
    fn rejects_exact_singularity_but_accepts_nonzero_near_singularity() {
        let frequency = [1.0e9];
        let mut s_a = Array3::zeros((1, 2, 2));
        let mut s_b = Array3::zeros((1, 2, 2));
        s_a[[0, 0, 0]] = complex(1.0, 0.0);
        s_b[[0, 0, 0]] = complex(1.0, 0.0);
        let z0_a = valid_z0(1, 2);
        let z0_b = valid_z0(1, 2);
        assert_eq!(
            connect_matched(&frequency, &s_a, &z0_a, 0, &frequency, &s_b, &z0_b, 0).unwrap_err(),
            ConnectionError::Singular { frequency: 0 }
        );

        s_a[[0, 0, 0]] = complex(1.0 - 2.0_f64.powi(-40), 0.0);
        s_a[[0, 1, 0]] = complex(1.0e-6, 0.0);
        s_a[[0, 0, 1]] = complex(1.0e-6, 0.0);
        s_b[[0, 1, 0]] = complex(1.0e-6, 0.0);
        s_b[[0, 0, 1]] = complex(1.0e-6, 0.0);
        let result = connect_matched(&frequency, &s_a, &z0_a, 0, &frequency, &s_b, &z0_b, 0)
            .expect("non-zero near-singular denominator must not be cut off");
        assert!(result.s.iter().all(|value| is_finite(*value)));
        assert_ne!(result.s[[0, 0, 0]], ZERO);
    }

    #[test]
    fn inner_connect_handles_zero_leading_pivot_with_tiny_nonzero_candidate() {
        let frequency = [1.0e9];
        let mut s = Array3::zeros((1, 3, 3));
        let z0 = valid_z0(1, 3);

        // For k=0 and l=2, the internal system is
        // [[0, -1], [-1e-200, 1]].  It is invertible, but a norm_sqr-based
        // pivot ranking would underflow the non-zero first-column candidate.
        s[[0, 0, 0]] = complex(1.0e-200, 0.0);
        s[[0, 2, 0]] = complex(1.0, 0.0);
        s[[0, 2, 2]] = complex(1.0, 0.0);
        s[[0, 1, 1]] = complex(0.125, -0.25);

        let result = inner_connect_matched(&frequency, &s, &z0, 0, 2)
            .expect("tiny non-zero pivot candidate must not be treated as singular");
        assert_eq!(result.s.dim(), (1, 1, 1));
        // All external coupling terms are zero, so elimination preserves the
        // independently known S_EE value exactly.
        assert_eq!(result.s[[0, 0, 0]], complex(0.125, -0.25));
        assert!(result.s.iter().all(|value| is_finite(*value)));
    }

    #[test]
    fn inner_connect_preserves_minimum_subnormal_correction() {
        let frequency = [1.0e9];
        let minimum_subnormal = f64::from_bits(1);
        let mut s = Array3::zeros((1, 3, 3));
        let z0 = valid_z0(1, 3);

        // With k=0 and l=2, A = I - P S_II is
        // [[1, 0], [-m-im, 2+i]].  The only non-zero correction is
        // X[1,1] = (m+i m)/(2+i), which rounds to (m, 0).
        s[[0, 0, 0]] = complex(minimum_subnormal, minimum_subnormal);
        s[[0, 0, 2]] = complex(-1.0, -1.0);
        s[[0, 1, 2]] = complex(1.0, 0.0);
        s[[0, 2, 1]] = complex(1.0, 0.0);

        let result = inner_connect_matched(&frequency, &s, &z0, 0, 2)
            .expect("minimum subnormal correction must remain finite");
        assert_eq!(result.s[[0, 0, 0]], complex(minimum_subnormal, 0.0));
    }

    #[test]
    fn reports_finite_input_arithmetic_overflow() {
        let frequency = [1.0e9];
        let mut s_a = Array3::zeros((1, 2, 2));
        let mut s_b = Array3::zeros((1, 2, 2));
        s_a[[0, 0, 0]] = complex(f64::MAX, 0.0);
        s_b[[0, 0, 0]] = complex(2.0, 0.0);
        let z0_a = valid_z0(1, 2);
        let z0_b = valid_z0(1, 2);

        assert_eq!(
            connect_matched(&frequency, &s_a, &z0_a, 0, &frequency, &s_b, &z0_b, 0).unwrap_err(),
            ConnectionError::NonFiniteComputation {
                frequency: 0,
                row: 0,
                column: 0,
            }
        );
    }

    #[test]
    fn connect_matched_preserves_issue_44_extreme_components_exactly() {
        let frequency = [1.0e9];
        let huge = 2.0_f64.powi(1000);
        let tiny = 2.0_f64.powi(-1000);
        let expected_tiny = f64::from_bits(1_u64 << 49);

        let mut s_a = Array3::zeros((1, 2, 2));
        s_a[[0, 0, 1]] = complex(huge, tiny);
        s_a[[0, 1, 1]] = complex(2.0_f64.powi(500), 0.0);

        let mut s_b = Array3::zeros((1, 2, 2));
        s_b[[0, 0, 0]] = complex(2.0_f64.powi(-500), -2.0_f64.powi(-475));
        s_b[[0, 0, 1]] = complex(1.0, 0.0);

        let z0_a = valid_z0(1, 2);
        let z0_b = valid_z0(1, 2);
        let result = connect_matched(&frequency, &s_a, &z0_a, 1, &frequency, &s_b, &z0_b, 0)
            .expect("Issue #44 extreme quotient must remain finite");

        assert_eq!(
            result.s[[0, 0, 1]],
            complex(expected_tiny, -2.0_f64.powi(975))
        );
    }

    #[test]
    fn rejects_unrepresentable_complex_quotient_as_nonfinite_computation() {
        let frequency = [1.0e9];
        let mut s_a = Array3::zeros((1, 2, 2));
        s_a[[0, 0, 1]] = complex(f64::MAX, 0.0);
        s_a[[0, 1, 1]] = complex(0.5, 0.0);

        let mut s_b = Array3::zeros((1, 2, 2));
        s_b[[0, 0, 1]] = complex(1.0, 0.0);
        s_b[[0, 0, 0]] = complex(1.0, 0.0);

        let z0_a = valid_z0(1, 2);
        let z0_b = valid_z0(1, 2);
        assert_eq!(
            connect_matched(&frequency, &s_a, &z0_a, 1, &frequency, &s_b, &z0_b, 0).unwrap_err(),
            ConnectionError::NonFiniteComputation {
                frequency: 0,
                row: 0,
                column: 1,
            }
        );
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ConnectFixture {
        data: ConnectData,
        metadata: ConnectMetadata,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ConnectData {
        frequency_hz: Vec<f64>,
        s_a: Vec<Vec<Vec<ComplexValue>>>,
        s_b: Vec<Vec<Vec<ComplexValue>>>,
        s_connected: Vec<Vec<Vec<ComplexValue>>>,
        z0_a_ohm: Vec<Vec<ComplexValue>>,
        z0_b_ohm: Vec<Vec<ComplexValue>>,
        z0_connected_ohm: Vec<Vec<ComplexValue>>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ComplexValue {
        imag: f64,
        real: f64,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ConnectMetadata {
        case_id: String,
        junction_ports: JunctionPorts,
        numpy_version: String,
        operation: String,
        port_order: PortOrder,
        random_seeds: RandomSeeds,
        reference_impedance: ReferenceImpedance,
        schema: String,
        schema_version: u32,
        scikit_rf_version: String,
        shape: ConnectShape,
        tolerance_policy: TolerancePolicy,
        wave_definition: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct JunctionPorts {
        a: usize,
        b: usize,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PortOrder {
        a_survivors: Vec<usize>,
        b_survivors: Vec<usize>,
        output: Vec<PortLabel>,
        description: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct PortLabel {
        network: String,
        port: usize,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RandomSeeds {
        a: u64,
        b: u64,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ReferenceImpedance {
        external_complex_allowed: bool,
        junction_exactly_equal: bool,
        junction_frequency_dependent: bool,
        junction_real_strictly_positive: bool,
        unit: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ConnectShape {
        frequency: Vec<usize>,
        input_s_a: Vec<usize>,
        input_s_b: Vec<usize>,
        input_z0_a: Vec<usize>,
        input_z0_b: Vec<usize>,
        output_s: Vec<usize>,
        output_z0: Vec<usize>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct TolerancePolicy {
        atol: f64,
        comparison: String,
        justification: String,
        regeneration: String,
        rtol: f64,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct InnerConnectFixture {
        data: InnerConnectData,
        metadata: InnerConnectMetadata,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct InnerConnectData {
        frequency_hz: Vec<f64>,
        s: Vec<Vec<Vec<ComplexValue>>>,
        s_inner_connected: Vec<Vec<Vec<ComplexValue>>>,
        z0_ohm: Vec<Vec<ComplexValue>>,
        z0_inner_connected_ohm: Vec<Vec<ComplexValue>>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct InnerConnectMetadata {
        case_id: String,
        junction_ports: InnerJunctionPorts,
        numpy_version: String,
        operation: String,
        port_order: InnerPortOrder,
        random_seed: u64,
        reference_impedance: ReferenceImpedance,
        schema: String,
        schema_version: u32,
        scikit_rf_version: String,
        shape: InnerConnectShape,
        tolerance_policy: TolerancePolicy,
        wave_definition: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct InnerJunctionPorts {
        k: usize,
        l: usize,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct InnerPortOrder {
        description: String,
        output: Vec<usize>,
        survivors: Vec<usize>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct InnerConnectShape {
        frequency: Vec<usize>,
        input_s: Vec<usize>,
        input_z0: Vec<usize>,
        output_s: Vec<usize>,
        output_z0: Vec<usize>,
    }

    fn fixture_array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
        let nfreq = values.len();
        let nports = values[0].len();
        Array3::from_shape_fn((nfreq, nports, nports), |(frequency, row, column)| {
            let value = &values[frequency][row][column];
            complex(value.real, value.imag)
        })
    }

    fn fixture_array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
        let nfreq = values.len();
        let nports = values[0].len();
        Array2::from_shape_fn((nfreq, nports), |(frequency, port)| {
            let value = &values[frequency][port];
            complex(value.real, value.imag)
        })
    }

    fn assert_close(actual: Complex64, expected: Complex64, rtol: f64, atol: f64) {
        let difference = (actual - expected).norm();
        let bound = atol + rtol * expected.norm();
        assert!(
            difference <= bound,
            "actual={actual:?}, expected={expected:?}, difference={difference:e}, bound={bound:e}"
        );
    }

    #[test]
    fn matches_checked_in_matched_connection_oracle_and_contract_metadata() {
        let fixture: ConnectFixture =
            serde_json::from_str(CONNECT_FIXTURE_JSON).expect("fixture must parse");
        assert_eq!(
            fixture.metadata.case_id,
            "power_wave_connect_matched_three_to_four_port_real_frequency_dependent_z0"
        );
        assert_eq!(fixture.metadata.operation, "connect_matched");
        assert_eq!(fixture.metadata.wave_definition, "power");
        assert_eq!(fixture.metadata.schema, "rfkit-rs.oracle.fixture");
        assert_eq!(fixture.metadata.schema_version, 1);
        assert_eq!(fixture.metadata.numpy_version, "2.5.1");
        assert_eq!(fixture.metadata.scikit_rf_version, "2.0.1");
        assert_eq!(fixture.metadata.random_seeds.a, 20_260_939);
        assert_eq!(fixture.metadata.random_seeds.b, 20_260_940);
        assert_eq!(fixture.metadata.junction_ports.a, 1);
        assert_eq!(fixture.metadata.junction_ports.b, 2);
        assert!(
            fixture
                .metadata
                .reference_impedance
                .external_complex_allowed
        );
        assert!(fixture.metadata.reference_impedance.junction_exactly_equal);
        assert!(
            fixture
                .metadata
                .reference_impedance
                .junction_frequency_dependent
        );
        assert!(
            fixture
                .metadata
                .reference_impedance
                .junction_real_strictly_positive
        );
        assert_eq!(fixture.metadata.reference_impedance.unit, "ohm");
        assert_eq!(fixture.metadata.port_order.a_survivors, vec![0, 2]);
        assert_eq!(fixture.metadata.port_order.b_survivors, vec![0, 1, 3]);
        assert_eq!(
            fixture.metadata.port_order.description,
            "A unconnected ports in original order, followed by B unconnected ports in original order"
        );
        let output_order: Vec<(&str, usize)> = fixture
            .metadata
            .port_order
            .output
            .iter()
            .map(|port| (port.network.as_str(), port.port))
            .collect();
        assert_eq!(
            output_order,
            vec![("A", 0), ("A", 2), ("B", 0), ("B", 1), ("B", 3)]
        );
        assert_eq!(fixture.metadata.shape.frequency, vec![3]);
        assert_eq!(fixture.metadata.shape.input_s_a, vec![3, 3, 3]);
        assert_eq!(fixture.metadata.shape.input_s_b, vec![3, 4, 4]);
        assert_eq!(fixture.metadata.shape.input_z0_a, vec![3, 3]);
        assert_eq!(fixture.metadata.shape.input_z0_b, vec![3, 4]);
        assert_eq!(fixture.metadata.shape.output_s, vec![3, 5, 5]);
        assert_eq!(fixture.metadata.shape.output_z0, vec![3, 5]);
        assert_eq!(fixture.metadata.tolerance_policy.rtol, 1e-12);
        assert_eq!(fixture.metadata.tolerance_policy.atol, 1e-12);
        assert_eq!(
            fixture.metadata.tolerance_policy.comparison,
            "abs(actual-expected) <= atol + rtol*abs(expected)"
        );
        assert!(!fixture.metadata.tolerance_policy.justification.is_empty());
        assert!(!fixture.metadata.tolerance_policy.regeneration.is_empty());

        let s_a = fixture_array3(&fixture.data.s_a);
        let s_b = fixture_array3(&fixture.data.s_b);
        let z0_a = fixture_array2(&fixture.data.z0_a_ohm);
        let z0_b = fixture_array2(&fixture.data.z0_b_ohm);
        let expected_s = fixture_array3(&fixture.data.s_connected);
        let expected_z0 = fixture_array2(&fixture.data.z0_connected_ohm);
        let result = connect_matched(
            &fixture.data.frequency_hz,
            &s_a,
            &z0_a,
            fixture.metadata.junction_ports.a,
            &fixture.data.frequency_hz,
            &s_b,
            &z0_b,
            fixture.metadata.junction_ports.b,
        )
        .expect("oracle connection must succeed");

        assert_eq!(result.frequency_hz, fixture.data.frequency_hz);
        assert_eq!(result.z0, expected_z0, "output z0/order must be exact");
        assert_eq!(result.s.dim(), expected_s.dim());
        for (actual, expected) in result.s.iter().zip(expected_s.iter()) {
            assert_close(
                *actual,
                *expected,
                fixture.metadata.tolerance_policy.rtol,
                fixture.metadata.tolerance_policy.atol,
            );
        }
    }

    #[test]
    fn matches_checked_in_inner_connect_oracle_and_contract_metadata() {
        let fixture: InnerConnectFixture =
            serde_json::from_str(INNER_CONNECT_FIXTURE_JSON).expect("fixture must parse");
        assert_eq!(
            fixture.metadata.case_id,
            "power_wave_inner_connect_matched_five_port_real_frequency_dependent_z0"
        );
        assert_eq!(fixture.metadata.operation, "inner_connect_matched");
        assert_eq!(fixture.metadata.wave_definition, "power");
        assert_eq!(fixture.metadata.schema, "rfkit-rs.oracle.fixture");
        assert_eq!(fixture.metadata.schema_version, 1);
        assert_eq!(fixture.metadata.numpy_version, "2.5.1");
        assert_eq!(fixture.metadata.scikit_rf_version, "2.0.1");
        assert_eq!(fixture.metadata.random_seed, 20_260_941);
        assert_eq!(fixture.metadata.junction_ports.k, 1);
        assert_eq!(fixture.metadata.junction_ports.l, 3);
        assert!(
            fixture
                .metadata
                .reference_impedance
                .external_complex_allowed
        );
        assert!(fixture.metadata.reference_impedance.junction_exactly_equal);
        assert!(
            fixture
                .metadata
                .reference_impedance
                .junction_frequency_dependent
        );
        assert!(
            fixture
                .metadata
                .reference_impedance
                .junction_real_strictly_positive
        );
        assert_eq!(fixture.metadata.reference_impedance.unit, "ohm");
        assert_eq!(fixture.metadata.port_order.survivors, vec![0, 2, 4]);
        assert_eq!(fixture.metadata.port_order.output, vec![0, 2, 4]);
        assert_eq!(
            fixture.metadata.port_order.description,
            "Input ports excluding k and l, in original order"
        );
        assert_eq!(fixture.metadata.shape.frequency, vec![3]);
        assert_eq!(fixture.metadata.shape.input_s, vec![3, 5, 5]);
        assert_eq!(fixture.metadata.shape.input_z0, vec![3, 5]);
        assert_eq!(fixture.metadata.shape.output_s, vec![3, 3, 3]);
        assert_eq!(fixture.metadata.shape.output_z0, vec![3, 3]);
        assert_eq!(fixture.metadata.tolerance_policy.rtol, 1e-12);
        assert_eq!(fixture.metadata.tolerance_policy.atol, 1e-12);
        assert_eq!(
            fixture.metadata.tolerance_policy.comparison,
            "abs(actual-expected) <= atol + rtol*abs(expected)"
        );
        assert!(!fixture.metadata.tolerance_policy.justification.is_empty());
        assert!(!fixture.metadata.tolerance_policy.regeneration.is_empty());

        let s = fixture_array3(&fixture.data.s);
        let z0 = fixture_array2(&fixture.data.z0_ohm);
        let expected_s = fixture_array3(&fixture.data.s_inner_connected);
        let expected_z0 = fixture_array2(&fixture.data.z0_inner_connected_ohm);
        let result = inner_connect_matched(
            &fixture.data.frequency_hz,
            &s,
            &z0,
            fixture.metadata.junction_ports.k,
            fixture.metadata.junction_ports.l,
        )
        .expect("oracle inner connection must succeed");

        assert_eq!(result.frequency_hz, fixture.data.frequency_hz);
        assert_eq!(result.z0, expected_z0, "output z0/order must be exact");
        assert_eq!(result.s.dim(), expected_s.dim());
        for (actual, expected) in result.s.iter().zip(expected_s.iter()) {
            assert_close(
                *actual,
                *expected,
                fixture.metadata.tolerance_policy.rtol,
                fixture.metadata.tolerance_policy.atol,
            );
        }
        for frequency in 0..fixture.data.frequency_hz.len() {
            let junction_k = z0[[frequency, fixture.metadata.junction_ports.k]];
            let junction_l = z0[[frequency, fixture.metadata.junction_ports.l]];
            assert_eq!(junction_k, junction_l);
            assert!(junction_k.re > 0.0);
            assert_eq!(junction_k.im, 0.0);
            assert_ne!(junction_k.re, 50.0);
        }
    }
}
