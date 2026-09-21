//! Direct physical connection of two Kurokawa power-wave networks.
//!
//! This module eliminates one port from each of two frequency-major scattering
//! networks by imposing the physical junction conditions
//! `V_A = V_B` and `I_A + I_B = 0`.  Currents point into each network.  For a
//! finite reference `z` with nonzero real part, the repository's Kurokawa
//! power waves are
//!
//! ```text
//! a = (V + z I) / (2 sqrt(abs(Re(z))))
//! b = (V - conj(z) I) / (2 sqrt(abs(Re(z))))
//! q = sqrt(abs(Re(z))) / Re(z)
//! I = q (a - b)
//! V = q (conj(z) a + z b).
//! ```
//!
//! With internal incident waves ordered as
//! `i = [a_A, a_B]` and internal reflected waves ordered as
//! `b_i = [b_A, b_B]`, the junction equations are `C i + D b_i = 0` with
//!
//! ```text
//! C = [[qA,             qB],
//!      [qA*conj(zA), -qB*conj(zB)]]
//! D = [[-qA,             -qB],
//!      [qA*zA,       -qB*zB]].
//! ```
//!
//! Since the two input networks are physically disjoint before this junction,
//! their internal scattering block is diagonal.  The implementation therefore
//! forms only the two-by-two system `C + D*S_ii`, solves
//! `(C + D*S_ii) X = -D`, and evaluates
//! `S_out = S_ee + S_ei*X*S_ie` directly.  It never converts through Z/Y,
//! constructs a whole-network inverse, inserts a mismatch network, changes a
//! wave convention, or regularizes a near-singular system.  A pivot is
//! singular only when the existing exact-pivot solver evaluates it as exactly
//! zero.  All finite-input arithmetic is checked so a finite overflow cannot
//! escape as a successful non-finite network.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::linalg;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Identifies which direct-connection input supplied a failing value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NetworkSide {
    A,
    B,
}

/// Failure modes for the direct physical junction kernel.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum DirectConnectionError {
    #[error(
        "{network:?} direct connection S-parameter shape must be (nfreq, nport, nport), got {shape:?}"
    )]
    InvalidSShape {
        network: NetworkSide,
        shape: (usize, usize, usize),
    },

    #[error(
        "{network:?} direct connection reference-impedance shape must be (nfreq, nport), got {shape:?}"
    )]
    InvalidZ0Shape {
        network: NetworkSide,
        shape: (usize, usize),
    },

    #[error("{network:?} direct connection frequency axis must not be empty")]
    EmptyFrequency { network: NetworkSide },

    #[error(
        "{network:?} direct connection frequency axis length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    FrequencyShape {
        network: NetworkSide,
        expected: usize,
        actual: usize,
    },

    #[error("A and B direct connection frequency axes have different lengths: A={a}, B={b}")]
    FrequencyLengthMismatch { a: usize, b: usize },

    #[error("direct connection frequency axes differ at index {index}: A={a:?}, B={b:?}")]
    FrequencyMismatch { index: usize, a: f64, b: f64 },

    #[error("{network:?} direct connection frequency is non-finite at index {index}: {value:?}")]
    NonFiniteFrequency {
        network: NetworkSide,
        index: usize,
        value: f64,
    },

    #[error(
        "{network:?} direct connection S-parameter is non-finite at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteS {
        network: NetworkSide,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "{network:?} direct connection reference impedance is non-finite at frequency {frequency}, port {port}"
    )]
    NonFiniteZ0 {
        network: NetworkSide,
        frequency: usize,
        port: usize,
    },

    #[error("{network:?} direct connection port {port} is out of range for {nports} ports")]
    InvalidPort {
        network: NetworkSide,
        port: usize,
        nports: usize,
    },

    #[error(
        "{network:?} direct connection reference impedance has a zero real part at frequency {frequency}, port {port}"
    )]
    ZeroRealReferenceImpedance {
        network: NetworkSide,
        frequency: usize,
        port: usize,
    },

    #[error("direct connection leaves no external ports")]
    NoExternalPorts,

    #[error(
        "direct power-wave connection is exactly singular at frequency {frequency}, selected ports A={port_a}, B={port_b}, pivot {pivot}"
    )]
    Singular {
        frequency: usize,
        port_a: usize,
        port_b: usize,
        pivot: usize,
    },

    #[error(
        "non-finite value while evaluating direct power-wave connection at frequency {frequency}, selected ports A={port_a}, B={port_b}, row {row}, column {column}"
    )]
    NonFiniteComputation {
        frequency: usize,
        port_a: usize,
        port_b: usize,
        row: usize,
        column: usize,
    },
}

/// Result of a direct physical junction connection.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConnectedNetwork {
    /// A's frequency axis, copied bit-for-bit.
    pub(crate) frequency_hz: Vec<f64>,
    /// Output frequency-major S-parameter stack.
    pub(crate) s: Array3<Complex64>,
    /// Surviving A references followed by surviving B references.
    pub(crate) z0: Array2<Complex64>,
}

/// Connect one A port and one B port by direct voltage/current continuity.
///
/// The raw-array boundary is deliberate: `Network` values can be created by
/// serde with malformed dimensions, so every shape and axis invariant is
/// checked before any selected-port indexing.  The output order is A's
/// survivors in their original order followed by B's survivors in their
/// original order, including when either input is one-port.
#[allow(clippy::too_many_arguments)]
pub(crate) fn connect_direct(
    frequency_a: &[f64],
    s_a: &Array3<Complex64>,
    z0_a: &Array2<Complex64>,
    port_a: usize,
    frequency_b: &[f64],
    s_b: &Array3<Complex64>,
    z0_b: &Array2<Complex64>,
    port_b: usize,
) -> Result<ConnectedNetwork, DirectConnectionError> {
    let shape_a = validate_input(NetworkSide::A, frequency_a, s_a, z0_a)?;
    let shape_b = validate_input(NetworkSide::B, frequency_b, s_b, z0_b)?;

    if frequency_a.len() != frequency_b.len() {
        return Err(DirectConnectionError::FrequencyLengthMismatch {
            a: frequency_a.len(),
            b: frequency_b.len(),
        });
    }
    for (index, (&a, &b)) in frequency_a.iter().zip(frequency_b).enumerate() {
        if a != b {
            return Err(DirectConnectionError::FrequencyMismatch { index, a, b });
        }
    }

    if port_a >= shape_a.nports {
        return Err(DirectConnectionError::InvalidPort {
            network: NetworkSide::A,
            port: port_a,
            nports: shape_a.nports,
        });
    }
    if port_b >= shape_b.nports {
        return Err(DirectConnectionError::InvalidPort {
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
        .ok_or(DirectConnectionError::NoExternalPorts)?;
    if n_external == 0 {
        return Err(DirectConnectionError::NoExternalPorts);
    }

    // Validate every reference before selecting either junction coordinate.
    // Complex and negative-real references are intentional parts of this
    // operation's domain; only a zero real part is undefined for q.
    validate_reference_real_parts(NetworkSide::A, z0_a)?;
    validate_reference_real_parts(NetworkSide::B, z0_b)?;

    let external = external_coordinates(&external_a, &external_b);

    let mut output_s = Array3::from_elem((shape_a.nfreq, n_external, n_external), ZERO);
    let mut output_z0 = Array2::from_elem((shape_a.nfreq, n_external), ZERO);

    for frequency in 0..shape_a.nfreq {
        for (output_port, &input_port) in external_a.iter().enumerate() {
            output_z0[[frequency, output_port]] = z0_a[[frequency, input_port]];
        }
        for (offset, &input_port) in external_b.iter().enumerate() {
            output_z0[[frequency, external_a.len() + offset]] = z0_b[[frequency, input_port]];
        }

        let z_a = z0_a[[frequency, port_a]];
        let z_b = z0_b[[frequency, port_b]];
        let q_a = q_for_reference(z_a, frequency, port_a, port_b)?;
        let q_b = q_for_reference(z_b, frequency, port_a, port_b)?;

        // Build C and D from the physical voltage/current equations.  The
        // arrays are row-major because that is the storage expected by the
        // shared exact-pivot solver.
        let mut c = [ZERO; 4];
        c[0] = q_a;
        c[1] = q_b;
        c[2] = checked_mul(q_a, z_a.conj(), frequency, port_a, port_b, 1, 0)?;
        c[3] = checked_neg(
            checked_mul(q_b, z_b.conj(), frequency, port_a, port_b, 1, 1)?,
            frequency,
            port_a,
            port_b,
            1,
            1,
        )?;

        let mut d = [ZERO; 4];
        d[0] = checked_neg(q_a, frequency, port_a, port_b, 0, 0)?;
        d[1] = checked_neg(q_b, frequency, port_a, port_b, 0, 1)?;
        d[2] = checked_mul(q_a, z_a, frequency, port_a, port_b, 1, 0)?;
        d[3] = checked_neg(
            checked_mul(q_b, z_b, frequency, port_a, port_b, 1, 1)?,
            frequency,
            port_a,
            port_b,
            1,
            1,
        )?;

        let s_aa = s_a[[frequency, port_a, port_a]];
        let s_bb = s_b[[frequency, port_b, port_b]];

        // A2 = C + D*S_ii, where S_ii is diag(s_aa, s_bb).
        let mut a2 = [ZERO; 4];
        a2[0] = checked_add(
            c[0],
            checked_mul(d[0], s_aa, frequency, port_a, port_b, 0, 0)?,
            frequency,
            port_a,
            port_b,
            0,
            0,
        )?;
        a2[1] = checked_add(
            c[1],
            checked_mul(d[1], s_bb, frequency, port_a, port_b, 0, 1)?,
            frequency,
            port_a,
            port_b,
            0,
            1,
        )?;
        a2[2] = checked_add(
            c[2],
            checked_mul(d[2], s_aa, frequency, port_a, port_b, 1, 0)?,
            frequency,
            port_a,
            port_b,
            1,
            0,
        )?;
        a2[3] = checked_add(
            c[3],
            checked_mul(d[3], s_bb, frequency, port_a, port_b, 1, 1)?,
            frequency,
            port_a,
            port_b,
            1,
            1,
        )?;

        // Solve A2*X = -D.  The existing solver performs scale-safe pivot
        // ranking and exact-zero pivot detection; no operation-specific
        // tolerance or fallback is introduced here.
        let mut x = [
            checked_neg(d[0], frequency, port_a, port_b, 0, 0)?,
            checked_neg(d[1], frequency, port_a, port_b, 0, 1)?,
            checked_neg(d[2], frequency, port_a, port_b, 1, 0)?,
            checked_neg(d[3], frequency, port_a, port_b, 1, 1)?,
        ];
        match linalg::solve_multiple_rhs(&mut a2, &mut x, 2) {
            Ok(()) => {}
            Err(linalg::SolveError::InvalidStorage { .. }) => {
                unreachable!("direct-connection solver storage is fixed 2x2")
            }
            Err(linalg::SolveError::Singular { pivot }) => {
                return Err(DirectConnectionError::Singular {
                    frequency,
                    port_a,
                    port_b,
                    pivot,
                });
            }
            Err(linalg::SolveError::NonFinite { row, column }) => {
                return Err(DirectConnectionError::NonFiniteComputation {
                    frequency,
                    port_a,
                    port_b,
                    row,
                    column,
                });
            }
        }

        for (output_row, &(row_side, row_port)) in external.iter().enumerate() {
            for (output_column, &(column_side, column_port)) in external.iter().enumerate() {
                let left = match row_side {
                    NetworkSide::A => [s_a[[frequency, row_port, port_a]], ZERO],
                    NetworkSide::B => [ZERO, s_b[[frequency, row_port, port_b]]],
                };
                let right = match column_side {
                    NetworkSide::A => [s_a[[frequency, port_a, column_port]], ZERO],
                    NetworkSide::B => [ZERO, s_b[[frequency, port_b, column_port]]],
                };

                let mut correction = ZERO;
                for internal_row in 0..2 {
                    for internal_column in 0..2 {
                        let term = checked_mul(
                            left[internal_row],
                            x[internal_row * 2 + internal_column],
                            frequency,
                            port_a,
                            port_b,
                            output_row,
                            output_column,
                        )?;
                        let term = checked_mul(
                            term,
                            right[internal_column],
                            frequency,
                            port_a,
                            port_b,
                            output_row,
                            output_column,
                        )?;
                        correction = checked_add(
                            correction,
                            term,
                            frequency,
                            port_a,
                            port_b,
                            output_row,
                            output_column,
                        )?;
                    }
                }

                let direct = match (row_side, column_side) {
                    (NetworkSide::A, NetworkSide::A) => s_a[[frequency, row_port, column_port]],
                    (NetworkSide::A, NetworkSide::B) | (NetworkSide::B, NetworkSide::A) => ZERO,
                    (NetworkSide::B, NetworkSide::B) => s_b[[frequency, row_port, column_port]],
                };
                output_s[[frequency, output_row, output_column]] = checked_add(
                    direct,
                    correction,
                    frequency,
                    port_a,
                    port_b,
                    output_row,
                    output_column,
                )?;
            }
        }
    }

    // Keep a result-wide guard near the public boundary.  It is redundant
    // with checked arithmetic today, but protects against future edits that
    // add an unguarded assignment.
    for (index, &value) in output_s.indexed_iter() {
        if !is_finite(value) {
            return Err(DirectConnectionError::NonFiniteComputation {
                frequency: index.0,
                port_a,
                port_b,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in output_z0.indexed_iter() {
        if !is_finite(value) {
            return Err(DirectConnectionError::NonFiniteComputation {
                frequency: index.0,
                port_a,
                port_b,
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
) -> Result<InputShape, DirectConnectionError> {
    let shape = s.dim();
    if shape.1 == 0 || shape.1 != shape.2 {
        return Err(DirectConnectionError::InvalidSShape { network, shape });
    }
    if frequency.is_empty() || shape.0 == 0 {
        return Err(DirectConnectionError::EmptyFrequency { network });
    }
    if frequency.len() != shape.0 {
        return Err(DirectConnectionError::FrequencyShape {
            network,
            expected: shape.0,
            actual: frequency.len(),
        });
    }
    let z0_shape = z0.dim();
    if z0_shape != (shape.0, shape.1) {
        return Err(DirectConnectionError::InvalidZ0Shape {
            network,
            shape: z0_shape,
        });
    }

    for (index, &value) in frequency.iter().enumerate() {
        if !value.is_finite() {
            return Err(DirectConnectionError::NonFiniteFrequency {
                network,
                index,
                value,
            });
        }
    }
    for (index, &value) in s.indexed_iter() {
        if !is_finite(value) {
            return Err(DirectConnectionError::NonFiniteS {
                network,
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in z0.indexed_iter() {
        if !is_finite(value) {
            return Err(DirectConnectionError::NonFiniteZ0 {
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

fn validate_reference_real_parts(
    network: NetworkSide,
    z0: &Array2<Complex64>,
) -> Result<(), DirectConnectionError> {
    for (index, &reference) in z0.indexed_iter() {
        if reference.re == 0.0 {
            return Err(DirectConnectionError::ZeroRealReferenceImpedance {
                network,
                frequency: index.0,
                port: index.1,
            });
        }
    }
    Ok(())
}

fn survivors(nports: usize, selected: usize) -> Vec<usize> {
    (0..nports).filter(|&port| port != selected).collect()
}

fn external_coordinates(external_a: &[usize], external_b: &[usize]) -> Vec<(NetworkSide, usize)> {
    let mut coordinates = Vec::with_capacity(external_a.len() + external_b.len());
    coordinates.extend(
        external_a
            .iter()
            .copied()
            .map(|port| (NetworkSide::A, port)),
    );
    coordinates.extend(
        external_b
            .iter()
            .copied()
            .map(|port| (NetworkSide::B, port)),
    );
    coordinates
}

fn q_for_reference(
    reference: Complex64,
    frequency: usize,
    port_a: usize,
    port_b: usize,
) -> Result<Complex64, DirectConnectionError> {
    let q = Complex64::new(reference.re.abs().sqrt() / reference.re, 0.0);
    checked_value(q, frequency, port_a, port_b, 0, 0)
}

fn checked_add(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    port_a: usize,
    port_b: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, DirectConnectionError> {
    checked_value(left + right, frequency, port_a, port_b, row, column)
}

fn checked_mul(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    port_a: usize,
    port_b: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, DirectConnectionError> {
    checked_value(left * right, frequency, port_a, port_b, row, column)
}

fn checked_neg(
    value: Complex64,
    frequency: usize,
    port_a: usize,
    port_b: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, DirectConnectionError> {
    checked_value(-value, frequency, port_a, port_b, row, column)
}

fn checked_value(
    value: Complex64,
    frequency: usize,
    port_a: usize,
    port_b: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, DirectConnectionError> {
    if is_finite(value) {
        Ok(value)
    } else {
        Err(DirectConnectionError::NonFiniteComputation {
            frequency,
            port_a,
            port_b,
            row,
            column,
        })
    }
}

fn is_finite(value: Complex64) -> bool {
    linalg::is_finite(value)
}
