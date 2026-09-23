//! Simultaneous direct power-wave cascade of two coupled even-port networks.
//!
//! The operation eliminates all paired internal ports in one joint physical
//! boundary solve.  For each input, ports are ordered as
//! `[left_0..left_(N-1), right_0..right_(N-1)]`; every right port of A is
//! connected to the corresponding left port of B.  The external coordinates
//! are `[A.left..., B.right...]` and the internal coordinates are
//! `[A.right..., B.left...]`.
//!
//! With the Kurokawa equations used by [`crate::direct_connection`], the
//! physical boundary is `C a_i + D b_i = 0`.  Partitioning the two disjoint
//! source networks gives
//!
//! ```text
//! b_i = S_ie a_e + S_ii a_i
//! (C + D S_ii) X = -D S_ie
//! S_out = S_ee + S_ei X
//! ```
//!
//! This is intentionally a simultaneous N-port elimination.  In particular,
//! the full within-group blocks of `S_ii` are retained; a sequence of
//! one-port connections is not equivalent on the complete domain because an
//! intermediate junction can be singular while the joint system is not.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::direct_connection::NetworkSide;
use crate::linalg;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Failure modes for simultaneous direct even-port power-wave cascading.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum CascadeError {
    #[error(
        "{network:?} direct cascade S-parameter shape must be (nfreq, nport, nport), got {shape:?}"
    )]
    InvalidSShape {
        network: NetworkSide,
        shape: (usize, usize, usize),
    },

    #[error(
        "{network:?} direct cascade reference-impedance shape must be (nfreq, nport), got {shape:?}"
    )]
    InvalidZ0Shape {
        network: NetworkSide,
        shape: (usize, usize),
    },

    #[error("{network:?} direct cascade frequency axis must not be empty")]
    EmptyFrequency { network: NetworkSide },

    #[error(
        "{network:?} direct cascade frequency axis length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    FrequencyShape {
        network: NetworkSide,
        expected: usize,
        actual: usize,
    },

    #[error("A and B direct cascade frequency axes have different lengths: A={a}, B={b}")]
    FrequencyLengthMismatch { a: usize, b: usize },

    #[error("direct cascade frequency axes differ at index {index}: A={a:?}, B={b:?}")]
    FrequencyMismatch { index: usize, a: f64, b: f64 },

    #[error("{network:?} direct cascade requires a positive even port count, got {nports}")]
    InvalidPortCount { network: NetworkSide, nports: usize },

    #[error("direct cascade input port counts differ: A={a}, B={b}")]
    PortCountMismatch { a: usize, b: usize },

    #[error("{network:?} direct cascade frequency is non-finite at index {index}: {value:?}")]
    NonFiniteFrequency {
        network: NetworkSide,
        index: usize,
        value: f64,
    },

    #[error(
        "{network:?} direct cascade S-parameter is non-finite at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteS {
        network: NetworkSide,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "{network:?} direct cascade reference impedance is non-finite at frequency {frequency}, port {port}"
    )]
    NonFiniteZ0 {
        network: NetworkSide,
        frequency: usize,
        port: usize,
    },

    #[error(
        "{network:?} direct cascade reference impedance has a zero real part at frequency {frequency}, port {port}"
    )]
    ZeroRealReferenceImpedance {
        network: NetworkSide,
        frequency: usize,
        port: usize,
    },

    #[error(
        "simultaneous direct power-wave cascade is exactly singular at frequency {frequency}, pivot {pivot}"
    )]
    Singular { frequency: usize, pivot: usize },

    #[error(
        "non-finite value while evaluating simultaneous direct power-wave cascade at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteComputation {
        frequency: usize,
        row: usize,
        column: usize,
    },
}

/// Result of a simultaneous direct cascade.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CascadedNetwork {
    /// A's frequency axis, copied bit-for-bit.
    pub(crate) frequency_hz: Vec<f64>,
    /// Frequency-major output S parameters in `[A.left..., B.right...]` order.
    pub(crate) s: Array3<Complex64>,
    /// Surviving A-left references followed by B-right references.
    pub(crate) z0: Array2<Complex64>,
}

#[derive(Debug, Clone, Copy)]
struct InputShape {
    nfreq: usize,
    nports: usize,
}

/// Cascade two equal-sized ordered 2N-port networks in one joint solve.
pub(crate) fn cascade_direct(
    frequency_a: &[f64],
    s_a: &Array3<Complex64>,
    z0_a: &Array2<Complex64>,
    frequency_b: &[f64],
    s_b: &Array3<Complex64>,
    z0_b: &Array2<Complex64>,
) -> Result<CascadedNetwork, CascadeError> {
    // Validate all shape/axis/data invariants before any group-coordinate
    // indexing.  This boundary is reachable by serde-created malformed
    // Network values.
    let shape_a = validate_input(NetworkSide::A, frequency_a, s_a, z0_a)?;
    let shape_b = validate_input(NetworkSide::B, frequency_b, s_b, z0_b)?;

    if frequency_a.len() != frequency_b.len() {
        return Err(CascadeError::FrequencyLengthMismatch {
            a: frequency_a.len(),
            b: frequency_b.len(),
        });
    }
    for (index, (&a, &b)) in frequency_a.iter().zip(frequency_b).enumerate() {
        if a != b {
            return Err(CascadeError::FrequencyMismatch { index, a, b });
        }
    }
    if shape_a.nports % 2 != 0 {
        return Err(CascadeError::InvalidPortCount {
            network: NetworkSide::A,
            nports: shape_a.nports,
        });
    }
    if shape_b.nports % 2 != 0 {
        return Err(CascadeError::InvalidPortCount {
            network: NetworkSide::B,
            nports: shape_b.nports,
        });
    }
    if shape_a.nports != shape_b.nports {
        return Err(CascadeError::PortCountMismatch {
            a: shape_a.nports,
            b: shape_b.nports,
        });
    }

    let group_size = shape_a.nports / 2;
    let internal_size = group_size * 2;
    let external_size = internal_size;
    let mut output_s = Array3::from_elem((shape_a.nfreq, external_size, external_size), ZERO);
    let mut output_z0 = Array2::from_elem((shape_a.nfreq, external_size), ZERO);

    for frequency in 0..shape_a.nfreq {
        // Output references are exact source values, not arithmetic results.
        for port in 0..group_size {
            output_z0[[frequency, port]] = z0_a[[frequency, port]];
            output_z0[[frequency, group_size + port]] = z0_b[[frequency, group_size + port]];
        }

        let mut c = vec![ZERO; internal_size * internal_size];
        let mut d = vec![ZERO; internal_size * internal_size];
        build_boundary_matrices(frequency, group_size, z0_a, z0_b, &mut c, &mut d)?;

        // Construct S_ee, S_ei, S_ie, and S_ii for the external order
        // [A.left, B.right] and internal order [A.right, B.left].
        let mut s_ee = vec![ZERO; external_size * external_size];
        let mut s_ei = vec![ZERO; external_size * internal_size];
        let mut s_ie = vec![ZERO; internal_size * external_size];
        let mut s_ii = vec![ZERO; internal_size * internal_size];
        for row in 0..external_size {
            for column in 0..external_size {
                s_ee[row * external_size + column] =
                    source_s_ee(s_a, s_b, frequency, group_size, row, column);
            }
            for column in 0..internal_size {
                s_ei[row * internal_size + column] =
                    source_s_ei(s_a, s_b, frequency, group_size, row, column);
            }
        }
        for row in 0..internal_size {
            for column in 0..external_size {
                s_ie[row * external_size + column] =
                    source_s_ie(s_a, s_b, frequency, group_size, row, column);
            }
            for column in 0..internal_size {
                s_ii[row * internal_size + column] =
                    source_s_ii(s_a, s_b, frequency, group_size, row, column);
            }
        }

        // A = C + D*S_ii and B = -D*S_ie.  Both are square 2N-by-2N
        // matrices because the external and internal group sizes agree.
        let mut system = vec![ZERO; internal_size * internal_size];
        let mut rhs = vec![ZERO; internal_size * external_size];
        for row in 0..internal_size {
            for column in 0..internal_size {
                let mut value = c[row * internal_size + column];
                for index in 0..internal_size {
                    let product = checked_mul(
                        d[row * internal_size + index],
                        s_ii[index * internal_size + column],
                        frequency,
                        row,
                        column,
                    )?;
                    value = checked_add(value, product, frequency, row, column)?;
                }
                system[row * internal_size + column] = value;
            }
            for column in 0..external_size {
                let mut value = ZERO;
                for index in 0..internal_size {
                    let product = checked_mul(
                        d[row * internal_size + index],
                        s_ie[index * external_size + column],
                        frequency,
                        row,
                        column,
                    )?;
                    value = checked_add(value, product, frequency, row, column)?;
                }
                rhs[row * external_size + column] = checked_neg(value, frequency, row, column)?;
            }
        }

        // The shared solver deliberately accepts n RHS columns.  Here the
        // number of external columns equals the internal dimension (2N), so
        // the complete -D*S_ie matrix is solved in one call.
        linalg::solve_multiple_rhs(&mut system, &mut rhs, internal_size).map_err(|error| {
            match error {
                linalg::SolveError::InvalidStorage { .. } => {
                    unreachable!("simultaneous cascade solver storage is square by construction")
                }
                linalg::SolveError::Singular { pivot } => {
                    CascadeError::Singular { frequency, pivot }
                }
                linalg::SolveError::NonFinite { row, column } => {
                    CascadeError::NonFiniteComputation {
                        frequency,
                        row,
                        column,
                    }
                }
            }
        })?;

        for row in 0..external_size {
            for column in 0..external_size {
                let mut value = s_ee[row * external_size + column];
                for index in 0..internal_size {
                    let product = checked_mul(
                        s_ei[row * internal_size + index],
                        rhs[index * external_size + column],
                        frequency,
                        row,
                        column,
                    )?;
                    value = checked_add(value, product, frequency, row, column)?;
                }
                output_s[[frequency, row, column]] = value;
            }
        }
    }

    // Keep a result-wide guard close to the public boundary in case a future
    // edit adds an unguarded assignment.
    for (index, &value) in output_s.indexed_iter() {
        if !linalg::is_finite(value) {
            return Err(CascadeError::NonFiniteComputation {
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in output_z0.indexed_iter() {
        if !linalg::is_finite(value) {
            return Err(CascadeError::NonFiniteComputation {
                frequency: index.0,
                row: index.1,
                column: index.1,
            });
        }
    }

    Ok(CascadedNetwork {
        frequency_hz: frequency_a.to_vec(),
        s: output_s,
        z0: output_z0,
    })
}

fn validate_input(
    network: NetworkSide,
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<InputShape, CascadeError> {
    let shape = s.dim();
    if shape.1 == 0 || shape.1 != shape.2 {
        return Err(CascadeError::InvalidSShape { network, shape });
    }
    if frequency.is_empty() || shape.0 == 0 {
        return Err(CascadeError::EmptyFrequency { network });
    }
    if frequency.len() != shape.0 {
        return Err(CascadeError::FrequencyShape {
            network,
            expected: shape.0,
            actual: frequency.len(),
        });
    }
    let z0_shape = z0.dim();
    if z0_shape != (shape.0, shape.1) {
        return Err(CascadeError::InvalidZ0Shape {
            network,
            shape: z0_shape,
        });
    }
    for (index, &value) in frequency.iter().enumerate() {
        if !value.is_finite() {
            return Err(CascadeError::NonFiniteFrequency {
                network,
                index,
                value,
            });
        }
    }
    for (index, &value) in s.indexed_iter() {
        if !linalg::is_finite(value) {
            return Err(CascadeError::NonFiniteS {
                network,
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in z0.indexed_iter() {
        if !linalg::is_finite(value) {
            return Err(CascadeError::NonFiniteZ0 {
                network,
                frequency: index.0,
                port: index.1,
            });
        }
        if value.re == 0.0 {
            return Err(CascadeError::ZeroRealReferenceImpedance {
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

fn build_boundary_matrices(
    frequency: usize,
    group_size: usize,
    z0_a: &Array2<Complex64>,
    z0_b: &Array2<Complex64>,
    c: &mut [Complex64],
    d: &mut [Complex64],
) -> Result<(), CascadeError> {
    let dimension = group_size * 2;
    for index in 0..group_size {
        let z_a = z0_a[[frequency, group_size + index]];
        let z_b = z0_b[[frequency, index]];
        let q_a = q_for_reference(z_a, frequency, index, index)?;
        let q_b = q_for_reference(z_b, frequency, index, index + group_size)?;

        // Current equations occupy rows [0, N), voltage equations rows
        // [N, 2N). Columns are [A.right, B.left].
        c[index * dimension + index] = q_a;
        c[index * dimension + group_size + index] = q_b;
        c[(group_size + index) * dimension + index] =
            checked_mul(q_a, z_a.conj(), frequency, group_size + index, index)?;
        c[(group_size + index) * dimension + group_size + index] = checked_neg(
            checked_mul(
                q_b,
                z_b.conj(),
                frequency,
                group_size + index,
                index + group_size,
            )?,
            frequency,
            group_size + index,
            index + group_size,
        )?;

        d[index * dimension + index] = checked_neg(q_a, frequency, index, index)?;
        d[index * dimension + group_size + index] =
            checked_neg(q_b, frequency, index, index + group_size)?;
        d[(group_size + index) * dimension + index] =
            checked_mul(q_a, z_a, frequency, group_size + index, index)?;
        d[(group_size + index) * dimension + group_size + index] = checked_neg(
            checked_mul(q_b, z_b, frequency, group_size + index, index + group_size)?,
            frequency,
            group_size + index,
            index + group_size,
        )?;
    }
    Ok(())
}

fn source_s_ee(
    s_a: &Array3<Complex64>,
    s_b: &Array3<Complex64>,
    frequency: usize,
    group_size: usize,
    row: usize,
    column: usize,
) -> Complex64 {
    match (row < group_size, column < group_size) {
        (true, true) => s_a[[frequency, row, column]],
        (false, false) => s_b[[frequency, row, column]],
        _ => ZERO,
    }
}

fn source_s_ei(
    s_a: &Array3<Complex64>,
    s_b: &Array3<Complex64>,
    frequency: usize,
    group_size: usize,
    row: usize,
    column: usize,
) -> Complex64 {
    match (row < group_size, column < group_size) {
        (true, true) => s_a[[frequency, row, group_size + column]],
        (false, false) => {
            s_b[[
                frequency,
                group_size + row - group_size,
                column - group_size,
            ]]
        }
        _ => ZERO,
    }
}

fn source_s_ie(
    s_a: &Array3<Complex64>,
    s_b: &Array3<Complex64>,
    frequency: usize,
    group_size: usize,
    row: usize,
    column: usize,
) -> Complex64 {
    match (row < group_size, column < group_size) {
        (true, true) => s_a[[frequency, group_size + row, column]],
        (false, false) => {
            s_b[[
                frequency,
                row - group_size,
                group_size + column - group_size,
            ]]
        }
        _ => ZERO,
    }
}

fn source_s_ii(
    s_a: &Array3<Complex64>,
    s_b: &Array3<Complex64>,
    frequency: usize,
    group_size: usize,
    row: usize,
    column: usize,
) -> Complex64 {
    match (row < group_size, column < group_size) {
        (true, true) => s_a[[frequency, group_size + row, group_size + column]],
        (false, false) => s_b[[frequency, row - group_size, column - group_size]],
        _ => ZERO,
    }
}

fn q_for_reference(
    reference: Complex64,
    frequency: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, CascadeError> {
    let q = Complex64::new(reference.re.abs().sqrt() / reference.re, 0.0);
    checked_value(q, frequency, row, column)
}

fn checked_add(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, CascadeError> {
    checked_value(left + right, frequency, row, column)
}

fn checked_mul(
    left: Complex64,
    right: Complex64,
    frequency: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, CascadeError> {
    checked_value(left * right, frequency, row, column)
}

fn checked_neg(
    value: Complex64,
    frequency: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, CascadeError> {
    checked_value(-value, frequency, row, column)
}

fn checked_value(
    value: Complex64,
    frequency: usize,
    row: usize,
    column: usize,
) -> Result<Complex64, CascadeError> {
    if linalg::is_finite(value) {
        Ok(value)
    } else {
        Err(CascadeError::NonFiniteComputation {
            frequency,
            row,
            column,
        })
    }
}
