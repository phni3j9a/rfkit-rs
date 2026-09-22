//! Power-wave inverse cascade for even-port networks.
//!
//! This kernel implements the wave-reversal relation for an ordered 2N-port
//! network without forming a dense inverse.  For each frequency sample it
//! solves `S X = I` with the core exact-pivot solver and then applies the
//! group exchange `P X P`, where `P` swaps the first and second N-port
//! groups.  The source references are conjugated and exchanged because the
//! reversed boundary has `V' = P V`, `I' = -P I` under the Kurokawa
//! power-wave definition.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::linalg;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

use crate::InverseCascadeStage;

/// Failure modes for the even-port power-wave inverse-cascade kernel.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum InverseCascadeError {
    #[error("inverse-cascade frequency axis must not be empty")]
    EmptyFrequency,

    #[error(
        "inverse-cascade frequency length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    FrequencyLengthMismatch { expected: usize, actual: usize },

    #[error(
        "inverse-cascade requires a square positive even-port S-parameter shape, got {shape:?}"
    )]
    InvalidSShape { shape: (usize, usize, usize) },

    #[error("inverse-cascade reference-impedance shape must be (nfreq, nport), got {shape:?}")]
    InvalidZ0Shape { shape: (usize, usize) },

    #[error("inverse-cascade requires an even positive port count, got {nports}")]
    InvalidPortCount { nports: usize },

    #[error("inverse-cascade frequency is non-finite at index {index}: {value:?}")]
    NonFiniteFrequency { index: usize, value: f64 },

    #[error(
        "inverse-cascade {stage} input S-parameter is non-finite at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteS {
        stage: InverseCascadeStage,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "inverse-cascade reference impedance is non-finite at frequency {frequency}, port {port}"
    )]
    NonFiniteZ0 { frequency: usize, port: usize },

    #[error(
        "inverse-cascade reference impedance has a zero real part at frequency {frequency}, port {port}"
    )]
    ZeroRealReferenceImpedance { frequency: usize, port: usize },

    #[error(
        "inverse-cascade {stage} system is exactly singular at frequency {frequency}, pivot {pivot}"
    )]
    Singular {
        stage: InverseCascadeStage,
        frequency: usize,
        pivot: usize,
    },

    #[error(
        "inverse-cascade {stage} computation became non-finite at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteComputation {
        stage: InverseCascadeStage,
        frequency: usize,
        row: usize,
        column: usize,
    },
}

/// Evaluate the power-wave inverse cascade and the transformed references.
pub(crate) fn inverse_cascade_power(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<(Array3<Complex64>, Array2<Complex64>), InverseCascadeError> {
    let nfreq = frequency.len();
    if nfreq == 0 {
        return Err(InverseCascadeError::EmptyFrequency);
    }

    let s_shape = s.dim();
    if s_shape.0 != nfreq {
        return Err(InverseCascadeError::FrequencyLengthMismatch {
            expected: nfreq,
            actual: s_shape.0,
        });
    }
    if s_shape.1 == 0 || s_shape.1 != s_shape.2 {
        return Err(InverseCascadeError::InvalidSShape { shape: s_shape });
    }
    let nport = s_shape.1;
    if nport % 2 != 0 {
        return Err(InverseCascadeError::InvalidPortCount { nports: nport });
    }

    let z0_shape = z0.dim();
    if z0_shape != (nfreq, nport) {
        return Err(InverseCascadeError::InvalidZ0Shape { shape: z0_shape });
    }

    for (index, &value) in frequency.iter().enumerate() {
        if !value.is_finite() {
            return Err(InverseCascadeError::NonFiniteFrequency { index, value });
        }
    }
    for (index, &value) in s.indexed_iter() {
        if !linalg::is_finite(value) {
            return Err(InverseCascadeError::NonFiniteS {
                stage: InverseCascadeStage::FullS,
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in z0.indexed_iter() {
        if !linalg::is_finite(value) {
            return Err(InverseCascadeError::NonFiniteZ0 {
                frequency: index.0,
                port: index.1,
            });
        }
        if value.re == 0.0 {
            return Err(InverseCascadeError::ZeroRealReferenceImpedance {
                frequency: index.0,
                port: index.1,
            });
        }
    }

    let group_size = nport / 2;
    let mut inverse = Array3::from_elem((nfreq, nport, nport), ZERO);
    let mut output_z0 = Array2::from_elem((nfreq, nport), ZERO);

    for frequency_index in 0..nfreq {
        let mut full_matrix = vec![ZERO; nport * nport];
        let mut full_rhs = identity(nport);
        for row in 0..nport {
            for column in 0..nport {
                full_matrix[row * nport + column] = s[[frequency_index, row, column]];
            }
        }
        solve(
            &mut full_matrix,
            &mut full_rhs,
            nport,
            InverseCascadeStage::FullS,
            frequency_index,
        )?;

        // Both directional transmission blocks are part of the explicit
        // inverse-cascade domain.  Solve them independently so an invertible
        // full S with an isolated direction is rejected with useful stage
        // context rather than being accepted as a merely algebraic inverse.
        let mut forward = vec![ZERO; group_size * group_size];
        let mut reverse = vec![ZERO; group_size * group_size];
        let mut forward_rhs = identity(group_size);
        let mut reverse_rhs = identity(group_size);
        for row in 0..group_size {
            for column in 0..group_size {
                forward[row * group_size + column] = s[[frequency_index, group_size + row, column]];
                reverse[row * group_size + column] = s[[frequency_index, row, group_size + column]];
            }
        }
        solve(
            &mut forward,
            &mut forward_rhs,
            group_size,
            InverseCascadeStage::ForwardTransmission,
            frequency_index,
        )?;
        solve(
            &mut reverse,
            &mut reverse_rhs,
            group_size,
            InverseCascadeStage::ReverseTransmission,
            frequency_index,
        )?;

        for row in 0..nport {
            let swapped_row = swap_group(row, group_size);
            for column in 0..nport {
                let swapped_column = swap_group(column, group_size);
                let value = full_rhs[swapped_row * nport + swapped_column];
                if !linalg::is_finite(value) {
                    return Err(InverseCascadeError::NonFiniteComputation {
                        stage: InverseCascadeStage::FullS,
                        frequency: frequency_index,
                        row,
                        column,
                    });
                }
                inverse[[frequency_index, row, column]] = value;
            }
        }
        for port in 0..nport {
            let old_port = swap_group(port, group_size);
            let value = z0[[frequency_index, old_port]].conj();
            if !linalg::is_finite(value) {
                return Err(InverseCascadeError::NonFiniteComputation {
                    stage: InverseCascadeStage::FullS,
                    frequency: frequency_index,
                    row: port,
                    column: port,
                });
            }
            output_z0[[frequency_index, port]] = value;
        }
    }

    Ok((inverse, output_z0))
}

fn identity(n: usize) -> Vec<Complex64> {
    let mut result = vec![ZERO; n * n];
    for index in 0..n {
        result[index * n + index] = Complex64::new(1.0, 0.0);
    }
    result
}

fn swap_group(index: usize, group_size: usize) -> usize {
    if index < group_size {
        index + group_size
    } else {
        index - group_size
    }
}

fn solve(
    matrix: &mut [Complex64],
    rhs: &mut [Complex64],
    dimension: usize,
    stage: InverseCascadeStage,
    frequency: usize,
) -> Result<(), InverseCascadeError> {
    linalg::solve_multiple_rhs(matrix, rhs, dimension).map_err(|error| match error {
        linalg::SolveError::InvalidStorage { n, a_len, b_len } => {
            InverseCascadeError::NonFiniteComputation {
                stage,
                frequency,
                row: n,
                column: a_len.max(b_len),
            }
        }
        linalg::SolveError::Singular { pivot } => InverseCascadeError::Singular {
            stage,
            frequency,
            pivot,
        },
        linalg::SolveError::NonFinite { row, column } => {
            InverseCascadeError::NonFiniteComputation {
                stage,
                frequency,
                row,
                column,
            }
        }
    })
}
