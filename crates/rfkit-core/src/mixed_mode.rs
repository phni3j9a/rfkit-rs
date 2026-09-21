//! Equal-reference-pair mixed-mode conversion for Kurokawa power waves.
//!
//! The public methods in [`crate::Network`] deliberately keep this kernel
//! small and explicit.  For an adjacent pair `(u, v)`, with `u` positive,
//! the real orthogonal coordinate rows are
//!
//! ```text
//! d = (u - v) / sqrt(2),   c = (u + v) / sqrt(2).
//! ```
//!
//! Equal single-ended references `z` map to natural modal references `2z`
//! and `z/2`.  This is a wave-coordinate change, not an impedance
//! renormalization or a conversion through Z/Y parameters.  The inverse uses
//! the transpose of the same real transform.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::linalg;
use crate::{MixedModeDirection, MixedModeMode, MixedModeScaling};

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const SQRT_TWO: f64 = std::f64::consts::SQRT_2;
const INV_SQRT_TWO: f64 = 1.0 / SQRT_TWO;

/// Errors produced by the equal-pair mixed-mode kernel.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum MixedModeError {
    #[error("{direction} mixed-mode conversion frequency axis must not be empty")]
    EmptyFrequency { direction: MixedModeDirection },

    #[error(
        "{direction} mixed-mode conversion frequency length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    FrequencyLengthMismatch {
        direction: MixedModeDirection,
        expected: usize,
        actual: usize,
    },

    #[error("{direction} mixed-mode conversion received an invalid S-parameter shape {shape:?}")]
    InvalidSShape {
        direction: MixedModeDirection,
        shape: (usize, usize, usize),
    },

    #[error(
        "{direction} mixed-mode conversion received an invalid reference-impedance shape {shape:?}"
    )]
    InvalidZ0Shape {
        direction: MixedModeDirection,
        shape: (usize, usize),
    },

    #[error(
        "{direction} mixed-mode conversion pair_count must be at least one and no greater than floor(nports/2): pair_count={pair_count}, nports={nports}"
    )]
    PairCountOutOfRange {
        direction: MixedModeDirection,
        pair_count: usize,
        nports: usize,
    },

    #[error(
        "{direction} mixed-mode conversion received a non-finite S-parameter at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteS {
        direction: MixedModeDirection,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "{direction} mixed-mode conversion received a non-finite reference impedance at frequency {frequency}, port {port}"
    )]
    NonFiniteZ0 {
        direction: MixedModeDirection,
        frequency: usize,
        port: usize,
    },

    #[error(
        "{direction} mixed-mode conversion has a zero-real reference impedance at frequency {frequency}, port {port}"
    )]
    ZeroRealReferenceImpedance {
        direction: MixedModeDirection,
        frequency: usize,
        port: usize,
    },

    #[error(
        "single-ended references in pair {pair} are not exactly equal at frequency {frequency}: positive={positive:?}, negative={negative:?}"
    )]
    UnequalPairReferences {
        direction: MixedModeDirection,
        frequency: usize,
        pair: usize,
        positive: Complex64,
        negative: Complex64,
    },

    #[error(
        "{direction} mixed-mode {mode:?} reference scaling at frequency {frequency}, pair {pair} with {scaling:?} of {value:?} loses finite information or is not finite"
    )]
    ReferenceScalingLoss {
        direction: MixedModeDirection,
        frequency: usize,
        pair: usize,
        mode: MixedModeMode,
        scaling: MixedModeScaling,
        value: Complex64,
    },

    #[error(
        "{direction} mixed-mode references at frequency {frequency}, pair {pair} are not exactly natural: differential={differential:?}, common={common:?}, zd/2={from_differential:?}, 2*zc={from_common:?}"
    )]
    InverseReferenceMismatch {
        direction: MixedModeDirection,
        frequency: usize,
        pair: usize,
        differential: Complex64,
        common: Complex64,
        from_differential: Complex64,
        from_common: Complex64,
    },

    #[error(
        "{direction} mixed-mode conversion produced a non-finite computation at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteComputation {
        direction: MixedModeDirection,
        frequency: usize,
        row: usize,
        column: usize,
    },
}

/// Convert S and z0 to the modal coordinate order `[d..., c..., unpaired]`.
pub(crate) fn to_mixed_mode_equal_pair_power(
    frequency_length: usize,
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    pair_count: usize,
) -> Result<(Array3<Complex64>, Array2<Complex64>), MixedModeError> {
    let direction = MixedModeDirection::ToMixedMode;
    validate_common(direction, frequency_length, s, z0, pair_count)?;
    let (nfreq, nport, _) = s.dim();

    validate_finite_inputs(direction, s, z0)?;
    let Some(paired) = paired_port_count(pair_count) else {
        return Err(MixedModeError::PairCountOutOfRange {
            direction,
            pair_count,
            nports: nport,
        });
    };

    let mut modal_z0 = z0.clone();
    for frequency in 0..nfreq {
        for pair in 0..pair_count {
            let positive = z0[[frequency, 2 * pair]];
            let negative = z0[[frequency, 2 * pair + 1]];
            if positive != negative {
                return Err(MixedModeError::UnequalPairReferences {
                    direction,
                    frequency,
                    pair,
                    positive,
                    negative,
                });
            }

            modal_z0[[frequency, pair]] = scale_reference(
                positive,
                2.0,
                direction,
                frequency,
                pair,
                MixedModeMode::Differential,
                MixedModeScaling::Double,
            )?;
            modal_z0[[frequency, pair_count + pair]] = scale_reference(
                positive,
                0.5,
                direction,
                frequency,
                pair,
                MixedModeMode::Common,
                MixedModeScaling::Half,
            )?;
        }
        // `paired <= nport` was established without an overflowing
        // multiplication.  Preserve all unpaired references exactly.
        for port in paired..nport {
            modal_z0[[frequency, port]] = z0[[frequency, port]];
        }
    }

    let modal_s = transform(s, pair_count, direction)?;
    Ok((modal_s, modal_z0))
}

/// Convert modal S and z0 in `[d..., c..., unpaired]` order back to adjacent
/// single-ended pairs.
pub(crate) fn to_single_ended_equal_pair_power(
    frequency_length: usize,
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    pair_count: usize,
) -> Result<(Array3<Complex64>, Array2<Complex64>), MixedModeError> {
    let direction = MixedModeDirection::ToSingleEnded;
    validate_common(direction, frequency_length, s, z0, pair_count)?;
    let (nfreq, nport, _) = s.dim();

    validate_finite_inputs(direction, s, z0)?;
    let Some(paired) = paired_port_count(pair_count) else {
        return Err(MixedModeError::PairCountOutOfRange {
            direction,
            pair_count,
            nports: nport,
        });
    };
    let mut single_ended_z0 = z0.clone();

    for frequency in 0..nfreq {
        for pair in 0..pair_count {
            let differential = z0[[frequency, pair]];
            let common = z0[[frequency, pair_count + pair]];
            let from_differential = scale_reference(
                differential,
                0.5,
                direction,
                frequency,
                pair,
                MixedModeMode::Differential,
                MixedModeScaling::Half,
            )?;
            let from_common = scale_reference(
                common,
                2.0,
                direction,
                frequency,
                pair,
                MixedModeMode::Common,
                MixedModeScaling::Double,
            )?;
            if from_differential != from_common {
                return Err(MixedModeError::InverseReferenceMismatch {
                    direction,
                    frequency,
                    pair,
                    differential,
                    common,
                    from_differential,
                    from_common,
                });
            }
            single_ended_z0[[frequency, 2 * pair]] = from_differential;
            single_ended_z0[[frequency, 2 * pair + 1]] = from_differential;
        }
        for port in paired..nport {
            single_ended_z0[[frequency, port]] = z0[[frequency, port]];
        }
    }

    let single_ended_s = transform(s, pair_count, direction)?;
    Ok((single_ended_s, single_ended_z0))
}

fn validate_common(
    direction: MixedModeDirection,
    frequency_length: usize,
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    pair_count: usize,
) -> Result<(), MixedModeError> {
    if frequency_length == 0 {
        return Err(MixedModeError::EmptyFrequency { direction });
    }
    let (s_frequency_length, nport_rows, nport_columns) = s.dim();
    if s_frequency_length != frequency_length {
        return Err(MixedModeError::FrequencyLengthMismatch {
            direction,
            expected: frequency_length,
            actual: s_frequency_length,
        });
    }
    if nport_rows == 0 || nport_rows != nport_columns {
        return Err(MixedModeError::InvalidSShape {
            direction,
            shape: (s_frequency_length, nport_rows, nport_columns),
        });
    }
    if z0.dim() != (s_frequency_length, nport_rows) {
        return Err(MixedModeError::InvalidZ0Shape {
            direction,
            shape: z0.dim(),
        });
    }
    if pair_count == 0 || pair_count > nport_rows / 2 {
        return Err(MixedModeError::PairCountOutOfRange {
            direction,
            pair_count,
            nports: nport_rows,
        });
    }
    // This is redundant for ordinary arrays but keeps the indexing proof
    // explicit if an unusual usize/platform boundary is ever exercised.
    if pair_count.checked_mul(2).is_none() {
        return Err(MixedModeError::PairCountOutOfRange {
            direction,
            pair_count,
            nports: nport_rows,
        });
    }
    Ok(())
}

fn validate_finite_inputs(
    direction: MixedModeDirection,
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<(), MixedModeError> {
    let (nfreq, nport, _) = s.dim();
    for frequency in 0..nfreq {
        for row in 0..nport {
            for column in 0..nport {
                if !linalg::is_finite(s[[frequency, row, column]]) {
                    return Err(MixedModeError::NonFiniteS {
                        direction,
                        frequency,
                        row,
                        column,
                    });
                }
            }
        }
        for port in 0..nport {
            let reference = z0[[frequency, port]];
            if !linalg::is_finite(reference) {
                return Err(MixedModeError::NonFiniteZ0 {
                    direction,
                    frequency,
                    port,
                });
            }
            if reference.re == 0.0 {
                return Err(MixedModeError::ZeroRealReferenceImpedance {
                    direction,
                    frequency,
                    port,
                });
            }
        }
    }
    Ok(())
}

fn paired_port_count(pair_count: usize) -> Option<usize> {
    pair_count.checked_mul(2)
}

fn scale_reference(
    value: Complex64,
    factor: f64,
    direction: MixedModeDirection,
    frequency: usize,
    pair: usize,
    mode: MixedModeMode,
    scaling: MixedModeScaling,
) -> Result<Complex64, MixedModeError> {
    let real = scale_component(value.re, factor);
    let imag = scale_component(value.im, factor);
    let Ok(real) = real else {
        return Err(MixedModeError::ReferenceScalingLoss {
            direction,
            frequency,
            pair,
            mode,
            scaling,
            value,
        });
    };
    let Ok(imag) = imag else {
        return Err(MixedModeError::ReferenceScalingLoss {
            direction,
            frequency,
            pair,
            mode,
            scaling,
            value,
        });
    };
    let scaled = Complex64::new(real, imag);
    if !linalg::is_finite(scaled) || scaled.re == 0.0 {
        return Err(MixedModeError::ReferenceScalingLoss {
            direction,
            frequency,
            pair,
            mode,
            scaling,
            value,
        });
    }
    Ok(scaled)
}

fn scale_component(value: f64, factor: f64) -> Result<f64, ()> {
    let scaled = value * factor;
    if !scaled.is_finite() || (value != 0.0 && scaled == 0.0) {
        return Err(());
    }
    let round_trip = if factor == 2.0 {
        scaled * 0.5
    } else {
        scaled * 2.0
    };
    if !round_trip.is_finite() || round_trip != value {
        return Err(());
    }
    Ok(scaled)
}

fn transform(
    input: &Array3<Complex64>,
    pair_count: usize,
    direction: MixedModeDirection,
) -> Result<Array3<Complex64>, MixedModeError> {
    let (nfreq, nport, _) = input.dim();
    // Every row of U (and therefore every row of Uᵀ) has at most two nonzero
    // entries.  Materialize those ordered entries once so each output scalar
    // below evaluates only their Cartesian product (at most four terms),
    // rather than scanning all nport² coordinate pairs.
    let terms = transform_terms(pair_count, direction, nport);
    let mut output = Array3::from_elem((nfreq, nport, nport), ZERO);

    for frequency in 0..nfreq {
        for output_row in 0..nport {
            for output_column in 0..nport {
                let mut value = ZERO;
                for &(left, left_coefficient) in &terms[output_row] {
                    for &(right, right_coefficient) in &terms[output_column] {
                        let mut term = input[[frequency, left, right]] * left_coefficient;
                        if !linalg::is_finite(term) {
                            return Err(MixedModeError::NonFiniteComputation {
                                direction,
                                frequency,
                                row: output_row,
                                column: output_column,
                            });
                        }
                        term *= right_coefficient;
                        if !linalg::is_finite(term) {
                            return Err(MixedModeError::NonFiniteComputation {
                                direction,
                                frequency,
                                row: output_row,
                                column: output_column,
                            });
                        }
                        value += term;
                        if !linalg::is_finite(value) {
                            return Err(MixedModeError::NonFiniteComputation {
                                direction,
                                frequency,
                                row: output_row,
                                column: output_column,
                            });
                        }
                    }
                }
                output[[frequency, output_row, output_column]] = value;
            }
        }
    }

    Ok(output)
}

fn transform_terms(
    pair_count: usize,
    direction: MixedModeDirection,
    nport: usize,
) -> Vec<Vec<(usize, f64)>> {
    (0..nport)
        .map(|output_index| match direction {
            MixedModeDirection::ToMixedMode => {
                if output_index < pair_count {
                    let pair_start = 2 * output_index;
                    vec![(pair_start, INV_SQRT_TWO), (pair_start + 1, -INV_SQRT_TWO)]
                } else if output_index < 2 * pair_count {
                    let pair_start = 2 * (output_index - pair_count);
                    vec![(pair_start, INV_SQRT_TWO), (pair_start + 1, INV_SQRT_TWO)]
                } else {
                    vec![(output_index, 1.0)]
                }
            }
            MixedModeDirection::ToSingleEnded => {
                // For Uᵀ, output_index is a single-ended coordinate and the
                // tuple indices are modal coordinates. U is real, so no
                // conjugation is involved.
                if output_index < 2 * pair_count {
                    let pair = output_index / 2;
                    let polarity = if output_index % 2 == 0 { 1.0 } else { -1.0 };
                    vec![
                        (pair, polarity * INV_SQRT_TWO),
                        (pair_count + pair, INV_SQRT_TWO),
                    ]
                } else {
                    vec![(output_index, 1.0)]
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_terms_are_sparse_and_ordered_for_both_directions() {
        for direction in [
            MixedModeDirection::ToMixedMode,
            MixedModeDirection::ToSingleEnded,
        ] {
            let terms = transform_terms(2, direction, 5);
            assert!(terms.iter().all(|coordinate| coordinate.len() <= 2));
            assert!(terms.iter().all(|coordinate| {
                coordinate
                    .windows(2)
                    .all(|window| window[0].0 < window[1].0)
            }));
        }
    }
}
