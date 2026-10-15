//! Two-port transducer power gain from explicit source and load impedances.
//!
//! The public operation in this module is deliberately a borrowing
//! diagnostic.  It re-expresses the stored two-port S matrix at the caller's
//! per-sample source/load references using the checked direct Kurokawa
//! wave-change kernel, then reports the squared magnitude of the forward
//! coordinate `S_new[1, 0]`.  No source/load solve is used as the evaluation
//! path; the physical V/I definition is retained in the public documentation
//! and in independent tests.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::linalg;
use crate::power_waves::{self, DirectRenormalizationError};

/// Failure modes for the sampled two-port transducer-gain diagnostic.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum TwoPortTransducerGainError {
    #[error("frequency axis must not be empty")]
    EmptyFrequency,

    #[error(
        "frequency axis length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    FrequencyLengthMismatch { expected: usize, actual: usize },

    #[error("S-parameter shape must be exactly (nfreq, 2, 2), got {shape:?}")]
    InvalidSShape { shape: (usize, usize, usize) },

    #[error("reference-impedance shape must be exactly (nfreq, 2), got {shape:?}")]
    InvalidZ0Shape { shape: (usize, usize) },

    #[error(
        "source-impedance length does not match the frequency axis: expected {expected}, got {actual}"
    )]
    SourceImpedanceLengthMismatch { expected: usize, actual: usize },

    #[error(
        "load-impedance length does not match the frequency axis: expected {expected}, got {actual}"
    )]
    LoadImpedanceLengthMismatch { expected: usize, actual: usize },

    #[error("frequency is non-finite at index {index}: {value:?}")]
    NonFiniteFrequency { index: usize, value: f64 },

    #[error("S-parameter is non-finite at frequency {frequency}, row {row}, column {column}")]
    NonFiniteS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("stored reference impedance is non-finite at frequency {frequency}, port {port}")]
    NonFiniteZ0 { frequency: usize, port: usize },

    #[error(
        "stored reference impedance must have a strictly positive real part at frequency {frequency}, port {port}: {value:?}"
    )]
    NonPositiveRealZ0 {
        frequency: usize,
        port: usize,
        value: Complex64,
    },

    #[error("source impedance is non-finite at frequency {frequency}")]
    NonFiniteSourceImpedance { frequency: usize },

    #[error(
        "source impedance must have a strictly positive real part at frequency {frequency}: {value:?}"
    )]
    NonPositiveRealSourceImpedance { frequency: usize, value: Complex64 },

    #[error("load impedance is non-finite at frequency {frequency}")]
    NonFiniteLoadImpedance { frequency: usize },

    #[error(
        "load impedance must have a strictly positive real part at frequency {frequency}: {value:?}"
    )]
    NonPositiveRealLoadImpedance { frequency: usize, value: Complex64 },

    #[error(
        "direct wave-change system is exactly singular at frequency {frequency}, pivot {pivot}"
    )]
    SingularWaveChange { frequency: usize, pivot: usize },

    #[error("direct wave-change received an invalid S-parameter shape {shape:?}")]
    InvalidWaveChangeSShape { shape: (usize, usize, usize) },

    #[error(
        "direct wave-change received an invalid {reference:?} reference-impedance shape {shape:?}"
    )]
    InvalidWaveChangeZ0Shape {
        reference: power_waves::DirectRenormalizationReference,
        shape: (usize, usize),
    },

    #[error(
        "direct wave-change received a non-finite S-parameter at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteWaveChangeS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "direct wave-change received a non-finite {reference:?} reference impedance at frequency {frequency}, port {port}"
    )]
    NonFiniteWaveChangeZ0 {
        reference: power_waves::DirectRenormalizationReference,
        frequency: usize,
        port: usize,
    },

    #[error(
        "direct wave-change has a zero-real {reference:?} reference impedance at frequency {frequency}, port {port}"
    )]
    ZeroRealWaveChangeZ0 {
        reference: power_waves::DirectRenormalizationReference,
        frequency: usize,
        port: usize,
    },

    #[error(
        "direct wave-change produced a non-finite computation at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteWaveChangeComputation {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "transducer power gain arithmetic became non-finite or unrepresentable at frequency {frequency} while evaluating {stage}"
    )]
    Arithmetic {
        frequency: usize,
        stage: TwoPortTransducerGainArithmetic,
    },
}

/// Arithmetic stage for a transducer-power-gain output failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TwoPortTransducerGainArithmetic {
    /// The finite forward-wave amplitude `hypot(Re(S21), Im(S21))`.
    ForwardAmplitude,
    /// The final power ratio `|S21|²`.
    PowerRatio,
}

impl std::fmt::Display for TwoPortTransducerGainArithmetic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ForwardAmplitude => "forward-wave amplitude",
            Self::PowerRatio => "forward power ratio",
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct InputShape {
    nfreq: usize,
}

/// Evaluate the two-port transducer power ratio for every stored sample.
pub(crate) fn two_port_transducer_gain_power(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    source_impedance_ohm: &[Complex64],
    load_impedance_ohm: &[Complex64],
) -> Result<Vec<f64>, TwoPortTransducerGainError> {
    let shape = validate_input(frequency, s, z0, source_impedance_ohm, load_impedance_ohm)?;

    // The table is assembled only after all caller inputs have been checked.
    // `from_shape_fn` indexes the borrowed slices at their already validated
    // exact length and creates the explicit target `[Zs, Zl]` coordinates.
    let target_z0 = Array2::from_shape_fn((shape.nfreq, 2), |(sample, port)| {
        if port == 0 {
            source_impedance_ohm[sample]
        } else {
            load_impedance_ohm[sample]
        }
    });

    // This is intentionally the sole RF evaluation path.  It evaluates the
    // complete direct wave-change system for every sample before S21 is
    // inspected, so an exact singularity is reported even when the forward
    // transmission would be zero.
    let target_s = power_waves::renormalize_s_power_direct(s, z0, &target_z0)
        .map_err(map_wave_change_error)?;

    let mut output = Vec::with_capacity(shape.nfreq);
    for frequency_index in 0..shape.nfreq {
        let forward = target_s[[frequency_index, 1, 0]];
        output.push(power_ratio(forward, frequency_index)?);
    }
    Ok(output)
}

fn validate_input(
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    source_impedance_ohm: &[Complex64],
    load_impedance_ohm: &[Complex64],
) -> Result<InputShape, TwoPortTransducerGainError> {
    if frequency.is_empty() {
        return Err(TwoPortTransducerGainError::EmptyFrequency);
    }

    let s_shape = s.dim();
    if s_shape.0 != frequency.len() {
        return Err(TwoPortTransducerGainError::FrequencyLengthMismatch {
            expected: frequency.len(),
            actual: s_shape.0,
        });
    }
    if s_shape.1 != 2 || s_shape.2 != 2 {
        return Err(TwoPortTransducerGainError::InvalidSShape { shape: s_shape });
    }

    let z0_shape = z0.dim();
    if z0_shape != (frequency.len(), 2) {
        return Err(TwoPortTransducerGainError::InvalidZ0Shape { shape: z0_shape });
    }

    if source_impedance_ohm.len() != frequency.len() {
        return Err(TwoPortTransducerGainError::SourceImpedanceLengthMismatch {
            expected: frequency.len(),
            actual: source_impedance_ohm.len(),
        });
    }
    if load_impedance_ohm.len() != frequency.len() {
        return Err(TwoPortTransducerGainError::LoadImpedanceLengthMismatch {
            expected: frequency.len(),
            actual: load_impedance_ohm.len(),
        });
    }

    // Complete validation precedes every sampled S/z0/source/load indexing.
    // Frequency labels are pointwise data: finiteness is required, but no
    // ordering, uniqueness, sign, or interpolation policy is imposed.
    for (index, &value) in frequency.iter().enumerate() {
        if !value.is_finite() {
            return Err(TwoPortTransducerGainError::NonFiniteFrequency { index, value });
        }
    }
    for (index, &value) in s.indexed_iter() {
        if !linalg::is_finite(value) {
            return Err(TwoPortTransducerGainError::NonFiniteS {
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in z0.indexed_iter() {
        if !linalg::is_finite(value) {
            return Err(TwoPortTransducerGainError::NonFiniteZ0 {
                frequency: index.0,
                port: index.1,
            });
        }
        if value.re <= 0.0 {
            return Err(TwoPortTransducerGainError::NonPositiveRealZ0 {
                frequency: index.0,
                port: index.1,
                value,
            });
        }
    }
    for (frequency_index, &value) in source_impedance_ohm.iter().enumerate() {
        if !linalg::is_finite(value) {
            return Err(TwoPortTransducerGainError::NonFiniteSourceImpedance {
                frequency: frequency_index,
            });
        }
        if value.re <= 0.0 {
            return Err(TwoPortTransducerGainError::NonPositiveRealSourceImpedance {
                frequency: frequency_index,
                value,
            });
        }
    }
    for (frequency_index, &value) in load_impedance_ohm.iter().enumerate() {
        if !linalg::is_finite(value) {
            return Err(TwoPortTransducerGainError::NonFiniteLoadImpedance {
                frequency: frequency_index,
            });
        }
        if value.re <= 0.0 {
            return Err(TwoPortTransducerGainError::NonPositiveRealLoadImpedance {
                frequency: frequency_index,
                value,
            });
        }
    }

    Ok(InputShape {
        nfreq: frequency.len(),
    })
}

fn power_ratio(forward: Complex64, frequency: usize) -> Result<f64, TwoPortTransducerGainError> {
    if forward == Complex64::new(0.0, 0.0) {
        return Ok(0.0);
    }

    // `hypot` avoids the overflow/underflow of forming re²+im² before the
    // final square.  A nonzero amplitude whose squared result rounds to zero
    // is an unrepresentable nonzero gain, not a value to clamp to zero.
    let amplitude = forward.re.hypot(forward.im);
    if !amplitude.is_finite() || amplitude == 0.0 {
        return Err(TwoPortTransducerGainError::Arithmetic {
            frequency,
            stage: TwoPortTransducerGainArithmetic::ForwardAmplitude,
        });
    }
    let gain = amplitude * amplitude;
    if !gain.is_finite() || gain == 0.0 {
        return Err(TwoPortTransducerGainError::Arithmetic {
            frequency,
            stage: TwoPortTransducerGainArithmetic::PowerRatio,
        });
    }
    Ok(gain)
}

fn map_wave_change_error(error: DirectRenormalizationError) -> TwoPortTransducerGainError {
    match error {
        DirectRenormalizationError::InvalidSShape { shape } => {
            TwoPortTransducerGainError::InvalidWaveChangeSShape { shape }
        }
        DirectRenormalizationError::InvalidZ0Shape { reference, shape } => {
            TwoPortTransducerGainError::InvalidWaveChangeZ0Shape { reference, shape }
        }
        DirectRenormalizationError::NonFiniteS {
            frequency,
            row,
            column,
        } => TwoPortTransducerGainError::NonFiniteWaveChangeS {
            frequency,
            row,
            column,
        },
        DirectRenormalizationError::NonFiniteZ0 {
            reference,
            frequency,
            port,
        } => TwoPortTransducerGainError::NonFiniteWaveChangeZ0 {
            reference,
            frequency,
            port,
        },
        DirectRenormalizationError::ZeroRealReferenceImpedance {
            reference,
            frequency,
            port,
        } => TwoPortTransducerGainError::ZeroRealWaveChangeZ0 {
            reference,
            frequency,
            port,
        },
        DirectRenormalizationError::Singular { frequency, pivot } => {
            TwoPortTransducerGainError::SingularWaveChange { frequency, pivot }
        }
        DirectRenormalizationError::NonFiniteComputation {
            frequency,
            row,
            column,
        } => TwoPortTransducerGainError::NonFiniteWaveChangeComputation {
            frequency,
            row,
            column,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    fn c(real: f64, imag: f64) -> Complex64 {
        Complex64::new(real, imag)
    }

    fn network_data() -> (Vec<f64>, Array3<Complex64>, Array2<Complex64>) {
        (
            vec![-2.0, 1.0, 1.0],
            Array3::from_shape_fn((3, 2, 2), |(frequency, row, column)| {
                c(
                    0.03 * (frequency + 1) as f64 + 0.01 * row as f64,
                    -0.02 * (column + 1) as f64,
                )
            }),
            Array2::from_shape_fn((3, 2), |(frequency, port)| {
                c(
                    40.0 + 3.0 * frequency as f64 + port as f64,
                    1.0 - port as f64,
                )
            }),
        )
    }

    #[test]
    fn ideal_through_matches_physical_power_ratio() {
        let frequency = vec![1.0];
        let s = Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        )
        .unwrap();
        let z0 = Array2::from_elem((1, 2), c(50.0, 0.0));
        let source = [c(35.0, 8.0)];
        let load = [c(73.0, -11.0)];
        let actual = two_port_transducer_gain_power(&frequency, &s, &z0, &source, &load).unwrap();
        let expected = 4.0 * source[0].re * load[0].re / (source[0] + load[0]).norm_sqr();
        assert!((actual[0] - expected).abs() <= 1.0e-14);
    }

    #[test]
    fn exact_zero_forward_wave_is_zero_but_singular_system_still_errors() {
        let frequency = vec![1.0];
        let s = Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(-3.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        )
        .unwrap();
        let z0 = Array2::from_elem((1, 2), c(50.0, 0.0));
        let source = [c(25.0, 0.0)];
        let load = [c(50.0, 0.0)];
        assert_eq!(
            two_port_transducer_gain_power(&frequency, &s, &z0, &source, &load),
            Err(TwoPortTransducerGainError::SingularWaveChange {
                frequency: 0,
                pivot: 0,
            })
        );
    }

    #[test]
    fn validation_rejects_malformed_shapes_and_references_before_indexing() {
        let frequency = vec![1.0, 2.0];
        let s = Array3::<Complex64>::zeros((1, 2, 2));
        let z0 = Array2::<Complex64>::zeros((1, 2));
        assert_eq!(
            two_port_transducer_gain_power(&frequency, &s, &z0, &[], &[]),
            Err(TwoPortTransducerGainError::FrequencyLengthMismatch {
                expected: 2,
                actual: 1,
            })
        );

        let s = Array3::<Complex64>::zeros((2, 1, 2));
        let z0 = Array2::<Complex64>::zeros((2, 2));
        assert!(matches!(
            two_port_transducer_gain_power(
                &frequency,
                &s,
                &z0,
                &[c(50.0, 0.0); 2],
                &[c(50.0, 0.0); 2]
            ),
            Err(TwoPortTransducerGainError::InvalidSShape { .. })
        ));

        let (frequency, s, mut z0) = network_data();
        z0[[1, 1]] = c(0.0, 2.0);
        assert!(matches!(
            two_port_transducer_gain_power(
                &frequency,
                &s,
                &z0,
                &[c(50.0, 0.0); 3],
                &[c(50.0, 0.0); 3]
            ),
            Err(TwoPortTransducerGainError::NonPositiveRealZ0 {
                frequency: 1,
                port: 1,
                ..
            })
        ));
    }

    #[test]
    fn power_ratio_reports_overflow_and_nonzero_underflow() {
        let overflow_amplitude = power_ratio(c(f64::MAX, f64::MAX), 4).unwrap_err();
        assert_eq!(
            overflow_amplitude,
            TwoPortTransducerGainError::Arithmetic {
                frequency: 4,
                stage: TwoPortTransducerGainArithmetic::ForwardAmplitude,
            }
        );
        let overflow_ratio = power_ratio(c(f64::MAX / 2.0_f64.sqrt(), 0.0), 5).unwrap_err();
        assert_eq!(
            overflow_ratio,
            TwoPortTransducerGainError::Arithmetic {
                frequency: 5,
                stage: TwoPortTransducerGainArithmetic::PowerRatio,
            }
        );
        let underflow = power_ratio(c(f64::from_bits(1), 0.0), 6).unwrap_err();
        assert_eq!(
            underflow,
            TwoPortTransducerGainError::Arithmetic {
                frequency: 6,
                stage: TwoPortTransducerGainArithmetic::PowerRatio,
            }
        );
        let representable_subnormal = power_ratio(c(2.0_f64.powi(-537), 0.0), 7).unwrap();
        assert!(representable_subnormal > 0.0);
    }
}
