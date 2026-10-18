//! Exact sample-axis concatenation for two frequency-major networks.
//!
//! This is a structural operation: it copies observations in caller-provided
//! order and does not interpret frequency labels, S parameters, or reference
//! impedances.  The implementation is intentionally independent of the
//! port-axis direct-sum kernel in [`crate::combine`].

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::mem::size_of;
use thiserror::Error;

use crate::ConnectionInput;

/// Failure modes for exact frequency-sample concatenation.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum ConcatenateFrequencySamplesError {
    #[error(
        "{input} frequency-sample concatenation S-parameter shape must be (nfreq, nport, nport), got {shape:?}"
    )]
    InvalidSShape {
        input: ConnectionInput,
        shape: (usize, usize, usize),
    },

    #[error(
        "{input} frequency-sample concatenation reference-impedance shape must be (nfreq, nport), got {shape:?}"
    )]
    InvalidZ0Shape {
        input: ConnectionInput,
        shape: (usize, usize),
    },

    #[error("{input} frequency-sample concatenation frequency axis must not be empty")]
    EmptyFrequency { input: ConnectionInput },

    #[error(
        "{input} frequency-sample concatenation frequency axis length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    FrequencyShape {
        input: ConnectionInput,
        expected: usize,
        actual: usize,
    },

    #[error("frequency-sample concatenation requires equal port counts: A={a}, B={b}")]
    PortCountMismatch { a: usize, b: usize },

    #[error(
        "frequency-sample concatenation sample-count addition overflow for A={a} and B={b} samples"
    )]
    SampleCountOverflow { a: usize, b: usize },

    #[error(
        "frequency-sample concatenation {quantity} size is not representable for A={samples_a}, B={samples_b}, output {samples} samples and {nports} ports"
    )]
    SizeOverflow {
        quantity: &'static str,
        samples_a: usize,
        samples_b: usize,
        samples: usize,
        nports: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InputShape {
    samples: usize,
    nports: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OutputShape {
    samples: usize,
    nports: usize,
}

#[derive(Debug, Clone, Copy)]
struct SizeContext {
    samples_a: usize,
    samples_b: usize,
    samples: usize,
    nports: usize,
    max_index: usize,
}

#[derive(Debug, Clone, Copy)]
struct BufferCheck {
    first: usize,
    second: usize,
    third: usize,
    element_size: usize,
    elements_quantity: &'static str,
    bytes_quantity: &'static str,
}

type ConcatenatedArrays = (Vec<f64>, Array3<Complex64>, Array2<Complex64>);

/// Check one source structure before any sample-indexed access.
fn validate_input(
    input: ConnectionInput,
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<InputShape, ConcatenateFrequencySamplesError> {
    if frequency.is_empty() {
        return Err(ConcatenateFrequencySamplesError::EmptyFrequency { input });
    }

    let s_shape = s.dim();
    if s_shape.0 != frequency.len() {
        return Err(ConcatenateFrequencySamplesError::FrequencyShape {
            input,
            expected: frequency.len(),
            actual: s_shape.0,
        });
    }
    if s_shape.1 == 0 || s_shape.1 != s_shape.2 {
        return Err(ConcatenateFrequencySamplesError::InvalidSShape {
            input,
            shape: s_shape,
        });
    }
    if z0.dim() != (s_shape.0, s_shape.1) {
        return Err(ConcatenateFrequencySamplesError::InvalidZ0Shape {
            input,
            shape: z0.dim(),
        });
    }

    Ok(InputShape {
        samples: s_shape.0,
        nports: s_shape.1,
    })
}

/// Check output dimensions and all owned-buffer arithmetic before allocation.
///
/// `ndarray` uses signed `isize` offsets for indexing.  In addition to its
/// element-count bound, each owned buffer's byte size is checked against that
/// signed domain before a standard-layout array or frequency `Vec` is made.
fn checked_output_shape(
    samples_a: usize,
    samples_b: usize,
    nports: usize,
) -> Result<OutputShape, ConcatenateFrequencySamplesError> {
    let samples = samples_a.checked_add(samples_b).ok_or(
        ConcatenateFrequencySamplesError::SampleCountOverflow {
            a: samples_a,
            b: samples_b,
        },
    )?;
    let max_index = isize::MAX as usize;

    if nports == 0 {
        return Err(ConcatenateFrequencySamplesError::SizeOverflow {
            quantity: "output dimensions",
            samples_a,
            samples_b,
            samples,
            nports,
        });
    }
    if samples_a > max_index || samples_b > max_index || samples > max_index || nports > max_index {
        return Err(ConcatenateFrequencySamplesError::SizeOverflow {
            quantity: "ndarray index dimensions",
            samples_a,
            samples_b,
            samples,
            nports,
        });
    }

    let context = SizeContext {
        samples_a,
        samples_b,
        samples,
        nports,
        max_index,
    };
    checked_elements_and_bytes(
        BufferCheck {
            first: samples,
            second: nports,
            third: nports,
            element_size: size_of::<Complex64>(),
            elements_quantity: "output S elements",
            bytes_quantity: "output S allocation bytes",
        },
        context,
    )?;
    checked_elements_and_bytes(
        BufferCheck {
            first: samples,
            second: nports,
            third: 1,
            element_size: size_of::<Complex64>(),
            elements_quantity: "output reference-impedance elements",
            bytes_quantity: "output reference-impedance allocation bytes",
        },
        context,
    )?;
    checked_elements_and_bytes(
        BufferCheck {
            first: samples,
            second: 1,
            third: 1,
            element_size: size_of::<f64>(),
            elements_quantity: "output frequency elements",
            bytes_quantity: "output frequency allocation bytes",
        },
        context,
    )?;

    Ok(OutputShape { samples, nports })
}

/// Check a three-dimensional element product and its byte allocation bound.
fn checked_elements_and_bytes(
    buffer: BufferCheck,
    context: SizeContext,
) -> Result<(), ConcatenateFrequencySamplesError> {
    let Some(elements) = buffer
        .first
        .checked_mul(buffer.second)
        .and_then(|count| count.checked_mul(buffer.third))
        .filter(|&count| count <= context.max_index)
    else {
        return Err(ConcatenateFrequencySamplesError::SizeOverflow {
            quantity: buffer.elements_quantity,
            samples_a: context.samples_a,
            samples_b: context.samples_b,
            samples: context.samples,
            nports: context.nports,
        });
    };

    if elements
        .checked_mul(buffer.element_size)
        .filter(|&bytes| bytes <= context.max_index)
        .is_none()
    {
        return Err(ConcatenateFrequencySamplesError::SizeOverflow {
            quantity: buffer.bytes_quantity,
            samples_a: context.samples_a,
            samples_b: context.samples_b,
            samples: context.samples,
            nports: context.nports,
        });
    }

    Ok(())
}

/// Concatenate A's observations followed by B's observations.
pub(crate) fn concatenate_frequency_samples(
    frequency_a: &[f64],
    s_a: &Array3<Complex64>,
    z0_a: &Array2<Complex64>,
    frequency_b: &[f64],
    s_b: &Array3<Complex64>,
    z0_b: &Array2<Complex64>,
) -> Result<ConcatenatedArrays, ConcatenateFrequencySamplesError> {
    // Validate both source structures before checking port counts or touching
    // any sample-indexed value.  This matters for serde-created malformed
    // Networks, whose private fields bypass `Network::new`.
    let shape_a = validate_input(ConnectionInput::A, frequency_a, s_a, z0_a)?;
    let shape_b = validate_input(ConnectionInput::B, frequency_b, s_b, z0_b)?;

    if shape_a.nports != shape_b.nports {
        return Err(ConcatenateFrequencySamplesError::PortCountMismatch {
            a: shape_a.nports,
            b: shape_b.nports,
        });
    }

    let output = checked_output_shape(shape_a.samples, shape_b.samples, shape_a.nports)?;

    let mut frequency_hz = Vec::with_capacity(output.samples);
    frequency_hz.extend_from_slice(frequency_a);
    frequency_hz.extend_from_slice(frequency_b);

    let output_s = Array3::from_shape_fn(
        (output.samples, output.nports, output.nports),
        |(sample, row, column)| {
            if sample < shape_a.samples {
                s_a[[sample, row, column]]
            } else {
                s_b[[sample - shape_a.samples, row, column]]
            }
        },
    );
    let output_z0 = Array2::from_shape_fn((output.samples, output.nports), |(sample, port)| {
        if sample < shape_a.samples {
            z0_a[[sample, port]]
        } else {
            z0_b[[sample - shape_a.samples, port]]
        }
    });

    Ok((frequency_hz, output_s, output_z0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_output_shape_rejects_sample_addition_overflow() {
        let error = checked_output_shape(usize::MAX, 1, 1).unwrap_err();
        assert!(matches!(
            error,
            ConcatenateFrequencySamplesError::SampleCountOverflow { .. }
        ));
    }

    #[test]
    fn checked_output_shape_rejects_element_overflow_without_allocation() {
        let error = checked_output_shape((isize::MAX as usize / 2) + 1, 1, 2).unwrap_err();
        assert!(matches!(
            error,
            ConcatenateFrequencySamplesError::SizeOverflow {
                quantity: "output S elements",
                ..
            }
        ));
    }

    #[test]
    fn checked_output_shape_rejects_complex_byte_overflow_without_allocation() {
        let samples = (isize::MAX as usize / size_of::<Complex64>()) + 1;
        let error = checked_output_shape(samples, 1, 1).unwrap_err();
        assert!(matches!(
            error,
            ConcatenateFrequencySamplesError::SizeOverflow {
                quantity: "output S allocation bytes",
                ..
            }
        ));
    }

    #[test]
    fn checked_frequency_byte_overflow_is_rejected_without_allocation() {
        let samples = (isize::MAX as usize / size_of::<f64>()) + 1;
        let error = checked_elements_and_bytes(
            BufferCheck {
                first: samples,
                second: 1,
                third: 1,
                element_size: size_of::<f64>(),
                elements_quantity: "output frequency elements",
                bytes_quantity: "output frequency allocation bytes",
            },
            SizeContext {
                samples_a: samples,
                samples_b: 0,
                samples,
                nports: 1,
                max_index: isize::MAX as usize,
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ConcatenateFrequencySamplesError::SizeOverflow {
                quantity: "output frequency allocation bytes",
                ..
            }
        ));
    }
}
