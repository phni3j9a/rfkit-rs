//! Direct-sum assembly for two independent frequency-major networks.
//!
//! This operation is deliberately a coordinate assembly rather than a
//! physical connection.  The output scattering matrix is the block diagonal
//! direct sum of the two input matrices, and the reference table is the two
//! input tables concatenated in port order.  No S or reference scalar is
//! interpreted or transformed here.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use std::mem::size_of;
use thiserror::Error;

use crate::ConnectionInput;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Failure modes for uncoupled direct-sum assembly.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum CombineUncoupledError {
    #[error(
        "{input} uncoupled combination S-parameter shape must be (nfreq, nport, nport), got {shape:?}"
    )]
    InvalidSShape {
        input: ConnectionInput,
        shape: (usize, usize, usize),
    },

    #[error(
        "{input} uncoupled combination reference-impedance shape must be (nfreq, nport), got {shape:?}"
    )]
    InvalidZ0Shape {
        input: ConnectionInput,
        shape: (usize, usize),
    },

    #[error("{input} uncoupled combination frequency axis must not be empty")]
    EmptyFrequency { input: ConnectionInput },

    #[error(
        "{input} uncoupled combination frequency axis length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    FrequencyShape {
        input: ConnectionInput,
        expected: usize,
        actual: usize,
    },

    #[error("A and B uncoupled-combination frequency axes have different lengths: A={a}, B={b}")]
    FrequencyLengthMismatch { a: usize, b: usize },

    #[error("uncoupled-combination frequency axes differ at index {index}: A={a:?}, B={b:?}")]
    FrequencyMismatch { index: usize, a: f64, b: f64 },

    #[error("{input} uncoupled-combination frequency is non-finite at index {index}: {value:?}")]
    NonFiniteFrequency {
        input: ConnectionInput,
        index: usize,
        value: f64,
    },

    #[error("uncoupled-combination port-count addition overflow for A={a} and B={b} ports")]
    PortCountOverflow { a: usize, b: usize },

    #[error(
        "uncoupled-combination {quantity} size is not representable for {nfreq} frequency samples and {nports} output ports"
    )]
    SizeOverflow {
        quantity: &'static str,
        nfreq: usize,
        nports: usize,
    },
}

/// Result of a successful direct-sum assembly.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CombinedNetwork {
    pub(crate) frequency_hz: Vec<f64>,
    pub(crate) s: Array3<Complex64>,
    pub(crate) z0: Array2<Complex64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InputShape {
    nfreq: usize,
    nports: usize,
}

/// Check a raw input before any sampled S/z0 indexing.
fn validate_input(
    input: ConnectionInput,
    frequency: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<InputShape, CombineUncoupledError> {
    if frequency.is_empty() {
        return Err(CombineUncoupledError::EmptyFrequency { input });
    }

    let s_shape = s.dim();
    if s_shape.0 != frequency.len() {
        return Err(CombineUncoupledError::FrequencyShape {
            input,
            expected: frequency.len(),
            actual: s_shape.0,
        });
    }
    if s_shape.1 == 0 || s_shape.1 != s_shape.2 {
        return Err(CombineUncoupledError::InvalidSShape {
            input,
            shape: s_shape,
        });
    }
    if z0.dim() != (s_shape.0, s_shape.1) {
        return Err(CombineUncoupledError::InvalidZ0Shape {
            input,
            shape: z0.dim(),
        });
    }

    // Frequency labels are the only scalar values with an operation-specific
    // validity rule.  S and z0 are intentionally opaque coordinate payloads.
    for (index, &value) in frequency.iter().enumerate() {
        if !value.is_finite() {
            return Err(CombineUncoupledError::NonFiniteFrequency {
                input,
                index,
                value,
            });
        }
    }

    Ok(InputShape {
        nfreq: s_shape.0,
        nports: s_shape.1,
    })
}

/// Validate output dimensions and ndarray's signed-index limit before arrays
/// are allocated.  This function is kept separate so overflow boundaries can
/// be tested without constructing a huge ndarray.
fn checked_output_shape(
    nfreq: usize,
    nports_a: usize,
    nports_b: usize,
) -> Result<(usize, usize), CombineUncoupledError> {
    let nports =
        nports_a
            .checked_add(nports_b)
            .ok_or(CombineUncoupledError::PortCountOverflow {
                a: nports_a,
                b: nports_b,
            })?;

    // ndarray's contiguous index calculations use isize offsets.  Requiring
    // every dimension and contiguous element count to fit that domain avoids
    // wrapping offsets before the first allocation or sampled write.
    let max_index = isize::MAX as usize;
    if nfreq > max_index || nports > max_index {
        return Err(CombineUncoupledError::SizeOverflow {
            quantity: "ndarray index dimensions",
            nfreq,
            nports,
        });
    }

    let Some(s_elements) = nfreq
        .checked_mul(nports)
        .and_then(|count| count.checked_mul(nports))
        .filter(|&count| count <= max_index)
    else {
        return Err(CombineUncoupledError::SizeOverflow {
            quantity: "output S elements",
            nfreq,
            nports,
        });
    };
    checked_allocation_bytes(
        s_elements,
        "output S allocation bytes",
        nfreq,
        nports,
        max_index,
    )?;

    let Some(z0_elements) = nfreq
        .checked_mul(nports)
        .filter(|&count| count <= max_index)
    else {
        return Err(CombineUncoupledError::SizeOverflow {
            quantity: "output reference-impedance elements",
            nfreq,
            nports,
        });
    };
    checked_allocation_bytes(
        z0_elements,
        "output reference-impedance allocation bytes",
        nfreq,
        nports,
        max_index,
    )?;

    Ok((nfreq, nports))
}

/// Check the byte-sized allocation and pointer-offset domain used by an owned
/// ndarray buffer without constructing the buffer.  `ndarray` checks the
/// element count against `isize::MAX`, while `Vec<Complex64>` also needs the
/// element count multiplied by the element size to remain representable in
/// the signed allocation/offset domain.
fn checked_allocation_bytes(
    elements: usize,
    quantity: &'static str,
    nfreq: usize,
    nports: usize,
    max_index: usize,
) -> Result<(), CombineUncoupledError> {
    match elements
        .checked_mul(size_of::<Complex64>())
        .filter(|&bytes| bytes <= max_index)
    {
        Some(_) => Ok(()),
        None => Err(CombineUncoupledError::SizeOverflow {
            quantity,
            nfreq,
            nports,
        }),
    }
}

/// Assemble `A ⊕ B` with A's ports followed by B's ports.
pub(crate) fn combine_uncoupled(
    frequency_a: &[f64],
    s_a: &Array3<Complex64>,
    z0_a: &Array2<Complex64>,
    frequency_b: &[f64],
    s_b: &Array3<Complex64>,
    z0_b: &Array2<Complex64>,
) -> Result<CombinedNetwork, CombineUncoupledError> {
    // Validate both malformed source structures before comparing sampled data
    // or allocating the output arrays.
    let shape_a = validate_input(ConnectionInput::A, frequency_a, s_a, z0_a)?;
    let shape_b = validate_input(ConnectionInput::B, frequency_b, s_b, z0_b)?;

    if frequency_a.len() != frequency_b.len() {
        return Err(CombineUncoupledError::FrequencyLengthMismatch {
            a: frequency_a.len(),
            b: frequency_b.len(),
        });
    }
    for (index, (&a, &b)) in frequency_a.iter().zip(frequency_b).enumerate() {
        if a != b {
            return Err(CombineUncoupledError::FrequencyMismatch { index, a, b });
        }
    }

    let (nfreq, nports) = checked_output_shape(shape_a.nfreq, shape_a.nports, shape_b.nports)?;

    // `from_elem` yields a fresh standard-layout allocation.  The cross blocks
    // are never written, so their values are canonical +0.0 + j+0.0.
    let mut output_s = Array3::from_elem((nfreq, nports, nports), ZERO);
    let mut output_z0 = Array2::from_elem((nfreq, nports), ZERO);

    for frequency in 0..nfreq {
        for row in 0..shape_a.nports {
            for column in 0..shape_a.nports {
                output_s[[frequency, row, column]] = s_a[[frequency, row, column]];
            }
            output_z0[[frequency, row]] = z0_a[[frequency, row]];
        }

        for row in 0..shape_b.nports {
            let output_row = shape_a.nports + row;
            output_z0[[frequency, output_row]] = z0_b[[frequency, row]];
            for column in 0..shape_b.nports {
                output_s[[frequency, output_row, shape_a.nports + column]] =
                    s_b[[frequency, row, column]];
            }
        }
    }

    Ok(CombinedNetwork {
        frequency_hz: frequency_a.to_vec(),
        s: output_s,
        z0: output_z0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_output_shape_rejects_port_addition_overflow() {
        let error = checked_output_shape(1, usize::MAX, 1).unwrap_err();
        assert!(matches!(
            error,
            CombineUncoupledError::PortCountOverflow { .. }
        ));
    }

    #[test]
    fn checked_output_shape_rejects_element_overflow_without_allocation() {
        let error = checked_output_shape(usize::MAX, 1, 1).unwrap_err();
        assert!(matches!(error, CombineUncoupledError::SizeOverflow { .. }));
    }

    #[test]
    fn checked_output_shape_rejects_s_byte_overflow_without_allocation() {
        let nfreq = (isize::MAX as usize / size_of::<Complex64>()) + 1;
        let error = checked_output_shape(nfreq, 1, 1).unwrap_err();
        assert!(matches!(
            error,
            CombineUncoupledError::SizeOverflow {
                quantity: "output S allocation bytes",
                ..
            }
        ));
    }

    #[test]
    fn checked_allocation_bytes_rejects_s_and_z0_boundaries_without_allocation() {
        let elements = (isize::MAX as usize / size_of::<Complex64>()) + 1;
        for quantity in [
            "output S allocation bytes",
            "output reference-impedance allocation bytes",
        ] {
            let error = checked_allocation_bytes(elements, quantity, 1, 1, isize::MAX as usize)
                .unwrap_err();
            assert!(matches!(
                error,
                CombineUncoupledError::SizeOverflow {
                    quantity: actual,
                    ..
                } if actual == quantity
            ));
        }
    }
}
