//! Internal power-wave network-parameter conversions.
//!
//! The implementation in this module follows Kurokawa's power-wave
//! definition, independently of scikit-rf naming.  Port current `I_i` is
//! directed into the network, and the adopted convention is
//!
//! ```text
//! a_i = (V_i + z0_i I_i) / (2 sqrt(|Re z0_i|))
//! b_i = (V_i - conj(z0_i) I_i) / (2 sqrt(|Re z0_i|))
//! b = S a
//! V = Z I
//! ```
//!
//! Substituting these wave definitions into `b = S a` and collecting the
//! voltage/current terms gives the `F`/`G` expression used by the kernel:
//! `F = diag(1 / (2 sqrt(|Re z0|)))`, `G = diag(z0)`, and
//! `Z = F^-1 (I - S)^-1 (S G + G*) F`.  A zero real part in any reference
//! impedance makes this normalization undefined, so the kernel returns a
//! private error rather than producing a non-finite value.  It deliberately
//! works on the frequency-major arrays used by [`crate::Network`] and does
//! not expose a public conversion API yet.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Failure modes for the internal power-wave conversion kernel.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum PowerWaveError {
    #[error("S-parameter shape must be (nfreq, nport, nport), got {shape:?}")]
    InvalidSShape { shape: (usize, usize, usize) },

    #[error("Z-parameter shape must be (nfreq, nport, nport), got {shape:?}")]
    InvalidZShape { shape: (usize, usize, usize) },

    #[error("reference-impedance shape must be (nfreq, nport), got {shape:?}")]
    InvalidZ0Shape { shape: (usize, usize) },

    #[error("reference impedance has zero real part at frequency {frequency}, port {port}")]
    ZeroRealReferenceImpedance { frequency: usize, port: usize },

    #[error(
        "power-wave conversion system is exactly singular at frequency {frequency}, pivot {pivot}"
    )]
    Singular { frequency: usize, pivot: usize },

    #[error("non-finite S-parameter at frequency {frequency}, row {row}, column {column}")]
    NonFiniteS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("non-finite Z-parameter at frequency {frequency}, row {row}, column {column}")]
    NonFiniteZ {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("non-finite reference impedance at frequency {frequency}, port {port}")]
    NonFiniteZ0 { frequency: usize, port: usize },

    #[error(
        "non-finite value while solving power-wave conversion system at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteComputation {
        frequency: usize,
        row: usize,
        column: usize,
    },
}

/// Convert frequency-major power-wave S-parameters to Z-parameters.
///
/// This uses the Kurokawa convention (independent of scikit-rf naming): port
/// current `I_i` is directed into the network, with
///
/// ```text
/// a_i = (V_i + z0_i I_i) / (2 sqrt(|Re z0_i|))
/// b_i = (V_i - conj(z0_i) I_i) / (2 sqrt(|Re z0_i|))
/// b = S a
/// V = Z I
/// ```
///
/// These definitions produce `F = diag(1 / (2 * sqrt(|Re z0|)))` and
/// `G = diag(z0)` when the voltage/current equations are collected.  A zero
/// real part of `z0_i` is therefore undefined and is reported as a private
/// error instead of returning a non-finite result.
///
/// For every frequency sample this forms
///
/// ```text
/// A = (I - S) F
/// B = (S G + G*) F
/// ```
///
/// with `G = diag(z0)` and
/// `F = diag(1 / (2 * sqrt(abs(Re(z0)))))`, then solves `A Z = B` by
/// Gaussian elimination with partial pivoting.  No explicit matrix inverse,
/// near-singular tolerance, or pivot nudge is used.  A pivot is singular only
/// when the selected complex value is exactly zero.
#[allow(dead_code)] // Internal kernel is staged for future Network conversion call sites; unit tests exercise it now.
pub(crate) fn s_to_z_power(
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<Array3<Complex64>, PowerWaveError> {
    let (nfreq, nport_rows, nport_columns) = s.dim();
    if nport_rows == 0 || nport_rows != nport_columns {
        return Err(PowerWaveError::InvalidSShape {
            shape: (nfreq, nport_rows, nport_columns),
        });
    }
    let (z0_nfreq, z0_nport) = z0.dim();
    if (z0_nfreq, z0_nport) != (nfreq, nport_rows) {
        return Err(PowerWaveError::InvalidZ0Shape {
            shape: (z0_nfreq, z0_nport),
        });
    }

    let nport = nport_rows;
    let mut z = Array3::from_elem((nfreq, nport, nport), ZERO);

    for frequency in 0..nfreq {
        let mut normalization = vec![ZERO; nport];
        for port in 0..nport {
            let impedance = z0[[frequency, port]];
            if !is_finite(impedance) {
                return Err(PowerWaveError::NonFiniteZ0 { frequency, port });
            }
            if impedance.re == 0.0 {
                return Err(PowerWaveError::ZeroRealReferenceImpedance { frequency, port });
            }

            let scale = 2.0 * impedance.re.abs().sqrt();
            let f = 1.0 / scale;
            let f = Complex64::new(f, 0.0);
            if !is_finite(f) {
                return Err(PowerWaveError::NonFiniteComputation {
                    frequency,
                    row: port,
                    column: port,
                });
            }
            normalization[port] = f;
        }

        // A and B are stored row-major.  The right multiplication by the
        // diagonal F therefore scales each column by that column's factor.
        let mut a = vec![ZERO; nport * nport];
        let mut b = vec![ZERO; nport * nport];
        for row in 0..nport {
            for column in 0..nport {
                let s_value = s[[frequency, row, column]];
                if !is_finite(s_value) {
                    return Err(PowerWaveError::NonFiniteS {
                        frequency,
                        row,
                        column,
                    });
                }

                let index = row * nport + column;
                let identity = if row == column { 1.0 } else { 0.0 };
                let a_value = (Complex64::new(identity, 0.0) - s_value) * normalization[column];

                let mut b_value = s_value * z0[[frequency, column]];
                if row == column {
                    b_value += z0[[frequency, row]].conj();
                }
                b_value *= normalization[column];

                if !is_finite(a_value) || !is_finite(b_value) {
                    return Err(PowerWaveError::NonFiniteComputation {
                        frequency,
                        row,
                        column,
                    });
                }
                a[index] = a_value;
                b[index] = b_value;
            }
        }

        solve_multiple_rhs(&mut a, &mut b, nport, frequency)?;

        for row in 0..nport {
            for column in 0..nport {
                let value = b[row * nport + column];
                if !is_finite(value) {
                    return Err(PowerWaveError::NonFiniteComputation {
                        frequency,
                        row,
                        column,
                    });
                }
                z[[frequency, row, column]] = value;
            }
        }
    }

    Ok(z)
}

/// Convert frequency-major power-wave Z-parameters to S-parameters.
///
/// This is the inverse of [`s_to_z_power`] under the same Kurokawa
/// convention.  For every frequency sample it forms
///
/// ```text
/// A = F (Z + G)
/// B = F (Z - G*)
/// ```
///
/// with `G = diag(z0)` and
/// `F = diag(1 / (2 * sqrt(abs(Re(z0)))))`, then solves `S A = B`.  The
/// existing exact-pivot left solver is reused by transposing the two systems
/// without conjugation: `A^T S^T = B^T`.  This avoids an explicit matrix
/// inverse and preserves the exact-zero-pivot singularity rule.
#[allow(dead_code)] // Internal kernel is staged for future Network conversion call sites; unit tests exercise it now.
pub(crate) fn z_to_s_power(
    z: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<Array3<Complex64>, PowerWaveError> {
    let (nfreq, nport_rows, nport_columns) = z.dim();
    if nport_rows == 0 || nport_rows != nport_columns {
        return Err(PowerWaveError::InvalidZShape {
            shape: (nfreq, nport_rows, nport_columns),
        });
    }
    let (z0_nfreq, z0_nport) = z0.dim();
    if (z0_nfreq, z0_nport) != (nfreq, nport_rows) {
        return Err(PowerWaveError::InvalidZ0Shape {
            shape: (z0_nfreq, z0_nport),
        });
    }

    let nport = nport_rows;
    let mut s = Array3::from_elem((nfreq, nport, nport), ZERO);

    for frequency in 0..nfreq {
        let mut normalization = vec![ZERO; nport];
        for port in 0..nport {
            let impedance = z0[[frequency, port]];
            if !is_finite(impedance) {
                return Err(PowerWaveError::NonFiniteZ0 { frequency, port });
            }
            if impedance.re == 0.0 {
                return Err(PowerWaveError::ZeroRealReferenceImpedance { frequency, port });
            }

            let scale = 2.0 * impedance.re.abs().sqrt();
            let f = 1.0 / scale;
            let f = Complex64::new(f, 0.0);
            if !is_finite(f) {
                return Err(PowerWaveError::NonFiniteComputation {
                    frequency,
                    row: port,
                    column: port,
                });
            }
            normalization[port] = f;
        }

        // A and B are stored as plain transposes so the existing left solver
        // computes A^T S^T = B^T.  Transposition here deliberately does not
        // conjugate complex values.
        let mut a_transpose = vec![ZERO; nport * nport];
        let mut b_transpose = vec![ZERO; nport * nport];
        for row in 0..nport {
            for column in 0..nport {
                let z_value = z[[frequency, row, column]];
                if !is_finite(z_value) {
                    return Err(PowerWaveError::NonFiniteZ {
                        frequency,
                        row,
                        column,
                    });
                }

                // F multiplies from the left, so it scales each row by its
                // corresponding normalization factor.
                let mut a_value = z_value;
                let mut b_value = z_value;
                if row == column {
                    a_value += z0[[frequency, row]];
                    b_value -= z0[[frequency, row]].conj();
                }
                a_value *= normalization[row];
                b_value *= normalization[row];

                if !is_finite(a_value) || !is_finite(b_value) {
                    return Err(PowerWaveError::NonFiniteComputation {
                        frequency,
                        row,
                        column,
                    });
                }

                // A^T[column, row] = A[row, column] and likewise for B.
                let index = column * nport + row;
                a_transpose[index] = a_value;
                b_transpose[index] = b_value;
            }
        }

        solve_multiple_rhs(&mut a_transpose, &mut b_transpose, nport, frequency)?;

        for row in 0..nport {
            for column in 0..nport {
                // The solved value is S^T[row, column] = S[column, row].
                let value = b_transpose[row * nport + column];
                if !is_finite(value) {
                    return Err(PowerWaveError::NonFiniteComputation {
                        frequency,
                        row,
                        column,
                    });
                }
                s[[frequency, column, row]] = value;
            }
        }
    }

    Ok(s)
}

/// Renormalize frequency-major power-wave S-parameters between two explicit
/// reference-impedance arrays without changing the underlying impedance
/// network.
///
/// This is Kurokawa's power-wave renormalization: Eq. (19) first obtains the
/// network impedance matrix from the source-referenced S matrix, and Eq. (18)
/// then expresses that same matrix as a target-referenced S matrix.  Both
/// reference arrays are `(nfreq, nport)` and may vary by frequency and port;
/// complex reference impedances are accepted when their real parts are
/// non-zero.
///
/// The operation intentionally delegates to the existing conversion kernels
/// rather than taking an identity shortcut or introducing a separate matrix
/// formula.  Consequently it preserves their exact validation and failure
/// policy: malformed shapes, non-finite values, and zero-real reference
/// impedances are rejected, and only an exactly-zero selected pivot is
/// considered singular (no near-singular tolerance, pivot nudge, or fallback).
#[allow(dead_code)] // Internal kernel is staged for future Network conversion call sites; unit tests exercise it now.
pub(crate) fn renormalize_s_power(
    s: &Array3<Complex64>,
    source_z0: &Array2<Complex64>,
    target_z0: &Array2<Complex64>,
) -> Result<Array3<Complex64>, PowerWaveError> {
    let z = s_to_z_power(s, source_z0)?;
    z_to_s_power(&z, target_z0)
}

/// Solve `A X = B` in place for a square `A` and multiple right-hand sides.
///
/// The only singularity criterion is an exactly-zero selected pivot.  This is
/// intentional: a tolerance or diagonal nudge would silently change the
/// requested network conversion semantics.
fn solve_multiple_rhs(
    a: &mut [Complex64],
    b: &mut [Complex64],
    n: usize,
    frequency: usize,
) -> Result<(), PowerWaveError> {
    for pivot_index in 0..n {
        let mut pivot_row = pivot_index;
        let mut pivot_magnitude = a[pivot_index * n + pivot_index].norm_sqr();

        for row in (pivot_index + 1)..n {
            let candidate = a[row * n + pivot_index];
            if !is_finite(candidate) {
                return Err(PowerWaveError::NonFiniteComputation {
                    frequency,
                    row,
                    column: pivot_index,
                });
            }
            let candidate_magnitude = candidate.norm_sqr();
            if candidate_magnitude > pivot_magnitude {
                pivot_row = row;
                pivot_magnitude = candidate_magnitude;
            }
        }

        let pivot = a[pivot_row * n + pivot_index];
        if !is_finite(pivot) {
            return Err(PowerWaveError::NonFiniteComputation {
                frequency,
                row: pivot_row,
                column: pivot_index,
            });
        }
        if pivot == ZERO {
            return Err(PowerWaveError::Singular {
                frequency,
                pivot: pivot_index,
            });
        }

        if pivot_row != pivot_index {
            for column in 0..n {
                a.swap(pivot_index * n + column, pivot_row * n + column);
                b.swap(pivot_index * n + column, pivot_row * n + column);
            }
        }

        let pivot_value = a[pivot_index * n + pivot_index];
        for row in (pivot_index + 1)..n {
            let factor = a[row * n + pivot_index] / pivot_value;
            if !is_finite(factor) {
                return Err(PowerWaveError::NonFiniteComputation {
                    frequency,
                    row,
                    column: pivot_index,
                });
            }

            a[row * n + pivot_index] = ZERO;
            for column in (pivot_index + 1)..n {
                let index = row * n + column;
                a[index] -= factor * a[pivot_index * n + column];
                if !is_finite(a[index]) {
                    return Err(PowerWaveError::NonFiniteComputation {
                        frequency,
                        row,
                        column,
                    });
                }
            }
            for rhs_column in 0..n {
                let index = row * n + rhs_column;
                b[index] -= factor * b[pivot_index * n + rhs_column];
                if !is_finite(b[index]) {
                    return Err(PowerWaveError::NonFiniteComputation {
                        frequency,
                        row,
                        column: rhs_column,
                    });
                }
            }
        }
    }

    for row in (0..n).rev() {
        let pivot = a[row * n + row];
        if !is_finite(pivot) {
            return Err(PowerWaveError::NonFiniteComputation {
                frequency,
                row,
                column: row,
            });
        }
        if pivot == ZERO {
            return Err(PowerWaveError::Singular {
                frequency,
                pivot: row,
            });
        }

        for rhs_column in 0..n {
            let mut value = b[row * n + rhs_column];
            for column in (row + 1)..n {
                value -= a[row * n + column] * b[column * n + rhs_column];
                if !is_finite(value) {
                    return Err(PowerWaveError::NonFiniteComputation {
                        frequency,
                        row,
                        column: rhs_column,
                    });
                }
            }
            let solution = value / pivot;
            if !is_finite(solution) {
                return Err(PowerWaveError::NonFiniteComputation {
                    frequency,
                    row,
                    column: rhs_column,
                });
            }
            b[row * n + rhs_column] = solution;
        }
    }

    Ok(())
}

fn is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    const FIXTURE_JSON: &str =
        include_str!("../../../tools/oracle/fixtures/power_wave_s_to_z_three_port_complex_z0.json");
    const Z_TO_S_FIXTURE_JSON: &str =
        include_str!("../../../tools/oracle/fixtures/power_wave_z_to_s_three_port_complex_z0.json");
    const RENORMALIZE_FIXTURES: &[(&str, &str)] = &[
        (
            "power_wave_renormalize_one_port_real_scalar_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_renormalize_one_port_real_scalar_z0.json"
            ),
        ),
        (
            "power_wave_renormalize_two_port_complex_per_port_constant_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_renormalize_two_port_complex_per_port_constant_z0.json"
            ),
        ),
        (
            "power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0.json"
            ),
        ),
        (
            "power_wave_renormalize_eight_port_real_frequency_dependent_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_renormalize_eight_port_real_frequency_dependent_z0.json"
            ),
        ),
        (
            "power_wave_renormalize_three_port_reciprocal_real_equal_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_renormalize_three_port_reciprocal_real_equal_z0.json"
            ),
        ),
    ];
    const MATRIX_S_TO_Z_FIXTURES: &[(&str, &str)] = &[
        (
            "power_wave_s_to_z_one_port_real_scalar_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_s_to_z_one_port_real_scalar_z0.json"
            ),
        ),
        (
            "power_wave_s_to_z_two_port_complex_per_port_constant_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_s_to_z_two_port_complex_per_port_constant_z0.json"
            ),
        ),
        (
            "power_wave_s_to_z_four_port_real_frequency_dependent_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_s_to_z_four_port_real_frequency_dependent_z0.json"
            ),
        ),
        (
            "power_wave_s_to_z_eight_port_complex_per_port_frequency_dependent_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_s_to_z_eight_port_complex_per_port_frequency_dependent_z0.json"
            ),
        ),
        (
            "power_wave_s_to_z_three_port_reciprocal_real_equal_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_s_to_z_three_port_reciprocal_real_equal_z0.json"
            ),
        ),
    ];
    const MATRIX_Z_TO_S_FIXTURES: &[(&str, &str)] = &[
        (
            "power_wave_z_to_s_one_port_real_scalar_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_z_to_s_one_port_real_scalar_z0.json"
            ),
        ),
        (
            "power_wave_z_to_s_two_port_complex_per_port_constant_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_z_to_s_two_port_complex_per_port_constant_z0.json"
            ),
        ),
        (
            "power_wave_z_to_s_four_port_real_frequency_dependent_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_z_to_s_four_port_real_frequency_dependent_z0.json"
            ),
        ),
        (
            "power_wave_z_to_s_eight_port_complex_per_port_frequency_dependent_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_z_to_s_eight_port_complex_per_port_frequency_dependent_z0.json"
            ),
        ),
        (
            "power_wave_z_to_s_three_port_reciprocal_real_equal_z0",
            include_str!(
                "../../../tools/oracle/fixtures/power_wave_z_to_s_three_port_reciprocal_real_equal_z0.json"
            ),
        ),
    ];
    const ACTIVE_S_TO_Z_FIXTURE: &str = include_str!(
        "../../../tools/oracle/fixtures/power_wave_s_to_z_three_port_active_real_equal_z0.json"
    );
    const ACTIVE_Z_TO_S_FIXTURE: &str = include_str!(
        "../../../tools/oracle/fixtures/power_wave_z_to_s_three_port_active_real_equal_z0.json"
    );
    const ACTIVE_RENORMALIZE_FIXTURE: &str = include_str!(
        "../../../tools/oracle/fixtures/power_wave_renormalize_three_port_active_real_equal_z0.json"
    );
    const ACTIVE_S_TO_Z_FIXTURES: &[(&str, &str)] = &[(
        "power_wave_s_to_z_three_port_active_real_equal_z0",
        ACTIVE_S_TO_Z_FIXTURE,
    )];
    const ACTIVE_Z_TO_S_FIXTURES: &[(&str, &str)] = &[(
        "power_wave_z_to_s_three_port_active_real_equal_z0",
        ACTIVE_Z_TO_S_FIXTURE,
    )];
    const ACTIVE_RENORMALIZE_FIXTURES: &[(&str, &str)] = &[(
        "power_wave_renormalize_three_port_active_real_equal_z0",
        ACTIVE_RENORMALIZE_FIXTURE,
    )];

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FixtureDocument {
        data: FixtureData,
        metadata: FixtureMetadata,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FixtureData {
        frequency_hz: Vec<f64>,
        s: Vec<Vec<Vec<ComplexValue>>>,
        z0_ohm: Vec<Vec<ComplexValue>>,
        z_ohm: Vec<Vec<Vec<ComplexValue>>>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ComplexValue {
        imag: f64,
        real: f64,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FixtureMetadata {
        case_id: String,
        input_case_id: String,
        numpy_version: String,
        operation: String,
        random_seed: u64,
        reference_impedance: ReferenceImpedanceMetadata,
        schema: String,
        schema_version: u32,
        scikit_rf_version: String,
        shape: FixtureShape,
        tolerance_policy: TolerancePolicy,
        wave_definition: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ReferenceImpedanceMetadata {
        complex: bool,
        frequency_dependent: bool,
        per_port: bool,
        unit: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FixtureShape {
        frequency: Vec<usize>,
        input_s: Vec<usize>,
        input_z0: Vec<usize>,
        output_z: Vec<usize>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct TolerancePolicy {
        atol_ohm: f64,
        comparison: String,
        justification: String,
        regeneration: String,
        rtol: f64,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ZToSFixtureDocument {
        data: ZToSFixtureData,
        metadata: ZToSFixtureMetadata,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ZToSFixtureData {
        frequency_hz: Vec<f64>,
        s: Vec<Vec<Vec<ComplexValue>>>,
        z0_ohm: Vec<Vec<ComplexValue>>,
        z_ohm: Vec<Vec<Vec<ComplexValue>>>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ZToSFixtureMetadata {
        case_id: String,
        numpy_version: String,
        operation: String,
        random_seed: u64,
        reference_impedance: ReferenceImpedanceMetadata,
        schema: String,
        schema_version: u32,
        scikit_rf_version: String,
        shape: ZToSFixtureShape,
        tolerance_policy: ZToSTolerancePolicy,
        wave_definition: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ZToSFixtureShape {
        frequency: Vec<usize>,
        input_z: Vec<usize>,
        input_z0: Vec<usize>,
        output_s: Vec<usize>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ZToSTolerancePolicy {
        atol: f64,
        comparison: String,
        justification: String,
        regeneration: String,
        rtol: f64,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ActiveNetworkMetadata {
        criterion: String,
        matrix_field: String,
        observed_minimum: f64,
        required_minimum: f64,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct MatrixFixtureDocument {
        data: MatrixFixtureData,
        metadata: MatrixFixtureMetadata,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct MatrixFixtureData {
        frequency_hz: Vec<f64>,
        s: Vec<Vec<Vec<ComplexValue>>>,
        z0_ohm: Vec<Vec<ComplexValue>>,
        z_ohm: Vec<Vec<Vec<ComplexValue>>>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct MatrixFixtureMetadata {
        #[serde(default)]
        active_network: Option<ActiveNetworkMetadata>,
        case_id: String,
        numpy_version: String,
        operation: String,
        random_seed: u64,
        reference_impedance: ReferenceImpedanceMetadata,
        schema: String,
        schema_version: u32,
        scikit_rf_version: String,
        shape: MatrixFixtureShape,
        tolerance_policy: MatrixTolerancePolicy,
        wave_definition: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct MatrixFixtureShape {
        frequency: Vec<usize>,
        #[serde(default)]
        input_s: Option<Vec<usize>>,
        #[serde(default)]
        input_z: Option<Vec<usize>>,
        #[serde(default)]
        input_z0: Option<Vec<usize>>,
        #[serde(default)]
        output_s: Option<Vec<usize>>,
        #[serde(default)]
        output_z: Option<Vec<usize>>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct MatrixTolerancePolicy {
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        atol_ohm: Option<f64>,
        comparison: String,
        justification: String,
        regeneration: String,
        rtol: f64,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RenormalizationFixtureDocument {
        data: RenormalizationFixtureData,
        metadata: RenormalizationFixtureMetadata,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RenormalizationFixtureData {
        frequency_hz: Vec<f64>,
        s_input: Vec<Vec<Vec<ComplexValue>>>,
        s_renormalized: Vec<Vec<Vec<ComplexValue>>>,
        z0_source_ohm: Vec<Vec<ComplexValue>>,
        z0_target_ohm: Vec<Vec<ComplexValue>>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RenormalizationFixtureMetadata {
        #[serde(default)]
        active_network: Option<ActiveNetworkMetadata>,
        case_id: String,
        numpy_version: String,
        operation: String,
        random_seed: u64,
        reference_impedance: RenormalizationReferenceImpedanceMetadata,
        schema: String,
        schema_version: u32,
        scikit_rf_version: String,
        shape: RenormalizationFixtureShape,
        tolerance_policy: RenormalizationTolerancePolicy,
        wave_definition: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RenormalizationReferenceImpedanceMetadata {
        source: ReferenceImpedanceMetadata,
        target: ReferenceImpedanceMetadata,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RenormalizationFixtureShape {
        frequency: Vec<usize>,
        s_input: Vec<usize>,
        s_renormalized: Vec<usize>,
        z0_source: Vec<usize>,
        z0_target: Vec<usize>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RenormalizationTolerancePolicy {
        atol: f64,
        comparison: String,
        justification: String,
        regeneration: String,
        rtol: f64,
    }

    type Matrix2 = [[Complex64; 2]; 2];

    /// Multiply two 2x2 matrices for the Eq. (18) invariant test.
    ///
    /// This intentionally stays independent of the production linear solver;
    /// the test's inverse is the closed-form 2x2 adjugate formula below.
    fn multiply_2x2(lhs: Matrix2, rhs: Matrix2) -> Matrix2 {
        let mut product = [[ZERO; 2]; 2];
        for row in 0..2 {
            for column in 0..2 {
                for inner in 0..2 {
                    product[row][column] += lhs[row][inner] * rhs[inner][column];
                }
            }
        }
        product
    }

    /// Invert a 2x2 matrix with the closed-form adjugate expression.
    fn inverse_2x2(matrix: Matrix2) -> Matrix2 {
        let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        assert_ne!(determinant, ZERO, "test matrix must be nonsingular");
        let reciprocal_determinant = 1.0 / determinant;
        [
            [
                matrix[1][1] * reciprocal_determinant,
                -matrix[0][1] * reciprocal_determinant,
            ],
            [
                -matrix[1][0] * reciprocal_determinant,
                matrix[0][0] * reciprocal_determinant,
            ],
        ]
    }

    /// Reconstruct S using Kurokawa Eq. (18), independently of production
    /// `solve_multiple_rhs`:
    /// `S = F (Z - G*) (Z + G)^-1 F^-1`.
    fn reconstruct_s_from_z_eq18(
        z: &Array3<Complex64>,
        z0: &Array2<Complex64>,
        frequency: usize,
    ) -> Matrix2 {
        let mut f = [[ZERO; 2]; 2];
        let mut f_inverse = [[ZERO; 2]; 2];
        let mut z_minus_g_conjugate = [[ZERO; 2]; 2];
        let mut z_plus_g = [[ZERO; 2]; 2];

        for port in 0..2 {
            let impedance = z0[[frequency, port]];
            let scale = 2.0 * impedance.re.abs().sqrt();
            let normalization = Complex64::new(1.0 / scale, 0.0);
            f[port][port] = normalization;
            f_inverse[port][port] = 1.0 / normalization;
        }

        for row in 0..2 {
            for column in 0..2 {
                let z_value = z[[frequency, row, column]];
                z_minus_g_conjugate[row][column] = z_value;
                z_plus_g[row][column] = z_value;
                if row == column {
                    z_minus_g_conjugate[row][column] -= z0[[frequency, row]].conj();
                    z_plus_g[row][column] += z0[[frequency, row]];
                }
            }
        }

        let middle = multiply_2x2(z_minus_g_conjugate, inverse_2x2(z_plus_g));
        multiply_2x2(multiply_2x2(f, middle), f_inverse)
    }

    fn matrix_complex(value: &ComplexValue) -> Complex64 {
        Complex64::new(value.real, value.imag)
    }

    #[derive(Debug, Clone, Copy)]
    struct MatrixCaseSpec {
        nfreq: usize,
        nport: usize,
        complex_z0: bool,
        frequency_dependent_z0: bool,
        per_port_z0: bool,
        reciprocal: bool,
        active: bool,
    }

    fn matrix_case_spec(case_id: &str) -> MatrixCaseSpec {
        match case_id {
            "power_wave_s_to_z_one_port_real_scalar_z0"
            | "power_wave_z_to_s_one_port_real_scalar_z0" => MatrixCaseSpec {
                nfreq: 3,
                nport: 1,
                complex_z0: false,
                frequency_dependent_z0: false,
                per_port_z0: false,
                reciprocal: false,
                active: false,
            },
            "power_wave_s_to_z_two_port_complex_per_port_constant_z0"
            | "power_wave_z_to_s_two_port_complex_per_port_constant_z0" => MatrixCaseSpec {
                nfreq: 4,
                nport: 2,
                complex_z0: true,
                frequency_dependent_z0: false,
                per_port_z0: true,
                reciprocal: false,
                active: false,
            },
            "power_wave_s_to_z_four_port_real_frequency_dependent_z0"
            | "power_wave_z_to_s_four_port_real_frequency_dependent_z0" => MatrixCaseSpec {
                nfreq: 3,
                nport: 4,
                complex_z0: false,
                frequency_dependent_z0: true,
                per_port_z0: false,
                reciprocal: false,
                active: false,
            },
            "power_wave_s_to_z_eight_port_complex_per_port_frequency_dependent_z0"
            | "power_wave_z_to_s_eight_port_complex_per_port_frequency_dependent_z0" => {
                MatrixCaseSpec {
                    nfreq: 3,
                    nport: 8,
                    complex_z0: true,
                    frequency_dependent_z0: true,
                    per_port_z0: true,
                    reciprocal: false,
                    active: false,
                }
            }
            "power_wave_s_to_z_three_port_reciprocal_real_equal_z0"
            | "power_wave_z_to_s_three_port_reciprocal_real_equal_z0" => MatrixCaseSpec {
                nfreq: 3,
                nport: 3,
                complex_z0: false,
                frequency_dependent_z0: false,
                per_port_z0: false,
                reciprocal: true,
                active: false,
            },
            "power_wave_s_to_z_three_port_active_real_equal_z0"
            | "power_wave_z_to_s_three_port_active_real_equal_z0" => MatrixCaseSpec {
                nfreq: 3,
                nport: 3,
                complex_z0: false,
                frequency_dependent_z0: false,
                per_port_z0: false,
                reciprocal: false,
                active: true,
            },
            _ => panic!("unexpected power-wave matrix case id: {case_id}"),
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct RenormalizationCaseSpec {
        nfreq: usize,
        nport: usize,
        complex_z0: bool,
        frequency_dependent_z0: bool,
        per_port_z0: bool,
        reciprocal: bool,
        active: bool,
    }

    fn renormalization_case_spec(case_id: &str) -> RenormalizationCaseSpec {
        match case_id {
            "power_wave_renormalize_one_port_real_scalar_z0" => RenormalizationCaseSpec {
                nfreq: 3,
                nport: 1,
                complex_z0: false,
                frequency_dependent_z0: false,
                per_port_z0: false,
                reciprocal: false,
                active: false,
            },
            "power_wave_renormalize_two_port_complex_per_port_constant_z0" => {
                RenormalizationCaseSpec {
                    nfreq: 4,
                    nport: 2,
                    complex_z0: true,
                    frequency_dependent_z0: false,
                    per_port_z0: true,
                    reciprocal: false,
                    active: false,
                }
            }
            "power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0" => {
                RenormalizationCaseSpec {
                    nfreq: 3,
                    nport: 4,
                    complex_z0: true,
                    frequency_dependent_z0: true,
                    per_port_z0: true,
                    reciprocal: false,
                    active: false,
                }
            }
            "power_wave_renormalize_eight_port_real_frequency_dependent_z0" => {
                RenormalizationCaseSpec {
                    nfreq: 3,
                    nport: 8,
                    complex_z0: false,
                    frequency_dependent_z0: true,
                    per_port_z0: false,
                    reciprocal: false,
                    active: false,
                }
            }
            "power_wave_renormalize_three_port_reciprocal_real_equal_z0" => {
                RenormalizationCaseSpec {
                    nfreq: 3,
                    nport: 3,
                    complex_z0: false,
                    frequency_dependent_z0: false,
                    per_port_z0: false,
                    reciprocal: true,
                    active: false,
                }
            }
            "power_wave_renormalize_three_port_active_real_equal_z0" => RenormalizationCaseSpec {
                nfreq: 3,
                nport: 3,
                complex_z0: false,
                frequency_dependent_z0: false,
                per_port_z0: false,
                reciprocal: false,
                active: true,
            },
            _ => panic!("unexpected power-wave renormalization case id: {case_id}"),
        }
    }

    fn matrix_parameter_array(
        values: &[Vec<Vec<ComplexValue>>],
        nfreq: usize,
        nport: usize,
    ) -> Array3<Complex64> {
        Array3::from_shape_fn((nfreq, nport, nport), |(frequency, row, column)| {
            matrix_complex(&values[frequency][row][column])
        })
    }

    fn matrix_z0_array(
        values: &[Vec<ComplexValue>],
        nfreq: usize,
        nport: usize,
    ) -> Array2<Complex64> {
        Array2::from_shape_fn((nfreq, nport), |(frequency, port)| {
            matrix_complex(&values[frequency][port])
        })
    }

    const ACTIVE_REQUIRED_SIGMA_MAX: f64 = 1.2;
    const ACTIVE_NETWORK_CRITERION: &str =
        "largest singular value of the relevant power-wave S matrix is strictly greater than 1";
    const ACTIVE_METADATA_ROUNDING_TOLERANCE: f64 = 1e-12;

    /// Validate active-network metadata without introducing an SVD dependency.
    ///
    /// NumPy records the true sigma-max minimum in the fixture.  The Rust
    /// tests independently prove the strict active bound with column norms in
    /// `assert_active_column_norm_lower_bound`, using
    /// `sigma_max >= max(column 2-norm)`.
    fn validate_active_network_metadata<'a>(
        metadata: Option<&'a ActiveNetworkMetadata>,
        expected_active: bool,
        expected_matrix_field: &str,
    ) -> Option<&'a ActiveNetworkMetadata> {
        if !expected_active {
            assert!(
                metadata.is_none(),
                "non-active fixture must not grow active-network metadata"
            );
            return None;
        }

        let active = metadata.expect("active fixture must record active-network metadata");
        assert_eq!(active.criterion, ACTIVE_NETWORK_CRITERION);
        assert_eq!(active.matrix_field, expected_matrix_field);
        assert_eq!(active.required_minimum, ACTIVE_REQUIRED_SIGMA_MAX);
        assert!(active.observed_minimum.is_finite());
        assert!(
            active.observed_minimum > active.required_minimum,
            "recorded active sigma-max minimum must exceed its required bound"
        );
        Some(active)
    }

    fn column_two_norm(matrix: &[Vec<ComplexValue>], column: usize) -> f64 {
        matrix
            .iter()
            .map(|row| {
                let value = matrix_complex(&row[column]);
                value.norm_sqr()
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Independently certify each active frequency with a valid sigma-max lower bound.
    fn assert_active_column_norm_lower_bound(
        case_id: &str,
        matrices: &[Vec<Vec<ComplexValue>>],
        active: &ActiveNetworkMetadata,
    ) {
        let mut minimum_column_bound = f64::INFINITY;
        for (frequency, matrix) in matrices.iter().enumerate() {
            assert_eq!(
                matrix.len(),
                3,
                "{case_id} active matrix must be three-port"
            );
            assert!(matrix.iter().all(|row| row.len() == 3));
            let column_bound = (0..3)
                .map(|column| column_two_norm(matrix, column))
                .fold(0.0_f64, f64::max);
            assert!(
                column_bound.is_finite(),
                "{case_id} active column bound is non-finite at frequency {frequency}"
            );
            assert!(
                column_bound > active.required_minimum,
                "{case_id} active column bound {column_bound:?} does not exceed required sigma-max minimum at frequency {frequency}"
            );
            minimum_column_bound = minimum_column_bound.min(column_bound);
        }
        assert!(
            active.observed_minimum + ACTIVE_METADATA_ROUNDING_TOLERANCE >= minimum_column_bound,
            "{case_id} recorded sigma-max minimum {:?} is below the independently certified column-norm lower bound {minimum_column_bound:?}",
            active.observed_minimum
        );
    }

    fn validate_renormalization_fixture_contract(
        fixture: &RenormalizationFixtureDocument,
        expected_case_id: &str,
        spec: RenormalizationCaseSpec,
    ) -> (f64, f64) {
        let metadata = &fixture.metadata;
        let data = &fixture.data;
        let expected_reciprocal = spec.reciprocal;
        let active_network = validate_active_network_metadata(
            metadata.active_network.as_ref(),
            spec.active,
            "s_input",
        );
        assert_eq!(metadata.case_id, expected_case_id);
        assert_eq!(metadata.operation, "renormalize_s");
        assert_eq!(metadata.wave_definition, "power");
        assert_eq!(metadata.schema, "rfkit-rs.oracle.fixture");
        assert_eq!(metadata.schema_version, 1);
        assert_eq!(metadata.numpy_version, "2.5.1");
        assert_eq!(metadata.scikit_rf_version, "2.0.1");
        assert!(metadata.random_seed > 0);

        for reference_impedance in [
            &metadata.reference_impedance.source,
            &metadata.reference_impedance.target,
        ] {
            assert_eq!(reference_impedance.complex, spec.complex_z0);
            assert_eq!(
                reference_impedance.frequency_dependent,
                spec.frequency_dependent_z0
            );
            assert_eq!(reference_impedance.per_port, spec.per_port_z0);
            assert_eq!(reference_impedance.unit, "ohm");
        }

        assert_eq!(metadata.shape.frequency, vec![spec.nfreq]);
        assert_eq!(
            metadata.shape.s_input,
            vec![spec.nfreq, spec.nport, spec.nport]
        );
        assert_eq!(
            metadata.shape.s_renormalized,
            vec![spec.nfreq, spec.nport, spec.nport]
        );
        assert_eq!(metadata.shape.z0_source, vec![spec.nfreq, spec.nport]);
        assert_eq!(metadata.shape.z0_target, vec![spec.nfreq, spec.nport]);

        let policy = &metadata.tolerance_policy;
        assert_eq!(policy.rtol, 1e-12);
        assert_eq!(policy.atol, 1e-12);
        assert_eq!(
            policy.comparison,
            "abs(actual-expected) <= atol + rtol*abs(expected)"
        );
        assert!(!policy.justification.is_empty());
        assert!(!policy.regeneration.is_empty());

        assert_eq!(data.frequency_hz.len(), spec.nfreq);
        assert!(data.frequency_hz.iter().all(|value| value.is_finite()));
        for matrix in [&data.s_input, &data.s_renormalized] {
            assert_eq!(matrix.len(), spec.nfreq);
            for frequency_matrix in matrix {
                assert_eq!(frequency_matrix.len(), spec.nport);
                assert!(frequency_matrix.iter().all(|row| row.len() == spec.nport));
                assert!(
                    frequency_matrix
                        .iter()
                        .flatten()
                        .all(|value| { value.real.is_finite() && value.imag.is_finite() })
                );
            }
        }

        let source_z0 = matrix_z0_array(&data.z0_source_ohm, spec.nfreq, spec.nport);
        let target_z0 = matrix_z0_array(&data.z0_target_ohm, spec.nfreq, spec.nport);
        for z0 in [&source_z0, &target_z0] {
            assert_eq!(z0.dim(), (spec.nfreq, spec.nport));
            assert!(z0.iter().all(|value| {
                value.re.is_finite()
                    && value.im.is_finite()
                    && value.re > 0.0
                    && if spec.complex_z0 {
                        value.im != 0.0
                    } else {
                        value.im == 0.0
                    }
            }));
            let has_imaginary = z0.iter().any(|value| value.im != 0.0);
            assert_eq!(has_imaginary, spec.complex_z0);

            let rows_differ = (1..spec.nfreq).any(|frequency| {
                (0..spec.nport).any(|port| z0[[frequency, port]] != z0[[0, port]])
            });
            assert_eq!(rows_differ, spec.frequency_dependent_z0);

            let ports_differ = (0..spec.nfreq).any(|frequency| {
                (1..spec.nport).any(|port| z0[[frequency, port]] != z0[[frequency, 0]])
            });
            assert_eq!(ports_differ, spec.per_port_z0);
        }
        assert!(
            source_z0
                .iter()
                .zip(target_z0.iter())
                .all(|(source, target)| (*source - *target).norm() > 1.0)
        );

        if spec.nport > 1 {
            if expected_reciprocal {
                let has_complex_off_diagonal = data.s_input.iter().any(|matrix| {
                    matrix.iter().enumerate().any(|(row, row_values)| {
                        row_values
                            .iter()
                            .enumerate()
                            .skip(row + 1)
                            .any(|(_, value)| value.imag != 0.0)
                    })
                });
                assert!(
                    has_complex_off_diagonal,
                    "{expected_case_id} reciprocal input must exercise complex transpose symmetry"
                );
            }
            for (frequency, matrix) in data.s_input.iter().enumerate() {
                if expected_reciprocal {
                    for (row, row_values) in matrix.iter().enumerate().take(spec.nport) {
                        for (column, column_values) in
                            matrix.iter().enumerate().skip(row + 1).take(spec.nport)
                        {
                            assert_eq!(
                                row_values[column].real, column_values[row].real,
                                "{expected_case_id} reciprocal S input real part differs at frequency {frequency}, ports ({row}, {column})"
                            );
                            assert_eq!(
                                row_values[column].imag, column_values[row].imag,
                                "{expected_case_id} reciprocal S input imaginary part differs at frequency {frequency}, ports ({row}, {column})"
                            );
                        }
                    }
                } else {
                    assert!(
                        (0..spec.nport).any(|row| {
                            ((row + 1)..spec.nport).any(|column| {
                                matrix[row][column].real != matrix[column][row].real
                                    || matrix[row][column].imag != matrix[column][row].imag
                            })
                        }),
                        "{expected_case_id} S input is symmetric at frequency {frequency}"
                    );
                }
            }
        }

        if let Some(active_network) = active_network {
            assert_active_column_norm_lower_bound(expected_case_id, &data.s_input, active_network);
        }

        (policy.rtol, policy.atol)
    }

    fn validate_matrix_fixture_contract(
        fixture: &MatrixFixtureDocument,
        expected_case_id: &str,
        expected_operation: &str,
        spec: MatrixCaseSpec,
    ) -> (f64, f64) {
        let MatrixCaseSpec {
            nfreq: expected_nfreq,
            nport: expected_nport,
            complex_z0: expected_complex_z0,
            frequency_dependent_z0: expected_frequency_dependent_z0,
            per_port_z0: expected_per_port_z0,
            reciprocal: expected_reciprocal,
            active: expected_active,
        } = spec;
        let metadata = &fixture.metadata;
        let data = &fixture.data;
        let active_network = validate_active_network_metadata(
            metadata.active_network.as_ref(),
            expected_active,
            "s",
        );
        assert_eq!(metadata.case_id, expected_case_id);
        assert_eq!(metadata.operation, expected_operation);
        assert_eq!(metadata.wave_definition, "power");
        assert_eq!(metadata.schema, "rfkit-rs.oracle.fixture");
        assert_eq!(metadata.schema_version, 1);
        assert_eq!(metadata.numpy_version, "2.5.1");
        assert_eq!(metadata.scikit_rf_version, "2.0.1");
        assert!(metadata.random_seed > 0);
        assert_eq!(metadata.reference_impedance.complex, expected_complex_z0);
        assert_eq!(
            metadata.reference_impedance.frequency_dependent,
            expected_frequency_dependent_z0
        );
        assert_eq!(metadata.reference_impedance.per_port, expected_per_port_z0);
        assert_eq!(metadata.reference_impedance.unit, "ohm");

        assert_eq!(metadata.shape.frequency, vec![expected_nfreq]);
        assert_eq!(
            metadata.shape.input_z0,
            Some(vec![expected_nfreq, expected_nport])
        );
        match expected_operation {
            "s_to_z" => {
                assert_eq!(
                    metadata.shape.input_s,
                    Some(vec![expected_nfreq, expected_nport, expected_nport])
                );
                assert!(metadata.shape.input_z.is_none());
                assert!(metadata.shape.output_s.is_none());
                assert_eq!(
                    metadata.shape.output_z,
                    Some(vec![expected_nfreq, expected_nport, expected_nport])
                );
            }
            "z_to_s" => {
                assert!(metadata.shape.input_s.is_none());
                assert_eq!(
                    metadata.shape.input_z,
                    Some(vec![expected_nfreq, expected_nport, expected_nport])
                );
                assert_eq!(
                    metadata.shape.output_s,
                    Some(vec![expected_nfreq, expected_nport, expected_nport])
                );
                assert!(metadata.shape.output_z.is_none());
            }
            _ => panic!("unexpected power-wave operation: {expected_operation}"),
        }

        assert_eq!(data.frequency_hz.len(), expected_nfreq);
        assert!(data.frequency_hz.iter().all(|value| value.is_finite()));
        for matrix in [&data.s, &data.z_ohm] {
            assert_eq!(matrix.len(), expected_nfreq);
            for frequency_matrix in matrix.iter() {
                assert_eq!(frequency_matrix.len(), expected_nport);
                assert!(
                    frequency_matrix
                        .iter()
                        .all(|row| row.len() == expected_nport)
                );
            }
        }
        assert_eq!(data.z0_ohm.len(), expected_nfreq);
        assert!(data.z0_ohm.iter().all(|row| row.len() == expected_nport));

        let z0 = data
            .z0_ohm
            .iter()
            .map(|row| row.iter().map(matrix_complex).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let has_imaginary = z0.iter().flatten().any(|value| value.im != 0.0);
        assert_eq!(has_imaginary, expected_complex_z0);
        assert!(
            z0.iter()
                .flatten()
                .any(|value| *value != Complex64::new(50.0, 0.0)),
            "matrix fixtures must not collapse to an implicit 50-ohm assumption"
        );

        let rows_differ = z0.windows(2).any(|rows| rows[0] != rows[1]);
        assert_eq!(rows_differ, expected_frequency_dependent_z0);
        if !expected_frequency_dependent_z0 {
            assert!(z0.iter().all(|row| row == &z0[0]));
        }
        let ports_differ = z0
            .iter()
            .any(|row| row.windows(2).any(|ports| ports[0] != ports[1]));
        assert_eq!(ports_differ, expected_per_port_z0);
        if !expected_per_port_z0 {
            assert!(
                z0.iter()
                    .all(|row| { row.iter().all(|value| *value == row[0]) })
            );
        }

        let input = if expected_operation == "s_to_z" {
            &data.s
        } else {
            &data.z_ohm
        };
        if expected_nport > 1 {
            if expected_reciprocal {
                let has_complex_off_diagonal = input.iter().any(|matrix| {
                    matrix.iter().enumerate().any(|(row, row_values)| {
                        row_values
                            .iter()
                            .enumerate()
                            .skip(row + 1)
                            .any(|(_, value)| matrix_complex(value).im != 0.0)
                    })
                });
                assert!(
                    has_complex_off_diagonal,
                    "{expected_case_id} reciprocal input must exercise complex transpose symmetry"
                );
            }
            for (frequency, matrix) in input.iter().enumerate() {
                let values = matrix
                    .iter()
                    .map(|row| row.iter().map(matrix_complex).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                if expected_reciprocal {
                    for (row, row_values) in values.iter().enumerate().take(expected_nport) {
                        for (column, column_values) in
                            values.iter().enumerate().skip(row + 1).take(expected_nport)
                        {
                            assert_eq!(
                                row_values[column], column_values[row],
                                "{expected_case_id} reciprocal input differs at frequency {frequency}, ports ({row}, {column})"
                            );
                        }
                    }
                } else {
                    assert!(
                        (0..expected_nport).any(|row| {
                            ((row + 1)..expected_nport)
                                .any(|column| values[row][column] != values[column][row])
                        }),
                        "{expected_case_id} input matrix is symmetric at frequency {frequency}"
                    );
                }
            }
        }

        if let Some(active_network) = active_network {
            // S-to-Z uses data.s as its direct active input; Z-to-S uses the
            // same field for its independently generated active oracle output.
            assert_active_column_norm_lower_bound(expected_case_id, &data.s, active_network);
        }

        let policy = &metadata.tolerance_policy;
        assert_eq!(policy.rtol, 1e-12);
        assert_eq!(
            policy.comparison,
            if expected_operation == "s_to_z" {
                "abs(actual-expected) <= atol_ohm + rtol*abs(expected)"
            } else {
                "abs(actual-expected) <= atol + rtol*abs(expected)"
            }
        );
        assert!(!policy.justification.is_empty());
        assert!(!policy.regeneration.is_empty());
        match expected_operation {
            "s_to_z" => {
                assert!(policy.atol.is_none());
                assert_eq!(policy.atol_ohm, Some(1e-12));
                (policy.rtol, policy.atol_ohm.expect("S-to-Z atol_ohm"))
            }
            "z_to_s" => {
                assert_eq!(policy.atol, Some(1e-12));
                assert!(policy.atol_ohm.is_none());
                (policy.rtol, policy.atol.expect("Z-to-S atol"))
            }
            _ => unreachable!(),
        }
    }

    fn assert_matrix_output_matches(
        case_id: &str,
        actual: &Array3<Complex64>,
        expected: &[Vec<Vec<ComplexValue>>],
        rtol: f64,
        atol: f64,
    ) {
        let nfreq = expected.len();
        let nport = expected[0].len();
        assert_eq!(actual.dim(), (nfreq, nport, nport));
        for frequency in 0..nfreq {
            for row in 0..nport {
                for column in 0..nport {
                    let expected_value = matrix_complex(&expected[frequency][row][column]);
                    let actual_value = actual[[frequency, row, column]];
                    let difference = (actual_value - expected_value).norm();
                    let tolerance = atol + rtol * expected_value.norm();
                    assert!(
                        difference <= tolerance,
                        "{case_id} output[{frequency},{row},{column}] differs: actual={actual_value:?}, expected={expected_value:?}, difference={difference:e}, tolerance={tolerance:e}"
                    );
                }
            }
        }
    }

    fn assert_matrix_output_preserves_symmetry(
        case_id: &str,
        actual: &Array3<Complex64>,
        expected: &[Vec<Vec<ComplexValue>>],
        rtol: f64,
        atol: f64,
    ) {
        let nfreq = expected.len();
        let nport = expected[0].len();
        assert_eq!(actual.dim(), (nfreq, nport, nport));
        for frequency in 0..nfreq {
            for row in 0..nport {
                for column in (row + 1)..nport {
                    let expected_value = matrix_complex(&expected[frequency][row][column]);
                    let expected_transpose = matrix_complex(&expected[frequency][column][row]);
                    let tolerance =
                        atol + rtol * expected_value.norm().max(expected_transpose.norm());
                    let expected_difference = (expected_value - expected_transpose).norm();
                    assert!(
                        expected_difference <= tolerance,
                        "{case_id} oracle output[{frequency},{row},{column}] is not symmetric: difference={expected_difference:e}, tolerance={tolerance:e}"
                    );

                    let actual_value = actual[[frequency, row, column]];
                    let actual_transpose = actual[[frequency, column, row]];
                    let actual_difference = (actual_value - actual_transpose).norm();
                    assert!(
                        actual_difference <= tolerance,
                        "{case_id} computed output[{frequency},{row},{column}] is not symmetric: difference={actual_difference:e}, tolerance={tolerance:e}"
                    );
                }
            }
        }
    }

    /// Fixed, modest-magnitude three-port data for renormalization invariants.
    /// Every frequency and port has a distinct complex reference impedance so
    /// the test cannot accidentally exercise only scalar or real-z0 behavior.
    fn renormalization_test_case() -> (Array3<Complex64>, Array2<Complex64>, Array2<Complex64>) {
        let s = Array3::from_shape_vec(
            (2, 3, 3),
            vec![
                Complex64::new(0.10, 0.02),
                Complex64::new(-0.04, 0.01),
                Complex64::new(0.025, -0.03),
                Complex64::new(0.06, -0.02),
                Complex64::new(-0.08, 0.03),
                Complex64::new(0.035, 0.015),
                Complex64::new(-0.02, 0.04),
                Complex64::new(0.05, -0.01),
                Complex64::new(0.07, -0.025),
                Complex64::new(-0.06, 0.015),
                Complex64::new(0.03, -0.02),
                Complex64::new(0.045, 0.025),
                Complex64::new(0.02, 0.035),
                Complex64::new(-0.055, 0.01),
                Complex64::new(0.04, -0.015),
                Complex64::new(0.015, -0.025),
                Complex64::new(0.065, 0.02),
                Complex64::new(-0.09, 0.035),
            ],
        )
        .expect("fixed S shape must be valid");
        let source_z0 = Array2::from_shape_vec(
            (2, 3),
            vec![
                Complex64::new(50.0, 2.0),
                Complex64::new(63.0, -1.0),
                Complex64::new(71.0, 3.0),
                Complex64::new(52.0, 2.5),
                Complex64::new(66.0, -1.5),
                Complex64::new(74.0, 3.5),
            ],
        )
        .expect("fixed source z0 shape must be valid");
        let target_z0 = Array2::from_shape_vec(
            (2, 3),
            vec![
                Complex64::new(41.0, -1.5),
                Complex64::new(79.0, 2.0),
                Complex64::new(58.0, -2.5),
                Complex64::new(44.0, -1.0),
                Complex64::new(83.0, 1.5),
                Complex64::new(61.0, -2.0),
            ],
        )
        .expect("fixed target z0 shape must be valid");

        (s, source_z0, target_z0)
    }

    fn assert_array3_close(
        actual: &Array3<Complex64>,
        expected: &Array3<Complex64>,
        rtol: f64,
        atol: f64,
    ) {
        assert_eq!(actual.dim(), expected.dim());
        for frequency in 0..actual.dim().0 {
            for row in 0..actual.dim().1 {
                for column in 0..actual.dim().2 {
                    let actual_value = actual[[frequency, row, column]];
                    let expected_value = expected[[frequency, row, column]];
                    let difference = (actual_value - expected_value).norm();
                    let tolerance = atol + rtol * expected_value.norm();
                    assert!(
                        difference <= tolerance,
                        "output[{frequency},{row},{column}] differs: actual={actual_value:?}, expected={expected_value:?}, difference={difference:e}, tolerance={tolerance:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn renormalization_preserves_s_for_identity_reference_change() {
        let (source_s, source_z0, _target_z0) = renormalization_test_case();

        let actual = renormalize_s_power(&source_s, &source_z0, &source_z0)
            .expect("identity reference change must use the valid conversion path");

        // The fixed, well-conditioned matrices leave substantially more than
        // this 1e-13 mixed bound above binary64 round-off accumulated across
        // the two exact-pivot conversion stages.
        assert_array3_close(&actual, &source_s, 1e-13, 1e-13);
    }

    #[test]
    fn renormalization_round_trip_a_to_b_to_a_preserves_s() {
        let (source_s, source_z0, target_z0) = renormalization_test_case();

        let target_s = renormalize_s_power(&source_s, &source_z0, &target_z0)
            .expect("source-to-target renormalization must succeed");
        let round_trip_s = renormalize_s_power(&target_s, &target_z0, &source_z0)
            .expect("target-to-source renormalization must succeed");

        // This is a well-conditioned, modest-magnitude N-port case; 1e-13 is
        // a strict mixed tolerance that allows only accumulated binary64
        // round-off from the two composed conversions.
        assert_array3_close(&round_trip_s, &source_s, 1e-13, 1e-13);
    }

    #[test]
    fn renormalization_rejects_invalid_s_source_and_target_shapes() {
        let valid_s = Array3::from_elem((1, 2, 2), ZERO);
        let valid_z0 = Array2::from_elem((1, 2), Complex64::new(50.0, 0.0));

        let invalid_s = Array3::from_elem((1, 2, 3), ZERO);
        let error = renormalize_s_power(&invalid_s, &valid_z0, &valid_z0)
            .expect_err("non-square S must be rejected");
        assert_eq!(error, PowerWaveError::InvalidSShape { shape: (1, 2, 3) });

        let invalid_source_z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
        let error = renormalize_s_power(&valid_s, &invalid_source_z0, &valid_z0)
            .expect_err("invalid source z0 shape must be rejected");
        assert_eq!(error, PowerWaveError::InvalidZ0Shape { shape: (1, 1) });

        let invalid_target_z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
        let error = renormalize_s_power(&valid_s, &valid_z0, &invalid_target_z0)
            .expect_err("invalid target z0 shape must be rejected");
        assert_eq!(error, PowerWaveError::InvalidZ0Shape { shape: (1, 1) });
    }

    #[test]
    fn renormalization_rejects_non_finite_s_source_and_target_values() {
        let source_z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
        let target_z0 = Array2::from_elem((1, 1), Complex64::new(75.0, 0.0));

        let mut non_finite_s = Array3::from_elem((1, 1, 1), ZERO);
        non_finite_s[[0, 0, 0]] = Complex64::new(f64::NAN, 0.0);
        let error = renormalize_s_power(&non_finite_s, &source_z0, &target_z0)
            .expect_err("non-finite S must be rejected");
        assert_eq!(
            error,
            PowerWaveError::NonFiniteS {
                frequency: 0,
                row: 0,
                column: 0,
            }
        );

        let mut non_finite_source_z0 = source_z0.clone();
        non_finite_source_z0[[0, 0]] = Complex64::new(f64::INFINITY, 0.0);
        let valid_s = Array3::from_elem((1, 1, 1), ZERO);
        let error = renormalize_s_power(&valid_s, &non_finite_source_z0, &target_z0)
            .expect_err("non-finite source z0 must be rejected");
        assert_eq!(
            error,
            PowerWaveError::NonFiniteZ0 {
                frequency: 0,
                port: 0,
            }
        );

        let mut non_finite_target_z0 = target_z0.clone();
        non_finite_target_z0[[0, 0]] = Complex64::new(75.0, f64::NEG_INFINITY);
        let error = renormalize_s_power(&valid_s, &source_z0, &non_finite_target_z0)
            .expect_err("non-finite target z0 must be rejected");
        assert_eq!(
            error,
            PowerWaveError::NonFiniteZ0 {
                frequency: 0,
                port: 0,
            }
        );
    }

    #[test]
    fn renormalization_rejects_zero_real_source_and_target_z0() {
        let valid_s = Array3::from_elem((1, 1, 1), ZERO);
        let valid_z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
        let zero_real_source = Array2::from_elem((1, 1), Complex64::new(0.0, 50.0));
        let error = renormalize_s_power(&valid_s, &zero_real_source, &valid_z0)
            .expect_err("zero-real source z0 must be rejected");
        assert_eq!(
            error,
            PowerWaveError::ZeroRealReferenceImpedance {
                frequency: 0,
                port: 0,
            }
        );

        let zero_real_target = Array2::from_elem((1, 1), Complex64::new(0.0, -50.0));
        let error = renormalize_s_power(&valid_s, &valid_z0, &zero_real_target)
            .expect_err("zero-real target z0 must be rejected");
        assert_eq!(
            error,
            PowerWaveError::ZeroRealReferenceImpedance {
                frequency: 0,
                port: 0,
            }
        );
    }

    #[test]
    fn renormalization_propagates_exact_source_stage_singularity() {
        let source_s = Array3::from_elem((1, 1, 1), Complex64::new(1.0, 0.0));
        let source_z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
        let target_z0 = Array2::from_elem((1, 1), Complex64::new(75.0, 0.0));

        let error = renormalize_s_power(&source_s, &source_z0, &target_z0)
            .expect_err("I-S must remain exactly singular in the source stage");
        assert_eq!(
            error,
            PowerWaveError::Singular {
                frequency: 0,
                pivot: 0,
            }
        );
    }

    #[test]
    fn renormalization_propagates_exact_target_stage_singularity() {
        // For real source z0=50 and S=-3, Eq. (19) gives Z=-25 exactly.
        // Target z0=25 then makes Z+G exactly zero in Eq. (18).
        let source_s = Array3::from_elem((1, 1, 1), Complex64::new(-3.0, 0.0));
        let source_z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
        let target_z0 = Array2::from_elem((1, 1), Complex64::new(25.0, 0.0));

        let error = renormalize_s_power(&source_s, &source_z0, &target_z0)
            .expect_err("Z+G must remain exactly singular in the target stage");
        assert_eq!(
            error,
            PowerWaveError::Singular {
                frequency: 0,
                pivot: 0,
            }
        );
    }

    #[test]
    fn one_port_zero_s_has_conjugate_reference_impedance() {
        let s = Array3::from_elem((1, 1, 1), ZERO);
        let z0_value = Complex64::new(50.0, 12.5);
        let z0 = Array2::from_elem((1, 1), z0_value);

        let z = s_to_z_power(&s, &z0).expect("well-formed one-port conversion must succeed");
        let actual = z[[0, 0, 0]];
        let expected = z0_value.conj();
        assert!(
            (actual - expected).norm() <= 1e-12,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn one_port_conjugate_z0_has_zero_s() {
        let z0_value = Complex64::new(50.0, 12.5);
        let z = Array3::from_elem((1, 1, 1), z0_value.conj());
        let z0 = Array2::from_elem((1, 1), z0_value);

        let s = z_to_s_power(&z, &z0).expect("well-formed one-port conversion must succeed");
        let actual = s[[0, 0, 0]];
        assert!(actual.norm() <= 1e-13, "expected zero, got {actual:?}");
    }

    #[test]
    fn exact_identity_s_is_reported_as_singular() {
        let mut identity = Array3::from_elem((1, 2, 2), ZERO);
        identity[[0, 0, 0]] = Complex64::new(1.0, 0.0);
        identity[[0, 1, 1]] = Complex64::new(1.0, 0.0);
        let z0 = Array2::from_elem((1, 2), Complex64::new(50.0, 0.0));

        let error = s_to_z_power(&identity, &z0).expect_err("I-S must be exactly singular");
        assert_eq!(
            error,
            PowerWaveError::Singular {
                frequency: 0,
                pivot: 0,
            }
        );
    }

    #[test]
    fn exact_negative_reference_impedance_z_is_reported_as_singular() {
        let z0_value = Complex64::new(50.0, 12.5);
        let z = Array3::from_elem((1, 1, 1), -z0_value);
        let z0 = Array2::from_elem((1, 1), z0_value);

        let error = z_to_s_power(&z, &z0).expect_err("Z+G must be exactly singular");
        assert_eq!(
            error,
            PowerWaveError::Singular {
                frequency: 0,
                pivot: 0,
            }
        );
    }

    #[test]
    fn rejects_non_square_z_matrices() {
        let z = Array3::zeros((1, 2, 3));
        let z0 = Array2::from_elem((1, 2), Complex64::new(50.0, 0.0));

        let error = z_to_s_power(&z, &z0).expect_err("non-square Z must be rejected");
        assert_eq!(error, PowerWaveError::InvalidZShape { shape: (1, 2, 3) });
    }

    #[test]
    fn rejects_non_finite_z_values() {
        let mut z = Array3::from_elem((1, 1, 1), Complex64::new(50.0, 0.0));
        z[[0, 0, 0]] = Complex64::new(f64::NAN, 0.0);
        let z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));

        let error = z_to_s_power(&z, &z0).expect_err("non-finite Z must be rejected");
        assert_eq!(
            error,
            PowerWaveError::NonFiniteZ {
                frequency: 0,
                row: 0,
                column: 0,
            }
        );
    }

    #[test]
    fn zero_real_reference_impedance_is_an_explicit_error() {
        let s = Array3::from_elem((1, 1, 1), ZERO);
        let z0 = Array2::from_elem((1, 1), Complex64::new(0.0, 50.0));

        let error = s_to_z_power(&s, &z0).expect_err("purely imaginary z0 is undefined here");
        assert_eq!(
            error,
            PowerWaveError::ZeroRealReferenceImpedance {
                frequency: 0,
                port: 0,
            }
        );
    }

    #[test]
    fn reconstructs_s_from_z_with_kurokawa_eq18() {
        let mut source_s = Array3::from_elem((2, 2, 2), ZERO);
        source_s[[0, 0, 0]] = Complex64::new(0.12, 0.03);
        source_s[[0, 0, 1]] = Complex64::new(-0.04, 0.02);
        source_s[[0, 1, 0]] = Complex64::new(0.06, -0.05);
        source_s[[0, 1, 1]] = Complex64::new(-0.15, 0.04);
        source_s[[1, 0, 0]] = Complex64::new(-0.08, 0.06);
        source_s[[1, 0, 1]] = Complex64::new(0.03, 0.01);
        source_s[[1, 1, 0]] = Complex64::new(0.09, -0.02);
        source_s[[1, 1, 1]] = Complex64::new(0.11, 0.07);

        let z0 = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(50.0, 2.0),
                Complex64::new(63.0, -1.5),
                Complex64::new(54.0, 2.75),
                Complex64::new(68.5, -0.75),
            ],
        )
        .expect("fixed two-frequency z0 shape must be valid");

        // The test data must exercise complex, per-port, frequency-dependent
        // reference impedances rather than accidentally reducing to scalar z0.
        assert!(z0.iter().any(|value| value.im != 0.0));
        assert_ne!(z0[[0, 0]], z0[[0, 1]]);
        assert_ne!(z0[[1, 0]], z0[[1, 1]]);
        assert_ne!(z0[[0, 0]], z0[[1, 0]]);
        assert_ne!(z0[[0, 1]], z0[[1, 1]]);

        let z = s_to_z_power(&source_s, &z0).expect("well-conditioned round-trip must succeed");
        let direct_reconstructed_s =
            z_to_s_power(&z, &z0).expect("well-conditioned inverse round-trip must succeed");
        assert_eq!(direct_reconstructed_s.dim(), source_s.dim());

        // The matrices are modest in magnitude and well-conditioned.  A
        // 1e-13 mixed bound is comfortably above accumulated binary64
        // round-off for this two-by-two closed-form check while remaining
        // strict relative to the O(1) S-parameter values.
        const RTOL: f64 = 1e-13;
        const ATOL: f64 = 1e-13;
        for frequency in 0..2 {
            let reconstructed = reconstruct_s_from_z_eq18(&z, &z0, frequency);
            for row in 0..2 {
                for column in 0..2 {
                    let expected = source_s[[frequency, row, column]];
                    let actual = reconstructed[row][column];
                    let difference = (actual - expected).norm();
                    let tolerance = ATOL + RTOL * expected.norm();
                    assert!(
                        difference <= tolerance,
                        "S[{frequency},{row},{column}] differs: actual={actual:?}, expected={expected:?}, difference={difference:e}, tolerance={tolerance:e}"
                    );
                }
            }

            for row in 0..2 {
                for column in 0..2 {
                    let expected = source_s[[frequency, row, column]];
                    let actual = direct_reconstructed_s[[frequency, row, column]];
                    let difference = (actual - expected).norm();
                    let tolerance = ATOL + RTOL * expected.norm();
                    assert!(
                        difference <= tolerance,
                        "direct S[{frequency},{row},{column}] differs: actual={actual:?}, expected={expected:?}, difference={difference:e}, tolerance={tolerance:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn matches_power_wave_three_port_complex_z0_fixture() {
        let fixture: FixtureDocument =
            serde_json::from_str(FIXTURE_JSON).expect("checked-in fixture must parse");

        assert_eq!(
            fixture.metadata.case_id,
            "power_wave_s_to_z_three_port_complex_z0"
        );
        assert_eq!(fixture.metadata.input_case_id, "three_port_complex_z0");
        assert_eq!(fixture.metadata.operation, "s_to_z");
        assert_eq!(fixture.metadata.wave_definition, "power");
        assert_eq!(fixture.metadata.schema, "rfkit-rs.oracle.fixture");
        assert_eq!(fixture.metadata.schema_version, 1);
        assert_eq!(fixture.metadata.numpy_version, "2.5.1");
        assert_eq!(fixture.metadata.scikit_rf_version, "2.0.1");
        assert_eq!(fixture.metadata.random_seed, 20_250_308);

        assert_eq!(fixture.metadata.shape.frequency, vec![4]);
        assert_eq!(fixture.metadata.shape.input_s, vec![4, 3, 3]);
        assert_eq!(fixture.metadata.shape.input_z0, vec![4, 3]);
        assert_eq!(fixture.metadata.shape.output_z, vec![4, 3, 3]);
        assert!(fixture.metadata.reference_impedance.complex);
        assert!(fixture.metadata.reference_impedance.frequency_dependent);
        assert!(fixture.metadata.reference_impedance.per_port);
        assert_eq!(fixture.metadata.reference_impedance.unit, "ohm");

        assert_eq!(fixture.metadata.tolerance_policy.rtol, 1e-12);
        assert_eq!(fixture.metadata.tolerance_policy.atol_ohm, 1e-12);
        assert_eq!(
            fixture.metadata.tolerance_policy.comparison,
            "abs(actual-expected) <= atol_ohm + rtol*abs(expected)"
        );
        assert!(!fixture.metadata.tolerance_policy.justification.is_empty());
        assert!(!fixture.metadata.tolerance_policy.regeneration.is_empty());

        // The shape contract is frequency-major: the first S and z0 axis is
        // frequency, followed by output/input port axes.
        assert_eq!(fixture.data.frequency_hz.len(), 4);
        assert_eq!(fixture.data.s.len(), 4);
        assert!(fixture.data.s.iter().all(|matrix| matrix.len() == 3));
        assert!(
            fixture
                .data
                .s
                .iter()
                .all(|matrix| matrix.iter().all(|row| row.len() == 3))
        );
        assert_eq!(fixture.data.z0_ohm.len(), 4);
        assert!(fixture.data.z0_ohm.iter().all(|row| row.len() == 3));
        assert_eq!(fixture.data.z_ohm.len(), 4);
        assert!(fixture.data.z_ohm.iter().all(|matrix| matrix.len() == 3));
        assert!(
            fixture
                .data
                .z_ohm
                .iter()
                .all(|matrix| matrix.iter().all(|row| row.len() == 3))
        );

        // Check that the data itself exercises complex, per-port,
        // frequency-dependent reference impedances rather than relying only
        // on the metadata flags.
        assert!(
            fixture
                .data
                .z0_ohm
                .iter()
                .flatten()
                .any(|value| value.imag != 0.0)
        );
        assert!(fixture.data.z0_ohm[0][0].real != fixture.data.z0_ohm[0][1].real);
        assert!(fixture.data.z0_ohm[0][1].real != fixture.data.z0_ohm[0][2].real);
        assert!(fixture.data.z0_ohm[0][0].real != fixture.data.z0_ohm[1][0].real);

        let s = Array3::from_shape_fn((4, 3, 3), |(frequency, row, column)| {
            let value = &fixture.data.s[frequency][row][column];
            Complex64::new(value.real, value.imag)
        });
        let z0 = Array2::from_shape_fn((4, 3), |(frequency, port)| {
            let value = &fixture.data.z0_ohm[frequency][port];
            Complex64::new(value.real, value.imag)
        });

        let actual = s_to_z_power(&s, &z0).expect("fixture conversion must succeed");
        assert_eq!(actual.dim(), (4, 3, 3));

        let rtol = fixture.metadata.tolerance_policy.rtol;
        let atol_ohm = fixture.metadata.tolerance_policy.atol_ohm;
        for frequency in 0..4 {
            for row in 0..3 {
                for column in 0..3 {
                    let expected_value = &fixture.data.z_ohm[frequency][row][column];
                    let expected = Complex64::new(expected_value.real, expected_value.imag);
                    let difference = (actual[[frequency, row, column]] - expected).norm();
                    let tolerance = atol_ohm + rtol * expected.norm();
                    assert!(
                        difference <= tolerance,
                        "Z[{frequency},{row},{column}] differs: actual={:?}, expected={expected:?}, difference={difference:e}, tolerance={tolerance:e}",
                        actual[[frequency, row, column]]
                    );
                }
            }
        }
    }

    #[test]
    fn matches_power_wave_z_to_s_three_port_complex_z0_fixture() {
        let fixture: ZToSFixtureDocument =
            serde_json::from_str(Z_TO_S_FIXTURE_JSON).expect("checked-in fixture must parse");

        assert_eq!(
            fixture.metadata.case_id,
            "power_wave_z_to_s_three_port_complex_z0"
        );
        assert_eq!(fixture.metadata.operation, "z_to_s");
        assert_eq!(fixture.metadata.wave_definition, "power");
        assert_eq!(fixture.metadata.schema, "rfkit-rs.oracle.fixture");
        assert_eq!(fixture.metadata.schema_version, 1);
        assert_eq!(fixture.metadata.numpy_version, "2.5.1");
        assert_eq!(fixture.metadata.scikit_rf_version, "2.0.1");
        assert_eq!(fixture.metadata.random_seed, 20_260_826);

        assert_eq!(fixture.metadata.shape.frequency, vec![4]);
        assert_eq!(fixture.metadata.shape.input_z, vec![4, 3, 3]);
        assert_eq!(fixture.metadata.shape.input_z0, vec![4, 3]);
        assert_eq!(fixture.metadata.shape.output_s, vec![4, 3, 3]);
        assert!(fixture.metadata.reference_impedance.complex);
        assert!(fixture.metadata.reference_impedance.frequency_dependent);
        assert!(fixture.metadata.reference_impedance.per_port);
        assert_eq!(fixture.metadata.reference_impedance.unit, "ohm");

        assert_eq!(fixture.metadata.tolerance_policy.rtol, 1e-12);
        assert_eq!(fixture.metadata.tolerance_policy.atol, 1e-12);
        assert_eq!(
            fixture.metadata.tolerance_policy.comparison,
            "abs(actual-expected) <= atol + rtol*abs(expected)"
        );
        assert!(!fixture.metadata.tolerance_policy.justification.is_empty());
        assert!(!fixture.metadata.tolerance_policy.regeneration.is_empty());

        // The shape contract is frequency-major: the first Z and z0 axis is
        // frequency, followed by output/input port axes.
        assert_eq!(fixture.data.frequency_hz.len(), 4);
        assert_eq!(fixture.data.z_ohm.len(), 4);
        assert!(fixture.data.z_ohm.iter().all(|matrix| matrix.len() == 3));
        assert!(
            fixture
                .data
                .z_ohm
                .iter()
                .all(|matrix| matrix.iter().all(|row| row.len() == 3))
        );
        assert_eq!(fixture.data.z0_ohm.len(), 4);
        assert!(fixture.data.z0_ohm.iter().all(|row| row.len() == 3));
        assert_eq!(fixture.data.s.len(), 4);
        assert!(fixture.data.s.iter().all(|matrix| matrix.len() == 3));
        assert!(
            fixture
                .data
                .s
                .iter()
                .all(|matrix| matrix.iter().all(|row| row.len() == 3))
        );

        // Check that the data itself exercises complex, per-port,
        // frequency-dependent reference impedances rather than relying only
        // on the metadata flags.
        assert!(
            fixture
                .data
                .z0_ohm
                .iter()
                .flatten()
                .any(|value| value.imag != 0.0)
        );
        assert!(fixture.data.z0_ohm[0][0].real != fixture.data.z0_ohm[0][1].real);
        assert!(fixture.data.z0_ohm[0][1].real != fixture.data.z0_ohm[0][2].real);
        assert!(fixture.data.z0_ohm[0][0].real != fixture.data.z0_ohm[1][0].real);

        let z = Array3::from_shape_fn((4, 3, 3), |(frequency, row, column)| {
            let value = &fixture.data.z_ohm[frequency][row][column];
            Complex64::new(value.real, value.imag)
        });
        let z0 = Array2::from_shape_fn((4, 3), |(frequency, port)| {
            let value = &fixture.data.z0_ohm[frequency][port];
            Complex64::new(value.real, value.imag)
        });

        let actual = z_to_s_power(&z, &z0).expect("fixture conversion must succeed");
        assert_eq!(actual.dim(), (4, 3, 3));

        let rtol = fixture.metadata.tolerance_policy.rtol;
        let atol = fixture.metadata.tolerance_policy.atol;
        for frequency in 0..4 {
            for row in 0..3 {
                for column in 0..3 {
                    let expected_value = &fixture.data.s[frequency][row][column];
                    let expected = Complex64::new(expected_value.real, expected_value.imag);
                    let difference = (actual[[frequency, row, column]] - expected).norm();
                    let tolerance = atol + rtol * expected.norm();
                    assert!(
                        difference <= tolerance,
                        "S[{frequency},{row},{column}] differs: actual={:?}, expected={expected:?}, difference={difference:e}, tolerance={tolerance:e}",
                        actual[[frequency, row, column]]
                    );
                }
            }
        }
    }

    #[test]
    fn matches_power_wave_s_to_z_conformance_matrix() {
        for &(case_id, json) in MATRIX_S_TO_Z_FIXTURES {
            let spec = matrix_case_spec(case_id);
            let fixture: MatrixFixtureDocument =
                serde_json::from_str(json).expect("checked-in matrix fixture must parse");
            let (rtol, atol_ohm) =
                validate_matrix_fixture_contract(&fixture, case_id, "s_to_z", spec);
            let s = matrix_parameter_array(&fixture.data.s, spec.nfreq, spec.nport);
            let z0 = matrix_z0_array(&fixture.data.z0_ohm, spec.nfreq, spec.nport);
            let actual = s_to_z_power(&s, &z0).expect("matrix S-to-Z conversion must succeed");
            assert_matrix_output_matches(case_id, &actual, &fixture.data.z_ohm, rtol, atol_ohm);
            if spec.reciprocal {
                assert_matrix_output_preserves_symmetry(
                    case_id,
                    &actual,
                    &fixture.data.z_ohm,
                    rtol,
                    atol_ohm,
                );
            }
        }
    }

    #[test]
    fn matches_power_wave_z_to_s_conformance_matrix() {
        for &(case_id, json) in MATRIX_Z_TO_S_FIXTURES {
            let spec = matrix_case_spec(case_id);
            let fixture: MatrixFixtureDocument =
                serde_json::from_str(json).expect("checked-in matrix fixture must parse");
            let (rtol, atol) = validate_matrix_fixture_contract(&fixture, case_id, "z_to_s", spec);
            let z = matrix_parameter_array(&fixture.data.z_ohm, spec.nfreq, spec.nport);
            let z0 = matrix_z0_array(&fixture.data.z0_ohm, spec.nfreq, spec.nport);
            let actual = z_to_s_power(&z, &z0).expect("matrix Z-to-S conversion must succeed");
            assert_matrix_output_matches(case_id, &actual, &fixture.data.s, rtol, atol);
            if spec.reciprocal {
                assert_matrix_output_preserves_symmetry(
                    case_id,
                    &actual,
                    &fixture.data.s,
                    rtol,
                    atol,
                );
            }
        }
    }

    #[test]
    fn matches_power_wave_renormalization_conformance_matrix() {
        for &(case_id, json) in RENORMALIZE_FIXTURES {
            let spec = renormalization_case_spec(case_id);
            let fixture: RenormalizationFixtureDocument =
                serde_json::from_str(json).expect("checked-in renormalization fixture must parse");
            let (rtol, atol) = validate_renormalization_fixture_contract(&fixture, case_id, spec);
            let source_s = matrix_parameter_array(&fixture.data.s_input, spec.nfreq, spec.nport);
            let source_z0 = matrix_z0_array(&fixture.data.z0_source_ohm, spec.nfreq, spec.nport);
            let target_z0 = matrix_z0_array(&fixture.data.z0_target_ohm, spec.nfreq, spec.nport);
            let actual = renormalize_s_power(&source_s, &source_z0, &target_z0)
                .expect("renormalization fixture conversion must succeed");
            assert_matrix_output_matches(
                case_id,
                &actual,
                &fixture.data.s_renormalized,
                rtol,
                atol,
            );
            if spec.reciprocal {
                assert_matrix_output_preserves_symmetry(
                    case_id,
                    &actual,
                    &fixture.data.s_renormalized,
                    rtol,
                    atol,
                );
            }
        }
    }

    #[test]
    fn matches_power_wave_active_s_to_z_conformance_fixture() {
        for &(case_id, json) in ACTIVE_S_TO_Z_FIXTURES {
            let spec = matrix_case_spec(case_id);
            let fixture: MatrixFixtureDocument =
                serde_json::from_str(json).expect("checked-in active fixture must parse");
            let (rtol, atol_ohm) =
                validate_matrix_fixture_contract(&fixture, case_id, "s_to_z", spec);
            let s = matrix_parameter_array(&fixture.data.s, spec.nfreq, spec.nport);
            let z0 = matrix_z0_array(&fixture.data.z0_ohm, spec.nfreq, spec.nport);
            let actual = s_to_z_power(&s, &z0).expect("active S-to-Z conversion must succeed");
            assert_matrix_output_matches(case_id, &actual, &fixture.data.z_ohm, rtol, atol_ohm);
        }
    }

    #[test]
    fn matches_power_wave_active_z_to_s_conformance_fixture() {
        for &(case_id, json) in ACTIVE_Z_TO_S_FIXTURES {
            let spec = matrix_case_spec(case_id);
            let fixture: MatrixFixtureDocument =
                serde_json::from_str(json).expect("checked-in active fixture must parse");
            let (rtol, atol) = validate_matrix_fixture_contract(&fixture, case_id, "z_to_s", spec);
            let z = matrix_parameter_array(&fixture.data.z_ohm, spec.nfreq, spec.nport);
            let z0 = matrix_z0_array(&fixture.data.z0_ohm, spec.nfreq, spec.nport);
            let actual = z_to_s_power(&z, &z0).expect("active Z-to-S conversion must succeed");
            assert_matrix_output_matches(case_id, &actual, &fixture.data.s, rtol, atol);
        }
    }

    #[test]
    fn matches_power_wave_active_renormalization_conformance_fixture() {
        for &(case_id, json) in ACTIVE_RENORMALIZE_FIXTURES {
            let spec = renormalization_case_spec(case_id);
            let fixture: RenormalizationFixtureDocument =
                serde_json::from_str(json).expect("checked-in active fixture must parse");
            let (rtol, atol) = validate_renormalization_fixture_contract(&fixture, case_id, spec);
            let source_s = matrix_parameter_array(&fixture.data.s_input, spec.nfreq, spec.nport);
            let source_z0 = matrix_z0_array(&fixture.data.z0_source_ohm, spec.nfreq, spec.nport);
            let target_z0 = matrix_z0_array(&fixture.data.z0_target_ohm, spec.nfreq, spec.nport);
            let actual = renormalize_s_power(&source_s, &source_z0, &target_z0)
                .expect("active renormalization conversion must succeed");
            assert_matrix_output_matches(
                case_id,
                &actual,
                &fixture.data.s_renormalized,
                rtol,
                atol,
            );
        }
    }
}
