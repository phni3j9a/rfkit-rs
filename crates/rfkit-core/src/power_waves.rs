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

    #[error("reference-impedance shape must be (nfreq, nport), got {shape:?}")]
    InvalidZ0Shape { shape: (usize, usize) },

    #[error("reference impedance has zero real part at frequency {frequency}, port {port}")]
    ZeroRealReferenceImpedance { frequency: usize, port: usize },

    #[error("S-to-Z system is exactly singular at frequency {frequency}, pivot {pivot}")]
    Singular { frequency: usize, pivot: usize },

    #[error("non-finite S-parameter at frequency {frequency}, row {row}, column {column}")]
    NonFiniteS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("non-finite reference impedance at frequency {frequency}, port {port}")]
    NonFiniteZ0 { frequency: usize, port: usize },

    #[error(
        "non-finite value while solving S-to-Z system at frequency {frequency}, row {row}, column {column}"
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
}
