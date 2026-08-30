//! Internal frequency-major impedance/admittance conversions.
//!
//! The impedance and admittance matrices are related independently at each
//! frequency by
//!
//! ```text
//! Y = Z^-1,    Z = Y^-1.
//! ```
//!
//! Both directions use the shared N-port Gaussian solver to solve `A X = I`.
//! There is deliberately no two-port shortcut, matrix-rank cutoff,
//! regularization, or tolerance-based runtime classification.  A matrix is
//! singular only when the selected pivot is exactly the complex zero value.

use ndarray::Array3;
use num_complex::Complex64;
use thiserror::Error;

use crate::linalg;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// Failure modes for the internal impedance/admittance conversion kernels.
#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum ImpedanceAdmittanceError {
    #[error("Z-parameter shape must be (nfreq, nport, nport), got {shape:?}")]
    InvalidZShape { shape: (usize, usize, usize) },

    #[error("Y-parameter shape must be (nfreq, nport, nport), got {shape:?}")]
    InvalidYShape { shape: (usize, usize, usize) },

    #[error("non-finite Z-parameter at frequency {frequency}, row {row}, column {column}")]
    NonFiniteZ {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("non-finite Y-parameter at frequency {frequency}, row {row}, column {column}")]
    NonFiniteY {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "non-finite value while solving impedance/admittance conversion system at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteComputation {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "impedance/admittance conversion system is exactly singular at frequency {frequency}, pivot {pivot}"
    )]
    Singular { frequency: usize, pivot: usize },
}

#[derive(Clone, Copy)]
enum Quantity {
    Impedance,
    Admittance,
}

impl Quantity {
    fn invalid_shape(self, shape: (usize, usize, usize)) -> ImpedanceAdmittanceError {
        match self {
            Self::Impedance => ImpedanceAdmittanceError::InvalidZShape { shape },
            Self::Admittance => ImpedanceAdmittanceError::InvalidYShape { shape },
        }
    }

    fn non_finite(self, frequency: usize, row: usize, column: usize) -> ImpedanceAdmittanceError {
        match self {
            Self::Impedance => ImpedanceAdmittanceError::NonFiniteZ {
                frequency,
                row,
                column,
            },
            Self::Admittance => ImpedanceAdmittanceError::NonFiniteY {
                frequency,
                row,
                column,
            },
        }
    }
}

/// Convert frequency-major impedance matrices to admittance matrices.
///
/// For each frequency slice this copies `Z` into a row-major linear system and
/// solves `Z Y = I`.  The returned array has the same frequency and port
/// dimensions as the input and is expressed in siemens (`S`).
#[allow(dead_code)] // Internal kernel is staged for a future Network call site.
pub(crate) fn z_to_y(z: &Array3<Complex64>) -> Result<Array3<Complex64>, ImpedanceAdmittanceError> {
    invert_frequency_major(z, Quantity::Impedance)
}

/// Convert frequency-major admittance matrices to impedance matrices.
///
/// For each frequency slice this copies `Y` into a row-major linear system and
/// solves `Y Z = I`.  The returned array has the same frequency and port
/// dimensions as the input and is expressed in ohms (`ohm`).
#[allow(dead_code)] // Internal kernel is staged for a future Network call site.
pub(crate) fn y_to_z(y: &Array3<Complex64>) -> Result<Array3<Complex64>, ImpedanceAdmittanceError> {
    invert_frequency_major(y, Quantity::Admittance)
}

/// Invert each frequency slice of a frequency-major N-port stack.
fn invert_frequency_major(
    input: &Array3<Complex64>,
    quantity: Quantity,
) -> Result<Array3<Complex64>, ImpedanceAdmittanceError> {
    let shape = input.dim();
    let (nfreq, nport_rows, nport_columns) = shape;
    if nport_rows == 0 || nport_rows != nport_columns {
        return Err(quantity.invalid_shape(shape));
    }

    let nport = nport_rows;
    let mut output = Array3::from_elem(shape, ZERO);
    for frequency in 0..nfreq {
        let mut a = vec![ZERO; nport * nport];
        let mut b = vec![ZERO; nport * nport];
        for row in 0..nport {
            for column in 0..nport {
                let value = input[[frequency, row, column]];
                if !linalg::is_finite(value) {
                    return Err(quantity.non_finite(frequency, row, column));
                }

                let index = row * nport + column;
                a[index] = value;
                b[index] = if row == column { ONE } else { ZERO };
            }
        }

        linalg::solve_multiple_rhs(&mut a, &mut b, nport).map_err(|error| match error {
            linalg::SolveError::InvalidStorage { .. } => {
                unreachable!("impedance/admittance solver storage is square")
            }
            linalg::SolveError::Singular { pivot } => {
                ImpedanceAdmittanceError::Singular { frequency, pivot }
            }
            linalg::SolveError::NonFinite { row, column } => {
                ImpedanceAdmittanceError::NonFiniteComputation {
                    frequency,
                    row,
                    column,
                }
            }
        })?;

        for row in 0..nport {
            for column in 0..nport {
                let value = b[row * nport + column];
                if !linalg::is_finite(value) {
                    return Err(ImpedanceAdmittanceError::NonFiniteComputation {
                        frequency,
                        row,
                        column,
                    });
                }
                output[[frequency, row, column]] = value;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    const Z_TO_Y_FIXTURE: &str = include_str!(
        "../../../tools/oracle/fixtures/impedance_admittance_z_to_y_three_port_well_conditioned.json"
    );
    const Y_TO_Z_FIXTURE: &str = include_str!(
        "../../../tools/oracle/fixtures/impedance_admittance_y_to_z_three_port_well_conditioned.json"
    );
    const Z_TO_Y_NEAR_FIXTURE: &str = include_str!(
        "../../../tools/oracle/fixtures/impedance_admittance_z_to_y_three_port_near_singular.json"
    );
    const Y_TO_Z_NEAR_FIXTURE: &str = include_str!(
        "../../../tools/oracle/fixtures/impedance_admittance_y_to_z_three_port_near_singular.json"
    );

    const IMPEDANCE_ADMITTANCE_FIXTURES: &[(&str, &str, &str, &str, &str)] = &[
        (
            "impedance_admittance_z_to_y_three_port_well_conditioned",
            "z_to_y",
            "ohm",
            "S",
            Z_TO_Y_FIXTURE,
        ),
        (
            "impedance_admittance_y_to_z_three_port_well_conditioned",
            "y_to_z",
            "S",
            "ohm",
            Y_TO_Z_FIXTURE,
        ),
        (
            "impedance_admittance_z_to_y_three_port_near_singular",
            "z_to_y",
            "ohm",
            "S",
            Z_TO_Y_NEAR_FIXTURE,
        ),
        (
            "impedance_admittance_y_to_z_three_port_near_singular",
            "y_to_z",
            "S",
            "ohm",
            Y_TO_Z_NEAR_FIXTURE,
        ),
    ];

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
        y_s: Vec<Vec<Vec<ComplexValue>>>,
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
        #[serde(default)]
        near_singular: Option<NearSingularMetadata>,
        case_id: String,
        input_unit: String,
        numpy_version: String,
        operation: String,
        output_unit: String,
        random_seed: u64,
        schema: String,
        schema_version: u32,
        scikit_rf_version: String,
        shape: FixtureShape,
        tolerance_policy: TolerancePolicy,
        wave_definition: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FixtureShape {
        frequency: Vec<usize>,
        #[serde(default)]
        input_y: Option<Vec<usize>>,
        #[serde(default)]
        input_z: Option<Vec<usize>>,
        #[serde(default)]
        output_y: Option<Vec<usize>>,
        #[serde(default)]
        output_z: Option<Vec<usize>>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct TolerancePolicy {
        #[serde(default)]
        atol_ohm: Option<f64>,
        #[serde(default)]
        atol_s: Option<f64>,
        comparison: String,
        justification: String,
        regeneration: String,
        rtol: f64,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct NearSingularMetadata {
        binary_exponent: i32,
        determinant: f64,
        determinant_factors: Vec<f64>,
        determinant_factors_nonzero: bool,
        matrix_structure: String,
        small_diagonal_port: usize,
        small_diagonal_value: f64,
        system_matrix: String,
    }

    fn complex(real: f64, imag: f64) -> Complex64 {
        Complex64::new(real, imag)
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
        let difference = (actual - expected).norm();
        assert!(
            difference <= tolerance,
            "actual {actual:?} differs from expected {expected:?} by {difference:?}"
        );
    }

    fn assert_array3_close(
        actual: &Array3<Complex64>,
        expected: &Array3<Complex64>,
        tolerance: f64,
    ) {
        assert_eq!(actual.dim(), expected.dim());
        for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
            assert_complex_close(actual_value, expected_value, tolerance);
            assert!(
                linalg::is_finite(actual_value),
                "actual value at flattened index {index} is non-finite"
            );
        }
    }

    fn arbitrary_three_port_stack() -> Array3<Complex64> {
        Array3::from_shape_fn((2, 3, 3), |(frequency, row, column)| {
            if row == column {
                complex(
                    8.0 + frequency as f64 + row as f64,
                    0.75 + 0.1 * frequency as f64 - 0.05 * row as f64,
                )
            } else {
                complex(
                    0.2 * (row + 1) as f64 * (column + 1) as f64,
                    0.07 * (row as f64 - column as f64) + 0.03 * frequency as f64,
                )
            }
        })
    }

    fn fixture_array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
        assert!(!values.is_empty(), "fixture must contain frequencies");
        let nfreq = values.len();
        let nport = values[0].len();
        assert!(nport > 0, "fixture must contain ports");
        assert!(values.iter().all(|matrix| matrix.len() == nport));
        assert!(
            values
                .iter()
                .all(|matrix| matrix.iter().all(|row| row.len() == nport))
        );

        let flattened = values
            .iter()
            .flat_map(|matrix| matrix.iter())
            .flat_map(|row| row.iter())
            .map(|value| complex(value.real, value.imag))
            .collect();
        Array3::from_shape_vec((nfreq, nport, nport), flattened)
            .expect("fixture dimensions must agree")
    }

    fn assert_recorded_output(
        actual: &Array3<Complex64>,
        expected: &Array3<Complex64>,
        rtol: f64,
        atol: f64,
    ) {
        assert_eq!(actual.dim(), expected.dim());
        for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
            assert!(
                linalg::is_finite(actual_value),
                "actual output at flattened index {index} is non-finite"
            );
            assert!(
                linalg::is_finite(expected_value),
                "recorded output at flattened index {index} is non-finite"
            );
            let difference = (actual_value - expected_value).norm();
            let bound = atol + rtol * expected_value.norm();
            assert!(
                difference <= bound,
                "output at flattened index {index} differs by {difference:?}, bound {bound:?}; actual {actual_value:?}, expected {expected_value:?}"
            );
        }
    }

    #[test]
    fn matches_all_direct_impedance_admittance_fixtures_with_recorded_tolerance() {
        for &(case_id, operation, input_unit, output_unit, fixture_json) in
            IMPEDANCE_ADMITTANCE_FIXTURES
        {
            let fixture: FixtureDocument =
                serde_json::from_str(fixture_json).expect("fixture JSON must deserialize");
            let metadata = &fixture.metadata;
            let data = &fixture.data;
            assert_eq!(metadata.case_id, case_id);
            assert_eq!(metadata.operation, operation);
            assert_eq!(metadata.input_unit, input_unit);
            assert_eq!(metadata.output_unit, output_unit);
            assert_eq!(metadata.numpy_version, "2.5.1");
            assert_eq!(metadata.scikit_rf_version, "2.0.1");
            assert_eq!(metadata.schema, "rfkit-rs.oracle.fixture");
            assert_eq!(metadata.schema_version, 1);
            assert_eq!(metadata.wave_definition, "not_applicable");
            let expected_seed = match case_id {
                "impedance_admittance_z_to_y_three_port_well_conditioned" => 20_260_933,
                "impedance_admittance_y_to_z_three_port_well_conditioned" => 20_260_934,
                "impedance_admittance_z_to_y_three_port_near_singular" => 20_260_935,
                "impedance_admittance_y_to_z_three_port_near_singular" => 20_260_936,
                _ => unreachable!("fixture table contains an unknown case"),
            };
            assert_eq!(metadata.random_seed, expected_seed);
            assert_eq!(metadata.shape.frequency, vec![3]);
            assert!(data.frequency_hz.iter().all(|value| value.is_finite()));
            assert_eq!(data.frequency_hz.len(), 3);

            if case_id.contains("near_singular") {
                let near = metadata
                    .near_singular
                    .as_ref()
                    .expect("near fixture must record its construction");
                assert_eq!(near.binary_exponent, -20);
                assert_eq!(
                    near.determinant_factors,
                    vec![(2.0_f64).powi(-20), 0.625, 0.75]
                );
                assert_eq!(
                    near.determinant,
                    near.determinant_factors.iter().product::<f64>()
                );
                assert!(near.determinant_factors_nonzero);
                assert_eq!(near.matrix_structure, "upper_triangular");
                assert_eq!(near.small_diagonal_port, 0);
                assert_eq!(near.small_diagonal_value, (2.0_f64).powi(-20));
                let expected_system_matrix = if operation == "z_to_y" { "Z" } else { "Y" };
                assert_eq!(near.system_matrix, expected_system_matrix);
            } else {
                assert!(metadata.near_singular.is_none());
            }

            let (input, expected, expected_input_shape, expected_output_shape) =
                if operation == "z_to_y" {
                    (
                        fixture_array3(&data.z_ohm),
                        fixture_array3(&data.y_s),
                        &metadata.shape.input_z,
                        &metadata.shape.output_y,
                    )
                } else {
                    (
                        fixture_array3(&data.y_s),
                        fixture_array3(&data.z_ohm),
                        &metadata.shape.input_y,
                        &metadata.shape.output_z,
                    )
                };
            assert_eq!(input.dim(), (3, 3, 3));
            assert_eq!(expected.dim(), (3, 3, 3));
            assert_eq!(expected_input_shape.as_deref(), Some([3, 3, 3].as_slice()));
            assert_eq!(expected_output_shape.as_deref(), Some([3, 3, 3].as_slice()));

            let actual = if operation == "z_to_y" {
                z_to_y(&input).expect("recorded Z fixture must be invertible")
            } else {
                y_to_z(&input).expect("recorded Y fixture must be invertible")
            };
            let tolerance_policy = &metadata.tolerance_policy;
            assert_eq!(tolerance_policy.rtol, 1e-12);
            let atol = if operation == "z_to_y" {
                assert!(tolerance_policy.atol_ohm.is_none());
                tolerance_policy
                    .atol_s
                    .expect("Z-to-Y fixture needs atol_s")
            } else {
                assert!(tolerance_policy.atol_s.is_none());
                tolerance_policy
                    .atol_ohm
                    .expect("Y-to-Z fixture needs atol_ohm")
            };
            assert_eq!(atol, 1e-12);
            assert!(!tolerance_policy.comparison.is_empty());
            assert!(!tolerance_policy.justification.is_empty());
            assert!(!tolerance_policy.regeneration.is_empty());
            assert_recorded_output(&actual, &expected, tolerance_policy.rtol, atol);
        }
    }

    #[test]
    fn one_port_analytical_inverse_works_in_both_directions() {
        let z = Array3::from_shape_vec((1, 1, 1), vec![complex(4.0, 3.0)]).unwrap();
        let y = z_to_y(&z).expect("one-port Z must be invertible");
        assert_complex_close(y[[0, 0, 0]], complex(0.16, -0.12), 1e-15);

        let recovered_z = y_to_z(&y).expect("one-port Y must be invertible");
        assert_complex_close(recovered_z[[0, 0, 0]], z[[0, 0, 0]], 1e-14);
    }

    #[test]
    fn arbitrary_multiport_round_trips_in_both_directions() {
        let z = arbitrary_three_port_stack();
        let y = z_to_y(&z).expect("arbitrary Z stack must be invertible");
        let recovered_z = y_to_z(&y).expect("computed Y stack must be invertible");
        assert_array3_close(&recovered_z, &z, 1e-13);

        let arbitrary_y = arbitrary_three_port_stack().mapv(|value| value / complex(5.0, -1.5));
        let z_from_y = y_to_z(&arbitrary_y).expect("arbitrary Y stack must be invertible");
        let recovered_y = z_to_y(&z_from_y).expect("computed Z stack must be invertible");
        assert_array3_close(&recovered_y, &arbitrary_y, 1e-13);
    }

    #[test]
    fn rejects_non_square_and_zero_port_shapes_for_both_quantities() {
        let non_square = Array3::zeros((2, 2, 3));
        assert_eq!(
            z_to_y(&non_square),
            Err(ImpedanceAdmittanceError::InvalidZShape { shape: (2, 2, 3) })
        );
        assert_eq!(
            y_to_z(&non_square),
            Err(ImpedanceAdmittanceError::InvalidYShape { shape: (2, 2, 3) })
        );

        let zero_port = Array3::zeros((2, 0, 0));
        assert_eq!(
            z_to_y(&zero_port),
            Err(ImpedanceAdmittanceError::InvalidZShape { shape: (2, 0, 0) })
        );
        assert_eq!(
            y_to_z(&zero_port),
            Err(ImpedanceAdmittanceError::InvalidYShape { shape: (2, 0, 0) })
        );
    }

    #[test]
    fn rejects_non_finite_inputs_with_quantity_and_location() {
        let mut z = Array3::zeros((2, 2, 2));
        z[[0, 0, 0]] = complex(1.0, 0.0);
        z[[0, 1, 1]] = complex(1.0, 0.0);
        z[[1, 0, 1]] = complex(f64::NAN, 0.0);
        assert_eq!(
            z_to_y(&z),
            Err(ImpedanceAdmittanceError::NonFiniteZ {
                frequency: 1,
                row: 0,
                column: 1,
            })
        );

        let mut y = Array3::zeros((2, 2, 2));
        y[[0, 0, 0]] = complex(1.0, 0.0);
        y[[0, 1, 1]] = complex(1.0, 0.0);
        y[[0, 1, 0]] = complex(0.0, f64::INFINITY);
        assert_eq!(
            y_to_z(&y),
            Err(ImpedanceAdmittanceError::NonFiniteY {
                frequency: 0,
                row: 1,
                column: 0,
            })
        );
    }

    #[test]
    fn reports_non_finite_computation_for_finite_input() {
        // The smallest positive subnormal is finite and non-zero, so it does
        // not satisfy the exact-singular criterion.  Its reciprocal overflows
        // and must be reported as a computation failure instead.
        let smallest_positive = f64::from_bits(1);
        let z = Array3::from_shape_vec((1, 1, 1), vec![complex(smallest_positive, 0.0)]).unwrap();
        assert_eq!(
            z_to_y(&z),
            Err(ImpedanceAdmittanceError::NonFiniteComputation {
                frequency: 0,
                row: 0,
                column: 0,
            })
        );

        let y = Array3::from_shape_vec((1, 1, 1), vec![complex(smallest_positive, 0.0)]).unwrap();
        assert_eq!(
            y_to_z(&y),
            Err(ImpedanceAdmittanceError::NonFiniteComputation {
                frequency: 0,
                row: 0,
                column: 0,
            })
        );

        // The input components themselves are finite, but the pivot
        // magnitude calculation overflows.  That intermediate is also a
        // deterministic non-finite computation, not an exact singularity.
        let huge = Array3::from_shape_vec((1, 1, 1), vec![complex(f64::MAX, f64::MAX)]).unwrap();
        assert_eq!(
            z_to_y(&huge),
            Err(ImpedanceAdmittanceError::NonFiniteComputation {
                frequency: 0,
                row: 0,
                column: 0,
            })
        );
    }

    #[test]
    fn reports_exact_singular_matrices_with_frequency_and_pivot() {
        let singular = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                complex(1.0, 0.0),
                complex(2.0, 0.0),
                complex(2.0, 0.0),
                complex(4.0, 0.0),
                complex(3.0, 0.0),
                complex(0.0, 0.0),
                complex(0.0, 0.0),
                complex(5.0, 0.0),
            ],
        )
        .unwrap();
        assert_eq!(
            z_to_y(&singular),
            Err(ImpedanceAdmittanceError::Singular {
                frequency: 0,
                pivot: 1,
            })
        );

        let mut singular_y = Array3::zeros((2, 2, 2));
        singular_y[[0, 0, 0]] = complex(3.0, 0.0);
        singular_y[[0, 0, 1]] = complex(1.0, 0.0);
        singular_y[[0, 1, 0]] = complex(6.0, 0.0);
        singular_y[[0, 1, 1]] = complex(2.0, 0.0);
        assert_eq!(
            y_to_z(&singular_y),
            Err(ImpedanceAdmittanceError::Singular {
                frequency: 0,
                pivot: 1,
            })
        );
    }
}
