use approx::assert_relative_eq;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, MaxSingularValuePowerArithmetic, Network};
use serde_json::json;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn network(
    frequency_hz: Vec<f64>,
    nports: usize,
    values: Vec<Complex64>,
    z0: Vec<Complex64>,
) -> Network {
    let nfreq = frequency_hz.len();
    Network::new(
        Frequency::from_hz(frequency_hz).unwrap(),
        Array3::from_shape_vec((nfreq, nports, nports), values).unwrap(),
        Array2::from_shape_vec((nfreq, nports), z0).unwrap(),
    )
    .unwrap()
}

fn real_z0(nfreq: usize, nports: usize) -> Vec<Complex64> {
    vec![c(50.0, 0.0); nfreq * nports]
}

#[test]
fn one_port_zero_and_complex_magnitude_are_sample_aligned() {
    let source = network(
        vec![3.0, -0.0, 3.0],
        1,
        vec![c(-0.3, 0.4), c(0.0, 0.0), c(1.2, -1.6)],
        vec![c(41.0, 2.0), c(42.0, -1.0), c(43.0, 3.0)],
    );
    let snapshot = source.clone();
    let actual = source.max_singular_value_power().unwrap();
    assert_eq!(actual.len(), 3);
    assert_relative_eq!(actual[0], 0.5, epsilon = 1.0e-14);
    assert_eq!(actual[1], 0.0);
    assert_relative_eq!(actual[2], 2.0, epsilon = 1.0e-14);
    assert_eq!(source, snapshot);
    assert_eq!(source.frequency().hz()[1].to_bits(), (-0.0f64).to_bits());
}

#[test]
fn coherent_and_nonnormal_examples_require_full_svd() {
    let source = network(
        vec![1.0, 2.0],
        2,
        vec![
            c(0.6, 0.0),
            c(0.6, 0.0),
            c(0.6, 0.0),
            c(0.6, 0.0),
            c(0.0, 0.0),
            c(2.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
        ],
        real_z0(2, 2),
    );
    let actual = source.max_singular_value_power().unwrap();
    assert_relative_eq!(actual[0], 1.2, epsilon = 1.0e-14);
    assert_relative_eq!(actual[1], 2.0, epsilon = 1.0e-14);
    // Every individual column of the first sample has norm below one, while
    // coherent excitation reaches 1.2; the second sample is nilpotent despite
    // zero eigenvalues and still has singular value 2.
}

#[test]
fn singular_repeated_and_unitary_cases_are_valid() {
    let source = network(
        vec![1.0, 2.0, 3.0],
        3,
        vec![
            // zero matrix
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            // rank-one matrix with repeated zero singular values
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            // scaled unitary permutation
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.5, 0.0),
            c(0.0, 0.0),
            c(1.5, 0.0),
            c(0.0, 0.0),
            c(1.5, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
        ],
        real_z0(3, 3),
    );
    let actual = source.max_singular_value_power().unwrap();
    assert_eq!(actual[0], 0.0);
    assert_relative_eq!(actual[1], 1.0, epsilon = 1.0e-14);
    assert_relative_eq!(actual[2], 1.5, epsilon = 1.0e-14);
}

#[test]
fn six_port_block_diagonal_case_uses_the_full_dense_dimension() {
    let source_s = Array3::from_shape_fn((2, 6, 6), |(frequency, row, column)| {
        if row / 2 == column / 2 {
            if frequency == 0 {
                c(0.3, 0.4)
            } else {
                c(0.6, 0.8)
            }
        } else {
            c(0.0, 0.0)
        }
    });
    let source = Network::new(
        Frequency::from_hz(vec![1.0, 2.0]).unwrap(),
        source_s,
        Array2::from_elem((2, 6), c(50.0, 0.0)),
    )
    .unwrap();

    let actual = source.max_singular_value_power().unwrap();
    assert_relative_eq!(actual[0], 1.0, epsilon = 1.0e-14);
    assert_relative_eq!(actual[1], 2.0, epsilon = 1.0e-14);
}

#[test]
fn near_degenerate_dense_complex_values_and_scaling_invariant_are_stable() {
    // A unitary DFT basis times nearly repeated positive singular values is
    // dense and complex, while its singular values remain analytically known.
    let root = c(-0.5, 3.0_f64.sqrt() / 2.0);
    let normalization = 1.0 / 3.0_f64.sqrt();
    let dft = Array2::from_shape_fn((3, 3), |(row, column)| {
        let root_power = match (row * column) % 3 {
            0 => c(1.0, 0.0),
            1 => root,
            _ => root * root,
        };
        c(root_power.re * normalization, root_power.im * normalization)
    });
    let diagonal = Array2::from_shape_fn((3, 3), |(row, column)| {
        if row == column {
            c(
                match row {
                    0 => 1.0,
                    1 => 1.0 + 1.0e-12,
                    _ => 1.0 - 2.0e-12,
                },
                0.0,
            )
        } else {
            c(0.0, 0.0)
        }
    });
    let dense = dft.dot(&diagonal);
    let source_s = Array3::from_shape_fn((1, 3, 3), |(_, row, column)| dense[[row, column]]);
    let source = Network::new(
        Frequency::from_hz(vec![-2.5]).unwrap(),
        source_s.clone(),
        Array2::from_elem((1, 3), c(50.0, 1.0)),
    )
    .unwrap();
    let base_sigma = source.max_singular_value_power().unwrap();
    assert_relative_eq!(base_sigma[0], 1.0 + 1.0e-12, epsilon = 5.0e-13);

    let alpha = c(-1.7, 0.9);
    let scaled_s = source_s.mapv(|value| alpha * value);
    let scaled = Network::new(
        Frequency::from_hz(vec![-2.5]).unwrap(),
        scaled_s,
        Array2::from_elem((1, 3), c(50.0, 1.0)),
    )
    .unwrap();
    let scaled_sigma = scaled.max_singular_value_power().unwrap();
    assert_relative_eq!(
        scaled_sigma[0],
        alpha.norm() * base_sigma[0],
        epsilon = 5.0e-13
    );
}

#[test]
fn port_permutation_and_unitary_coordinate_changes_preserve_sigma_max() {
    let source = network(
        vec![1.0, 2.0],
        3,
        vec![
            c(0.2, 0.1),
            c(0.3, -0.2),
            c(-0.1, 0.05),
            c(-0.4, 0.25),
            c(0.1, -0.3),
            c(0.2, 0.04),
            c(0.05, -0.1),
            c(0.4, 0.2),
            c(-0.2, 0.15),
            c(-0.2, 0.05),
            c(0.1, 0.2),
            c(0.25, -0.1),
            c(0.3, 0.0),
            c(-0.1, 0.4),
            c(0.2, -0.2),
            c(0.1, 0.1),
            c(-0.3, 0.2),
            c(0.15, -0.05),
        ],
        vec![
            c(41.0, 2.0),
            c(58.0, -1.0),
            c(73.0, 3.0),
            c(44.0, 1.0),
            c(61.0, -2.0),
            c(76.0, 4.0),
        ],
    );
    let original = source.max_singular_value_power().unwrap();
    let permuted = source.permute_ports(&[2, 0, 1]).unwrap();
    let reordered = permuted.max_singular_value_power().unwrap();
    for (left, right) in original.iter().zip(reordered.iter()) {
        assert_relative_eq!(left, right, epsilon = 1.0e-13);
    }

    // A real orthogonal change of coordinates is a special unitary change.
    // Applying it on both sides with the transpose preserves the Euclidean
    // singular values while changing every matrix entry.
    let u = Array2::from_shape_vec(
        (3, 3),
        vec![
            c(1.0 / 3.0_f64.sqrt(), 0.0),
            c(1.0 / 3.0_f64.sqrt(), 0.0),
            c(1.0 / 3.0_f64.sqrt(), 0.0),
            c(-1.0 / 2.0_f64.sqrt(), 0.0),
            c(1.0 / 2.0_f64.sqrt(), 0.0),
            c(0.0, 0.0),
            c(-1.0 / 6.0_f64.sqrt(), 0.0),
            c(-1.0 / 6.0_f64.sqrt(), 0.0),
            c(2.0 / 6.0_f64.sqrt(), 0.0),
        ],
    )
    .unwrap();
    let u_h = u.t().mapv(|value| value.conj()).to_owned();
    let transformed_s = Array3::from_shape_fn((2, 3, 3), |(frequency, row, column)| {
        let matrix = source.s().slice(ndarray::s![frequency, .., ..]).to_owned();
        (u.dot(&matrix).dot(&u_h))[[row, column]]
    });
    let transformed = Network::new(
        source.frequency().clone(),
        transformed_s,
        source.z0().clone(),
    )
    .unwrap();
    let transformed_sigma = transformed.max_singular_value_power().unwrap();
    for (left, right) in original.iter().zip(transformed_sigma.iter()) {
        assert_relative_eq!(left, right, epsilon = 1.0e-12);
    }
}

#[test]
fn large_and_small_finite_scales_are_not_replaced_by_gram_squaring() {
    let source = network(
        vec![1.0, 2.0],
        2,
        vec![
            c(1.0e200, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0e-200, 0.0),
            c(1.0e-200, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0e-200, 0.0),
        ],
        real_z0(2, 2),
    );
    let actual = source.max_singular_value_power().unwrap();
    assert_relative_eq!(actual[0], 1.0e200, epsilon = 1.0e185);
    assert_relative_eq!(actual[1], 1.0e-200, epsilon = 1.0e-214);
}

#[test]
fn finite_input_overflow_is_a_structured_arithmetic_error() {
    let source = network(
        vec![1.0],
        2,
        vec![
            c(f64::MAX, 0.0),
            c(f64::MAX, 0.0),
            c(f64::MAX, 0.0),
            c(f64::MAX, 0.0),
        ],
        real_z0(1, 2),
    );
    assert_eq!(
        source.max_singular_value_power().unwrap_err(),
        Error::NonFiniteMaxSingularValuePowerComputation {
            frequency: 0,
            stage: MaxSingularValuePowerArithmetic::SingularValues,
        }
    );
}

#[test]
fn finite_extreme_complex_input_does_not_panic_on_unordered_svd() {
    let source = network(
        vec![-1.0],
        2,
        vec![c(9.0e307, 9.0e307), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
        real_z0(1, 2),
    );
    let result = catch_unwind(AssertUnwindSafe(|| source.max_singular_value_power()));
    assert!(result.is_ok(), "finite extreme S input must not panic");
    assert_eq!(
        result.unwrap().unwrap_err(),
        Error::NonFiniteMaxSingularValuePowerComputation {
            frequency: 0,
            stage: MaxSingularValuePowerArithmetic::SingularValues,
        }
    );
}

#[test]
fn malformed_serde_shapes_are_errors_not_panics() {
    let valid = network(vec![1.0], 1, vec![c(0.1, 0.0)], vec![c(50.0, 0.0)]);
    let mut malformed = serde_json::to_value(&valid).unwrap();
    malformed["frequency"]["hz"] = json!([]);
    malformed["s"]["dim"] = json!([0, 1, 1]);
    malformed["s"]["data"] = json!([]);
    malformed["z0"]["dim"] = json!([0, 1]);
    malformed["z0"]["data"] = json!([]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| malformed.max_singular_value_power()));
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().unwrap_err(),
        Error::EmptyMaxSingularValuePowerFrequency
    );

    let mut malformed = serde_json::to_value(&valid).unwrap();
    let scalar = malformed["s"]["data"][0].clone();
    malformed["frequency"]["hz"] = json!([1.0, 2.0]);
    malformed["s"]["dim"] = json!([1, 1, 1]);
    malformed["s"]["data"] = json!([scalar]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| malformed.max_singular_value_power()));
    assert!(result.is_ok());
    assert!(matches!(
        result.unwrap(),
        Err(Error::MaxSingularValuePowerFrequencyLengthMismatch { .. })
            | Err(Error::InvalidMaxSingularValuePowerSShape { .. })
    ));
}

#[test]
fn invalid_data_and_reference_domains_are_structured() {
    let nonfinite_frequency = network(
        vec![f64::INFINITY],
        1,
        vec![c(0.1, 0.0)],
        vec![c(50.0, 0.0)],
    );
    assert_eq!(
        nonfinite_frequency.max_singular_value_power().unwrap_err(),
        Error::NonFiniteMaxSingularValuePowerFrequency {
            index: 0,
            value: f64::INFINITY,
        }
    );

    let nonfinite_s = network(vec![1.0], 1, vec![c(f64::NAN, 0.0)], vec![c(50.0, 0.0)]);
    assert_eq!(
        nonfinite_s.max_singular_value_power().unwrap_err(),
        Error::NonFiniteMaxSingularValuePowerS {
            frequency: 0,
            row: 0,
            column: 0,
        }
    );

    for reference in [c(0.0, 2.0), c(-1.0, 4.0)] {
        let network = network(vec![1.0], 1, vec![c(0.1, 0.0)], vec![reference]);
        assert!(matches!(
            network.max_singular_value_power(),
            Err(
                Error::NonPositiveRealMaxSingularValuePowerReferenceImpedance {
                    frequency: 0,
                    port: 0,
                    ..
                }
            )
        ));
    }

    let nonfinite_z0 = network(vec![1.0], 1, vec![c(0.1, 0.0)], vec![c(f64::NAN, 0.0)]);
    assert_eq!(
        nonfinite_z0.max_singular_value_power().unwrap_err(),
        Error::NonFiniteMaxSingularValuePowerZ0 {
            frequency: 0,
            port: 0,
        }
    );
}

#[test]
fn arithmetic_error_mapping_retains_operation_specific_stage() {
    let error = Error::NonFiniteMaxSingularValuePowerComputation {
        frequency: 3,
        stage: MaxSingularValuePowerArithmetic::SingularValues,
    };
    assert!(error.to_string().contains("maximum singular-value power"));
    assert!(error.to_string().contains("SVD singular values"));
}
