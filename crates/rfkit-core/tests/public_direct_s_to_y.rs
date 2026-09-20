use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{ConversionStage, Error, Frequency, Network, ParameterKind};
use serde::Deserialize;
use serde_json::json;
use std::panic::{AssertUnwindSafe, catch_unwind};

const S_TO_Y_FIXTURE_JSON: &str =
    include_str!("../../../tools/oracle/fixtures/power_wave_s_to_y_three_port_complex_z0.json");
const DIRECT_S_TO_Y_SINGULAR_FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/power_wave_s_to_y_three_port_singular_i_minus_s_complex_z0.json"
);

#[derive(Debug, Deserialize)]
struct FixtureDocument {
    data: FixtureData,
    metadata: FixtureMetadata,
}

#[derive(Debug, Deserialize)]
struct FixtureData {
    frequency_hz: Vec<f64>,
    s: Vec<Vec<Vec<ComplexValue>>>,
    y_s: Vec<Vec<Vec<ComplexValue>>>,
    z0_ohm: Vec<Vec<ComplexValue>>,
}

#[derive(Debug, Deserialize)]
struct ComplexValue {
    real: f64,
    imag: f64,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    operation: String,
    tolerance_policy: TolerancePolicy,
}

#[derive(Debug, Deserialize)]
struct TolerancePolicy {
    rtol: f64,
    atol_s: f64,
}

fn complex(value: &ComplexValue) -> Complex64 {
    Complex64::new(value.real, value.imag)
}

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
    assert!(!values.is_empty());
    let nfreq = values.len();
    let nport = values[0].len();
    assert!(nport > 0);
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
        .map(complex)
        .collect();
    Array3::from_shape_vec((nfreq, nport, nport), flattened).unwrap()
}

fn z0_array(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    assert!(!values.is_empty());
    let nfreq = values.len();
    let nport = values[0].len();
    assert!(nport > 0);
    assert!(values.iter().all(|row| row.len() == nport));
    let flattened = values
        .iter()
        .flat_map(|row| row.iter())
        .map(complex)
        .collect();
    Array2::from_shape_vec((nfreq, nport), flattened).unwrap()
}

fn assert_array3_close(
    actual: &Array3<Complex64>,
    expected: &Array3<Complex64>,
    rtol: f64,
    atol: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual_value - expected_value).norm();
        let bound = atol + rtol * expected_value.norm();
        assert!(
            difference <= bound,
            "index {index}: actual={actual_value:?}, expected={expected_value:?}, difference={difference:e}, bound={bound:e}"
        );
    }
}

fn identity(nport: usize) -> Array3<Complex64> {
    Array3::from_shape_fn((1, nport, nport), |(_, row, column)| {
        if row == column {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })
}

fn asymmetric_case() -> (Frequency, Array3<Complex64>, Array2<Complex64>) {
    let frequency = Frequency::from_hz(vec![2.0e9, 1.0e9]).unwrap();
    let s = Array3::from_shape_fn((2, 3, 3), |(f, row, column)| {
        let real = if row == column {
            0.08 + 0.01 * f as f64 + 0.006 * row as f64
        } else {
            0.011 * (row + 1) as f64 - 0.004 * (column + 1) as f64
        };
        let imag = if row == column {
            -0.02 + 0.004 * row as f64
        } else {
            0.003 * (f + row + 1) as f64 - 0.002 * column as f64
        };
        Complex64::new(real, imag)
    });
    let z0 = Array2::from_shape_fn((2, 3), |(f, port)| {
        Complex64::new(
            [42.0, -57.0, 71.0][port] + 1.5 * f as f64,
            [2.0, -1.5, 3.25][port] + 0.2 * f as f64,
        )
    });
    (frequency, s, z0)
}

fn assert_direct_wave_equation(network: &Network, y: &Array3<Complex64>, tolerance: f64) {
    let s = network.s();
    let z0 = network.z0();
    let (nfreq, nport, nport_columns) = s.dim();
    assert_eq!(y.dim(), (nfreq, nport, nport_columns));
    assert_eq!(nport, nport_columns);

    for frequency in 0..nfreq {
        let normalization: Vec<_> = (0..nport)
            .map(|port| Complex64::new(1.0 / (2.0 * z0[[frequency, port]].re.abs().sqrt()), 0.0))
            .collect();

        for row in 0..nport {
            for column in 0..nport {
                let left = (0..nport)
                    .map(|index| {
                        let mut a = s[[frequency, row, index]] * z0[[frequency, index]];
                        if row == index {
                            a += z0[[frequency, row]].conj();
                        }
                        a * normalization[index] * y[[frequency, index, column]]
                    })
                    .sum::<Complex64>();
                let identity = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let right = (identity - s[[frequency, row, column]]) * normalization[column];
                assert!(
                    (left - right).norm() <= tolerance,
                    "A*Y != B at [{frequency},{row},{column}]: left={left:?}, right={right:?}"
                );
            }
        }
    }
}

#[test]
fn direct_s_to_y_matches_pinned_fixture_and_composed_common_domain() {
    let fixture: FixtureDocument = serde_json::from_str(S_TO_Y_FIXTURE_JSON).unwrap();
    assert_eq!(fixture.metadata.operation, "s_to_y");
    let frequency = Frequency::from_hz(fixture.data.frequency_hz.clone()).unwrap();
    let s = array3(&fixture.data.s);
    let expected = array3(&fixture.data.y_s);
    let z0 = z0_array(&fixture.data.z0_ohm);
    let network = Network::new(frequency.clone(), s.clone(), z0.clone()).unwrap();

    let direct = network.to_y_direct_power().unwrap();
    assert_array3_close(
        &direct,
        &expected,
        fixture.metadata.tolerance_policy.rtol,
        fixture.metadata.tolerance_policy.atol_s,
    );
    let composed = network.to_y_power().unwrap();
    assert_array3_close(&direct, &composed, 1.0e-12, 1.0e-12);
    assert_eq!(network.frequency(), &frequency);
    assert_eq!(network.s(), &s);
    assert_eq!(network.z0(), &z0);
}

#[test]
fn direct_s_to_y_matches_pinned_singular_i_minus_s_fixture() {
    let fixture: FixtureDocument =
        serde_json::from_str(DIRECT_S_TO_Y_SINGULAR_FIXTURE_JSON).unwrap();
    assert_eq!(fixture.metadata.operation, "s_to_y_direct");
    let frequency = Frequency::from_hz(fixture.data.frequency_hz.clone()).unwrap();
    let s = array3(&fixture.data.s);
    let expected = array3(&fixture.data.y_s);
    let z0 = z0_array(&fixture.data.z0_ohm);
    let network = Network::new(frequency, s, z0).unwrap();

    let direct = network.to_y_direct_power().unwrap();
    assert_array3_close(
        &direct,
        &expected,
        fixture.metadata.tolerance_policy.rtol,
        fixture.metadata.tolerance_policy.atol_s,
    );
    assert_direct_wave_equation(&network, &direct, 2.0e-12);
    assert!(matches!(
        network.to_y_power(),
        Err(Error::Singular {
            stage: ConversionStage::SToZ,
            ..
        })
    ));
}

#[test]
fn ideal_open_and_floating_series_use_direct_path_while_composed_path_stays_restricted() {
    let open_frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let open_z0 = Array2::from_shape_vec(
        (1, 2),
        vec![Complex64::new(50.0, 2.0), Complex64::new(-60.0, 3.0)],
    )
    .unwrap();
    let open = Network::new(open_frequency, identity(2), open_z0).unwrap();
    let direct_open = open.to_y_direct_power().unwrap();
    assert!(direct_open.iter().all(|value| value.norm() <= 1.0e-14));
    assert!(matches!(
        open.to_y_power(),
        Err(Error::Singular {
            stage: ConversionStage::SToZ,
            frequency: 0,
            pivot: 0,
        })
    ));

    let frequency = Frequency::from_hz(vec![1.0e9, 2.0e9]).unwrap();
    let floating_y = Array3::from_shape_vec(
        (2, 2, 2),
        vec![
            Complex64::new(0.01, 0.0),
            Complex64::new(-0.01, 0.0),
            Complex64::new(-0.01, 0.0),
            Complex64::new(0.01, 0.0),
            Complex64::new(0.005, 0.0),
            Complex64::new(-0.005, 0.0),
            Complex64::new(-0.005, 0.0),
            Complex64::new(0.005, 0.0),
        ],
    )
    .unwrap();
    let z0 = Array2::from_elem((2, 2), Complex64::new(50.0, 0.0));
    let floating = Network::from_y_direct_power(frequency, floating_y.clone(), z0).unwrap();
    let recovered = floating.to_y_direct_power().unwrap();
    assert_array3_close(&recovered, &floating_y, 1.0e-12, 1.0e-12);
    let composed_floating = floating.to_y_power();
    assert!(matches!(
        composed_floating,
        Err(Error::Singular {
            stage: ConversionStage::SToZ,
            frequency: 0,
            ..
        })
    ));
}

#[test]
fn direct_s_to_y_round_trip_preserves_asymmetric_nport_and_complex_negative_z0() {
    let (frequency, s, z0) = asymmetric_case();
    let source_s = s.clone();
    let source_z0 = z0.clone();
    let network = Network::new(frequency.clone(), s, z0).unwrap();
    let y = network.to_y_direct_power().unwrap();

    assert_direct_wave_equation(&network, &y, 2.0e-12);
    let reconstructed =
        Network::from_y_direct_power(frequency.clone(), y.clone(), source_z0.clone()).unwrap();
    assert_array3_close(reconstructed.s(), &source_s, 1.0e-12, 1.0e-12);
    assert_array3_close(
        &reconstructed.to_y_direct_power().unwrap(),
        &y,
        1.0e-12,
        1.0e-12,
    );
    assert_eq!(network.frequency(), &frequency);
    assert_eq!(network.s(), &source_s);
    assert_eq!(network.z0(), &source_z0);
}

#[test]
fn exact_and_near_singular_direct_systems_have_explicit_policy() {
    let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
    let exact = Network::new(
        frequency.clone(),
        Array3::from_elem((1, 1, 1), Complex64::new(-1.0, 0.0)),
        z0.clone(),
    )
    .unwrap();
    assert_eq!(
        exact.to_y_direct_power().unwrap_err(),
        Error::Singular {
            stage: ConversionStage::SToY,
            frequency: 0,
            pivot: 0,
        }
    );

    let epsilon = 1.0e-10;
    let near = Network::new(
        frequency,
        Array3::from_elem((1, 1, 1), Complex64::new(-1.0 + epsilon, 0.0)),
        z0,
    )
    .unwrap();
    let y = near.to_y_direct_power().unwrap();
    assert!(y[[0, 0, 0]].is_finite());
    assert!(y[[0, 0, 0]].re > 1.0e8);
}

#[test]
fn direct_s_to_y_reports_shapes_references_and_nonfinite_computation_at_s_to_y_stage() {
    let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let valid_z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));

    let non_square = Network::new(
        frequency.clone(),
        Array3::from_elem((1, 1, 1), Complex64::new(0.0, 0.0)),
        valid_z0.clone(),
    )
    .unwrap();
    let mut non_square_value = serde_json::to_value(&non_square).unwrap();
    let scalar = non_square_value["s"]["data"][0].clone();
    non_square_value["s"]["dim"] = json!([1, 1, 2]);
    non_square_value["s"]["data"]
        .as_array_mut()
        .unwrap()
        .push(scalar);
    let non_square: Network = serde_json::from_value(non_square_value).unwrap();
    assert_eq!(
        non_square.to_y_direct_power().unwrap_err(),
        Error::InvalidShape {
            stage: ConversionStage::SToY,
            parameter: ParameterKind::S,
            shape: vec![1, 1, 2],
        }
    );

    let valid = Network::new(
        frequency.clone(),
        Array3::from_elem((1, 1, 1), Complex64::new(0.0, 0.0)),
        valid_z0.clone(),
    )
    .unwrap();
    let mut invalid_z0_value = serde_json::to_value(&valid).unwrap();
    let z0_value = invalid_z0_value["z0"]["data"][0].clone();
    invalid_z0_value["z0"]["dim"] = json!([2, 1]);
    invalid_z0_value["z0"]["data"]
        .as_array_mut()
        .unwrap()
        .push(z0_value);
    let invalid_z0: Network = serde_json::from_value(invalid_z0_value).unwrap();
    assert_eq!(
        invalid_z0.to_y_direct_power().unwrap_err(),
        Error::InvalidShape {
            stage: ConversionStage::SToY,
            parameter: ParameterKind::Z0,
            shape: vec![2, 1],
        }
    );

    let mut non_finite_s = Array3::from_elem((1, 1, 1), Complex64::new(0.0, 0.0));
    non_finite_s[[0, 0, 0]] = Complex64::new(f64::NAN, 0.0);
    let non_finite_s = Network::new(frequency.clone(), non_finite_s, valid_z0.clone()).unwrap();
    assert_eq!(
        non_finite_s.to_y_direct_power().unwrap_err(),
        Error::NonFiniteS {
            stage: ConversionStage::SToY,
            frequency: 0,
            row: 0,
            column: 0,
        }
    );

    let non_finite_z0 = Network::new(
        frequency.clone(),
        Array3::zeros((1, 1, 1)),
        Array2::from_elem((1, 1), Complex64::new(f64::INFINITY, 0.0)),
    )
    .unwrap();
    assert_eq!(
        non_finite_z0.to_y_direct_power().unwrap_err(),
        Error::NonFiniteZ0 {
            stage: ConversionStage::SToY,
            frequency: 0,
            port: 0,
        }
    );

    let zero_real_z0 = Network::new(
        frequency.clone(),
        Array3::zeros((1, 1, 1)),
        Array2::from_elem((1, 1), Complex64::new(0.0, 10.0)),
    )
    .unwrap();
    assert_eq!(
        zero_real_z0.to_y_direct_power().unwrap_err(),
        Error::ZeroRealReferenceImpedance {
            stage: ConversionStage::SToY,
            frequency: 0,
            port: 0,
        }
    );

    let overflow = Network::new(
        frequency,
        Array3::from_elem((1, 1, 1), Complex64::new(f64::MAX, 0.0)),
        Array2::from_elem((1, 1), Complex64::new(f64::MAX, 0.0)),
    )
    .unwrap();
    assert!(matches!(
        overflow.to_y_direct_power(),
        Err(Error::NonFiniteComputation {
            stage: ConversionStage::SToY,
            frequency: 0,
            ..
        })
    ));
}

#[test]
fn direct_s_to_y_rejects_malformed_serde_frequency_axes_without_panicking() {
    let network = Network::new(
        Frequency::from_hz(vec![1.0e9]).unwrap(),
        Array3::zeros((1, 1, 1)),
        Array2::from_elem((1, 1), Complex64::new(50.0, 0.0)),
    )
    .unwrap();

    let mut empty_value = serde_json::to_value(&network).unwrap();
    empty_value["frequency"]["hz"] = json!([]);
    let empty: Network = serde_json::from_value(empty_value).unwrap();
    let empty_result = catch_unwind(AssertUnwindSafe(|| empty.to_y_direct_power()));
    assert!(empty_result.is_ok());
    assert_eq!(
        empty_result.unwrap().unwrap_err(),
        Error::EmptyConversionFrequency {
            stage: ConversionStage::SToY,
        }
    );

    let mut mismatch_value = serde_json::to_value(&network).unwrap();
    mismatch_value["frequency"]["hz"] = json!([1.0e9, 2.0e9]);
    let mismatch: Network = serde_json::from_value(mismatch_value).unwrap();
    let mismatch_result = catch_unwind(AssertUnwindSafe(|| mismatch.to_y_direct_power()));
    assert!(mismatch_result.is_ok());
    assert_eq!(
        mismatch_result.unwrap().unwrap_err(),
        Error::ConversionFrequencyLengthMismatch {
            stage: ConversionStage::SToY,
            expected: 2,
            actual: 1,
        }
    );
}
