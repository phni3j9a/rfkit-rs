use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{ConversionStage, Error, Frequency, Network, ParameterKind};
use serde::Deserialize;
use serde_json::json;
use std::panic::{AssertUnwindSafe, catch_unwind};

const Z_TO_S_FIXTURE_JSON: &str =
    include_str!("../../../tools/oracle/fixtures/power_wave_z_to_s_three_port_complex_z0.json");
const Y_TO_S_FIXTURE_JSON: &str =
    include_str!("../../../tools/oracle/fixtures/power_wave_y_to_s_three_port_complex_z0.json");
const Z_TO_S_NEAR_FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/power_wave_z_to_s_three_port_near_singular_real_equal_z0.json"
);

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
    #[serde(default)]
    y_s: Option<Vec<Vec<Vec<ComplexValue>>>>,
    #[serde(default)]
    z_ohm: Option<Vec<Vec<Vec<ComplexValue>>>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComplexValue {
    imag: f64,
    real: f64,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    operation: String,
    tolerance_policy: TolerancePolicy,
}

#[derive(Debug, Deserialize)]
struct TolerancePolicy {
    rtol: f64,
    #[serde(default)]
    atol: Option<f64>,
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
    let values = values
        .iter()
        .flat_map(|matrix| matrix.iter())
        .flat_map(|row| row.iter())
        .map(complex)
        .collect();
    Array3::from_shape_vec((nfreq, nport, nport), values).unwrap()
}

fn z0_array(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    assert!(!values.is_empty());
    let nfreq = values.len();
    let nport = values[0].len();
    assert!(nport > 0);
    assert!(values.iter().all(|row| row.len() == nport));
    let values = values
        .iter()
        .flat_map(|row| row.iter())
        .map(complex)
        .collect();
    Array2::from_shape_vec((nfreq, nport), values).unwrap()
}

fn assert_array3_close(
    actual: &Array3<Complex64>,
    expected: &Array3<Complex64>,
    rtol: f64,
    atol: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let difference = (actual - expected).norm();
        let bound = atol + rtol * expected.norm();
        assert!(
            difference <= bound,
            "index {index}: actual={actual:?}, expected={expected:?}, difference={difference:e}, bound={bound:e}"
        );
    }
}

fn assert_identity_product(left: &Array3<Complex64>, right: &Array3<Complex64>, tolerance: f64) {
    assert_eq!(left.dim(), right.dim());
    let (nfreq, nport, nport_columns) = left.dim();
    assert_eq!(nport, nport_columns);
    for frequency in 0..nfreq {
        for row in 0..nport {
            for column in 0..nport {
                let actual = (0..nport)
                    .map(|index| left[[frequency, row, index]] * right[[frequency, index, column]])
                    .sum::<Complex64>();
                let expected = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert!((actual - expected).norm() <= tolerance);
            }
        }
    }
}

#[test]
fn from_z_power_matches_pinned_z_to_s_and_near_singular_fixtures() {
    for json in [Z_TO_S_FIXTURE_JSON, Z_TO_S_NEAR_FIXTURE_JSON] {
        let fixture: FixtureDocument = serde_json::from_str(json).unwrap();
        assert_eq!(fixture.metadata.operation, "z_to_s");
        let frequency = Frequency::from_hz(fixture.data.frequency_hz.clone()).unwrap();
        let z = array3(fixture.data.z_ohm.as_deref().unwrap());
        let z0 = z0_array(&fixture.data.z0_ohm);
        let expected = array3(&fixture.data.s);
        let network = Network::from_z_power(frequency.clone(), z.clone(), z0.clone()).unwrap();
        let tolerance = &fixture.metadata.tolerance_policy;
        assert_array3_close(
            network.s(),
            &expected,
            tolerance.rtol,
            tolerance.atol.unwrap_or(1.0e-12),
        );
        assert_eq!(network.frequency(), &frequency);
        assert_eq!(network.z0(), &z0);

        let recovered = network.to_z_power().unwrap();
        assert_array3_close(
            &recovered,
            &z,
            tolerance.rtol,
            tolerance.atol.unwrap_or(1.0e-12),
        );
    }
}

#[test]
fn from_y_via_z_power_matches_pinned_y_to_s_fixture_and_round_trips_y() {
    let fixture: FixtureDocument = serde_json::from_str(Y_TO_S_FIXTURE_JSON).unwrap();
    assert_eq!(fixture.metadata.operation, "y_to_s");
    let frequency = Frequency::from_hz(fixture.data.frequency_hz.clone()).unwrap();
    let y = array3(fixture.data.y_s.as_deref().unwrap());
    let z0 = z0_array(&fixture.data.z0_ohm);
    let expected = array3(&fixture.data.s);
    let network = Network::from_y_via_z_power(frequency.clone(), y.clone(), z0.clone()).unwrap();
    let tolerance = &fixture.metadata.tolerance_policy;
    assert_array3_close(
        network.s(),
        &expected,
        tolerance.rtol,
        tolerance.atol.unwrap_or(1.0e-12),
    );
    assert_eq!(network.frequency(), &frequency);
    assert_eq!(network.z0(), &z0);
    assert_identity_product(
        &network.to_z_power().unwrap(),
        &network.to_y_power().unwrap(),
        1.0e-12,
    );
    assert_array3_close(&network.to_y_power().unwrap(), &y, 1.0e-12, 1.0e-12);
}

#[test]
fn constructors_cover_analytical_zero_and_negative_reference_cases() {
    let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
    let from_z = Network::from_z_power(
        frequency.clone(),
        Array3::from_elem((1, 1, 1), Complex64::new(100.0, 0.0)),
        z0.clone(),
    )
    .unwrap();
    assert!((from_z.s()[[0, 0, 0]].re - 1.0 / 3.0).abs() < 1.0e-14);

    let from_y = Network::from_y_via_z_power(
        frequency.clone(),
        Array3::from_elem((1, 1, 1), Complex64::new(0.02, 0.0)),
        z0.clone(),
    )
    .unwrap();
    assert!(from_y.s()[[0, 0, 0]].norm() < 1.0e-14);

    let zero_z =
        Network::from_z_power(frequency.clone(), Array3::zeros((1, 1, 1)), z0.clone()).unwrap();
    assert_eq!(zero_z.s()[[0, 0, 0]], Complex64::new(-1.0, 0.0));

    let negative_z0 = Complex64::new(-50.0, 10.0);
    let negative = Network::from_z_power(
        frequency.clone(),
        Array3::from_elem((1, 1, 1), Complex64::new(100.0, 0.0)),
        Array2::from_elem((1, 1), negative_z0),
    )
    .unwrap();
    let expected = (Complex64::new(100.0, 0.0) - negative_z0.conj())
        / (Complex64::new(100.0, 0.0) + negative_z0);
    assert!((negative.s()[[0, 0, 0]] - expected).norm() < 1.0e-14);
    assert!(
        (negative.to_z_power().unwrap()[[0, 0, 0]] - Complex64::new(100.0, 0.0)).norm() < 1.0e-12
    );
}

#[test]
fn constructors_preserve_asymmetric_nport_frequency_order_complex_z0_and_inputs() {
    let frequency_values = vec![2.0e9, -3.0, 2.0e9];
    let frequency = Frequency::from_hz(frequency_values.clone()).unwrap();
    let z = Array3::from_shape_fn((3, 3, 3), |(f, row, column)| {
        if row == column {
            Complex64::new(90.0 + f as f64 * 3.0 + row as f64, 5.0 + row as f64)
        } else {
            Complex64::new(
                0.7 * (row + 1) as f64 - 0.2 * column as f64,
                0.3 * (f + row + column + 1) as f64,
            )
        }
    });
    let z0 = Array2::from_shape_fn((3, 3), |(f, port)| {
        let real = [
            [51.0, -62.0, 73.0],
            [54.0, -65.0, 76.0],
            [57.0, -68.0, 79.0],
        ][f][port];
        Complex64::new(real, 1.5 * (port + 1) as f64)
    });
    let z_before = z.clone();
    let z0_before = z0.clone();
    let network = Network::from_z_power(frequency.clone(), z.clone(), z0.clone()).unwrap();
    assert_eq!(network.frequency().hz(), frequency_values.as_slice());
    assert_eq!(network.z0(), &z0);
    assert_eq!(z, z_before);
    assert_eq!(z0, z0_before);
    assert_array3_close(&network.to_z_power().unwrap(), &z, 1.0e-12, 1.0e-12);

    let y = Array3::from_shape_fn((3, 3, 3), |(f, row, column)| {
        if row == column {
            Complex64::new(
                0.02 + 0.001 * f as f64 + 0.001 * row as f64,
                0.0005 * (row + 1) as f64,
            )
        } else {
            Complex64::new(0.0004 * (row + 1) as f64, -0.0002 * (f + column + 1) as f64)
        }
    });
    let y_before = y.clone();
    let network = Network::from_y_via_z_power(frequency, y.clone(), z0).unwrap();
    assert_eq!(network.frequency().hz(), frequency_values.as_slice());
    assert_eq!(y, y_before);
    assert_array3_close(&network.to_y_power().unwrap(), &y, 1.0e-12, 1.0e-12);

    let nonfinite_frequency = Network::from_z_power(
        Frequency::from_hz(vec![f64::NAN]).unwrap(),
        Array3::from_elem((1, 1, 1), Complex64::new(100.0, 0.0)),
        Array2::from_elem((1, 1), Complex64::new(50.0, 0.0)),
    )
    .unwrap();
    assert!(nonfinite_frequency.frequency().hz()[0].is_nan());
}

#[test]
fn constructors_report_parameter_specific_frequency_and_shape_errors() {
    let z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
    let z = Array3::from_elem((2, 1, 1), Complex64::new(50.0, 0.0));
    let error =
        Network::from_z_power(Frequency::from_hz(vec![1.0]).unwrap(), z, z0.clone()).unwrap_err();
    assert_eq!(
        error,
        Error::ParameterFrequencyLengthMismatch {
            parameter: ParameterKind::Z,
            expected: 1,
            actual: 2,
        }
    );

    let y = Array3::from_elem((2, 1, 1), Complex64::new(0.02, 0.0));
    let error = Network::from_y_via_z_power(Frequency::from_hz(vec![1.0]).unwrap(), y, z0.clone())
        .unwrap_err();
    assert_eq!(
        error,
        Error::ParameterFrequencyLengthMismatch {
            parameter: ParameterKind::Y,
            expected: 1,
            actual: 2,
        }
    );

    let malformed_empty: Frequency = serde_json::from_value(json!({ "hz": [] })).unwrap();
    let error = Network::from_z_power(
        malformed_empty.clone(),
        Array3::zeros((1, 1, 1)),
        z0.clone(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::EmptyParameterFrequency {
            parameter: ParameterKind::Z,
        }
    );
    let error = Network::from_y_via_z_power(malformed_empty, Array3::zeros((1, 1, 1)), z0.clone())
        .unwrap_err();
    assert_eq!(
        error,
        Error::EmptyParameterFrequency {
            parameter: ParameterKind::Y,
        }
    );

    let error = Network::from_z_power(
        Frequency::from_hz(vec![1.0]).unwrap(),
        Array3::zeros((1, 0, 0)),
        Array2::zeros((1, 0)),
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::InvalidShape {
            stage: ConversionStage::ZToS,
            parameter: ParameterKind::Z,
            shape: vec![1, 0, 0],
        }
    );

    let error = Network::from_y_via_z_power(
        Frequency::from_hz(vec![1.0]).unwrap(),
        Array3::zeros((1, 1, 2)),
        z0,
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::InvalidShape {
            stage: ConversionStage::YToZ,
            parameter: ParameterKind::Y,
            shape: vec![1, 1, 2],
        }
    );
}

#[test]
fn constructors_report_distinct_z_y_z0_nonfinite_reference_and_singular_stages() {
    let frequency = Frequency::from_hz(vec![1.0]).unwrap();
    let valid_z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));

    let error = Network::from_z_power(
        frequency.clone(),
        Array3::from_elem((1, 1, 1), Complex64::new(f64::NAN, 0.0)),
        valid_z0.clone(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::NonFiniteZ {
            stage: ConversionStage::ZToS,
            frequency: 0,
            row: 0,
            column: 0,
        }
    );

    let error = Network::from_y_via_z_power(
        frequency.clone(),
        Array3::from_elem((1, 1, 1), Complex64::new(f64::INFINITY, 0.0)),
        valid_z0.clone(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::NonFiniteY {
            stage: ConversionStage::YToZ,
            frequency: 0,
            row: 0,
            column: 0,
        }
    );

    let error = Network::from_z_power(
        frequency.clone(),
        Array3::from_elem((1, 1, 1), Complex64::new(50.0, 0.0)),
        Array2::zeros((1, 2)),
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::InvalidShape {
            stage: ConversionStage::ZToS,
            parameter: ParameterKind::Z0,
            shape: vec![1, 2],
        }
    );

    let error = Network::from_z_power(
        frequency.clone(),
        Array3::zeros((1, 1, 1)),
        Array2::from_elem((1, 1), Complex64::new(f64::INFINITY, 0.0)),
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::NonFiniteZ0 {
            stage: ConversionStage::ZToS,
            frequency: 0,
            port: 0,
        }
    );

    let error = Network::from_y_via_z_power(
        frequency.clone(),
        Array3::from_elem((1, 1, 1), Complex64::new(0.02, 0.0)),
        Array2::from_elem((1, 1), Complex64::new(0.0, 10.0)),
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::ZeroRealReferenceImpedance {
            stage: ConversionStage::ZToS,
            frequency: 0,
            port: 0,
        }
    );

    let error = Network::from_z_power(
        frequency.clone(),
        Array3::from_elem((1, 1, 1), Complex64::new(-50.0, 0.0)),
        valid_z0.clone(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::Singular {
            stage: ConversionStage::ZToS,
            frequency: 0,
            pivot: 0,
        }
    );

    let error = Network::from_y_via_z_power(
        frequency.clone(),
        Array3::zeros((1, 1, 1)),
        valid_z0.clone(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::Singular {
            stage: ConversionStage::YToZ,
            frequency: 0,
            pivot: 0,
        }
    );

    let error = Network::from_y_via_z_power(
        frequency,
        Array3::from_elem((1, 1, 1), Complex64::new(-0.02, 0.0)),
        valid_z0,
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::Singular {
            stage: ConversionStage::ZToS,
            frequency: 0,
            pivot: 0,
        }
    );
}

#[test]
fn malformed_empty_frequency_never_panics_at_parameter_ingress() {
    let frequency: Frequency = serde_json::from_value(json!({ "hz": [] })).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        Network::from_z_power(frequency, Array3::zeros((0, 1, 1)), Array2::zeros((0, 1)))
    }));
    assert!(result.is_ok());
    assert!(matches!(
        result.unwrap(),
        Err(Error::EmptyParameterFrequency {
            parameter: ParameterKind::Z
        })
    ));
}
