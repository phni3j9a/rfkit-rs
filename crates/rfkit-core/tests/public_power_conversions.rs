use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{ConversionStage, Error, Frequency, Network, ParameterKind};
use serde::Deserialize;

const S_TO_Z_FIXTURE_JSON: &str =
    include_str!("../../../tools/oracle/fixtures/power_wave_s_to_z_three_port_complex_z0.json");
const S_TO_Y_FIXTURE_JSON: &str =
    include_str!("../../../tools/oracle/fixtures/power_wave_s_to_y_three_port_complex_z0.json");
const S_TO_Z_NEAR_FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/power_wave_s_to_z_three_port_near_singular_real_equal_z0.json"
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
    case_id: String,
    operation: String,
    shape: FixtureShape,
    tolerance_policy: TolerancePolicy,
}

#[derive(Debug, Deserialize)]
struct FixtureShape {
    frequency: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct TolerancePolicy {
    rtol: f64,
    #[serde(default)]
    atol_s: Option<f64>,
    #[serde(default)]
    atol_ohm: Option<f64>,
}

fn complex(value: &ComplexValue) -> Complex64 {
    Complex64::new(value.real, value.imag)
}

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
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

    let values = values
        .iter()
        .flat_map(|matrix| matrix.iter())
        .flat_map(|row| row.iter())
        .map(complex)
        .collect();
    Array3::from_shape_vec((nfreq, nport, nport), values)
        .expect("fixture matrix dimensions must agree")
}

fn z0_array(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    assert!(!values.is_empty(), "fixture must contain frequencies");
    let nfreq = values.len();
    let nport = values[0].len();
    assert!(nport > 0, "fixture must contain ports");
    assert!(values.iter().all(|row| row.len() == nport));

    let values = values
        .iter()
        .flat_map(|row| row.iter())
        .map(complex)
        .collect();
    Array2::from_shape_vec((nfreq, nport), values)
        .expect("fixture reference-impedance dimensions must agree")
}

fn fixture_network(fixture: &FixtureDocument) -> Network {
    let frequency = Frequency::from_hz(fixture.data.frequency_hz.clone())
        .expect("fixture frequency axis must be valid");
    Network::new(
        frequency,
        array3(&fixture.data.s),
        z0_array(&fixture.data.z0_ohm),
    )
    .expect("fixture network must satisfy the public constructor contract")
}

fn assert_array3_close(
    actual: &Array3<Complex64>,
    expected: &Array3<Complex64>,
    rtol: f64,
    atol: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate()
    {
        assert!(
            actual_value.re.is_finite() && actual_value.im.is_finite(),
            "actual value at flattened index {index} is non-finite"
        );
        let difference = (actual_value - expected_value).norm();
        let tolerance = atol + rtol * expected_value.norm();
        assert!(
            difference <= tolerance,
            "value at flattened index {index} differs by {difference:e}, tolerance {tolerance:e}: actual={actual_value:?}, expected={expected_value:?}"
        );
    }
}

fn assert_identity_product(z: &Array3<Complex64>, y: &Array3<Complex64>, tolerance: f64) {
    assert_eq!(z.dim(), y.dim());
    let (nfreq, nport, nport_columns) = z.dim();
    assert_eq!(nport, nport_columns);
    for frequency in 0..nfreq {
        for row in 0..nport {
            for column in 0..nport {
                let actual = (0..nport)
                    .map(|index| z[[frequency, row, index]] * y[[frequency, index, column]])
                    .sum::<Complex64>();
                let expected = if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert!(
                    (actual - expected).norm() <= tolerance,
                    "Z*Y[{frequency},{row},{column}] = {actual:?}, expected {expected:?}"
                );
            }
        }
    }
}

#[test]
fn public_methods_reuse_direct_pinned_fixtures_and_preserve_order() {
    let z_fixture: FixtureDocument =
        serde_json::from_str(S_TO_Z_FIXTURE_JSON).expect("S-to-Z fixture must parse");
    assert_eq!(
        z_fixture.metadata.case_id,
        "power_wave_s_to_z_three_port_complex_z0"
    );
    assert_eq!(z_fixture.metadata.operation, "s_to_z");
    assert_eq!(z_fixture.metadata.shape.frequency, vec![4]);
    let z_network = fixture_network(&z_fixture);
    let source_s = z_network.s().clone();
    let source_z0 = z_network.z0().clone();
    let actual_z = z_network.to_z_power().expect("public S-to-Z must succeed");
    let expected_z = array3(
        z_fixture
            .data
            .z_ohm
            .as_deref()
            .expect("S-to-Z fixture must contain expected Z"),
    );
    assert_array3_close(
        &actual_z,
        &expected_z,
        z_fixture.metadata.tolerance_policy.rtol,
        z_fixture
            .metadata
            .tolerance_policy
            .atol_ohm
            .expect("S-to-Z fixture must record an ohm tolerance"),
    );
    assert_eq!(actual_z.dim(), (4, 3, 3));
    assert_eq!(z_network.s(), &source_s);
    assert_eq!(z_network.z0(), &source_z0);
    assert_ne!(source_s[[0, 0, 1]], source_s[[0, 1, 0]]);
    assert_ne!(source_z0[[0, 0]], source_z0[[0, 1]]);
    assert_ne!(source_z0[[0, 0]], source_z0[[1, 0]]);

    let y_fixture: FixtureDocument =
        serde_json::from_str(S_TO_Y_FIXTURE_JSON).expect("S-to-Y fixture must parse");
    assert_eq!(
        y_fixture.metadata.case_id,
        "power_wave_s_to_y_three_port_complex_z0"
    );
    assert_eq!(y_fixture.metadata.operation, "s_to_y");
    assert_eq!(y_fixture.metadata.shape.frequency, vec![3]);
    let y_network = fixture_network(&y_fixture);
    let source_s = y_network.s().clone();
    let source_z0 = y_network.z0().clone();
    let actual_y = y_network.to_y_power().expect("public S-to-Y must succeed");
    let expected_y = array3(
        y_fixture
            .data
            .y_s
            .as_deref()
            .expect("S-to-Y fixture must contain expected Y"),
    );
    assert_array3_close(
        &actual_y,
        &expected_y,
        y_fixture.metadata.tolerance_policy.rtol,
        y_fixture
            .metadata
            .tolerance_policy
            .atol_s
            .expect("S-to-Y fixture must record a siemens tolerance"),
    );
    assert_eq!(actual_y.dim(), (3, 3, 3));
    assert_eq!(y_network.s(), &source_s);
    assert_eq!(y_network.z0(), &source_z0);
}

#[test]
fn public_one_port_complex_z0_is_analytical_and_z_y_are_inverses() {
    let s_value = Complex64::new(0.23, -0.17);
    let z0_value = Complex64::new(73.5, 12.25);
    let frequency = Frequency::from_hz(vec![2.45e9]).unwrap();
    let s = Array3::from_elem((1, 1, 1), s_value);
    let z0 = Array2::from_elem((1, 1), z0_value);
    let network = Network::new(frequency, s, z0).unwrap();

    let z = network.to_z_power().unwrap();
    let y = network.to_y_power().unwrap();
    let expected_z = (s_value * z0_value + z0_value.conj()) / (Complex64::new(1.0, 0.0) - s_value);
    let expected_y = Complex64::new(1.0, 0.0) / expected_z;
    assert!((z[[0, 0, 0]] - expected_z).norm() <= 1e-12);
    assert!((y[[0, 0, 0]] - expected_y).norm() <= 1e-14);
    assert_identity_product(&z, &y, 1e-13);
}

#[test]
fn public_y_path_preserves_z_y_identity_for_complex_multiport_data() {
    let fixture: FixtureDocument =
        serde_json::from_str(S_TO_Y_FIXTURE_JSON).expect("S-to-Y fixture must parse");
    let network = fixture_network(&fixture);
    let z = network.to_z_power().unwrap();
    let y = network.to_y_power().unwrap();
    assert_identity_product(&z, &y, 1e-12);
}

#[test]
fn public_near_singular_fixture_remains_finite() {
    let fixture: FixtureDocument =
        serde_json::from_str(S_TO_Z_NEAR_FIXTURE_JSON).expect("near fixture must parse");
    assert_eq!(fixture.metadata.operation, "s_to_z");
    let network = fixture_network(&fixture);
    let actual = network
        .to_z_power()
        .expect("finite exact-nonzero near-singular system must succeed");
    let expected = array3(
        fixture
            .data
            .z_ohm
            .as_deref()
            .expect("near fixture must contain expected Z"),
    );
    assert_array3_close(
        &actual,
        &expected,
        fixture.metadata.tolerance_policy.rtol,
        fixture
            .metadata
            .tolerance_policy
            .atol_ohm
            .expect("near fixture must record an ohm tolerance"),
    );
}

fn one_port_network(s: Complex64, z0: Complex64) -> Network {
    Network::new(
        Frequency::from_hz(vec![1.0e9]).unwrap(),
        Array3::from_elem((1, 1, 1), s),
        Array2::from_elem((1, 1), z0),
    )
    .unwrap()
}

#[test]
fn public_errors_preserve_conversion_stage_and_locations() {
    let error = one_port_network(Complex64::new(1.0, 0.0), Complex64::new(50.0, 0.0))
        .to_z_power()
        .expect_err("S=1 must be exactly singular in S-to-Z");
    assert_eq!(
        error,
        Error::Singular {
            stage: ConversionStage::SToZ,
            frequency: 0,
            pivot: 0,
        }
    );

    let error = one_port_network(Complex64::new(-1.0, 0.0), Complex64::new(50.0, 0.0))
        .to_y_power()
        .expect_err("S=-1 must make the composed Z-to-Y stage singular");
    assert_eq!(
        error,
        Error::Singular {
            stage: ConversionStage::ZToY,
            frequency: 0,
            pivot: 0,
        }
    );

    let error = one_port_network(Complex64::new(0.0, 0.0), Complex64::new(0.0, 50.0))
        .to_y_power()
        .expect_err("zero-real z0 must be rejected");
    assert_eq!(
        error,
        Error::ZeroRealReferenceImpedance {
            stage: ConversionStage::SToZ,
            frequency: 0,
            port: 0,
        }
    );

    let error = one_port_network(Complex64::new(0.0, 0.0), Complex64::new(f64::INFINITY, 0.0))
        .to_z_power()
        .expect_err("non-finite z0 must be rejected");
    assert_eq!(
        error,
        Error::NonFiniteZ0 {
            stage: ConversionStage::SToZ,
            frequency: 0,
            port: 0,
        }
    );

    let error = one_port_network(Complex64::new(f64::NAN, 0.0), Complex64::new(50.0, 0.0))
        .to_y_power()
        .expect_err("non-finite S must be rejected");
    assert_eq!(
        error,
        Error::NonFiniteS {
            stage: ConversionStage::SToZ,
            frequency: 0,
            row: 0,
            column: 0,
        }
    );

    let error = one_port_network(Complex64::new(f64::MAX, 0.0), Complex64::new(50.0, 0.0))
        .to_y_power()
        .expect_err("finite S overflow must be reported as computation context");
    assert_eq!(
        error,
        Error::NonFiniteComputation {
            stage: ConversionStage::SToZ,
            frequency: 0,
            row: 0,
            column: 0,
        }
    );
}

#[test]
fn public_error_vocabulary_retains_parameter_kind_for_shape_context() {
    let error = Error::InvalidShape {
        stage: ConversionStage::ZToY,
        parameter: ParameterKind::Z,
        shape: vec![2, 3, 4],
    };
    assert!(error.to_string().contains("Z→Y"));
    assert!(error.to_string().contains("Z-parameter"));
    assert!(error.to_string().contains("[2, 3, 4]"));
}
