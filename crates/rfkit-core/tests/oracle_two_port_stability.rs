use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str =
    include_str!("../../../tools/oracle/fixtures/two_port_stability_power_four_frequency.json");

const EXPECTED_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_CASE_ID: &str = "two_port_stability_power_four_frequency";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_OPERATION: &str = "network_two_port_stability_power";
const EXPECTED_RANDOM_SEED: u64 = 20_260_954;
const EXPECTED_RTOL: f64 = 1.0e-12;
const EXPECTED_ATOL: f64 = 1.0e-12;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureDocument {
    data: FixtureData,
    metadata: FixtureMetadata,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureData {
    delta: Vec<ComplexValue>,
    frequency_hz: Vec<f64>,
    rollet_k: Vec<f64>,
    s_input: Vec<Vec<Vec<ComplexValue>>>,
    z0_input_ohm: Vec<Vec<ComplexValue>>,
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
    input_recipe: String,
    numpy_version: String,
    operation: String,
    random_seed: u64,
    reference_impedance: ReferenceImpedance,
    sample_classes: Vec<String>,
    schema: String,
    schema_version: u32,
    scikit_rf_commit: String,
    scikit_rf_version: String,
    shape: DeclaredShape,
    tolerance_policy: TolerancePolicy,
    units: Units,
    wave_definition: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceImpedance {
    complex: bool,
    frequency_dependent: bool,
    per_port: bool,
    real_part: String,
    unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeclaredShape {
    frequency: Vec<usize>,
    input_s: Vec<usize>,
    input_z0: Vec<usize>,
    output_delta: Vec<usize>,
    output_rollet_k: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TolerancePolicy {
    atol: f64,
    comparison: String,
    justification: String,
    regeneration: String,
    rtol: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Units {
    delta: String,
    frequency: String,
    rollet_k: String,
    s: String,
    z0: String,
}

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
    assert_eq!(values.len(), 4);
    assert!(values.iter().all(|matrix| matrix.len() == 2));
    assert!(
        values
            .iter()
            .flat_map(|matrix| matrix.iter())
            .all(|row| row.len() == 2)
    );
    Array3::from_shape_fn((4, 2, 2), |(frequency, row, column)| {
        let value = &values[frequency][row][column];
        Complex64::new(value.real, value.imag)
    })
}

fn array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    assert_eq!(values.len(), 4);
    assert!(values.iter().all(|row| row.len() == 2));
    Array2::from_shape_fn((4, 2), |(frequency, port)| {
        let value = &values[frequency][port];
        Complex64::new(value.real, value.imag)
    })
}

fn largest_singular_value_2x2(source_s: &Array3<Complex64>, frequency: usize) -> f64 {
    let s11 = source_s[[frequency, 0, 0]];
    let s12 = source_s[[frequency, 0, 1]];
    let s21 = source_s[[frequency, 1, 0]];
    let s22 = source_s[[frequency, 1, 1]];
    let frobenius_squared = s11.norm_sqr() + s12.norm_sqr() + s21.norm_sqr() + s22.norm_sqr();
    let determinant = s11 * s22 - s12 * s21;
    let gram_discriminant =
        (frobenius_squared * frobenius_squared - 4.0 * determinant.norm_sqr()).max(0.0);
    (0.5 * (frobenius_squared + gram_discriminant.sqrt())).sqrt()
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    let difference = (actual - expected).abs();
    let bound = EXPECTED_ATOL + EXPECTED_RTOL * expected.abs();
    assert!(
        difference <= bound,
        "{context}: actual={actual:.17e}, expected={expected:.17e}, difference={difference:.17e}, bound={bound:.17e}"
    );
}

fn assert_complex_close(actual: Complex64, expected: &ComplexValue, context: &str) {
    assert_close(actual.re, expected.real, &format!("{context}.real"));
    assert_close(actual.im, expected.imag, &format!("{context}.imag"));
}

fn validate_fixture_contract(fixture: &FixtureDocument) {
    let metadata = &fixture.metadata;
    assert_eq!(metadata.schema, EXPECTED_SCHEMA);
    assert_eq!(metadata.schema_version, EXPECTED_SCHEMA_VERSION);
    assert_eq!(metadata.case_id, EXPECTED_CASE_ID);
    assert_eq!(metadata.numpy_version, EXPECTED_NUMPY_VERSION);
    assert_eq!(metadata.scikit_rf_version, EXPECTED_SCIKIT_RF_VERSION);
    assert_eq!(metadata.scikit_rf_commit, EXPECTED_SCIKIT_RF_COMMIT);
    assert_eq!(metadata.operation, EXPECTED_OPERATION);
    assert_eq!(metadata.random_seed, EXPECTED_RANDOM_SEED);
    assert_eq!(metadata.wave_definition, "power");
    assert_eq!(
        metadata.input_recipe,
        "independent four-frequency two-port base S stack plus deterministic complex perturbation from NumPy default_rng seed 20260954 at scale 1e-3; sample classes are derived from true largest singular values with one passive and three active/non-passive samples; unequal real/complex positive-real z0 varies by frequency and port; no undefined zero-transmission samples"
    );
    assert_eq!(
        metadata.sample_classes,
        vec!["passive", "active", "active", "active"]
    );
    assert_eq!(
        (
            metadata.reference_impedance.complex,
            metadata.reference_impedance.frequency_dependent,
            metadata.reference_impedance.per_port,
            metadata.reference_impedance.real_part.as_str(),
            metadata.reference_impedance.unit.as_str(),
        ),
        (true, true, true, "strictly positive", "ohm")
    );
    assert_eq!(metadata.shape.frequency, vec![4]);
    assert_eq!(metadata.shape.input_s, vec![4, 2, 2]);
    assert_eq!(metadata.shape.input_z0, vec![4, 2]);
    assert_eq!(metadata.shape.output_delta, vec![4]);
    assert_eq!(metadata.shape.output_rollet_k, vec![4]);
    assert_eq!(metadata.tolerance_policy.rtol, EXPECTED_RTOL);
    assert_eq!(metadata.tolerance_policy.atol, EXPECTED_ATOL);
    assert!(metadata.tolerance_policy.comparison.contains("delta"));
    assert!(metadata.tolerance_policy.comparison.contains("rollet_k"));
    assert!(metadata.tolerance_policy.justification.contains("binary64"));
    assert!(
        metadata
            .tolerance_policy
            .regeneration
            .contains("Network.stability")
    );
    assert_eq!(
        (
            metadata.units.delta.as_str(),
            metadata.units.frequency.as_str(),
            metadata.units.rollet_k.as_str(),
            metadata.units.s.as_str(),
            metadata.units.z0.as_str(),
        ),
        (
            "dimensionless",
            "Hz",
            "dimensionless",
            "dimensionless",
            "ohm"
        )
    );
}

#[test]
fn public_two_port_stability_matches_canonical_scikit_rf_fixture() {
    let fixture: FixtureDocument = serde_json::from_str(FIXTURE_JSON).unwrap();
    validate_fixture_contract(&fixture);
    assert_eq!(fixture.data.frequency_hz.len(), 4);
    assert_eq!(fixture.data.delta.len(), 4);
    assert_eq!(fixture.data.rollet_k.len(), 4);

    let source_s = array3(&fixture.data.s_input);
    let source_z0 = array2(&fixture.data.z0_input_ohm);
    let source_frequency = fixture.data.frequency_hz.clone();
    let network = Network::new(
        Frequency::from_hz(source_frequency.clone()).unwrap(),
        source_s.clone(),
        source_z0.clone(),
    )
    .unwrap();
    let snapshot = network.clone();
    let actual = network.two_port_stability_power().unwrap();
    assert_eq!(actual.len(), 4);

    for frequency in 0..4 {
        let s11 = source_s[[frequency, 0, 0]];
        let s12 = source_s[[frequency, 0, 1]];
        let s21 = source_s[[frequency, 1, 0]];
        let s22 = source_s[[frequency, 1, 1]];
        // Independently restate the determinant relation before trusting the
        // serialized expected output; NumPy generated this same value via
        // np.linalg.det in the pinned oracle environment.
        let expected_delta = s11 * s22 - s12 * s21;
        assert_complex_close(
            expected_delta,
            &fixture.data.delta[frequency],
            &format!("fixture delta[{frequency}]"),
        );
        assert_complex_close(
            actual[frequency].delta,
            &fixture.data.delta[frequency],
            &format!("output delta[{frequency}]"),
        );
        assert!(actual[frequency].rollet_k.is_some());
        assert_close(
            actual[frequency].rollet_k.unwrap(),
            fixture.data.rollet_k[frequency],
            &format!("output rollet_k[{frequency}]"),
        );

        let largest_singular_value = largest_singular_value_2x2(&source_s, frequency);
        match fixture.metadata.sample_classes[frequency].as_str() {
            "passive" => assert!(
                largest_singular_value < 1.0,
                "sample {frequency} should be genuinely passive: sigma_max={largest_singular_value}"
            ),
            "active" => assert!(
                largest_singular_value > 1.0,
                "sample {frequency} should be genuinely active/non-passive: sigma_max={largest_singular_value}"
            ),
            class => panic!("unexpected sample class {class:?}"),
        }
    }

    assert_eq!(network, snapshot);
    assert_eq!(network.frequency().hz(), source_frequency.as_slice());
    assert_eq!(network.s(), &source_s);
    assert_eq!(network.z0(), &source_z0);
}
