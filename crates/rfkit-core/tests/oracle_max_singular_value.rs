use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/max_singular_value_power_four_port_complex_z0.json"
);

const EXPECTED_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_CASE_ID: &str = "max_singular_value_power_four_port_complex_z0";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_OPERATION: &str = "network_max_singular_value_power";
const EXPECTED_RANDOM_SEED: u64 = 20_260_957;
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
    frequency_hz: Vec<f64>,
    is_passive: Vec<bool>,
    s_input: Vec<Vec<Vec<ComplexValue>>>,
    sigma_max: Vec<f64>,
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
    scikit_rf_is_passive_comparison: ScikitRfPassivityComparison,
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
struct ScikitRfPassivityComparison {
    classification_domain: String,
    method: String,
    tol: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeclaredShape {
    frequency: Vec<usize>,
    input_s: Vec<usize>,
    input_z0: Vec<usize>,
    output_is_passive: Vec<usize>,
    output_sigma_max: Vec<usize>,
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
    frequency: String,
    s: String,
    sigma_max: String,
    z0: String,
}

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
    assert_eq!(values.len(), 4);
    assert!(values.iter().all(|matrix| matrix.len() == 4));
    assert!(
        values
            .iter()
            .flat_map(|matrix| matrix.iter())
            .all(|row| row.len() == 4)
    );
    Array3::from_shape_fn((4, 4, 4), |(frequency, row, column)| {
        let value = &values[frequency][row][column];
        Complex64::new(value.real, value.imag)
    })
}

fn array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    assert_eq!(values.len(), 4);
    assert!(values.iter().all(|row| row.len() == 4));
    Array2::from_shape_fn((4, 4), |(frequency, port)| {
        let value = &values[frequency][port];
        Complex64::new(value.real, value.imag)
    })
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    let difference = (actual - expected).abs();
    let bound = EXPECTED_ATOL + EXPECTED_RTOL * expected.abs();
    assert!(
        difference <= bound,
        "{context}: actual={actual:.17e}, expected={expected:.17e}, difference={difference:.17e}, bound={bound:.17e}"
    );
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
        "independent four-frequency coupled non-reciprocal four-port S stack from NumPy default_rng seed 20260957; each sample is multiplied by fixed binary-exact factors [1.0, 2.0, 4.5, 5.0] (SVD is not used to construct exact inputs); unequal complex positive-real frequency-dependent per-port references; no prior fixture values reused"
    );
    assert_eq!(
        metadata.sample_classes,
        vec!["passive", "passive", "active", "active"]
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
    assert_eq!(metadata.shape.input_s, vec![4, 4, 4]);
    assert_eq!(metadata.shape.input_z0, vec![4, 4]);
    assert_eq!(metadata.shape.output_is_passive, vec![4]);
    assert_eq!(metadata.shape.output_sigma_max, vec![4]);
    assert_eq!(
        metadata.scikit_rf_is_passive_comparison.method,
        "Network.is_passive"
    );
    assert_eq!(metadata.scikit_rf_is_passive_comparison.tol, 1.0e-12);
    assert!(
        metadata
            .scikit_rf_is_passive_comparison
            .classification_domain
            .contains("comfortably away")
    );
    assert_eq!(metadata.tolerance_policy.rtol, EXPECTED_RTOL);
    assert_eq!(metadata.tolerance_policy.atol, EXPECTED_ATOL);
    assert!(metadata.tolerance_policy.comparison.contains("sigma_max"));
    assert!(metadata.tolerance_policy.comparison.contains("is_passive"));
    assert!(metadata.tolerance_policy.justification.contains("binary64"));
    assert!(
        metadata
            .tolerance_policy
            .regeneration
            .contains("numpy.linalg.svd")
    );
    assert!(
        metadata
            .tolerance_policy
            .regeneration
            .contains("Network.is_passive")
    );
    assert_eq!(
        (
            metadata.units.frequency.as_str(),
            metadata.units.s.as_str(),
            metadata.units.sigma_max.as_str(),
            metadata.units.z0.as_str(),
        ),
        (
            "Hz",
            "dimensionless",
            "dimensionless amplitude ratio",
            "ohm"
        )
    );
}

#[test]
fn public_max_singular_value_matches_numpy_fixture_and_pinned_boolean_evidence() {
    let fixture: FixtureDocument = serde_json::from_str(FIXTURE_JSON).unwrap();
    validate_fixture_contract(&fixture);
    assert_eq!(fixture.data.frequency_hz.len(), 4);
    assert_eq!(fixture.data.sigma_max.len(), 4);
    assert_eq!(fixture.data.is_passive.len(), 4);

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
    let actual = network.max_singular_value_power().unwrap();
    assert_eq!(actual.len(), 4);

    for (frequency, actual_sigma) in actual.iter().enumerate() {
        assert_close(
            *actual_sigma,
            fixture.data.sigma_max[frequency],
            &format!("sigma_max[{frequency}]"),
        );
        assert!(actual_sigma.is_finite());
        assert_eq!(
            fixture.data.is_passive[frequency],
            *actual_sigma < 1.0,
            "limited pinned is_passive comparison must agree away from one"
        );
        assert_eq!(
            fixture.data.is_passive[frequency],
            fixture.metadata.sample_classes[frequency] == "passive"
        );
    }

    assert_eq!(network, snapshot);
    assert_eq!(network.frequency().hz(), source_frequency.as_slice());
    assert_eq!(network.s(), &source_s);
    assert_eq!(network.z0(), &source_z0);
}
