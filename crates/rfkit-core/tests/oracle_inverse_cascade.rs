use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/power_wave_inverse_cascade_four_port_real_unequal_z0.json"
);
const EXPECTED_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_CASE_ID: &str = "power_wave_inverse_cascade_four_port_real_unequal_z0";
const EXPECTED_OPERATION: &str = "network_inverse_cascade_power";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_RANDOM_SEED: u64 = 20_260_955;
const EXPECTED_NFREQ: usize = 3;
const EXPECTED_NPORTS: usize = 4;
const EXPECTED_RTOL: f64 = 1.0e-12;
const EXPECTED_ATOL: f64 = 1.0e-12;
const EXPECTED_FREQUENCY_HZ: [f64; EXPECTED_NFREQ] = [0.91e9, 1.73e9, 2.87e9];

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
    s_input: Vec<Vec<Vec<ComplexValue>>>,
    s_inverse: Vec<Vec<Vec<ComplexValue>>>,
    z0_input_ohm: Vec<Vec<ComplexValue>>,
    z0_inverse_ohm: Vec<Vec<ComplexValue>>,
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
    determinant_evidence: DeterminantEvidence,
    group_convention: GroupConvention,
    input_recipe: String,
    numpy_version: String,
    operation: String,
    random_seed: u64,
    reference_impedance: ReferenceImpedance,
    schema: String,
    schema_version: u32,
    scikit_rf_commit: String,
    scikit_rf_version: String,
    shape: DeclaredShape,
    solve_difference_max_abs: f64,
    tolerance_policy: TolerancePolicy,
    units: Units,
    wave_definition: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeterminantEvidence {
    forward_transmission_abs: Vec<f64>,
    full_s_abs: Vec<f64>,
    reverse_transmission_abs: Vec<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupConvention {
    description: String,
    input: Vec<String>,
    output: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceImpedance {
    complex: bool,
    frequency_dependent: bool,
    per_port: bool,
    real_part: String,
    swapped_output: bool,
    unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeclaredShape {
    frequency: Vec<usize>,
    input_s: Vec<usize>,
    input_z0: Vec<usize>,
    output_s: Vec<usize>,
    output_z0: Vec<usize>,
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
    z0: String,
}

fn complex(value: &ComplexValue) -> Complex64 {
    Complex64::new(value.real, value.imag)
}

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
    assert_eq!(values.len(), EXPECTED_NFREQ);
    assert!(values.iter().all(|matrix| matrix.len() == EXPECTED_NPORTS));
    assert!(
        values
            .iter()
            .flat_map(|matrix| matrix.iter())
            .all(|row| row.len() == EXPECTED_NPORTS)
    );
    Array3::from_shape_fn(
        (EXPECTED_NFREQ, EXPECTED_NPORTS, EXPECTED_NPORTS),
        |(f, row, column)| complex(&values[f][row][column]),
    )
}

fn array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    assert_eq!(values.len(), EXPECTED_NFREQ);
    assert!(values.iter().all(|row| row.len() == EXPECTED_NPORTS));
    Array2::from_shape_fn((EXPECTED_NFREQ, EXPECTED_NPORTS), |(f, port)| {
        complex(&values[f][port])
    })
}

fn assert_bits(actual: f64, expected: f64, context: &str) {
    assert_eq!(actual.to_bits(), expected.to_bits(), "{context}");
}

fn assert_complex_close(actual: Complex64, expected: &ComplexValue, context: &str) {
    let expected = complex(expected);
    let difference = (actual - expected).norm();
    let bound = EXPECTED_ATOL + EXPECTED_RTOL * expected.norm();
    assert!(
        difference <= bound,
        "{context}: actual={actual:?}, expected={expected:?}, difference={difference:e}, bound={bound:e}"
    );
}

fn validate_contract(metadata: &FixtureMetadata) {
    assert_eq!(metadata.schema, EXPECTED_SCHEMA);
    assert_eq!(metadata.schema_version, EXPECTED_SCHEMA_VERSION);
    assert_eq!(metadata.case_id, EXPECTED_CASE_ID);
    assert_eq!(metadata.operation, EXPECTED_OPERATION);
    assert_eq!(metadata.numpy_version, EXPECTED_NUMPY_VERSION);
    assert_eq!(metadata.scikit_rf_version, EXPECTED_SCIKIT_RF_VERSION);
    assert_eq!(metadata.scikit_rf_commit, EXPECTED_SCIKIT_RF_COMMIT);
    assert_eq!(metadata.random_seed, EXPECTED_RANDOM_SEED);
    assert_eq!(metadata.wave_definition, "power");
    assert_eq!(
        metadata.input_recipe,
        "independent multi-frequency four-port complex S stack from NumPy default_rng seed 20260955, modest random non-reciprocal perturbations with deterministic diagonal terms, and unequal real-positive frequency-dependent per-port references; no prior fixture values reused"
    );
    assert_eq!(
        metadata.group_convention.description,
        "P solve(S,I) P with fixed equal ordered groups"
    );
    assert_eq!(
        metadata.group_convention.input,
        vec!["left_0", "left_1", "right_0", "right_1"]
    );
    assert_eq!(
        metadata.group_convention.output,
        vec!["old_right_0", "old_right_1", "old_left_0", "old_left_1"]
    );
    assert_eq!(
        (
            metadata.reference_impedance.complex,
            metadata.reference_impedance.frequency_dependent,
            metadata.reference_impedance.per_port,
            metadata.reference_impedance.real_part.as_str(),
            metadata.reference_impedance.swapped_output,
            metadata.reference_impedance.unit.as_str(),
        ),
        (false, true, true, "strictly positive", true, "ohm")
    );
    assert_eq!(metadata.shape.frequency, vec![EXPECTED_NFREQ]);
    assert_eq!(
        metadata.shape.input_s,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS, EXPECTED_NPORTS]
    );
    assert_eq!(
        metadata.shape.input_z0,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS]
    );
    assert_eq!(
        metadata.shape.output_s,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS, EXPECTED_NPORTS]
    );
    assert_eq!(
        metadata.shape.output_z0,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS]
    );
    assert_eq!(metadata.tolerance_policy.rtol, EXPECTED_RTOL);
    assert_eq!(metadata.tolerance_policy.atol, EXPECTED_ATOL);
    assert!(metadata.tolerance_policy.comparison.contains("s_inverse"));
    assert!(metadata.tolerance_policy.justification.contains("binary64"));
    assert!(
        metadata
            .tolerance_policy
            .regeneration
            .contains("Network.inv")
    );
    assert_eq!(metadata.units.frequency, "Hz");
    assert_eq!(metadata.units.s, "dimensionless");
    assert_eq!(metadata.units.z0, "ohm");
    assert_eq!(
        metadata.determinant_evidence.full_s_abs.len(),
        EXPECTED_NFREQ
    );
    assert_eq!(
        metadata.determinant_evidence.forward_transmission_abs.len(),
        EXPECTED_NFREQ
    );
    assert_eq!(
        metadata.determinant_evidence.reverse_transmission_abs.len(),
        EXPECTED_NFREQ
    );
    assert!(metadata.solve_difference_max_abs <= EXPECTED_ATOL);
}

#[test]
fn pinned_scikit_rf_inverse_fixture_matches_public_network_method() {
    let fixture: FixtureDocument = serde_json::from_str(FIXTURE_JSON).unwrap();
    validate_contract(&fixture.metadata);
    assert_eq!(fixture.data.frequency_hz.len(), EXPECTED_NFREQ);
    for (index, &frequency) in fixture.data.frequency_hz.iter().enumerate() {
        assert_bits(
            frequency,
            EXPECTED_FREQUENCY_HZ[index],
            &format!("frequency[{index}]"),
        );
    }
    let source_s = array3(&fixture.data.s_input);
    let expected_s = &fixture.data.s_inverse;
    let source_z0 = array2(&fixture.data.z0_input_ohm);
    let expected_z0 = array2(&fixture.data.z0_inverse_ohm);
    assert!(
        source_s
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(
        source_z0
            .iter()
            .all(|value| value.re.is_finite() && value.re > 0.0 && value.im == 0.0)
    );
    assert!(
        expected_z0
            .iter()
            .all(|value| value.re.is_finite() && value.re > 0.0 && value.im == 0.0)
    );

    let network = Network::new(
        Frequency::from_hz(fixture.data.frequency_hz.clone()).unwrap(),
        source_s.clone(),
        source_z0.clone(),
    )
    .unwrap();
    let snapshot = network.clone();
    let inverse = network.inverse_cascade_power().unwrap();
    assert_eq!(
        inverse.frequency().hz(),
        fixture.data.frequency_hz.as_slice()
    );
    assert_eq!(inverse.z0(), &expected_z0);
    for ((frequency, row, column), actual) in inverse.s().indexed_iter() {
        assert_complex_close(
            *actual,
            &expected_s[frequency][row][column],
            &format!("s_inverse[{frequency},{row},{column}]"),
        );
        assert!(actual.re.is_finite() && actual.im.is_finite());
    }

    // The exact source references and data remain borrowed/unchanged, while
    // the output reference contract is independently restated as P*z0.
    assert_eq!(network, snapshot);
    let expected_swapped = Array2::from_shape_fn((EXPECTED_NFREQ, EXPECTED_NPORTS), |(f, port)| {
        source_z0[[f, if port < 2 { port + 2 } else { port - 2 }]]
    });
    assert_eq!(inverse.z0(), &expected_swapped);
}
