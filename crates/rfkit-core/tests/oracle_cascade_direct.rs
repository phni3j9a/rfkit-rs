use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/power_wave_cascade_direct_four_port_complex_z0.json"
);
const EXPECTED_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_CASE_ID: &str = "power_wave_cascade_direct_four_port_complex_z0";
const EXPECTED_OPERATION: &str = "network_cascade_direct_power";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_RANDOM_SEED: u64 = 20_260_956;
const EXPECTED_NFREQ: usize = 3;
const EXPECTED_NPORTS: usize = 4;
const EXPECTED_RTOL: f64 = 1.0e-12;
const EXPECTED_ATOL: f64 = 1.0e-12;
const EXPECTED_FREQUENCY_HZ: [f64; EXPECTED_NFREQ] = [0.87e9, 1.61e9, 2.93e9];

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
    s_a: Vec<Vec<Vec<ComplexValue>>>,
    s_b: Vec<Vec<Vec<ComplexValue>>>,
    s_cascaded: Vec<Vec<Vec<ComplexValue>>>,
    z0_a_ohm: Vec<Vec<ComplexValue>>,
    z0_b_ohm: Vec<Vec<ComplexValue>>,
    z0_output_ohm: Vec<Vec<ComplexValue>>,
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
    tolerance_policy: TolerancePolicy,
    units: Units,
    wave_definition: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupConvention {
    description: String,
    input_a: Vec<String>,
    input_b: Vec<String>,
    output: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceImpedance {
    complex: bool,
    frequency_dependent: bool,
    per_port: bool,
    real_part: String,
    survivors_exact: bool,
    unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeclaredShape {
    frequency: Vec<usize>,
    input_s_a: Vec<usize>,
    input_s_b: Vec<usize>,
    input_z0_a: Vec<usize>,
    input_z0_b: Vec<usize>,
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
        "independent coupled four-port A/B S stacks from NumPy default_rng seed 20260956 with deterministic diagonal terms and full non-reciprocal within-group/cross-group perturbations; unequal complex frequency-dependent per-port references with strictly positive real parts; no prior fixture values reused"
    );
    assert_eq!(
        metadata.group_convention.description,
        "simultaneous right-to-left group junction with full S_ii coupling"
    );
    assert_eq!(
        metadata.group_convention.input_a,
        vec!["left_0", "left_1", "right_0", "right_1"]
    );
    assert_eq!(
        metadata.group_convention.input_b,
        vec!["left_0", "left_1", "right_0", "right_1"]
    );
    assert_eq!(
        metadata.group_convention.output,
        vec!["a_left_0", "a_left_1", "b_right_0", "b_right_1"]
    );
    assert_eq!(
        (
            metadata.reference_impedance.complex,
            metadata.reference_impedance.frequency_dependent,
            metadata.reference_impedance.per_port,
            metadata.reference_impedance.real_part.as_str(),
            metadata.reference_impedance.survivors_exact,
            metadata.reference_impedance.unit.as_str(),
        ),
        (true, true, true, "strictly positive", true, "ohm")
    );
    assert_eq!(metadata.shape.frequency, vec![EXPECTED_NFREQ]);
    assert_eq!(
        metadata.shape.input_s_a,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS, EXPECTED_NPORTS]
    );
    assert_eq!(
        metadata.shape.input_s_b,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS, EXPECTED_NPORTS]
    );
    assert_eq!(
        metadata.shape.input_z0_a,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS]
    );
    assert_eq!(
        metadata.shape.input_z0_b,
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
    assert!(metadata.tolerance_policy.comparison.contains("s_cascaded"));
    assert!(metadata.tolerance_policy.justification.contains("binary64"));
    assert!(
        metadata
            .tolerance_policy
            .regeneration
            .contains("Network.cascade")
    );
    assert!(
        metadata
            .tolerance_policy
            .regeneration
            .contains("renormalize")
    );
    assert_eq!(metadata.units.frequency, "Hz");
    assert_eq!(metadata.units.s, "dimensionless");
    assert_eq!(metadata.units.z0, "ohm");
}

#[test]
fn pinned_scikit_rf_cascade_fixture_matches_public_network_method() {
    let fixture: FixtureDocument = serde_json::from_str(FIXTURE_JSON).unwrap();
    validate_contract(&fixture.metadata);
    let source_frequency = fixture.data.frequency_hz.clone();
    assert_eq!(source_frequency.len(), EXPECTED_NFREQ);
    for (index, &frequency) in source_frequency.iter().enumerate() {
        assert_eq!(frequency.to_bits(), EXPECTED_FREQUENCY_HZ[index].to_bits());
    }
    let source_a = array3(&fixture.data.s_a);
    let source_b = array3(&fixture.data.s_b);
    let z0_a = array2(&fixture.data.z0_a_ohm);
    let z0_b = array2(&fixture.data.z0_b_ohm);
    let expected_z0 = array2(&fixture.data.z0_output_ohm);
    assert!(
        source_a
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(
        source_b
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(
        z0_a.iter()
            .all(|value| value.re.is_finite() && value.re > 0.0)
    );
    assert!(
        z0_b.iter()
            .all(|value| value.re.is_finite() && value.re > 0.0)
    );
    let expected_survivor_z0 =
        Array2::from_shape_fn((EXPECTED_NFREQ, EXPECTED_NPORTS), |(f, port)| {
            if port < EXPECTED_NPORTS / 2 {
                z0_a[[f, port]]
            } else {
                z0_b[[f, port]]
            }
        });

    let a = Network::new(
        Frequency::from_hz(source_frequency.clone()).unwrap(),
        source_a,
        z0_a,
    )
    .unwrap();
    let b = Network::new(
        Frequency::from_hz(source_frequency).unwrap(),
        source_b,
        z0_b,
    )
    .unwrap();
    let a_snapshot = a.clone();
    let b_snapshot = b.clone();
    let cascaded = a.cascade_direct_power(&b).unwrap();
    assert_eq!(cascaded.frequency(), a.frequency());
    assert_eq!(expected_z0, expected_survivor_z0);
    assert_eq!(cascaded.z0(), &expected_survivor_z0);
    for ((frequency, row, column), actual) in cascaded.s().indexed_iter() {
        assert_complex_close(
            *actual,
            &fixture.data.s_cascaded[frequency][row][column],
            &format!("s_cascaded[{frequency},{row},{column}]"),
        );
        assert!(actual.re.is_finite() && actual.im.is_finite());
    }
    assert_eq!(a, a_snapshot);
    assert_eq!(b, b_snapshot);
}
