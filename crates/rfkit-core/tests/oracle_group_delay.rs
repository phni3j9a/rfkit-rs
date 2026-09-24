use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/group_delay_secant_power_three_port_branch_crossing.json"
);
const EXPECTED_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_CASE_ID: &str = "group_delay_secant_power_three_port_branch_crossing";
const EXPECTED_OPERATION: &str = "network_group_delay_secant_power";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_RANDOM_SEED: u64 = 20_260_958;
const EXPECTED_NFREQ: usize = 7;
const EXPECTED_NPORTS: usize = 3;
const EXPECTED_PORT_OUT: usize = 2;
const EXPECTED_PORT_IN: usize = 0;
const EXPECTED_FREQUENCY_HZ: [f64; EXPECTED_NFREQ] =
    [0.0, 31.0e6, 80.0e6, 143.0e6, 225.0e6, 320.0e6, 429.0e6];
const EXPECTED_RTOL: f64 = 1.0e-12;
const EXPECTED_ATOL: f64 = 1.0e-21;

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
    group_delay_seconds: Vec<f64>,
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
    phase_definition: String,
    random_seed: u64,
    reference_impedance: ReferenceImpedance,
    sampling_limit: String,
    schema: String,
    schema_version: u32,
    scikit_rf_commit: String,
    scikit_rf_version: String,
    selected_entry: SelectedEntry,
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
struct SelectedEntry {
    label: String,
    port_in_zero_based: usize,
    port_out_zero_based: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeclaredShape {
    frequency: Vec<usize>,
    input_s: Vec<usize>,
    input_z0: Vec<usize>,
    output_delay_seconds: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TolerancePolicy {
    atol_s: f64,
    comparison: String,
    justification: String,
    regeneration: String,
    rtol: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Units {
    frequency: String,
    group_delay: String,
    phase: String,
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
        |(frequency, row, column)| complex(&values[frequency][row][column]),
    )
}

fn array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    assert_eq!(values.len(), EXPECTED_NFREQ);
    assert!(values.iter().all(|row| row.len() == EXPECTED_NPORTS));
    Array2::from_shape_fn((EXPECTED_NFREQ, EXPECTED_NPORTS), |(frequency, port)| {
        complex(&values[frequency][port])
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

fn validate_contract(fixture: &FixtureDocument) {
    let metadata = &fixture.metadata;
    assert_eq!(metadata.schema, EXPECTED_SCHEMA);
    assert_eq!(metadata.schema_version, EXPECTED_SCHEMA_VERSION);
    assert_eq!(metadata.case_id, EXPECTED_CASE_ID);
    assert_eq!(metadata.operation, EXPECTED_OPERATION);
    assert_eq!(metadata.numpy_version, EXPECTED_NUMPY_VERSION);
    assert_eq!(metadata.scikit_rf_version, EXPECTED_SCIKIT_RF_VERSION);
    assert_eq!(metadata.scikit_rf_commit, EXPECTED_SCIKIT_RF_COMMIT);
    assert_eq!(metadata.random_seed, EXPECTED_RANDOM_SEED);
    assert_eq!(metadata.wave_definition, "power");
    assert_eq!(metadata.selected_entry.label, "S31");
    assert_eq!(metadata.selected_entry.port_in_zero_based, EXPECTED_PORT_IN);
    assert_eq!(
        metadata.selected_entry.port_out_zero_based,
        EXPECTED_PORT_OUT
    );
    assert_eq!(
        metadata.input_recipe,
        "independent asymmetric three-port input from NumPy default_rng seed 20260958; nonuniform frequencies [0,31,80,143,225,320,429] MHz; selected S31 has positive varying amplitude and phase -2.9-2*pi*(1.3e-9*f+0.6e-18*f^2) with one principal-branch crossing; complex unequal per-port frequency-dependent references"
    );
    assert!(metadata.phase_definition.contains("s_rad_unwrap"));
    assert!(metadata.phase_definition.contains("adjacent interval"));
    assert!(metadata.sampling_limit.contains("magnitude >= pi"));
    assert!(metadata.sampling_limit.contains("exact half-turn"));
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
        metadata.shape.output_delay_seconds,
        vec![EXPECTED_NFREQ - 1]
    );
    assert_eq!(metadata.tolerance_policy.rtol, EXPECTED_RTOL);
    assert_eq!(metadata.tolerance_policy.atol_s, EXPECTED_ATOL);
    assert!(
        metadata
            .tolerance_policy
            .comparison
            .contains("group_delay_seconds")
    );
    assert!(
        metadata
            .tolerance_policy
            .justification
            .contains("seconds-scale")
    );
    assert!(
        metadata
            .tolerance_policy
            .regeneration
            .contains("s_rad_unwrap")
    );
    assert_eq!(metadata.units.frequency, "Hz");
    assert_eq!(metadata.units.group_delay, "s");
    assert_eq!(metadata.units.phase, "rad");
    assert_eq!(metadata.units.s, "dimensionless");
    assert_eq!(metadata.units.z0, "ohm");
}

#[test]
fn public_group_delay_matches_pinned_unwrapped_interval_fixture() {
    let fixture: FixtureDocument = serde_json::from_str(FIXTURE_JSON).unwrap();
    validate_contract(&fixture);
    assert_eq!(fixture.data.frequency_hz.len(), EXPECTED_NFREQ);
    assert_eq!(fixture.data.group_delay_seconds.len(), EXPECTED_NFREQ - 1);
    for (index, (&actual, &expected)) in fixture
        .data
        .frequency_hz
        .iter()
        .zip(EXPECTED_FREQUENCY_HZ.iter())
        .enumerate()
    {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "frequency_hz[{index}]"
        );
    }

    let source_frequency = fixture.data.frequency_hz.clone();
    let source_s = array3(&fixture.data.s_input);
    let source_z0 = array2(&fixture.data.z0_input_ohm);
    let network = Network::new(
        Frequency::from_hz(source_frequency.clone()).unwrap(),
        source_s.clone(),
        source_z0.clone(),
    )
    .unwrap();
    let snapshot = network.clone();
    let actual = network
        .group_delay_secant_power(EXPECTED_PORT_OUT, EXPECTED_PORT_IN)
        .unwrap();
    assert_eq!(actual.len(), EXPECTED_NFREQ - 1);
    for (interval, (&actual, &expected)) in actual
        .iter()
        .zip(fixture.data.group_delay_seconds.iter())
        .enumerate()
    {
        assert_close(
            actual,
            expected,
            &format!("group_delay_seconds[{interval}]"),
        );
        assert!(actual.is_finite());
    }
    assert_eq!(network, snapshot);
    assert_eq!(network.frequency().hz(), source_frequency.as_slice());
    assert_eq!(network.s(), &source_s);
    assert_eq!(network.z0(), &source_z0);
}
