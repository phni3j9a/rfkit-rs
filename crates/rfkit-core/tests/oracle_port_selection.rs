use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/port_selection_zero_incident_five_port_complex_z0.json"
);

const EXPECTED_CASE_ID: &str = "port_selection_zero_incident_five_port_complex_z0";
const EXPECTED_OPERATION: &str = "network_select_ports_zero_incident";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_RANDOM_SEED: u64 = 20_260_964;
const EXPECTED_PORTS: &[usize] = &[4, 1, 3];

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
    s_input: Vec<Vec<Vec<ComplexValue>>>,
    z0_input_ohm: Vec<Vec<ComplexValue>>,
    z0_ohm: Vec<Vec<ComplexValue>>,
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
    port_selection: PortSelection,
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
struct PortSelection {
    boundary: String,
    description: String,
    omitted_ports: Vec<usize>,
    ports_new_to_old: Vec<usize>,
    retained_ports: Vec<usize>,
    source_mapping: Vec<SourceMapping>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceMapping {
    new_port: usize,
    old_port: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceImpedance {
    complex: bool,
    frequency_dependent: bool,
    per_port: bool,
    real_positive: bool,
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
    comparison: String,
    regeneration: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Units {
    frequency: String,
    s: String,
    z0: String,
}

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
    let nfreq = values.len();
    let nports = values[0].len();
    Array3::from_shape_fn((nfreq, nports, nports), |(frequency, row, column)| {
        let value = &values[frequency][row][column];
        Complex64::new(value.real, value.imag)
    })
}

fn array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    let nfreq = values.len();
    let nports = values[0].len();
    Array2::from_shape_fn((nfreq, nports), |(frequency, port)| {
        let value = &values[frequency][port];
        Complex64::new(value.real, value.imag)
    })
}

fn assert_f64_bits(actual: f64, expected: f64, context: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{context}: expected exact binary64 copy, actual={actual:?}, expected={expected:?}"
    );
}

fn assert_complex_bits(actual: Complex64, expected: &ComplexValue, context: &str) {
    assert_f64_bits(actual.re, expected.real, &format!("{context}.real"));
    assert_f64_bits(actual.im, expected.imag, &format!("{context}.imag"));
}

#[test]
fn ordered_zero_incident_subset_matches_pinned_scikit_rf_fixture_exactly() {
    let fixture: FixtureDocument = serde_json::from_str(FIXTURE_JSON).unwrap();
    let metadata = &fixture.metadata;
    assert_eq!(metadata.schema, "rfkit-rs.oracle.fixture");
    assert_eq!(metadata.schema_version, 1);
    assert_eq!(metadata.case_id, EXPECTED_CASE_ID);
    assert_eq!(metadata.operation, EXPECTED_OPERATION);
    assert_eq!(metadata.numpy_version, EXPECTED_NUMPY_VERSION);
    assert_eq!(metadata.scikit_rf_version, EXPECTED_SCIKIT_RF_VERSION);
    assert_eq!(metadata.scikit_rf_commit, EXPECTED_SCIKIT_RF_COMMIT);
    assert_eq!(metadata.random_seed, EXPECTED_RANDOM_SEED);
    assert_eq!(metadata.wave_definition, "power");
    assert_eq!(
        metadata.input_recipe,
        "independent local NumPy default_rng input with seed 20260964, an asymmetric complex five-port S stack, and unequal complex frequency-dependent positive-real z0"
    );
    assert_eq!(
        metadata.port_selection.description,
        "ports[new_port] = old_port"
    );
    assert_eq!(metadata.port_selection.ports_new_to_old, EXPECTED_PORTS);
    assert_eq!(metadata.port_selection.retained_ports, EXPECTED_PORTS);
    assert_eq!(metadata.port_selection.omitted_ports, vec![0, 2]);
    assert_eq!(metadata.port_selection.boundary, "a_R=0; b_E=S_EE*a_E");
    assert_eq!(
        metadata
            .port_selection
            .source_mapping
            .iter()
            .map(|mapping| (mapping.new_port, mapping.old_port))
            .collect::<Vec<_>>(),
        vec![(0, 4), (1, 1), (2, 3)]
    );
    assert!(metadata.reference_impedance.complex);
    assert!(metadata.reference_impedance.frequency_dependent);
    assert!(metadata.reference_impedance.per_port);
    assert!(metadata.reference_impedance.real_positive);
    assert_eq!(metadata.reference_impedance.unit, "ohm");
    assert_eq!(metadata.units.frequency, "Hz");
    assert_eq!(metadata.units.s, "dimensionless");
    assert_eq!(metadata.units.z0, "ohm");
    assert_eq!(metadata.shape.frequency, vec![3]);
    assert_eq!(metadata.shape.input_s, vec![3, 5, 5]);
    assert_eq!(metadata.shape.input_z0, vec![3, 5]);
    assert_eq!(metadata.shape.output_s, vec![3, 3, 3]);
    assert_eq!(metadata.shape.output_z0, vec![3, 3]);
    assert_eq!(
        metadata.tolerance_policy.comparison,
        "exact canonical UTF-8 JSON bytes; frequency, S, and z0 are pure ordered coordinate copies under a_R=0 and require no arithmetic tolerance"
    );
    assert_eq!(
        metadata.tolerance_policy.regeneration,
        "exact canonical UTF-8 JSON bytes"
    );

    let source_s = array3(&fixture.data.s_input);
    let source_z0 = array2(&fixture.data.z0_input_ohm);
    let frequency = Frequency::from_hz(fixture.data.frequency_hz.clone()).unwrap();
    let network = Network::new(frequency, source_s, source_z0).unwrap();
    let selected = network.select_ports_zero_incident(EXPECTED_PORTS).unwrap();

    for (frequency, expected_frequency) in fixture.data.frequency_hz.iter().enumerate() {
        assert_f64_bits(
            selected.frequency().hz()[frequency],
            *expected_frequency,
            &format!("frequency[{frequency}]"),
        );
        for (new_port, &old_port) in EXPECTED_PORTS.iter().enumerate() {
            assert_complex_bits(
                selected.z0()[[frequency, new_port]],
                &fixture.data.z0_ohm[frequency][new_port],
                &format!("output z0[{frequency},{new_port}]"),
            );
            assert_complex_bits(
                selected.z0()[[frequency, new_port]],
                &fixture.data.z0_input_ohm[frequency][old_port],
                &format!("mapped z0[{frequency},{new_port}]"),
            );
            for (new_column, &old_column) in EXPECTED_PORTS.iter().enumerate() {
                assert_complex_bits(
                    selected.s()[[frequency, new_port, new_column]],
                    &fixture.data.s[frequency][new_port][new_column],
                    &format!("output S[{frequency},{new_port},{new_column}]"),
                );
                assert_complex_bits(
                    selected.s()[[frequency, new_port, new_column]],
                    &fixture.data.s_input[frequency][old_port][old_column],
                    &format!("mapped S[{frequency},{new_port},{new_column}]"),
                );
            }
        }
    }
}
