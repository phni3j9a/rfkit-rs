use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str =
    include_str!("../../../tools/oracle/fixtures/port_permutation_three_port_complex_z0.json");

const EXPECTED_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_CASE_ID: &str = "port_permutation_three_port_complex_z0";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_OPERATION: &str = "network_port_permutation";
const EXPECTED_RANDOM_SEED: u64 = 20_260_947;
const EXPECTED_INPUT_RECIPE: &str = "independent local NumPy default_rng input with a non-symmetric complex three-port S stack and unequal complex per-port/frequency-dependent z0";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_WAVE_DEFINITION: &str = "power";
const EXPECTED_ORDER: &[usize] = &[2, 0, 1];
const EXPECTED_FREQUENCY_SHAPE: &[usize] = &[3];
const EXPECTED_S_SHAPE: &[usize] = &[3, 3, 3];
const EXPECTED_Z0_SHAPE: &[usize] = &[3, 3];
const EXPECTED_TOLERANCE_COMPARISON: &str = "exact canonical UTF-8 JSON bytes; frequency, S, and z0 are pure reindexing outputs and are copied exactly";
const EXPECTED_TOLERANCE_REGENERATION: &str = "exact canonical UTF-8 JSON bytes";

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
    port_order: PortOrder,
    random_seed: u64,
    reference_impedance: ReferenceImpedance,
    schema: String,
    schema_version: u32,
    scikit_rf_commit: String,
    scikit_rf_version: String,
    shape: DeclaredShape,
    tolerance_policy: TolerancePolicy,
    wave_definition: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortOrder {
    description: String,
    order_new_to_old: Vec<usize>,
    renumbered_from_ports: Vec<usize>,
    renumbered_to_ports: Vec<usize>,
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

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
    assert!(!values.is_empty());
    let nfreq = values.len();
    let nports = values[0].len();
    assert!(nports > 0);
    assert!(values.iter().all(|matrix| matrix.len() == nports));
    assert!(
        values
            .iter()
            .flat_map(|matrix| matrix.iter())
            .all(|row| row.len() == nports)
    );
    Array3::from_shape_fn((nfreq, nports, nports), |(frequency, row, column)| {
        let value = &values[frequency][row][column];
        Complex64::new(value.real, value.imag)
    })
}

fn array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    assert!(!values.is_empty());
    let nfreq = values.len();
    let nports = values[0].len();
    assert!(nports > 0);
    assert!(values.iter().all(|row| row.len() == nports));
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

fn validate_fixture_contract(fixture: &FixtureDocument) {
    let metadata = &fixture.metadata;
    assert_eq!(metadata.schema, EXPECTED_SCHEMA);
    assert_eq!(metadata.schema_version, EXPECTED_SCHEMA_VERSION);
    assert_eq!(metadata.case_id, EXPECTED_CASE_ID);
    assert_eq!(metadata.numpy_version, EXPECTED_NUMPY_VERSION);
    assert_eq!(metadata.operation, EXPECTED_OPERATION);
    assert_eq!(metadata.random_seed, EXPECTED_RANDOM_SEED);
    assert_eq!(metadata.scikit_rf_commit, EXPECTED_SCIKIT_RF_COMMIT);
    assert_eq!(metadata.scikit_rf_version, EXPECTED_SCIKIT_RF_VERSION);
    assert_eq!(metadata.wave_definition, EXPECTED_WAVE_DEFINITION);
    assert_eq!(metadata.input_recipe, EXPECTED_INPUT_RECIPE);

    assert_eq!(
        metadata.port_order.description,
        "order[new_port] = old_port"
    );
    assert_eq!(metadata.port_order.order_new_to_old, EXPECTED_ORDER);
    assert_eq!(metadata.port_order.renumbered_from_ports, EXPECTED_ORDER);
    assert_eq!(metadata.port_order.renumbered_to_ports, vec![0, 1, 2]);
    assert_eq!(
        metadata
            .port_order
            .source_mapping
            .iter()
            .map(|mapping| (mapping.new_port, mapping.old_port))
            .collect::<Vec<_>>(),
        vec![(0, 2), (1, 0), (2, 1)]
    );

    assert!(metadata.reference_impedance.complex);
    assert!(metadata.reference_impedance.frequency_dependent);
    assert!(metadata.reference_impedance.per_port);
    assert_eq!(metadata.reference_impedance.unit, "ohm");

    assert_eq!(metadata.shape.frequency, EXPECTED_FREQUENCY_SHAPE);
    assert_eq!(metadata.shape.input_s, EXPECTED_S_SHAPE);
    assert_eq!(metadata.shape.input_z0, EXPECTED_Z0_SHAPE);
    assert_eq!(metadata.shape.output_s, EXPECTED_S_SHAPE);
    assert_eq!(metadata.shape.output_z0, EXPECTED_Z0_SHAPE);
    assert_eq!(
        metadata.tolerance_policy.comparison,
        EXPECTED_TOLERANCE_COMPARISON
    );
    assert_eq!(
        metadata.tolerance_policy.regeneration,
        EXPECTED_TOLERANCE_REGENERATION
    );

    let data = &fixture.data;
    assert_eq!(data.frequency_hz.len(), EXPECTED_FREQUENCY_SHAPE[0]);
    assert_eq!(data.s_input.len(), EXPECTED_S_SHAPE[0]);
    assert_eq!(data.s.len(), EXPECTED_S_SHAPE[0]);
    assert_eq!(data.z0_input_ohm.len(), EXPECTED_Z0_SHAPE[0]);
    assert_eq!(data.z0_ohm.len(), EXPECTED_Z0_SHAPE[0]);
    for frequency in 0..EXPECTED_FREQUENCY_SHAPE[0] {
        assert_eq!(data.s_input[frequency].len(), EXPECTED_S_SHAPE[1]);
        assert_eq!(data.s[frequency].len(), EXPECTED_S_SHAPE[1]);
        assert_eq!(data.z0_input_ohm[frequency].len(), EXPECTED_Z0_SHAPE[1]);
        assert_eq!(data.z0_ohm[frequency].len(), EXPECTED_Z0_SHAPE[1]);
        for row in 0..EXPECTED_S_SHAPE[1] {
            assert_eq!(data.s_input[frequency][row].len(), EXPECTED_S_SHAPE[2]);
            assert_eq!(data.s[frequency][row].len(), EXPECTED_S_SHAPE[2]);
        }
    }
}

#[test]
fn public_port_permutation_matches_pinned_scikit_rf_fixture_exactly() {
    let fixture: FixtureDocument = serde_json::from_str(FIXTURE_JSON).unwrap();
    validate_fixture_contract(&fixture);

    let source_s = array3(&fixture.data.s_input);
    let source_z0 = array2(&fixture.data.z0_input_ohm);
    let expected_s = array3(&fixture.data.s);
    let expected_z0 = array2(&fixture.data.z0_ohm);
    let frequency = Frequency::from_hz(fixture.data.frequency_hz.clone()).unwrap();
    let network = Network::new(frequency, source_s.clone(), source_z0.clone()).unwrap();

    // Verify the serialized oracle itself is an independently indexed copy
    // before invoking the Rust operation, so a transposed or inverse fixture
    // cannot mask an implementation error.
    for frequency in 0..EXPECTED_FREQUENCY_SHAPE[0] {
        for new_row in 0..EXPECTED_ORDER.len() {
            let old_row = EXPECTED_ORDER[new_row];
            assert_complex_bits(
                expected_z0[[frequency, new_row]],
                &fixture.data.z0_input_ohm[frequency][old_row],
                &format!("fixture z0[{frequency},{new_row}]"),
            );
            for new_column in 0..EXPECTED_ORDER.len() {
                let old_column = EXPECTED_ORDER[new_column];
                assert_complex_bits(
                    expected_s[[frequency, new_row, new_column]],
                    &fixture.data.s_input[frequency][old_row][old_column],
                    &format!("fixture S[{frequency},{new_row},{new_column}]"),
                );
            }
        }
    }

    let permuted = network.permute_ports(EXPECTED_ORDER).unwrap();
    assert_eq!(permuted.nports(), EXPECTED_ORDER.len());
    assert_eq!(permuted.frequency().len(), EXPECTED_FREQUENCY_SHAPE[0]);

    for (frequency, expected_frequency) in fixture.data.frequency_hz.iter().enumerate() {
        assert_f64_bits(
            permuted.frequency().hz()[frequency],
            *expected_frequency,
            &format!("output frequency[{frequency}]"),
        );
    }
    for frequency in 0..EXPECTED_FREQUENCY_SHAPE[0] {
        for row in 0..EXPECTED_ORDER.len() {
            assert_complex_bits(
                permuted.z0()[[frequency, row]],
                &fixture.data.z0_ohm[frequency][row],
                &format!("output z0[{frequency},{row}]"),
            );
            for column in 0..EXPECTED_ORDER.len() {
                assert_complex_bits(
                    permuted.s()[[frequency, row, column]],
                    &fixture.data.s[frequency][row][column],
                    &format!("output S[{frequency},{row},{column}]"),
                );
            }
        }
    }

    // The borrowed input remains unchanged after the operation.
    for frequency in 0..EXPECTED_FREQUENCY_SHAPE[0] {
        for row in 0..EXPECTED_ORDER.len() {
            assert_complex_bits(
                network.z0()[[frequency, row]],
                &fixture.data.z0_input_ohm[frequency][row],
                &format!("input z0[{frequency},{row}]"),
            );
            for column in 0..EXPECTED_ORDER.len() {
                assert_complex_bits(
                    network.s()[[frequency, row, column]],
                    &fixture.data.s_input[frequency][row][column],
                    &format!("input S[{frequency},{row},{column}]"),
                );
            }
        }
    }
}
