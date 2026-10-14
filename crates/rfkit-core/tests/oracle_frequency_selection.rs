use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/frequency_sample_selection_three_port_complex_z0.json"
);

const EXPECTED_CASE_ID: &str = "frequency_sample_selection_three_port_complex_z0";
const EXPECTED_OPERATION: &str = "network_select_frequency_samples";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_RANDOM_SEED: u64 = 20_260_967;
const EXPECTED_SAMPLES: &[usize] = &[3, 1];

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
    frequency_input_hz: Vec<f64>,
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
    frequency_selection: FrequencySelection,
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
struct FrequencySelection {
    description: String,
    samples_new_to_old: Vec<usize>,
    source_mapping: Vec<SourceMapping>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceMapping {
    new_sample: usize,
    old_sample: usize,
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
fn ordered_frequency_selection_matches_pinned_scikit_rf_integer_array_fixture_exactly() {
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
        "independent local NumPy default_rng input with seed 20260967, an asymmetric complex three-port S stack, four irregular frequency samples, and unequal complex frequency-dependent per-port z0"
    );
    assert_eq!(
        metadata.frequency_selection.description,
        "samples[new_sample] = old_sample"
    );
    assert_eq!(
        metadata.frequency_selection.samples_new_to_old,
        EXPECTED_SAMPLES
    );
    assert_eq!(
        metadata
            .frequency_selection
            .source_mapping
            .iter()
            .map(|mapping| (mapping.new_sample, mapping.old_sample))
            .collect::<Vec<_>>(),
        vec![(0, 3), (1, 1)]
    );
    assert!(metadata.reference_impedance.complex);
    assert!(metadata.reference_impedance.frequency_dependent);
    assert!(metadata.reference_impedance.per_port);
    assert_eq!(metadata.reference_impedance.unit, "ohm");
    assert_eq!(metadata.units.frequency, "Hz");
    assert_eq!(metadata.units.s, "dimensionless");
    assert_eq!(metadata.units.z0, "ohm");
    assert_eq!(metadata.shape.frequency, vec![4]);
    assert_eq!(metadata.shape.input_s, vec![4, 3, 3]);
    assert_eq!(metadata.shape.input_z0, vec![4, 3]);
    assert_eq!(metadata.shape.output_s, vec![2, 3, 3]);
    assert_eq!(metadata.shape.output_z0, vec![2, 3]);
    assert_eq!(
        metadata.tolerance_policy.comparison,
        "exact canonical UTF-8 JSON bytes; frequency, S, and z0 are pure ordered observation copies and require no arithmetic tolerance"
    );
    assert_eq!(
        metadata.tolerance_policy.regeneration,
        "exact canonical UTF-8 JSON bytes"
    );

    let source_s = array3(&fixture.data.s_input);
    let source_z0 = array2(&fixture.data.z0_input_ohm);
    let source = Network::new(
        Frequency::from_hz(fixture.data.frequency_input_hz.clone()).unwrap(),
        source_s,
        source_z0,
    )
    .unwrap();
    let selected = source.select_frequency_samples(EXPECTED_SAMPLES).unwrap();

    assert_eq!(selected.frequency().len(), fixture.data.frequency_hz.len());
    for (new_frequency, expected_frequency) in fixture.data.frequency_hz.iter().enumerate() {
        assert_f64_bits(
            selected.frequency().hz()[new_frequency],
            *expected_frequency,
            &format!("frequency[{new_frequency}]"),
        );
        for row in 0..3 {
            for column in 0..3 {
                assert_complex_bits(
                    selected.s()[[new_frequency, row, column]],
                    &fixture.data.s[new_frequency][row][column],
                    &format!("S[{new_frequency},{row},{column}]"),
                );
            }
            assert_complex_bits(
                selected.z0()[[new_frequency, row]],
                &fixture.data.z0_ohm[new_frequency][row],
                &format!("z0[{new_frequency},{row}]"),
            );
        }
    }

    // The independently serialized expected arrays make the mapping check
    // explicit as well as checking the public selector result.
    for (new_frequency, &old_frequency) in EXPECTED_SAMPLES.iter().enumerate() {
        for row in 0..3 {
            for column in 0..3 {
                assert_eq!(
                    fixture.data.s[new_frequency][row][column].real,
                    fixture.data.s_input[old_frequency][row][column].real
                );
                assert_eq!(
                    fixture.data.s[new_frequency][row][column].imag,
                    fixture.data.s_input[old_frequency][row][column].imag
                );
            }
            assert_eq!(
                fixture.data.z0_ohm[new_frequency][row].real,
                fixture.data.z0_input_ohm[old_frequency][row].real
            );
            assert_eq!(
                fixture.data.z0_ohm[new_frequency][row].imag,
                fixture.data.z0_input_ohm[old_frequency][row].imag
            );
        }
    }
}
