use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/frequency_sample_concatenation_three_port_complex_z0.json"
);

const EXPECTED_CASE_ID: &str = "frequency_sample_concatenation_three_port_complex_z0";
const EXPECTED_OPERATION: &str = "network_concatenate_frequency_samples";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_RANDOM_SEED: u64 = 20_261_017;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureDocument {
    data: FixtureData,
    metadata: FixtureMetadata,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureData {
    frequency_a_hz: Vec<f64>,
    frequency_b_hz: Vec<f64>,
    frequency_hz: Vec<f64>,
    s: Vec<Vec<Vec<ComplexValue>>>,
    s_a: Vec<Vec<Vec<ComplexValue>>>,
    s_b: Vec<Vec<Vec<ComplexValue>>>,
    z0_a_ohm: Vec<Vec<ComplexValue>>,
    z0_b_ohm: Vec<Vec<ComplexValue>>,
    z0_ohm: Vec<Vec<ComplexValue>>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ComplexValue {
    imag: f64,
    real: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureMetadata {
    case_id: String,
    frequency_concatenation: FrequencyConcatenation,
    input_recipe: String,
    numpy_version: String,
    operation: String,
    random_seed: u64,
    reference_impedance: ReferenceImpedance,
    schema: String,
    schema_version: u32,
    scikit_rf_commit: String,
    scikit_rf_version: String,
    shape: Shape,
    tolerance_policy: TolerancePolicy,
    units: Units,
    wave_definition: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrequencyConcatenation {
    a_samples: usize,
    b_samples: usize,
    description: String,
    output_mapping: Vec<OutputMapping>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputMapping {
    network: String,
    sample: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceImpedance {
    complex: bool,
    frequency_dependent: bool,
    per_port: bool,
    unequal: bool,
    unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Shape {
    frequency: Vec<usize>,
    frequency_a: Vec<usize>,
    frequency_b: Vec<usize>,
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

fn complex(value: &ComplexValue) -> Complex64 {
    Complex64::new(value.real, value.imag)
}

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
    let nfreq = values.len();
    let nports = values[0].len();
    Array3::from_shape_fn((nfreq, nports, nports), |(frequency, row, column)| {
        complex(&values[frequency][row][column])
    })
}

fn array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    let nfreq = values.len();
    let nports = values[0].len();
    Array2::from_shape_fn((nfreq, nports), |(frequency, port)| {
        complex(&values[frequency][port])
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
fn pinned_scikit_rf_stitch_fixture_matches_a_then_b_sequence_exactly() {
    let fixture: FixtureDocument = serde_json::from_str(FIXTURE_JSON).unwrap();
    let metadata = &fixture.metadata;
    assert_eq!(metadata.schema, "rfkit-rs.oracle.fixture");
    assert_eq!(metadata.schema_version, 1);
    assert_eq!(metadata.case_id, EXPECTED_CASE_ID);
    assert_eq!(metadata.operation, EXPECTED_OPERATION);
    assert_eq!(
        metadata.input_recipe,
        "one independent NumPy default_rng stream with seed 20261017, two separate asymmetric non-reciprocal three-port S draws with A=3 and B=2 samples, separated frequency bands, and independently authored complex per-port/frequency-dependent references"
    );
    assert_eq!(metadata.numpy_version, EXPECTED_NUMPY_VERSION);
    assert_eq!(metadata.scikit_rf_version, EXPECTED_SCIKIT_RF_VERSION);
    assert_eq!(metadata.scikit_rf_commit, EXPECTED_SCIKIT_RF_COMMIT);
    assert_eq!(metadata.random_seed, EXPECTED_RANDOM_SEED);
    assert_eq!(metadata.wave_definition, "power");
    assert_eq!(
        metadata.frequency_concatenation.description,
        "output samples are A in order followed by B in order"
    );
    assert_eq!(metadata.frequency_concatenation.a_samples, 3);
    assert_eq!(metadata.frequency_concatenation.b_samples, 2);
    assert_eq!(
        metadata
            .frequency_concatenation
            .output_mapping
            .iter()
            .map(|mapping| (mapping.network.as_str(), mapping.sample))
            .collect::<Vec<_>>(),
        vec![("A", 0), ("A", 1), ("A", 2), ("B", 0), ("B", 1),]
    );
    assert!(metadata.reference_impedance.complex);
    assert!(metadata.reference_impedance.frequency_dependent);
    assert!(metadata.reference_impedance.per_port);
    assert!(metadata.reference_impedance.unequal);
    assert_eq!(metadata.reference_impedance.unit, "ohm");
    assert_eq!(metadata.units.frequency, "Hz");
    assert_eq!(metadata.units.s, "dimensionless");
    assert_eq!(metadata.units.z0, "ohm");
    assert_eq!(metadata.shape.frequency_a, vec![3]);
    assert_eq!(metadata.shape.frequency_b, vec![2]);
    assert_eq!(metadata.shape.frequency, vec![5]);
    assert_eq!(metadata.shape.input_s_a, vec![3, 3, 3]);
    assert_eq!(metadata.shape.input_s_b, vec![2, 3, 3]);
    assert_eq!(metadata.shape.input_z0_a, vec![3, 3]);
    assert_eq!(metadata.shape.input_z0_b, vec![2, 3]);
    assert_eq!(metadata.shape.output_s, vec![5, 3, 3]);
    assert_eq!(metadata.shape.output_z0, vec![5, 3]);
    assert_eq!(
        metadata.tolerance_policy.comparison,
        "exact canonical UTF-8 JSON bytes; frequency, S, and z0 are exact sequence copies with no arithmetic tolerance"
    );
    assert_eq!(
        metadata.tolerance_policy.regeneration,
        "exact canonical UTF-8 JSON bytes"
    );

    let a = Network::new(
        Frequency::from_hz(fixture.data.frequency_a_hz.clone()).unwrap(),
        array3(&fixture.data.s_a),
        array2(&fixture.data.z0_a_ohm),
    )
    .unwrap();
    let b = Network::new(
        Frequency::from_hz(fixture.data.frequency_b_hz.clone()).unwrap(),
        array3(&fixture.data.s_b),
        array2(&fixture.data.z0_b_ohm),
    )
    .unwrap();
    let concatenated = a.concatenate_frequency_samples(&b).unwrap();

    assert_eq!(
        concatenated.frequency().len(),
        fixture.data.frequency_hz.len()
    );
    for (index, expected_frequency) in fixture.data.frequency_hz.iter().enumerate() {
        assert_f64_bits(
            concatenated.frequency().hz()[index],
            *expected_frequency,
            &format!("frequency[{index}]"),
        );
        for row in 0..3 {
            for column in 0..3 {
                assert_complex_bits(
                    concatenated.s()[[index, row, column]],
                    &fixture.data.s[index][row][column],
                    &format!("S[{index},{row},{column}]"),
                );
            }
            assert_complex_bits(
                concatenated.z0()[[index, row]],
                &fixture.data.z0_ohm[index][row],
                &format!("z0[{index},{row}]"),
            );
        }
    }

    // Independently serialized inputs make the A/B boundary and aligned
    // sample copies explicit rather than relying only on the stitched output.
    for sample in 0..3 {
        assert_eq!(
            fixture.data.frequency_hz[sample].to_bits(),
            fixture.data.frequency_a_hz[sample].to_bits()
        );
        for row in 0..3 {
            for column in 0..3 {
                assert_eq!(
                    fixture.data.s[sample][row][column].real,
                    fixture.data.s_a[sample][row][column].real
                );
                assert_eq!(
                    fixture.data.s[sample][row][column].imag,
                    fixture.data.s_a[sample][row][column].imag
                );
            }
            assert_eq!(
                fixture.data.z0_ohm[sample][row],
                fixture.data.z0_a_ohm[sample][row]
            );
        }
    }
    for sample in 0..2 {
        let output_sample = 3 + sample;
        assert_eq!(
            fixture.data.frequency_hz[output_sample].to_bits(),
            fixture.data.frequency_b_hz[sample].to_bits()
        );
        for row in 0..3 {
            for column in 0..3 {
                assert_eq!(
                    fixture.data.s[output_sample][row][column].real,
                    fixture.data.s_b[sample][row][column].real
                );
                assert_eq!(
                    fixture.data.s[output_sample][row][column].imag,
                    fixture.data.s_b[sample][row][column].imag
                );
            }
            assert_eq!(
                fixture.data.z0_ohm[output_sample][row],
                fixture.data.z0_b_ohm[sample][row]
            );
        }
    }
}
