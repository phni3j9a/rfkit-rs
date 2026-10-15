use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/combine_uncoupled_two_plus_three_port_complex_z0.json"
);
const EXPECTED_CASE_ID: &str = "combine_uncoupled_two_plus_three_port_complex_z0";
const EXPECTED_OPERATION: &str = "network_combine_uncoupled";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_RANDOM_SEED: u64 = 20_261_016;

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
    s_combined: Vec<Vec<Vec<ComplexValue>>>,
    z0_a_ohm: Vec<Vec<ComplexValue>>,
    z0_b_ohm: Vec<Vec<ComplexValue>>,
    z0_combined_ohm: Vec<Vec<ComplexValue>>,
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
    shape: Shape,
    tolerance_policy: TolerancePolicy,
    units: Units,
    wave_definition: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortOrder {
    a_ports: usize,
    b_ports: usize,
    description: String,
    output: Vec<OutputPort>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputPort {
    network: String,
    port: usize,
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

fn assert_complex_bits(actual: Complex64, expected: Complex64, context: &str) {
    assert_eq!(actual.re.to_bits(), expected.re.to_bits(), "{context}.real");
    assert_eq!(actual.im.to_bits(), expected.im.to_bits(), "{context}.imag");
}

#[test]
fn pinned_scikit_rf_concat_ports_fixture_matches_direct_sum_exactly() {
    let fixture: FixtureDocument = serde_json::from_str(FIXTURE_JSON).unwrap();
    let metadata = &fixture.metadata;
    assert_eq!(metadata.schema, "rfkit-rs.oracle.fixture");
    assert_eq!(metadata.schema_version, 1);
    assert_eq!(metadata.case_id, EXPECTED_CASE_ID);
    assert_eq!(metadata.operation, EXPECTED_OPERATION);
    assert_eq!(
        metadata.input_recipe,
        "one independent NumPy default_rng stream with seed 20261016, an asymmetric non-reciprocal two-port A and three-port B, three finite frequency samples, and unequal complex frequency-dependent references"
    );
    assert_eq!(metadata.numpy_version, EXPECTED_NUMPY_VERSION);
    assert_eq!(metadata.scikit_rf_version, EXPECTED_SCIKIT_RF_VERSION);
    assert_eq!(metadata.scikit_rf_commit, EXPECTED_SCIKIT_RF_COMMIT);
    assert_eq!(metadata.random_seed, EXPECTED_RANDOM_SEED);
    assert_eq!(metadata.wave_definition, "power");
    assert_eq!(metadata.port_order.a_ports, 2);
    assert_eq!(metadata.port_order.b_ports, 3);
    assert_eq!(
        metadata.port_order.description,
        "A ports in original order followed by B ports in original order"
    );
    assert_eq!(
        metadata
            .port_order
            .output
            .iter()
            .map(|port| (port.network.as_str(), port.port))
            .collect::<Vec<_>>(),
        vec![("A", 0), ("A", 1), ("B", 0), ("B", 1), ("B", 2)]
    );
    assert!(metadata.reference_impedance.complex);
    assert!(metadata.reference_impedance.frequency_dependent);
    assert!(metadata.reference_impedance.per_port);
    assert!(metadata.reference_impedance.unequal);
    assert_eq!(metadata.reference_impedance.unit, "ohm");
    assert_eq!(metadata.units.frequency, "Hz");
    assert_eq!(metadata.units.s, "dimensionless");
    assert_eq!(metadata.units.z0, "ohm");
    assert_eq!(metadata.shape.frequency, vec![3]);
    assert_eq!(metadata.shape.input_s_a, vec![3, 2, 2]);
    assert_eq!(metadata.shape.input_s_b, vec![3, 3, 3]);
    assert_eq!(metadata.shape.input_z0_a, vec![3, 2]);
    assert_eq!(metadata.shape.input_z0_b, vec![3, 3]);
    assert_eq!(metadata.shape.output_s, vec![3, 5, 5]);
    assert_eq!(metadata.shape.output_z0, vec![3, 5]);
    assert_eq!(
        metadata.tolerance_policy.comparison,
        "exact canonical UTF-8 JSON bytes; the direct-sum output is block placement and canonical complex zeros with no floating arithmetic"
    );
    assert_eq!(
        metadata.tolerance_policy.regeneration,
        "exact canonical UTF-8 JSON bytes"
    );

    let a = Network::new(
        Frequency::from_hz(fixture.data.frequency_hz.clone()).unwrap(),
        array3(&fixture.data.s_a),
        array2(&fixture.data.z0_a_ohm),
    )
    .unwrap();
    let b = Network::new(
        Frequency::from_hz(fixture.data.frequency_hz.clone()).unwrap(),
        array3(&fixture.data.s_b),
        array2(&fixture.data.z0_b_ohm),
    )
    .unwrap();
    let combined = a.combine_uncoupled(&b).unwrap();
    let expected_s = array3(&fixture.data.s_combined);
    let expected_z0 = array2(&fixture.data.z0_combined_ohm);
    assert_eq!(combined.s().dim(), expected_s.dim());
    assert_eq!(combined.z0().dim(), expected_z0.dim());
    for (index, (&actual, &expected)) in combined.s().iter().zip(expected_s.iter()).enumerate() {
        assert_complex_bits(actual, expected, &format!("s[{index}]"));
    }
    for (index, (&actual, &expected)) in combined.z0().iter().zip(expected_z0.iter()).enumerate() {
        assert_complex_bits(actual, expected, &format!("z0[{index}]"));
    }
    for (&actual, &expected) in combined
        .frequency()
        .hz()
        .iter()
        .zip(&fixture.data.frequency_hz)
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    for frequency in 0..combined.frequency().len() {
        for row in 0..a.nports() {
            for column in a.nports()..combined.nports() {
                assert_complex_bits(
                    combined.s()[[frequency, row, column]],
                    Complex64::new(0.0, 0.0),
                    "cross A->B block",
                );
            }
        }
        for row in a.nports()..combined.nports() {
            for column in 0..a.nports() {
                assert_complex_bits(
                    combined.s()[[frequency, row, column]],
                    Complex64::new(0.0, 0.0),
                    "cross B->A block",
                );
            }
        }
    }
}
