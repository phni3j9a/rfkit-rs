use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/two_port_transducer_gain_power_four_frequency.json"
);
const EXPECTED_CASE_ID: &str = "two_port_transducer_gain_power_four_frequency";
const EXPECTED_OPERATION: &str = "network_two_port_transducer_gain_power";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_RANDOM_SEED: u64 = 20_261_015;
const RTOL: f64 = 1.0e-12;
const ATOL: f64 = 1.0e-12;

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
    load_impedance_ohm: Vec<ComplexValue>,
    s_input: Vec<Vec<Vec<ComplexValue>>>,
    source_impedance_ohm: Vec<ComplexValue>,
    transducer_gain: Vec<f64>,
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
    physical_definition: PhysicalDefinition,
    port_orientation: PortOrientation,
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
struct PhysicalDefinition {
    available_source_power: String,
    current_direction: String,
    load_power: String,
    phasors: String,
    ratio: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortOrientation {
    forward_s_entry: ForwardSEntry,
    load_output_port_zero_based: usize,
    source_input_port_zero_based: usize,
    target_reference_columns: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ForwardSEntry {
    column_input_zero_based: usize,
    row_output_zero_based: usize,
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
struct Shape {
    frequency: Vec<usize>,
    input_s: Vec<usize>,
    input_z0: Vec<usize>,
    source_impedance: Vec<usize>,
    load_impedance: Vec<usize>,
    output_transducer_gain: Vec<usize>,
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
    impedance: String,
    s: String,
    transducer_gain: String,
}

fn complex(value: &ComplexValue) -> Complex64 {
    Complex64::new(value.real, value.imag)
}

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
    Array3::from_shape_fn((4, 2, 2), |(frequency, row, column)| {
        complex(&values[frequency][row][column])
    })
}

fn array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    Array2::from_shape_fn((4, 2), |(frequency, port)| {
        complex(&values[frequency][port])
    })
}

fn vector(values: &[ComplexValue]) -> Vec<Complex64> {
    values.iter().map(complex).collect()
}

fn solve_two_by_two(matrix: [[Complex64; 2]; 2], rhs: [Complex64; 2]) -> [Complex64; 2] {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    [
        (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
        (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
    ]
}

fn independent_original_coordinate_gain(
    frequency: usize,
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    source: &[Complex64],
    load: &[Complex64],
) -> f64 {
    let mut impedance = [[Complex64::new(0.0, 0.0); 2]; 2];
    let mut a = [[Complex64::new(0.0, 0.0); 2]; 2];
    let mut b = [[Complex64::new(0.0, 0.0); 2]; 2];
    let normalization = [
        1.0 / (2.0 * z0[[frequency, 0]].re.sqrt()),
        1.0 / (2.0 * z0[[frequency, 1]].re.sqrt()),
    ];
    for row in 0..2 {
        for column in 0..2 {
            let value = s[[frequency, row, column]];
            let identity = if row == column {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            a[row][column] = (identity - value) * normalization[column];
            let mut rhs = value * z0[[frequency, column]];
            if row == column {
                rhs += z0[[frequency, row]].conj();
            }
            b[row][column] = rhs * normalization[column];
        }
    }
    for column in 0..2 {
        let result = solve_two_by_two(a, [b[0][column], b[1][column]]);
        impedance[0][column] = result[0];
        impedance[1][column] = result[1];
    }

    let loaded = [
        [impedance[0][0] + source[frequency], impedance[0][1]],
        [impedance[1][0], impedance[1][1] + load[frequency]],
    ];
    let currents = solve_two_by_two(loaded, [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
    4.0 * source[frequency].re * load[frequency].re * currents[1].norm_sqr()
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    let difference = (actual - expected).abs();
    let bound = ATOL + RTOL * expected.abs();
    assert!(
        difference <= bound,
        "{context}: actual={actual:.17e}, expected={expected:.17e}, difference={difference:.17e}, bound={bound:.17e}"
    );
}

#[test]
fn public_two_port_transducer_gain_matches_pinned_scikit_rf_fixture_and_v_i_guard() {
    let fixture: FixtureDocument = serde_json::from_str(FIXTURE_JSON).unwrap();
    assert_eq!(fixture.metadata.schema, "rfkit-rs.oracle.fixture");
    assert_eq!(fixture.metadata.schema_version, 1);
    assert_eq!(fixture.metadata.case_id, EXPECTED_CASE_ID);
    assert_eq!(fixture.metadata.operation, EXPECTED_OPERATION);
    assert_eq!(fixture.metadata.physical_definition.phasors, "RMS");
    assert_eq!(
        fixture.metadata.physical_definition.current_direction,
        "into the network"
    );
    assert_eq!(
        fixture.metadata.physical_definition.available_source_power,
        "abs(Vs)^2/(4*Re(Zs))"
    );
    assert_eq!(
        fixture.metadata.physical_definition.load_power,
        "Re(Zl)*abs(I1)^2"
    );
    assert_eq!(
        fixture.metadata.physical_definition.ratio,
        "P_load/P_available_source"
    );
    assert_eq!(
        fixture
            .metadata
            .port_orientation
            .source_input_port_zero_based,
        0
    );
    assert_eq!(
        fixture
            .metadata
            .port_orientation
            .load_output_port_zero_based,
        1
    );
    assert_eq!(
        fixture
            .metadata
            .port_orientation
            .forward_s_entry
            .row_output_zero_based,
        1
    );
    assert_eq!(
        fixture
            .metadata
            .port_orientation
            .forward_s_entry
            .column_input_zero_based,
        0
    );
    assert_eq!(
        fixture.metadata.port_orientation.target_reference_columns,
        vec!["source_impedance_ohm", "load_impedance_ohm"]
    );
    assert_eq!(fixture.metadata.numpy_version, EXPECTED_NUMPY_VERSION);
    assert_eq!(
        fixture.metadata.scikit_rf_version,
        EXPECTED_SCIKIT_RF_VERSION
    );
    assert_eq!(fixture.metadata.scikit_rf_commit, EXPECTED_SCIKIT_RF_COMMIT);
    assert_eq!(fixture.metadata.random_seed, EXPECTED_RANDOM_SEED);
    assert_eq!(fixture.metadata.wave_definition, "power");
    assert!(fixture.metadata.input_recipe.contains("20261015"));
    assert!(fixture.metadata.input_recipe.contains("source/load"));
    assert!(fixture.metadata.reference_impedance.complex);
    assert!(fixture.metadata.reference_impedance.frequency_dependent);
    assert!(fixture.metadata.reference_impedance.per_port);
    assert_eq!(
        fixture.metadata.reference_impedance.real_part,
        "strictly positive"
    );
    assert_eq!(fixture.metadata.reference_impedance.unit, "ohm");
    assert_eq!(fixture.metadata.shape.frequency, vec![4]);
    assert_eq!(fixture.metadata.shape.input_s, vec![4, 2, 2]);
    assert_eq!(fixture.metadata.shape.input_z0, vec![4, 2]);
    assert_eq!(fixture.metadata.shape.source_impedance, vec![4]);
    assert_eq!(fixture.metadata.shape.load_impedance, vec![4]);
    assert_eq!(fixture.metadata.shape.output_transducer_gain, vec![4]);
    assert_eq!(fixture.metadata.tolerance_policy.rtol, RTOL);
    assert_eq!(fixture.metadata.tolerance_policy.atol, ATOL);
    assert!(
        fixture
            .metadata
            .tolerance_policy
            .comparison
            .contains("transducer_gain")
    );
    assert!(
        fixture
            .metadata
            .tolerance_policy
            .justification
            .contains("V/I")
    );
    assert!(
        fixture
            .metadata
            .tolerance_policy
            .regeneration
            .contains("renormalize")
    );
    assert_eq!(fixture.metadata.units.frequency, "Hz");
    assert_eq!(fixture.metadata.units.impedance, "ohm");
    assert_eq!(fixture.metadata.units.s, "dimensionless");
    assert_eq!(
        fixture.metadata.units.transducer_gain,
        "dimensionless linear power ratio"
    );

    let s = array3(&fixture.data.s_input);
    let z0 = array2(&fixture.data.z0_input_ohm);
    let source = vector(&fixture.data.source_impedance_ohm);
    let load = vector(&fixture.data.load_impedance_ohm);
    let network = Network::new(
        Frequency::from_hz(fixture.data.frequency_hz.clone()).unwrap(),
        s.clone(),
        z0.clone(),
    )
    .unwrap();
    let snapshot = network.clone();
    let actual = network
        .two_port_transducer_gain_power(&source, &load)
        .unwrap();
    assert_eq!(actual.len(), fixture.data.transducer_gain.len());
    for (frequency, (&actual, &expected)) in actual
        .iter()
        .zip(fixture.data.transducer_gain.iter())
        .enumerate()
    {
        assert_close(actual, expected, &format!("fixture output[{frequency}]"));
        let independent = independent_original_coordinate_gain(frequency, &s, &z0, &source, &load);
        assert_close(independent, expected, &format!("V/I guard[{frequency}]"));
    }
    assert_eq!(network, snapshot);
}
