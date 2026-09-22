use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/power_wave_connect_direct_three_to_four_port_complex_z0.json"
);

const EXPECTED_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_CASE_ID: &str = "power_wave_connect_direct_three_to_four_port_complex_z0";
const EXPECTED_OPERATION: &str = "connect_direct_power";
const EXPECTED_RANDOM_SEED: u64 = 20_260_951;
const EXPECTED_INPUT_RECIPE: &str = "one independent NumPy default_rng stream with seed 20260951, an asymmetric non-reciprocal three-port A and four-port B, three finite frequency samples, and unequal complex frequency-dependent positive-real references on every selected and surviving port";
const EXPECTED_NFREQ: usize = 3;
const EXPECTED_NPORTS_A: usize = 3;
const EXPECTED_NPORTS_B: usize = 4;
const EXPECTED_PORT_A: usize = 1;
const EXPECTED_PORT_B: usize = 2;
const EXPECTED_FREQUENCY_HZ: [f64; EXPECTED_NFREQ] = [0.83e9, 1.29e9, 2.11e9];
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
    s_a: Vec<Vec<Vec<ComplexValue>>>,
    s_b: Vec<Vec<Vec<ComplexValue>>>,
    s_connected: Vec<Vec<Vec<ComplexValue>>>,
    z0_a_ohm: Vec<Vec<ComplexValue>>,
    z0_b_ohm: Vec<Vec<ComplexValue>>,
    z0_connected_ohm: Vec<Vec<ComplexValue>>,
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
    junction_ports: JunctionPorts,
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
struct JunctionPorts {
    a: usize,
    b: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortOrder {
    a_survivors: Vec<usize>,
    b_survivors: Vec<usize>,
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
    positive_real: bool,
    selected_references_unequal: bool,
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

fn complex(value: &ComplexValue) -> Complex64 {
    Complex64::new(value.real, value.imag)
}

fn array3(values: &[Vec<Vec<ComplexValue>>], nports: usize) -> Array3<Complex64> {
    assert_eq!(values.len(), EXPECTED_NFREQ);
    assert!(values.iter().all(|matrix| matrix.len() == nports));
    assert!(
        values
            .iter()
            .flat_map(|matrix| matrix.iter())
            .all(|row| row.len() == nports)
    );
    Array3::from_shape_fn(
        (EXPECTED_NFREQ, nports, nports),
        |(frequency, row, column)| complex(&values[frequency][row][column]),
    )
}

fn array2(values: &[Vec<ComplexValue>], nports: usize) -> Array2<Complex64> {
    assert_eq!(values.len(), EXPECTED_NFREQ);
    assert!(values.iter().all(|row| row.len() == nports));
    Array2::from_shape_fn((EXPECTED_NFREQ, nports), |(frequency, port)| {
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

fn assert_s_close(actual: Complex64, expected: Complex64, context: &str) {
    let difference = (actual - expected).norm();
    let bound = EXPECTED_ATOL + EXPECTED_RTOL * expected.norm();
    assert!(
        difference <= bound,
        "{context}: actual={actual:?}, expected={expected:?}, difference={difference:e}, bound={bound:e}"
    );
}

#[test]
fn pinned_scikit_rf_direct_connection_fixture_matches_public_network_method() {
    let fixture: FixtureDocument =
        serde_json::from_str(FIXTURE_JSON).expect("direct connection fixture must parse");
    let metadata = fixture.metadata;
    let data = fixture.data;

    assert_eq!(metadata.schema, EXPECTED_SCHEMA);
    assert_eq!(metadata.schema_version, EXPECTED_SCHEMA_VERSION);
    assert_eq!(metadata.case_id, EXPECTED_CASE_ID);
    assert_eq!(metadata.operation, EXPECTED_OPERATION);
    assert_eq!(metadata.input_recipe, EXPECTED_INPUT_RECIPE);
    assert_eq!(metadata.numpy_version, EXPECTED_NUMPY_VERSION);
    assert_eq!(metadata.scikit_rf_version, EXPECTED_SCIKIT_RF_VERSION);
    assert_eq!(metadata.scikit_rf_commit, EXPECTED_SCIKIT_RF_COMMIT);
    assert_eq!(metadata.wave_definition, "power");
    assert_eq!(metadata.random_seed, EXPECTED_RANDOM_SEED);
    assert_eq!(metadata.junction_ports.a, EXPECTED_PORT_A);
    assert_eq!(metadata.junction_ports.b, EXPECTED_PORT_B);
    assert_eq!(
        metadata.port_order.description,
        "A unconnected ports in original order, followed by B unconnected ports in original order"
    );
    assert_eq!(metadata.port_order.a_survivors, vec![0, 2]);
    assert_eq!(metadata.port_order.b_survivors, vec![0, 1, 3]);
    assert_eq!(
        metadata
            .port_order
            .output
            .iter()
            .map(|port| (port.network.as_str(), port.port))
            .collect::<Vec<_>>(),
        vec![("A", 0), ("A", 2), ("B", 0), ("B", 1), ("B", 3)]
    );
    assert!(metadata.reference_impedance.complex);
    assert!(metadata.reference_impedance.frequency_dependent);
    assert!(metadata.reference_impedance.per_port);
    assert!(metadata.reference_impedance.positive_real);
    assert!(metadata.reference_impedance.selected_references_unequal);
    assert_eq!(metadata.reference_impedance.unit, "ohm");
    assert_eq!(metadata.shape.frequency, vec![EXPECTED_NFREQ]);
    assert_eq!(
        metadata.shape.input_s_a,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS_A, EXPECTED_NPORTS_A]
    );
    assert_eq!(
        metadata.shape.input_s_b,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS_B, EXPECTED_NPORTS_B]
    );
    assert_eq!(
        metadata.shape.input_z0_a,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS_A]
    );
    assert_eq!(
        metadata.shape.input_z0_b,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS_B]
    );
    assert_eq!(metadata.shape.output_s, vec![EXPECTED_NFREQ, 5, 5]);
    assert_eq!(metadata.shape.output_z0, vec![EXPECTED_NFREQ, 5]);
    assert_eq!(metadata.tolerance_policy.rtol, EXPECTED_RTOL);
    assert_eq!(metadata.tolerance_policy.atol, EXPECTED_ATOL);
    assert!(
        metadata
            .tolerance_policy
            .comparison
            .contains("only data.s_connected")
    );
    assert!(
        metadata
            .tolerance_policy
            .justification
            .contains("public scikit-rf network.connect")
    );
    assert!(
        metadata
            .tolerance_policy
            .regeneration
            .contains("checked exactly")
    );

    assert_eq!(data.frequency_hz.len(), EXPECTED_NFREQ);
    for (index, frequency) in data.frequency_hz.iter().enumerate() {
        assert_f64_bits(
            *frequency,
            EXPECTED_FREQUENCY_HZ[index],
            &format!("fixture frequency_hz[{index}]"),
        );
        assert!(frequency.is_finite());
    }
    let source_s_a = array3(&data.s_a, EXPECTED_NPORTS_A);
    let source_s_b = array3(&data.s_b, EXPECTED_NPORTS_B);
    let source_z0_a = array2(&data.z0_a_ohm, EXPECTED_NPORTS_A);
    let source_z0_b = array2(&data.z0_b_ohm, EXPECTED_NPORTS_B);
    let expected_s = array3(&data.s_connected, 5);
    let fixture_output_z0 = array2(&data.z0_connected_ohm, 5);

    for ((frequency, row, column), value) in source_s_a.indexed_iter() {
        assert!(
            value.re.is_finite() && value.im.is_finite(),
            "A S at ({frequency},{row},{column}) must be finite"
        );
    }
    for ((frequency, row, column), value) in source_s_b.indexed_iter() {
        assert!(
            value.re.is_finite() && value.im.is_finite(),
            "B S at ({frequency},{row},{column}) must be finite"
        );
    }
    for ((frequency, port), value) in source_z0_a.indexed_iter() {
        assert!(value.re.is_finite() && value.im.is_finite());
        assert!(
            value.re > 0.0,
            "A z0 at ({frequency},{port}) must be positive-real"
        );
    }
    for ((frequency, port), value) in source_z0_b.indexed_iter() {
        assert!(value.re.is_finite() && value.im.is_finite());
        assert!(
            value.re > 0.0,
            "B z0 at ({frequency},{port}) must be positive-real"
        );
    }
    assert!(
        (0..EXPECTED_NFREQ).any(|frequency| {
            (0..EXPECTED_NPORTS_A).any(|row| {
                (0..EXPECTED_NPORTS_A).any(|column| {
                    source_s_a[[frequency, row, column]] != source_s_a[[frequency, column, row]]
                })
            })
        }),
        "A fixture input must be non-reciprocal"
    );
    assert!(
        (0..EXPECTED_NFREQ).any(|frequency| {
            (0..EXPECTED_NPORTS_B).any(|row| {
                (0..EXPECTED_NPORTS_B).any(|column| {
                    source_s_b[[frequency, row, column]] != source_s_b[[frequency, column, row]]
                })
            })
        }),
        "B fixture input must be non-reciprocal"
    );
    assert!(
        source_z0_a
            .column(EXPECTED_PORT_A)
            .iter()
            .zip(source_z0_b.column(EXPECTED_PORT_B).iter())
            .all(|(a, b)| a != b),
        "selected reference coordinates must be unequal"
    );
    assert!(
        source_z0_a
            .column(EXPECTED_PORT_A)
            .iter()
            .chain(source_z0_b.column(EXPECTED_PORT_B).iter())
            .all(|value| value.im != 0.0),
        "selected references must exercise the complex domain"
    );

    // Exact input/grid/reference checks are deliberately performed before the
    // public operation.  The only oracle comparison below is connected S.
    for (index, frequency) in data.frequency_hz.iter().enumerate() {
        assert_f64_bits(
            *frequency,
            data.frequency_hz[index],
            &format!("frequency_hz[{index}]"),
        );
    }
    for ((frequency, port), actual) in source_z0_a.indexed_iter() {
        assert_complex_bits(
            *actual,
            &data.z0_a_ohm[frequency][port],
            &format!("z0_a_ohm[{frequency}][{port}]"),
        );
    }
    for ((frequency, port), actual) in source_z0_b.indexed_iter() {
        assert_complex_bits(
            *actual,
            &data.z0_b_ohm[frequency][port],
            &format!("z0_b_ohm[{frequency}][{port}]"),
        );
    }

    let frequency = Frequency::from_hz(data.frequency_hz.clone()).expect("fixture frequency axis");
    let network_a = Network::new(frequency.clone(), source_s_a.clone(), source_z0_a.clone())
        .expect("fixture A Network");
    let network_b = Network::new(frequency, source_s_b.clone(), source_z0_b.clone())
        .expect("fixture B Network");
    let source_a_snapshot = network_a.s().clone();
    let source_b_snapshot = network_b.s().clone();
    let source_z0_a_snapshot = network_a.z0().clone();
    let source_z0_b_snapshot = network_b.z0().clone();
    let connected = network_a
        .connect_direct_power(EXPECTED_PORT_A, &network_b, EXPECTED_PORT_B)
        .expect("direct fixture connection");

    assert_eq!(connected.frequency().hz(), data.frequency_hz.as_slice());
    let survivors_a = metadata.port_order.a_survivors;
    let survivors_b = metadata.port_order.b_survivors;
    let independently_expected_z0 = Array2::from_shape_fn(
        (EXPECTED_NFREQ, survivors_a.len() + survivors_b.len()),
        |(frequency, output_port)| {
            if output_port < survivors_a.len() {
                source_z0_a[[frequency, survivors_a[output_port]]]
            } else {
                source_z0_b[[frequency, survivors_b[output_port - survivors_a.len()]]]
            }
        },
    );
    assert_eq!(connected.z0(), &independently_expected_z0);
    assert_eq!(connected.z0(), &fixture_output_z0);
    assert_eq!(connected.s().dim(), (EXPECTED_NFREQ, 5, 5));

    for ((frequency, row, column), actual) in connected.s().indexed_iter() {
        assert_s_close(
            *actual,
            expected_s[[frequency, row, column]],
            &format!("s_connected[{frequency},{row},{column}]"),
        );
        assert!(actual.re.is_finite() && actual.im.is_finite());
    }

    // The public method borrows both inputs and must preserve every source
    // coordinate exactly.
    assert_eq!(network_a.s(), &source_a_snapshot);
    assert_eq!(network_b.s(), &source_b_snapshot);
    assert_eq!(network_a.z0(), &source_z0_a_snapshot);
    assert_eq!(network_b.z0(), &source_z0_b_snapshot);
    assert_eq!(network_a.frequency().hz(), data.frequency_hz.as_slice());
    assert_eq!(network_b.frequency().hz(), data.frequency_hz.as_slice());
}
