use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/power_wave_inner_connect_direct_five_port_complex_z0.json"
);

const EXPECTED_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_CASE_ID: &str = "power_wave_inner_connect_direct_five_port_complex_z0";
// The fixture metadata records the historical generator operation. The Rust
// conformance call itself intentionally uses the consolidated public method.
const EXPECTED_OPERATION: &str = "inner_connect_direct_power";
const EXPECTED_RANDOM_SEED: u64 = 20_260_952;
const EXPECTED_INPUT_RECIPE: &str = "one independent NumPy default_rng stream with seed 20260952, an asymmetric non-reciprocal five-port with a full selected 2x2 internal coupling block, three finite frequency samples, and unequal complex frequency-dependent positive-real references on every port";
const EXPECTED_NFREQ: usize = 3;
const EXPECTED_NPORTS: usize = 5;
const EXPECTED_PORT_A: usize = 1;
const EXPECTED_PORT_B: usize = 3;
const EXPECTED_FREQUENCY_HZ: [f64; EXPECTED_NFREQ] = [0.67e9, 1.21e9, 1.97e9];
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
    s: Vec<Vec<Vec<ComplexValue>>>,
    s_inner_connected: Vec<Vec<Vec<ComplexValue>>>,
    z0_ohm: Vec<Vec<ComplexValue>>,
    z0_inner_connected_ohm: Vec<Vec<ComplexValue>>,
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
    input_structure: InputStructure,
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
    wave_definitions: WaveDefinitions,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputStructure {
    asymmetric: bool,
    full_selected_internal_block: bool,
    nonreciprocal: bool,
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
    description: String,
    input: Vec<usize>,
    output: Vec<usize>,
    survivors: Vec<usize>,
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
    input_s: Vec<usize>,
    input_z0: Vec<usize>,
    intermediate_s: Vec<usize>,
    intermediate_z0: Vec<usize>,
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
struct WaveDefinitions {
    inner_connect_raw: String,
    input: String,
    output: String,
    restoration: String,
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
fn pinned_scikit_rf_direct_inner_connection_fixture_matches_public_network_method() {
    let fixture: FixtureDocument =
        serde_json::from_str(FIXTURE_JSON).expect("direct inner fixture must parse");
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
    assert_eq!(metadata.random_seed, EXPECTED_RANDOM_SEED);
    assert_eq!(metadata.wave_definition, "power");
    assert_eq!(
        metadata.wave_definitions.input, "power",
        "fixture input must be constructed as power waves"
    );
    assert_eq!(metadata.wave_definitions.inner_connect_raw, "pseudo");
    assert_eq!(metadata.wave_definitions.output, "power");
    assert_eq!(
        metadata.wave_definitions.restoration,
        "public result.renormalize(result.z0, s_def=\"power\")"
    );
    assert!(metadata.input_structure.asymmetric);
    assert!(metadata.input_structure.full_selected_internal_block);
    assert!(metadata.input_structure.nonreciprocal);
    assert_eq!(metadata.junction_ports.a, EXPECTED_PORT_A);
    assert_eq!(metadata.junction_ports.b, EXPECTED_PORT_B);
    assert_eq!(
        metadata.port_order.description,
        "Input ports excluding selected a and b, in original order"
    );
    assert_eq!(metadata.port_order.input, vec![0, 1, 2, 3, 4]);
    assert_eq!(metadata.port_order.survivors, vec![0, 2, 4]);
    assert_eq!(metadata.port_order.output, vec![0, 2, 4]);
    assert!(metadata.reference_impedance.complex);
    assert!(metadata.reference_impedance.frequency_dependent);
    assert!(metadata.reference_impedance.per_port);
    assert!(metadata.reference_impedance.positive_real);
    assert!(metadata.reference_impedance.selected_references_unequal);
    assert_eq!(metadata.reference_impedance.unit, "ohm");
    assert_eq!(metadata.shape.frequency, vec![EXPECTED_NFREQ]);
    assert_eq!(
        metadata.shape.input_s,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS, EXPECTED_NPORTS]
    );
    assert_eq!(
        metadata.shape.input_z0,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS]
    );
    assert_eq!(metadata.shape.intermediate_s, vec![EXPECTED_NFREQ, 3, 3]);
    assert_eq!(metadata.shape.intermediate_z0, vec![EXPECTED_NFREQ, 3]);
    assert_eq!(metadata.shape.output_s, vec![EXPECTED_NFREQ, 3, 3]);
    assert_eq!(metadata.shape.output_z0, vec![EXPECTED_NFREQ, 3]);
    assert_eq!(metadata.tolerance_policy.rtol, EXPECTED_RTOL);
    assert_eq!(metadata.tolerance_policy.atol, EXPECTED_ATOL);
    assert!(
        metadata
            .tolerance_policy
            .comparison
            .contains("only data.s_inner_connected")
    );
    assert!(
        metadata
            .tolerance_policy
            .justification
            .contains("innerconnect")
    );
    assert!(
        metadata
            .tolerance_policy
            .justification
            .contains("renormalize")
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

    let source_s = array3(&data.s, EXPECTED_NPORTS);
    let source_z0 = array2(&data.z0_ohm, EXPECTED_NPORTS);
    let expected_s = array3(&data.s_inner_connected, 3);
    let fixture_output_z0 = array2(&data.z0_inner_connected_ohm, 3);
    for ((frequency, row, column), value) in source_s.indexed_iter() {
        assert!(
            value.re.is_finite() && value.im.is_finite(),
            "input S at ({frequency},{row},{column}) must be finite"
        );
    }
    for ((frequency, port), value) in source_z0.indexed_iter() {
        assert!(
            value.re.is_finite() && value.im.is_finite(),
            "input z0 at ({frequency},{port}) must be finite"
        );
        assert!(value.re > 0.0);
        assert_ne!(value.im, 0.0);
    }
    assert!(
        (0..EXPECTED_NFREQ).any(|frequency| {
            (0..EXPECTED_NPORTS).any(|row| {
                (0..EXPECTED_NPORTS).any(|column| {
                    source_s[[frequency, row, column]] != source_s[[frequency, column, row]]
                })
            })
        }),
        "input S must be non-reciprocal"
    );
    for frequency in 0..EXPECTED_NFREQ {
        assert_ne!(
            source_s[[frequency, EXPECTED_PORT_A, EXPECTED_PORT_B]],
            Complex64::new(0.0, 0.0)
        );
        assert_ne!(
            source_s[[frequency, EXPECTED_PORT_B, EXPECTED_PORT_A]],
            Complex64::new(0.0, 0.0)
        );
        assert_ne!(
            source_s[[frequency, EXPECTED_PORT_A, EXPECTED_PORT_B]],
            source_s[[frequency, EXPECTED_PORT_B, EXPECTED_PORT_A]],
            "selected internal block must retain both non-reciprocal off-diagonal terms"
        );
        assert_ne!(
            source_z0[[frequency, EXPECTED_PORT_A]],
            source_z0[[frequency, EXPECTED_PORT_B]],
            "selected references must be unequal"
        );
    }

    // Reconstruct the input arrays and verify every serialized non-output
    // value is consumed exactly before invoking the public operation.
    for ((frequency, row, column), actual) in source_s.indexed_iter() {
        assert_complex_bits(
            *actual,
            &data.s[frequency][row][column],
            &format!("s[{frequency}][{row}][{column}]"),
        );
    }
    for ((frequency, port), actual) in source_z0.indexed_iter() {
        assert_complex_bits(
            *actual,
            &data.z0_ohm[frequency][port],
            &format!("z0_ohm[{frequency}][{port}]"),
        );
    }

    let frequency = Frequency::from_hz(data.frequency_hz.clone()).expect("fixture frequency axis");
    let source = Network::new(frequency, source_s.clone(), source_z0.clone())
        .expect("direct inner fixture Network");
    let source_frequency_snapshot = source.frequency().hz().to_vec();
    let source_s_snapshot = source.s().clone();
    let source_z0_snapshot = source.z0().clone();

    let reduced = source
        .inner_connect_power(EXPECTED_PORT_A, EXPECTED_PORT_B)
        .expect("direct inner fixture connection");

    assert_eq!(reduced.frequency().hz(), data.frequency_hz.as_slice());
    for (index, frequency) in reduced.frequency().hz().iter().enumerate() {
        assert_f64_bits(
            *frequency,
            data.frequency_hz[index],
            &format!("result.frequency_hz[{index}]"),
        );
    }
    let survivors = metadata.port_order.survivors;
    let independently_expected_z0 = Array2::from_shape_fn(
        (EXPECTED_NFREQ, survivors.len()),
        |(frequency, output_port)| source_z0[[frequency, survivors[output_port]]],
    );
    assert_eq!(reduced.z0(), &independently_expected_z0);
    assert_eq!(reduced.z0(), &fixture_output_z0);
    assert_eq!(reduced.s().dim(), (EXPECTED_NFREQ, 3, 3));

    for ((frequency, row, column), actual) in reduced.s().indexed_iter() {
        assert_s_close(
            *actual,
            expected_s[[frequency, row, column]],
            &format!("s_inner_connected[{frequency},{row},{column}]"),
        );
        assert!(actual.re.is_finite() && actual.im.is_finite());
    }

    // The operation borrows the source network and must preserve every source
    // frequency, S coordinate, and reference bit-for-bit.
    assert_eq!(
        source.frequency().hz(),
        source_frequency_snapshot.as_slice()
    );
    assert_eq!(source.s(), &source_s_snapshot);
    assert_eq!(source.z0(), &source_z0_snapshot);
}
