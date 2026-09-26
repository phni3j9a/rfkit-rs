use std::fmt;

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network, PortLoad};
use serde::Deserialize;

const FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/power_wave_terminate_port_impedance_five_port_complex_z0.json"
);
const MIXED_OPEN_FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/power_wave_terminate_port_mixed_open_five_port_complex_z0.json"
);

const EXPECTED_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_CASE_ID: &str = "power_wave_terminate_port_impedance_five_port_complex_z0";
const EXPECTED_OPERATION: &str = "terminate_port_impedance_power";
const EXPECTED_INPUT_RECIPE: &str = "independent local NumPy default_rng input with an asymmetric complex five-port S stack, a frequency-dependent unequal complex positive-real source-reference profile, and explicit finite loads [0, 38+12j, 73-9j] ohm; selected source port is the middle port 2";
const EXPECTED_NFREQ: usize = 3;
const EXPECTED_NPORTS: usize = 5;
const EXPECTED_PORT: usize = 2;
const EXPECTED_RTOL: f64 = 1.0e-12;
const EXPECTED_ATOL: f64 = 1.0e-12;
const EXPECTED_MIXED_OPEN_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_MIXED_OPEN_SCHEMA_VERSION: u32 = 1;
const EXPECTED_MIXED_OPEN_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_MIXED_OPEN_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_MIXED_OPEN_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_MIXED_OPEN_CASE_ID: &str =
    "power_wave_terminate_port_mixed_open_five_port_complex_z0";
const EXPECTED_MIXED_OPEN_OPERATION: &str = "terminate_port_power";
const EXPECTED_MIXED_OPEN_INPUT_RECIPE: &str = "independent local NumPy default_rng input with seed 20260963, an asymmetric non-reciprocal five-port S stack, four finite frequency samples, a middle selected port 2, unequal complex frequency-dependent positive-real source references, and mixed physical loads [Open, 31+7j, Open, -17+4j] ohm";
const EXPECTED_MIXED_OPEN_NFREQ: usize = 4;
const EXPECTED_MIXED_OPEN_NPORTS: usize = 5;
const EXPECTED_MIXED_OPEN_PORT: usize = 2;
const EXPECTED_MIXED_OPEN_RTOL: f64 = 1.0e-12;
const EXPECTED_MIXED_OPEN_ATOL: f64 = 1.0e-12;

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
    load_ohm: Vec<ComplexValue>,
    s: Vec<Vec<Vec<ComplexValue>>>,
    s_terminated: Vec<Vec<Vec<ComplexValue>>>,
    z0_ohm: Vec<Vec<ComplexValue>>,
    z0_survivor_ohm: Vec<Vec<ComplexValue>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MixedOpenFixtureDocument {
    data: MixedOpenFixtureData,
    metadata: MixedOpenFixtureMetadata,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MixedOpenFixtureData {
    frequency_hz: Vec<f64>,
    loads: Vec<MixedOpenLoad>,
    s: Vec<Vec<Vec<ComplexValue>>>,
    s_terminated: Vec<Vec<Vec<ComplexValue>>>,
    z0_ohm: Vec<Vec<ComplexValue>>,
    z0_survivor_ohm: Vec<Vec<ComplexValue>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MixedOpenLoad {
    kind: String,
    #[serde(default)]
    value: Option<ComplexValue>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MixedOpenFixtureMetadata {
    case_id: String,
    input_recipe: String,
    load_boundary: MixedOpenLoadBoundary,
    numpy_version: String,
    operation: String,
    port: usize,
    port_order: PortOrder,
    random_seed: u64,
    reference_impedance: ReferenceImpedance,
    schema: String,
    schema_version: u32,
    scikit_rf_commit: String,
    scikit_rf_version: String,
    shape: MixedOpenDeclaredShape,
    tolerance_policy: TolerancePolicy,
    wave_definition: String,
    wave_definitions: MixedOpenWaveDefinitions,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MixedOpenLoadBoundary {
    finite_only: bool,
    includes_ideal_open: bool,
    includes_ideal_short: bool,
    load_excitation: String,
    open_sentinel: String,
    unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MixedOpenDeclaredShape {
    frequency: Vec<usize>,
    input_s: Vec<usize>,
    input_z0: Vec<usize>,
    loads: Vec<usize>,
    output_s: Vec<usize>,
    output_z0: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MixedOpenWaveDefinitions {
    connect_raw: String,
    input: String,
    load: String,
    output: String,
    restoration: String,
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
    input_recipe: String,
    load_boundary: LoadBoundary,
    numpy_version: String,
    operation: String,
    port: usize,
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
struct LoadBoundary {
    finite_only: bool,
    includes_ideal_short: bool,
    load_excitation: String,
    open_sentinel: String,
    unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortOrder {
    description: String,
    input: Vec<usize>,
    output: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceImpedance {
    complex: bool,
    frequency_dependent: bool,
    per_port: bool,
    positive_real: bool,
    source_field: String,
    survivor_field: String,
    unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeclaredShape {
    frequency: Vec<usize>,
    input_s: Vec<usize>,
    input_z0: Vec<usize>,
    load_ohm: Vec<usize>,
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

#[derive(Debug)]
enum FixtureError {
    Parse(serde_json::Error),
}

impl fmt::Display for FixtureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => write!(formatter, "fixture JSON parse failed: {error}"),
        }
    }
}

impl std::error::Error for FixtureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
        }
    }
}

fn parse_fixture(json: &str) -> Result<FixtureDocument, FixtureError> {
    serde_json::from_str(json).map_err(FixtureError::Parse)
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

fn mixed_open_array3(
    values: &[Vec<Vec<ComplexValue>>],
    nfreq: usize,
    nports: usize,
) -> Array3<Complex64> {
    assert_eq!(values.len(), nfreq);
    assert!(values.iter().all(|matrix| matrix.len() == nports));
    assert!(
        values
            .iter()
            .flat_map(|matrix| matrix.iter())
            .all(|row| row.len() == nports)
    );
    Array3::from_shape_fn((nfreq, nports, nports), |(frequency, row, column)| {
        complex(&values[frequency][row][column])
    })
}

fn mixed_open_array2(
    values: &[Vec<ComplexValue>],
    nfreq: usize,
    nports: usize,
) -> Array2<Complex64> {
    assert_eq!(values.len(), nfreq);
    assert!(values.iter().all(|row| row.len() == nports));
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

fn assert_s_close(actual: Complex64, expected: Complex64, context: &str) {
    let difference = (actual - expected).norm();
    let bound = EXPECTED_ATOL + EXPECTED_RTOL * expected.norm();
    assert!(
        difference <= bound,
        "{context}: actual={actual:?}, expected={expected:?}, difference={difference:e}, bound={bound:e}"
    );
}

#[test]
fn pinned_scikit_rf_termination_fixture_matches_public_network_method() {
    let fixture = parse_fixture(FIXTURE_JSON).expect("termination fixture must parse");
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
    assert_eq!(metadata.random_seed, 20_260_950);
    assert_eq!(metadata.port, EXPECTED_PORT);
    assert_eq!(
        metadata.port_order.description,
        "source ports except the selected port, in original order"
    );
    assert_eq!(metadata.port_order.input, vec![0, 1, 2, 3, 4]);
    assert_eq!(metadata.port_order.output, vec![0, 1, 3, 4]);
    assert!(
        metadata.load_boundary.finite_only,
        "oracle load domain must remain finite-only"
    );
    assert!(metadata.load_boundary.includes_ideal_short);
    assert_eq!(metadata.load_boundary.load_excitation, "none");
    assert_eq!(metadata.load_boundary.open_sentinel, "rejected");
    assert_eq!(metadata.load_boundary.unit, "ohm");
    assert!(metadata.reference_impedance.complex);
    assert!(metadata.reference_impedance.frequency_dependent);
    assert!(metadata.reference_impedance.per_port);
    assert!(metadata.reference_impedance.positive_real);
    assert_eq!(metadata.reference_impedance.source_field, "z0_ohm");
    assert_eq!(
        metadata.reference_impedance.survivor_field,
        "z0_survivor_ohm"
    );
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
    assert_eq!(metadata.shape.load_ohm, vec![EXPECTED_NFREQ]);
    assert_eq!(metadata.shape.output_s, vec![EXPECTED_NFREQ, 4, 4]);
    assert_eq!(metadata.shape.output_z0, vec![EXPECTED_NFREQ, 4]);
    assert_eq!(metadata.tolerance_policy.rtol, EXPECTED_RTOL);
    assert_eq!(metadata.tolerance_policy.atol, EXPECTED_ATOL);
    assert!(
        metadata
            .tolerance_policy
            .comparison
            .contains("only data.s_terminated")
    );
    assert!(
        metadata
            .tolerance_policy
            .justification
            .contains("public scikit-rf connect")
    );
    assert!(metadata.tolerance_policy.justification.contains("z2s"));
    assert!(
        metadata
            .tolerance_policy
            .regeneration
            .contains("checked exactly")
    );

    assert_eq!(data.frequency_hz.len(), EXPECTED_NFREQ);
    assert_eq!(data.load_ohm.len(), EXPECTED_NFREQ);
    assert_eq!(data.s.len(), EXPECTED_NFREQ);
    assert_eq!(data.s_terminated.len(), EXPECTED_NFREQ);
    assert_eq!(data.z0_ohm.len(), EXPECTED_NFREQ);
    assert_eq!(data.z0_survivor_ohm.len(), EXPECTED_NFREQ);
    for (index, frequency) in data.frequency_hz.iter().enumerate() {
        assert!(frequency.is_finite(), "frequency {index} must be finite");
    }

    let source_s = array3(&data.s, EXPECTED_NPORTS);
    let source_z0 = array2(&data.z0_ohm, EXPECTED_NPORTS);
    let expected_s = array3(&data.s_terminated, 4);
    let expected_survivor_z0 = array2(&data.z0_survivor_ohm, 4);
    let load = data.load_ohm.iter().map(complex).collect::<Vec<_>>();
    let loads = load
        .iter()
        .copied()
        .map(PortLoad::ImpedanceOhm)
        .collect::<Vec<_>>();
    assert_eq!(
        load,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(38.0, 12.0),
            Complex64::new(73.0, -9.0),
        ]
    );
    for (index, value) in load.iter().enumerate() {
        assert!(value.re.is_finite() && value.im.is_finite());
        assert!(
            value.re >= 0.0,
            "oracle load {index} should include finite explicit values"
        );
    }
    for ((frequency, port), value) in source_z0.indexed_iter() {
        assert!(value.re.is_finite() && value.im.is_finite());
        assert!(
            value.re > 0.0,
            "source z0 at ({frequency},{port}) must be positive-real"
        );
    }
    for ((frequency, row, column), value) in source_s.indexed_iter() {
        assert!(
            value.re.is_finite() && value.im.is_finite(),
            "source S at ({frequency},{row},{column}) must be finite"
        );
    }
    for ((frequency, row, column), value) in expected_s.indexed_iter() {
        assert!(
            value.re.is_finite() && value.im.is_finite(),
            "expected S at ({frequency},{row},{column}) must be finite"
        );
    }
    for ((frequency, port), value) in expected_survivor_z0.indexed_iter() {
        assert!(
            value.re.is_finite() && value.im.is_finite(),
            "expected survivor z0 at ({frequency},{port}) must be finite"
        );
    }

    // Validate the canonical source/reference/load fields by exact binary
    // value before exercising the public method.  This prevents a fixture
    // drift from being hidden by a matching reimplementation in the test.
    for (index, frequency) in data.frequency_hz.iter().enumerate() {
        assert_f64_bits(
            *frequency,
            data.frequency_hz[index],
            &format!("frequency_hz[{index}]"),
        );
    }
    for ((frequency, port), actual) in source_z0.indexed_iter() {
        assert_complex_bits(
            *actual,
            &data.z0_ohm[frequency][port],
            &format!("z0_ohm[{frequency}][{port}]"),
        );
    }
    for (index, actual) in load.iter().enumerate() {
        assert_complex_bits(
            *actual,
            &data.load_ohm[index],
            &format!("load_ohm[{index}]"),
        );
    }

    let frequency = Frequency::from_hz(data.frequency_hz.clone()).expect("fixture frequency axis");
    let source = Network::new(frequency, source_s.clone(), source_z0.clone())
        .expect("fixture source Network");
    let terminated = source
        .terminate_port_power(EXPECTED_PORT, &loads)
        .expect("finite fixture load termination");

    assert_eq!(terminated.frequency().hz().len(), EXPECTED_NFREQ);
    for (index, actual) in terminated.frequency().hz().iter().enumerate() {
        assert_f64_bits(
            *actual,
            data.frequency_hz[index],
            &format!("output frequency[{index}]"),
        );
    }
    let survivors = metadata.port_order.output;
    assert_eq!(survivors, vec![0, 1, 3, 4]);
    let independently_expected_z0 = Array2::from_shape_fn(
        (EXPECTED_NFREQ, survivors.len()),
        |(frequency, output_port)| source_z0[[frequency, survivors[output_port]]],
    );
    assert_eq!(terminated.z0(), &independently_expected_z0);
    assert_eq!(terminated.z0(), &expected_survivor_z0);
    assert_eq!(terminated.s().dim(), (EXPECTED_NFREQ, 4, 4));

    // The only floating comparison against the pinned oracle output is the
    // reduced S matrix.  Metadata, inputs, frequency labels, and survivor
    // references above are exact contract checks.
    for ((frequency, row, column), actual) in terminated.s().indexed_iter() {
        assert_s_close(
            *actual,
            expected_s[[frequency, row, column]],
            &format!("s_terminated[{frequency},{row},{column}]"),
        );
        assert!(actual.re.is_finite() && actual.im.is_finite());
    }

    // The public method is borrowing/pure; retaining the original arrays also
    // catches an implementation that mutates the source while reducing it.
    assert_eq!(source.s(), &source_s);
    assert_eq!(source.z0(), &source_z0);
    assert_eq!(source.frequency().hz(), data.frequency_hz.as_slice());
}

#[test]
fn pinned_scikit_rf_mixed_open_termination_fixture_matches_public_network_method() {
    let fixture: MixedOpenFixtureDocument =
        serde_json::from_str(MIXED_OPEN_FIXTURE_JSON).expect("mixed-open fixture must parse");
    let metadata = fixture.metadata;
    let data = fixture.data;

    assert_eq!(metadata.schema, EXPECTED_MIXED_OPEN_SCHEMA);
    assert_eq!(metadata.schema_version, EXPECTED_MIXED_OPEN_SCHEMA_VERSION);
    assert_eq!(metadata.case_id, EXPECTED_MIXED_OPEN_CASE_ID);
    assert_eq!(metadata.operation, EXPECTED_MIXED_OPEN_OPERATION);
    assert_eq!(metadata.input_recipe, EXPECTED_MIXED_OPEN_INPUT_RECIPE);
    assert_eq!(metadata.numpy_version, EXPECTED_MIXED_OPEN_NUMPY_VERSION);
    assert_eq!(
        metadata.scikit_rf_version,
        EXPECTED_MIXED_OPEN_SCIKIT_RF_VERSION
    );
    assert_eq!(
        metadata.scikit_rf_commit,
        EXPECTED_MIXED_OPEN_SCIKIT_RF_COMMIT
    );
    assert_eq!(metadata.wave_definition, "power");
    assert_eq!(metadata.random_seed, 20_260_963);
    assert_eq!(metadata.port, EXPECTED_MIXED_OPEN_PORT);
    assert_eq!(
        metadata.port_order.description,
        "source ports except the selected port, in original order"
    );
    assert_eq!(metadata.port_order.input, vec![0, 1, 2, 3, 4]);
    assert_eq!(metadata.port_order.output, vec![0, 1, 3, 4]);
    assert!(!metadata.load_boundary.finite_only);
    assert!(metadata.load_boundary.includes_ideal_open);
    assert!(!metadata.load_boundary.includes_ideal_short);
    assert_eq!(metadata.load_boundary.load_excitation, "none");
    assert_eq!(
        metadata.load_boundary.open_sentinel,
        "explicit PortLoad::Open variant"
    );
    assert_eq!(metadata.load_boundary.unit, "ohm");
    assert!(metadata.reference_impedance.complex);
    assert!(metadata.reference_impedance.frequency_dependent);
    assert!(metadata.reference_impedance.per_port);
    assert!(metadata.reference_impedance.positive_real);
    assert_eq!(metadata.reference_impedance.source_field, "z0_ohm");
    assert_eq!(
        metadata.reference_impedance.survivor_field,
        "z0_survivor_ohm"
    );
    assert_eq!(metadata.reference_impedance.unit, "ohm");
    assert_eq!(metadata.shape.frequency, vec![EXPECTED_MIXED_OPEN_NFREQ]);
    assert_eq!(
        metadata.shape.input_s,
        vec![
            EXPECTED_MIXED_OPEN_NFREQ,
            EXPECTED_MIXED_OPEN_NPORTS,
            EXPECTED_MIXED_OPEN_NPORTS
        ]
    );
    assert_eq!(
        metadata.shape.input_z0,
        vec![EXPECTED_MIXED_OPEN_NFREQ, EXPECTED_MIXED_OPEN_NPORTS]
    );
    assert_eq!(metadata.shape.loads, vec![EXPECTED_MIXED_OPEN_NFREQ]);
    assert_eq!(
        metadata.shape.output_s,
        vec![EXPECTED_MIXED_OPEN_NFREQ, 4, 4]
    );
    assert_eq!(metadata.shape.output_z0, vec![EXPECTED_MIXED_OPEN_NFREQ, 4]);
    assert_eq!(metadata.tolerance_policy.rtol, EXPECTED_MIXED_OPEN_RTOL);
    assert_eq!(metadata.tolerance_policy.atol, EXPECTED_MIXED_OPEN_ATOL);
    assert!(
        metadata
            .tolerance_policy
            .comparison
            .contains("only data.s_terminated")
    );
    assert!(
        metadata
            .tolerance_policy
            .justification
            .contains("public skrf.network.connect")
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
            .contains("wave definitions")
    );
    assert_eq!(metadata.wave_definitions.connect_raw, "power");
    assert_eq!(metadata.wave_definitions.input, "power");
    assert_eq!(metadata.wave_definitions.load, "power");
    assert_eq!(metadata.wave_definitions.output, "power");
    assert_eq!(
        metadata.wave_definitions.restoration,
        "public result.renormalize(result.z0, s_def=\"power\")"
    );

    assert_eq!(data.frequency_hz.len(), EXPECTED_MIXED_OPEN_NFREQ);
    assert_eq!(data.loads.len(), EXPECTED_MIXED_OPEN_NFREQ);
    assert_eq!(data.s.len(), EXPECTED_MIXED_OPEN_NFREQ);
    assert_eq!(data.s_terminated.len(), EXPECTED_MIXED_OPEN_NFREQ);
    assert_eq!(data.z0_ohm.len(), EXPECTED_MIXED_OPEN_NFREQ);
    assert_eq!(data.z0_survivor_ohm.len(), EXPECTED_MIXED_OPEN_NFREQ);

    let loads = data
        .loads
        .iter()
        .map(|load| match (load.kind.as_str(), load.value.as_ref()) {
            ("open", None) => PortLoad::Open,
            ("impedance_ohm", Some(value)) => PortLoad::ImpedanceOhm(complex(value)),
            (kind, value) => panic!("invalid mixed-open load {kind:?} / {value:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(loads.len(), EXPECTED_MIXED_OPEN_NFREQ);
    assert_eq!(loads[0], PortLoad::Open);
    assert_eq!(loads[2], PortLoad::Open);
    assert_eq!(loads[1], PortLoad::ImpedanceOhm(Complex64::new(31.0, 7.0)));
    assert_eq!(loads[3], PortLoad::ImpedanceOhm(Complex64::new(-17.0, 4.0)));

    let source_s = mixed_open_array3(
        &data.s,
        EXPECTED_MIXED_OPEN_NFREQ,
        EXPECTED_MIXED_OPEN_NPORTS,
    );
    let source_z0 = mixed_open_array2(
        &data.z0_ohm,
        EXPECTED_MIXED_OPEN_NFREQ,
        EXPECTED_MIXED_OPEN_NPORTS,
    );
    let expected_s = mixed_open_array3(&data.s_terminated, EXPECTED_MIXED_OPEN_NFREQ, 4);
    let expected_survivor_z0 =
        mixed_open_array2(&data.z0_survivor_ohm, EXPECTED_MIXED_OPEN_NFREQ, 4);

    for (index, frequency) in data.frequency_hz.iter().enumerate() {
        assert!(frequency.is_finite(), "frequency {index} must be finite");
    }
    for ((frequency, port), value) in source_z0.indexed_iter() {
        assert!(value.re.is_finite() && value.im.is_finite());
        assert!(
            value.re > 0.0,
            "source z0 at ({frequency},{port}) must be positive-real"
        );
        assert!(value.im != 0.0, "source z0 must remain complex");
    }
    for ((frequency, row, column), value) in source_s.indexed_iter() {
        assert!(
            value.re.is_finite() && value.im.is_finite(),
            "source S at ({frequency},{row},{column}) must be finite"
        );
    }
    for ((frequency, row, column), value) in expected_s.indexed_iter() {
        assert!(
            value.re.is_finite() && value.im.is_finite(),
            "expected S at ({frequency},{row},{column}) must be finite"
        );
    }

    let frequency =
        Frequency::from_hz(data.frequency_hz.clone()).expect("mixed-open fixture frequency axis");
    let source = Network::new(frequency, source_s.clone(), source_z0.clone())
        .expect("mixed-open fixture source Network");
    let terminated = source
        .terminate_port_power(EXPECTED_MIXED_OPEN_PORT, &loads)
        .expect("mixed-open fixture load termination");

    assert_eq!(terminated.frequency().hz().len(), EXPECTED_MIXED_OPEN_NFREQ);
    for (index, actual) in terminated.frequency().hz().iter().enumerate() {
        assert_f64_bits(
            *actual,
            data.frequency_hz[index],
            &format!("mixed-open output frequency[{index}]"),
        );
    }
    assert_eq!(terminated.z0(), &expected_survivor_z0);
    assert_eq!(terminated.s().dim(), (EXPECTED_MIXED_OPEN_NFREQ, 4, 4));

    // Only the restored reduced S is numerically compared with the pinned
    // oracle.  Tagged loads, source arrays, labels, survivor references,
    // ordering, and wave-definition metadata remain exact contract fields.
    for ((frequency, row, column), actual) in terminated.s().indexed_iter() {
        assert_s_close(
            *actual,
            expected_s[[frequency, row, column]],
            &format!("mixed-open s_terminated[{frequency},{row},{column}]"),
        );
        assert!(actual.re.is_finite() && actual.im.is_finite());
    }
    assert_eq!(source.s(), &source_s);
    assert_eq!(source.z0(), &source_z0);
    assert_eq!(source.frequency().hz(), data.frequency_hz.as_slice());
}
