use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/reference_plane_shift_lossless_power_four_port_media_connect.json"
);

const EXPECTED_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_CASE_ID: &str = "reference_plane_shift_lossless_power_four_port_media_connect";
const EXPECTED_OPERATION: &str = "network_shift_reference_planes_lossless_power";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_RANDOM_SEED: u64 = 20_260_965;
const EXPECTED_NFREQ: usize = 3;
const EXPECTED_NPORTS: usize = 4;
const EXPECTED_FREQUENCY_HZ: [f64; EXPECTED_NFREQ] = [0.9e9, 1.6e9, 2.8e9];
const EXPECTED_PHASE_RAD: [[f64; EXPECTED_NPORTS]; EXPECTED_NFREQ] = [
    [0.0, 0.2, -0.3, 0.7],
    [0.4, -0.8, 0.15, 1.2],
    [-0.9, 0.3, 0.6, -0.4],
];
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
    one_way_phase_rad: Vec<Vec<f64>>,
    s_input: Vec<Vec<Vec<ComplexValue>>>,
    s_shifted: Vec<Vec<Vec<ComplexValue>>>,
    z0_input_ohm: Vec<Vec<ComplexValue>>,
    z0_output_ohm: Vec<Vec<ComplexValue>>,
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
    line_construction: LineConstruction,
    numpy_version: String,
    operation: String,
    output_port_mapping: Vec<String>,
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
struct LineConstruction {
    characteristic_impedance: String,
    length: f64,
    mapping: String,
    medium: String,
    s_def: String,
    unit: String,
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
struct DeclaredShape {
    frequency: Vec<usize>,
    input_s: Vec<usize>,
    input_z0: Vec<usize>,
    one_way_phase_rad: Vec<usize>,
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
struct Units {
    frequency: String,
    one_way_phase: String,
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

fn phase_array(values: &[Vec<f64>]) -> Array2<f64> {
    assert_eq!(values.len(), EXPECTED_NFREQ);
    assert!(values.iter().all(|row| row.len() == EXPECTED_NPORTS));
    Array2::from_shape_fn((EXPECTED_NFREQ, EXPECTED_NPORTS), |(frequency, port)| {
        values[frequency][port]
    })
}

fn assert_close(actual: Complex64, expected: Complex64, context: &str) {
    for (actual, expected, component) in [
        (actual.re, expected.re, "real"),
        (actual.im, expected.im, "imag"),
    ] {
        let difference = (actual - expected).abs();
        let bound = EXPECTED_ATOL + EXPECTED_RTOL * expected.abs();
        assert!(
            difference <= bound,
            "{context}.{component}: actual={actual:.17e}, expected={expected:.17e}, difference={difference:.17e}, bound={bound:.17e}"
        );
    }
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
    assert_eq!(
        metadata.input_recipe,
        "independent asymmetric four-port S stack from NumPy default_rng seed 20260965; frequencies [0.9,1.6,2.8] GHz; unequal real-positive frequency-dependent per-port references; signed one-way phase table [[0,0.2,-0.3,0.7],[0.4,-0.8,0.15,1.2],[-0.9,0.3,0.6,-0.4]] rad; expected output from public DefinedGammaZ0.line plus sequential network.connect; no prior fixture values reused"
    );
    assert_eq!(
        (
            metadata.reference_impedance.complex,
            metadata.reference_impedance.frequency_dependent,
            metadata.reference_impedance.per_port,
            metadata.reference_impedance.real_part.as_str(),
            metadata.reference_impedance.unit.as_str(),
        ),
        (false, true, true, "strictly positive", "ohm")
    );
    assert_eq!(
        metadata.output_port_mapping,
        vec![
            "line_0".to_owned(),
            "line_1".to_owned(),
            "line_2".to_owned(),
            "line_3".to_owned()
        ]
    );
    assert_eq!(
        (
            metadata.line_construction.characteristic_impedance.as_str(),
            metadata.line_construction.length,
            metadata.line_construction.mapping.as_str(),
            metadata.line_construction.medium.as_str(),
            metadata.line_construction.s_def.as_str(),
            metadata.line_construction.unit.as_str(),
        ),
        (
            "source z0[f,port]",
            1.0,
            "sequential connect(current, current_index, line, 0); each line survivor remains at the replaced physical port index",
            "DefinedGammaZ0",
            "power",
            "m",
        )
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
        metadata.shape.one_way_phase_rad,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS]
    );
    assert_eq!(
        metadata.shape.output_s,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS, EXPECTED_NPORTS]
    );
    assert_eq!(
        metadata.shape.output_z0,
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS]
    );
    assert_eq!(metadata.tolerance_policy.rtol, EXPECTED_RTOL);
    assert_eq!(metadata.tolerance_policy.atol, EXPECTED_ATOL);
    assert!(
        metadata
            .tolerance_policy
            .comparison
            .contains("only data.s_shifted is numeric output")
    );
    assert!(
        metadata
            .tolerance_policy
            .justification
            .contains("lossless-reference-plane transformation")
    );
    assert!(
        metadata
            .tolerance_policy
            .regeneration
            .contains("DefinedGammaZ0.line")
    );
    assert!(
        metadata
            .tolerance_policy
            .regeneration
            .contains("D S D identity check")
    );
    assert_eq!(
        (
            metadata.units.frequency.as_str(),
            metadata.units.one_way_phase.as_str(),
            metadata.units.s.as_str(),
            metadata.units.z0.as_str(),
        ),
        ("Hz", "rad", "dimensionless", "ohm")
    );
}

#[test]
fn public_reference_plane_shift_matches_pinned_media_connect_fixture() {
    let fixture: FixtureDocument = serde_json::from_str(FIXTURE_JSON).unwrap();
    validate_contract(&fixture);
    assert_eq!(fixture.data.frequency_hz.len(), EXPECTED_NFREQ);
    assert_eq!(fixture.data.one_way_phase_rad.len(), EXPECTED_NFREQ);
    assert_eq!(fixture.data.s_input.len(), EXPECTED_NFREQ);
    assert_eq!(fixture.data.s_shifted.len(), EXPECTED_NFREQ);
    assert_eq!(fixture.data.z0_input_ohm.len(), EXPECTED_NFREQ);
    assert_eq!(fixture.data.z0_output_ohm.len(), EXPECTED_NFREQ);

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
    for (frequency, expected_row) in EXPECTED_PHASE_RAD.iter().enumerate() {
        for (port, expected) in expected_row.iter().enumerate() {
            assert_eq!(
                fixture.data.one_way_phase_rad[frequency][port].to_bits(),
                expected.to_bits(),
                "one_way_phase_rad[{frequency}][{port}]"
            );
        }
    }

    let source_frequency = fixture.data.frequency_hz.clone();
    let source_s = array3(&fixture.data.s_input);
    let source_z0 = array2(&fixture.data.z0_input_ohm);
    let phase = phase_array(&fixture.data.one_way_phase_rad);
    let network = Network::new(
        Frequency::from_hz(source_frequency.clone()).unwrap(),
        source_s.clone(),
        source_z0.clone(),
    )
    .unwrap();
    let snapshot = network.clone();
    let shifted = network
        .shift_reference_planes_lossless_power(&phase)
        .unwrap();
    let expected_s = array3(&fixture.data.s_shifted);
    let expected_z0 = array2(&fixture.data.z0_output_ohm);

    assert_eq!(
        shifted.s().dim(),
        (EXPECTED_NFREQ, EXPECTED_NPORTS, EXPECTED_NPORTS)
    );
    for (index, (actual, expected)) in shifted.s().iter().zip(expected_s.iter()).enumerate() {
        assert_close(*actual, *expected, &format!("s_shifted[{index}]"));
        assert!(actual.re.is_finite());
        assert!(actual.im.is_finite());
    }
    assert_eq!(shifted.z0(), &expected_z0);
    assert_eq!(shifted.z0(), &source_z0);
    assert_eq!(shifted.frequency().hz(), source_frequency.as_slice());
    assert_eq!(network, snapshot);
    assert_eq!(network.frequency().hz(), source_frequency.as_slice());
    assert_eq!(network.s(), &source_s);
    assert_eq!(network.z0(), &source_z0);
}
