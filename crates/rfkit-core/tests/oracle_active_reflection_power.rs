use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str =
    include_str!("../../../tools/oracle/fixtures/active_reflection_power_four_port_real_z0.json");

const EXPECTED_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_CASE_ID: &str = "active_reflection_power_four_port_real_z0";
const EXPECTED_OPERATION: &str = "network_active_reflection_power";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_RANDOM_SEED: u64 = 20_260_966;
const EXPECTED_RTOL: f64 = 1.0e-12;
const EXPECTED_ATOL: f64 = 1.0e-12;

#[derive(Debug, Deserialize)]
struct FixtureDocument {
    data: FixtureData,
    metadata: FixtureMetadata,
}

#[derive(Debug, Deserialize)]
struct FixtureData {
    active_reflection: Vec<Vec<ComplexValue>>,
    frequency_hz: Vec<f64>,
    incident: Vec<Vec<ComplexValue>>,
    s_input: Vec<Vec<Vec<ComplexValue>>>,
    z0_input_ohm: Vec<Vec<ComplexValue>>,
}

#[derive(Debug, Deserialize)]
struct ComplexValue {
    imag: f64,
    real: f64,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    case_id: String,
    drive_definition: String,
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
    zero_policy: String,
}

#[derive(Debug, Deserialize)]
struct ReferenceImpedance {
    complex: bool,
    frequency_dependent: bool,
    per_port: bool,
    real_part: String,
    unit: String,
}

#[derive(Debug, Deserialize)]
struct DeclaredShape {
    frequency: Vec<usize>,
    input_incident: Vec<usize>,
    input_s: Vec<usize>,
    input_z0: Vec<usize>,
    output_active_reflection: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct TolerancePolicy {
    atol: f64,
    comparison: String,
    justification: String,
    mutation_guard: String,
    regeneration: String,
    rtol: f64,
}

#[derive(Debug, Deserialize)]
struct Units {
    active_reflection: String,
    frequency: String,
    incident: String,
    s: String,
    z0: String,
}

fn complex(value: &ComplexValue) -> Complex64 {
    Complex64::new(value.real, value.imag)
}

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
    let nfreq = values.len();
    let nport = values.first().map_or(0, Vec::len);
    assert!(nfreq > 0);
    assert!(nport > 0);
    assert!(values.iter().all(|matrix| matrix.len() == nport));
    assert!(
        values
            .iter()
            .flat_map(|matrix| matrix.iter())
            .all(|row| row.len() == nport)
    );
    Array3::from_shape_fn((nfreq, nport, nport), |(frequency, row, column)| {
        complex(&values[frequency][row][column])
    })
}

fn array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    let nfreq = values.len();
    let nport = values.first().map_or(0, Vec::len);
    assert!(nfreq > 0);
    assert!(nport > 0);
    assert!(values.iter().all(|row| row.len() == nport));
    Array2::from_shape_fn((nfreq, nport), |(frequency, port)| {
        complex(&values[frequency][port])
    })
}

fn assert_close(actual: Complex64, expected: Complex64, context: &str) {
    let difference = (actual - expected).norm();
    let bound = EXPECTED_ATOL + EXPECTED_RTOL * expected.norm();
    assert!(
        difference <= bound,
        "{context}: actual={actual:.17e}, expected={expected:.17e}, difference={difference:.17e}, bound={bound:.17e}"
    );
}

fn validate_contract(fixture: &FixtureDocument, nfreq: usize, nport: usize) {
    let metadata = &fixture.metadata;
    assert_eq!(metadata.schema, EXPECTED_SCHEMA);
    assert_eq!(metadata.schema_version, 1);
    assert_eq!(metadata.case_id, EXPECTED_CASE_ID);
    assert_eq!(metadata.operation, EXPECTED_OPERATION);
    assert_eq!(metadata.numpy_version, EXPECTED_NUMPY_VERSION);
    assert_eq!(metadata.scikit_rf_version, EXPECTED_SCIKIT_RF_VERSION);
    assert_eq!(metadata.scikit_rf_commit, EXPECTED_SCIKIT_RF_COMMIT);
    assert_eq!(metadata.random_seed, EXPECTED_RANDOM_SEED);
    assert_eq!(metadata.wave_definition, "power");
    assert!(
        metadata
            .drive_definition
            .contains("incident Kurokawa power-wave amplitudes")
    );
    assert!(metadata.input_recipe.contains("default_rng seed 20260966"));
    assert!(
        metadata
            .input_recipe
            .contains("every incident component is finite and nonzero")
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
    assert_eq!(metadata.shape.frequency, vec![nfreq]);
    assert_eq!(metadata.shape.input_s, vec![nfreq, nport, nport]);
    assert_eq!(metadata.shape.input_z0, vec![nfreq, nport]);
    assert_eq!(metadata.shape.input_incident, vec![nfreq, nport]);
    assert_eq!(metadata.shape.output_active_reflection, vec![nfreq, nport]);
    assert_eq!(metadata.tolerance_policy.rtol, EXPECTED_RTOL);
    assert_eq!(metadata.tolerance_policy.atol, EXPECTED_ATOL);
    assert!(
        metadata
            .tolerance_policy
            .comparison
            .contains("active_reflection")
    );
    assert!(
        metadata
            .tolerance_policy
            .justification
            .contains("Network.s_active")
    );
    assert!(
        metadata
            .tolerance_policy
            .mutation_guard
            .contains("fresh copy")
    );
    assert!(metadata.tolerance_policy.regeneration.contains("S @ a / a"));
    assert_eq!(
        (
            metadata.units.active_reflection.as_str(),
            metadata.units.frequency.as_str(),
            metadata.units.incident.as_str(),
            metadata.units.s.as_str(),
            metadata.units.z0.as_str(),
        ),
        (
            "dimensionless",
            "Hz",
            "sqrt(W), common arbitrary normalization",
            "dimensionless",
            "ohm",
        )
    );
    assert!(metadata.zero_policy.contains("no exact zero"));
}

#[test]
fn public_active_reflection_matches_pinned_scikit_rf_fixture() {
    let fixture: FixtureDocument =
        serde_json::from_str(FIXTURE_JSON).expect("active-reflection fixture must parse");
    let nfreq = fixture.data.frequency_hz.len();
    let nport = fixture.data.incident.first().map_or(0, Vec::len);
    validate_contract(&fixture, nfreq, nport);

    assert_eq!(fixture.data.active_reflection.len(), nfreq);
    assert_eq!(fixture.data.incident.len(), nfreq);
    let source_s = array3(&fixture.data.s_input);
    let source_z0 = array2(&fixture.data.z0_input_ohm);
    let incident = array2(&fixture.data.incident);
    let source_frequency = fixture.data.frequency_hz.clone();
    assert!(
        incident
            .iter()
            .all(|value| *value != Complex64::new(0.0, 0.0))
    );

    let network = Network::new(
        Frequency::from_hz(source_frequency).unwrap(),
        source_s.clone(),
        source_z0,
    )
    .unwrap();
    let network_snapshot = network.clone();
    let incident_snapshot = incident.clone();
    let actual = network
        .active_reflection_power(&incident)
        .expect("active-reflection oracle input must be accepted");

    assert_eq!(actual.dim(), (nfreq, nport));
    for frequency in 0..nfreq {
        for port in 0..nport {
            let expected = complex(&fixture.data.active_reflection[frequency][port]);
            let value =
                actual[[frequency, port]].expect("canonical all-nonzero drive must produce Some");
            assert_close(
                value,
                expected,
                &format!("active_reflection[{frequency},{port}]"),
            );
            assert!(value.re.is_finite() && value.im.is_finite());
        }
    }

    assert_eq!(incident, incident_snapshot);
    assert_eq!(network, network_snapshot);
    assert!(
        (actual[[0, 0]].unwrap() - source_s[[0, 0, 0]]).norm() > 1.0e-6,
        "coherent drive must not collapse to the stored diagonal S entry"
    );
}
