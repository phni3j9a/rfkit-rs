use num_complex::Complex64;
use rfkit_touchstone::parse_touchstone_v2_0_power;
use serde::Deserialize;

const Z_FIXTURE: &str =
    include_str!("../../../tools/oracle/fixtures/touchstone_v2_0_z_full_three_port.json");
const Y_FIXTURE: &str =
    include_str!("../../../tools/oracle/fixtures/touchstone_v2_0_y_full_three_port.json");

#[derive(Debug, Deserialize)]
struct Fixture {
    metadata: Metadata,
    data: Data,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    case_id: String,
    input_parameter: String,
    input_recipe: String,
    parameter_unit: String,
    numpy_version: String,
    operation: String,
    port_count: usize,
    scikit_rf_commit: String,
    scikit_rf_version: String,
    shape: Shape,
    tolerance_policy: TolerancePolicy,
}

#[derive(Debug, Deserialize)]
struct Shape {
    frequency: Vec<usize>,
    parameter: Vec<usize>,
    s: Vec<usize>,
    z0: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct TolerancePolicy {
    rtol: f64,
    atol: f64,
}

#[derive(Debug, Deserialize)]
struct Data {
    frequency_hz: Vec<f64>,
    nports: usize,
    parameter: String,
    parameter_matrix: Vec<Vec<Vec<ComplexValue>>>,
    s: Vec<Vec<Vec<ComplexValue>>>,
    touchstone_text: String,
    z0_ohm: Vec<Vec<ComplexValue>>,
}

#[derive(Debug, Deserialize)]
struct ComplexValue {
    real: f64,
    imag: f64,
}

impl From<&ComplexValue> for Complex64 {
    fn from(value: &ComplexValue) -> Self {
        Self::new(value.real, value.imag)
    }
}

#[test]
fn pinned_scikit_rf_z_reader_fixture_matches_power_parser_and_physical_z() {
    assert_fixture(
        Z_FIXTURE,
        "touchstone_v2_0_z_full_three_port",
        "Z",
        "touchstone_v2_0_z_full_parse",
    );
}

#[test]
fn pinned_scikit_rf_y_reader_fixture_matches_power_parser_and_physical_y() {
    assert_fixture(
        Y_FIXTURE,
        "touchstone_v2_0_y_full_three_port",
        "Y",
        "touchstone_v2_0_y_full_parse",
    );
}

fn assert_fixture(
    json: &str,
    expected_case: &str,
    expected_parameter: &str,
    expected_operation: &str,
) {
    let fixture: Fixture = serde_json::from_str(json).expect("valid physical Touchstone fixture");
    assert_eq!(fixture.metadata.case_id, expected_case);
    assert_eq!(fixture.metadata.input_parameter, expected_parameter);
    assert_eq!(
        fixture.metadata.parameter_unit,
        if expected_parameter == "Z" {
            "ohm"
        } else {
            "siemens"
        }
    );
    assert_eq!(fixture.metadata.operation, expected_operation);
    assert_eq!(fixture.metadata.port_count, fixture.data.nports);
    assert_eq!(fixture.data.parameter, expected_parameter);
    assert_eq!(fixture.metadata.numpy_version, "2.5.1");
    assert_eq!(fixture.metadata.scikit_rf_version, "2.0.1");
    assert_eq!(
        fixture.metadata.scikit_rf_commit,
        "bd651e923cac6020de49a096e1d7e9b5f949f884"
    );
    assert!(!fixture.metadata.input_recipe.is_empty());
    assert!(!fixture.data.touchstone_text.is_empty());
    assert_eq!(
        fixture.metadata.shape.frequency,
        vec![fixture.data.frequency_hz.len()]
    );
    assert_eq!(
        fixture.metadata.shape.parameter,
        vec![
            fixture.data.parameter_matrix.len(),
            fixture.data.nports,
            fixture.data.nports
        ]
    );
    assert_eq!(
        fixture.metadata.shape.s,
        vec![
            fixture.data.s.len(),
            fixture.data.nports,
            fixture.data.nports
        ]
    );
    assert_eq!(
        fixture.metadata.shape.z0,
        vec![fixture.data.z0_ohm.len(), fixture.data.nports]
    );
    assert!((fixture.metadata.tolerance_policy.rtol - 1.0e-12).abs() < f64::EPSILON);
    assert!((fixture.metadata.tolerance_policy.atol - 1.0e-12).abs() < f64::EPSILON);

    let network = parse_touchstone_v2_0_power(&fixture.data.touchstone_text)
        .expect("physical Touchstone v2 fixture must parse");
    assert_eq!(
        network.frequency().hz(),
        fixture.data.frequency_hz.as_slice()
    );
    assert_eq!(
        network.s().dim(),
        (
            fixture.data.s.len(),
            fixture.data.nports,
            fixture.data.nports
        )
    );
    assert_eq!(
        network.z0().dim(),
        (fixture.data.z0_ohm.len(), fixture.data.nports)
    );

    for (frequency, matrix) in fixture.data.s.iter().enumerate() {
        for (row, values) in matrix.iter().enumerate() {
            for (column, expected) in values.iter().enumerate() {
                let actual = network.s()[[frequency, row, column]];
                let expected = Complex64::from(expected);
                let bound = fixture.metadata.tolerance_policy.atol
                    + fixture.metadata.tolerance_policy.rtol * expected.norm();
                assert!(
                    (actual - expected).norm() <= bound,
                    "S[{frequency},{row},{column}] differs: actual={actual:?}, expected={expected:?}, bound={bound:?}"
                );
            }
        }
    }
    for (frequency, row) in fixture.data.z0_ohm.iter().enumerate() {
        for (port, expected) in row.iter().enumerate() {
            assert_eq!(
                network.z0()[[frequency, port]],
                Complex64::from(expected),
                "z0[{frequency},{port}] differs"
            );
        }
    }

    let physical = if expected_parameter == "Z" {
        network.to_z_power().expect("Z output extraction succeeds")
    } else {
        network
            .to_y_direct_power()
            .expect("direct Y output extraction succeeds")
    };
    for (frequency, matrix) in fixture.data.parameter_matrix.iter().enumerate() {
        for (row, values) in matrix.iter().enumerate() {
            for (column, expected) in values.iter().enumerate() {
                let expected = Complex64::from(expected);
                assert!(
                    (physical[[frequency, row, column]] - expected).norm() < 1.0e-11,
                    "{expected_parameter}[{frequency},{row},{column}] differs"
                );
            }
        }
    }
}
