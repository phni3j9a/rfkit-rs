use num_complex::Complex64;
use rfkit_touchstone::parse_touchstone_v1_0_s;
use serde::Deserialize;

const FIXTURE: &str =
    include_str!("../../../tools/oracle/fixtures/touchstone_v1_0_s_three_port.json");

#[derive(Debug, Deserialize)]
struct Fixture {
    metadata: Metadata,
    data: Data,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    case_id: String,
    operation: String,
    port_count: usize,
    scikit_rf_version: String,
    tolerance_policy: TolerancePolicy,
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
fn pinned_scikit_rf_touchstone_fixture_matches_public_parser() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid canonical fixture");
    assert_eq!(fixture.metadata.case_id, "touchstone_v1_0_s_three_port");
    assert_eq!(fixture.metadata.operation, "touchstone_v1_0_s_parse");
    assert_eq!(fixture.metadata.port_count, fixture.data.nports);
    assert_eq!(fixture.metadata.scikit_rf_version, "2.0.1");
    assert!(fixture.metadata.tolerance_policy.rtol.is_finite());
    assert!(fixture.metadata.tolerance_policy.atol.is_finite());
    assert!(fixture.metadata.tolerance_policy.rtol >= 0.0);
    assert!(fixture.metadata.tolerance_policy.atol >= 0.0);

    let network = parse_touchstone_v1_0_s(&fixture.data.touchstone_text, fixture.data.nports)
        .expect("Touchstone fixture must parse");
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
                let difference = (actual - expected).norm();
                let bound = fixture.metadata.tolerance_policy.atol
                    + fixture.metadata.tolerance_policy.rtol * expected.norm();
                assert!(
                    difference <= bound,
                    "S[{frequency},{row},{column}] differs by {difference:?}, allowed {bound:?}"
                );
            }
        }
    }
    for (frequency, row) in fixture.data.z0_ohm.iter().enumerate() {
        for (port, expected) in row.iter().enumerate() {
            let actual = network.z0()[[frequency, port]];
            let expected = Complex64::from(expected);
            let difference = (actual - expected).norm();
            let bound = fixture.metadata.tolerance_policy.atol
                + fixture.metadata.tolerance_policy.rtol * expected.norm();
            assert!(
                difference <= bound,
                "z0[{frequency},{port}] differs by {difference:?}, allowed {bound:?}"
            );
        }
    }
}
