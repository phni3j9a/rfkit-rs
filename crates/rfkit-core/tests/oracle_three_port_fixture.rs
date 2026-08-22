use std::fmt;

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Error as NetworkError, Frequency, Network};
use serde::Deserialize;

const FIXTURE_JSON: &str =
    include_str!("../../../tools/oracle/fixtures/three_port_complex_z0.json");

const EXPECTED_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_CASE_ID: &str = "three_port_complex_z0";
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_OPERATION: &str = "network_fixture";
const EXPECTED_RANDOM_SEED: u64 = 20_250_308;
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_WAVE_DEFINITION: &str = "power";
const EXPECTED_FREQUENCY_SHAPE: &[usize] = &[4];
const EXPECTED_S_SHAPE: &[usize] = &[4, 3, 3];
const EXPECTED_Z0_SHAPE: &[usize] = &[4, 3];
const EXPECTED_TOLERANCE_COMPARISON: &str = "exact canonical UTF-8 JSON bytes";
const EXPECTED_TOLERANCE_FLOATING_POINT: &str =
    "IEEE-754 binary64 values serialized by Python json";
const EXPECTED_TOLERANCE_NUMERIC: &str = "not applicable to regeneration; downstream numerical comparisons must define operation-specific tolerances";

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
    z0_ohm: Vec<Vec<ComplexValue>>,
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
    numpy_version: String,
    operation: String,
    random_seed: u64,
    schema: String,
    schema_version: u32,
    scikit_rf_version: String,
    shape: DeclaredShape,
    tolerance_policy: TolerancePolicy,
    wave_definition: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeclaredShape {
    frequency: Vec<usize>,
    s: Vec<usize>,
    z0: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TolerancePolicy {
    comparison: String,
    floating_point: String,
    numeric_tolerance: String,
}

#[derive(Debug)]
enum FixtureError {
    Parse(serde_json::Error),
    MetadataMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    ShapeMismatch {
        field: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    Network(NetworkError),
}

impl fmt::Display for FixtureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => write!(formatter, "fixture JSON parse failed: {error}"),
            Self::MetadataMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "fixture metadata mismatch for {field}: expected {expected}, found {actual}"
            ),
            Self::ShapeMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "fixture shape mismatch for {field}: expected {expected:?}, found {actual:?}"
            ),
            Self::Network(error) => write!(formatter, "network construction failed: {error}"),
        }
    }
}

impl std::error::Error for FixtureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::Network(error) => Some(error),
            Self::MetadataMismatch { .. } | Self::ShapeMismatch { .. } => None,
        }
    }
}

fn parse_fixture(json: &str) -> Result<FixtureDocument, FixtureError> {
    serde_json::from_str(json).map_err(FixtureError::Parse)
}

fn expect_metadata<T>(field: &'static str, expected: T, actual: T) -> Result<(), FixtureError>
where
    T: fmt::Debug + PartialEq,
{
    if actual == expected {
        Ok(())
    } else {
        Err(FixtureError::MetadataMismatch {
            field,
            expected: format!("{expected:?}"),
            actual: format!("{actual:?}"),
        })
    }
}

fn validate_metadata(metadata: &FixtureMetadata) -> Result<(), FixtureError> {
    expect_metadata("metadata.schema", EXPECTED_SCHEMA, metadata.schema.as_str())?;
    expect_metadata(
        "metadata.schema_version",
        EXPECTED_SCHEMA_VERSION,
        metadata.schema_version,
    )?;
    expect_metadata(
        "metadata.case_id",
        EXPECTED_CASE_ID,
        metadata.case_id.as_str(),
    )?;
    expect_metadata(
        "metadata.numpy_version",
        EXPECTED_NUMPY_VERSION,
        metadata.numpy_version.as_str(),
    )?;
    expect_metadata(
        "metadata.operation",
        EXPECTED_OPERATION,
        metadata.operation.as_str(),
    )?;
    expect_metadata(
        "metadata.random_seed",
        EXPECTED_RANDOM_SEED,
        metadata.random_seed,
    )?;
    expect_metadata(
        "metadata.scikit_rf_version",
        EXPECTED_SCIKIT_RF_VERSION,
        metadata.scikit_rf_version.as_str(),
    )?;
    expect_metadata(
        "metadata.wave_definition",
        EXPECTED_WAVE_DEFINITION,
        metadata.wave_definition.as_str(),
    )?;
    expect_metadata(
        "metadata.shape.frequency",
        EXPECTED_FREQUENCY_SHAPE,
        metadata.shape.frequency.as_slice(),
    )?;
    expect_metadata(
        "metadata.shape.s",
        EXPECTED_S_SHAPE,
        metadata.shape.s.as_slice(),
    )?;
    expect_metadata(
        "metadata.shape.z0",
        EXPECTED_Z0_SHAPE,
        metadata.shape.z0.as_slice(),
    )?;
    expect_metadata(
        "metadata.tolerance_policy.comparison",
        EXPECTED_TOLERANCE_COMPARISON,
        metadata.tolerance_policy.comparison.as_str(),
    )?;
    expect_metadata(
        "metadata.tolerance_policy.floating_point",
        EXPECTED_TOLERANCE_FLOATING_POINT,
        metadata.tolerance_policy.floating_point.as_str(),
    )?;
    expect_metadata(
        "metadata.tolerance_policy.numeric_tolerance",
        EXPECTED_TOLERANCE_NUMERIC,
        metadata.tolerance_policy.numeric_tolerance.as_str(),
    )?;
    Ok(())
}

fn shape_mismatch(field: impl Into<String>, expected: &[usize], actual: &[usize]) -> FixtureError {
    FixtureError::ShapeMismatch {
        field: field.into(),
        expected: expected.to_vec(),
        actual: actual.to_vec(),
    }
}

fn validate_data_shape(data: &FixtureData, declared: &DeclaredShape) -> Result<(), FixtureError> {
    if data.frequency_hz.len() != declared.frequency[0] {
        return Err(shape_mismatch(
            "data.frequency_hz",
            &declared.frequency,
            &[data.frequency_hz.len()],
        ));
    }

    if data.s.len() != declared.s[0] {
        return Err(shape_mismatch("data.s", &declared.s, &[data.s.len()]));
    }
    for (frequency_index, matrix) in data.s.iter().enumerate() {
        if matrix.len() != declared.s[1] {
            return Err(shape_mismatch(
                format!("data.s[{frequency_index}]"),
                &declared.s[1..],
                &[matrix.len()],
            ));
        }
        for (row_index, row) in matrix.iter().enumerate() {
            if row.len() != declared.s[2] {
                return Err(shape_mismatch(
                    format!("data.s[{frequency_index}][{row_index}]"),
                    &declared.s[2..],
                    &[row.len()],
                ));
            }
        }
    }

    if data.z0_ohm.len() != declared.z0[0] {
        return Err(shape_mismatch(
            "data.z0_ohm",
            &declared.z0,
            &[data.z0_ohm.len()],
        ));
    }
    for (frequency_index, row) in data.z0_ohm.iter().enumerate() {
        if row.len() != declared.z0[1] {
            return Err(shape_mismatch(
                format!("data.z0_ohm[{frequency_index}]"),
                &declared.z0[1..],
                &[row.len()],
            ));
        }
    }

    Ok(())
}

fn fixture_to_network(fixture: &FixtureDocument) -> Result<Network, FixtureError> {
    validate_metadata(&fixture.metadata)?;
    validate_data_shape(&fixture.data, &fixture.metadata.shape)?;

    let [nfreq, nport, _] = <[usize; 3]>::try_from(fixture.metadata.shape.s.as_slice())
        .expect("metadata shape was validated against the fixture contract");
    let frequency =
        Frequency::from_hz(fixture.data.frequency_hz.clone()).map_err(FixtureError::Network)?;

    let s = Array3::from_shape_fn((nfreq, nport, nport), |(frequency, row, column)| {
        let value = &fixture.data.s[frequency][row][column];
        Complex64::new(value.real, value.imag)
    });
    let z0 = Array2::from_shape_fn((nfreq, nport), |(frequency, port)| {
        let value = &fixture.data.z0_ohm[frequency][port];
        Complex64::new(value.real, value.imag)
    });

    Network::new(frequency, s, z0).map_err(FixtureError::Network)
}

#[test]
fn consumes_checked_in_fixture_and_maps_every_value_exactly() {
    let fixture = parse_fixture(FIXTURE_JSON).expect("checked-in fixture must parse");
    let network = fixture_to_network(&fixture).expect("fixture must map to Network");

    assert_eq!(
        network.nports(),
        3,
        "fixture contract must remain three-port"
    );
    assert_eq!(network.frequency().len(), 4);
    assert!(
        network.z0().iter().any(|value| value.im != 0.0),
        "fixture must exercise complex reference impedances"
    );

    for (frequency_index, expected_frequency) in fixture.data.frequency_hz.iter().enumerate() {
        assert_eq!(
            network.frequency().hz()[frequency_index],
            *expected_frequency,
            "frequency sample {frequency_index} changed during mapping"
        );
    }
    for (frequency_index, matrix) in fixture.data.s.iter().enumerate() {
        for (row_index, row) in matrix.iter().enumerate() {
            for (column_index, expected) in row.iter().enumerate() {
                assert_eq!(
                    network.s()[[frequency_index, row_index, column_index]],
                    Complex64::new(expected.real, expected.imag),
                    "S[{frequency_index},{row_index},{column_index}] changed during mapping"
                );
            }
        }
    }
    for (frequency_index, row) in fixture.data.z0_ohm.iter().enumerate() {
        for (port_index, expected) in row.iter().enumerate() {
            assert_eq!(
                network.z0()[[frequency_index, port_index]],
                Complex64::new(expected.real, expected.imag),
                "z0[{frequency_index},{port_index}] changed during mapping"
            );
        }
    }
}

#[test]
fn rejects_frequency_shape_mismatch_without_truncating_data() {
    let mut fixture = parse_fixture(FIXTURE_JSON).expect("checked-in fixture must parse");
    fixture.data.frequency_hz.pop();

    let error = fixture_to_network(&fixture).expect_err("short frequency data must fail");
    assert_eq!(
        error.to_string(),
        "fixture shape mismatch for data.frequency_hz: expected [4], found [3]"
    );
}

#[test]
fn rejects_ragged_s_and_z0_shapes_with_the_failing_dimension() {
    let mut fixture = parse_fixture(FIXTURE_JSON).expect("checked-in fixture must parse");
    fixture.data.s[1].pop();
    let error = fixture_to_network(&fixture).expect_err("ragged S data must fail");
    assert_eq!(
        error.to_string(),
        "fixture shape mismatch for data.s[1]: expected [3, 3], found [2]"
    );

    let mut fixture = parse_fixture(FIXTURE_JSON).expect("checked-in fixture must parse");
    fixture.data.z0_ohm[2].pop();
    let error = fixture_to_network(&fixture).expect_err("short z0 row must fail");
    assert_eq!(
        error.to_string(),
        "fixture shape mismatch for data.z0_ohm[2]: expected [3], found [2]"
    );
}

#[test]
fn rejects_unknown_schema_fields_instead_of_defaulting_them() {
    let mut document: serde_json::Value =
        serde_json::from_str(FIXTURE_JSON).expect("checked-in fixture must parse");
    document["metadata"]["unexpected_field"] = serde_json::Value::Bool(true);

    let error = serde_json::from_value::<FixtureDocument>(document)
        .expect_err("unknown metadata fields must fail schema parsing");
    assert!(
        error
            .to_string()
            .contains("unknown field `unexpected_field`"),
        "unexpected schema error: {error}"
    );
}
