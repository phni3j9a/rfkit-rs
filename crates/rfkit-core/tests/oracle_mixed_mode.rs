use std::fmt;

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use serde::Deserialize;

const FORWARD_FIXTURE_JSON: &str =
    include_str!("../../../tools/oracle/fixtures/mixed_mode_forward_five_port_complex_z0.json");
const INVERSE_FIXTURE_JSON: &str =
    include_str!("../../../tools/oracle/fixtures/mixed_mode_inverse_five_port_complex_z0.json");

const EXPECTED_SCHEMA: &str = "rfkit-rs.oracle.fixture";
const EXPECTED_SCHEMA_VERSION: u32 = 1;
const EXPECTED_NUMPY_VERSION: &str = "2.5.1";
const EXPECTED_SCIKIT_RF_VERSION: &str = "2.0.1";
const EXPECTED_SCIKIT_RF_COMMIT: &str = "bd651e923cac6020de49a096e1d7e9b5f949f884";
const EXPECTED_WAVE_DEFINITION: &str = "power";
const EXPECTED_PAIR_COUNT: usize = 2;
const EXPECTED_NFREQ: usize = 3;
const EXPECTED_NPORTS: usize = 5;
const EXPECTED_RTOL: f64 = 1.0e-12;
const EXPECTED_ATOL: f64 = 1.0e-12;
const FORWARD_INPUT_RECIPE: &str = "independent local NumPy default_rng input with a non-symmetric complex five-port S stack; adjacent pairs (0+,1-) and (2+,3-) have two distinct complex equal references at every frequency and port 4 is an unpaired complex reference; no inverse fixture data is reused";
const INVERSE_INPUT_RECIPE: &str = "independent local NumPy default_rng mixed-coordinate input with a non-symmetric complex five-port S stack; modal references are authored as (2z_pair,z_pair/2) for adjacent pairs and gmm2se receives an explicit (frequency,4) target z0_se array; no forward output is reused";

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
    s_input: Vec<Vec<Vec<ComplexValue>>>,
    z0_se_target_ohm: Option<Vec<Vec<ComplexValue>>>,
    z0_source_ohm: Vec<Vec<ComplexValue>>,
    z0_target_ohm: Vec<Vec<ComplexValue>>,
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
    coordinate_order: CoordinateOrder,
    direction: String,
    input_recipe: String,
    numpy_version: String,
    operation: String,
    pair_count: usize,
    pairing: Pairing,
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
struct CoordinateOrder {
    input: Vec<String>,
    output: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pairing {
    pairs: Vec<Pair>,
    polarity: String,
    unpaired_ports: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pair {
    negative: usize,
    positive: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceImpedance {
    natural_modal_relationship: String,
    output: ReferenceFlags,
    source: ReferenceFlags,
    source_field: String,
    target_field: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReferenceFlags {
    complex: bool,
    frequency_dependent: bool,
    per_port: bool,
    unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeclaredShape {
    frequency: Vec<usize>,
    input_s: Vec<usize>,
    input_z0: Vec<usize>,
    output_s: Vec<usize>,
    output_z0: Vec<usize>,
    target_z0_se: Option<Vec<usize>>,
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
    MetadataMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
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
        }
    }
}

impl std::error::Error for FixtureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::MetadataMismatch { .. } => None,
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

fn validate_metadata(
    metadata: &FixtureMetadata,
    case_id: &str,
    direction: &str,
    operation: &str,
    random_seed: u64,
    input_recipe: &str,
    inverse: bool,
) -> Result<(), FixtureError> {
    expect_metadata("metadata.schema", EXPECTED_SCHEMA, metadata.schema.as_str())?;
    expect_metadata(
        "metadata.schema_version",
        EXPECTED_SCHEMA_VERSION,
        metadata.schema_version,
    )?;
    expect_metadata("metadata.case_id", case_id, metadata.case_id.as_str())?;
    expect_metadata("metadata.direction", direction, metadata.direction.as_str())?;
    expect_metadata("metadata.operation", operation, metadata.operation.as_str())?;
    expect_metadata(
        "metadata.input_recipe",
        input_recipe,
        metadata.input_recipe.as_str(),
    )?;
    expect_metadata("metadata.random_seed", random_seed, metadata.random_seed)?;
    expect_metadata(
        "metadata.numpy_version",
        EXPECTED_NUMPY_VERSION,
        metadata.numpy_version.as_str(),
    )?;
    expect_metadata(
        "metadata.scikit_rf_version",
        EXPECTED_SCIKIT_RF_VERSION,
        metadata.scikit_rf_version.as_str(),
    )?;
    expect_metadata(
        "metadata.scikit_rf_commit",
        EXPECTED_SCIKIT_RF_COMMIT,
        metadata.scikit_rf_commit.as_str(),
    )?;
    expect_metadata(
        "metadata.wave_definition",
        EXPECTED_WAVE_DEFINITION,
        metadata.wave_definition.as_str(),
    )?;
    expect_metadata(
        "metadata.pair_count",
        EXPECTED_PAIR_COUNT,
        metadata.pair_count,
    )?;
    expect_metadata(
        "metadata.coordinate_order.input",
        if inverse {
            ["d0", "d1", "c0", "c1", "se4"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        } else {
            ["se0", "se1", "se2", "se3", "se4"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        },
        metadata.coordinate_order.input.clone(),
    )?;
    expect_metadata(
        "metadata.coordinate_order.output",
        if inverse {
            ["se0", "se1", "se2", "se3", "se4"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        } else {
            ["d0", "d1", "c0", "c1", "se4"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        },
        metadata.coordinate_order.output.clone(),
    )?;
    expect_metadata(
        "metadata.pairing.unpaired_ports",
        vec![4],
        metadata.pairing.unpaired_ports.clone(),
    )?;
    expect_metadata(
        "metadata.pairing.polarity",
        "positive_then_negative; (0+,1-) and (2+,3-)",
        metadata.pairing.polarity.as_str(),
    )?;
    if metadata.pairing.pairs.len() != EXPECTED_PAIR_COUNT
        || metadata.pairing.pairs[0].positive != 0
        || metadata.pairing.pairs[0].negative != 1
        || metadata.pairing.pairs[1].positive != 2
        || metadata.pairing.pairs[1].negative != 3
    {
        return Err(FixtureError::MetadataMismatch {
            field: "metadata.pairing.pairs",
            expected: "[(positive=0,negative=1),(positive=2,negative=3)]".to_owned(),
            actual: format!("{:?}", metadata.pairing.pairs.len()),
        });
    }
    expect_metadata(
        "metadata.reference_impedance.natural_modal_relationship",
        "d=2*z_pair; c=z_pair/2",
        metadata
            .reference_impedance
            .natural_modal_relationship
            .as_str(),
    )?;
    for (field, flags) in [
        (
            "metadata.reference_impedance.source",
            &metadata.reference_impedance.source,
        ),
        (
            "metadata.reference_impedance.output",
            &metadata.reference_impedance.output,
        ),
    ] {
        expect_metadata(field, true, flags.complex)?;
        expect_metadata(field, true, flags.frequency_dependent)?;
        expect_metadata(field, true, flags.per_port)?;
        expect_metadata(field, "ohm", flags.unit.as_str())?;
    }
    expect_metadata(
        "metadata.reference_impedance.source_field",
        "z0_source_ohm",
        metadata.reference_impedance.source_field.as_str(),
    )?;
    expect_metadata(
        "metadata.reference_impedance.target_field",
        "z0_target_ohm",
        metadata.reference_impedance.target_field.as_str(),
    )?;
    expect_metadata(
        "metadata.shape.frequency",
        vec![EXPECTED_NFREQ],
        metadata.shape.frequency.clone(),
    )?;
    expect_metadata(
        "metadata.shape.input_s",
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS, EXPECTED_NPORTS],
        metadata.shape.input_s.clone(),
    )?;
    expect_metadata(
        "metadata.shape.input_z0",
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS],
        metadata.shape.input_z0.clone(),
    )?;
    expect_metadata(
        "metadata.shape.output_s",
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS, EXPECTED_NPORTS],
        metadata.shape.output_s.clone(),
    )?;
    expect_metadata(
        "metadata.shape.output_z0",
        vec![EXPECTED_NFREQ, EXPECTED_NPORTS],
        metadata.shape.output_z0.clone(),
    )?;
    expect_metadata(
        "metadata.shape.target_z0_se",
        if inverse {
            Some(vec![EXPECTED_NFREQ, 4])
        } else {
            None
        },
        metadata.shape.target_z0_se.clone(),
    )?;
    assert!((metadata.tolerance_policy.rtol - EXPECTED_RTOL).abs() <= f64::EPSILON);
    assert!((metadata.tolerance_policy.atol - EXPECTED_ATOL).abs() <= f64::EPSILON);
    assert!(
        metadata
            .tolerance_policy
            .comparison
            .contains("only data.s is numeric output")
    );
    assert!(!metadata.tolerance_policy.justification.is_empty());
    assert!(
        metadata
            .tolerance_policy
            .regeneration
            .contains("exact contract fields")
    );
    Ok(())
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
        |(frequency, row, column)| {
            let value = &values[frequency][row][column];
            Complex64::new(value.real, value.imag)
        },
    )
}

fn array2(values: &[Vec<ComplexValue>], nports: usize) -> Array2<Complex64> {
    assert_eq!(values.len(), EXPECTED_NFREQ);
    assert!(values.iter().all(|row| row.len() == nports));
    Array2::from_shape_fn((EXPECTED_NFREQ, nports), |(frequency, port)| {
        let value = &values[frequency][port];
        Complex64::new(value.real, value.imag)
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
        "{context}: difference={difference:.17e}, bound={bound:.17e}, actual={actual:?}, expected={expected:?}"
    );
}

fn assert_frequency_exact(network: &Network, fixture: &FixtureData) {
    assert_eq!(network.frequency().len(), EXPECTED_NFREQ);
    for (index, expected) in fixture.frequency_hz.iter().enumerate() {
        assert_f64_bits(
            network.frequency().hz()[index],
            *expected,
            &format!("frequency[{index}]"),
        );
    }
}

fn assert_network_matches_fixture(
    network: &Network,
    fixture: &FixtureData,
    compare_s: bool,
    context: &str,
) {
    assert_frequency_exact(network, fixture);
    assert_eq!(network.nports(), EXPECTED_NPORTS);
    for frequency in 0..EXPECTED_NFREQ {
        for port in 0..EXPECTED_NPORTS {
            assert_complex_bits(
                network.z0()[[frequency, port]],
                &fixture.z0_target_ohm[frequency][port],
                &format!("{context}.z0[{frequency},{port}]"),
            );
        }
        for row in 0..EXPECTED_NPORTS {
            for column in 0..EXPECTED_NPORTS {
                let expected = &fixture.s[frequency][row][column];
                let actual = network.s()[[frequency, row, column]];
                if compare_s {
                    assert_s_close(
                        actual,
                        Complex64::new(expected.real, expected.imag),
                        &format!("{context}.s[{frequency},{row},{column}]"),
                    );
                } else {
                    assert_complex_bits(
                        actual,
                        expected,
                        &format!("{context}.s[{frequency},{row},{column}]"),
                    );
                }
            }
        }
    }
}

fn assert_source_contract(fixture: &FixtureDocument, inverse: bool) {
    let expected_frequency = if inverse {
        [1.07e9, 1.89e9, 2.83e9]
    } else {
        [0.91e9, 1.73e9, 2.57e9]
    };
    for (index, expected) in expected_frequency.into_iter().enumerate() {
        assert_f64_bits(
            fixture.data.frequency_hz[index],
            expected,
            &format!("fixture input frequency[{index}]"),
        );
    }
    assert_eq!(fixture.data.s_input.len(), EXPECTED_NFREQ);
    assert_eq!(fixture.data.z0_source_ohm.len(), EXPECTED_NFREQ);
    for frequency in 0..EXPECTED_NFREQ {
        assert_eq!(fixture.data.s_input[frequency].len(), EXPECTED_NPORTS);
        assert_eq!(fixture.data.z0_source_ohm[frequency].len(), EXPECTED_NPORTS);
        for row in 0..EXPECTED_NPORTS {
            assert_eq!(fixture.data.s_input[frequency][row].len(), EXPECTED_NPORTS);
        }
        if inverse {
            assert_ne!(
                fixture.data.z0_source_ohm[frequency][0],
                fixture.data.z0_source_ohm[frequency][2]
            );
        } else {
            assert_ne!(
                fixture.data.z0_source_ohm[frequency][0].real,
                fixture.data.z0_source_ohm[frequency][2].real
            );
            assert_eq!(
                fixture.data.z0_source_ohm[frequency][0],
                fixture.data.z0_source_ohm[frequency][1]
            );
            assert_eq!(
                fixture.data.z0_source_ohm[frequency][2],
                fixture.data.z0_source_ohm[frequency][3]
            );
        }
    }
    if inverse {
        let target = fixture
            .data
            .z0_se_target_ohm
            .as_ref()
            .expect("inverse fixture records explicit target z0_se");
        assert_eq!(target.len(), EXPECTED_NFREQ);
        assert!(target.iter().all(|row| row.len() == 4));
        for row in target {
            assert_eq!(row[0].real, row[1].real);
            assert_eq!(row[0].imag, row[1].imag);
            assert_eq!(row[2].real, row[3].real);
            assert_eq!(row[2].imag, row[3].imag);
        }
        for frequency in 0..EXPECTED_NFREQ {
            let modal = &fixture.data.z0_source_ohm[frequency];
            let target = &target[frequency];
            assert_f64_bits(
                modal[0].real,
                2.0 * target[0].real,
                &format!("inverse modal z0 d0 real[{frequency}]"),
            );
            assert_f64_bits(
                modal[0].imag,
                2.0 * target[0].imag,
                &format!("inverse modal z0 d0 imag[{frequency}]"),
            );
            assert_f64_bits(
                modal[1].real,
                2.0 * target[2].real,
                &format!("inverse modal z0 d1 real[{frequency}]"),
            );
            assert_f64_bits(
                modal[1].imag,
                2.0 * target[2].imag,
                &format!("inverse modal z0 d1 imag[{frequency}]"),
            );
            assert_f64_bits(
                modal[2].real,
                0.5 * target[0].real,
                &format!("inverse modal z0 c0 real[{frequency}]"),
            );
            assert_f64_bits(
                modal[2].imag,
                0.5 * target[0].imag,
                &format!("inverse modal z0 c0 imag[{frequency}]"),
            );
            assert_f64_bits(
                modal[3].real,
                0.5 * target[2].real,
                &format!("inverse modal z0 c1 real[{frequency}]"),
            );
            assert_f64_bits(
                modal[3].imag,
                0.5 * target[2].imag,
                &format!("inverse modal z0 c1 imag[{frequency}]"),
            );
        }
    } else {
        assert!(fixture.data.z0_se_target_ohm.is_none());
    }
}

#[test]
fn forward_mixed_mode_matches_pinned_scikit_rf_fixture() {
    let fixture: FixtureDocument = parse_fixture(FORWARD_FIXTURE_JSON).unwrap();
    validate_metadata(
        &fixture.metadata,
        "mixed_mode_forward_five_port_complex_z0",
        "single_ended_to_mixed_mode",
        "network_se2gmm_power",
        20_260_948,
        FORWARD_INPUT_RECIPE,
        false,
    )
    .unwrap();
    assert_source_contract(&fixture, false);
    let source_s = array3(&fixture.data.s_input);
    let source_z0 = array2(&fixture.data.z0_source_ohm, EXPECTED_NPORTS);
    let frequency = Frequency::from_hz(fixture.data.frequency_hz.clone()).unwrap();
    let network = Network::new(frequency, source_s, source_z0).unwrap();
    let original = network.clone();
    let transformed = network
        .to_mixed_mode_equal_pair_power(EXPECTED_PAIR_COUNT)
        .unwrap();
    assert_network_matches_fixture(&transformed, &fixture.data, true, "forward");
    assert_eq!(network, original);
}

#[test]
fn inverse_mixed_mode_matches_independent_pinned_scikit_rf_fixture() {
    let fixture: FixtureDocument = parse_fixture(INVERSE_FIXTURE_JSON).unwrap();
    validate_metadata(
        &fixture.metadata,
        "mixed_mode_inverse_five_port_complex_z0",
        "mixed_mode_to_single_ended",
        "network_gmm2se_power",
        20_260_949,
        INVERSE_INPUT_RECIPE,
        true,
    )
    .unwrap();
    assert_source_contract(&fixture, true);
    let source_s = array3(&fixture.data.s_input);
    let source_z0 = array2(&fixture.data.z0_source_ohm, EXPECTED_NPORTS);
    let frequency = Frequency::from_hz(fixture.data.frequency_hz.clone()).unwrap();
    let network = Network::new(frequency, source_s, source_z0).unwrap();
    let original = network.clone();
    let transformed = network
        .to_single_ended_equal_pair_power(EXPECTED_PAIR_COUNT)
        .unwrap();
    assert_network_matches_fixture(&transformed, &fixture.data, true, "inverse");
    assert_eq!(network, original);
}
