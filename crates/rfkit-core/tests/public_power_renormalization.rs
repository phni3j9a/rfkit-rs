use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{ConversionStage, Error, Frequency, Network, ParameterKind};
use serde::Deserialize;

const RENORMALIZATION_FIXTURES: &[(&str, &str)] = &[
    (
        "power_wave_renormalize_one_port_real_scalar_z0",
        include_str!(
            "../../../tools/oracle/fixtures/power_wave_renormalize_one_port_real_scalar_z0.json"
        ),
    ),
    (
        "power_wave_renormalize_two_port_complex_per_port_constant_z0",
        include_str!(
            "../../../tools/oracle/fixtures/power_wave_renormalize_two_port_complex_per_port_constant_z0.json"
        ),
    ),
    (
        "power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0",
        include_str!(
            "../../../tools/oracle/fixtures/power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0.json"
        ),
    ),
    (
        "power_wave_renormalize_eight_port_real_frequency_dependent_z0",
        include_str!(
            "../../../tools/oracle/fixtures/power_wave_renormalize_eight_port_real_frequency_dependent_z0.json"
        ),
    ),
    (
        "power_wave_renormalize_three_port_reciprocal_real_equal_z0",
        include_str!(
            "../../../tools/oracle/fixtures/power_wave_renormalize_three_port_reciprocal_real_equal_z0.json"
        ),
    ),
    (
        "power_wave_renormalize_three_port_active_real_equal_z0",
        include_str!(
            "../../../tools/oracle/fixtures/power_wave_renormalize_three_port_active_real_equal_z0.json"
        ),
    ),
];

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
    s_input: Vec<Vec<Vec<ComplexValue>>>,
    s_renormalized: Vec<Vec<Vec<ComplexValue>>>,
    z0_source_ohm: Vec<Vec<ComplexValue>>,
    z0_target_ohm: Vec<Vec<ComplexValue>>,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    operation: String,
    tolerance_policy: TolerancePolicy,
}

#[derive(Debug, Deserialize)]
struct TolerancePolicy {
    atol: f64,
    rtol: f64,
}

#[derive(Debug, Deserialize)]
struct ComplexValue {
    imag: f64,
    real: f64,
}

fn complex(value: &ComplexValue) -> Complex64 {
    Complex64::new(value.real, value.imag)
}

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
    let nfreq = values.len();
    let nport = values
        .first()
        .expect("fixture must contain at least one frequency")
        .len();
    let flattened = values
        .iter()
        .flat_map(|matrix| matrix.iter())
        .flat_map(|row| row.iter())
        .map(complex)
        .collect::<Vec<_>>();
    Array3::from_shape_vec((nfreq, nport, nport), flattened)
        .expect("fixture S matrices must be square and rectangular")
}

fn array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    let nfreq = values.len();
    let nport = values
        .first()
        .expect("fixture must contain at least one frequency")
        .len();
    let flattened = values
        .iter()
        .flat_map(|row| row.iter())
        .map(complex)
        .collect::<Vec<_>>();
    Array2::from_shape_vec((nfreq, nport), flattened)
        .expect("fixture reference impedances must be rectangular")
}

fn fixture_network(fixture: &FixtureDocument) -> Network {
    Network::new(
        Frequency::from_hz(fixture.data.frequency_hz.clone()).unwrap(),
        array3(&fixture.data.s_input),
        array2(&fixture.data.z0_source_ohm),
    )
    .unwrap()
}

fn assert_array3_close(
    actual: &Array3<Complex64>,
    expected: &Array3<Complex64>,
    rtol: f64,
    atol: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate()
    {
        let difference = (actual_value - expected_value).norm();
        let tolerance = atol + rtol * expected_value.norm();
        assert!(
            difference <= tolerance,
            "value at flattened index {index} differs by {difference:e}, tolerance {tolerance:e}: actual={actual_value:?}, expected={expected_value:?}"
        );
    }
}

fn one_port_network(s: Complex64, z0: Complex64) -> Network {
    Network::new(
        Frequency::from_hz(vec![1.0e9]).unwrap(),
        Array3::from_elem((1, 1, 1), s),
        Array2::from_elem((1, 1), z0),
    )
    .unwrap()
}

#[test]
fn public_path_matches_all_existing_renormalization_fixtures() {
    for (case_id, json) in RENORMALIZATION_FIXTURES {
        let fixture: FixtureDocument = serde_json::from_str(json).expect("fixture must parse");
        assert_eq!(&fixture.metadata.operation, "renormalize_s", "{case_id}");
        let source = fixture_network(&fixture);
        let source_frequency = source.frequency().clone();
        let source_s = source.s().clone();
        let source_z0 = source.z0().clone();
        let target_z0 = array2(&fixture.data.z0_target_ohm);
        let expected_s = array3(&fixture.data.s_renormalized);

        let target = source
            .renormalize_power(target_z0.clone())
            .unwrap_or_else(|error| panic!("{case_id} failed: {error}"));

        assert_eq!(target.frequency(), &source_frequency, "{case_id}");
        assert_eq!(target.z0(), &target_z0, "{case_id}");
        assert_eq!(target.nports(), source.nports(), "{case_id}");
        assert_ne!(target.s().as_ptr(), source.s().as_ptr(), "{case_id}");
        assert_array3_close(
            target.s(),
            &expected_s,
            fixture.metadata.tolerance_policy.rtol,
            fixture.metadata.tolerance_policy.atol,
        );

        // The public result must retain the same physical Z network.  The
        // conversion stages introduce only binary64 round-off, so use a
        // tighter mixed bound than the fixture output comparison.
        let source_z = source.to_z_power().unwrap();
        let target_z = target.to_z_power().unwrap();
        assert_array3_close(&target_z, &source_z, 2.0e-11, 2.0e-11);

        assert_eq!(source.s(), &source_s, "{case_id}");
        assert_eq!(source.z0(), &source_z0, "{case_id}");
    }
}

#[test]
fn public_renormalization_has_analytical_non_50_ohm_behavior() {
    let source_s_value = Complex64::new(0.25, 0.0);
    let source_z0_value = Complex64::new(50.0, 0.0);
    let target_z0_value = Complex64::new(75.0, 0.0);
    let source = one_port_network(source_s_value, source_z0_value);

    let source_z = source.to_z_power().unwrap()[[0, 0, 0]];
    let expected_z = (source_s_value * source_z0_value + source_z0_value.conj())
        / (Complex64::new(1.0, 0.0) - source_s_value);
    assert!((source_z - expected_z).norm() <= 1.0e-12);

    let target = source
        .renormalize_power(Array2::from_elem((1, 1), target_z0_value))
        .unwrap();
    let expected_s = (expected_z - target_z0_value) / (expected_z + target_z0_value);
    assert!((target.s()[[0, 0, 0]] - expected_s).norm() <= 1.0e-12);
    assert!((target.to_z_power().unwrap()[[0, 0, 0]] - expected_z).norm() <= 1.0e-12);
}

#[test]
fn public_renormalization_supports_negative_real_references() {
    let source_z0_value = Complex64::new(-50.0, 0.0);
    let target_z0_value = Complex64::new(-100.0, 0.0);
    let source = one_port_network(Complex64::new(0.2, 0.0), source_z0_value);
    let source_z = source.to_z_power().unwrap();
    let target = source
        .renormalize_power(Array2::from_elem((1, 1), target_z0_value))
        .unwrap();

    assert_eq!(target.z0()[[0, 0]], target_z0_value);
    assert!((target.to_z_power().unwrap()[[0, 0, 0]] - source_z[[0, 0, 0]]).norm() <= 1.0e-12);
}

#[test]
fn public_renormalization_accepts_finite_near_singular_source_system() {
    // This exact binary64 boundary leaves I-S = 2^-20.  It is deliberately
    // very close to singular while remaining nonzero, so the existing exact-
    // pivot policy must accept it without a condition-number cutoff.
    let source_s = Complex64::new(1.0 - 2.0_f64.powi(-20), 0.0);
    let source = one_port_network(source_s, Complex64::new(64.0, 0.0));
    let source_z = source.to_z_power().unwrap();
    assert!(
        source_z
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let target = source
        .renormalize_power(Array2::from_elem((1, 1), Complex64::new(91.75, 0.0)))
        .expect("finite nonzero near-singular source system must succeed");
    assert!(
        target
            .s()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let target_z = target.to_z_power().unwrap();
    assert!(
        target_z
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    // The resulting Z is about 1.34e8 ohms and the near-singular solve
    // amplifies binary64 round-off.  A 1e-11 relative bound allows the
    // observed accumulated error through both conversion stages while still
    // requiring the same finite physical impedance.
    assert_array3_close(&target_z, &source_z, 1.0e-11, 1.0e-5);
}

#[test]
fn public_renormalization_round_trips_and_identity_use_the_composed_path() {
    let source = one_port_network(Complex64::new(0.25, -0.1), Complex64::new(50.0, 0.0));
    let target_z0 = Array2::from_elem((1, 1), Complex64::new(75.0, 0.0));
    let source_z0 = source.z0().clone();
    let source_s = source.s().clone();

    let identity = source.renormalize_power(source_z0.clone()).unwrap();
    assert_array3_close(&identity.s().to_owned(), &source_s, 1.0e-13, 1.0e-13);

    let target = source.renormalize_power(target_z0).unwrap();
    let round_trip = target.renormalize_power(source_z0).unwrap();
    assert_array3_close(&round_trip.s().to_owned(), &source_s, 1.0e-13, 1.0e-13);
}

#[test]
fn public_renormalization_maps_target_shape_and_stage_errors() {
    let source = one_port_network(Complex64::new(0.0, 0.0), Complex64::new(50.0, 0.0));
    let error = source
        .renormalize_power(Array2::from_elem((1, 2), Complex64::new(75.0, 0.0)))
        .unwrap_err();
    assert_eq!(
        error,
        Error::InvalidShape {
            stage: ConversionStage::ZToS,
            parameter: ParameterKind::Z0,
            shape: vec![1, 2],
        }
    );

    let source = one_port_network(Complex64::new(0.0, 0.0), Complex64::new(0.0, 50.0));
    assert_eq!(
        source
            .renormalize_power(Array2::from_elem((1, 1), Complex64::new(75.0, 0.0)))
            .unwrap_err(),
        Error::ZeroRealReferenceImpedance {
            stage: ConversionStage::SToZ,
            frequency: 0,
            port: 0,
        }
    );

    let source = one_port_network(Complex64::new(0.0, 0.0), Complex64::new(50.0, 0.0));
    assert_eq!(
        source
            .renormalize_power(Array2::from_elem((1, 1), Complex64::new(0.0, -75.0)))
            .unwrap_err(),
        Error::ZeroRealReferenceImpedance {
            stage: ConversionStage::ZToS,
            frequency: 0,
            port: 0,
        }
    );

    let source = one_port_network(Complex64::new(0.0, 0.0), Complex64::new(50.0, 0.0));
    assert_eq!(
        source
            .renormalize_power(Array2::from_elem((1, 1), Complex64::new(f64::NAN, 0.0)))
            .unwrap_err(),
        Error::NonFiniteZ0 {
            stage: ConversionStage::ZToS,
            frequency: 0,
            port: 0,
        }
    );

    let source = one_port_network(Complex64::new(0.0, 0.0), Complex64::new(f64::INFINITY, 0.0));
    assert_eq!(
        source
            .renormalize_power(Array2::from_elem((1, 1), Complex64::new(75.0, 0.0)))
            .unwrap_err(),
        Error::NonFiniteZ0 {
            stage: ConversionStage::SToZ,
            frequency: 0,
            port: 0,
        }
    );
}

#[test]
fn public_renormalization_preserves_source_stage_errors_even_for_equal_references() {
    let singular_source = one_port_network(Complex64::new(1.0, 0.0), Complex64::new(50.0, 0.0));
    let error = singular_source
        .renormalize_power(Array2::from_elem((1, 1), Complex64::new(50.0, 0.0)))
        .unwrap_err();
    assert_eq!(
        error,
        Error::Singular {
            stage: ConversionStage::SToZ,
            frequency: 0,
            pivot: 0,
        }
    );
}

#[test]
fn public_renormalization_distinguishes_target_singularity_and_nonfinite_source() {
    let target_singular_source =
        one_port_network(Complex64::new(-3.0, 0.0), Complex64::new(50.0, 0.0));
    let error = target_singular_source
        .renormalize_power(Array2::from_elem((1, 1), Complex64::new(25.0, 0.0)))
        .unwrap_err();
    assert_eq!(
        error,
        Error::Singular {
            stage: ConversionStage::ZToS,
            frequency: 0,
            pivot: 0,
        }
    );

    let nonfinite_source =
        one_port_network(Complex64::new(f64::NAN, 0.0), Complex64::new(50.0, 0.0));
    let error = nonfinite_source
        .renormalize_power(Array2::from_elem((1, 1), Complex64::new(75.0, 0.0)))
        .unwrap_err();
    assert_eq!(
        error,
        Error::NonFiniteS {
            stage: ConversionStage::SToZ,
            frequency: 0,
            row: 0,
            column: 0,
        }
    );
}

#[test]
fn public_renormalization_maps_finite_input_overflow_to_source_stage() {
    // The input values are finite, but S*z0 overflows while forming the
    // source-stage power-wave system.  The public boundary must retain that
    // computation failure and its matrix location.
    let source = one_port_network(Complex64::new(f64::MAX, 0.0), Complex64::new(50.0, 0.0));
    let error = source
        .renormalize_power(Array2::from_elem((1, 1), Complex64::new(75.0, 0.0)))
        .unwrap_err();
    assert_eq!(
        error,
        Error::NonFiniteComputation {
            stage: ConversionStage::SToZ,
            frequency: 0,
            row: 0,
            column: 0,
        }
    );
}
