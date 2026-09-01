//! Internal Kurokawa power-wave S/Y conversions.
//!
//! These kernels deliberately compose the already verified S↔Z power-wave and
//! Z↔Y N-port conversions.  For a frequency-major stack, with port currents
//! directed into the network, the power-wave definition is
//!
//! ```text
//! a_i = (V_i + z0_i I_i) / (2 sqrt(|Re z0_i|))
//! b_i = (V_i - conj(z0_i) I_i) / (2 sqrt(|Re z0_i|))
//! b = S a
//! V = Z I
//! ```
//!
//! Thus S→Y is the composition `S → Z → Y`, and Y→S is `Y → Z → S` under
//! the same explicit Kurokawa convention.  Reference impedances remain
//! complex, per-port, and frequency-dependent through each stage.  No public
//! Network API is exposed here; callers are kept on the internal, verified
//! conversion boundary until a public API contract is specified.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::impedance_admittance::{self, ImpedanceAdmittanceError};
use crate::power_waves::{self, PowerWaveError};

/// Failure modes for a composed power-wave S/Y conversion.
///
/// Each stage retains its original private error as the source so callers and
/// tests can distinguish a source power-wave failure from a Z↔Y stage failure
/// without flattening the deterministic error context.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum PowerWaveAdmittanceError {
    #[error("power-wave S-to-Z stage failed: {0}")]
    SToZ(#[source] PowerWaveError),

    #[error("impedance-to-admittance Z-to-Y stage failed: {0}")]
    ZToY(#[source] ImpedanceAdmittanceError),

    #[error("admittance-to-impedance Y-to-Z stage failed: {0}")]
    YToZ(#[source] ImpedanceAdmittanceError),

    #[error("power-wave Z-to-S stage failed: {0}")]
    ZToS(#[source] PowerWaveError),
}

/// Convert frequency-major Kurokawa power-wave S-parameters to admittance.
///
/// The implementation is intentionally the exact composition
/// [`crate::power_waves::s_to_z_power`] followed by
/// [`crate::impedance_admittance::z_to_y`].  This preserves the existing
/// complex-Z0 semantics and exact-zero singularity policy at each stage.
#[allow(dead_code)] // Internal kernel is staged for a future Network call site.
pub(crate) fn s_to_y_power(
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<Array3<Complex64>, PowerWaveAdmittanceError> {
    let z = power_waves::s_to_z_power(s, z0).map_err(PowerWaveAdmittanceError::SToZ)?;
    impedance_admittance::z_to_y(&z).map_err(PowerWaveAdmittanceError::ZToY)
}

/// Convert frequency-major admittance to Kurokawa power-wave S-parameters.
///
/// The implementation is intentionally the exact composition
/// [`crate::impedance_admittance::y_to_z`] followed by
/// [`crate::power_waves::z_to_s_power`].  This preserves the existing
/// complex-Z0 semantics and exact-zero singularity policy at each stage.
#[allow(dead_code)] // Internal kernel is staged for a future Network call site.
pub(crate) fn y_to_s_power(
    y: &Array3<Complex64>,
    z0: &Array2<Complex64>,
) -> Result<Array3<Complex64>, PowerWaveAdmittanceError> {
    let z = impedance_admittance::y_to_z(y).map_err(PowerWaveAdmittanceError::YToZ)?;
    power_waves::z_to_s_power(&z, z0).map_err(PowerWaveAdmittanceError::ZToS)
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use ndarray::{Array2, Array3};
    use serde::Deserialize;

    use super::*;

    const ZERO: Complex64 = Complex64::new(0.0, 0.0);

    const S_TO_Y_FIXTURE_JSON: &str =
        include_str!("../../../tools/oracle/fixtures/power_wave_s_to_y_three_port_complex_z0.json");
    const Y_TO_S_FIXTURE_JSON: &str =
        include_str!("../../../tools/oracle/fixtures/power_wave_y_to_s_three_port_complex_z0.json");

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
        #[serde(default)]
        s: Option<Vec<Vec<Vec<ComplexValue>>>>,
        #[serde(default)]
        y_s: Option<Vec<Vec<Vec<ComplexValue>>>>,
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
        input_unit: String,
        numpy_version: String,
        operation: String,
        output_unit: String,
        random_seed: u64,
        reference_impedance: ReferenceImpedanceMetadata,
        schema: String,
        schema_version: u32,
        scikit_rf_version: String,
        shape: FixtureShape,
        tolerance_policy: TolerancePolicy,
        wave_definition: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ReferenceImpedanceMetadata {
        complex: bool,
        frequency_dependent: bool,
        per_port: bool,
        unit: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FixtureShape {
        frequency: Vec<usize>,
        #[serde(default)]
        input_s: Option<Vec<usize>>,
        #[serde(default)]
        input_y: Option<Vec<usize>>,
        input_z0: Vec<usize>,
        #[serde(default)]
        output_s: Option<Vec<usize>>,
        #[serde(default)]
        output_y: Option<Vec<usize>>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct TolerancePolicy {
        #[serde(default)]
        atol: Option<f64>,
        #[serde(default)]
        atol_s: Option<f64>,
        comparison: String,
        justification: String,
        regeneration: String,
        rtol: f64,
    }

    fn complex(real: f64, imag: f64) -> Complex64 {
        Complex64::new(real, imag)
    }

    fn assert_array3_close(
        actual: &Array3<Complex64>,
        expected: &Array3<Complex64>,
        rtol: f64,
        atol: f64,
    ) {
        assert_eq!(actual.dim(), expected.dim());
        for frequency in 0..actual.dim().0 {
            for row in 0..actual.dim().1 {
                for column in 0..actual.dim().2 {
                    let actual_value = actual[[frequency, row, column]];
                    let expected_value = expected[[frequency, row, column]];
                    let difference = (actual_value - expected_value).norm();
                    let tolerance = atol + rtol * expected_value.norm();
                    assert!(
                        difference <= tolerance,
                        "output[{frequency},{row},{column}] differs: actual={actual_value:?}, expected={expected_value:?}, difference={difference:e}, tolerance={tolerance:e}"
                    );
                }
            }
        }
    }

    fn fixture_array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
        assert!(!values.is_empty(), "fixture must contain frequencies");
        let nfreq = values.len();
        let nport = values[0].len();
        assert!(nport > 0, "fixture must contain ports");
        assert!(values.iter().all(|matrix| matrix.len() == nport));
        assert!(
            values
                .iter()
                .all(|matrix| matrix.iter().all(|row| row.len() == nport))
        );
        let flattened = values
            .iter()
            .flat_map(|matrix| matrix.iter())
            .flat_map(|row| row.iter())
            .map(|value| complex(value.real, value.imag))
            .collect();
        Array3::from_shape_vec((nfreq, nport, nport), flattened)
            .expect("fixture dimensions must agree")
    }

    fn fixture_z0(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
        assert!(!values.is_empty(), "fixture must contain frequencies");
        let nfreq = values.len();
        let nport = values[0].len();
        assert!(nport > 0, "fixture must contain ports");
        assert!(values.iter().all(|row| row.len() == nport));
        let flattened = values
            .iter()
            .flat_map(|row| row.iter())
            .map(|value| complex(value.real, value.imag))
            .collect();
        Array2::from_shape_vec((nfreq, nport), flattened).expect("fixture z0 dimensions must agree")
    }

    fn assert_recorded_output(
        actual: &Array3<Complex64>,
        expected: &Array3<Complex64>,
        rtol: f64,
        atol: f64,
    ) {
        assert_eq!(actual.dim(), expected.dim());
        for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
            assert!(
                actual_value.re.is_finite() && actual_value.im.is_finite(),
                "actual output at flattened index {index} is non-finite"
            );
            let difference = (actual_value - expected_value).norm();
            let bound = atol + rtol * expected_value.norm();
            assert!(
                difference <= bound,
                "output at flattened index {index} differs by {difference:?}, bound {bound:?}; actual {actual_value:?}, expected {expected_value:?}"
            );
        }
    }

    fn multiport_s_case() -> (Array3<Complex64>, Array2<Complex64>) {
        let s = Array3::from_shape_vec(
            (2, 3, 3),
            vec![
                complex(0.10, 0.02),
                complex(-0.04, 0.01),
                complex(0.025, -0.03),
                complex(0.06, -0.02),
                complex(-0.08, 0.03),
                complex(0.035, 0.015),
                complex(-0.02, 0.04),
                complex(0.05, -0.01),
                complex(0.07, -0.025),
                complex(-0.06, 0.015),
                complex(0.03, -0.02),
                complex(0.045, 0.025),
                complex(0.02, 0.035),
                complex(-0.055, 0.01),
                complex(0.04, -0.015),
                complex(0.015, -0.025),
                complex(0.065, 0.02),
                complex(-0.09, 0.035),
            ],
        )
        .expect("fixed S shape must be valid");
        let z0 = Array2::from_shape_vec(
            (2, 3),
            vec![
                complex(50.0, 2.0),
                complex(63.0, -1.0),
                complex(71.0, 3.0),
                complex(52.0, 2.5),
                complex(66.0, -1.5),
                complex(74.0, 3.5),
            ],
        )
        .expect("fixed z0 shape must be valid");
        (s, z0)
    }

    fn multiport_y_case() -> (Array3<Complex64>, Array2<Complex64>) {
        let y = Array3::from_shape_vec(
            (2, 3, 3),
            vec![
                complex(0.021, 0.002),
                complex(-0.0012, 0.0006),
                complex(0.0008, -0.0005),
                complex(0.0011, -0.0007),
                complex(0.024, 0.0015),
                complex(-0.0009, 0.0004),
                complex(-0.0006, 0.0009),
                complex(0.0014, -0.0003),
                complex(0.027, 0.0025),
                complex(0.022, 0.0022),
                complex(-0.001, 0.0005),
                complex(0.0007, -0.0004),
                complex(0.0013, -0.0008),
                complex(0.025, 0.0017),
                complex(-0.0011, 0.0002),
                complex(-0.0005, 0.0008),
                complex(0.0012, -0.0002),
                complex(0.028, 0.0028),
            ],
        )
        .expect("fixed Y shape must be valid");
        let z0 = Array2::from_shape_vec(
            (2, 3),
            vec![
                complex(50.0, 2.0),
                complex(63.0, -1.0),
                complex(71.0, 3.0),
                complex(52.0, 2.5),
                complex(66.0, -1.5),
                complex(74.0, 3.5),
            ],
        )
        .expect("fixed z0 shape must be valid");
        (y, z0)
    }

    #[test]
    fn matches_direct_s_to_y_fixture_with_recorded_tolerance() {
        let fixture: FixtureDocument =
            serde_json::from_str(S_TO_Y_FIXTURE_JSON).expect("S-to-Y fixture must parse");
        let metadata = &fixture.metadata;
        assert_eq!(metadata.case_id, "power_wave_s_to_y_three_port_complex_z0");
        assert_eq!(metadata.operation, "s_to_y");
        assert_eq!(metadata.input_unit, "dimensionless");
        assert_eq!(metadata.output_unit, "S");
        assert_eq!(metadata.wave_definition, "power");
        assert_eq!(metadata.schema, "rfkit-rs.oracle.fixture");
        assert_eq!(metadata.schema_version, 1);
        assert_eq!(metadata.numpy_version, "2.5.1");
        assert_eq!(metadata.scikit_rf_version, "2.0.1");
        assert_eq!(metadata.random_seed, 20_260_937);
        assert_eq!(metadata.shape.frequency, vec![3]);
        assert_eq!(
            metadata.shape.input_s.as_deref(),
            Some([3, 3, 3].as_slice())
        );
        assert_eq!(metadata.shape.input_z0, vec![3, 3]);
        assert_eq!(
            metadata.shape.output_y.as_deref(),
            Some([3, 3, 3].as_slice())
        );
        assert!(metadata.shape.input_y.is_none());
        assert!(metadata.shape.output_s.is_none());
        assert!(metadata.reference_impedance.complex);
        assert!(metadata.reference_impedance.frequency_dependent);
        assert!(metadata.reference_impedance.per_port);
        assert_eq!(metadata.reference_impedance.unit, "ohm");
        assert_eq!(metadata.tolerance_policy.rtol, 1e-12);
        assert_eq!(metadata.tolerance_policy.atol_s, Some(1e-12));
        assert!(metadata.tolerance_policy.atol.is_none());
        assert!(!metadata.tolerance_policy.comparison.is_empty());
        assert!(!metadata.tolerance_policy.justification.is_empty());
        assert!(!metadata.tolerance_policy.regeneration.is_empty());

        let s = fixture
            .data
            .s
            .as_ref()
            .expect("S-to-Y fixture needs S input");
        let expected_y = fixture
            .data
            .y_s
            .as_ref()
            .expect("S-to-Y fixture needs Y output");
        let z0 = fixture_z0(&fixture.data.z0_ohm);
        assert_eq!(fixture.data.frequency_hz.len(), 3);
        assert!(
            fixture
                .data
                .frequency_hz
                .iter()
                .all(|value| value.is_finite())
        );
        assert_eq!(s.len(), 3);
        assert_eq!(expected_y.len(), 3);
        assert!(
            z0.iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        assert!(z0.iter().any(|value| value.im != 0.0));
        assert_ne!(z0[[0, 0]], z0[[0, 1]]);
        assert_ne!(z0[[0, 0]], z0[[1, 0]]);

        let s = fixture_array3(s);
        let expected_y = fixture_array3(expected_y);
        let actual = s_to_y_power(&s, &z0).expect("direct S-to-Y fixture must succeed");
        assert_recorded_output(
            &actual,
            &expected_y,
            metadata.tolerance_policy.rtol,
            metadata
                .tolerance_policy
                .atol_s
                .expect("S-to-Y fixture requires atol_s"),
        );
    }

    #[test]
    fn matches_direct_y_to_s_fixture_with_recorded_tolerance() {
        let fixture: FixtureDocument =
            serde_json::from_str(Y_TO_S_FIXTURE_JSON).expect("Y-to-S fixture must parse");
        let metadata = &fixture.metadata;
        assert_eq!(metadata.case_id, "power_wave_y_to_s_three_port_complex_z0");
        assert_eq!(metadata.operation, "y_to_s");
        assert_eq!(metadata.input_unit, "S");
        assert_eq!(metadata.output_unit, "dimensionless");
        assert_eq!(metadata.wave_definition, "power");
        assert_eq!(metadata.schema, "rfkit-rs.oracle.fixture");
        assert_eq!(metadata.schema_version, 1);
        assert_eq!(metadata.numpy_version, "2.5.1");
        assert_eq!(metadata.scikit_rf_version, "2.0.1");
        assert_eq!(metadata.random_seed, 20_260_938);
        assert_eq!(metadata.shape.frequency, vec![3]);
        assert_eq!(
            metadata.shape.input_y.as_deref(),
            Some([3, 3, 3].as_slice())
        );
        assert_eq!(metadata.shape.input_z0, vec![3, 3]);
        assert_eq!(
            metadata.shape.output_s.as_deref(),
            Some([3, 3, 3].as_slice())
        );
        assert!(metadata.shape.input_s.is_none());
        assert!(metadata.shape.output_y.is_none());
        assert!(metadata.reference_impedance.complex);
        assert!(metadata.reference_impedance.frequency_dependent);
        assert!(metadata.reference_impedance.per_port);
        assert_eq!(metadata.reference_impedance.unit, "ohm");
        assert_eq!(metadata.tolerance_policy.rtol, 1e-12);
        assert_eq!(metadata.tolerance_policy.atol, Some(1e-12));
        assert!(metadata.tolerance_policy.atol_s.is_none());
        assert!(!metadata.tolerance_policy.comparison.is_empty());
        assert!(!metadata.tolerance_policy.justification.is_empty());
        assert!(!metadata.tolerance_policy.regeneration.is_empty());

        let y = fixture
            .data
            .y_s
            .as_ref()
            .expect("Y-to-S fixture needs Y input");
        let expected_s = fixture
            .data
            .s
            .as_ref()
            .expect("Y-to-S fixture needs S output");
        let z0 = fixture_z0(&fixture.data.z0_ohm);
        assert_eq!(fixture.data.frequency_hz.len(), 3);
        assert!(
            fixture
                .data
                .frequency_hz
                .iter()
                .all(|value| value.is_finite())
        );
        assert_eq!(y.len(), 3);
        assert_eq!(expected_s.len(), 3);
        assert!(
            z0.iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        assert!(z0.iter().any(|value| value.im != 0.0));
        assert_ne!(z0[[0, 0]], z0[[0, 1]]);
        assert_ne!(z0[[0, 0]], z0[[1, 0]]);

        let y = fixture_array3(y);
        let expected_s = fixture_array3(expected_s);
        let actual = y_to_s_power(&y, &z0).expect("direct Y-to-S fixture must succeed");
        assert_recorded_output(
            &actual,
            &expected_s,
            metadata.tolerance_policy.rtol,
            metadata
                .tolerance_policy
                .atol
                .expect("Y-to-S fixture requires atol"),
        );
    }

    fn assert_source_power_wave(error: &PowerWaveAdmittanceError, expected: &PowerWaveError) {
        assert!(matches!(error, PowerWaveAdmittanceError::SToZ(actual) if actual == expected));
        let source = error.source().expect("stage error must expose its source");
        assert_eq!(source.downcast_ref::<PowerWaveError>(), Some(expected));
    }

    fn assert_source_impedance_admittance(
        error: &PowerWaveAdmittanceError,
        expected: &ImpedanceAdmittanceError,
    ) {
        assert!(matches!(error, PowerWaveAdmittanceError::ZToY(actual) if actual == expected));
        let source = error.source().expect("stage error must expose its source");
        assert_eq!(
            source.downcast_ref::<ImpedanceAdmittanceError>(),
            Some(expected)
        );
    }

    #[test]
    fn one_port_analytical_power_wave_s_to_y() {
        let z0_value = complex(40.0, 8.0);
        let s = Array3::from_elem((1, 1, 1), ZERO);
        let z0 = Array2::from_elem((1, 1), z0_value);

        let y = s_to_y_power(&s, &z0).expect("one-port S-to-Y must succeed");
        let expected = 1.0 / z0_value.conj();
        assert!((y[[0, 0, 0]] - expected).norm() <= 1e-15);
    }

    #[test]
    fn one_port_analytical_power_wave_y_to_s() {
        let z0_value = complex(40.0, 8.0);
        let y = Array3::from_elem((1, 1, 1), 1.0 / z0_value.conj());
        let z0 = Array2::from_elem((1, 1), z0_value);

        let s = y_to_s_power(&y, &z0).expect("one-port Y-to-S must succeed");
        assert!(s[[0, 0, 0]].norm() <= 1e-14, "expected zero S, got {s:?}");
    }

    #[test]
    fn multiport_s_to_y_to_s_round_trip_preserves_s() {
        let (source_s, z0) = multiport_s_case();
        let y = s_to_y_power(&source_s, &z0).expect("S-to-Y must succeed");
        let recovered_s = y_to_s_power(&y, &z0).expect("Y-to-S must succeed");
        assert_array3_close(&recovered_s, &source_s, 1e-13, 1e-13);
    }

    #[test]
    fn multiport_y_to_s_to_y_round_trip_preserves_y() {
        let (source_y, z0) = multiport_y_case();
        let s = y_to_s_power(&source_y, &z0).expect("Y-to-S must succeed");
        let recovered_y = s_to_y_power(&s, &z0).expect("S-to-Y must succeed");
        assert_array3_close(&recovered_y, &source_y, 1e-13, 1e-13);
    }

    #[test]
    fn rejects_invalid_shapes_with_source_stage_context() {
        let valid_z0 = Array2::from_elem((1, 2), complex(50.0, 0.0));
        let invalid_s = Array3::zeros((1, 2, 3));
        let error = s_to_y_power(&invalid_s, &valid_z0).expect_err("invalid S shape");
        assert_eq!(
            error,
            PowerWaveAdmittanceError::SToZ(PowerWaveError::InvalidSShape { shape: (1, 2, 3) })
        );

        let invalid_y = Array3::zeros((1, 2, 3));
        let error = y_to_s_power(&invalid_y, &valid_z0).expect_err("invalid Y shape");
        assert_eq!(
            error,
            PowerWaveAdmittanceError::YToZ(ImpedanceAdmittanceError::InvalidYShape {
                shape: (1, 2, 3)
            })
        );

        let valid_s = Array3::zeros((1, 2, 2));
        let invalid_z0 = Array2::from_elem((1, 1), complex(50.0, 0.0));
        let error = s_to_y_power(&valid_s, &invalid_z0).expect_err("invalid S z0 shape");
        assert_eq!(
            error,
            PowerWaveAdmittanceError::SToZ(PowerWaveError::InvalidZ0Shape { shape: (1, 1) })
        );
        let mut valid_y = Array3::zeros((1, 2, 2));
        valid_y[[0, 0, 0]] = complex(0.01, 0.0);
        valid_y[[0, 1, 1]] = complex(0.02, 0.0);
        let error = y_to_s_power(&valid_y, &invalid_z0).expect_err("invalid Y z0 shape");
        assert_eq!(
            error,
            PowerWaveAdmittanceError::ZToS(PowerWaveError::InvalidZ0Shape { shape: (1, 1) })
        );
    }

    #[test]
    fn rejects_non_finite_s_y_and_z0_with_source_context() {
        let z0 = Array2::from_elem((1, 1), complex(50.0, 0.0));
        let mut non_finite_s = Array3::from_elem((1, 1, 1), ZERO);
        non_finite_s[[0, 0, 0]] = complex(f64::NAN, 0.0);
        let error = s_to_y_power(&non_finite_s, &z0).expect_err("non-finite S");
        assert_source_power_wave(
            &error,
            &PowerWaveError::NonFiniteS {
                frequency: 0,
                row: 0,
                column: 0,
            },
        );

        let mut non_finite_y = Array3::from_elem((1, 1, 1), ZERO);
        non_finite_y[[0, 0, 0]] = complex(0.0, f64::INFINITY);
        let error = y_to_s_power(&non_finite_y, &z0).expect_err("non-finite Y");
        assert!(matches!(
            error,
            PowerWaveAdmittanceError::YToZ(ImpedanceAdmittanceError::NonFiniteY {
                frequency: 0,
                row: 0,
                column: 0,
            })
        ));
        let source = error
            .source()
            .expect("Y stage error must expose its source");
        assert!(source.downcast_ref::<ImpedanceAdmittanceError>().is_some());

        let mut non_finite_z0 = z0.clone();
        non_finite_z0[[0, 0]] = complex(f64::INFINITY, 0.0);
        let valid_s = Array3::from_elem((1, 1, 1), ZERO);
        let error = s_to_y_power(&valid_s, &non_finite_z0).expect_err("non-finite z0");
        assert_source_power_wave(
            &error,
            &PowerWaveError::NonFiniteZ0 {
                frequency: 0,
                port: 0,
            },
        );
        let valid_y = Array3::from_elem((1, 1, 1), complex(0.01, 0.0));
        let error = y_to_s_power(&valid_y, &non_finite_z0).expect_err("non-finite z0");
        assert!(matches!(
            error,
            PowerWaveAdmittanceError::ZToS(PowerWaveError::NonFiniteZ0 {
                frequency: 0,
                port: 0,
            })
        ));
    }

    #[test]
    fn rejects_non_finite_computation_in_each_direction() {
        let z0 = Array2::from_elem((1, 1), complex(50.0, 0.0));

        let huge_s = Array3::from_elem((1, 1, 1), complex(f64::MAX, 0.0));
        let error = s_to_y_power(&huge_s, &z0).expect_err("finite S overflow must be reported");
        assert!(matches!(
            error,
            PowerWaveAdmittanceError::SToZ(PowerWaveError::NonFiniteComputation {
                frequency: 0,
                row: 0,
                column: 0,
            })
        ));

        // Keep the input finite while forcing a checked elimination
        // subtraction overflow.  The tiny pivot also verifies that the
        // shared solver does not use squared pivot magnitudes.
        let overflow_y = Array3::from_shape_vec(
            (1, 2, 2),
            vec![
                complex(1.0e-200, 0.0),
                complex(f64::MAX, 0.0),
                complex(1.0e-200, 0.0),
                complex(-f64::MAX, 0.0),
            ],
        )
        .unwrap();
        let overflow_z0 = Array2::from_elem((1, 2), complex(50.0, 0.0));
        let error = y_to_s_power(&overflow_y, &overflow_z0)
            .expect_err("finite Y overflow must be reported");
        assert!(matches!(
            error,
            PowerWaveAdmittanceError::YToZ(ImpedanceAdmittanceError::NonFiniteComputation {
                frequency: 0,
                row: 1,
                column: 1,
            })
        ));

        // A finite, very small reference impedance produces a finite S-to-Z
        // result, but its reciprocal overflows in the second stage.
        // Keep the normalization pivot magnitude finite while making the
        // resulting impedance small enough that its reciprocal overflows.
        let tiny_z0 = Array2::from_elem((1, 1), complex(4.0e-309, 0.0));
        let zero_s = Array3::from_elem((1, 1, 1), ZERO);
        let error = s_to_y_power(&zero_s, &tiny_z0)
            .expect_err("finite tiny Z must report second-stage overflow");
        assert!(matches!(
            error,
            PowerWaveAdmittanceError::ZToY(ImpedanceAdmittanceError::NonFiniteComputation {
                frequency: 0,
                row: 0,
                column: 0,
            })
        ));

        // The first Y-to-Z stage remains finite for 4e-155, while the final
        // Z-to-S normalization overflows with the same tiny reference.
        let tiny_y = Array3::from_elem((1, 1, 1), complex(4e-155, 0.0));
        let error = y_to_s_power(&tiny_y, &tiny_z0)
            .expect_err("finite tiny Y must report final-stage overflow");
        assert!(matches!(
            error,
            PowerWaveAdmittanceError::ZToS(PowerWaveError::NonFiniteComputation {
                frequency: 0,
                row: 0,
                column: 0,
            })
        ));
    }

    #[test]
    fn rejects_zero_real_reference_impedance_in_each_direction() {
        let zero_real_z0 = Array2::from_elem((1, 1), complex(0.0, 50.0));
        let valid_s = Array3::from_elem((1, 1, 1), ZERO);
        let error = s_to_y_power(&valid_s, &zero_real_z0).expect_err("zero-real z0");
        assert!(matches!(
            error,
            PowerWaveAdmittanceError::SToZ(PowerWaveError::ZeroRealReferenceImpedance {
                frequency: 0,
                port: 0,
            })
        ));

        let valid_y = Array3::from_elem((1, 1, 1), complex(0.01, 0.0));
        let error = y_to_s_power(&valid_y, &zero_real_z0).expect_err("zero-real z0");
        assert!(matches!(
            error,
            PowerWaveAdmittanceError::ZToS(PowerWaveError::ZeroRealReferenceImpedance {
                frequency: 0,
                port: 0,
            })
        ));
    }

    #[test]
    fn reports_exact_singularities_at_each_composition_stage() {
        let z0 = Array2::from_elem((1, 1), complex(1.0, 0.0));

        // S=1 makes the first S-to-Z system I-S exactly singular.
        let source_s_singular = Array3::from_elem((1, 1, 1), complex(1.0, 0.0));
        let error = s_to_y_power(&source_s_singular, &z0).expect_err("S-to-Z singularity");
        assert!(matches!(
            error,
            PowerWaveAdmittanceError::SToZ(PowerWaveError::Singular {
                frequency: 0,
                pivot: 0,
            })
        ));

        // S=-1 produces Z=0 exactly, so the subsequent Z-to-Y inversion is
        // the stage that reports the exact singularity.
        let source_s_z_singular = Array3::from_elem((1, 1, 1), complex(-1.0, 0.0));
        let error = s_to_y_power(&source_s_z_singular, &z0).expect_err("Z-to-Y singularity");
        assert_source_impedance_admittance(
            &error,
            &ImpedanceAdmittanceError::Singular {
                frequency: 0,
                pivot: 0,
            },
        );

        // Y=0 makes the first Y-to-Z inversion exactly singular.
        let source_y_singular = Array3::from_elem((1, 1, 1), ZERO);
        let error = y_to_s_power(&source_y_singular, &z0).expect_err("Y-to-Z singularity");
        assert!(matches!(
            error,
            PowerWaveAdmittanceError::YToZ(ImpedanceAdmittanceError::Singular {
                frequency: 0,
                pivot: 0,
            })
        ));

        // Y=-1 gives Z=-1 exactly.  The final Z-to-S system Z+G is then
        // exactly singular.
        let source_y_z_singular = Array3::from_elem((1, 1, 1), complex(-1.0, 0.0));
        let error = y_to_s_power(&source_y_z_singular, &z0).expect_err("Z-to-S singularity");
        assert!(matches!(
            error,
            PowerWaveAdmittanceError::ZToS(PowerWaveError::Singular {
                frequency: 0,
                pivot: 0,
            })
        ));
    }
}
