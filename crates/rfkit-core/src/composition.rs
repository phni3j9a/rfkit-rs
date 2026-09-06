//! Internal explicit-grid matched connection for frequency-major N-port data.
//!
//! [`connect_matched_on_grid`] composes two networks whose source frequency
//! axes may differ by first interpolating each one onto the caller-provided
//! target axis and then delegating the resulting arrays to the existing
//! exactly-matched real-junction connection kernel.  The operation order is
//! deliberately part of the contract: A interpolation, B interpolation, and
//! matched connection happen in that order, and a failure is attributed to the
//! stage that produced it.
//!
//! The final connection uses the Kurokawa power-wave matched-junction
//! semantics implemented by [`crate::connection::connect_matched`].  The
//! selected A and B junction reference impedances must therefore be finite,
//! real, strictly positive, and exactly equal at every target frequency.
//! External surviving-port impedances may remain finite complex and
//! frequency-dependent.  The output order is A survivors in their original
//! order followed by B survivors in their original order.
//!
//! The target axis is never inferred, intersected, subset, sorted, or
//! extrapolated by this wrapper.  All source/target validation and Cartesian
//! linear interpolation semantics are delegated to
//! [`crate::interpolation::interpolate_cartesian_linear`].  All connection
//! shape, port, junction, survivor, singularity, and finite-computation
//! checks are delegated to [`crate::connection::connect_matched`].

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

use crate::connection::{ConnectedNetwork, ConnectionError, connect_matched};
use crate::interpolation::{InterpolationError, interpolate_cartesian_linear};

/// Identifies the stage that failed while composing two networks on an
/// explicit target frequency grid.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum CompositionError {
    /// A's source data or the shared target grid failed interpolation
    /// validation or computation.
    #[error("A interpolation failed: {0}")]
    AInterpolation(#[source] InterpolationError),

    /// B's source data failed interpolation validation or computation after A
    /// interpolation completed successfully.
    #[error("B interpolation failed: {0}")]
    BInterpolation(#[source] InterpolationError),

    /// The interpolated arrays failed the existing matched-connection kernel's
    /// validation or numerical evaluation.
    #[error("matched connection failed: {0}")]
    Connection(#[source] ConnectionError),
}

/// Interpolate two source networks onto an explicit target grid and connect
/// one port from each through an exactly matched real junction.
///
/// A is interpolated first, then B, and only then are the two interpolated
/// networks passed to [`connect_matched`].  The target grid is copied by the
/// interpolation kernel and is returned unchanged on success.  No target-grid
/// selection, intersection, subsetting, sorting, or extrapolation is
/// performed here.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn connect_matched_on_grid(
    frequency_a: &[f64],
    s_a: &Array3<Complex64>,
    z0_a: &Array2<Complex64>,
    port_a: usize,
    frequency_b: &[f64],
    s_b: &Array3<Complex64>,
    z0_b: &Array2<Complex64>,
    port_b: usize,
    target_frequency_hz: &[f64],
) -> Result<ConnectedNetwork, CompositionError> {
    let interpolated_a = interpolate_cartesian_linear(frequency_a, s_a, z0_a, target_frequency_hz)
        .map_err(CompositionError::AInterpolation)?;

    let interpolated_b = interpolate_cartesian_linear(frequency_b, s_b, z0_b, target_frequency_hz)
        .map_err(CompositionError::BInterpolation)?;

    connect_matched(
        &interpolated_a.frequency_hz,
        &interpolated_a.s,
        &interpolated_a.z0,
        port_a,
        &interpolated_b.frequency_hz,
        &interpolated_b.s,
        &interpolated_b.z0,
        port_b,
    )
    .map_err(CompositionError::Connection)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};
    use serde::Deserialize;

    const COMPOSITION_FIXTURE_JSON: &str = include_str!(
        "../../../tools/oracle/fixtures/power_wave_connect_matched_explicit_grid_three_to_four_port_complex_z0.json"
    );

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
        s_a: Vec<Vec<Vec<ComplexValue>>>,
        s_b: Vec<Vec<Vec<ComplexValue>>>,
        s_connected: Vec<Vec<Vec<ComplexValue>>>,
        source_frequency_a_hz: Vec<f64>,
        source_frequency_b_hz: Vec<f64>,
        target_frequency_hz: Vec<f64>,
        z0_a_ohm: Vec<Vec<ComplexValue>>,
        z0_b_ohm: Vec<Vec<ComplexValue>>,
        z0_connected_ohm: Vec<Vec<ComplexValue>>,
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
        interpolation: InterpolationMetadata,
        junction_ports: JunctionPorts,
        numpy_version: String,
        operation: String,
        port_order: PortOrder,
        random_seeds: RandomSeeds,
        reference_impedance: ReferenceImpedance,
        schema: String,
        schema_version: u32,
        scikit_rf_version: String,
        shape: FixtureShape,
        tolerance_policy: FixtureTolerance,
        wave_definition: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct InterpolationMetadata {
        basis: String,
        coords: String,
        kind: String,
        scipy_version: String,
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
        a_survivors: Vec<usize>,
        b_survivors: Vec<usize>,
        description: String,
        output: Vec<PortReference>,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct PortReference {
        network: String,
        port: usize,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RandomSeeds {
        a: u64,
        b: u64,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ReferenceImpedance {
        external_complex: bool,
        external_frequency_dependent: bool,
        junction_exactly_equal_after_interpolation: bool,
        junction_frequency_dependent: bool,
        junction_real_strictly_positive: bool,
        unit: String,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FixtureShape {
        frequency: Vec<usize>,
        input_s_a: Vec<usize>,
        input_s_b: Vec<usize>,
        input_z0_a: Vec<usize>,
        input_z0_b: Vec<usize>,
        output_s: Vec<usize>,
        output_z0: Vec<usize>,
        source_frequency_a: Vec<usize>,
        source_frequency_b: Vec<usize>,
        target_frequency: Vec<usize>,
    }

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct FixtureTolerance {
        atol: f64,
        comparison: String,
        justification: String,
        regeneration: String,
        rtol: f64,
    }

    fn complex(real: f64, imag: f64) -> Complex64 {
        Complex64::new(real, imag)
    }

    fn affine_s(frequency: f64, row: usize, column: usize, side_offset: f64) -> Complex64 {
        let row = row as f64;
        let column = column as f64;
        complex(
            side_offset + 0.000_000_000_25 * frequency + 0.007 * row - 0.004 * column,
            -side_offset * 0.4 - 0.000_000_000_15 * frequency + 0.003 * row + 0.006 * column,
        )
    }

    fn affine_z0(frequency: f64, port: usize, side_offset: f64) -> Complex64 {
        let port = port as f64;
        complex(
            side_offset + 0.000_000_000_8 * frequency + 2.75 * port,
            1.25 + 0.000_000_000_35 * frequency + 0.45 * port,
        )
    }

    fn make_s(frequency: &[f64], nports: usize, side_offset: f64) -> Array3<Complex64> {
        Array3::from_shape_fn(
            (frequency.len(), nports, nports),
            |(frequency_index, row, column)| {
                affine_s(frequency[frequency_index], row, column, side_offset)
            },
        )
    }

    fn make_z0(
        frequency: &[f64],
        nports: usize,
        side_offset: f64,
        junction_port: usize,
    ) -> Array2<Complex64> {
        Array2::from_shape_fn((frequency.len(), nports), |(frequency_index, port)| {
            if port == junction_port {
                // Keep the matched junction profile exactly real and equal on
                // both sides after either source grid is interpolated.
                complex(73.5, 0.0)
            } else {
                affine_z0(frequency[frequency_index], port, side_offset)
            }
        })
    }

    fn assert_close(actual: Complex64, expected: Complex64) {
        let difference = (actual - expected).norm();
        assert!(
            difference <= 2.0e-12,
            "actual={actual:?}, expected={expected:?}, difference={difference:e}"
        );
    }

    fn fixture_array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
        assert!(!values.is_empty());
        let nfreq = values.len();
        let nrows = values[0].len();
        assert!(nrows > 0);
        let ncolumns = values[0][0].len();
        assert!(ncolumns > 0);
        let flattened = values.iter().flat_map(|frequency| {
            frequency.iter().flat_map(|row| {
                row.iter()
                    .map(|value| Complex64::new(value.real, value.imag))
            })
        });
        Array3::from_shape_vec((nfreq, nrows, ncolumns), flattened.collect())
            .expect("fixture S dimensions must agree")
    }

    fn fixture_array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
        assert!(!values.is_empty());
        let nfreq = values.len();
        let nports = values[0].len();
        assert!(nports > 0);
        let flattened = values.iter().flat_map(|frequency| {
            frequency
                .iter()
                .map(|value| Complex64::new(value.real, value.imag))
        });
        Array2::from_shape_vec((nfreq, nports), flattened.collect())
            .expect("fixture z0 dimensions must agree")
    }

    fn assert_recorded_output(
        actual: impl IntoIterator<Item = Complex64>,
        expected: impl IntoIterator<Item = Complex64>,
        rtol: f64,
        atol: f64,
    ) {
        for (index, (actual_value, expected_value)) in actual.into_iter().zip(expected).enumerate()
        {
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

    #[test]
    fn interpolates_distinct_irregular_grids_before_matched_connection() {
        let frequency_a = [1.0e9, 2.5e9, 5.5e9, 10.0e9];
        let frequency_b = [1.0e9, 1.75e9, 4.75e9, 7.0e9, 10.0e9];
        let target = [1.0e9, 1.75e9, 2.5e9, 4.0e9, 7.0e9, 10.0e9];
        let s_a = make_s(&frequency_a, 3, 0.015);
        let s_b = make_s(&frequency_b, 4, -0.021);
        let z0_a = make_z0(&frequency_a, 3, 40.0, 1);
        let z0_b = make_z0(&frequency_b, 4, 82.0, 2);

        let result = connect_matched_on_grid(
            &frequency_a,
            &s_a,
            &z0_a,
            1,
            &frequency_b,
            &s_b,
            &z0_b,
            2,
            &target,
        )
        .expect("analytical explicit-grid connection must succeed");

        assert_eq!(result.frequency_hz, target);
        assert_eq!(result.s.dim(), (target.len(), 2 + 3, 2 + 3));
        assert_eq!(result.z0.dim(), (target.len(), 5));

        // A and B survivors must remain in their source order.  This also
        // verifies that external complex, frequency-dependent z0 survives
        // composition without a real-50-ohm assumption.  S expectations are
        // constructed directly from the affine definitions below, rather
        // than by invoking the interpolation kernel a second time.
        for (target_index, &frequency) in target.iter().enumerate() {
            for (output_port, &input_port) in [0usize, 2].iter().enumerate() {
                assert_close(
                    result.z0[[target_index, output_port]],
                    affine_z0(frequency, input_port, 40.0),
                );
            }
            for (offset, &input_port) in [0usize, 1, 3].iter().enumerate() {
                assert_close(
                    result.z0[[target_index, 2 + offset]],
                    affine_z0(frequency, input_port, 82.0),
                );
            }

            let a_junction = affine_s(frequency, 1, 1, 0.015);
            let b_junction = affine_s(frequency, 2, 2, -0.021);
            let denominator = complex(1.0, 0.0) - a_junction * b_junction;
            let a_survivors = [0usize, 2];
            let b_survivors = [0usize, 1, 3];
            for (output_row, &row) in a_survivors.iter().enumerate() {
                for (output_column, &column) in a_survivors.iter().enumerate() {
                    let expected = affine_s(frequency, row, column, 0.015)
                        + affine_s(frequency, row, 1, 0.015)
                            * b_junction
                            * affine_s(frequency, 1, column, 0.015)
                            / denominator;
                    assert_close(
                        result.s[[target_index, output_row, output_column]],
                        expected,
                    );
                }
                for (b_offset, &column) in b_survivors.iter().enumerate() {
                    let expected = affine_s(frequency, row, 1, 0.015)
                        * affine_s(frequency, 2, column, -0.021)
                        / denominator;
                    assert_close(
                        result.s[[target_index, output_row, a_survivors.len() + b_offset]],
                        expected,
                    );
                }
            }
            for (b_offset, &row) in b_survivors.iter().enumerate() {
                for (output_column, &column) in a_survivors.iter().enumerate() {
                    let expected = affine_s(frequency, row, 2, -0.021)
                        * affine_s(frequency, 1, column, 0.015)
                        / denominator;
                    assert_close(
                        result.s[[target_index, a_survivors.len() + b_offset, output_column]],
                        expected,
                    );
                }
                for (other_offset, &column) in b_survivors.iter().enumerate() {
                    let expected = affine_s(frequency, row, column, -0.021)
                        + affine_s(frequency, row, 2, -0.021)
                            * a_junction
                            * affine_s(frequency, 2, column, -0.021)
                            / denominator;
                    assert_close(
                        result.s[[
                            target_index,
                            a_survivors.len() + b_offset,
                            a_survivors.len() + other_offset,
                        ]],
                        expected,
                    );
                }
            }
        }
    }

    #[test]
    fn same_grid_matches_direct_connect_exactly() {
        let frequency = [1.0e9, 2.0e9, 3.5e9];
        let s_a = make_s(&frequency, 3, 0.012);
        let s_b = make_s(&frequency, 2, -0.018);
        let z0_a = make_z0(&frequency, 3, 41.0, 1);
        let z0_b = make_z0(&frequency, 2, 79.0, 0);

        let composed = connect_matched_on_grid(
            &frequency, &s_a, &z0_a, 1, &frequency, &s_b, &z0_b, 0, &frequency,
        )
        .unwrap();
        let direct =
            connect_matched(&frequency, &s_a, &z0_a, 1, &frequency, &s_b, &z0_b, 0).unwrap();

        assert_eq!(composed, direct);
    }

    #[test]
    fn attributes_a_and_b_interpolation_failures_deterministically() {
        let valid_frequency = [1.0, 2.0, 3.0];
        let target = [1.0, 2.5, 3.0];
        let valid_s_a = make_s(&valid_frequency, 2, 0.01);
        let valid_s_b = make_s(&valid_frequency, 2, -0.01);
        let valid_z0_a = make_z0(&valid_frequency, 2, 40.0, 0);
        let valid_z0_b = make_z0(&valid_frequency, 2, 80.0, 0);

        let invalid_s_a = Array3::zeros((3, 2, 3));
        assert_eq!(
            connect_matched_on_grid(
                &valid_frequency,
                &invalid_s_a,
                &valid_z0_a,
                0,
                &valid_frequency,
                &valid_s_b,
                &valid_z0_b,
                0,
                &target,
            ),
            Err(CompositionError::AInterpolation(
                InterpolationError::InvalidSShape { shape: (3, 2, 3) }
            ))
        );

        let invalid_s_b = Array3::zeros((3, 2, 3));
        assert_eq!(
            connect_matched_on_grid(
                &valid_frequency,
                &valid_s_a,
                &valid_z0_a,
                0,
                &valid_frequency,
                &invalid_s_b,
                &valid_z0_b,
                0,
                &target,
            ),
            Err(CompositionError::BInterpolation(
                InterpolationError::InvalidSShape { shape: (3, 2, 3) }
            ))
        );

        // A accepts every target knot, while B cannot extrapolate the final
        // target value.  The wrapper must preserve that failure as a B-stage
        // interpolation error rather than reporting a connection mismatch.
        let frequency_a_span = [1.0, 2.0, 4.0];
        let frequency_b_span = [1.0, 2.0, 3.0];
        let target_outside_b = [1.0, 2.5, 4.0];
        let s_a = make_s(&frequency_a_span, 2, 0.01);
        let s_b = make_s(&frequency_b_span, 2, -0.01);
        let z0_a = make_z0(&frequency_a_span, 2, 40.0, 0);
        let z0_b = make_z0(&frequency_b_span, 2, 80.0, 0);
        assert_eq!(
            connect_matched_on_grid(
                &frequency_a_span,
                &s_a,
                &z0_a,
                0,
                &frequency_b_span,
                &s_b,
                &z0_b,
                0,
                &target_outside_b,
            ),
            Err(CompositionError::BInterpolation(
                InterpolationError::TargetFrequencyOutOfRange {
                    index: 2,
                    value: 4.0,
                    lower: 1.0,
                    upper: 3.0,
                }
            ))
        );
    }

    #[test]
    fn attributes_post_interpolation_connection_failures() {
        let frequency_a = [1.0, 2.0, 4.0];
        let frequency_b = [1.0, 1.5, 4.0];
        let target = [1.0, 1.5, 2.0, 4.0];
        let s_a = make_s(&frequency_a, 2, 0.01);
        let s_b = make_s(&frequency_b, 2, -0.01);
        let z0_a = make_z0(&frequency_a, 2, 40.0, 0);
        let z0_b = make_z0(&frequency_b, 2, 80.0, 0);

        assert_eq!(
            connect_matched_on_grid(
                &frequency_a,
                &s_a,
                &z0_a,
                2,
                &frequency_b,
                &s_b,
                &z0_b,
                0,
                &target,
            ),
            Err(CompositionError::Connection(ConnectionError::InvalidPort {
                network: crate::connection::NetworkSide::A,
                port: 2,
                nports: 2,
            }))
        );

        let mut mismatched_z0_b = z0_b.clone();
        mismatched_z0_b[[1, 0]] = complex(74.0, 0.0);
        assert_eq!(
            connect_matched_on_grid(
                &frequency_a,
                &s_a,
                &z0_a,
                0,
                &frequency_b,
                &s_b,
                &mismatched_z0_b,
                0,
                &target,
            ),
            Err(CompositionError::Connection(
                ConnectionError::MismatchedJunctionZ0 {
                    frequency: 1,
                    a: complex(73.5, 0.0),
                    b: complex(74.0, 0.0),
                }
            ))
        );

        let mut invalid_junction_z0_b = z0_b.clone();
        invalid_junction_z0_b[[1, 0]] = complex(73.5, 1.0);
        assert_eq!(
            connect_matched_on_grid(
                &frequency_a,
                &s_a,
                &z0_a,
                0,
                &frequency_b,
                &s_b,
                &invalid_junction_z0_b,
                0,
                &target,
            ),
            Err(CompositionError::Connection(
                ConnectionError::InvalidJunctionZ0 {
                    network: crate::connection::NetworkSide::B,
                    frequency: 1,
                    port: 0,
                    value: complex(73.5, 1.0),
                }
            ))
        );
    }

    #[test]
    fn preserves_connection_error_identity_for_zero_survivor_singularity_and_nonfinite() {
        let frequency = [1.0, 2.0, 3.0];
        let target = [1.0, 2.0, 3.0];
        let one_port_s = make_s(&frequency, 1, 0.01);
        let two_port_s = make_s(&frequency, 2, -0.01);
        let one_port_z0 = make_z0(&frequency, 1, 40.0, 0);
        let two_port_z0 = make_z0(&frequency, 2, 80.0, 0);

        assert_eq!(
            connect_matched_on_grid(
                &frequency,
                &one_port_s,
                &one_port_z0,
                0,
                &frequency,
                &one_port_s,
                &one_port_z0,
                0,
                &target,
            ),
            Err(CompositionError::Connection(
                ConnectionError::NoExternalPorts
            ))
        );

        let mut singular_s_a = two_port_s.clone();
        let mut singular_s_b = two_port_s.clone();
        singular_s_a[[0, 0, 0]] = complex(1.0, 0.0);
        singular_s_b[[0, 0, 0]] = complex(1.0, 0.0);
        assert_eq!(
            connect_matched_on_grid(
                &frequency,
                &singular_s_a,
                &two_port_z0,
                0,
                &frequency,
                &singular_s_b,
                &two_port_z0,
                0,
                &target,
            ),
            Err(CompositionError::Connection(ConnectionError::Singular {
                frequency: 0
            }))
        );

        // Inputs remain finite, but the matched kernel's product overflows;
        // the wrapper must preserve this as a connection-stage error rather
        // than misattributing it to either interpolation stage.
        let short_frequency = [1.0, 2.0];
        let mut overflowing_s_a = Array3::zeros((2, 2, 2));
        let mut overflowing_s_b = Array3::zeros((2, 2, 2));
        overflowing_s_a[[0, 0, 0]] = complex(f64::MAX, 0.0);
        overflowing_s_b[[0, 0, 0]] = complex(2.0, 0.0);
        let overflowing_z0_a = make_z0(&short_frequency, 2, 40.0, 0);
        let overflowing_z0_b = make_z0(&short_frequency, 2, 80.0, 0);
        assert_eq!(
            connect_matched_on_grid(
                &short_frequency,
                &overflowing_s_a,
                &overflowing_z0_a,
                0,
                &short_frequency,
                &overflowing_s_b,
                &overflowing_z0_b,
                0,
                &short_frequency,
            ),
            Err(CompositionError::Connection(
                ConnectionError::NonFiniteComputation {
                    frequency: 0,
                    row: 0,
                    column: 0,
                }
            ))
        );

        let mut nonfinite_s = two_port_s.clone();
        nonfinite_s[[1, 1, 1]] = complex(f64::NAN, 0.0);
        assert_eq!(
            connect_matched_on_grid(
                &frequency,
                &two_port_s,
                &two_port_z0,
                0,
                &frequency,
                &nonfinite_s,
                &two_port_z0,
                0,
                &target,
            ),
            Err(CompositionError::BInterpolation(
                InterpolationError::NonFiniteS {
                    frequency: 1,
                    row: 1,
                    column: 1,
                }
            ))
        );
    }

    #[test]
    fn matches_explicit_grid_composition_oracle_and_contract_metadata() {
        let fixture: FixtureDocument =
            serde_json::from_str(COMPOSITION_FIXTURE_JSON).expect("fixture must parse");
        let metadata = &fixture.metadata;
        assert_eq!(
            metadata.case_id,
            "power_wave_connect_matched_explicit_grid_three_to_four_port_complex_z0"
        );
        assert_eq!(metadata.interpolation.basis, "s");
        assert_eq!(metadata.interpolation.coords, "cart");
        assert_eq!(metadata.interpolation.kind, "linear");
        assert_eq!(metadata.interpolation.scipy_version, "1.18.1");
        assert_eq!(metadata.junction_ports.a, 1);
        assert_eq!(metadata.junction_ports.b, 2);
        assert_eq!(metadata.numpy_version, "2.5.1");
        assert_eq!(metadata.operation, "connect_matched_on_grid");
        assert_eq!(metadata.random_seeds.a, 20_260_943);
        assert_eq!(metadata.random_seeds.b, 20_260_944);
        assert!(metadata.reference_impedance.external_complex);
        assert!(metadata.reference_impedance.external_frequency_dependent);
        assert!(
            metadata
                .reference_impedance
                .junction_exactly_equal_after_interpolation
        );
        assert!(!metadata.reference_impedance.junction_frequency_dependent);
        assert!(metadata.reference_impedance.junction_real_strictly_positive);
        assert_eq!(metadata.reference_impedance.unit, "ohm");
        assert_eq!(metadata.schema, "rfkit-rs.oracle.fixture");
        assert_eq!(metadata.schema_version, 1);
        assert_eq!(metadata.scikit_rf_version, "2.0.1");
        assert_eq!(metadata.wave_definition, "power");
        assert_eq!(metadata.port_order.a_survivors, vec![0, 2]);
        assert_eq!(metadata.port_order.b_survivors, vec![0, 1, 3]);
        assert_eq!(
            metadata.port_order.description,
            "A unconnected ports in original order, followed by B unconnected ports in original order"
        );
        assert_eq!(
            metadata.port_order.output,
            vec![
                PortReference {
                    network: "A".to_owned(),
                    port: 0
                },
                PortReference {
                    network: "A".to_owned(),
                    port: 2
                },
                PortReference {
                    network: "B".to_owned(),
                    port: 0
                },
                PortReference {
                    network: "B".to_owned(),
                    port: 1
                },
                PortReference {
                    network: "B".to_owned(),
                    port: 3
                },
            ]
        );
        assert_eq!(metadata.shape.frequency, vec![8]);
        assert_eq!(metadata.shape.source_frequency_a, vec![5]);
        assert_eq!(metadata.shape.source_frequency_b, vec![6]);
        assert_eq!(metadata.shape.target_frequency, vec![8]);
        assert_eq!(metadata.shape.input_s_a, vec![5, 3, 3]);
        assert_eq!(metadata.shape.input_s_b, vec![6, 4, 4]);
        assert_eq!(metadata.shape.input_z0_a, vec![5, 3]);
        assert_eq!(metadata.shape.input_z0_b, vec![6, 4]);
        assert_eq!(metadata.shape.output_s, vec![8, 5, 5]);
        assert_eq!(metadata.shape.output_z0, vec![8, 5]);
        assert_eq!(metadata.tolerance_policy.rtol, 1.0e-12);
        assert_eq!(metadata.tolerance_policy.atol, 1.0e-12);
        assert_eq!(
            metadata.tolerance_policy.comparison,
            "abs(actual-expected) <= atol + rtol*abs(expected)"
        );
        assert!(metadata.tolerance_policy.justification.contains("SciPy"));
        assert!(
            metadata
                .tolerance_policy
                .regeneration
                .contains("z0_connected_ohm")
        );

        let data = &fixture.data;
        assert_eq!(data.frequency_hz, data.target_frequency_hz);
        assert_eq!(
            data.source_frequency_a_hz,
            vec![0.73e9, 1.21e9, 1.89e9, 2.67e9, 3.41e9]
        );
        assert_eq!(
            data.source_frequency_b_hz,
            vec![0.73e9, 0.96e9, 1.48e9, 2.22e9, 2.93e9, 3.41e9]
        );
        assert_eq!(
            data.target_frequency_hz,
            vec![
                0.73e9, 0.96e9, 1.21e9, 1.73e9, 2.22e9, 2.67e9, 2.93e9, 3.41e9
            ]
        );
        assert_ne!(data.source_frequency_a_hz, data.source_frequency_b_hz);
        assert_eq!(data.s_a.len(), 5);
        assert_eq!(data.s_b.len(), 6);
        assert_eq!(data.z0_a_ohm.len(), 5);
        assert_eq!(data.z0_b_ohm.len(), 6);
        assert_eq!(data.s_connected.len(), 8);
        assert_eq!(data.z0_connected_ohm.len(), 8);
        for frequency in &data.z0_a_ohm {
            assert_eq!(frequency[1].real, 73.5);
            assert_eq!(frequency[1].imag, 0.0);
        }
        for frequency in &data.z0_b_ohm {
            assert_eq!(frequency[2].real, 73.5);
            assert_eq!(frequency[2].imag, 0.0);
        }
        assert!(
            data.z0_a_ohm
                .iter()
                .flatten()
                .any(|value| value.imag != 0.0)
        );
        assert!(
            data.z0_b_ohm
                .iter()
                .flatten()
                .any(|value| value.imag != 0.0)
        );

        let s_a = fixture_array3(&data.s_a);
        let s_b = fixture_array3(&data.s_b);
        let z0_a = fixture_array2(&data.z0_a_ohm);
        let z0_b = fixture_array2(&data.z0_b_ohm);
        let expected_s = fixture_array3(&data.s_connected);
        let expected_z0 = fixture_array2(&data.z0_connected_ohm);
        let result = connect_matched_on_grid(
            &data.source_frequency_a_hz,
            &s_a,
            &z0_a,
            metadata.junction_ports.a,
            &data.source_frequency_b_hz,
            &s_b,
            &z0_b,
            metadata.junction_ports.b,
            &data.target_frequency_hz,
        )
        .expect("fixture explicit-grid composition must succeed");
        assert_eq!(result.frequency_hz, data.target_frequency_hz);
        assert_eq!(result.s.dim(), (8, 5, 5));
        assert_eq!(result.z0.dim(), (8, 5));
        assert_recorded_output(
            result.s.iter().copied(),
            expected_s.iter().copied(),
            metadata.tolerance_policy.rtol,
            metadata.tolerance_policy.atol,
        );
        assert_recorded_output(
            result.z0.iter().copied(),
            expected_z0.iter().copied(),
            metadata.tolerance_policy.rtol,
            metadata.tolerance_policy.atol,
        );
    }
}
