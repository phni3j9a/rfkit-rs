use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, InterpolationAxis, Network};
use serde::Deserialize;

const EXPLICIT_GRID_FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/power_wave_connect_matched_explicit_grid_three_to_four_port_complex_z0.json"
);

#[derive(Debug, Deserialize)]
struct FixtureDocument {
    data: FixtureData,
    metadata: FixtureMetadata,
}

#[derive(Debug, Deserialize)]
struct FixtureData {
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
struct ComplexValue {
    real: f64,
    imag: f64,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    junction_ports: JunctionPorts,
    tolerance_policy: TolerancePolicy,
}

#[derive(Debug, Deserialize)]
struct JunctionPorts {
    a: usize,
    b: usize,
}

#[derive(Debug, Deserialize)]
struct TolerancePolicy {
    atol: f64,
    rtol: f64,
}

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
    assert!(!values.is_empty());
    let nfreq = values.len();
    let nports = values[0].len();
    assert!(nports > 0);
    assert!(values.iter().all(|matrix| matrix.len() == nports));
    assert!(
        values
            .iter()
            .all(|matrix| matrix.iter().all(|row| row.len() == nports))
    );
    let flattened = values
        .iter()
        .flat_map(|matrix| matrix.iter())
        .flat_map(|row| row.iter())
        .map(|value| c(value.real, value.imag))
        .collect::<Vec<_>>();
    Array3::from_shape_vec((nfreq, nports, nports), flattened).unwrap()
}

fn array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    assert!(!values.is_empty());
    let nfreq = values.len();
    let nports = values[0].len();
    assert!(nports > 0);
    assert!(values.iter().all(|row| row.len() == nports));
    let flattened = values
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| c(value.real, value.imag))
        .collect::<Vec<_>>();
    Array2::from_shape_vec((nfreq, nports), flattened).unwrap()
}

fn network(frequency_hz: Vec<f64>, s: Array3<Complex64>, z0: Array2<Complex64>) -> Network {
    Network::new(Frequency::from_hz(frequency_hz).unwrap(), s, z0).unwrap()
}

fn zero_network(frequency_hz: Vec<f64>, nports: usize, z0: Complex64) -> Network {
    let nfreq = frequency_hz.len();
    network(
        frequency_hz,
        Array3::zeros((nfreq, nports, nports)),
        Array2::from_elem((nfreq, nports), z0),
    )
}

/// Explicit-grid migration is caller composition. The two interpolation
/// stages remain ordered A then B, and connection errors are returned without
/// a grid wrapper.
fn explicit_connect(
    a: &Network,
    port_a: usize,
    b: &Network,
    port_b: usize,
    target: &Frequency,
) -> rfkit_core::Result<Network> {
    let interpolated_a = a.interpolate_cartesian_linear(target)?;
    let interpolated_b = b.interpolate_cartesian_linear(target)?;
    interpolated_a.connect_power(port_a, &interpolated_b, port_b)
}

fn assert_array3_close(
    actual: &Array3<Complex64>,
    expected: &Array3<Complex64>,
    rtol: f64,
    atol: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual_value - expected_value).norm();
        let bound = atol + rtol * expected_value.norm();
        assert!(
            difference <= bound,
            "flattened index {index}: actual={actual_value:?}, expected={expected_value:?}, difference={difference:e}, bound={bound:e}"
        );
    }
}

fn assert_array2_close(
    actual: &Array2<Complex64>,
    expected: &Array2<Complex64>,
    rtol: f64,
    atol: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual_value - expected_value).norm();
        let bound = atol + rtol * expected_value.norm();
        assert!(
            difference <= bound,
            "flattened index {index}: actual={actual_value:?}, expected={expected_value:?}, difference={difference:e}, bound={bound:e}"
        );
    }
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    let difference = (actual - expected).norm();
    assert!(
        difference <= 2.0e-12,
        "actual={actual:?}, expected={expected:?}, difference={difference:e}"
    );
}

fn affine_s(frequency: f64, row: usize, column: usize, side_offset: f64) -> Complex64 {
    let row = row as f64;
    let column = column as f64;
    c(
        side_offset + 0.025 * frequency + 0.007 * row - 0.004 * column,
        -side_offset * 0.4 - 0.015 * frequency + 0.003 * row + 0.006 * column,
    )
}

fn affine_z0(frequency: f64, port: usize, side_offset: f64) -> Complex64 {
    let port = port as f64;
    c(
        side_offset + 0.08 * frequency + 2.75 * port,
        1.25 + 0.035 * frequency + 0.45 * port,
    )
}

fn frequency_dependent_junction(frequency: f64) -> Complex64 {
    c(61.25 + 0.5 * frequency, 0.0)
}

fn affine_s_network(frequency: &[f64], nports: usize, side_offset: f64) -> Array3<Complex64> {
    Array3::from_shape_fn((frequency.len(), nports, nports), |(index, row, column)| {
        affine_s(frequency[index], row, column, side_offset)
    })
}

fn affine_z0_network(
    frequency: &[f64],
    nports: usize,
    side_offset: f64,
    junction_port: usize,
) -> Array2<Complex64> {
    Array2::from_shape_fn((frequency.len(), nports), |(index, port)| {
        if port == junction_port {
            frequency_dependent_junction(frequency[index])
        } else {
            affine_z0(frequency[index], port, side_offset)
        }
    })
}

#[test]
fn explicit_interpolation_then_connect_is_analytical_owned_and_nonmutating() {
    let frequency_a = vec![1.0, 2.5, 5.5, 10.0];
    let frequency_b = vec![1.0, 1.75, 4.75, 7.0, 10.0];
    let target_frequency_hz = vec![1.0, 1.75, 2.5, 4.0, 7.0, 10.0];
    let target = Frequency::from_hz(target_frequency_hz.clone()).unwrap();
    let target_before = target.clone();
    let s_a = affine_s_network(&frequency_a, 3, 0.015);
    let s_b = affine_s_network(&frequency_b, 4, -0.021);
    let z0_a = affine_z0_network(&frequency_a, 3, 40.0, 1);
    let z0_b = affine_z0_network(&frequency_b, 4, 82.0, 2);
    let a = network(frequency_a.clone(), s_a.clone(), z0_a.clone());
    let b = network(frequency_b.clone(), s_b.clone(), z0_b.clone());

    let connected = explicit_connect(&a, 1, &b, 2, &target).unwrap();

    assert_eq!(connected.frequency().hz(), target_frequency_hz.as_slice());
    assert_eq!(connected.s().dim(), (target_frequency_hz.len(), 5, 5));
    assert_eq!(connected.z0().dim(), (target_frequency_hz.len(), 5));
    assert_ne!(connected.s().as_ptr(), a.s().as_ptr());
    assert_ne!(connected.z0().as_ptr(), a.z0().as_ptr());
    assert_eq!(target, target_before);
    assert_eq!(a.frequency().hz(), frequency_a.as_slice());
    assert_eq!(a.s(), &s_a);
    assert_eq!(a.z0(), &z0_a);
    assert_eq!(b.frequency().hz(), frequency_b.as_slice());
    assert_eq!(b.s(), &s_b);
    assert_eq!(b.z0(), &z0_b);

    let a_survivors = [0usize, 2];
    let b_survivors = [0usize, 1, 3];
    for (target_index, &frequency) in target_frequency_hz.iter().enumerate() {
        for (output_port, &input_port) in a_survivors.iter().enumerate() {
            assert_complex_close(
                connected.z0()[[target_index, output_port]],
                affine_z0(frequency, input_port, 40.0),
            );
        }
        for (offset, &input_port) in b_survivors.iter().enumerate() {
            assert_complex_close(
                connected.z0()[[target_index, a_survivors.len() + offset]],
                affine_z0(frequency, input_port, 82.0),
            );
        }

        let a_junction = affine_s(frequency, 1, 1, 0.015);
        let b_junction = affine_s(frequency, 2, 2, -0.021);
        let denominator = c(1.0, 0.0) - a_junction * b_junction;
        for (output_row, &row) in a_survivors.iter().enumerate() {
            for (output_column, &column) in a_survivors.iter().enumerate() {
                let expected = affine_s(frequency, row, column, 0.015)
                    + affine_s(frequency, row, 1, 0.015)
                        * b_junction
                        * affine_s(frequency, 1, column, 0.015)
                        / denominator;
                assert_complex_close(
                    connected.s()[[target_index, output_row, output_column]],
                    expected,
                );
            }
            for (b_offset, &column) in b_survivors.iter().enumerate() {
                let expected = affine_s(frequency, row, 1, 0.015)
                    * affine_s(frequency, 2, column, -0.021)
                    / denominator;
                assert_complex_close(
                    connected.s()[[target_index, output_row, a_survivors.len() + b_offset]],
                    expected,
                );
            }
        }
        for (b_offset, &row) in b_survivors.iter().enumerate() {
            for (output_column, &column) in a_survivors.iter().enumerate() {
                let expected = affine_s(frequency, row, 2, -0.021)
                    * affine_s(frequency, 1, column, 0.015)
                    / denominator;
                assert_complex_close(
                    connected.s()[[target_index, a_survivors.len() + b_offset, output_column]],
                    expected,
                );
            }
            for (other_offset, &column) in b_survivors.iter().enumerate() {
                let expected = affine_s(frequency, row, column, -0.021)
                    + affine_s(frequency, row, 2, -0.021)
                        * a_junction
                        * affine_s(frequency, 2, column, -0.021)
                        / denominator;
                assert_complex_close(
                    connected.s()[[
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
fn explicit_interpolation_then_connect_matches_recorded_complex_z0_fixture() {
    let fixture: FixtureDocument = serde_json::from_str(EXPLICIT_GRID_FIXTURE_JSON).unwrap();
    let a = network(
        fixture.data.source_frequency_a_hz.clone(),
        array3(&fixture.data.s_a),
        array2(&fixture.data.z0_a_ohm),
    );
    let b = network(
        fixture.data.source_frequency_b_hz.clone(),
        array3(&fixture.data.s_b),
        array2(&fixture.data.z0_b_ohm),
    );
    let target = Frequency::from_hz(fixture.data.target_frequency_hz.clone()).unwrap();
    let connected = explicit_connect(
        &a,
        fixture.metadata.junction_ports.a,
        &b,
        fixture.metadata.junction_ports.b,
        &target,
    )
    .unwrap();

    assert_eq!(connected.frequency(), &target);
    assert_array3_close(
        connected.s(),
        &array3(&fixture.data.s_connected),
        fixture.metadata.tolerance_policy.rtol,
        fixture.metadata.tolerance_policy.atol,
    );
    assert_array2_close(
        connected.z0(),
        &array2(&fixture.data.z0_connected_ohm),
        fixture.metadata.tolerance_policy.rtol,
        fixture.metadata.tolerance_policy.atol,
    );
}

#[test]
fn explicit_composition_equals_spelled_out_interpolation_and_connection() {
    let frequency_a = vec![1.0, 2.5, 5.5, 10.0];
    let frequency_b = vec![1.0, 1.75, 4.75, 7.0, 10.0];
    let target = Frequency::from_hz(vec![1.0, 1.75, 2.5, 4.0, 7.0, 10.0]).unwrap();
    let a = network(
        frequency_a.clone(),
        affine_s_network(&frequency_a, 3, 0.015),
        affine_z0_network(&frequency_a, 3, 40.0, 1),
    );
    let b = network(
        frequency_b.clone(),
        affine_s_network(&frequency_b, 4, -0.021),
        affine_z0_network(&frequency_b, 4, 82.0, 2),
    );

    let composed = explicit_connect(&a, 1, &b, 2, &target).unwrap();
    let interpolated_a = a.interpolate_cartesian_linear(&target).unwrap();
    let interpolated_b = b.interpolate_cartesian_linear(&target).unwrap();
    let explicit = interpolated_a.connect_power(1, &interpolated_b, 2).unwrap();

    assert_eq!(composed, explicit);
}

#[test]
fn same_grid_explicit_composition_equals_connection() {
    let frequency_hz = vec![1.0, 2.0, 3.5];
    let a = network(
        frequency_hz.clone(),
        affine_s_network(&frequency_hz, 3, 0.012),
        affine_z0_network(&frequency_hz, 3, 41.0, 1),
    );
    let b = network(
        frequency_hz.clone(),
        affine_s_network(&frequency_hz, 2, -0.018),
        affine_z0_network(&frequency_hz, 2, 79.0, 0),
    );
    let target = Frequency::from_hz(frequency_hz).unwrap();
    let explicit_grid = explicit_connect(&a, 1, &b, 0, &target).unwrap();
    let direct = a.connect_power(1, &b, 0).unwrap();

    assert_eq!(explicit_grid, direct);
}

#[test]
fn explicit_composition_accepts_single_target_and_preserves_signed_zero() {
    let source_frequency_hz = vec![-1.0, -0.0, 1.0];
    let target_frequency_hz = vec![0.0];
    let target = Frequency::from_hz(target_frequency_hz.clone()).unwrap();
    let a = zero_network(source_frequency_hz.clone(), 2, c(73.5, 0.0));
    let b = zero_network(source_frequency_hz, 2, c(73.5, 0.0));
    let connected = explicit_connect(&a, 0, &b, 0, &target).unwrap();

    assert_eq!(connected.frequency().len(), 1);
    assert_eq!(connected.frequency().hz()[0].to_bits(), 0.0f64.to_bits());
    assert_eq!(target.hz()[0].to_bits(), 0.0f64.to_bits());
    assert_eq!(connected.s().dim(), (1, 2, 2));
}

#[test]
fn explicit_composition_reports_interpolation_failures_without_grid_wrapper() {
    let valid = zero_network(vec![1.0, 2.0, 3.0], 2, c(50.0, 0.0));
    let target = Frequency::from_hz(vec![1.0, 2.5, 3.0]).unwrap();
    let one_sample_a = zero_network(vec![1.0], 2, c(50.0, 0.0));
    assert_eq!(
        explicit_connect(&one_sample_a, 0, &valid, 0, &target),
        Err(Error::InterpolationTooFewSourceSamples { actual: 1 })
    );

    let one_sample_b = zero_network(vec![1.0], 2, c(50.0, 0.0));
    assert_eq!(
        explicit_connect(&valid, 0, &one_sample_b, 0, &target),
        Err(Error::InterpolationTooFewSourceSamples { actual: 1 })
    );

    let target_outside_b = Frequency::from_hz(vec![1.0, 2.5, 4.0]).unwrap();
    let b_short = zero_network(vec![1.0, 2.0, 3.0], 2, c(50.0, 0.0));
    let a_long = zero_network(vec![1.0, 2.0, 4.0], 2, c(50.0, 0.0));
    assert_eq!(
        explicit_connect(&a_long, 0, &b_short, 0, &target_outside_b),
        Err(Error::InterpolationTargetOutOfRange {
            index: 2,
            value: 4.0,
            lower: 1.0,
            upper: 3.0,
        })
    );
}

#[test]
fn explicit_composition_reports_malformed_axes_and_data_at_interpolation_stage() {
    let valid = zero_network(vec![1.0, 2.0], 2, c(50.0, 0.0));
    let target = Frequency::from_hz(vec![1.5]).unwrap();

    let malformed_a = zero_network(vec![1.0, 1.0], 2, c(50.0, 0.0));
    assert_eq!(
        explicit_connect(&malformed_a, 0, &valid, 0, &target),
        Err(Error::InterpolationFrequencyNotStrictlyIncreasing {
            axis: InterpolationAxis::Source,
            index: 1,
            previous: 1.0,
            current: 1.0,
        })
    );

    let malformed_b = zero_network(vec![2.0, 1.0], 2, c(50.0, 0.0));
    assert_eq!(
        explicit_connect(&valid, 0, &malformed_b, 0, &target),
        Err(Error::InterpolationFrequencyNotStrictlyIncreasing {
            axis: InterpolationAxis::Source,
            index: 1,
            previous: 2.0,
            current: 1.0,
        })
    );

    let mut nonfinite_frequency = vec![1.0, 2.0];
    nonfinite_frequency[1] = f64::INFINITY;
    let nonfinite_a = zero_network(nonfinite_frequency, 2, c(50.0, 0.0));
    assert!(matches!(
        explicit_connect(&nonfinite_a, 0, &valid, 0, &target),
        Err(Error::NonFiniteInterpolationFrequency {
            axis: InterpolationAxis::Source,
            index: 1,
            value,
        }) if value == f64::INFINITY
    ));

    let mut bad_s = valid.s().clone();
    bad_s[[1, 1, 0]] = c(f64::INFINITY, 0.0);
    let bad_a = network(valid.frequency().hz().to_vec(), bad_s, valid.z0().clone());
    assert_eq!(
        explicit_connect(
            &bad_a,
            0,
            &valid,
            0,
            &Frequency::from_hz(vec![1.0, 2.0]).unwrap()
        ),
        Err(Error::NonFiniteInterpolationS {
            frequency: 1,
            row: 1,
            column: 0,
        })
    );

    let mut bad_z0 = valid.z0().clone();
    bad_z0[[0, 1]] = c(50.0, f64::NAN);
    let bad_b = network(valid.frequency().hz().to_vec(), valid.s().clone(), bad_z0);
    assert_eq!(
        explicit_connect(
            &valid,
            0,
            &bad_b,
            0,
            &Frequency::from_hz(vec![1.0, 2.0]).unwrap()
        ),
        Err(Error::NonFiniteInterpolationZ0 {
            frequency: 0,
            port: 1,
        })
    );
}

#[test]
fn explicit_composition_reports_target_validation_before_connection() {
    let a = zero_network(vec![1.0, 2.0], 2, c(50.0, 0.0));
    let b = zero_network(vec![1.0, 2.0], 2, c(50.0, 0.0));

    let target = Frequency::from_hz(vec![f64::NAN]).unwrap();
    assert!(matches!(
        explicit_connect(&a, 0, &b, 0, &target),
        Err(Error::NonFiniteInterpolationFrequency {
            axis: InterpolationAxis::Target,
            index: 0,
            value,
        }) if value.is_nan()
    ));

    let target = Frequency::from_hz(vec![1.5, 1.5]).unwrap();
    assert_eq!(
        explicit_connect(&a, 0, &b, 0, &target),
        Err(Error::InterpolationFrequencyNotStrictlyIncreasing {
            axis: InterpolationAxis::Target,
            index: 1,
            previous: 1.5,
            current: 1.5,
        })
    );

    let target = Frequency::from_hz(vec![0.5]).unwrap();
    assert_eq!(
        explicit_connect(&a, 99, &b, 99, &target),
        Err(Error::InterpolationTargetOutOfRange {
            index: 0,
            value: 0.5,
            lower: 1.0,
            upper: 2.0,
        })
    );
}

#[test]
fn explicit_composition_keeps_connection_diagnostics_unwrapped() {
    let a = zero_network(vec![1.0, 2.0], 2, c(50.0, 0.0));
    let b = zero_network(vec![1.0, 2.0], 2, c(50.0, 0.0));
    let target = Frequency::from_hz(vec![1.0, 1.5, 2.0]).unwrap();
    assert_eq!(
        explicit_connect(&a, 2, &b, 0, &target),
        Err(Error::InvalidDirectConnectionPort {
            input: rfkit_core::ConnectionInput::A,
            port: 2,
            nports: 2,
        })
    );

    let one_a = zero_network(vec![1.0, 2.0], 1, c(50.0, 0.0));
    let one_b = zero_network(vec![1.0, 2.0], 1, c(50.0, 0.0));
    let target = Frequency::from_hz(vec![1.0, 2.0]).unwrap();
    assert_eq!(
        explicit_connect(&one_a, 0, &one_b, 0, &target),
        Err(Error::NoExternalDirectConnectionPorts)
    );

    let mut singular_a_s = Array3::zeros((2, 2, 2));
    let mut singular_b_s = Array3::zeros((2, 2, 2));
    singular_a_s[[0, 0, 0]] = c(1.0, 0.0);
    singular_b_s[[0, 0, 0]] = c(1.0, 0.0);
    singular_a_s[[1, 0, 0]] = c(1.0, 0.0);
    singular_b_s[[1, 0, 0]] = c(1.0, 0.0);
    let singular_a = network(
        vec![1.0, 2.0],
        singular_a_s,
        Array2::from_elem((2, 2), c(50.0, 0.0)),
    );
    let singular_b = network(
        vec![1.0, 2.0],
        singular_b_s,
        Array2::from_elem((2, 2), c(50.0, 0.0)),
    );
    assert_eq!(
        explicit_connect(&singular_a, 0, &singular_b, 0, &target),
        Err(Error::SingularConnection { frequency: 0 })
    );

    let mut overflowing_a_s = Array3::zeros((2, 2, 2));
    let mut overflowing_b_s = Array3::zeros((2, 2, 2));
    // Keep the selector on the matched path while making the final external
    // Schur correction genuinely unrepresentable.  A selected diagonal
    // product alone is not sufficient: exact Schur division can cancel it
    // while the external network remains zero.
    overflowing_a_s[[0, 1, 0]] = c(f64::MAX, 0.0);
    overflowing_a_s[[0, 0, 1]] = c(f64::MAX, 0.0);
    overflowing_b_s[[0, 0, 0]] = c(2.0, 0.0);
    let overflowing_a = network(
        vec![1.0, 2.0],
        overflowing_a_s,
        Array2::from_elem((2, 2), c(50.0, 0.0)),
    );
    let overflowing_b = network(
        vec![1.0, 2.0],
        overflowing_b_s,
        Array2::from_elem((2, 2), c(50.0, 0.0)),
    );
    assert_eq!(
        explicit_connect(&overflowing_a, 0, &overflowing_b, 0, &target),
        Err(Error::NonFiniteConnectionComputation {
            frequency: 0,
            row: 0,
            column: 0,
        })
    );
}

#[test]
fn explicit_composition_uses_direct_domain_for_post_interpolation_mismatch() {
    let source_frequency_a = vec![1.0, 3.0];
    let source_frequency_b = vec![1.0, 2.0, 3.0];
    let target = Frequency::from_hz(vec![1.5]).unwrap();
    let a = network(
        source_frequency_a,
        Array3::zeros((2, 2, 2)),
        Array2::from_shape_vec(
            (2, 2),
            vec![c(50.0, 0.0), c(80.0, 0.0), c(60.0, 0.0), c(80.0, 0.0)],
        )
        .unwrap(),
    );
    let mut z0_b = Array2::from_elem((3, 2), c(50.0, 0.0));
    z0_b[[0, 0]] = c(50.0, 0.0);
    z0_b[[1, 0]] = c(54.0, 0.0);
    z0_b[[2, 0]] = c(60.0, 0.0);
    let b = network(source_frequency_b, Array3::zeros((3, 2, 2)), z0_b);
    let connected = explicit_connect(&a, 0, &b, 0, &target).unwrap();
    assert_eq!(connected.nports(), 2);
    assert!(connected.s().iter().all(|value| value.is_finite()));

    let mut complex_z0_b = b.z0().clone();
    complex_z0_b[[1, 0]] = c(54.0, 1.0);
    let complex_b = network(b.frequency().hz().to_vec(), b.s().clone(), complex_z0_b);
    let connected = explicit_connect(&a, 0, &complex_b, 0, &target).unwrap();
    assert_eq!(connected.nports(), 2);
    assert!(connected.s().iter().all(|value| value.is_finite()));
}
