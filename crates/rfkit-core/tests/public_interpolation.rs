use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, InterpolationAxis, InterpolationQuantity, Network};
use serde::Deserialize;

const INTERPOLATION_FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/interpolation_cartesian_linear_three_port_complex_z0.json"
);

#[derive(Debug, Deserialize)]
struct FixtureDocument {
    data: FixtureData,
    metadata: FixtureMetadata,
}

#[derive(Debug, Deserialize)]
struct FixtureData {
    s: Vec<Vec<Vec<ComplexValue>>>,
    s_input: Vec<Vec<Vec<ComplexValue>>>,
    source_frequency_hz: Vec<f64>,
    target_frequency_hz: Vec<f64>,
    z0_input_ohm: Vec<Vec<ComplexValue>>,
    z0_ohm: Vec<Vec<ComplexValue>>,
}

#[derive(Debug, Deserialize)]
struct ComplexValue {
    imag: f64,
    real: f64,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    tolerance_policy: TolerancePolicy,
}

#[derive(Debug, Deserialize)]
struct TolerancePolicy {
    atol: f64,
    rtol: f64,
}

fn complex(value: &ComplexValue) -> Complex64 {
    Complex64::new(value.real, value.imag)
}

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
    assert!(!values.is_empty());
    let nfreq = values.len();
    let nport = values[0].len();
    assert!(nport > 0);
    assert!(values.iter().all(|matrix| matrix.len() == nport));
    assert!(
        values
            .iter()
            .all(|matrix| matrix.iter().all(|row| row.len() == nport))
    );
    let values = values
        .iter()
        .flat_map(|matrix| matrix.iter())
        .flat_map(|row| row.iter())
        .map(complex)
        .collect::<Vec<_>>();
    Array3::from_shape_vec((nfreq, nport, nport), values).unwrap()
}

fn array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    assert!(!values.is_empty());
    let nfreq = values.len();
    let nport = values[0].len();
    assert!(nport > 0);
    assert!(values.iter().all(|row| row.len() == nport));
    let values = values
        .iter()
        .flat_map(|row| row.iter())
        .map(complex)
        .collect::<Vec<_>>();
    Array2::from_shape_vec((nfreq, nport), values).unwrap()
}

fn network(frequency_hz: Vec<f64>, s: Array3<Complex64>, z0: Array2<Complex64>) -> Network {
    Network::new(Frequency::from_hz(frequency_hz).unwrap(), s, z0).unwrap()
}

fn assert_array3_close(
    actual: &Array3<Complex64>,
    expected: &Array3<Complex64>,
    rtol: f64,
    atol: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual_value.re.is_finite() && actual_value.im.is_finite(),
            "actual S value at flattened index {index} is non-finite"
        );
        let difference = (actual_value - expected_value).norm();
        let bound = atol + rtol * expected_value.norm();
        assert!(
            difference <= bound,
            "S value at flattened index {index} differs by {difference:?}, bound {bound:?}; actual={actual_value:?}, expected={expected_value:?}"
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
        assert!(
            actual_value.re.is_finite() && actual_value.im.is_finite(),
            "actual z0 value at flattened index {index} is non-finite"
        );
        let difference = (actual_value - expected_value).norm();
        let bound = atol + rtol * expected_value.norm();
        assert!(
            difference <= bound,
            "z0 value at flattened index {index} differs by {difference:?}, bound {bound:?}; actual={actual_value:?}, expected={expected_value:?}"
        );
    }
}

fn affine_s(frequency: f64, row: usize, column: usize) -> Complex64 {
    let row = row as f64;
    let column = column as f64;
    Complex64::new(
        0.125 * frequency + 0.75 * row - 0.5 * column,
        -0.375 * frequency + 0.25 * row + 1.25 * column,
    )
}

fn affine_z0(frequency: f64, port: usize) -> Complex64 {
    Complex64::new(
        37.0 + 1.75 * frequency + 4.5 * port as f64,
        2.0 - 0.625 * frequency + 0.8 * port as f64,
    )
}

#[test]
fn public_interpolation_handles_irregular_affine_complex_three_port_data() {
    let source_frequency_hz = vec![1.0, 2.5, 6.0, 10.0];
    let target_frequency_hz = vec![1.0, 1.75, 2.5, 4.25, 6.0, 8.0, 10.0];
    let source_s = Array3::from_shape_fn(
        (source_frequency_hz.len(), 3, 3),
        |(frequency, row, column)| affine_s(source_frequency_hz[frequency], row, column),
    );
    let source_z0 = Array2::from_shape_fn((source_frequency_hz.len(), 3), |(frequency, port)| {
        affine_z0(source_frequency_hz[frequency], port)
    });
    let source = network(
        source_frequency_hz.clone(),
        source_s.clone(),
        source_z0.clone(),
    );
    let target = Frequency::from_hz(target_frequency_hz.clone()).unwrap();
    let target_before = target.clone();

    let result = source.interpolate_cartesian_linear(&target).unwrap();

    assert_eq!(result.frequency().hz(), target_frequency_hz.as_slice());
    assert_eq!(result.nports(), 3);
    assert_eq!(result.s().dim(), (target_frequency_hz.len(), 3, 3));
    assert_eq!(result.z0().dim(), (target_frequency_hz.len(), 3));
    assert_ne!(result.s().as_ptr(), source.s().as_ptr());
    assert_ne!(result.z0().as_ptr(), source.z0().as_ptr());
    assert_eq!(target, target_before);
    assert_eq!(source.frequency().hz(), source_frequency_hz.as_slice());
    assert_eq!(source.s(), &source_s);
    assert_eq!(source.z0(), &source_z0);

    for (target_index, &frequency) in target_frequency_hz.iter().enumerate() {
        for row in 0..3 {
            for column in 0..3 {
                assert!(
                    (result.s()[[target_index, row, column]] - affine_s(frequency, row, column))
                        .norm()
                        <= 1.0e-12
                );
            }
        }
        for port in 0..3 {
            assert!(
                (result.z0()[[target_index, port]] - affine_z0(frequency, port)).norm() <= 1.0e-12
            );
        }
    }
}

#[test]
fn public_interpolation_matches_the_pinned_complex_z0_fixture() {
    let fixture: FixtureDocument =
        serde_json::from_str(INTERPOLATION_FIXTURE_JSON).expect("fixture must parse");
    let source = network(
        fixture.data.source_frequency_hz.clone(),
        array3(&fixture.data.s_input),
        array2(&fixture.data.z0_input_ohm),
    );
    let target = Frequency::from_hz(fixture.data.target_frequency_hz.clone()).unwrap();
    let result = source
        .interpolate_cartesian_linear(&target)
        .expect("public fixture interpolation must succeed");

    assert_eq!(result.frequency().hz(), target.hz());
    assert_array3_close(
        result.s(),
        &array3(&fixture.data.s),
        fixture.metadata.tolerance_policy.rtol,
        fixture.metadata.tolerance_policy.atol,
    );
    assert_array2_close(
        result.z0(),
        &array2(&fixture.data.z0_ohm),
        fixture.metadata.tolerance_policy.rtol,
        fixture.metadata.tolerance_policy.atol,
    );
}

#[test]
fn public_interpolation_copies_exact_knots_and_supports_same_grid_identity() {
    let source_frequency_hz = vec![1.0, 2.5, 6.0];
    let source_s = Array3::from_shape_fn((3, 2, 2), |(frequency, row, column)| {
        Complex64::new(
            0.25 + 3.0 * frequency as f64 + row as f64,
            -0.5 + column as f64,
        )
    });
    let source_z0 = Array2::from_shape_fn((3, 2), |(frequency, port)| {
        Complex64::new(
            40.0 + 5.0 * frequency as f64 + port as f64,
            port as f64 - 0.5,
        )
    });
    let source = network(
        source_frequency_hz.clone(),
        source_s.clone(),
        source_z0.clone(),
    );

    let same_grid = Frequency::from_hz(source_frequency_hz.clone()).unwrap();
    let identity = source.interpolate_cartesian_linear(&same_grid).unwrap();
    assert_eq!(identity.frequency(), &same_grid);
    assert_eq!(identity.s(), &source_s);
    assert_eq!(identity.z0(), &source_z0);
    assert_ne!(identity.s().as_ptr(), source.s().as_ptr());
    assert_ne!(identity.z0().as_ptr(), source.z0().as_ptr());

    let subset = Frequency::from_hz(vec![1.0, 6.0]).unwrap();
    let endpoints = source.interpolate_cartesian_linear(&subset).unwrap();
    assert_eq!(
        endpoints.s().slice(ndarray::s![0, .., ..]),
        source_s.slice(ndarray::s![0, .., ..])
    );
    assert_eq!(
        endpoints.s().slice(ndarray::s![1, .., ..]),
        source_s.slice(ndarray::s![2, .., ..])
    );
    assert_eq!(
        endpoints.z0().slice(ndarray::s![0, ..]),
        source_z0.slice(ndarray::s![0, ..])
    );
    assert_eq!(
        endpoints.z0().slice(ndarray::s![1, ..]),
        source_z0.slice(ndarray::s![2, ..])
    );
}

#[test]
fn public_interpolation_accepts_complex_non_50_zero_and_negative_real_z0() {
    let source = network(
        vec![1.0, 3.0],
        Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                Complex64::new(0.1, -0.2),
                Complex64::new(0.2, 0.3),
                Complex64::new(-0.4, 0.5),
                Complex64::new(0.6, -0.7),
                Complex64::new(0.8, 0.9),
                Complex64::new(-1.0, 1.1),
                Complex64::new(1.2, -1.3),
                Complex64::new(-1.4, -1.5),
            ],
        )
        .unwrap(),
        Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 4.0),
                Complex64::new(-75.0, 2.0),
                Complex64::new(-20.0, -6.0),
                Complex64::new(125.0, 8.0),
            ],
        )
        .unwrap(),
    );
    let target = Frequency::from_hz(vec![2.0]).unwrap();
    let result = source.interpolate_cartesian_linear(&target).unwrap();

    assert_eq!(result.frequency(), &target);
    assert_eq!(result.z0()[[0, 0]], Complex64::new(-10.0, -1.0));
    assert_eq!(result.z0()[[0, 1]], Complex64::new(25.0, 5.0));
}

#[test]
fn public_interpolation_retains_negative_frequency_and_signed_zero_knot_behavior() {
    let source_frequency_hz = vec![-2.0, -0.0, 2.0];
    let source_s = Array3::from_shape_vec(
        (3, 1, 1),
        vec![
            Complex64::new(-2.0, 9.0),
            Complex64::new(17.25, -0.0),
            Complex64::new(-3.5, -7.0),
        ],
    )
    .unwrap();
    let source_z0 = Array2::from_shape_vec(
        (3, 1),
        vec![
            Complex64::new(48.0, -3.0),
            Complex64::new(-125.5, -0.0),
            Complex64::new(61.0, 5.0),
        ],
    )
    .unwrap();
    let source = network(source_frequency_hz, source_s.clone(), source_z0.clone());
    let target = Frequency::from_hz(vec![0.0, 1.0]).unwrap();

    let result = source.interpolate_cartesian_linear(&target).unwrap();

    assert_eq!(result.frequency().hz(), target.hz());
    assert_eq!(
        result.frequency().hz()[0].to_bits(),
        target.hz()[0].to_bits()
    );
    assert_eq!(target.hz()[0].to_bits(), 0.0f64.to_bits());
    assert_eq!(source.frequency().hz()[1].to_bits(), (-0.0f64).to_bits());

    let source_knot_s = source_s[[1, 0, 0]];
    let copied_s = result.s()[[0, 0, 0]];
    assert_eq!(copied_s.re.to_bits(), source_knot_s.re.to_bits());
    assert_eq!(copied_s.im.to_bits(), source_knot_s.im.to_bits());

    let source_knot_z0 = source_z0[[1, 0]];
    let copied_z0 = result.z0()[[0, 0]];
    assert_eq!(copied_z0.re.to_bits(), source_knot_z0.re.to_bits());
    assert_eq!(copied_z0.im.to_bits(), source_knot_z0.im.to_bits());

    assert_eq!(result.s()[[1, 0, 0]], Complex64::new(6.875, -3.5));
    assert_eq!(result.z0()[[1, 0]], Complex64::new(-32.25, 2.5));
}

#[test]
fn public_interpolation_retains_extreme_finite_cross_zero_handling() {
    let source_frequency_hz = vec![-f64::MAX, 0.0, f64::MAX];
    let source_s = Array3::from_shape_fn((3, 1, 1), |(frequency, _, _)| {
        Complex64::new(
            source_frequency_hz[frequency],
            -source_frequency_hz[frequency],
        )
    });
    let source_z0 = Array2::from_shape_fn((3, 1), |(frequency, _)| {
        Complex64::new(
            -source_frequency_hz[frequency],
            source_frequency_hz[frequency],
        )
    });
    let source = network(source_frequency_hz, source_s, source_z0);
    let target = Frequency::from_hz(vec![-f64::MAX / 2.0, 0.0, f64::MAX / 2.0]).unwrap();

    let result = source.interpolate_cartesian_linear(&target).unwrap();

    assert_eq!(
        result.s()[[0, 0, 0]],
        Complex64::new(-f64::MAX / 2.0, f64::MAX / 2.0)
    );
    assert_eq!(result.s()[[1, 0, 0]], Complex64::new(0.0, 0.0));
    assert_eq!(
        result.s()[[2, 0, 0]],
        Complex64::new(f64::MAX / 2.0, -f64::MAX / 2.0)
    );
    assert_eq!(
        result.z0()[[0, 0]],
        Complex64::new(f64::MAX / 2.0, -f64::MAX / 2.0)
    );
    assert_eq!(
        result.z0()[[2, 0]],
        Complex64::new(-f64::MAX / 2.0, f64::MAX / 2.0)
    );
}

#[test]
fn public_interpolation_does_not_bypass_one_sample_or_nonfinite_data_validation() {
    let one_sample = network(
        vec![1.0],
        Array3::from_elem((1, 1, 1), Complex64::new(0.1, 0.0)),
        Array2::from_elem((1, 1), Complex64::new(50.0, 0.0)),
    );
    let same_grid = Frequency::from_hz(vec![1.0]).unwrap();
    assert_eq!(
        one_sample
            .interpolate_cartesian_linear(&same_grid)
            .unwrap_err(),
        Error::InterpolationTooFewSourceSamples { actual: 1 }
    );

    let mut nonfinite_s = Array3::from_elem((2, 1, 1), Complex64::new(0.0, 0.0));
    nonfinite_s[[1, 0, 0]] = Complex64::new(f64::NAN, 0.0);
    let source = network(
        vec![1.0, 2.0],
        nonfinite_s,
        Array2::from_elem((2, 1), Complex64::new(50.0, 0.0)),
    );
    let error = source
        .interpolate_cartesian_linear(&Frequency::from_hz(vec![1.0, 2.0]).unwrap())
        .unwrap_err();
    assert_eq!(
        error,
        Error::NonFiniteInterpolationS {
            frequency: 1,
            row: 0,
            column: 0,
        }
    );

    let mut nonfinite_z0 = Array2::from_elem((2, 1), Complex64::new(50.0, 0.0));
    nonfinite_z0[[0, 0]] = Complex64::new(50.0, f64::INFINITY);
    let source = network(vec![1.0, 2.0], Array3::zeros((2, 1, 1)), nonfinite_z0);
    let error = source
        .interpolate_cartesian_linear(&Frequency::from_hz(vec![1.0, 2.0]).unwrap())
        .unwrap_err();
    assert_eq!(
        error,
        Error::NonFiniteInterpolationZ0 {
            frequency: 0,
            port: 0,
        }
    );
}

#[test]
fn public_interpolation_reports_structured_axis_and_range_errors() {
    let valid_s = Array3::zeros((2, 1, 1));
    let valid_z0 = Array2::from_elem((2, 1), Complex64::new(50.0, 0.0));

    let source = network(vec![1.0, f64::NAN], valid_s.clone(), valid_z0.clone());
    let error = source
        .interpolate_cartesian_linear(&Frequency::from_hz(vec![1.0]).unwrap())
        .unwrap_err();
    assert!(matches!(
        error,
        Error::NonFiniteInterpolationFrequency {
            axis: InterpolationAxis::Source,
            index: 1,
            value,
        } if value.is_nan()
    ));

    let source = network(vec![1.0, 2.0], valid_s.clone(), valid_z0.clone());
    let error = source
        .interpolate_cartesian_linear(&Frequency::from_hz(vec![f64::INFINITY]).unwrap())
        .unwrap_err();
    assert_eq!(
        error,
        Error::NonFiniteInterpolationFrequency {
            axis: InterpolationAxis::Target,
            index: 0,
            value: f64::INFINITY,
        }
    );

    let source = network(vec![2.0, 1.0], valid_s.clone(), valid_z0.clone());
    let error = source
        .interpolate_cartesian_linear(&Frequency::from_hz(vec![1.5]).unwrap())
        .unwrap_err();
    assert_eq!(
        error,
        Error::InterpolationFrequencyNotStrictlyIncreasing {
            axis: InterpolationAxis::Source,
            index: 1,
            previous: 2.0,
            current: 1.0,
        }
    );

    let source = network(vec![1.0, 1.0], valid_s.clone(), valid_z0.clone());
    let error = source
        .interpolate_cartesian_linear(&Frequency::from_hz(vec![1.0]).unwrap())
        .unwrap_err();
    assert_eq!(
        error,
        Error::InterpolationFrequencyNotStrictlyIncreasing {
            axis: InterpolationAxis::Source,
            index: 1,
            previous: 1.0,
            current: 1.0,
        }
    );

    let source = network(vec![1.0, 2.0], valid_s.clone(), valid_z0.clone());
    let error = source
        .interpolate_cartesian_linear(&Frequency::from_hz(vec![1.5, 1.5]).unwrap())
        .unwrap_err();
    assert_eq!(
        error,
        Error::InterpolationFrequencyNotStrictlyIncreasing {
            axis: InterpolationAxis::Target,
            index: 1,
            previous: 1.5,
            current: 1.5,
        }
    );

    let source = network(vec![1.0, 2.0], valid_s, valid_z0);
    let error = source
        .interpolate_cartesian_linear(&Frequency::from_hz(vec![2.0, 1.5]).unwrap())
        .unwrap_err();
    assert_eq!(
        error,
        Error::InterpolationFrequencyNotStrictlyIncreasing {
            axis: InterpolationAxis::Target,
            index: 1,
            previous: 2.0,
            current: 1.5,
        }
    );

    for (target, value) in [(vec![0.5], 0.5), (vec![2.5], 2.5)] {
        let source = network(
            vec![1.0, 2.0],
            Array3::zeros((2, 1, 1)),
            Array2::from_elem((2, 1), Complex64::new(50.0, 0.0)),
        );
        let error = source
            .interpolate_cartesian_linear(&Frequency::from_hz(target).unwrap())
            .unwrap_err();
        assert_eq!(
            error,
            Error::InterpolationTargetOutOfRange {
                index: 0,
                value,
                lower: 1.0,
                upper: 2.0,
            }
        );
    }
}

#[test]
fn public_interpolation_quantity_display_is_actionable() {
    assert_eq!(InterpolationQuantity::S.to_string(), "S-parameter");
    assert_eq!(InterpolationQuantity::Z0.to_string(), "reference-impedance");
}
