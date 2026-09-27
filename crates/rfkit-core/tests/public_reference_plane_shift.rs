use approx::assert_relative_eq;
use ndarray::{Array2, Array3, ShapeBuilder};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, Network, ReferencePlaneShiftArithmetic};
use serde_json::{Value, json};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn network(frequency_hz: Vec<f64>, s: Array3<Complex64>, z0: Array2<Complex64>) -> Network {
    Network::new(Frequency::from_hz(frequency_hz).unwrap(), s, z0).unwrap()
}

fn positive_z0(nfreq: usize, nport: usize) -> Array2<Complex64> {
    Array2::from_shape_fn((nfreq, nport), |(frequency, port)| {
        c(31.0 + 7.0 * port as f64 + 2.0 * frequency as f64, 0.0)
    })
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_relative_eq!(
        actual.re,
        expected.re,
        epsilon = 2.0e-14,
        max_relative = 2.0e-13
    );
    assert_relative_eq!(
        actual.im,
        expected.im,
        epsilon = 2.0e-14,
        max_relative = 2.0e-13
    );
}

fn assert_bits(left: Complex64, right: Complex64) {
    assert_eq!(left.re.to_bits(), right.re.to_bits());
    assert_eq!(left.im.to_bits(), right.im.to_bits());
}

#[test]
fn one_port_reflection_receives_twice_the_one_way_phase() {
    let theta = 0.73;
    let source_value = c(0.41, -0.27);
    let source = network(
        vec![2.4e9],
        Array3::from_elem((1, 1, 1), source_value),
        Array2::from_elem((1, 1), c(73.0, 0.0)),
    );

    let shifted = source
        .shift_reference_planes_lossless_power(&Array2::from_elem((1, 1), theta))
        .unwrap();
    let expected = source_value * Complex64::from_polar(1.0, -2.0 * theta);
    assert_complex_close(shifted.s()[[0, 0, 0]], expected);
}

#[test]
fn two_port_transmission_uses_independent_endpoint_phases() {
    let source_value = c(0.23, -0.61);
    let source = network(
        vec![1.0e9],
        Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.0, 0.0), source_value, c(-0.4, 0.1), c(0.0, 0.0)],
        )
        .unwrap(),
        Array2::from_shape_vec((1, 2), vec![c(37.0, 0.0), c(91.0, 0.0)]).unwrap(),
    );
    let phase = Array2::from_shape_vec((1, 2), vec![0.41, -1.2]).unwrap();

    let shifted = source
        .shift_reference_planes_lossless_power(&phase)
        .unwrap();
    let expected = source_value * Complex64::from_polar(1.0, -(0.41 - 1.2));
    assert_complex_close(shifted.s()[[0, 0, 1]], expected);
}

#[test]
fn asymmetric_multi_sample_n_port_data_matches_dsd_and_keeps_input_immutable() {
    let frequency_hz = vec![-0.0, -4.0e9, 2.5e9];
    let source_s = Array3::from_shape_fn((3, 3, 3), |(frequency, row, column)| {
        c(
            0.08 * (frequency + 1) as f64 + 0.011 * row as f64 + 0.007 * column as f64,
            -0.03 * (frequency + 1) as f64 + 0.013 * row as f64 - 0.009 * column as f64,
        )
    });
    let source_z0 = Array2::from_shape_fn((3, 3), |(frequency, port)| {
        c([42.0, 68.0, 109.0][port] + frequency as f64, 0.0)
    });
    let source = network(frequency_hz.clone(), source_s.clone(), source_z0.clone());
    let phases = Array2::from_shape_fn((3, 3), |(frequency, port)| {
        [[0.0, 0.4, -0.7], [1.1, -0.2, 0.9], [-1.3, 0.8, 0.0]][frequency][port]
    });
    let source_before = source.clone();

    let shifted = source
        .shift_reference_planes_lossless_power(&phases)
        .unwrap();
    for frequency in 0..3 {
        for row in 0..3 {
            for column in 0..3 {
                let expected = Complex64::from_polar(1.0, -phases[[frequency, row]])
                    * source_s[[frequency, row, column]]
                    * Complex64::from_polar(1.0, -phases[[frequency, column]]);
                assert_complex_close(shifted.s()[[frequency, row, column]], expected);
            }
        }
    }
    assert_eq!(source, source_before);
    assert_eq!(shifted.frequency().hz(), frequency_hz.as_slice());
    assert_eq!(shifted.z0(), &source_z0);
    assert_ne!(shifted.s().as_ptr(), source.s().as_ptr());
    assert_ne!(shifted.z0().as_ptr(), source.z0().as_ptr());
}

#[test]
fn zero_positive_and_negative_phase_entries_are_supported() {
    let source_s = Array3::from_shape_vec(
        (1, 3, 3),
        vec![
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, -1.0),
            c(-1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 1.0),
            c(0.0, 1.0),
            c(0.0, -1.0),
            c(0.25, 0.5),
        ],
    )
    .unwrap();
    let phases = Array2::from_shape_vec((1, 3), vec![0.0, 0.7, -1.4]).unwrap();
    let source = network(vec![1.0], source_s, positive_z0(1, 3));
    let shifted = source
        .shift_reference_planes_lossless_power(&phases)
        .unwrap();

    for row in 0..3 {
        for column in 0..3 {
            let factor = Complex64::from_polar(1.0, -phases[[0, row]] - phases[[0, column]]);
            assert_complex_close(
                shifted.s()[[0, row, column]],
                source.s()[[0, row, column]] * factor,
            );
        }
    }
}

#[test]
fn preserves_frequency_and_reference_bits_for_finite_pointwise_metadata() {
    let frequency = vec![-0.0, f64::from_bits(0x4009_21fb_5444_2d18), -8.5];
    let z0 = Array2::from_shape_vec(
        (3, 2),
        vec![
            c(50.0, -0.0),
            c(71.0, 0.0),
            c(50.0, 0.0),
            c(71.0, -0.0),
            c(50.0, -0.0),
            c(71.0, 0.0),
        ],
    )
    .unwrap();
    let source = network(
        frequency.clone(),
        Array3::from_elem((3, 2, 2), c(0.2, -0.1)),
        z0.clone(),
    );
    let shifted = source
        .shift_reference_planes_lossless_power(&Array2::zeros((3, 2)))
        .unwrap();

    for (actual, expected) in shifted.frequency().hz().iter().zip(frequency.iter()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    for (actual, expected) in shifted.z0().iter().zip(z0.iter()) {
        assert_bits(*actual, *expected);
    }
}

#[test]
fn supports_nonstandard_owned_ndarray_layouts() {
    let standard_s = Array3::from_shape_fn((2, 3, 3), |(frequency, row, column)| {
        c(
            0.1 + 0.02 * frequency as f64 + 0.03 * row as f64 + 0.01 * column as f64,
            -0.04 + 0.01 * frequency as f64 - 0.02 * row as f64 + 0.03 * column as f64,
        )
    });
    let source_s = Array3::from_shape_vec(
        (2, 3, 3).strides((9, 1, 3)),
        standard_s.iter().copied().collect(),
    )
    .unwrap();
    let standard_z0 = positive_z0(2, 3);
    let source_z0 = Array2::from_shape_vec(
        (2, 3).strides((1, 2)),
        standard_z0.iter().copied().collect(),
    )
    .unwrap();
    let standard_phase = Array2::from_shape_fn((2, 3), |(frequency, port)| {
        0.2 * frequency as f64 - 0.3 * port as f64
    });
    let phase = Array2::from_shape_vec(
        (2, 3).strides((1, 2)),
        standard_phase.iter().copied().collect(),
    )
    .unwrap();
    assert!(!source_s.is_standard_layout());
    assert!(!source_z0.is_standard_layout());
    assert!(!phase.is_standard_layout());

    let source = network(vec![1.0, 2.0], source_s.clone(), source_z0.clone());
    let shifted = source
        .shift_reference_planes_lossless_power(&phase)
        .unwrap();
    for frequency in 0..2 {
        for row in 0..3 {
            for column in 0..3 {
                let expected = Complex64::from_polar(1.0, -phase[[frequency, row]])
                    * source_s[[frequency, row, column]]
                    * Complex64::from_polar(1.0, -phase[[frequency, column]]);
                assert_complex_close(shifted.s()[[frequency, row, column]], expected);
            }
        }
    }
}

#[test]
fn accepts_zero_singular_active_and_nonreciprocal_networks() {
    let source_s = Array3::from_shape_vec(
        (2, 2, 2),
        vec![
            c(0.0, 0.0),
            c(2.0, -1.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(f64::MAX / 4.0, 0.0),
            c(-0.7, 0.2),
            c(0.9, 0.1),
            c(0.0, 0.0),
        ],
    )
    .unwrap();
    let source = network(vec![0.0, 1.0], source_s, positive_z0(2, 2));
    let shifted = source
        .shift_reference_planes_lossless_power(
            &Array2::from_shape_vec((2, 2), vec![0.0, -0.7, 1.2, 0.4]).unwrap(),
        )
        .unwrap();
    assert!(
        shifted
            .s()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn rejects_nonfinite_inputs_and_invalid_reference_domain_even_for_zero_phase() {
    let phases = Array2::zeros((1, 1));
    let valid = network(
        vec![1.0],
        Array3::from_elem((1, 1, 1), c(0.2, 0.1)),
        Array2::from_elem((1, 1), c(50.0, 0.0)),
    );

    let mut nonfinite_phase = phases.clone();
    nonfinite_phase[[0, 0]] = f64::NAN;
    assert_eq!(
        valid.shift_reference_planes_lossless_power(&nonfinite_phase),
        Err(Error::NonFiniteReferencePlaneShiftPhase {
            frequency: 0,
            port: 0,
        })
    );

    let nonfinite_s = network(
        vec![1.0],
        Array3::from_elem((1, 1, 1), c(f64::INFINITY, 0.0)),
        Array2::from_elem((1, 1), c(50.0, 0.0)),
    );
    assert_eq!(
        nonfinite_s.shift_reference_planes_lossless_power(&phases),
        Err(Error::NonFiniteReferencePlaneShiftS {
            frequency: 0,
            row: 0,
            column: 0,
        })
    );

    for reference in [c(50.0, 1.0), c(0.0, 0.0), c(-50.0, 0.0)] {
        let invalid = network(
            vec![1.0],
            Array3::from_elem((1, 1, 1), c(0.2, 0.1)),
            Array2::from_elem((1, 1), reference),
        );
        assert_eq!(
            invalid.shift_reference_planes_lossless_power(&phases),
            Err(Error::InvalidReferencePlaneShiftZ0 {
                frequency: 0,
                port: 0,
                value: reference,
            })
        );
    }
}

#[test]
fn rejects_nonfinite_frequency_and_reference_impedance() {
    let phases = Array2::zeros((1, 1));
    let s = Array3::from_elem((1, 1, 1), c(0.2, 0.1));

    let nonfinite_frequency = network(
        vec![f64::NAN],
        s.clone(),
        Array2::from_elem((1, 1), c(50.0, 0.0)),
    );
    assert!(matches!(
        nonfinite_frequency.shift_reference_planes_lossless_power(&phases),
        Err(Error::NonFiniteReferencePlaneShiftFrequency { index: 0, value })
            if !value.is_finite()
    ));

    let nonfinite_z0 = network(
        vec![1.0],
        s,
        Array2::from_elem((1, 1), c(50.0, f64::INFINITY)),
    );
    assert_eq!(
        nonfinite_z0
            .shift_reference_planes_lossless_power(&phases)
            .unwrap_err(),
        Error::NonFiniteReferencePlaneShiftZ0 {
            frequency: 0,
            port: 0,
        }
    );
}

#[test]
fn accepts_duplicate_frequency_samples_as_pointwise_labels() {
    let source = network(
        vec![1.0, 1.0],
        Array3::from_shape_vec((2, 1, 1), vec![c(0.2, -0.1), c(-0.3, 0.4)]).unwrap(),
        Array2::from_elem((2, 1), c(50.0, 0.0)),
    );
    let shifted = source
        .shift_reference_planes_lossless_power(
            &Array2::from_shape_vec((2, 1), vec![0.0, 0.5]).unwrap(),
        )
        .unwrap();

    assert_eq!(shifted.frequency().hz(), &[1.0, 1.0]);
    assert_complex_close(shifted.s()[[0, 0, 0]], c(0.2, -0.1));
    assert_complex_close(
        shifted.s()[[1, 0, 0]],
        c(-0.3, 0.4) * Complex64::from_polar(1.0, -1.0),
    );
}

#[test]
fn returns_structured_error_for_unrepresentable_rotated_component() {
    let source = network(
        vec![1.0],
        Array3::from_elem((1, 1, 1), c(f64::MAX, f64::MAX)),
        Array2::from_elem((1, 1), c(50.0, 0.0)),
    );
    let result = source.shift_reference_planes_lossless_power(&Array2::from_elem(
        (1, 1),
        std::f64::consts::FRAC_PI_8,
    ));
    assert_eq!(
        result,
        Err(Error::NonFiniteReferencePlaneShiftComputation {
            frequency: 0,
            row: 0,
            column: 0,
            stage: ReferencePlaneShiftArithmetic::Output,
        })
    );
}

#[test]
fn preserves_representable_huge_s_after_combining_endpoint_factors() {
    let source = network(
        vec![1.0],
        Array3::from_elem((1, 1, 1), c(f64::MAX, 0.0)),
        Array2::from_elem((1, 1), c(50.0, 0.0)),
    );
    let shifted = source
        .shift_reference_planes_lossless_power(&Array2::from_elem((1, 1), 1.0e308))
        .unwrap();
    assert!(shifted.s()[[0, 0, 0]].re.is_finite());
    assert!(shifted.s()[[0, 0, 0]].im.is_finite());
}

#[test]
fn preserves_representable_huge_s_for_opposite_huge_phases() {
    let theta = f64::from_bits(0x7fe1_ccf3_85eb_c899);
    let source = network(
        vec![1.0],
        Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.0, 0.0), c(f64::MAX, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        )
        .unwrap(),
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    );
    let shifted = source
        .shift_reference_planes_lossless_power(
            &Array2::from_shape_vec((1, 2), vec![theta, -theta]).unwrap(),
        )
        .unwrap();
    let actual = shifted.s()[[0, 0, 1]];

    assert!(actual.re.is_finite());
    assert!(actual.im.is_finite());
    assert_relative_eq!(actual.re, f64::MAX, max_relative = 8.0e-15);
    assert_eq!(actual.im, 0.0);
}

#[test]
fn preserves_representable_huge_s_for_same_sign_huge_phases() {
    let source = network(
        vec![1.0],
        Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.0, 0.0), c(f64::MAX, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        )
        .unwrap(),
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    );
    let shifted = source
        .shift_reference_planes_lossless_power(
            &Array2::from_shape_vec((1, 2), vec![1.0e308, 1.0e308]).unwrap(),
        )
        .unwrap();
    let actual = shifted.s()[[0, 0, 1]];

    assert!(actual.re.is_finite());
    assert!(actual.im.is_finite());
}

#[test]
fn phase_negation_round_trip_recovers_moderate_data() {
    let source = network(
        vec![0.0, 1.0, 2.0],
        Array3::from_shape_fn((3, 3, 3), |(frequency, row, column)| {
            c(
                0.2 + 0.03 * frequency as f64 + 0.01 * row as f64,
                -0.1 + 0.02 * column as f64,
            )
        }),
        positive_z0(3, 3),
    );
    let phases = Array2::from_shape_fn((3, 3), |(frequency, port)| {
        [[0.1, -0.2, 0.7], [1.2, 0.4, -0.8], [-0.3, 0.9, 0.2]][frequency][port]
    });
    let shifted = source
        .shift_reference_planes_lossless_power(&phases)
        .unwrap();
    let recovered = shifted
        .shift_reference_planes_lossless_power(&phases.mapv(|value| -value))
        .unwrap();
    for (actual, expected) in recovered.s().iter().zip(source.s()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn phase_table_composition_matches_one_explicit_combined_shift() {
    let source = network(
        vec![1.0, 2.0],
        Array3::from_shape_fn((2, 2, 2), |(frequency, row, column)| {
            c(
                0.2 + 0.1 * frequency as f64 + 0.03 * row as f64,
                -0.1 + 0.02 * column as f64,
            )
        }),
        positive_z0(2, 2),
    );
    let first = Array2::from_shape_vec((2, 2), vec![0.1, -0.3, 0.4, 0.7]).unwrap();
    let second = Array2::from_shape_vec((2, 2), vec![-0.2, 0.6, -0.5, 0.1]).unwrap();
    let sequential = source
        .shift_reference_planes_lossless_power(&first)
        .unwrap()
        .shift_reference_planes_lossless_power(&second)
        .unwrap();
    let combined = source
        .shift_reference_planes_lossless_power(&(&first + &second))
        .unwrap();
    for (actual, expected) in sequential.s().iter().zip(combined.s()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn magnitude_and_singular_values_are_preserved() {
    let source = network(
        vec![1.0, 2.0],
        Array3::from_shape_fn((2, 3, 3), |(frequency, row, column)| {
            c(
                0.12 * (frequency + 1) as f64 + 0.03 * row as f64,
                -0.08 * (column + 1) as f64,
            )
        }),
        positive_z0(2, 3),
    );
    let phases = Array2::from_shape_vec((2, 3), vec![0.3, -0.7, 1.2, -0.4, 0.8, -1.1]).unwrap();
    let shifted = source
        .shift_reference_planes_lossless_power(&phases)
        .unwrap();
    for (actual, expected) in shifted.s().iter().zip(source.s()) {
        assert_relative_eq!(actual.norm(), expected.norm(), epsilon = 3.0e-14);
    }
    let source_sigma = source.max_singular_value_power().unwrap();
    let shifted_sigma = shifted.max_singular_value_power().unwrap();
    for (actual, expected) in shifted_sigma.iter().zip(source_sigma.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 5.0e-13, max_relative = 5.0e-13);
    }
}

#[test]
fn full_port_selection_covariance_holds_with_reordered_phase_table() {
    let source = network(
        vec![1.0, 2.0],
        Array3::from_shape_fn((2, 4, 4), |(frequency, row, column)| {
            c(
                0.1 + 0.01 * frequency as f64 + 0.02 * row as f64,
                -0.2 + 0.03 * column as f64,
            )
        }),
        positive_z0(2, 4),
    );
    let phases =
        Array2::from_shape_vec((2, 4), vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]).unwrap();
    let order = [3, 1, 0, 2];
    let source_shifted = source
        .shift_reference_planes_lossless_power(&phases)
        .unwrap();
    let reordered_source = source.select_ports_zero_incident(&order).unwrap();
    let reordered_phases =
        Array2::from_shape_fn((2, 4), |(frequency, port)| phases[[frequency, order[port]]]);
    let reordered_shifted = reordered_source
        .shift_reference_planes_lossless_power(&reordered_phases)
        .unwrap();
    let expected = source_shifted.select_ports_zero_incident(&order).unwrap();
    for (actual, expected) in reordered_shifted.s().iter().zip(expected.s()) {
        assert_complex_close(*actual, *expected);
    }
}

#[test]
fn malformed_serde_networks_and_phase_shapes_return_without_panics() {
    fn base() -> Value {
        serde_json::to_value(network(
            vec![1.0, 2.0],
            Array3::from_elem((2, 2, 2), c(0.1, -0.2)),
            Array2::from_elem((2, 2), c(50.0, 0.0)),
        ))
        .unwrap()
    }
    fn deserialize(value: Value) -> Network {
        serde_json::from_value(value).unwrap()
    }
    fn set_shape(value: &mut Value, field: &str, shape: &[usize], count: usize) {
        value[field]["dim"] = json!(shape);
        let original = value[field]["data"].as_array().unwrap().clone();
        value[field]["data"] = json!(
            (0..count)
                .map(|index| original[index % original.len()].clone())
                .collect::<Vec<_>>()
        );
    }
    fn assert_error(network: Network, phase: &Array2<f64>, expected: Error) {
        let result = catch_unwind(AssertUnwindSafe(|| {
            network.shift_reference_planes_lossless_power(phase)
        }));
        assert!(result.is_ok(), "reference-plane shift panicked");
        assert_eq!(result.unwrap().unwrap_err(), expected);
    }

    let phase = Array2::zeros((2, 2));
    let mut value = base();
    value["frequency"]["hz"] = json!([]);
    assert_error(
        deserialize(value),
        &Array2::zeros((2, 2)),
        Error::EmptyReferencePlaneShiftFrequency,
    );

    let mut value = base();
    value["frequency"]["hz"] = json!([1.0]);
    assert_error(
        deserialize(value),
        &Array2::zeros((1, 2)),
        Error::ReferencePlaneShiftFrequencyLengthMismatch {
            expected: 1,
            actual: 2,
        },
    );

    let mut value = base();
    set_shape(&mut value, "s", &[2, 2, 3], 12);
    assert_error(
        deserialize(value),
        &phase,
        Error::InvalidReferencePlaneShiftSShape {
            shape: vec![2, 2, 3],
        },
    );

    let mut value = base();
    set_shape(&mut value, "z0", &[2, 3], 6);
    assert_error(
        deserialize(value),
        &phase,
        Error::InvalidReferencePlaneShiftZ0Shape { shape: vec![2, 3] },
    );

    let mut value = base();
    value["s"]["dim"] = json!([3, 2, 2]);
    let original = value["s"]["data"].as_array().unwrap().clone();
    value["s"]["data"] = json!(
        (0..12)
            .map(|index| original[index % original.len()].clone())
            .collect::<Vec<_>>()
    );
    assert_error(
        deserialize(value),
        &phase,
        Error::ReferencePlaneShiftFrequencyLengthMismatch {
            expected: 2,
            actual: 3,
        },
    );

    let valid = deserialize(base());
    assert_eq!(
        valid.shift_reference_planes_lossless_power(&Array2::zeros((1, 2))),
        Err(Error::InvalidReferencePlaneShiftPhaseShape { shape: vec![1, 2] })
    );
}
