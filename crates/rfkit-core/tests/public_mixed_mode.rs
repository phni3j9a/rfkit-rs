use ndarray::{Array2, Array3, ArrayView2, Axis};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, MixedModeDirection, MixedModeMode, MixedModeScaling, Network};
use serde_json::{Value, json};
use std::panic::{AssertUnwindSafe, catch_unwind};

const TOLERANCE: f64 = 3.0e-13;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn assert_close(actual: Complex64, expected: Complex64, tolerance: f64) {
    let error = (actual - expected).norm();
    assert!(
        error <= tolerance,
        "actual={actual:?}, expected={expected:?}, error={error:e}"
    );
}

fn assert_array3_close(actual: &Array3<Complex64>, expected: &Array3<Complex64>) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        assert_close(actual_value, expected_value, TOLERANCE);
        assert!(
            actual_value.re.is_finite() && actual_value.im.is_finite(),
            "non-finite output at linear index {index}"
        );
    }
}

fn asymmetric_network(nfreq: usize, nport: usize) -> Network {
    let frequency = Frequency::from_hz(
        (0..nfreq)
            .map(|index| [f64::NAN, -2.5e9, 4.0e9, -0.0][index])
            .collect(),
    )
    .unwrap();
    let s = Array3::from_shape_fn((nfreq, nport, nport), |(f, row, column)| {
        c(
            0.021 * (1 + f + 3 * row + 5 * column) as f64,
            -0.013 * (1 + 2 * f + row + 7 * column) as f64,
        )
    });
    let z0 = Array2::from_shape_fn((nfreq, nport), |(f, port)| {
        let references = [
            c(42.0 + 1.5 * f as f64, 4.0 - f as f64),
            c(42.0 + 1.5 * f as f64, 4.0 - f as f64),
            c(-63.0 - 2.0 * f as f64, -5.0 + 0.5 * f as f64),
            c(-63.0 - 2.0 * f as f64, -5.0 + 0.5 * f as f64),
            c(77.0 + 0.25 * f as f64, -9.0),
            c(83.0 - 0.25 * f as f64, 2.5),
        ];
        references[port]
    });
    Network::new(frequency, s, z0).unwrap()
}

fn u_entry(nport: usize, pair_count: usize, row: usize, column: usize) -> f64 {
    let q = 1.0 / 2.0_f64.sqrt();
    if row < pair_count {
        let start = 2 * row;
        if column == start {
            q
        } else if column == start + 1 {
            -q
        } else {
            0.0
        }
    } else if row < 2 * pair_count {
        let start = 2 * (row - pair_count);
        if column == start || column == start + 1 {
            q
        } else {
            0.0
        }
    } else if column == row {
        1.0
    } else {
        debug_assert!(row < nport);
        0.0
    }
}

fn expected_forward(slice: ArrayView2<'_, Complex64>, pair_count: usize) -> Array2<Complex64> {
    let nport = slice.nrows();
    Array2::from_shape_fn((nport, nport), |(row, column)| {
        let mut value = c(0.0, 0.0);
        for left in 0..nport {
            for right in 0..nport {
                value += slice[[left, right]]
                    * u_entry(nport, pair_count, row, left)
                    * u_entry(nport, pair_count, column, right);
            }
        }
        value
    })
}

fn expected_inverse(slice: ArrayView2<'_, Complex64>, pair_count: usize) -> Array2<Complex64> {
    let nport = slice.nrows();
    Array2::from_shape_fn((nport, nport), |(row, column)| {
        let mut value = c(0.0, 0.0);
        for modal_row in 0..nport {
            for modal_column in 0..nport {
                value += slice[[modal_row, modal_column]]
                    * u_entry(nport, pair_count, modal_row, row)
                    * u_entry(nport, pair_count, modal_column, column);
            }
        }
        value
    })
}

#[test]
fn forward_matches_analytical_usut_for_two_port_and_preserves_source() {
    let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let s = Array3::from_shape_vec(
        (1, 2, 2),
        vec![c(0.1, 0.2), c(0.3, -0.1), c(-0.4, 0.5), c(0.7, 0.6)],
    )
    .unwrap();
    let z0 = Array2::from_shape_vec((1, 2), vec![c(50.0, 7.0), c(50.0, 7.0)]).unwrap();
    let input = Network::new(frequency, s, z0).unwrap();
    let input_before = input.clone();

    let output = input.to_mixed_mode_equal_pair_power(1).unwrap();
    let expected = expected_forward(input.s().index_axis(Axis(0), 0), 1);
    for row in 0..2 {
        for column in 0..2 {
            assert_close(
                output.s()[[0, row, column]],
                expected[[row, column]],
                TOLERANCE,
            );
        }
    }
    assert_eq!(output.z0()[[0, 0]], c(100.0, 14.0));
    assert_eq!(output.z0()[[0, 1]], c(25.0, 3.5));
    assert_eq!(input.s(), input_before.s());
    assert_eq!(input.z0(), input_before.z0());
    for (actual, expected) in input
        .frequency()
        .hz()
        .iter()
        .zip(input_before.frequency().hz())
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_ne!(output.s().as_ptr(), input.s().as_ptr());
    assert_ne!(output.z0().as_ptr(), input.z0().as_ptr());
}

#[test]
fn forward_supports_odd_multi_frequency_negative_complex_pair_references_and_unpaired_copy() {
    let input = asymmetric_network(3, 5);
    let input_before = input.clone();
    let output = input.to_mixed_mode_equal_pair_power(2).unwrap();

    for frequency in 0..3 {
        let expected = expected_forward(input.s().index_axis(Axis(0), frequency), 2);
        for row in 0..5 {
            for column in 0..5 {
                assert_close(
                    output.s()[[frequency, row, column]],
                    expected[[row, column]],
                    TOLERANCE,
                );
            }
        }
        assert_eq!(
            output.z0()[[frequency, 0]],
            input.z0()[[frequency, 0]] * 2.0
        );
        assert_eq!(
            output.z0()[[frequency, 1]],
            input.z0()[[frequency, 2]] * 2.0
        );
        assert_eq!(
            output.z0()[[frequency, 2]],
            input.z0()[[frequency, 0]] / 2.0
        );
        assert_eq!(
            output.z0()[[frequency, 3]],
            input.z0()[[frequency, 2]] / 2.0
        );
        assert_eq!(output.z0()[[frequency, 4]], input.z0()[[frequency, 4]]);
    }
    for (actual, expected) in output.frequency().hz().iter().zip(input.frequency().hz()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(input.s(), input_before.s());
    assert_eq!(input.z0(), input_before.z0());
    for (actual, expected) in input
        .frequency()
        .hz()
        .iter()
        .zip(input_before.frequency().hz())
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn supports_three_and_four_port_coordinate_layouts() {
    for (nport, pair_count) in [(3, 1), (4, 2)] {
        let input = asymmetric_network(2, nport);
        let output = input.to_mixed_mode_equal_pair_power(pair_count).unwrap();
        assert_eq!(output.s().dim(), (2, nport, nport));
        assert_eq!(output.z0().dim(), (2, nport));
        for frequency in 0..2 {
            let expected = expected_forward(input.s().index_axis(Axis(0), frequency), pair_count);
            for row in 0..nport {
                for column in 0..nport {
                    assert_close(
                        output.s()[[frequency, row, column]],
                        expected[[row, column]],
                        TOLERANCE,
                    );
                }
            }
        }
    }
}

#[test]
fn inverse_is_independently_verified_and_round_trip_restores_asymmetric_data() {
    let source = asymmetric_network(3, 6);
    let modal = source.to_mixed_mode_equal_pair_power(2).unwrap();
    let inverse = modal.to_single_ended_equal_pair_power(2).unwrap();

    for frequency in 0..3 {
        let expected = expected_inverse(modal.s().index_axis(Axis(0), frequency), 2);
        for row in 0..6 {
            for column in 0..6 {
                assert_close(
                    inverse.s()[[frequency, row, column]],
                    expected[[row, column]],
                    TOLERANCE,
                );
            }
        }
    }
    assert_array3_close(inverse.s(), source.s());
    assert_eq!(inverse.z0(), source.z0());
    for (actual, expected) in inverse.frequency().hz().iter().zip(source.frequency().hz()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    // The inverse is also authored directly, rather than obtained from the
    // forward method, so its input-coordinate interpretation is exercised.
    let modal_s = Array3::from_shape_fn((2, 6, 6), |(f, row, column)| {
        c(
            0.04 * (1 + f + row + 2 * column) as f64,
            -0.03 * (2 + 2 * f + 3 * row + column) as f64,
        )
    });
    let modal_z0 = Array2::from_shape_fn((2, 6), |(f, port)| match port {
        0 => c(2.0 * (40.0 + f as f64), 6.0),
        1 => c(2.0 * (-55.0 - f as f64), -4.0),
        2 => c((40.0 + f as f64) / 2.0, 1.5),
        3 => c((-55.0 - f as f64) / 2.0, -1.0),
        4 => c(91.0, -1.0),
        5 => c(97.0, 2.0),
        _ => unreachable!(),
    });
    let modal_network = Network::new(
        Frequency::from_hz(vec![8.0e9, 6.0e9]).unwrap(),
        modal_s,
        modal_z0,
    )
    .unwrap();
    let restored = modal_network.to_single_ended_equal_pair_power(2).unwrap();
    for frequency in 0..2 {
        let expected = expected_inverse(modal_network.s().index_axis(Axis(0), frequency), 2);
        for row in 0..6 {
            for column in 0..6 {
                assert_close(
                    restored.s()[[frequency, row, column]],
                    expected[[row, column]],
                    TOLERANCE,
                );
            }
        }
    }
    assert_eq!(restored.z0()[[0, 0]], c(40.0, 3.0));
    assert_eq!(restored.z0()[[0, 1]], c(40.0, 3.0));
    assert_eq!(restored.z0()[[0, 2]], c(-55.0, -2.0));
    assert_eq!(restored.z0()[[0, 3]], c(-55.0, -2.0));
}

fn waves(
    voltage: &[Complex64],
    current: &[Complex64],
    z0: &[Complex64],
) -> (Vec<Complex64>, Vec<Complex64>) {
    let mut a = Vec::with_capacity(voltage.len());
    let mut b = Vec::with_capacity(voltage.len());
    for ((&voltage, &current), &reference) in voltage.iter().zip(current).zip(z0) {
        let f = 1.0 / (2.0 * reference.re.abs().sqrt());
        a.push((voltage + reference * current) * f);
        b.push((voltage - reference.conj() * current) * f);
    }
    (a, b)
}

fn matrix_vector(
    matrix: &Array3<Complex64>,
    frequency: usize,
    vector: &[Complex64],
) -> Vec<Complex64> {
    (0..vector.len())
        .map(|row| {
            (0..vector.len())
                .map(|column| matrix[[frequency, row, column]] * vector[column])
                .sum()
        })
        .collect()
}

#[test]
fn forward_wave_coordinates_match_independent_vi_derivation() {
    let source = asymmetric_network(1, 5);
    let modal = source.to_mixed_mode_equal_pair_power(2).unwrap();
    let z_se: Vec<_> = source.z0().row(0).to_vec();
    let a_se = vec![
        c(0.12, -0.07),
        c(-0.03, 0.18),
        c(0.04, 0.09),
        c(-0.11, -0.02),
        c(0.06, 0.03),
    ];
    let b_se = matrix_vector(source.s(), 0, &a_se);
    let current: Vec<_> = a_se
        .iter()
        .zip(&b_se)
        .zip(&z_se)
        .map(|((&a, &b), &reference)| {
            let f = 1.0 / (2.0 * reference.re.abs().sqrt());
            (a - b) / (f * (2.0 * reference.re))
        })
        .collect();
    let voltage: Vec<_> = a_se
        .iter()
        .zip(&current)
        .zip(&z_se)
        .map(|((&a, &current), &reference)| {
            let f = 1.0 / (2.0 * reference.re.abs().sqrt());
            a / f - reference * current
        })
        .collect();
    let (a_check, b_check) = waves(&voltage, &current, &z_se);
    for port in 0..5 {
        assert_close(a_check[port], a_se[port], 2.0e-13);
        assert_close(b_check[port], b_se[port], 2.0e-13);
    }
    let mut a_expected = [c(0.0, 0.0); 5];
    let mut b_expected = [c(0.0, 0.0); 5];
    for mode in 0..5 {
        for port in 0..5 {
            let coefficient = u_entry(5, 2, mode, port);
            a_expected[mode] += coefficient * a_se[port];
            b_expected[mode] += coefficient * b_se[port];
        }
    }
    let z_modal: Vec<_> = modal.z0().row(0).to_vec();
    let voltage_modal = vec![
        voltage[0] - voltage[1],
        voltage[2] - voltage[3],
        (voltage[0] + voltage[1]) / 2.0,
        (voltage[2] + voltage[3]) / 2.0,
        voltage[4],
    ];
    let current_modal = vec![
        (current[0] - current[1]) / 2.0,
        (current[2] - current[3]) / 2.0,
        current[0] + current[1],
        current[2] + current[3],
        current[4],
    ];
    let (a_modal_from_vi, b_modal_from_vi) = waves(&voltage_modal, &current_modal, &z_modal);
    for mode in 0..5 {
        assert_close(a_modal_from_vi[mode], a_expected[mode], 2.0e-13);
        assert_close(b_modal_from_vi[mode], b_expected[mode], 2.0e-13);
    }
    let b_from_source = matrix_vector(source.s(), 0, &a_se);
    let a_modal_from_u: Vec<_> = (0..5)
        .map(|row| {
            (0..5)
                .map(|column| u_entry(5, 2, row, column) * a_se[column])
                .sum()
        })
        .collect();
    let b_modal_from_u: Vec<_> = (0..5)
        .map(|row| {
            (0..5)
                .map(|column| u_entry(5, 2, row, column) * b_from_source[column])
                .sum()
        })
        .collect();
    let b_from_modal = matrix_vector(modal.s(), 0, &a_modal_from_u);
    for mode in 0..5 {
        assert_close(a_modal_from_u[mode], a_expected[mode], 2.0e-13);
        assert_close(b_modal_from_u[mode], b_expected[mode], 2.0e-13);
        assert_close(b_from_modal[mode], b_expected[mode], 2.0e-13);
    }
}

#[test]
fn singular_and_ideal_networks_transform_without_z_or_y_inversion() {
    let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let open = Network::new(
        frequency.clone(),
        Array2::eye(4).insert_axis(Axis(0)),
        Array2::from_shape_fn((1, 4), |(_, port)| {
            if port < 2 {
                c(50.0, 0.0)
            } else {
                c(-70.0, 1.0)
            }
        }),
    )
    .unwrap();
    let modal = open.to_mixed_mode_equal_pair_power(2).unwrap();
    let expected = expected_forward(open.s().index_axis(Axis(0), 0), 2);
    assert_array3_close(modal.s(), &expected.insert_axis(Axis(0)));

    let short = Network::new(
        frequency.clone(),
        Array3::from_shape_fn((1, 4, 4), |(_, row, column)| {
            if row == column {
                c(-1.0, 0.0)
            } else {
                c(0.0, 0.0)
            }
        }),
        open.z0().clone(),
    )
    .unwrap();
    assert!(short.to_mixed_mode_equal_pair_power(2).is_ok());

    // A reciprocal, lossless ideal thru becomes independent differential and
    // common reflections with opposite sign; no Z/Y singularity is involved.
    let thru = Network::new(
        frequency,
        Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        )
        .unwrap(),
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    )
    .unwrap();
    let modal_thru = thru.to_mixed_mode_equal_pair_power(1).unwrap();
    assert_close(modal_thru.s()[[0, 0, 0]], c(-1.0, 0.0), TOLERANCE);
    assert_close(modal_thru.s()[[0, 0, 1]], c(0.0, 0.0), TOLERANCE);
    assert_close(modal_thru.s()[[0, 1, 0]], c(0.0, 0.0), TOLERANCE);
    assert_close(modal_thru.s()[[0, 1, 1]], c(1.0, 0.0), TOLERANCE);
}

fn valid_serialized_network(nfreq: usize, nport: usize) -> Value {
    let mut value = serde_json::to_value(asymmetric_network(nfreq, nport)).unwrap();
    value["frequency"]["hz"] = json!(
        (0..nfreq)
            .map(|index| 1.0e9 + index as f64)
            .collect::<Vec<_>>()
    );
    value
}

fn deserialize_network(value: Value) -> Network {
    serde_json::from_value(value).expect("malformed shape fixture should deserialize")
}

fn set_array_shape(value: &mut Value, field: &str, shape: &[usize], data_len: usize) {
    value[field]["dim"] = json!(shape);
    let data = value[field]["data"]
        .as_array()
        .expect("ndarray serde data is an array")
        .iter()
        .take(data_len)
        .cloned()
        .collect::<Vec<_>>();
    value[field]["data"] = json!(data);
}

fn assert_no_panic(network: Network, pair_count: usize, expected: Error) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        network.to_mixed_mode_equal_pair_power(pair_count)
    }));
    assert!(result.is_ok(), "mixed-mode conversion panicked");
    assert_eq!(result.unwrap().unwrap_err(), expected);
}

#[test]
fn rejects_pair_counts_and_malformed_serde_shapes_without_panicking() {
    let input = asymmetric_network(1, 3);
    assert_eq!(
        input.to_mixed_mode_equal_pair_power(0).unwrap_err(),
        Error::MixedModePairCountOutOfRange {
            direction: MixedModeDirection::ToMixedMode,
            pair_count: 0,
            nports: 3,
        }
    );
    assert_eq!(
        input
            .to_mixed_mode_equal_pair_power(usize::MAX)
            .unwrap_err(),
        Error::MixedModePairCountOutOfRange {
            direction: MixedModeDirection::ToMixedMode,
            pair_count: usize::MAX,
            nports: 3,
        }
    );

    let empty_frequency = deserialize_network({
        let mut value = valid_serialized_network(1, 3);
        value["frequency"]["hz"] = json!([]);
        value
    });
    assert_no_panic(
        empty_frequency,
        1,
        Error::EmptyMixedModeFrequency {
            direction: MixedModeDirection::ToMixedMode,
        },
    );

    let mismatched_frequency = deserialize_network({
        let mut value = valid_serialized_network(1, 3);
        value["frequency"]["hz"] = json!([1.0e9, 2.0e9]);
        value
    });
    assert_no_panic(
        mismatched_frequency,
        1,
        Error::MixedModeFrequencyLengthMismatch {
            direction: MixedModeDirection::ToMixedMode,
            expected: 2,
            actual: 1,
        },
    );

    let malformed_s = deserialize_network({
        let mut value = valid_serialized_network(1, 3);
        set_array_shape(&mut value, "s", &[1, 3, 2], 6);
        value
    });
    assert_no_panic(
        malformed_s,
        1,
        Error::InvalidMixedModeSShape {
            direction: MixedModeDirection::ToMixedMode,
            shape: vec![1, 3, 2],
        },
    );

    let malformed_z0 = deserialize_network({
        let mut value = valid_serialized_network(1, 3);
        set_array_shape(&mut value, "z0", &[1, 2], 2);
        value
    });
    assert_no_panic(
        malformed_z0,
        1,
        Error::InvalidMixedModeZ0Shape {
            direction: MixedModeDirection::ToMixedMode,
            shape: vec![1, 2],
        },
    );
}

#[test]
fn rejects_nonfinite_zero_real_unequal_and_scaling_loss_references() {
    let valid = asymmetric_network(1, 3);
    let mut nonfinite_s = valid.s().clone();
    nonfinite_s[[0, 1, 2]] = c(f64::NAN, 0.0);
    let network = Network::new(valid.frequency().clone(), nonfinite_s, valid.z0().clone()).unwrap();
    assert_eq!(
        network.to_mixed_mode_equal_pair_power(1).unwrap_err(),
        Error::NonFiniteMixedModeS {
            direction: MixedModeDirection::ToMixedMode,
            frequency: 0,
            row: 1,
            column: 2,
        }
    );

    let mut nonfinite_z0 = valid.z0().clone();
    nonfinite_z0[[0, 0]] = c(f64::INFINITY, 0.0);
    let network = Network::new(valid.frequency().clone(), valid.s().clone(), nonfinite_z0).unwrap();
    assert_eq!(
        network.to_mixed_mode_equal_pair_power(1).unwrap_err(),
        Error::NonFiniteMixedModeZ0 {
            direction: MixedModeDirection::ToMixedMode,
            frequency: 0,
            port: 0,
        }
    );

    let mut zero_real = valid.z0().clone();
    zero_real[[0, 0]] = c(0.0, 1.0);
    zero_real[[0, 1]] = c(0.0, 1.0);
    let network = Network::new(valid.frequency().clone(), valid.s().clone(), zero_real).unwrap();
    assert_eq!(
        network.to_mixed_mode_equal_pair_power(1).unwrap_err(),
        Error::ZeroRealMixedModeReferenceImpedance {
            direction: MixedModeDirection::ToMixedMode,
            frequency: 0,
            port: 0,
        }
    );

    let mut unequal = valid.z0().clone();
    unequal[[0, 1]] += c(0.0, 1.0e-12);
    let network = Network::new(valid.frequency().clone(), valid.s().clone(), unequal).unwrap();
    assert!(matches!(
        network.to_mixed_mode_equal_pair_power(1),
        Err(Error::UnequalMixedModePairReferences {
            direction: MixedModeDirection::ToMixedMode,
            frequency: 0,
            pair: 0,
            ..
        })
    ));

    let scaling_loss = Network::new(
        valid.frequency().clone(),
        valid.s().clone(),
        Array2::from_shape_vec(
            (1, 3),
            vec![c(f64::MAX, 0.0), c(f64::MAX, 0.0), c(70.0, 0.0)],
        )
        .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        scaling_loss.to_mixed_mode_equal_pair_power(1),
        Err(Error::MixedModeReferenceScalingLoss {
            direction: MixedModeDirection::ToMixedMode,
            frequency: 0,
            pair: 0,
            mode: MixedModeMode::Differential,
            scaling: MixedModeScaling::Double,
            ..
        })
    ));

    let half_loss = Network::new(
        valid.frequency().clone(),
        valid.s().clone(),
        Array2::from_shape_vec(
            (1, 3),
            vec![
                c(f64::from_bits(1), 0.0),
                c(f64::from_bits(1), 0.0),
                c(70.0, 0.0),
            ],
        )
        .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        half_loss.to_mixed_mode_equal_pair_power(1),
        Err(Error::MixedModeReferenceScalingLoss {
            mode: MixedModeMode::Common,
            scaling: MixedModeScaling::Half,
            ..
        })
    ));

    let arithmetic_overflow = Network::new(
        valid.frequency().clone(),
        Array3::from_elem((1, 2, 2), c(f64::MAX, f64::MAX)),
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    )
    .unwrap();
    assert!(matches!(
        arithmetic_overflow.to_mixed_mode_equal_pair_power(1),
        Err(Error::NonFiniteMixedModeComputation {
            direction: MixedModeDirection::ToMixedMode,
            frequency: 0,
            ..
        })
    ));
}

#[test]
fn inverse_rejects_incompatible_modal_references_and_preserves_input() {
    let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let s = Array3::from_shape_fn((1, 3, 3), |(_, row, column)| c((row + column) as f64, 0.0));
    let z0 =
        Array2::from_shape_vec((1, 3), vec![c(100.0, 0.0), c(30.0, 0.0), c(70.0, 0.0)]).unwrap();
    let modal = Network::new(frequency, s, z0).unwrap();
    let before = modal.clone();
    assert_eq!(
        modal.to_single_ended_equal_pair_power(1).unwrap_err(),
        Error::MixedModeInverseReferenceMismatch {
            direction: MixedModeDirection::ToSingleEnded,
            frequency: 0,
            pair: 0,
            differential: c(100.0, 0.0),
            common: c(30.0, 0.0),
            from_differential: c(50.0, 0.0),
            from_common: c(60.0, 0.0),
        }
    );
    assert_eq!(modal.s(), before.s());
    assert_eq!(modal.z0(), before.z0());
    assert_eq!(modal.frequency(), before.frequency());
}

#[test]
fn public_enums_remain_usable_for_structured_directional_diagnostics() {
    assert_eq!(
        MixedModeDirection::ToMixedMode.to_string(),
        "single-ended to mixed-mode"
    );
    assert_eq!(
        MixedModeDirection::ToSingleEnded.to_string(),
        "mixed-mode to single-ended"
    );
    assert_eq!(format!("{:?}", MixedModeMode::Differential), "Differential");
    assert_eq!(format!("{:?}", MixedModeScaling::Half), "Half");
}
