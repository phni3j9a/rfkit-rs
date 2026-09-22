use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, InverseCascadeStage, Network};
use serde_json::json;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn assert_close(actual: Complex64, expected: Complex64, tolerance: f64) {
    let difference = (actual - expected).norm();
    assert!(
        difference <= tolerance,
        "actual={actual:?}, expected={expected:?}, difference={difference:e}, tolerance={tolerance:e}"
    );
}

fn assert_array_close(actual: &Array3<Complex64>, expected: &Array3<Complex64>, tolerance: f64) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_close(actual, expected, tolerance);
        assert!(
            actual.re.is_finite() && actual.im.is_finite(),
            "index {index}"
        );
    }
}

fn two_port_source() -> Network {
    let frequency = Frequency::from_hz(vec![-0.0, -2.0e9, 1.0e9]).unwrap();
    let s = Array3::from_shape_fn((3, 2, 2), |(frequency, row, column)| {
        let scale = 1.0 + 0.17 * frequency as f64;
        match (row, column) {
            (0, 0) => c(0.12 * scale, -0.07),
            (0, 1) => c(0.35 * scale, 0.11),
            (1, 0) => c(-0.21 * scale, 0.19),
            (1, 1) => c(0.28 * scale, -0.13),
            _ => unreachable!(),
        }
    });
    let z0 = Array2::from_shape_vec(
        (3, 2),
        vec![
            c(42.0, 7.0),
            c(-68.0, -5.0),
            c(44.5, 8.0),
            c(-71.0, -4.5),
            c(47.0, 9.0),
            c(-74.0, -4.0),
        ],
    )
    .unwrap();
    Network::new(frequency, s, z0).unwrap()
}

fn direct_two_by_two_inverse(s: &Array3<Complex64>, frequency: usize) -> Array3<Complex64> {
    let a = s[[frequency, 0, 0]];
    let b = s[[frequency, 0, 1]];
    let c_value = s[[frequency, 1, 0]];
    let d = s[[frequency, 1, 1]];
    let determinant = a * d - b * c_value;
    let mut result = Array3::zeros((1, 2, 2));
    // P S^-1 P, with P exchanging the two physical groups.
    result[[0, 0, 0]] = a / determinant;
    result[[0, 0, 1]] = -c_value / determinant;
    result[[0, 1, 0]] = -b / determinant;
    result[[0, 1, 1]] = d / determinant;
    result
}

fn matrix_vector(
    s: &Array3<Complex64>,
    frequency: usize,
    input: &Array1<Complex64>,
) -> Array1<Complex64> {
    Array1::from_shape_fn(input.len(), |row| {
        (0..input.len())
            .map(|column| s[[frequency, row, column]] * input[column])
            .sum()
    })
}

fn waves(
    voltage: &Array1<Complex64>,
    current: &Array1<Complex64>,
    z0: &Array1<Complex64>,
) -> (Array1<Complex64>, Array1<Complex64>) {
    let a = Array1::from_shape_fn(voltage.len(), |port| {
        let scale = 1.0 / (2.0 * z0[port].re.abs().sqrt());
        (voltage[port] + z0[port] * current[port]) * scale
    });
    let b = Array1::from_shape_fn(voltage.len(), |port| {
        let scale = 1.0 / (2.0 * z0[port].re.abs().sqrt());
        (voltage[port] - z0[port].conj() * current[port]) * scale
    });
    (a, b)
}

#[test]
fn inverse_matches_analytic_two_port_and_preserves_exact_metadata() {
    let source = two_port_source();
    let source_before = source.clone();
    let inverse = source.inverse_cascade_power().unwrap();

    assert_eq!(inverse.frequency(), source.frequency());
    assert_eq!(inverse.frequency().hz()[0].to_bits(), (-0.0f64).to_bits());
    for frequency in 0..3 {
        let expected = direct_two_by_two_inverse(source.s(), frequency);
        for row in 0..2 {
            for column in 0..2 {
                assert_close(
                    inverse.s()[[frequency, row, column]],
                    expected[[0, row, column]],
                    2.0e-14,
                );
            }
        }
        assert_eq!(
            inverse.z0()[[frequency, 0]],
            source.z0()[[frequency, 1]].conj()
        );
        assert_eq!(
            inverse.z0()[[frequency, 1]],
            source.z0()[[frequency, 0]].conj()
        );
    }
    assert_eq!(source, source_before);

    let round_trip = inverse.inverse_cascade_power().unwrap();
    assert_array_close(round_trip.s(), source.s(), 2.0e-13);
    assert_eq!(round_trip.z0(), source.z0());
    assert_eq!(round_trip.frequency(), source.frequency());
}

#[test]
fn inverse_reversal_reconstructs_complex_reference_power_waves() {
    let source = two_port_source();
    let inverse = source.inverse_cascade_power().unwrap();
    let old_a = Array1::from_vec(vec![c(0.43, -0.27), c(-0.21, 0.62)]);

    for frequency in 0..source.frequency().len() {
        let old_z0 = source.z0().slice(ndarray::s![frequency, ..]).to_owned();
        let old_b = matrix_vector(source.s(), frequency, &old_a);
        let old_current = Array1::from_shape_fn(2, |port| {
            let q = old_z0[port].re.abs().sqrt() / old_z0[port].re;
            q * (old_a[port] - old_b[port])
        });
        let old_voltage = Array1::from_shape_fn(2, |port| {
            let scale = 2.0 * old_z0[port].re.abs().sqrt();
            scale * old_a[port] - old_z0[port] * old_current[port]
        });
        let (a_check, b_check) = waves(&old_voltage, &old_current, &old_z0);
        for port in 0..2 {
            assert_close(a_check[port], old_a[port], 2.0e-14);
            assert_close(b_check[port], old_b[port], 2.0e-14);
        }

        let new_voltage = Array1::from_vec(vec![old_voltage[1], old_voltage[0]]);
        let new_current = Array1::from_vec(vec![-old_current[1], -old_current[0]]);
        let new_z0 = inverse.z0().slice(ndarray::s![frequency, ..]).to_owned();
        let (new_a, new_b) = waves(&new_voltage, &new_current, &new_z0);
        assert_close(new_a[0], old_b[1], 2.0e-14);
        assert_close(new_a[1], old_b[0], 2.0e-14);
        assert_close(new_b[0], old_a[1], 2.0e-14);
        assert_close(new_b[1], old_a[0], 2.0e-14);

        let reconstructed = matrix_vector(inverse.s(), frequency, &new_a);
        for port in 0..2 {
            assert_close(reconstructed[port], new_b[port], 2.0e-13);
        }
    }
}

#[test]
fn inverse_of_ideal_through_preserves_group_exchange_and_reverses_references() {
    let source = simple_source(
        Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        )
        .unwrap(),
        Array2::from_shape_vec((1, 2), vec![c(42.0, 7.0), c(-68.0, -5.0)]).unwrap(),
    );
    let inverse = source.inverse_cascade_power().unwrap();
    assert_array_close(inverse.s(), source.s(), 0.0);
    assert_eq!(inverse.z0()[[0, 0]], source.z0()[[0, 1]].conj());
    assert_eq!(inverse.z0()[[0, 1]], source.z0()[[0, 0]].conj());
    let round_trip = inverse.inverse_cascade_power().unwrap();
    assert_array_close(round_trip.s(), source.s(), 0.0);
    assert_eq!(round_trip.z0(), source.z0());
}

#[test]
fn inverse_supports_coupled_four_port_group_order() {
    let frequency = Frequency::from_hz(vec![1.0e9, 2.0e9]).unwrap();
    let s = Array3::from_shape_fn((2, 4, 4), |(f, row, column)| {
        let mode = row % 2;
        let group = row / 2;
        let other_group = column / 2;
        let other_mode = column % 2;
        let same_mode = match (group, other_group) {
            (0, 0) => c(0.09 + 0.01 * f as f64, -0.02),
            (0, 1) => c(0.31 + 0.01 * mode as f64, 0.08),
            (1, 0) => c(-0.22 - 0.01 * mode as f64, 0.13),
            (1, 1) => c(0.27 + 0.01 * f as f64, -0.11),
            _ => unreachable!(),
        };
        if mode != other_mode {
            // Keep both transmission blocks dense and asymmetric.  These
            // small terms are deliberate cross-mode coupling, not a
            // block-diagonal modal fixture hidden behind a 4-port shape.
            c(
                0.013 + 0.002 * row as f64 - 0.001 * column as f64,
                0.004 + 0.001 * f as f64,
            )
        } else {
            same_mode
        }
    });
    let z0 = Array2::from_shape_fn((2, 4), |(f, port)| {
        c(
            [41.0, -52.0, 67.0, -79.0][port] + 0.5 * f as f64,
            [3.0, -4.0, 5.0, -6.0][port],
        )
    });
    let source = Network::new(frequency, s, z0).unwrap();
    let inverse = source.inverse_cascade_power().unwrap();
    assert_eq!(inverse.nports(), 4);
    for f in 0..2 {
        // Applying the returned matrix to `P*b` must produce `P*a`, where
        // `b = S*a`; this checks the full 4-port group orientation rather than
        // merely checking that the output is finite.
        let source_a =
            Array1::from_vec(vec![c(0.2, -0.1), c(-0.4, 0.3), c(0.5, 0.2), c(-0.1, 0.6)]);
        let source_b = matrix_vector(source.s(), f, &source_a);
        let reversed_a = Array1::from_vec(vec![source_b[2], source_b[3], source_b[0], source_b[1]]);
        let result = matrix_vector(inverse.s(), f, &reversed_a);
        let expected = Array1::from_vec(vec![source_a[2], source_a[3], source_a[0], source_a[1]]);
        for row in 0..4 {
            assert_close(result[row], expected[row], 2.0e-13);
        }
    }
}

#[test]
fn inverse_supports_a_larger_even_port_case_and_double_round_trip() {
    const GROUP: usize = 3;
    const NPORTS: usize = 2 * GROUP;
    let frequency = Frequency::from_hz(vec![0.8e9, 1.6e9]).unwrap();
    let forward = [
        [0.31, 0.024, 0.011],
        [0.017, 0.28, 0.021],
        [0.009, 0.019, 0.34],
    ];
    let reverse = [
        [0.23, 0.031, 0.014],
        [0.018, 0.26, 0.022],
        [0.012, 0.027, 0.25],
    ];
    let s = Array3::from_shape_fn((2, NPORTS, NPORTS), |(f, row, column)| {
        let group = row / GROUP;
        let other_group = column / GROUP;
        let local_row = row % GROUP;
        let local_column = column % GROUP;
        let frequency_scale = 1.0 + 0.02 * f as f64;
        match (group, other_group) {
            (0, 0) => {
                if local_row == local_column {
                    c(0.05 * frequency_scale, -0.012)
                } else {
                    c(0.009 + 0.002 * local_row as f64, 0.004)
                }
            }
            (0, 1) => c(
                forward[local_row][local_column] * frequency_scale,
                0.01 * (local_row + 1) as f64,
            ),
            (1, 0) => c(
                reverse[local_row][local_column] * frequency_scale,
                -0.008 * (local_column + 1) as f64,
            ),
            (1, 1) => {
                if local_row == local_column {
                    c(-0.04 * frequency_scale, 0.015)
                } else {
                    c(-0.007 - 0.001 * local_column as f64, 0.003)
                }
            }
            _ => unreachable!(),
        }
    });
    let z0 = Array2::from_shape_fn((2, NPORTS), |(f, port)| {
        c(
            [38.0, 47.0, 56.0, -63.0, -72.0, -81.0][port] + f as f64,
            [2.0, -3.0, 4.0, -5.0, 6.0, -7.0][port],
        )
    });
    let source = Network::new(frequency, s, z0).unwrap();
    let inverse = source.inverse_cascade_power().unwrap();
    assert_eq!(inverse.nports(), NPORTS);
    let round_trip = inverse.inverse_cascade_power().unwrap();
    assert_array_close(round_trip.s(), source.s(), 2.0e-12);
    assert_eq!(round_trip.z0(), source.z0());
    assert_eq!(round_trip.frequency(), source.frequency());
}

// Keep this bookkeeping local to the invariant test. A multi-mode cascade
// connects each ordered right-group/left-group pair, then closes the
// remaining pairs inside the aggregate with the existing public inner direct
// connection. No product-level N-port cascade API is introduced here.
fn connect_groupwise(
    first: &Network,
    second: &Network,
    first_group: &[usize],
    second_group: &[usize],
) -> rfkit_core::Result<Network> {
    assert_eq!(first_group.len(), second_group.len());
    assert!(!first_group.is_empty());
    let mut labels: Vec<(bool, usize)> = (0..first.nports()).map(|port| (true, port)).collect();
    labels.extend((0..second.nports()).map(|port| (false, port)));
    let mut output = first.connect_direct_power(first_group[0], second, second_group[0])?;
    labels.retain(|&(side, port)| {
        !(side && port == first_group[0] || !side && port == second_group[0])
    });
    for (&first_port, &second_port) in first_group[1..].iter().zip(&second_group[1..]) {
        let first_index = labels
            .iter()
            .position(|&(side, port)| side && port == first_port)
            .expect("first group port survives until its pair is connected");
        let second_index = labels
            .iter()
            .position(|&(side, port)| !side && port == second_port)
            .expect("second group port survives until its pair is connected");
        output = output.inner_connect_direct_power(first_index, second_index)?;
        labels
            .retain(|&(side, port)| !(side && port == first_port || !side && port == second_port));
    }
    Ok(output)
}

fn four_port_fixture(offset: f64) -> Network {
    let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let s = Array3::from_shape_fn((1, 4, 4), |(_, row, column)| {
        let left = row < 2;
        let other_left = column < 2;
        if left == other_left {
            if row == column {
                c(0.045 + offset, -0.012)
            } else {
                c(0.011 + 0.002 * row as f64, 0.004 - 0.001 * column as f64)
            }
        } else if left {
            // Reverse transmission S[left,right].
            c(
                0.22 + 0.017 * (row + column) as f64 + offset,
                0.018 * (row + 1) as f64,
            )
        } else {
            // Forward transmission S[right,left].
            c(
                0.28 + 0.013 * (row + column) as f64 + offset,
                -0.014 * (column + 1) as f64,
            )
        }
    });
    Network::new(frequency, s, Array2::from_elem((1, 4), c(50.0, 0.0))).unwrap()
}

fn cascade_four_port(
    left: &Network,
    dut: &Network,
    right: &Network,
) -> rfkit_core::Result<Network> {
    let left_dut = connect_groupwise(left, dut, &[2, 3], &[0, 1])?;
    connect_groupwise(&left_dut, right, &[2, 3], &[0, 1])
}

fn remove_left_then_right_four_port(
    measured: &Network,
    left: &Network,
    right: &Network,
) -> rfkit_core::Result<Network> {
    let left_inverse = left.inverse_cascade_power()?;
    let right_inverse = right.inverse_cascade_power()?;
    let without_left = connect_groupwise(&left_inverse, measured, &[2, 3], &[0, 1])?;
    connect_groupwise(&without_left, &right_inverse, &[2, 3], &[0, 1])
}

fn remove_right_then_left_four_port(
    measured: &Network,
    left: &Network,
    right: &Network,
) -> rfkit_core::Result<Network> {
    let left_inverse = left.inverse_cascade_power()?;
    let right_inverse = right.inverse_cascade_power()?;
    let without_right = connect_groupwise(measured, &right_inverse, &[2, 3], &[0, 1])?;
    connect_groupwise(&left_inverse, &without_right, &[2, 3], &[0, 1])
}

#[test]
fn inverse_cancels_both_orders_of_a_coupled_four_port_cascade() {
    let left = four_port_fixture(0.0);
    let dut = four_port_fixture(0.004);
    let right = four_port_fixture(-0.003);
    let measured = cascade_four_port(&left, &dut, &right).unwrap();
    let recovered_left_right = remove_left_then_right_four_port(&measured, &left, &right).unwrap();
    let recovered_right_left = remove_right_then_left_four_port(&measured, &left, &right).unwrap();
    assert_eq!(recovered_left_right.frequency(), dut.frequency());
    assert_eq!(recovered_left_right.z0(), dut.z0());
    assert_eq!(recovered_right_left.z0(), dut.z0());
    assert_array_close(recovered_left_right.s(), dut.s(), 2.0e-9);
    assert_array_close(recovered_right_left.s(), dut.s(), 2.0e-9);
}

fn simple_source(s: Array3<Complex64>, z0: Array2<Complex64>) -> Network {
    Network::new(Frequency::from_hz(vec![1.0e9]).unwrap(), s, z0).unwrap()
}

fn two_by_two(z0: Complex64) -> Array2<Complex64> {
    Array2::from_elem((1, 2), z0)
}

#[test]
fn inverse_distinguishes_full_and_directional_singularities() {
    let z0 = two_by_two(c(50.0, 0.0));
    let full_singular = simple_source(
        Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(1.0, 0.0), c(0.25, 0.0), c(0.5, 0.0), c(0.125, 0.0)],
        )
        .unwrap(),
        z0.clone(),
    );
    assert!(matches!(
        full_singular.inverse_cascade_power(),
        Err(Error::SingularInverseCascade {
            stage: InverseCascadeStage::FullS,
            frequency: 0,
            ..
        })
    ));

    let forward_singular = simple_source(
        Array3::from_shape_vec(
            (1, 4, 4),
            vec![
                c(0.1, 0.0),
                c(0.0, 0.0),
                c(0.2, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.2, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.3, 0.0),
                c(0.0, 0.0),
                c(0.4, 0.0),
                c(0.0, 0.0),
                c(0.5, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.3, 0.0),
            ],
        )
        .unwrap(),
        Array2::from_elem((1, 4), c(50.0, 0.0)),
    );
    assert!(matches!(
        forward_singular.inverse_cascade_power(),
        Err(Error::SingularInverseCascade {
            stage: InverseCascadeStage::ForwardTransmission,
            frequency: 0,
            ..
        })
    ));

    let reverse_singular = simple_source(
        Array3::from_shape_vec(
            (1, 4, 4),
            vec![
                c(1.0, 0.0),
                c(0.0, 0.0),
                c(0.2, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(1.0, 0.0),
                c(0.4, 0.0),
                c(0.0, 0.0),
                c(0.3, 0.0),
                c(0.0, 0.0),
                c(1.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.5, 0.0),
                c(0.0, 0.0),
                c(1.0, 0.0),
            ],
        )
        .unwrap(),
        Array2::from_elem((1, 4), c(50.0, 0.0)),
    );
    assert!(matches!(
        reverse_singular.inverse_cascade_power(),
        Err(Error::SingularInverseCascade {
            stage: InverseCascadeStage::ReverseTransmission,
            frequency: 0,
            ..
        })
    ));
}

#[test]
fn inverse_accepts_finite_near_singular_transmission_blocks() {
    let source = simple_source(
        Array3::from_shape_vec(
            (1, 2, 2),
            vec![
                c(0.1, 0.0),
                c(1.0e-200, 0.0),
                c(-2.0e-200, 0.0),
                c(0.2, 0.0),
            ],
        )
        .unwrap(),
        two_by_two(c(50.0, 1.0)),
    );
    let inverse = source.inverse_cascade_power().unwrap();
    assert!(
        inverse
            .s()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn inverse_reports_input_errors_and_never_panics_on_malformed_shapes() {
    let valid = two_port_source();
    let mut malformed_empty = serde_json::to_value(&valid).unwrap();
    malformed_empty["frequency"]["hz"] = json!([]);
    malformed_empty["s"]["dim"] = json!([0, 2, 2]);
    malformed_empty["s"]["data"] = json!([]);
    malformed_empty["z0"]["dim"] = json!([0, 2]);
    malformed_empty["z0"]["data"] = json!([]);
    let malformed_empty: Network = serde_json::from_value(malformed_empty).unwrap();
    assert_eq!(
        malformed_empty.inverse_cascade_power().unwrap_err(),
        Error::EmptyInverseCascadeFrequency
    );

    let mut zero_port = serde_json::to_value(&valid).unwrap();
    zero_port["frequency"]["hz"] = json!([1.0e9]);
    zero_port["s"]["dim"] = json!([1, 0, 0]);
    zero_port["s"]["data"] = json!([]);
    zero_port["z0"]["dim"] = json!([1, 0]);
    zero_port["z0"]["data"] = json!([]);
    let zero_port: Network = serde_json::from_value(zero_port).unwrap();
    assert_eq!(
        zero_port.inverse_cascade_power().unwrap_err(),
        Error::InvalidInverseCascadeSShape {
            shape: vec![1, 0, 0]
        }
    );

    let mut frequency_mismatch = serde_json::to_value(&valid).unwrap();
    frequency_mismatch["frequency"]["hz"] = json!([1.0e9]);
    let frequency_mismatch: Network = serde_json::from_value(frequency_mismatch).unwrap();
    assert_eq!(
        frequency_mismatch.inverse_cascade_power().unwrap_err(),
        Error::InverseCascadeFrequencyLengthMismatch {
            expected: 1,
            actual: 3,
        }
    );

    let mut malformed_z0 = serde_json::to_value(&valid).unwrap();
    let z0_scalar = malformed_z0["z0"]["data"][0].clone();
    malformed_z0["z0"]["dim"] = json!([3, 1]);
    malformed_z0["z0"]["data"] = json!([z0_scalar.clone(), z0_scalar.clone(), z0_scalar]);
    let malformed_z0: Network = serde_json::from_value(malformed_z0).unwrap();
    assert_eq!(
        malformed_z0.inverse_cascade_power().unwrap_err(),
        Error::InvalidInverseCascadeZ0Shape { shape: vec![3, 1] }
    );
    let odd = Network::new(
        Frequency::from_hz(vec![1.0e9]).unwrap(),
        Array3::zeros((1, 3, 3)),
        Array2::from_elem((1, 3), c(50.0, 0.0)),
    )
    .unwrap();
    assert_eq!(
        odd.inverse_cascade_power().unwrap_err(),
        Error::InvalidInverseCascadePortCount { nports: 3 }
    );
    let nonfinite_frequency = Network::new(
        Frequency::from_hz(vec![f64::NAN]).unwrap(),
        Array3::zeros((1, 2, 2)),
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    )
    .unwrap();
    assert!(matches!(
        nonfinite_frequency.inverse_cascade_power(),
        Err(Error::NonFiniteInverseCascadeFrequency { index: 0, .. })
    ));
    let nonfinite_s = simple_source(
        Array3::from_elem((1, 2, 2), c(f64::NAN, 0.0)),
        two_by_two(c(50.0, 0.0)),
    );
    assert!(matches!(
        nonfinite_s.inverse_cascade_power(),
        Err(Error::NonFiniteInverseCascadeS {
            stage: InverseCascadeStage::FullS,
            frequency: 0,
            ..
        })
    ));
    let nonfinite_z0 = simple_source(
        Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.1, 0.0), c(0.2, 0.0), c(0.3, 0.0), c(0.4, 0.0)],
        )
        .unwrap(),
        Array2::from_shape_vec((1, 2), vec![c(f64::NAN, 0.0), c(50.0, 0.0)]).unwrap(),
    );
    assert!(matches!(
        nonfinite_z0.inverse_cascade_power(),
        Err(Error::NonFiniteInverseCascadeZ0 {
            frequency: 0,
            port: 0,
        })
    ));
    let zero_real = simple_source(
        Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.1, 0.0), c(0.2, 0.0), c(0.3, 0.0), c(0.4, 0.0)],
        )
        .unwrap(),
        two_by_two(c(0.0, 50.0)),
    );
    assert_eq!(
        zero_real.inverse_cascade_power().unwrap_err(),
        Error::ZeroRealInverseCascadeReferenceImpedance {
            frequency: 0,
            port: 0,
        }
    );

    let mut malformed = serde_json::to_value(&valid).unwrap();
    malformed["frequency"]["hz"] = json!([]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| malformed.inverse_cascade_power()));
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().unwrap_err(),
        Error::EmptyInverseCascadeFrequency
    );

    let mut malformed = serde_json::to_value(&valid).unwrap();
    let scalar = malformed["s"]["data"][0].clone();
    let z0_scalar = malformed["z0"]["data"][0].clone();
    malformed["frequency"]["hz"] = json!([1.0e9]);
    malformed["s"]["dim"] = json!([1, 1, 2]);
    malformed["s"]["data"] = json!([scalar.clone(), scalar]);
    malformed["z0"]["dim"] = json!([1, 2]);
    malformed["z0"]["data"] = json!([z0_scalar.clone(), z0_scalar]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    assert_eq!(
        malformed.inverse_cascade_power().unwrap_err(),
        Error::InvalidInverseCascadeSShape {
            shape: vec![1, 1, 2]
        }
    );

    let arithmetic_overflow = simple_source(
        Array3::from_shape_vec(
            (1, 2, 2),
            vec![
                c(f64::MAX, 0.0),
                c(f64::MAX, 0.0),
                c(f64::MAX, 0.0),
                c(-f64::MAX, 0.0),
            ],
        )
        .unwrap(),
        two_by_two(c(50.0, 0.0)),
    );
    assert!(matches!(
        arithmetic_overflow.inverse_cascade_power(),
        Err(Error::NonFiniteInverseCascadeComputation {
            stage: InverseCascadeStage::FullS,
            frequency: 0,
            ..
        })
    ));
}
