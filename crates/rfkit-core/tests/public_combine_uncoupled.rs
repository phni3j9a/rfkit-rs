use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{ConnectionInput, Error, Frequency, Network};
use serde_json::{Value, json};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn network(frequency: &[f64], s: Array3<Complex64>, z0: Array2<Complex64>) -> Network {
    Network::new(Frequency::from_hz(frequency.to_vec()).unwrap(), s, z0).unwrap()
}

fn asymmetric_network(frequency: &[f64], nports: usize, offset: f64) -> Network {
    let s = Array3::from_shape_fn((frequency.len(), nports, nports), |(f, row, column)| {
        c(
            offset + 0.013 * f as f64 + 0.007 * row as f64 + 0.003 * column as f64,
            -offset * 0.4 - 0.011 * f as f64 + 0.005 * row as f64 - 0.002 * column as f64,
        )
    });
    let z0 = Array2::from_shape_fn((frequency.len(), nports), |(f, port)| {
        c(
            31.0 + offset + 3.5 * f as f64 + 2.25 * port as f64,
            -4.0 + 0.75 * f as f64 - 0.4 * port as f64,
        )
    });
    network(frequency, s, z0)
}

fn assert_complex_bits(actual: Complex64, expected: Complex64) {
    assert_eq!(actual.re.to_bits(), expected.re.to_bits());
    assert_eq!(actual.im.to_bits(), expected.im.to_bits());
}

fn assert_network_bits_equal(actual: &Network, expected: &Network) {
    assert_eq!(actual.frequency().len(), expected.frequency().len());
    for (&actual, &expected) in actual
        .frequency()
        .hz()
        .iter()
        .zip(expected.frequency().hz())
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(actual.s().dim(), expected.s().dim());
    for (&actual, &expected) in actual.s().iter().zip(expected.s()) {
        assert_complex_bits(actual, expected);
    }
    assert_eq!(actual.z0().dim(), expected.z0().dim());
    for (&actual, &expected) in actual.z0().iter().zip(expected.z0()) {
        assert_complex_bits(actual, expected);
    }
}

fn assert_s_close(actual: &Array3<Complex64>, expected: &Array3<Complex64>) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual - expected).norm();
        assert!(
            difference <= 2.0e-12,
            "index {index}: actual={actual:?}, expected={expected:?}, difference={difference:e}"
        );
    }
}

fn valid_serialized_network(nfreq: usize, nports: usize) -> Value {
    serde_json::to_value(asymmetric_network(
        &(0..nfreq)
            .map(|index| 1.0e9 + index as f64)
            .collect::<Vec<_>>(),
        nports,
        0.2,
    ))
    .unwrap()
}

fn deserialize_network(value: Value) -> Network {
    serde_json::from_value(value).expect("malformed shape fixture should deserialize")
}

#[test]
fn combines_asymmetric_networks_with_exact_blocks_zeros_and_opaque_scalars() {
    let frequency = vec![3.0e9, -0.0, 3.0e9];
    let mut s_a = Array3::from_shape_fn((3, 2, 2), |(f, row, column)| {
        c(
            0.1 + f as f64 + row as f64 * 0.01,
            -0.2 - column as f64 * 0.02,
        )
    });
    let nan_s = f64::from_bits(0x7ff8_1234_5678_9abc);
    s_a[[1, 0, 1]] = c(-0.0, nan_s);
    let mut z0_a = Array2::from_shape_fn((3, 2), |(f, port)| {
        c(37.0 + 2.0 * f as f64 + port as f64, -(port as f64))
    });
    z0_a[[2, 1]] = c(nan_s, -0.0);
    let a = network(&frequency, s_a.clone(), z0_a.clone());

    let mut s_b = Array3::from_shape_fn((3, 3, 3), |(f, row, column)| {
        c(
            -0.3 - 0.1 * f as f64 + row as f64 * 0.02,
            0.4 + column as f64 * 0.03,
        )
    });
    s_b[[0, 2, 1]] = c(nan_s, -0.0);
    let z0_b = Array2::from_shape_fn((3, 3), |(f, port)| {
        c(71.0 + 3.0 * f as f64 + 4.0 * port as f64, 1.5 - port as f64)
    });
    let b = network(&frequency, s_b.clone(), z0_b.clone());
    let a_before = a.clone();
    let b_before = b.clone();

    let combined = a.combine_uncoupled(&b).unwrap();
    assert_eq!(combined.frequency().len(), 3);
    for (&actual, &expected) in combined.frequency().hz().iter().zip(&frequency) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(combined.s().dim(), (3, 5, 5));
    assert_eq!(combined.z0().dim(), (3, 5));
    for f in 0..3 {
        for row in 0..5 {
            for column in 0..5 {
                let expected = if row < 2 && column < 2 {
                    s_a[[f, row, column]]
                } else if row >= 2 && column >= 2 {
                    s_b[[f, row - 2, column - 2]]
                } else {
                    c(0.0, 0.0)
                };
                assert_complex_bits(combined.s()[[f, row, column]], expected);
            }
        }
        for port in 0..5 {
            let expected = if port < 2 {
                z0_a[[f, port]]
            } else {
                z0_b[[f, port - 2]]
            };
            assert_complex_bits(combined.z0()[[f, port]], expected);
        }
    }
    assert!(combined.s().is_standard_layout());
    assert!(combined.z0().is_standard_layout());
    assert_network_bits_equal(&a, &a_before);
    assert_network_bits_equal(&b, &b_before);
    assert_ne!(combined.s().as_ptr(), a.s().as_ptr());
    assert_ne!(combined.z0().as_ptr(), b.z0().as_ptr());
}

#[test]
fn combines_one_port_inputs_and_recovers_each_source_block_by_selection() {
    let frequency = [1.0e9, 2.0e9];
    let a = asymmetric_network(&frequency, 1, 0.5);
    let b = asymmetric_network(&frequency, 1, -0.7);
    let combined = a.combine_uncoupled(&b).unwrap();
    assert_eq!(combined.nports(), 2);
    assert_network_bits_equal(&combined.select_ports_zero_incident(&[0]).unwrap(), &a);
    assert_network_bits_equal(&combined.select_ports_zero_incident(&[1]).unwrap(), &b);
}

#[test]
fn accepts_nonstandard_owned_layouts_and_keeps_result_standard() {
    let frequency = [1.0e9, 2.0e9];
    let a_standard = asymmetric_network(&frequency, 2, 0.3);
    let b_standard = asymmetric_network(&frequency, 3, -0.4);
    let a = network(
        &frequency,
        a_standard.s().clone().permuted_axes([0, 2, 1]),
        a_standard.z0().clone(),
    );
    let b_z0 = Array2::from_shape_fn((3, 2), |(port, frequency)| {
        b_standard.z0()[[frequency, port]]
    })
    .reversed_axes();
    let b = network(&frequency, b_standard.s().clone(), b_z0);
    assert!(!a.s().is_standard_layout());
    assert!(!b.z0().is_standard_layout());

    let combined = a.combine_uncoupled(&b).unwrap();
    assert!(combined.s().is_standard_layout());
    assert!(combined.z0().is_standard_layout());
    for f in 0..frequency.len() {
        for row in 0..a.nports() {
            for column in 0..a.nports() {
                assert_complex_bits(combined.s()[[f, row, column]], a.s()[[f, row, column]]);
            }
        }
    }
}

#[test]
fn rejects_grid_mismatches_and_nonfinite_labels_with_context() {
    let valid = asymmetric_network(&[1.0e9, 2.0e9], 2, 0.1);
    let length_mismatch = asymmetric_network(&[1.0e9], 3, 0.2);
    assert_eq!(
        valid.combine_uncoupled(&length_mismatch).unwrap_err(),
        Error::CombineUncoupledFrequencyLengthMismatch { a: 2, b: 1 }
    );

    let point_mismatch = asymmetric_network(&[1.0e9, 2.0000000001e9], 3, 0.2);
    assert!(matches!(
        valid.combine_uncoupled(&point_mismatch),
        Err(Error::CombineUncoupledFrequencyMismatch { index: 1, .. })
    ));

    let signed_zero_a = asymmetric_network(&[-0.0], 1, 0.1);
    let signed_zero_b = asymmetric_network(&[0.0], 1, 0.2);
    let signed_zero_combined = signed_zero_a.combine_uncoupled(&signed_zero_b).unwrap();
    assert_eq!(
        signed_zero_combined.frequency().hz()[0].to_bits(),
        (-0.0f64).to_bits()
    );

    let nonfinite_a = asymmetric_network(&[f64::NAN, 2.0e9], 2, 0.1);
    assert!(matches!(
        nonfinite_a.combine_uncoupled(&valid),
        Err(Error::NonFiniteCombineUncoupledFrequency {
            input: ConnectionInput::A,
            index: 0,
            value
        }) if value.is_nan()
    ));
    let nonfinite_b = asymmetric_network(&[1.0e9, f64::INFINITY], 2, 0.1);
    assert!(matches!(
        valid.combine_uncoupled(&nonfinite_b),
        Err(Error::NonFiniteCombineUncoupledFrequency {
            input: ConnectionInput::B,
            index: 1,
            ..
        })
    ));
}

#[test]
fn rejects_malformed_serde_inputs_without_panicking_before_indexing() {
    let valid = asymmetric_network(&[1.0e9], 2, 0.1);

    let mut malformed_s_value = valid_serialized_network(1, 2);
    malformed_s_value["s"]["dim"] = json!([1, 2, 3]);
    let scalar = malformed_s_value["s"]["data"][0].clone();
    malformed_s_value["s"]["data"] = json!([
        scalar.clone(),
        scalar.clone(),
        scalar.clone(),
        scalar.clone(),
        scalar.clone(),
        scalar,
    ]);
    let malformed_s = deserialize_network(malformed_s_value);
    let result = catch_unwind(AssertUnwindSafe(|| malformed_s.combine_uncoupled(&valid)));
    assert!(result.is_ok());
    assert!(matches!(
        result.unwrap(),
        Err(Error::InvalidCombineUncoupledSShape {
            input: ConnectionInput::A,
            shape
        }) if shape == vec![1, 2, 3]
    ));

    let mut malformed_b_z0_value = valid_serialized_network(1, 2);
    malformed_b_z0_value["z0"]["dim"] = json!([1, 1]);
    let scalar = malformed_b_z0_value["z0"]["data"][0].clone();
    malformed_b_z0_value["z0"]["data"] = json!([scalar]);
    let malformed_b_z0 = deserialize_network(malformed_b_z0_value);
    assert_eq!(
        valid.combine_uncoupled(&malformed_b_z0).unwrap_err(),
        Error::InvalidCombineUncoupledZ0Shape {
            input: ConnectionInput::B,
            shape: vec![1, 1],
        }
    );

    let mut empty_frequency_value = serde_json::to_value(&valid).unwrap();
    empty_frequency_value["frequency"]["hz"] = json!([]);
    let empty_frequency = deserialize_network(empty_frequency_value);
    assert_eq!(
        empty_frequency.combine_uncoupled(&valid).unwrap_err(),
        Error::EmptyCombineUncoupledFrequency {
            input: ConnectionInput::A,
        }
    );

    let mut frequency_shape_value = serde_json::to_value(&valid).unwrap();
    frequency_shape_value["frequency"]["hz"] = json!([1.0e9, 2.0e9]);
    let frequency_shape = deserialize_network(frequency_shape_value);
    assert_eq!(
        frequency_shape.combine_uncoupled(&valid).unwrap_err(),
        Error::CombineUncoupledFrequencyShape {
            input: ConnectionInput::A,
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn composition_is_associative_and_swap_matches_explicit_permutation() {
    let frequency = [1.0e9, 1.0e9, -2.0e9];
    let a = asymmetric_network(&frequency, 1, 0.1);
    let b = asymmetric_network(&frequency, 2, 0.2);
    let c = asymmetric_network(&frequency, 3, 0.3);

    let left = a
        .combine_uncoupled(&b)
        .unwrap()
        .combine_uncoupled(&c)
        .unwrap();
    let right = a
        .combine_uncoupled(&b.combine_uncoupled(&c).unwrap())
        .unwrap();
    assert_network_bits_equal(&left, &right);

    let swapped = b.combine_uncoupled(&a).unwrap();
    let swapped_to_ab = swapped.select_ports_zero_incident(&[2, 0, 1]).unwrap();
    let ab = a.combine_uncoupled(&b).unwrap();
    assert_network_bits_equal(&swapped_to_ab, &ab);

    let a_block = ab.select_ports_zero_incident(&[0]).unwrap();
    let b_block = ab.select_ports_zero_incident(&[1, 2]).unwrap();
    assert_network_bits_equal(&a_block, &a);
    assert_network_bits_equal(&b_block, &b);
}

#[test]
fn direct_sum_then_inner_connect_agrees_with_connect_for_matched_and_unequal_complex_junctions() {
    let frequency = [1.0e9, 1.5e9];
    let mut a_s = Array3::zeros((2, 2, 2));
    a_s[[0, 0, 0]] = c(0.08, 0.01);
    a_s[[0, 0, 1]] = c(0.12, -0.02);
    a_s[[0, 1, 0]] = c(-0.04, 0.03);
    a_s[[0, 1, 1]] = c(0.15, 0.02);
    a_s[[1, 0, 0]] = c(0.06, -0.02);
    a_s[[1, 0, 1]] = c(0.10, 0.01);
    a_s[[1, 1, 0]] = c(-0.02, 0.02);
    a_s[[1, 1, 1]] = c(0.14, -0.01);
    let b_s = Array3::from_shape_fn((2, 2, 2), |(f, row, column)| {
        c(
            -0.03 + 0.02 * f as f64 + 0.04 * row as f64 + 0.01 * column as f64,
            0.01 - 0.015 * f as f64 + 0.02 * row as f64 - 0.01 * column as f64,
        )
    });

    for (a_junction, b_junction) in [(c(50.0, 0.0), c(50.0, 0.0)), (c(43.0, 7.0), c(81.0, -5.0))] {
        let a = network(
            &frequency,
            a_s.clone(),
            Array2::from_shape_fn((2, 2), |(f, port)| {
                if port == 1 {
                    a_junction + c(f as f64, 0.0)
                } else {
                    c(37.0 + f as f64, -1.0)
                }
            }),
        );
        let b = network(
            &frequency,
            b_s.clone(),
            Array2::from_shape_fn((2, 2), |(f, port)| {
                if port == 0 {
                    b_junction + c(f as f64, 0.0)
                } else {
                    c(77.0 + f as f64, 2.0)
                }
            }),
        );

        let direct = a.connect_power(1, &b, 0).unwrap();
        let assembled = a
            .combine_uncoupled(&b)
            .unwrap()
            .inner_connect_power(1, 2)
            .unwrap();
        assert_eq!(direct.frequency(), assembled.frequency());
        assert_eq!(direct.z0(), assembled.z0());
        assert_s_close(direct.s(), assembled.s());
    }
}
