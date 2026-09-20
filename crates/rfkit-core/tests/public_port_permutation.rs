use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, Network};
use serde_json::{Value, json};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn complex(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn asymmetric_network(nfreq: usize, nport: usize) -> Network {
    let frequency_hz = (0..nfreq)
        .map(|index| 1.25e9 + 0.375e9 * index as f64)
        .collect();
    let s = Array3::from_shape_fn((nfreq, nport, nport), |(frequency, row, column)| {
        let real =
            0.125 * (frequency + 1) as f64 + 0.017 * (row + 1) as f64 + 0.003 * (column + 1) as f64;
        let imag = -0.071 * (frequency + 1) as f64 + 0.011 * (row + 1) as f64
            - 0.019 * (column + 1) as f64;
        complex(real, imag)
    });
    let z0 = Array2::from_shape_fn((nfreq, nport), |(frequency, port)| {
        complex(
            31.0 + 2.5 * (frequency + 1) as f64 + 7.0 * (port + 1) as f64,
            -4.0 + 0.75 * (frequency + 1) as f64 - 1.25 * (port + 1) as f64,
        )
    });
    Network::new(Frequency::from_hz(frequency_hz).unwrap(), s, z0).unwrap()
}

fn assert_same_bits(left: Complex64, right: Complex64) {
    assert_eq!(left.re.to_bits(), right.re.to_bits());
    assert_eq!(left.im.to_bits(), right.im.to_bits());
}

fn assert_network_bits_equal(left: &Network, right: &Network) {
    assert_eq!(left.frequency().len(), right.frequency().len());
    for (&left_frequency, &right_frequency) in
        left.frequency().hz().iter().zip(right.frequency().hz())
    {
        assert_eq!(left_frequency.to_bits(), right_frequency.to_bits());
    }
    assert_eq!(left.s().dim(), right.s().dim());
    for (&left_value, &right_value) in left.s().iter().zip(right.s()) {
        assert_same_bits(left_value, right_value);
    }
    assert_eq!(left.z0().dim(), right.z0().dim());
    for (&left_value, &right_value) in left.z0().iter().zip(right.z0()) {
        assert_same_bits(left_value, right_value);
    }
}

fn assert_permuted_bits(input: &Network, output: &Network, order: &[usize]) {
    assert_eq!(output.nports(), order.len());
    assert_eq!(output.frequency().len(), input.frequency().len());
    for (frequency, (&input_frequency, &output_frequency)) in input
        .frequency()
        .hz()
        .iter()
        .zip(output.frequency().hz())
        .enumerate()
    {
        assert_eq!(input_frequency.to_bits(), output_frequency.to_bits());
        for row in 0..order.len() {
            for column in 0..order.len() {
                assert_same_bits(
                    output.s()[[frequency, row, column]],
                    input.s()[[frequency, order[row], order[column]]],
                );
            }
            assert_same_bits(
                output.z0()[[frequency, row]],
                input.z0()[[frequency, order[row]]],
            );
        }
    }
}

#[test]
fn permutes_one_two_and_many_port_networks_with_multiple_frequencies() {
    for (nport, order) in [(1, vec![0]), (2, vec![1, 0]), (4, vec![2, 0, 3, 1])] {
        let input = asymmetric_network(3, nport);
        let input_before = input.clone();
        let order_before = order.clone();

        let output = input.permute_ports(&order).unwrap();

        assert_permuted_bits(&input, &output, &order);
        assert_network_bits_equal(&input, &input_before);
        assert_eq!(order, order_before);
        assert_ne!(output.s().as_ptr(), input.s().as_ptr());
        assert_ne!(output.z0().as_ptr(), input.z0().as_ptr());
    }
}

#[test]
fn identity_inverse_and_composition_are_exact_for_non_involutive_orders() {
    let input = asymmetric_network(4, 3);
    let input_before = input.clone();
    let first_order = vec![2, 0, 1];
    let second_order = vec![1, 2, 0];

    let first = input.permute_ports(&first_order).unwrap();
    let inverse = first.permute_ports(&second_order).unwrap();
    assert_network_bits_equal(&inverse, &input);

    let other_order = vec![0, 2, 1];
    let sequential = first.permute_ports(&other_order).unwrap();
    let composed_order: Vec<_> = other_order
        .iter()
        .map(|&new_port| first_order[new_port])
        .collect();
    let composed = input.permute_ports(&composed_order).unwrap();
    assert_network_bits_equal(&sequential, &composed);

    let identity = input.permute_ports(&[0, 1, 2]).unwrap();
    assert_network_bits_equal(&identity, &input);
    assert_network_bits_equal(&input, &input_before);
}

#[test]
fn preserves_exact_bits_and_supports_nonstandard_owned_ndarray_layouts() {
    let nan_payload = f64::from_bits(0x7ff8_1234_5678_9abc);
    let frequency = Frequency::from_hz(vec![-0.0, nan_payload]).unwrap();
    let standard_s = Array3::from_shape_fn((2, 3, 3), |(frequency, row, column)| {
        let real = if frequency == 0 && row == 0 && column == 1 {
            -0.0
        } else if frequency == 1 && row == 2 && column == 0 {
            f64::from_bits(0x7ff8_2468_1357_9bdf)
        } else {
            (100 * frequency + 17 * row + 3 * column) as f64
        };
        let imag = if frequency == 0 && row == 1 && column == 2 {
            f64::from_bits(0x8000_0000_0000_0000)
        } else {
            -((frequency + row + column) as f64)
        };
        complex(real, imag)
    });
    let standard_z0 = Array2::from_shape_fn((2, 3), |(frequency, port)| {
        if frequency == 1 && port == 1 {
            complex(-0.0, nan_payload)
        } else {
            complex((40 + 7 * frequency + port) as f64, -(port as f64))
        }
    });
    let nonstandard_s = standard_s.clone().permuted_axes([0, 2, 1]);
    let nonstandard_z0 =
        Array2::from_shape_fn((3, 2), |(port, frequency)| standard_z0[[frequency, port]])
            .reversed_axes();
    assert!(!nonstandard_s.is_standard_layout());
    assert!(!nonstandard_z0.is_standard_layout());

    let input = Network::new(frequency, nonstandard_s, nonstandard_z0).unwrap();
    let input_before = input.clone();
    let order = vec![2, 0, 1];
    let output = input.permute_ports(&order).unwrap();

    assert_permuted_bits(&input, &output, &order);
    assert!(output.s().is_standard_layout());
    assert!(output.z0().is_standard_layout());
    assert_network_bits_equal(&input, &input_before);
}

#[test]
fn permuted_finite_waves_obey_the_same_scattering_relation() {
    let input = asymmetric_network(2, 3);
    let order = [2, 0, 1];
    let output = input.permute_ports(&order).unwrap();

    for frequency in 0..2 {
        let old_a = [complex(0.7, -0.2), complex(-0.3, 0.8), complex(0.45, 0.1)];
        let old_b: Vec<_> = (0..3)
            .map(|row| {
                (0..3)
                    .map(|column| input.s()[[frequency, row, column]] * old_a[column])
                    .sum::<Complex64>()
            })
            .collect();
        let new_a: Vec<_> = order.iter().map(|&old_port| old_a[old_port]).collect();
        let expected_new_b: Vec<_> = order.iter().map(|&old_port| old_b[old_port]).collect();

        for (new_row, expected_b) in expected_new_b.iter().enumerate() {
            let actual_new_b: Complex64 = (0..3)
                .map(|new_column| output.s()[[frequency, new_row, new_column]] * new_a[new_column])
                .sum();
            assert!((actual_new_b - *expected_b).norm() < 1.0e-14);
        }
    }
}

#[test]
fn rejects_invalid_permutations_with_structured_context() {
    let network = asymmetric_network(1, 3);

    assert_eq!(
        network.permute_ports(&[]).unwrap_err(),
        Error::PortPermutationLengthMismatch {
            expected: 3,
            actual: 0,
        }
    );
    assert_eq!(
        network.permute_ports(&[0, 1]).unwrap_err(),
        Error::PortPermutationLengthMismatch {
            expected: 3,
            actual: 2,
        }
    );
    assert_eq!(
        network.permute_ports(&[0, 0, 2]).unwrap_err(),
        Error::PortPermutationDuplicate {
            port: 0,
            first_position: 0,
            second_position: 1,
        }
    );
    assert_eq!(
        network.permute_ports(&[0, 3, 1]).unwrap_err(),
        Error::PortPermutationOutOfRange {
            position: 1,
            port: 3,
            nports: 3,
        }
    );
}

fn valid_serialized_network(nfreq: usize, nport: usize) -> Value {
    serde_json::to_value(asymmetric_network(nfreq, nport)).unwrap()
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

fn assert_no_panic(error: Error, network: Network, order: &[usize]) {
    let result = catch_unwind(AssertUnwindSafe(|| network.permute_ports(order)));
    assert!(
        result.is_ok(),
        "permute_ports panicked for malformed network"
    );
    assert_eq!(result.unwrap().unwrap_err(), error);
}

#[test]
fn rejects_malformed_serde_networks_before_identity_indexing() {
    let empty_frequency = deserialize_network({
        let mut value = valid_serialized_network(1, 3);
        value["frequency"]["hz"] = json!([]);
        value
    });
    assert_no_panic(
        Error::EmptyPortPermutationFrequency,
        empty_frequency,
        &[0, 1, 2],
    );

    let mismatched_frequency = deserialize_network({
        let mut value = valid_serialized_network(1, 3);
        value["frequency"]["hz"] = json!([1.0e9, 2.0e9]);
        value
    });
    assert_no_panic(
        Error::PortPermutationFrequencyLengthMismatch {
            expected: 2,
            actual: 1,
        },
        mismatched_frequency,
        &[0, 1, 2],
    );

    let zero_port = deserialize_network({
        let mut value = valid_serialized_network(1, 3);
        set_array_shape(&mut value, "s", &[1, 0, 0], 0);
        value
    });
    assert_no_panic(
        Error::InvalidPortPermutationSShape {
            shape: vec![1, 0, 0],
        },
        zero_port,
        &[],
    );

    let non_square = deserialize_network({
        let mut value = valid_serialized_network(1, 3);
        set_array_shape(&mut value, "s", &[1, 2, 3], 6);
        value
    });
    assert_no_panic(
        Error::InvalidPortPermutationSShape {
            shape: vec![1, 2, 3],
        },
        non_square,
        &[0, 1],
    );

    let mismatched_z0 = deserialize_network({
        let mut value = valid_serialized_network(1, 3);
        set_array_shape(&mut value, "z0", &[1, 2], 2);
        value
    });
    assert_no_panic(
        Error::InvalidPortPermutationZ0Shape { shape: vec![1, 2] },
        mismatched_z0,
        &[0, 1, 2],
    );

    let mismatched_s_frequency = deserialize_network({
        let mut value = valid_serialized_network(2, 3);
        value["frequency"]["hz"] = json!([1.0e9]);
        value
    });
    assert_no_panic(
        Error::PortPermutationFrequencyLengthMismatch {
            expected: 1,
            actual: 2,
        },
        mismatched_s_frequency,
        &[0, 1, 2],
    );
}
