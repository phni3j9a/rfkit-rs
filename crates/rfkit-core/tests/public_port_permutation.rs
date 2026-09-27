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

fn assert_selected_bits(input: &Network, output: &Network, ports: &[usize]) {
    assert_eq!(output.nports(), ports.len());
    assert_eq!(output.frequency().len(), input.frequency().len());
    for (frequency, (&input_frequency, &output_frequency)) in input
        .frequency()
        .hz()
        .iter()
        .zip(output.frequency().hz())
        .enumerate()
    {
        assert_eq!(input_frequency.to_bits(), output_frequency.to_bits());
        for row in 0..ports.len() {
            for column in 0..ports.len() {
                assert_same_bits(
                    output.s()[[frequency, row, column]],
                    input.s()[[frequency, ports[row], ports[column]]],
                );
            }
            assert_same_bits(
                output.z0()[[frequency, row]],
                input.z0()[[frequency, ports[row]]],
            );
        }
    }
}

#[test]
fn selects_one_two_and_many_port_networks_with_multiple_frequencies() {
    for (nport, order) in [(1, vec![0]), (2, vec![1, 0]), (4, vec![2, 0, 3, 1])] {
        let input = asymmetric_network(3, nport);
        let input_before = input.clone();
        let order_before = order.clone();

        let output = input.select_ports_zero_incident(&order).unwrap();

        assert_selected_bits(&input, &output, &order);
        assert_network_bits_equal(&input, &input_before);
        assert_eq!(order, order_before);
        assert_ne!(output.s().as_ptr(), input.s().as_ptr());
        assert_ne!(output.z0().as_ptr(), input.z0().as_ptr());
    }
}

#[test]
fn identity_inverse_and_composition_are_exact_for_non_involutive_selections() {
    let input = asymmetric_network(4, 3);
    let input_before = input.clone();
    let first_order = vec![2, 0, 1];
    let second_order = vec![1, 2, 0];

    let first = input.select_ports_zero_incident(&first_order).unwrap();
    let inverse = first.select_ports_zero_incident(&second_order).unwrap();
    assert_network_bits_equal(&inverse, &input);

    let other_order = vec![0, 2, 1];
    let sequential = first.select_ports_zero_incident(&other_order).unwrap();
    let composed_order: Vec<_> = other_order
        .iter()
        .map(|&new_port| first_order[new_port])
        .collect();
    let composed = input.select_ports_zero_incident(&composed_order).unwrap();
    assert_network_bits_equal(&sequential, &composed);

    let identity = input.select_ports_zero_incident(&[0, 1, 2]).unwrap();
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
    let output = input.select_ports_zero_incident(&order).unwrap();

    assert_selected_bits(&input, &output, &order);
    let subset_ports = [1, 0];
    let subset = input.select_ports_zero_incident(&subset_ports).unwrap();
    assert_selected_bits(&input, &subset, &subset_ports);

    // The smaller selection visibly retains exact scalar bits from the
    // nonstandard source: the source port 1 reference carries a NaN payload
    // and a negative zero, while the selected S coordinate carrying a
    // negative-zero real component is retained too.  The NaN in the omitted
    // source port 2 coordinate is accepted by selection but cannot appear in
    // the selected S block.
    assert_eq!(subset.z0()[[1, 0]].im.to_bits(), nan_payload.to_bits());
    assert_eq!(subset.z0()[[1, 0]].re.to_bits(), (-0.0f64).to_bits());
    assert_eq!(subset.s()[[0, 0, 1]].re.to_bits(), (-0.0f64).to_bits());
    assert_eq!(input.s()[[1, 0, 2]].re.to_bits(), 0x7ff8_2468_1357_9bdf);
    assert!(
        subset
            .s()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(output.s().is_standard_layout());
    assert!(output.z0().is_standard_layout());
    assert!(subset.s().is_standard_layout());
    assert!(subset.z0().is_standard_layout());
    assert_network_bits_equal(&input, &input_before);
}

#[test]
fn reordered_finite_waves_obey_the_same_scattering_relation() {
    let input = asymmetric_network(2, 3);
    let order = [2, 0, 1];
    let output = input.select_ports_zero_incident(&order).unwrap();

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
fn selects_noncontiguous_subsets_and_preserves_exact_owned_coordinates() {
    let input = asymmetric_network(3, 5);
    let input_before = input.clone();
    let ports = vec![4, 1, 3];
    let ports_before = ports.clone();

    let output = input.select_ports_zero_incident(&ports).unwrap();

    assert_eq!(output.nports(), 3);
    assert_selected_bits(&input, &output, &ports);
    assert_network_bits_equal(&input, &input_before);
    assert_eq!(ports, ports_before);
    assert_ne!(output.s().as_ptr(), input.s().as_ptr());
    assert_ne!(output.z0().as_ptr(), input.z0().as_ptr());

    let one_port = input.select_ports_zero_incident(&[2]).unwrap();
    assert_eq!(one_port.s().dim(), (3, 1, 1));
    assert_eq!(one_port.z0().dim(), (3, 1));
    assert_selected_bits(&input, &one_port, &[2]);
}

#[test]
fn subset_composition_matches_the_explicit_composed_mapping_exactly() {
    let input = asymmetric_network(2, 5);
    let first = [4, 1, 3];
    let second = [2, 0];
    let sequential = input
        .select_ports_zero_incident(&first)
        .unwrap()
        .select_ports_zero_incident(&second)
        .unwrap();
    let composed = [first[second[0]], first[second[1]]];
    let direct = input.select_ports_zero_incident(&composed).unwrap();

    assert_network_bits_equal(&sequential, &direct);
}

#[test]
fn retained_scattering_relation_has_zero_incident_omitted_waves() {
    let input = asymmetric_network(1, 5);
    let retained = [4, 1, 3];
    let retained_input = [complex(0.7, -0.2), complex(-0.3, 0.8), complex(0.45, 0.1)];
    let mut full_input = [Complex64::new(0.0, 0.0); 5];
    for (new_port, &old_port) in retained.iter().enumerate() {
        full_input[old_port] = retained_input[new_port];
    }
    let full_output: Vec<_> = (0..5)
        .map(|row| {
            (0..5)
                .map(|column| input.s()[[0, row, column]] * full_input[column])
                .sum::<Complex64>()
        })
        .collect();
    let reduced = input.select_ports_zero_incident(&retained).unwrap();

    for (new_row, &old_row) in retained.iter().enumerate() {
        let reduced_output: Complex64 = (0..retained.len())
            .map(|new_column| reduced.s()[[0, new_row, new_column]] * retained_input[new_column])
            .sum();
        assert!((reduced_output - full_output[old_row]).norm() < 1.0e-14);
    }
}

#[test]
fn omitted_incident_waves_have_the_stored_reference_boundary() {
    let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let s = Array3::from_shape_fn((1, 4, 4), |(_, row, column)| {
        complex(
            0.03 * (1 + row + 2 * column) as f64,
            -0.02 * (1 + 2 * row + column) as f64,
        )
    });
    let z0 = Array2::from_shape_vec(
        (1, 4),
        vec![
            complex(42.0, 5.0),
            complex(37.0, 8.0),
            complex(61.0, -4.0),
            complex(-53.0, -7.0),
        ],
    )
    .unwrap();
    let input = Network::new(frequency, s, z0).unwrap();
    let retained = [0, 2];
    let retained_input = [complex(0.6, -0.1), complex(-0.25, 0.4)];
    let mut full_input = [Complex64::new(0.0, 0.0); 4];
    for (new_port, &old_port) in retained.iter().enumerate() {
        full_input[old_port] = retained_input[new_port];
    }
    let full_output: Vec<_> = (0..4)
        .map(|row| {
            (0..4)
                .map(|column| input.s()[[0, row, column]] * full_input[column])
                .sum::<Complex64>()
        })
        .collect();
    let selected = input.select_ports_zero_incident(&retained).unwrap();
    let selected_output: Vec<_> = (0..retained.len())
        .map(|row| {
            (0..retained.len())
                .map(|column| selected.s()[[0, row, column]] * retained_input[column])
                .sum::<Complex64>()
        })
        .collect();

    // The same nonzero retained excitation is applied to the full network
    // with a_R=0 and to the exact selected network.  This is the coordinate
    // projection being tested, not a physical termination solve.
    assert!(retained_input.iter().all(|wave| wave.norm() > 0.0));
    for (new_row, &old_row) in retained.iter().enumerate() {
        assert!((selected_output[new_row] - full_output[old_row]).norm() < 1.0e-14);
    }

    for old_port in [1, 3] {
        let reference = input.z0()[[0, old_port]];
        assert_ne!(reference.im, 0.0);
        if old_port == 1 {
            assert!(reference.re > 0.0);
        } else {
            assert!(reference.re < 0.0);
        }
        assert!(full_output[old_port].norm() > 1.0e-12);

        // Independently invert the repository's Kurokawa equations, rather
        // than assigning V=-z0*I.  With g=sqrt(abs(Re(z0))),
        // I=(g/Re(z0))*(a-b) and V=g*(a+b)-j*Im(z0)*I.
        let g = reference.re.abs().sqrt();
        let full_a = full_input[old_port];
        let full_b = full_output[old_port];
        assert_eq!(full_a, Complex64::new(0.0, 0.0));
        let current = complex(g / reference.re, 0.0) * (full_a - full_b);
        let voltage = complex(g, 0.0) * (full_a + full_b) - complex(0.0, reference.im) * current;
        let denominator = complex(2.0 * g, 0.0);
        let reconstructed_a = (voltage + reference * current) / denominator;
        let reconstructed_b = (voltage - reference.conj() * current) / denominator;

        assert!((reconstructed_a - full_a).norm() < 1.0e-14);
        assert!((reconstructed_b - full_b).norm() < 1.0e-14);
        assert!((voltage + reference * current).norm() < 1.0e-14);
        // The stored reference, rather than its conjugate, defines the zero-
        // incident boundary.  This deliberately makes no passive-load claim
        // for the negative-real reference at old port 3.
        assert!((voltage + reference.conj() * current).norm() > 1.0e-8);
    }
}

#[test]
fn singular_omitted_feedback_block_does_not_affect_exact_selection() {
    let mut s = Array3::zeros((1, 3, 3));
    s[[0, 1, 1]] = complex(0.5, 0.0);
    s[[0, 1, 2]] = complex(0.5, 0.0);
    s[[0, 2, 1]] = complex(0.5, 0.0);
    s[[0, 2, 2]] = complex(0.5, 0.0);
    s[[0, 0, 0]] = complex(0.25, -0.1);
    s[[0, 0, 1]] = complex(0.125, 0.0);
    s[[0, 0, 2]] = complex(-0.2, 0.0);
    s[[0, 1, 0]] = complex(0.3, 0.0);
    s[[0, 2, 0]] = complex(-0.4, 0.0);
    let input = Network::new(
        Frequency::from_hz(vec![2.4e9]).unwrap(),
        s,
        Array2::from_elem((1, 3), complex(50.0, 0.0)),
    )
    .unwrap();

    let omitted_block = [
        [input.s()[[0, 1, 1]], input.s()[[0, 1, 2]]],
        [input.s()[[0, 2, 1]], input.s()[[0, 2, 2]]],
    ];
    let determinant = (Complex64::new(1.0, 0.0) - omitted_block[0][0])
        * (Complex64::new(1.0, 0.0) - omitted_block[1][1])
        - (-omitted_block[0][1]) * (-omitted_block[1][0]);
    assert_eq!(determinant, Complex64::new(0.0, 0.0));
    assert_ne!(omitted_block[0][1], Complex64::new(0.0, 0.0));
    assert_ne!(omitted_block[1][0], Complex64::new(0.0, 0.0));
    assert_ne!(input.s()[[0, 0, 1]], Complex64::new(0.0, 0.0));
    assert_ne!(input.s()[[0, 0, 2]], Complex64::new(0.0, 0.0));
    assert_ne!(input.s()[[0, 1, 0]], Complex64::new(0.0, 0.0));
    assert_ne!(input.s()[[0, 2, 0]], Complex64::new(0.0, 0.0));

    let selected = input.select_ports_zero_incident(&[0]).unwrap();
    assert_eq!(selected.s()[[0, 0, 0]], input.s()[[0, 0, 0]]);
    assert_eq!(selected.z0()[[0, 0]], input.z0()[[0, 0]]);
}

#[test]
fn rejects_invalid_selections_with_structured_context() {
    let network = asymmetric_network(1, 3);

    assert_eq!(
        network.select_ports_zero_incident(&[]).unwrap_err(),
        Error::PortSelectionEmpty
    );
    assert_eq!(
        network.select_ports_zero_incident(&[0, 0, 2]).unwrap_err(),
        Error::PortSelectionDuplicate {
            port: 0,
            first_position: 0,
            second_position: 1,
        }
    );
    assert_eq!(
        network.select_ports_zero_incident(&[0, 3, 1]).unwrap_err(),
        Error::PortSelectionOutOfRange {
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
    let result = catch_unwind(AssertUnwindSafe(|| {
        network.select_ports_zero_incident(order)
    }));
    assert!(
        result.is_ok(),
        "select_ports_zero_incident panicked for malformed network"
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
        Error::EmptyPortSelectionFrequency,
        empty_frequency,
        &[0, 1, 2],
    );

    let mismatched_frequency = deserialize_network({
        let mut value = valid_serialized_network(1, 3);
        value["frequency"]["hz"] = json!([1.0e9, 2.0e9]);
        value
    });
    assert_no_panic(
        Error::PortSelectionFrequencyLengthMismatch {
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
        Error::InvalidPortSelectionSShape {
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
        Error::InvalidPortSelectionSShape {
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
        Error::InvalidPortSelectionZ0Shape { shape: vec![1, 2] },
        mismatched_z0,
        &[0, 1, 2],
    );

    let mismatched_s_frequency = deserialize_network({
        let mut value = valid_serialized_network(2, 3);
        value["frequency"]["hz"] = json!([1.0e9]);
        value
    });
    assert_no_panic(
        Error::PortSelectionFrequencyLengthMismatch {
            expected: 1,
            actual: 2,
        },
        mismatched_s_frequency,
        &[0, 1, 2],
    );
}
