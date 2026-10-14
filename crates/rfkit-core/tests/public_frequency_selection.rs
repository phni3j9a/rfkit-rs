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
        complex(
            0.125 * (frequency + 1) as f64 + 0.017 * (row + 1) as f64 + 0.003 * (column + 1) as f64,
            -0.071 * (frequency + 1) as f64 + 0.011 * (row + 1) as f64
                - 0.019 * (column + 1) as f64,
        )
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

fn assert_selected_bits(input: &Network, output: &Network, samples: &[usize]) {
    assert_eq!(output.frequency().len(), samples.len());
    assert_eq!(
        output.s().dim(),
        (samples.len(), input.nports(), input.nports())
    );
    assert_eq!(output.z0().dim(), (samples.len(), input.nports()));
    for (new_frequency, &old_frequency) in samples.iter().enumerate() {
        assert_eq!(
            output.frequency().hz()[new_frequency].to_bits(),
            input.frequency().hz()[old_frequency].to_bits()
        );
        for row in 0..input.nports() {
            for column in 0..input.nports() {
                assert_same_bits(
                    output.s()[[new_frequency, row, column]],
                    input.s()[[old_frequency, row, column]],
                );
            }
            assert_same_bits(
                output.z0()[[new_frequency, row]],
                input.z0()[[old_frequency, row]],
            );
        }
    }
}

#[test]
fn selects_one_port_asymmetric_partial_reordered_and_identity_observations() {
    let input = asymmetric_network(4, 3);
    let input_before = input.clone();
    let samples = [3, 1];
    let samples_before = samples;

    let output = input.select_frequency_samples(&samples).unwrap();

    assert_selected_bits(&input, &output, &samples);
    assert_ne!(output.s().as_ptr(), input.s().as_ptr());
    assert_ne!(output.z0().as_ptr(), input.z0().as_ptr());
    assert_network_bits_equal(&input, &input_before);
    assert_eq!(samples, samples_before);

    let one_port = input.select_ports_zero_incident(&[2]).unwrap();
    let one_sample = one_port.select_frequency_samples(&[0]).unwrap();
    assert_eq!(one_sample.s().dim(), (1, 1, 1));
    assert_eq!(one_sample.z0().dim(), (1, 1));
    assert_selected_bits(&one_port, &one_sample, &[0]);

    let one_sample_source = asymmetric_network(1, 3);
    let one_sample_output = one_sample_source.select_frequency_samples(&[0]).unwrap();
    assert_selected_bits(&one_sample_source, &one_sample_output, &[0]);

    let identity = input.select_frequency_samples(&[0, 1, 2, 3]).unwrap();
    assert_network_bits_equal(&identity, &input);
}

#[test]
fn duplicate_frequency_labels_are_selected_by_index_and_not_value() {
    let frequencies = Frequency::from_hz(vec![2.0e9, 2.0e9, -1.0e9]).unwrap();
    let s = Array3::from_shape_fn((3, 1, 1), |(frequency, _, _)| {
        complex(10.0 + frequency as f64, -(frequency as f64))
    });
    let z0 = Array2::from_shape_fn((3, 1), |(frequency, _)| {
        complex(40.0 + frequency as f64, 0.5 * frequency as f64)
    });
    let input = Network::new(frequencies, s, z0).unwrap();

    let selected = input.select_frequency_samples(&[1, 0]).unwrap();
    assert_eq!(selected.frequency().hz(), &[2.0e9, 2.0e9]);
    assert_eq!(selected.s()[[0, 0, 0]], complex(11.0, -1.0));
    assert_eq!(selected.s()[[1, 0, 0]], complex(10.0, 0.0));
    assert_eq!(selected.z0()[[0, 0]], complex(41.0, 0.5));
    assert_eq!(selected.z0()[[1, 0]], complex(40.0, 0.0));
}

#[test]
fn preserves_special_bits_nonstandard_layouts_and_unselected_nonfinite_data() {
    let nan_frequency = f64::from_bits(0x7ff8_1234_5678_9abc);
    let nan_s = f64::from_bits(0x7ff8_2468_1357_9bdf);
    let nan_z0 = f64::from_bits(0x7ff8_aaaa_bbbb_cccd);
    let frequency = Frequency::from_hz(vec![-0.0, 2.0, nan_frequency]).unwrap();
    let standard_s = Array3::from_shape_fn((3, 2, 2), |(sample, row, column)| {
        let real = match (sample, row, column) {
            // The source is transposed below, so this lands at input
            // S[1,0,1] after the nonstandard layout is installed.
            (1, 1, 0) => nan_s,
            (2, 1, 0) => -0.0,
            _ => (100 * sample + 10 * row + column) as f64,
        };
        let imag = if (sample, row, column) == (0, 1, 0) {
            -0.0
        } else {
            -((sample + row + column) as f64)
        };
        complex(real, imag)
    });
    let standard_z0 = Array2::from_shape_fn((3, 2), |(sample, port)| {
        if (sample, port) == (1, 1) {
            complex(-0.0, nan_z0)
        } else {
            complex((40 + 3 * sample + port) as f64, -(port as f64))
        }
    });
    let nonstandard_s = standard_s.clone().permuted_axes([0, 2, 1]);
    let nonstandard_z0 =
        Array2::from_shape_fn((2, 3), |(port, sample)| standard_z0[[sample, port]]).reversed_axes();
    assert!(!nonstandard_s.is_standard_layout());
    assert!(!nonstandard_z0.is_standard_layout());

    let input = Network::new(frequency, nonstandard_s, nonstandard_z0).unwrap();
    let selected_finite = input.select_frequency_samples(&[0]).unwrap();
    assert_selected_bits(&input, &selected_finite, &[0]);
    assert!(selected_finite.s().iter().all(|value| value.re.is_finite()));
    assert!(
        selected_finite
            .z0()
            .iter()
            .all(|value| value.re.is_finite())
    );

    // The source's unselected non-finite sample does not prevent a valid
    // structural selection, and the newly allocated output is standard layout.
    assert!(selected_finite.s().is_standard_layout());
    assert!(selected_finite.z0().is_standard_layout());

    let selected_nonfinite = input.select_frequency_samples(&[1]).unwrap();
    assert_eq!(
        selected_nonfinite.s()[[0, 0, 1]].re.to_bits(),
        nan_s.to_bits()
    );
    assert_eq!(
        selected_nonfinite.z0()[[0, 1]].im.to_bits(),
        nan_z0.to_bits()
    );
    assert_eq!(
        selected_nonfinite.z0()[[0, 1]].re.to_bits(),
        (-0.0f64).to_bits()
    );
    let selected_nonfinite_s_only = Network::new(
        selected_nonfinite.frequency().clone(),
        selected_nonfinite.s().clone(),
        Array2::from_elem((1, 2), complex(50.0, 0.0)),
    )
    .unwrap();
    assert!(matches!(
        selected_nonfinite_s_only.max_singular_value_power(),
        Err(Error::NonFiniteMaxSingularValuePowerS {
            frequency: 0,
            row: 0,
            column: 1,
        })
    ));

    let selected_signed_zero = input.select_frequency_samples(&[2]).unwrap();
    assert_eq!(
        selected_signed_zero.frequency().hz()[0].to_bits(),
        nan_frequency.to_bits()
    );
    assert_eq!(
        selected_signed_zero.s()[[0, 0, 1]].re.to_bits(),
        (-0.0f64).to_bits()
    );
}

#[test]
fn selection_composes_and_commutes_with_port_selection_bit_for_bit() {
    let input = asymmetric_network(4, 4);
    let first = input.select_frequency_samples(&[3, 1, 0]).unwrap();
    let sequential = first
        .select_frequency_samples(&[2, 0])
        .unwrap()
        .select_ports_zero_incident(&[2, 0])
        .unwrap();
    let direct = input
        .select_frequency_samples(&[0, 3])
        .unwrap()
        .select_ports_zero_incident(&[2, 0])
        .unwrap();
    assert_network_bits_equal(&sequential, &direct);

    let ports_first = input
        .select_ports_zero_incident(&[2, 0])
        .unwrap()
        .select_frequency_samples(&[3, 1, 0])
        .unwrap();
    let frequencies_first = input
        .select_frequency_samples(&[3, 1, 0])
        .unwrap()
        .select_ports_zero_incident(&[2, 0])
        .unwrap();
    assert_network_bits_equal(&ports_first, &frequencies_first);

    let complete = input.select_frequency_samples(&[3, 2, 1, 0]).unwrap();
    let inverse = complete.select_frequency_samples(&[3, 2, 1, 0]).unwrap();
    assert_network_bits_equal(&inverse, &input);
}

#[test]
fn agrees_with_interpolation_on_the_shared_finite_increasing_exact_knot_domain() {
    let input = asymmetric_network(4, 3);
    let selected = input.select_frequency_samples(&[1, 3]).unwrap();
    let target =
        Frequency::from_hz(vec![input.frequency().hz()[1], input.frequency().hz()[3]]).unwrap();
    let interpolated = input.interpolate_cartesian_linear(&target).unwrap();
    assert_network_bits_equal(&selected, &interpolated);
}

#[test]
fn rejects_selection_errors_with_position_and_source_context() {
    let input = asymmetric_network(3, 2);

    assert_eq!(
        input.select_frequency_samples(&[]).unwrap_err(),
        Error::FrequencySampleSelectionEmpty
    );
    assert_eq!(
        input.select_frequency_samples(&[0, 0, 2]).unwrap_err(),
        Error::FrequencySampleSelectionDuplicate {
            sample: 0,
            first_position: 0,
            second_position: 1,
        }
    );
    assert_eq!(
        input
            .select_frequency_samples(&[0, usize::MAX])
            .unwrap_err(),
        Error::FrequencySampleSelectionOutOfRange {
            position: 1,
            sample: usize::MAX,
            nsamples: 3,
        }
    );
    assert_eq!(
        input.select_frequency_samples(&[3, 0]).unwrap_err(),
        Error::FrequencySampleSelectionOutOfRange {
            position: 0,
            sample: 3,
            nsamples: 3,
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

fn assert_no_panic(error: Error, network: Network, samples: &[usize]) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        network.select_frequency_samples(samples)
    }));
    assert!(
        result.is_ok(),
        "select_frequency_samples panicked for malformed network"
    );
    assert_eq!(result.unwrap().unwrap_err(), error);
}

#[test]
fn validates_malformed_serde_networks_before_identity_or_selection_indexing() {
    let empty_frequency = deserialize_network({
        let mut value = valid_serialized_network(1, 2);
        value["frequency"]["hz"] = json!([]);
        value
    });
    assert_no_panic(
        Error::EmptyFrequencySampleSelectionFrequency,
        empty_frequency,
        &[0],
    );

    let mismatched_frequency = deserialize_network({
        let mut value = valid_serialized_network(1, 2);
        value["frequency"]["hz"] = json!([1.0e9, 2.0e9]);
        value
    });
    assert_no_panic(
        Error::FrequencySampleSelectionFrequencyLengthMismatch {
            expected: 2,
            actual: 1,
        },
        mismatched_frequency,
        &[0, 1],
    );

    // Identity selection by the malformed source frequency cardinality must
    // still validate the S frequency axis before touching any indexed data.
    let mismatched_s_frequency = deserialize_network({
        let mut value = valid_serialized_network(2, 2);
        value["frequency"]["hz"] = json!([1.0e9]);
        value
    });
    assert_no_panic(
        Error::FrequencySampleSelectionFrequencyLengthMismatch {
            expected: 1,
            actual: 2,
        },
        mismatched_s_frequency,
        &[0],
    );

    let zero_port = deserialize_network({
        let mut value = valid_serialized_network(1, 2);
        set_array_shape(&mut value, "s", &[1, 0, 0], 0);
        value
    });
    assert_no_panic(
        Error::InvalidFrequencySampleSelectionSShape {
            shape: vec![1, 0, 0],
        },
        zero_port,
        &[],
    );

    let non_square = deserialize_network({
        let mut value = valid_serialized_network(1, 2);
        set_array_shape(&mut value, "s", &[1, 1, 2], 2);
        value
    });
    assert_no_panic(
        Error::InvalidFrequencySampleSelectionSShape {
            shape: vec![1, 1, 2],
        },
        non_square,
        &[0, 1],
    );

    let mismatched_z0 = deserialize_network({
        let mut value = valid_serialized_network(1, 2);
        set_array_shape(&mut value, "z0", &[1, 1], 1);
        value
    });
    assert_no_panic(
        Error::InvalidFrequencySampleSelectionZ0Shape { shape: vec![1, 1] },
        mismatched_z0,
        &[0, 1],
    );

    let mismatched_z0_frequency = deserialize_network({
        let mut value = valid_serialized_network(3, 2);
        set_array_shape(&mut value, "z0", &[2, 2], 4);
        value
    });
    assert_no_panic(
        Error::InvalidFrequencySampleSelectionZ0Shape { shape: vec![2, 2] },
        mismatched_z0_frequency,
        &[0, 1, 2],
    );
}
