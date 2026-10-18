use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{ConnectionInput, Error, Frequency, Network};
use serde_json::{Value, json};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn network(frequency: &[f64], nports: usize, offset: f64) -> Network {
    let s = Array3::from_shape_fn(
        (frequency.len(), nports, nports),
        |(sample, row, column)| {
            c(
                offset + 0.13 * sample as f64 + 0.017 * row as f64 + 0.003 * column as f64,
                -offset * 0.5 + 0.011 * sample as f64 - 0.005 * row as f64 + 0.002 * column as f64,
            )
        },
    );
    let z0 = Array2::from_shape_fn((frequency.len(), nports), |(sample, port)| {
        c(
            25.0 + offset + 2.75 * sample as f64 + 1.5 * port as f64,
            -4.0 + 0.4 * sample as f64 - 0.25 * port as f64,
        )
    });
    Network::new(Frequency::from_hz(frequency.to_vec()).unwrap(), s, z0).unwrap()
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

fn serialized_network(nfreq: usize, nports: usize, offset: f64) -> Value {
    serde_json::to_value(network(
        &(0..nfreq)
            .map(|sample| 1.0e9 + sample as f64)
            .collect::<Vec<_>>(),
        nports,
        offset,
    ))
    .unwrap()
}

fn deserialize_network(value: Value) -> Network {
    serde_json::from_value(value).expect("malformed shape fixture should deserialize")
}

#[test]
fn concatenates_unequal_segments_exactly_and_recovers_each_split() {
    let nan = f64::from_bits(0x7ff8_1234_5678_9abc);
    let frequency_a = [nan, -0.0, -3.0e9];
    let frequency_b = [nan, 1.0e9, -0.0, 1.0e9];

    let mut s_a = Array3::from_shape_fn((3, 1, 1), |(sample, _, _)| {
        c(0.25 + sample as f64, -0.5 - sample as f64)
    });
    s_a[[1, 0, 0]] = c(-0.0, nan);
    let mut z0_a = Array2::from_shape_fn((3, 1), |(sample, _)| {
        c(20.0 + sample as f64, -2.0 - sample as f64)
    });
    z0_a[[2, 0]] = c(nan, -0.0);

    let mut s_b = Array3::from_shape_fn((4, 1, 1), |(sample, _, _)| {
        c(-1.25 - sample as f64, 0.75 + sample as f64)
    });
    s_b[[0, 0, 0]] = c(nan, -0.0);
    let mut z0_b = Array2::from_shape_fn((4, 1), |(sample, _)| {
        c(70.0 + sample as f64, 3.0 + sample as f64)
    });
    z0_b[[3, 0]] = c(-0.0, nan);

    let a = Network::new(Frequency::from_hz(frequency_a.to_vec()).unwrap(), s_a, z0_a).unwrap();
    let b = Network::new(Frequency::from_hz(frequency_b.to_vec()).unwrap(), s_b, z0_b).unwrap();
    let a_before = a.clone();
    let b_before = b.clone();

    let concatenated = a.concatenate_frequency_samples(&b).unwrap();
    assert_eq!(concatenated.frequency().len(), 7);
    assert_eq!(concatenated.s().dim(), (7, 1, 1));
    assert_eq!(concatenated.z0().dim(), (7, 1));
    assert!(concatenated.s().is_standard_layout());
    assert!(concatenated.z0().is_standard_layout());
    assert_ne!(concatenated.s().as_ptr(), a.s().as_ptr());
    assert_ne!(concatenated.z0().as_ptr(), b.z0().as_ptr());

    let expected_frequency = [nan, -0.0, -3.0e9, nan, 1.0e9, -0.0, 1.0e9];
    for (&actual, expected) in concatenated.frequency().hz().iter().zip(expected_frequency) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    for sample in 0..3 {
        assert_complex_bits(concatenated.s()[[sample, 0, 0]], a.s()[[sample, 0, 0]]);
        assert_complex_bits(concatenated.z0()[[sample, 0]], a.z0()[[sample, 0]]);
    }
    for sample in 0..4 {
        assert_complex_bits(concatenated.s()[[sample + 3, 0, 0]], b.s()[[sample, 0, 0]]);
        assert_complex_bits(concatenated.z0()[[sample + 3, 0]], b.z0()[[sample, 0]]);
    }

    assert_network_bits_equal(&a, &a_before);
    assert_network_bits_equal(&b, &b_before);
    assert_network_bits_equal(
        &concatenated.select_frequency_samples(&[0, 1, 2]).unwrap(),
        &a,
    );
    assert_network_bits_equal(
        &concatenated
            .select_frequency_samples(&[3, 4, 5, 6])
            .unwrap(),
        &b,
    );
}

#[test]
fn concatenation_is_associative_and_commutes_with_same_port_selection() {
    let a = network(&[3.0e9, -1.0e9], 3, 0.1);
    let b = network(&[3.0e9], 3, 0.2);
    let c = network(&[3.0e9, 3.0e9, 8.0e9], 3, 0.3);

    let left = a
        .concatenate_frequency_samples(&b)
        .unwrap()
        .concatenate_frequency_samples(&c)
        .unwrap();
    let right = a
        .concatenate_frequency_samples(&b.concatenate_frequency_samples(&c).unwrap())
        .unwrap();
    assert_network_bits_equal(&left, &right);

    let ports = [2, 0, 1];
    let selected_then_concatenated = a
        .select_ports_zero_incident(&ports)
        .unwrap()
        .concatenate_frequency_samples(&b.select_ports_zero_incident(&ports).unwrap())
        .unwrap();
    let concatenated_then_selected = a
        .concatenate_frequency_samples(&b)
        .unwrap()
        .select_ports_zero_incident(&ports)
        .unwrap();
    assert_network_bits_equal(&selected_then_concatenated, &concatenated_then_selected);
}

#[test]
fn accepts_self_concatenation_and_nonstandard_owned_layouts() {
    let standard = network(&[1.0e9, 2.0e9, 3.0e9], 3, 0.4);
    let nonstandard = Network::new(
        standard.frequency().clone(),
        standard.s().clone().reversed_axes(),
        standard.z0().clone().reversed_axes(),
    )
    .unwrap();
    assert!(!nonstandard.s().is_standard_layout());
    assert!(!nonstandard.z0().is_standard_layout());

    let concatenated = nonstandard
        .concatenate_frequency_samples(&nonstandard)
        .unwrap();
    assert!(concatenated.s().is_standard_layout());
    assert!(concatenated.z0().is_standard_layout());
    assert_network_bits_equal(
        &concatenated.select_frequency_samples(&[0, 1, 2]).unwrap(),
        &nonstandard,
    );
    assert_network_bits_equal(
        &concatenated.select_frequency_samples(&[3, 4, 5]).unwrap(),
        &nonstandard,
    );
}

#[test]
fn rejects_mismatched_ports_and_malformed_sources_with_side_context() {
    let valid = network(&[1.0e9, 2.0e9], 2, 0.1);
    let mismatched = network(&[1.0e9], 3, 0.2);
    assert_eq!(
        valid
            .concatenate_frequency_samples(&mismatched)
            .unwrap_err(),
        Error::ConcatenateFrequencySamplesPortCountMismatch { a: 2, b: 3 }
    );

    let mut malformed_a_value = serialized_network(1, 2, 0.1);
    malformed_a_value["s"]["dim"] = json!([1, 2, 3]);
    let scalar = malformed_a_value["s"]["data"][0].clone();
    malformed_a_value["s"]["data"] = json!([
        scalar.clone(),
        scalar.clone(),
        scalar.clone(),
        scalar.clone(),
        scalar.clone(),
        scalar,
    ]);
    let malformed_a = deserialize_network(malformed_a_value);
    let result = catch_unwind(AssertUnwindSafe(|| {
        malformed_a.concatenate_frequency_samples(&valid)
    }));
    assert!(result.is_ok());
    assert!(matches!(
        result.unwrap(),
        Err(Error::InvalidConcatenateFrequencySamplesSShape {
            input: ConnectionInput::A,
            shape
        }) if shape == vec![1, 2, 3]
    ));

    let mut malformed_b_value = serialized_network(1, 2, 0.2);
    malformed_b_value["z0"]["dim"] = json!([1, 1]);
    let scalar = malformed_b_value["z0"]["data"][0].clone();
    malformed_b_value["z0"]["data"] = json!([scalar]);
    let malformed_b = deserialize_network(malformed_b_value);
    assert_eq!(
        valid
            .concatenate_frequency_samples(&malformed_b)
            .unwrap_err(),
        Error::InvalidConcatenateFrequencySamplesZ0Shape {
            input: ConnectionInput::B,
            shape: vec![1, 1],
        }
    );

    let mut empty_frequency_value = serde_json::to_value(&valid).unwrap();
    empty_frequency_value["frequency"]["hz"] = json!([]);
    let empty_frequency = deserialize_network(empty_frequency_value);
    assert_eq!(
        empty_frequency
            .concatenate_frequency_samples(&valid)
            .unwrap_err(),
        Error::EmptyConcatenateFrequencySamplesFrequency {
            input: ConnectionInput::A,
        }
    );

    let mut frequency_shape_value = serde_json::to_value(&valid).unwrap();
    frequency_shape_value["frequency"]["hz"] = json!([1.0e9, 2.0e9, 3.0e9]);
    let frequency_shape = deserialize_network(frequency_shape_value);
    assert_eq!(
        frequency_shape
            .concatenate_frequency_samples(&valid)
            .unwrap_err(),
        Error::ConcatenateFrequencySamplesFrequencyShape {
            input: ConnectionInput::A,
            expected: 3,
            actual: 2,
        }
    );
}
