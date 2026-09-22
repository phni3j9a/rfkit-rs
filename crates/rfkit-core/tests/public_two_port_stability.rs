use approx::assert_relative_eq;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, Network, TwoPortStability, TwoPortStabilityArithmetic};
use serde_json::{Value, json};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn network(frequency_hz: Vec<f64>, values: Vec<Complex64>, z0: Vec<Complex64>) -> Network {
    let nfreq = frequency_hz.len();
    Network::new(
        Frequency::from_hz(frequency_hz).unwrap(),
        Array3::from_shape_vec((nfreq, 2, 2), values).unwrap(),
        Array2::from_shape_vec((nfreq, 2), z0).unwrap(),
    )
    .unwrap()
}

fn real_z0(nfreq: usize) -> Vec<Complex64> {
    vec![c(50.0, 0.0); 2 * nfreq]
}

fn assert_metric(actual: TwoPortStability, delta: Complex64, k: Option<f64>) {
    assert_relative_eq!(actual.delta.re, delta.re, epsilon = 1e-14);
    assert_relative_eq!(actual.delta.im, delta.im, epsilon = 1e-14);
    match (actual.rollet_k, k) {
        (Some(actual), Some(expected)) => assert_relative_eq!(actual, expected, epsilon = 1e-14),
        (None, None) => {}
        other => panic!("metric mismatch: {other:?}"),
    }
}

#[test]
fn analytic_values_cover_boundary_active_and_undefined_cases() {
    let source = network(
        vec![1.0e9, 2.0e9, 3.0e9, 4.0e9, 5.0e9],
        vec![
            // K=2.6, delta=-0.2.
            c(0.0, 0.0),
            c(0.1, 0.0),
            c(2.0, 0.0),
            c(0.0, 0.0),
            // Ideal through: K=1 and |delta|=1.
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            // K>1 while |delta|>1.
            c(2.0, 0.0),
            c(0.1, 0.0),
            c(0.1, 0.0),
            c(2.0, 0.0),
            // Finite negative K.
            c(0.8, 0.0),
            c(0.5, 0.0),
            c(0.4, 0.0),
            c(0.8, 0.0),
            // Finite sub-unity positive K.
            c(0.7, 0.0),
            c(0.5, 0.0),
            c(0.5, 0.0),
            c(0.7, 0.0),
        ],
        real_z0(5),
    );
    let actual = source.two_port_stability_power().unwrap();
    assert_eq!(actual.len(), 5);
    assert_metric(actual[0], c(-0.2, 0.0), Some(2.6));
    assert_metric(actual[1], c(-1.0, 0.0), Some(1.0));
    assert_metric(actual[2], c(3.99, 0.0), Some(446.005));
    assert_metric(actual[3], c(0.44, 0.0), Some(-0.216));
    assert_metric(actual[4], c(0.24, 0.0), Some(0.1552));
}

#[test]
fn complex_nonreciprocal_singular_and_mixed_unilateral_samples_are_supported() {
    let source = network(
        vec![-2.0, -0.0, 1.0, 1.0],
        vec![
            // Complex, nonreciprocal defined sample.
            c(0.2, 0.1),
            c(0.3, -0.2),
            c(-0.4, 0.25),
            c(0.1, -0.3),
            // S12 is exact signed-zero complex zero: undefined K, delta kept.
            c(0.1, 0.2),
            c(-0.0, 0.0),
            c(0.5, -0.1),
            c(0.2, 0.0),
            // S21 is exactly isolated.
            c(0.1, 0.0),
            c(0.4, 0.3),
            c(-0.0, -0.0),
            c(-0.2, 0.1),
            // Singular S (S11*S22 == S12*S21), but K remains finite.
            c(1.0, 0.0),
            c(2.0, 0.0),
            c(0.5, 0.0),
            c(1.0, 0.0),
        ],
        vec![
            c(50.0, 5.0),
            c(71.0, -3.0),
            c(55.0, 1.0),
            c(63.0, 7.0),
            c(58.0, 0.5),
            c(67.0, -4.0),
            c(61.0, 2.0),
            c(75.0, 3.0),
        ],
    );
    let actual = source.two_port_stability_power().unwrap();
    assert_eq!(actual.len(), 4);
    assert!(actual[0].rollet_k.is_some());
    assert_relative_eq!(actual[1].delta.re, 0.02, epsilon = 1e-14);
    assert_relative_eq!(actual[1].delta.im, 0.04, epsilon = 1e-14);
    assert_eq!(actual[1].rollet_k, None);
    assert_metric(actual[2], c(-0.02, 0.01), None);
    assert_eq!(actual[2].rollet_k, None);
    assert_metric(actual[3], c(0.0, 0.0), Some(-0.5));
}

#[test]
fn exact_unilateral_skips_unused_k_overflow_but_delta_failures_remain_errors() {
    let finite_delta = network(
        vec![1.0],
        vec![c(1.0e308, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(1.0e-308, 0.0)],
        real_z0(1),
    );
    let result = finite_delta.two_port_stability_power().unwrap();
    assert_relative_eq!(result[0].delta.re, 1.0, epsilon = 1.0e-14);
    assert!(result[0].delta.im.is_finite());
    assert_eq!(result[0].rollet_k, None);

    let determinant_overflow = network(
        vec![1.0],
        vec![c(1.0e308, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(1.0e308, 0.0)],
        real_z0(1),
    );
    assert_eq!(
        determinant_overflow.two_port_stability_power().unwrap_err(),
        Error::NonFiniteTwoPortStabilityComputation {
            frequency: 0,
            stage: TwoPortStabilityArithmetic::DeltaS11S22Product,
        }
    );
}

#[test]
fn output_aligns_with_opaque_frequency_labels_and_does_not_mutate_source() {
    let frequency = vec![-3.0, -0.0, -3.0, 2.0];
    let values = vec![
        c(0.1, 0.2),
        c(0.3, 0.1),
        c(-0.2, 0.4),
        c(0.2, -0.1),
        c(0.0, 0.0),
        c(0.2, 0.0),
        c(0.1, 0.0),
        c(0.0, 0.0),
        c(0.2, -0.1),
        c(0.4, 0.0),
        c(0.3, 0.0),
        c(0.2, 0.1),
        c(0.1, 0.0),
        c(0.6, 0.0),
        c(0.4, 0.0),
        c(0.1, 0.0),
    ];
    let z0 = vec![
        c(40.0, 2.0),
        c(63.0, -4.0),
        c(41.0, 2.5),
        c(64.0, -3.0),
        c(42.0, 3.0),
        c(65.0, -2.0),
        c(43.0, 3.5),
        c(66.0, -1.0),
    ];
    let source = network(frequency.clone(), values, z0);
    let snapshot = source.clone();
    let result = source.two_port_stability_power().unwrap();
    assert_eq!(result.len(), frequency.len());
    assert_eq!(source, snapshot);
    assert_eq!(source.frequency().hz()[1].to_bits(), (-0.0f64).to_bits());
    assert_eq!(source.frequency().hz(), frequency.as_slice());
}

#[test]
fn port_exchange_preserves_delta_and_k_state() {
    let source = network(
        vec![1.0e9, 2.0e9],
        vec![
            c(0.2, 0.1),
            c(0.3, -0.2),
            c(-0.4, 0.25),
            c(0.1, -0.3),
            c(0.1, 0.0),
            c(0.0, 0.0),
            c(0.2, 0.0),
            c(0.4, 0.0),
        ],
        vec![c(50.0, 5.0), c(71.0, -3.0), c(55.0, 1.0), c(63.0, 7.0)],
    );
    let exchanged = source.permute_ports(&[1, 0]).unwrap();
    let left = source.two_port_stability_power().unwrap();
    let right = exchanged.two_port_stability_power().unwrap();
    for (left, right) in left.iter().zip(right.iter()) {
        assert_eq!(left.delta, right.delta);
        assert_eq!(left.rollet_k.is_some(), right.rollet_k.is_some());
        if let (Some(left), Some(right)) = (left.rollet_k, right.rollet_k) {
            assert_relative_eq!(left, right, epsilon = 1e-14);
        }
    }
}

#[test]
fn positive_real_power_renormalization_preserves_finite_k() {
    let source = network(
        vec![1.0e9, 2.0e9],
        vec![
            c(0.1, 0.02),
            c(0.3, -0.1),
            c(-0.2, 0.15),
            c(0.4, -0.03),
            c(0.12, -0.04),
            c(0.24, 0.07),
            c(-0.18, 0.06),
            c(0.31, -0.02),
        ],
        vec![c(50.0, 0.0), c(75.0, 0.0), c(55.0, 0.0), c(68.0, 0.0)],
    );
    let target = Array2::from_shape_vec(
        (2, 2),
        vec![c(83.0, 4.0), c(47.0, -3.0), c(91.0, 2.0), c(62.0, 5.0)],
    )
    .unwrap();
    let before = source.two_port_stability_power().unwrap();
    let changed = source.renormalize_direct_power(target).unwrap();
    let after = changed.two_port_stability_power().unwrap();
    for (before, after) in before.iter().zip(after.iter()) {
        match (before.rollet_k, after.rollet_k) {
            (Some(before), Some(after)) => assert_relative_eq!(before, after, epsilon = 1e-11),
            (None, None) => {}
            other => panic!("renormalization changed K definedness: {other:?}"),
        }
    }
}

#[test]
fn accepts_finite_tiny_nonzero_transmission_when_arithmetic_is_representable() {
    let source = network(
        vec![1.0],
        vec![c(0.0, 0.0), c(1.0e-150, 0.0), c(1.0e-150, 0.0), c(0.0, 0.0)],
        real_z0(1),
    );
    let result = source.two_port_stability_power().unwrap();
    assert_relative_eq!(result[0].delta.re, -1.0e-300, epsilon = 1.0e-314);
    assert_eq!(result[0].delta.im, 0.0);
    assert_relative_eq!(result[0].rollet_k.unwrap(), 5.0e299, epsilon = 1.0e-12);
}

#[test]
fn arithmetic_underflow_and_overflow_are_errors_not_none() {
    let underflow = network(
        vec![1.0],
        vec![c(0.0, 0.0), c(1.0e-200, 0.0), c(1.0e-200, 0.0), c(0.0, 0.0)],
        real_z0(1),
    );
    assert_eq!(
        underflow.two_port_stability_power().unwrap_err(),
        Error::NonFiniteTwoPortStabilityComputation {
            frequency: 0,
            stage: TwoPortStabilityArithmetic::TransmissionMagnitudeProduct,
        }
    );

    let overflow = network(
        vec![1.0],
        vec![c(0.0, 0.0), c(1.0e200, 0.0), c(1.0e200, 0.0), c(0.0, 0.0)],
        real_z0(1),
    );
    assert!(matches!(
        overflow.two_port_stability_power(),
        Err(Error::NonFiniteTwoPortStabilityComputation { frequency: 0, .. })
    ));
}

#[test]
fn rejects_nonfinite_data_and_nonpositive_real_references() {
    let nonfinite_frequency = network(
        vec![f64::NAN],
        vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        real_z0(1),
    );
    assert!(matches!(
        nonfinite_frequency.two_port_stability_power(),
        Err(Error::NonFiniteTwoPortStabilityFrequency { index: 0, .. })
    ));

    let nonfinite_s = network(
        vec![1.0],
        vec![c(f64::INFINITY, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        real_z0(1),
    );
    assert_eq!(
        nonfinite_s.two_port_stability_power().unwrap_err(),
        Error::NonFiniteTwoPortStabilityS {
            frequency: 0,
            row: 0,
            column: 0,
        }
    );

    for (port, reference) in [
        (0, c(f64::NAN, 1.0)),
        (1, c(50.0, f64::NAN)),
        (0, c(f64::INFINITY, 1.0)),
    ] {
        let mut references = real_z0(2);
        references[2 + port] = reference;
        let invalid = network(
            vec![1.0, 2.0],
            vec![
                c(0.0, 0.0),
                c(0.1, 0.0),
                c(0.2, 0.0),
                c(0.3, 0.0),
                c(0.0, 0.0),
                c(0.1, 0.0),
                c(0.2, 0.0),
                c(0.3, 0.0),
            ],
            references,
        );
        assert_eq!(
            invalid.two_port_stability_power().unwrap_err(),
            Error::NonFiniteTwoPortStabilityZ0 { frequency: 1, port }
        );
    }

    for reference in [c(0.0, 1.0), c(-1.0, 2.0), c(-0.0, 3.0)] {
        let invalid = network(
            vec![1.0],
            vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
            vec![reference, c(50.0, 0.0)],
        );
        assert!(matches!(
            invalid.two_port_stability_power(),
            Err(Error::NonPositiveRealTwoPortStabilityReferenceImpedance {
                frequency: 0,
                port: 0,
                ..
            })
        ));
    }
}

fn valid_serialized_network() -> Value {
    serde_json::to_value(network(
        vec![1.0e9],
        vec![c(0.1, 0.0), c(0.2, 0.0), c(0.3, 0.0), c(0.4, 0.0)],
        real_z0(1),
    ))
    .unwrap()
}

fn deserialize_network(value: Value) -> Network {
    serde_json::from_value(value).expect("malformed shape fixture should deserialize")
}

fn set_array_shape(value: &mut Value, field: &str, shape: &[usize], data_len: usize) {
    value[field]["dim"] = json!(shape);
    let source_data = value[field]["data"]
        .as_array()
        .expect("ndarray serde data is an array")
        .clone();
    let mut data = Vec::with_capacity(data_len);
    for index in 0..data_len {
        data.push(source_data[index % source_data.len()].clone());
    }
    value[field]["data"] = json!(data);
}

fn assert_no_panic(network: Network, expected: Error) {
    let result = catch_unwind(AssertUnwindSafe(|| network.two_port_stability_power()));
    assert!(
        result.is_ok(),
        "two-port stability panicked on malformed data"
    );
    assert_eq!(result.unwrap().unwrap_err(), expected);
}

#[test]
fn rejects_empty_mismatched_wrong_port_and_malformed_shapes_without_indexing() {
    let empty = deserialize_network({
        let mut value = valid_serialized_network();
        value["frequency"]["hz"] = json!([]);
        value
    });
    assert_no_panic(empty, Error::EmptyTwoPortStabilityFrequency);

    let mismatched = deserialize_network({
        let mut value = valid_serialized_network();
        value["frequency"]["hz"] = json!([1.0e9, 2.0e9]);
        value
    });
    assert_no_panic(
        mismatched,
        Error::TwoPortStabilityFrequencyLengthMismatch {
            expected: 2,
            actual: 1,
        },
    );

    let wrong_ports = deserialize_network({
        let mut value = valid_serialized_network();
        set_array_shape(&mut value, "s", &[1, 3, 3], 9);
        set_array_shape(&mut value, "z0", &[1, 3], 3);
        value
    });
    assert_no_panic(
        wrong_ports,
        Error::InvalidTwoPortStabilitySShape {
            shape: vec![1, 3, 3],
        },
    );

    let malformed_z0 = deserialize_network({
        let mut value = valid_serialized_network();
        set_array_shape(&mut value, "z0", &[1, 3], 3);
        value
    });
    assert_no_panic(
        malformed_z0,
        Error::InvalidTwoPortStabilityZ0Shape { shape: vec![1, 3] },
    );
}
