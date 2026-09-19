use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use rfkit_touchstone::{Error, parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};
use serde_json::{Value, json};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn make_network(nports: usize, frequencies: &[f64], reference: f64) -> Network {
    let frequency = Frequency::from_hz(frequencies.to_vec()).unwrap();
    let s = Array3::from_shape_fn((frequencies.len(), nports, nports), |(f, row, column)| {
        Complex64::new(
            0.01 * (100 * f + 10 * row + column + 1) as f64,
            -0.02 * (100 * f + 10 * row + column + 1) as f64,
        )
    });
    let z0 = Array2::from_elem((frequencies.len(), nports), Complex64::new(reference, 0.0));
    Network::new(frequency, s, z0).unwrap()
}

fn malformed_network(base: &Network, mutate: impl FnOnce(&mut Value)) -> Network {
    let mut value = serde_json::to_value(base).unwrap();
    mutate(&mut value);
    serde_json::from_value(value).unwrap()
}

fn assert_roundtrip(network: &Network) {
    let text = write_touchstone_v1_0_s_ri_hz(network).unwrap();
    let parsed = parse_touchstone_v1_0_s(&text, network.nports()).unwrap();
    assert_eq!(parsed.frequency(), network.frequency());
    assert_eq!(parsed.s(), network.s());
    assert_eq!(parsed.z0(), network.z0());
}

#[test]
fn writes_one_two_three_four_and_five_port_physical_layouts() {
    let one = make_network(1, &[0.0], 75.0);
    assert_eq!(
        write_touchstone_v1_0_s_ri_hz(&one).unwrap(),
        "# Hz S RI R 75\n0 0.01 -0.02\n"
    );
    assert_roundtrip(&one);

    let two = make_network(2, &[1.0], 90.0);
    let two_text = write_touchstone_v1_0_s_ri_hz(&two).unwrap();
    assert_eq!(
        two_text,
        "# Hz S RI R 90\n1 0.01 -0.02 0.11 -0.22 0.02 -0.04 0.12 -0.24\n"
    );
    assert_roundtrip(&two);

    let three = make_network(3, &[1.0], 61.0);
    let three_text = write_touchstone_v1_0_s_ri_hz(&three).unwrap();
    assert_eq!(
        three_text,
        "# Hz S RI R 61\n1 0.01 -0.02 0.02 -0.04 0.03 -0.06\n0.11 -0.22 0.12 -0.24 0.13 -0.26\n0.21 -0.42 0.22 -0.44 0.23 -0.46\n"
    );
    assert_roundtrip(&three);

    let four = make_network(4, &[1.0], 62.0);
    let four_text = write_touchstone_v1_0_s_ri_hz(&four).unwrap();
    assert_eq!(
        four_text,
        "# Hz S RI R 62\n1 0.01 -0.02 0.02 -0.04 0.03 -0.06 0.04 -0.08\n0.11 -0.22 0.12 -0.24 0.13 -0.26 0.14 -0.28\n0.21 -0.42 0.22 -0.44 0.23 -0.46 0.24 -0.48\n0.31 -0.62 0.32 -0.64 0.33 -0.66 0.34 -0.68\n"
    );
    assert_roundtrip(&four);

    let five = make_network(5, &[1.0], 63.0);
    let five_text = write_touchstone_v1_0_s_ri_hz(&five).unwrap();
    assert_eq!(
        five_text,
        "# Hz S RI R 63\n1 0.01 -0.02 0.02 -0.04 0.03 -0.06 0.04 -0.08\n0.05 -0.1\n0.11 -0.22 0.12 -0.24 0.13 -0.26 0.14 -0.28\n0.15 -0.3\n0.21 -0.42 0.22 -0.44 0.23 -0.46 0.24 -0.48\n0.25 -0.5\n0.31 -0.62 0.32 -0.64 0.33 -0.66 0.34 -0.68\n0.35000000000000003 -0.7000000000000001\n0.41000000000000003 -0.8200000000000001 0.42 -0.84 0.43 -0.86 0.44 -0.88\n0.45 -0.9\n"
    );
    assert_roundtrip(&five);
}

#[test]
fn preserves_asymmetric_order_multiple_frequencies_and_non_50_reference() {
    let frequency = Frequency::from_hz(vec![0.0, 1.0e9]).unwrap();
    let s = Array3::from_shape_vec(
        (2, 2, 2),
        vec![
            Complex64::new(0.11, 0.12),
            Complex64::new(0.13, 0.14),
            Complex64::new(0.21, 0.22),
            Complex64::new(0.23, 0.24),
            Complex64::new(-0.31, 0.32),
            Complex64::new(-0.33, 0.34),
            Complex64::new(-0.41, 0.42),
            Complex64::new(-0.43, 0.44),
        ],
    )
    .unwrap();
    let z0 = Array2::from_elem((2, 2), Complex64::new(123.75, 0.0));
    let network = Network::new(frequency, s, z0).unwrap();
    let text = write_touchstone_v1_0_s_ri_hz(&network).unwrap();
    assert_eq!(
        text,
        "# Hz S RI R 123.75\n0 0.11 0.12 0.21 0.22 0.13 0.14 0.23 0.24\n1000000000 -0.31 0.32 -0.41 0.42 -0.33 0.34 -0.43 0.44\n"
    );
    assert_roundtrip(&network);
}

#[test]
fn roundtrip_preserves_finite_extreme_and_subnormal_binary64_values() {
    let values = [
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::MAX,
        -f64::MAX,
        1.0e-200,
        -1.0e200,
        -0.0,
    ];
    let frequencies = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, f64::MAX];
    let frequency = Frequency::from_hz(frequencies.to_vec()).unwrap();
    let s = Array3::from_shape_fn((values.len(), 1, 1), |(index, _, _)| {
        Complex64::new(values[index], -values[index])
    });
    let z0 = Array2::from_elem((values.len(), 1), Complex64::new(1.0e308, 0.0));
    let network = Network::new(frequency, s, z0).unwrap();
    let text = write_touchstone_v1_0_s_ri_hz(&network).unwrap();
    let parsed = parse_touchstone_v1_0_s(&text, 1).unwrap();
    assert_eq!(parsed.frequency(), network.frequency());
    for (actual, expected) in parsed.s().iter().zip(network.s().iter()) {
        assert_eq!(actual.re, expected.re);
        assert_eq!(actual.im, expected.im);
    }
    assert_eq!(parsed.z0(), network.z0());
}

#[test]
fn repeated_writes_are_deterministic_and_do_not_modify_input() {
    let network = make_network(5, &[0.0, 1.0, 2.0], 77.5);
    let before = network.clone();
    let first = write_touchstone_v1_0_s_ri_hz(&network).unwrap();
    let second = write_touchstone_v1_0_s_ri_hz(&network).unwrap();
    assert_eq!(first, second);
    assert_eq!(network, before);
    assert!(first.is_ascii());
    assert!(first.ends_with('\n'));
    assert!(!first.ends_with("\n\n"));
}

#[test]
fn read_transform_write_read_is_an_analytic_public_workflow() {
    let input = parse_touchstone_v1_0_s("# Hz S RI R 75\n0 0.2 0\n1000000000 0.4 0\n", 1).unwrap();
    let target_z0 = Array2::from_elem((2, 1), Complex64::new(100.0, 0.0));
    let transformed = input.renormalize_power(target_z0).unwrap();
    let text = write_touchstone_v1_0_s_ri_hz(&transformed).unwrap();
    let output = parse_touchstone_v1_0_s(&text, 1).unwrap();
    assert_eq!(output.frequency(), transformed.frequency());
    assert_eq!(output.z0(), transformed.z0());
    // For a one-port power-wave renormalization, z=75*(1+s)/(1-s) is
    // preserved; the output reference is explicitly 100 ohms.
    for index in 0..2 {
        let z = input.s()[[index, 0, 0]];
        let expected_z = 75.0 * (1.0 + z.re) / (1.0 - z.re);
        let s = output.s()[[index, 0, 0]];
        let recovered_z = 100.0 * (1.0 + s.re) / (1.0 - s.re);
        assert!((recovered_z - expected_z).abs() <= 1.0e-12);
    }
}

#[test]
fn rejects_every_writer_domain_class_with_actionable_errors() {
    let base = make_network(2, &[1.0, 2.0], 50.0);

    let empty = malformed_network(&base, |value| {
        value["frequency"]["hz"] = json!([]);
    });
    assert!(matches!(
        write_touchstone_v1_0_s_ri_hz(&empty),
        Err(Error::WriterEmptyFrequency)
    ));

    let invalid_s_shape = malformed_network(&base, |value| {
        value["s"]["dim"] = json!([2, 1, 4]);
    });
    assert!(matches!(
        write_touchstone_v1_0_s_ri_hz(&invalid_s_shape),
        Err(Error::WriterInvalidSShape { .. })
    ));

    let invalid_z0_shape = malformed_network(&base, |value| {
        value["z0"]["dim"] = json!([2, 1]);
        value["z0"]["data"] = json!([[50.0, 0.0], [50.0, 0.0]]);
    });
    assert!(matches!(
        write_touchstone_v1_0_s_ri_hz(&invalid_z0_shape),
        Err(Error::WriterInvalidZ0Shape { .. })
    ));

    for frequency in [f64::NAN, f64::INFINITY, -1.0] {
        let network = Network::new(
            Frequency::from_hz(vec![frequency]).unwrap(),
            Array3::zeros((1, 1, 1)),
            Array2::from_elem((1, 1), Complex64::new(50.0, 0.0)),
        )
        .unwrap();
        assert!(matches!(
            write_touchstone_v1_0_s_ri_hz(&network),
            Err(Error::WriterInvalidFrequency { .. })
        ));
    }

    let descending = make_network(1, &[2.0, 1.0], 50.0);
    assert!(matches!(
        write_touchstone_v1_0_s_ri_hz(&descending),
        Err(Error::WriterFrequencyNotStrictlyIncreasing { .. })
    ));
    let duplicate = make_network(1, &[1.0, 1.0], 50.0);
    assert!(matches!(
        write_touchstone_v1_0_s_ri_hz(&duplicate),
        Err(Error::WriterFrequencyNotStrictlyIncreasing { .. })
    ));

    let nonfinite_s = Network::new(
        Frequency::from_hz(vec![1.0]).unwrap(),
        Array3::from_elem((1, 1, 1), Complex64::new(f64::NAN, 0.0)),
        Array2::from_elem((1, 1), Complex64::new(50.0, 0.0)),
    )
    .unwrap();
    assert!(matches!(
        write_touchstone_v1_0_s_ri_hz(&nonfinite_s),
        Err(Error::WriterNonFiniteS { .. })
    ));

    for value in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(50.0, f64::INFINITY),
        Complex64::new(0.0, 0.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(50.0, 1.0),
    ] {
        let network = Network::new(
            Frequency::from_hz(vec![1.0]).unwrap(),
            Array3::zeros((1, 1, 1)),
            Array2::from_elem((1, 1), value),
        )
        .unwrap();
        assert!(matches!(
            write_touchstone_v1_0_s_ri_hz(&network),
            Err(Error::WriterInvalidZ0 { .. })
        ));
    }

    let mismatch = Network::new(
        Frequency::from_hz(vec![1.0, 2.0]).unwrap(),
        Array3::zeros((2, 1, 1)),
        Array2::from_shape_vec(
            (2, 1),
            vec![Complex64::new(50.0, 0.0), Complex64::new(51.0, 0.0)],
        )
        .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        write_touchstone_v1_0_s_ri_hz(&mismatch),
        Err(Error::WriterMismatchedZ0 { .. })
    ));
}

#[test]
fn malformed_serde_shapes_never_panic() {
    let base = make_network(2, &[1.0], 50.0);
    let malformed = [
        malformed_network(&base, |value| value["s"]["dim"] = json!([1, 1, 4])),
        malformed_network(&base, |value| value["s"]["dim"] = json!([1, 4, 1])),
        malformed_network(&base, |value| {
            value["s"]["dim"] = json!([0, 2, 2]);
            value["s"]["data"] = json!([]);
        }),
        malformed_network(&base, |value| {
            value["z0"]["dim"] = json!([1, 1]);
            value["z0"]["data"] = json!([[50.0, 0.0]]);
        }),
        malformed_network(&base, |value| value["z0"]["dim"] = json!([2, 1])),
    ];
    for network in malformed {
        let result = catch_unwind(AssertUnwindSafe(|| write_touchstone_v1_0_s_ri_hz(&network)));
        assert!(result.is_ok(), "writer panicked for malformed serde state");
        assert!(result.unwrap().is_err());
    }
}
