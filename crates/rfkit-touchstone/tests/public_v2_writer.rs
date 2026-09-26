use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use rfkit_touchstone::{
    Error, parse_touchstone_v2_0_s, write_touchstone_v1_0_s_ri_hz,
    write_touchstone_v2_0_s_full_ri_hz,
};
use serde_json::{Value, json};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn make_network(nports: usize, frequencies: &[f64], references: &[f64]) -> Network {
    assert_eq!(references.len(), nports);
    let frequency = Frequency::from_hz(frequencies.to_vec()).expect("fixed frequency grid");
    let s = Array3::from_shape_fn((frequencies.len(), nports, nports), |(f, row, column)| {
        let index = (f * nports * nports + row * nports + column + 1) as f64;
        Complex64::new(0.01 * index, -0.02 * index)
    });
    let z0 = Array2::from_shape_fn((frequencies.len(), nports), |(_, port)| {
        Complex64::new(references[port], 0.0)
    });
    Network::new(frequency, s, z0).expect("fixed Network dimensions")
}

fn malformed_network(base: &Network, mutate: impl FnOnce(&mut Value)) -> Network {
    let mut value = serde_json::to_value(base).expect("Network serializes");
    mutate(&mut value);
    serde_json::from_value(value).expect("malformed serde Network remains readable")
}

fn assert_bitwise_equal(actual: &Network, expected: &Network) {
    assert_eq!(
        actual.frequency().hz().len(),
        expected.frequency().hz().len()
    );
    for (actual, expected) in actual
        .frequency()
        .hz()
        .iter()
        .zip(expected.frequency().hz())
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(actual.s().dim(), expected.s().dim());
    for (actual, expected) in actual.s().iter().zip(expected.s().iter()) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }
    assert_eq!(actual.z0().dim(), expected.z0().dim());
    for (actual, expected) in actual.z0().iter().zip(expected.z0().iter()) {
        assert_eq!(actual.re.to_bits(), expected.re.to_bits());
        assert_eq!(actual.im.to_bits(), expected.im.to_bits());
    }
}

#[test]
fn emits_exact_headers_orders_and_one_record_line_per_frequency() {
    let one = make_network(1, &[1.0], &[73.0]);
    assert_eq!(
        write_touchstone_v2_0_s_full_ri_hz(&one).unwrap(),
        "[Version] 2.0\n# Hz S RI R 73\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Reference] 73\n[Matrix Format] Full\n[Network Data]\n1 0.01 -0.02\n[End]\n"
    );

    let two = make_network(2, &[1.0], &[37.0, 83.0]);
    assert_eq!(
        write_touchstone_v2_0_s_full_ri_hz(&two).unwrap(),
        "[Version] 2.0\n# Hz S RI R 37\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n[Reference] 37 83\n[Matrix Format] Full\n[Network Data]\n1 0.01 -0.02 0.02 -0.04 0.03 -0.06 0.04 -0.08\n[End]\n"
    );

    let five = make_network(5, &[1.0, 2.0], &[37.0, 48.0, 59.0, 70.0, 81.0]);
    let text = write_touchstone_v2_0_s_full_ri_hz(&five).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 10);
    assert_eq!(lines[0], "[Version] 2.0");
    assert_eq!(lines[1], "# Hz S RI R 37");
    assert_eq!(lines[4], "[Reference] 37 48 59 70 81");
    assert_eq!(lines[7].split_whitespace().count(), 1 + 2 * 25);
    assert_eq!(lines[9], "[End]");
    assert!(text.is_ascii());
    assert!(text.ends_with('\n'));
    assert!(!text.ends_with("\n\n"));
}

#[test]
fn preserves_asymmetric_data_unequal_references_and_exact_v2_roundtrip() {
    let frequencies = [0.0, 1.0e9, 2.0e9];
    let frequency = Frequency::from_hz(frequencies.to_vec()).unwrap();
    let s = Array3::from_shape_vec(
        (3, 2, 2),
        vec![
            Complex64::new(0.125, -0.25),
            Complex64::new(-0.375, 0.5),
            Complex64::new(0.625, -0.75),
            Complex64::new(-0.875, 1.0),
            Complex64::new(-1.125, 1.25),
            Complex64::new(1.375, -1.5),
            Complex64::new(-1.625, 1.75),
            Complex64::new(1.875, -2.0),
            Complex64::new(2.125, -2.25),
            Complex64::new(-2.375, 2.5),
            Complex64::new(2.625, -2.75),
            Complex64::new(-2.875, 3.0),
        ],
    )
    .unwrap();
    let references = [37.125, 83.875];
    let z0 = Array2::from_shape_fn((3, 2), |(_, port)| Complex64::new(references[port], 0.0));
    let network = Network::new(frequency, s, z0).unwrap();
    let before = network.clone();
    let first = write_touchstone_v2_0_s_full_ri_hz(&network).unwrap();
    let second = write_touchstone_v2_0_s_full_ri_hz(&network).unwrap();
    assert_eq!(first, second);
    assert_eq!(network, before);

    let reread = parse_touchstone_v2_0_s(&first).unwrap();
    assert_bitwise_equal(&reread, &network);
}

#[test]
fn preserves_nontrivial_binary64_and_signed_zero_values() {
    let frequency = Frequency::from_hz(vec![-0.0, 1.0, 2.0]).unwrap();
    let values = [
        Complex64::new(f64::from_bits(1), -0.0),
        Complex64::new(-f64::from_bits(1), f64::MAX),
        Complex64::new(f64::MIN_POSITIVE, -f64::MIN_POSITIVE),
    ];
    let s = Array3::from_shape_fn((3, 1, 1), |(index, _, _)| values[index]);
    let z0 = Array2::from_elem((3, 1), Complex64::new(61.25, 0.0));
    let network = Network::new(frequency, s, z0).unwrap();
    let text = write_touchstone_v2_0_s_full_ri_hz(&network).unwrap();
    assert!(text.contains("-0"));
    let reread = parse_touchstone_v2_0_s(&text).unwrap();
    assert_bitwise_equal(&reread, &network);
}

#[test]
fn accepts_unequal_ports_but_rejects_frequency_varying_single_port() {
    let network = make_network(3, &[1.0, 2.0], &[37.0, 61.0, 83.0]);
    let text = write_touchstone_v2_0_s_full_ri_hz(&network).unwrap();
    assert!(text.contains("[Reference] 37 61 83\n"));

    let varying = Network::new(
        Frequency::from_hz(vec![1.0, 2.0]).unwrap(),
        Array3::zeros((2, 3, 3)),
        Array2::from_shape_vec(
            (2, 3),
            vec![
                Complex64::new(37.0, 0.0),
                Complex64::new(61.0, 0.0),
                Complex64::new(83.0, 0.0),
                Complex64::new(37.0, 0.0),
                Complex64::new(62.0, 0.0),
                Complex64::new(83.0, 0.0),
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let error = write_touchstone_v2_0_s_full_ri_hz(&varying).unwrap_err();
    assert!(matches!(
        error,
        Error::WriterV2MismatchedZ0 {
            frequency: 1,
            port: 1,
            expected: 61.0,
            actual: Complex64 { re: 62.0, .. },
        }
    ));
    let message = error.to_string();
    assert!(message.contains("port 1"));
    assert!(message.contains("constant across samples"));
    assert!(!message.contains("all ports"));
}

#[test]
fn rejects_every_v2_writer_domain_class_with_context() {
    let base = make_network(2, &[1.0, 2.0], &[37.0, 83.0]);

    let empty = malformed_network(&base, |value| {
        value["frequency"]["hz"] = json!([]);
    });
    assert!(matches!(
        write_touchstone_v2_0_s_full_ri_hz(&empty),
        Err(Error::WriterEmptyFrequency)
    ));

    let invalid_s_shape = malformed_network(&base, |value| {
        value["s"]["dim"] = json!([2, 1, 4]);
    });
    assert!(matches!(
        write_touchstone_v2_0_s_full_ri_hz(&invalid_s_shape),
        Err(Error::WriterV2InvalidSShape { .. })
    ));

    let mismatched_frequency_shape = malformed_network(&base, |value| {
        value["s"]["dim"] = json!([1, 2, 2]);
        value["s"]["data"] = json!([[0.01, -0.02], [0.02, -0.04], [0.03, -0.06], [0.04, -0.08]]);
    });
    assert!(matches!(
        write_touchstone_v2_0_s_full_ri_hz(&mismatched_frequency_shape),
        Err(Error::WriterV2InvalidSShape { .. })
    ));

    let invalid_z0_shape = malformed_network(&base, |value| {
        value["z0"]["dim"] = json!([2, 1]);
        value["z0"]["data"] = json!([[37.0, 0.0], [37.0, 0.0]]);
    });
    assert!(matches!(
        write_touchstone_v2_0_s_full_ri_hz(&invalid_z0_shape),
        Err(Error::WriterV2InvalidZ0Shape { .. })
    ));

    for frequency in [f64::NAN, f64::INFINITY, -1.0] {
        let network = Network::new(
            Frequency::from_hz(vec![frequency]).unwrap(),
            Array3::zeros((1, 1, 1)),
            Array2::from_elem((1, 1), Complex64::new(50.0, 0.0)),
        )
        .unwrap();
        assert!(matches!(
            write_touchstone_v2_0_s_full_ri_hz(&network),
            Err(Error::WriterV2InvalidFrequency { .. })
        ));
    }

    for frequencies in [vec![2.0, 1.0], vec![1.0, 1.0]] {
        let network = make_network(1, &frequencies, &[50.0]);
        assert!(matches!(
            write_touchstone_v2_0_s_full_ri_hz(&network),
            Err(Error::WriterV2FrequencyNotStrictlyIncreasing { .. })
        ));
    }

    let nonfinite_s = Network::new(
        Frequency::from_hz(vec![1.0]).unwrap(),
        Array3::from_elem((1, 1, 1), Complex64::new(f64::NAN, 0.0)),
        Array2::from_elem((1, 1), Complex64::new(50.0, 0.0)),
    )
    .unwrap();
    assert!(matches!(
        write_touchstone_v2_0_s_full_ri_hz(&nonfinite_s),
        Err(Error::WriterV2NonFiniteS { .. })
    ));

    for value in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(f64::INFINITY, 0.0),
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
            write_touchstone_v2_0_s_full_ri_hz(&network),
            Err(Error::WriterV2InvalidZ0 { .. })
        ));
    }

    // A signed zero imaginary part is numerically real and therefore valid.
    let signed_zero = Network::new(
        Frequency::from_hz(vec![1.0]).unwrap(),
        Array3::zeros((1, 1, 1)),
        Array2::from_elem((1, 1), Complex64::new(50.0, -0.0)),
    )
    .unwrap();
    assert!(write_touchstone_v2_0_s_full_ri_hz(&signed_zero).is_ok());

    // v1 remains deliberately stricter: heterogeneous references are still
    // rejected rather than silently repaired or renormalized.
    let heterogeneous = make_network(2, &[1.0], &[37.0, 83.0]);
    assert!(matches!(
        write_touchstone_v1_0_s_ri_hz(&heterogeneous),
        Err(Error::WriterMismatchedZ0 { .. })
    ));
}

#[test]
fn malformed_serde_values_never_panic() {
    let base = make_network(2, &[1.0], &[37.0, 83.0]);
    let malformed = [
        malformed_network(&base, |value| value["s"]["dim"] = json!([1, 1, 4])),
        malformed_network(&base, |value| value["s"]["dim"] = json!([1, 4, 1])),
        malformed_network(&base, |value| {
            value["s"]["dim"] = json!([0, 2, 2]);
            value["s"]["data"] = json!([]);
        }),
        malformed_network(&base, |value| {
            value["z0"]["dim"] = json!([1, 1]);
            value["z0"]["data"] = json!([[37.0, 0.0]]);
        }),
        malformed_network(&base, |value| value["z0"]["dim"] = json!([2, 1])),
    ];
    for network in malformed {
        let result = catch_unwind(AssertUnwindSafe(|| {
            write_touchstone_v2_0_s_full_ri_hz(&network)
        }));
        assert!(
            result.is_ok(),
            "v2 writer panicked for malformed serde state"
        );
        assert!(result.unwrap().is_err());
    }
}
