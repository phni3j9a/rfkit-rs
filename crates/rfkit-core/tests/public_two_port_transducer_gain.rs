use ndarray::{Array2, Array3, ShapeBuilder};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, Network, TwoPortTransducerGainArithmetic};
use serde_json::json;
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

fn assert_close(actual: f64, expected: f64) {
    let bound = 2.0e-12 + 2.0e-12 * expected.abs();
    assert!(
        (actual - expected).abs() <= bound,
        "actual={actual:?} expected={expected:?}"
    );
}

fn solve_two_by_two(matrix: [[Complex64; 2]; 2], rhs: [Complex64; 2]) -> [Complex64; 2] {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    [
        (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
        (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
    ]
}

fn physical_gain_from_z(
    z: [[Complex64; 2]; 2],
    source: Complex64,
    load: Complex64,
    drive: Complex64,
) -> f64 {
    let currents = solve_two_by_two(
        [[z[0][0] + source, z[0][1]], [z[1][0], z[1][1] + load]],
        [drive, c(0.0, 0.0)],
    );
    let output_power = load.re * currents[1].norm_sqr();
    let available_power = drive.norm_sqr() / (4.0 * source.re);
    output_power / available_power
}

fn two_port_z_from_s(s: [[Complex64; 2]; 2], z0: [Complex64; 2]) -> [[Complex64; 2]; 2] {
    let normalization = [1.0 / (2.0 * z0[0].re.sqrt()), 1.0 / (2.0 * z0[1].re.sqrt())];
    let mut a = [[c(0.0, 0.0); 2]; 2];
    let mut b = [[c(0.0, 0.0); 2]; 2];
    for row in 0..2 {
        for column in 0..2 {
            let identity = if row == column {
                c(1.0, 0.0)
            } else {
                c(0.0, 0.0)
            };
            a[row][column] = (identity - s[row][column]) * normalization[column];
            let mut rhs = s[row][column] * z0[column];
            if row == column {
                rhs += z0[row].conj();
            }
            b[row][column] = rhs * normalization[column];
        }
    }
    let mut output = [[c(0.0, 0.0); 2]; 2];
    for column in 0..2 {
        let result = solve_two_by_two(a, [b[0][column], b[1][column]]);
        output[0][column] = result[0];
        output[1][column] = result[1];
    }
    output
}

#[test]
fn ideal_through_matches_analytical_complex_source_and_load_gain() {
    let source = [c(35.0, 8.0), c(62.0, -5.0), c(48.0, 11.0)];
    let load = [c(73.0, -11.0), c(41.0, 9.0), c(96.0, -4.0)];
    let network = network(
        vec![-3.0, 1.0, 1.0],
        vec![
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
        ],
        vec![c(50.0, 0.0); 6],
    );
    let actual = network
        .two_port_transducer_gain_power(&source, &load)
        .unwrap();
    for index in 0..3 {
        let expected =
            4.0 * source[index].re * load[index].re / (source[index] + load[index]).norm_sqr();
        assert_close(actual[index], expected);
    }
}

#[test]
fn independent_thevenin_load_solve_is_drive_scale_invariant() {
    let s = [c(0.11, 0.02), c(0.37, -0.04), c(0.19, 0.06), c(-0.08, 0.03)];
    let z0 = [c(47.0, 4.0), c(83.0, -6.0)];
    let source = c(31.0, 7.0);
    let load = c(68.0, -9.0);
    let z = two_port_z_from_s([[s[0], s[1]], [s[2], s[3]]], z0);
    let gain_one = physical_gain_from_z(z, source, load, c(1.0, 0.0));
    let gain_scaled = physical_gain_from_z(z, source, load, c(-4.0, 3.0));
    assert_close(gain_one, gain_scaled);

    let source_network = network(vec![1.0], s.to_vec(), z0.to_vec());
    let actual = source_network
        .two_port_transducer_gain_power(&[source], &[load])
        .unwrap();
    assert_close(actual[0], gain_one);
}

#[test]
fn stored_reference_change_and_port_exchange_preserve_physical_result() {
    let s = vec![
        c(0.12, 0.03),
        c(0.41, -0.05),
        c(0.27, 0.08),
        c(-0.09, 0.04),
        c(0.15, -0.01),
        c(0.33, 0.06),
        c(0.19, -0.07),
        c(-0.04, -0.02),
    ];
    let stored = vec![c(43.0, 5.0), c(71.0, -4.0), c(48.0, 7.0), c(86.0, 3.0)];
    let source = vec![c(34.0, 8.0), c(56.0, -2.0)];
    let load = vec![c(79.0, -6.0), c(68.0, 9.0)];
    let original = network(vec![1.0, 2.0], s.clone(), stored);
    let expected = original
        .two_port_transducer_gain_power(&source, &load)
        .unwrap();
    let changed = original
        .renormalize_direct_power(
            Array2::from_shape_vec(
                (2, 2),
                vec![c(63.0, -3.0), c(92.0, 7.0), c(52.0, 2.0), c(74.0, -8.0)],
            )
            .unwrap(),
        )
        .unwrap();
    let changed_gain = changed
        .two_port_transducer_gain_power(&source, &load)
        .unwrap();
    for (left, right) in expected.iter().zip(changed_gain.iter()) {
        assert_close(*left, *right);
    }

    let exchanged = changed.select_ports_zero_incident(&[1, 0]).unwrap();
    let forward = changed
        .two_port_transducer_gain_power(&source, &load)
        .unwrap();
    let reverse = exchanged
        .two_port_transducer_gain_power(&load, &source)
        .unwrap();
    for frequency in 0..2 {
        let original_s = [
            [
                changed.s()[[frequency, 0, 0]],
                changed.s()[[frequency, 0, 1]],
            ],
            [
                changed.s()[[frequency, 1, 0]],
                changed.s()[[frequency, 1, 1]],
            ],
        ];
        let original_z0 = [changed.z0()[[frequency, 0]], changed.z0()[[frequency, 1]]];
        let forward_expected = physical_gain_from_z(
            two_port_z_from_s(original_s, original_z0),
            source[frequency],
            load[frequency],
            c(1.0, 0.0),
        );
        let exchanged_s = [
            [
                exchanged.s()[[frequency, 0, 0]],
                exchanged.s()[[frequency, 0, 1]],
            ],
            [
                exchanged.s()[[frequency, 1, 0]],
                exchanged.s()[[frequency, 1, 1]],
            ],
        ];
        let exchanged_z0 = [
            exchanged.z0()[[frequency, 0]],
            exchanged.z0()[[frequency, 1]],
        ];
        let reverse_expected = physical_gain_from_z(
            two_port_z_from_s(exchanged_s, exchanged_z0),
            load[frequency],
            source[frequency],
            c(1.0, 0.0),
        );
        assert_close(forward[frequency], forward_expected);
        assert_close(reverse[frequency], reverse_expected);
    }
    assert!(
        forward
            .iter()
            .zip(reverse.iter())
            .any(|(left, right)| (left - right).abs() > 1.0e-6),
        "non-reciprocal forward and reverse gains should not be assumed equal"
    );
}

#[test]
fn singular_full_s_still_has_a_well_defined_loaded_response() {
    let source = network(
        vec![1.0],
        vec![c(0.1, 0.0), c(0.2, 0.0), c(0.3, 0.0), c(0.6, 0.0)],
        vec![c(50.0, 0.0), c(50.0, 0.0)],
    );
    let result = source
        .two_port_transducer_gain_power(&[c(37.0, 4.0)], &[c(81.0, -7.0)])
        .unwrap();
    assert!(result[0].is_finite());
    assert!(result[0] >= 0.0);
}

#[test]
fn exact_and_near_singular_wave_change_are_distinguished() {
    let exact = network(
        vec![1.0],
        vec![c(-3.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        vec![c(50.0, 0.0), c(50.0, 0.0)],
    );
    assert!(matches!(
        exact.two_port_transducer_gain_power(&[c(25.0, 0.0)], &[c(50.0, 0.0)]),
        Err(Error::SingularTwoPortTransducerGainPowerWaveChange { frequency: 0, .. })
    ));

    let near = network(
        vec![1.0],
        vec![
            c(-3.0 + 2.0_f64.powi(-20), 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
        ],
        vec![c(50.0, 0.0), c(50.0, 0.0)],
    );
    let result = near
        .two_port_transducer_gain_power(&[c(25.0, 0.0)], &[c(50.0, 0.0)])
        .unwrap();
    assert!(result[0].is_finite());
}

#[test]
fn passive_network_has_nonnegative_gain_at_most_one() {
    let passive = network(
        vec![1.0, 2.0],
        vec![
            c(0.0, 0.0),
            c(0.4, 0.0),
            c(0.4, 0.0),
            c(0.0, 0.0),
            c(0.05, 0.01),
            c(0.3, -0.02),
            c(0.3, -0.02),
            c(0.04, 0.01),
        ],
        vec![c(50.0, 0.0); 4],
    );
    let result = passive
        .two_port_transducer_gain_power(
            &[c(37.0, 5.0), c(73.0, -4.0)],
            &[c(81.0, -9.0), c(44.0, 7.0)],
        )
        .unwrap();
    assert!(result.iter().all(|value| *value >= 0.0 && *value <= 1.0));
}

#[test]
fn finite_nonreciprocal_active_network_is_not_clipped_or_classified() {
    let active = network(
        vec![1.0],
        vec![c(0.0, 0.0), c(0.0, 0.0), c(2.0, 0.0), c(0.0, 0.0)],
        vec![c(50.0, 0.0); 2],
    );
    let gain = active
        .two_port_transducer_gain_power(&[c(50.0, 0.0)], &[c(50.0, 0.0)])
        .unwrap();
    assert_eq!(gain, vec![4.0]);
}

#[test]
fn validation_and_nonstandard_layouts_are_structured_and_nonmutating() {
    let frequency = vec![1.0, 2.0];
    let s = Array3::from_shape_fn((2, 2, 2).f(), |(f, row, column)| {
        c(0.1 * (f + row + column) as f64, -0.02 * column as f64)
    });
    let z0 = Array2::from_shape_fn((2, 2).f(), |(f, port)| {
        c(45.0 + 3.0 * f as f64 + port as f64, 2.0 - port as f64)
    });
    let source = Network::new(Frequency::from_hz(frequency.clone()).unwrap(), s, z0).unwrap();
    let snapshot = source.clone();
    let result = source
        .two_port_transducer_gain_power(
            &[c(34.0, 6.0), c(57.0, -3.0)],
            &[c(77.0, -4.0), c(83.0, 9.0)],
        )
        .unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(source, snapshot);

    assert!(matches!(
        source.two_port_transducer_gain_power(&[c(50.0, 0.0)], &[c(50.0, 0.0)]),
        Err(
            Error::TwoPortTransducerGainPowerSourceImpedanceLengthMismatch {
                expected: 2,
                actual: 1,
            }
        )
    ));
    assert!(matches!(
        source.two_port_transducer_gain_power(&[c(50.0, 0.0), c(50.0, 0.0)], &[c(50.0, 0.0)],),
        Err(
            Error::TwoPortTransducerGainPowerLoadImpedanceLengthMismatch {
                expected: 2,
                actual: 1,
            }
        )
    ));
    assert!(matches!(
        source.two_port_transducer_gain_power(
            &[c(0.0, 1.0), c(50.0, 0.0)],
            &[c(50.0, 0.0), c(50.0, 0.0)],
        ),
        Err(Error::NonPositiveRealTwoPortTransducerGainPowerSourceImpedance { frequency: 0, .. })
    ));
    assert!(matches!(
        source.two_port_transducer_gain_power(
            &[c(50.0, 0.0), c(50.0, 0.0)],
            &[c(-1.0, 0.0), c(50.0, 0.0)],
        ),
        Err(Error::NonPositiveRealTwoPortTransducerGainPowerLoadImpedance { frequency: 0, .. })
    ));

    let mut malformed = serde_json::to_value(&source).unwrap();
    malformed["frequency"]["hz"] = json!([]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        malformed.two_port_transducer_gain_power(&[], &[])
    }));
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().unwrap_err(),
        Error::EmptyTwoPortTransducerGainPowerFrequency
    );

    let mut malformed = serde_json::to_value(&source).unwrap();
    malformed["frequency"]["hz"] = json!([1.0, 2.0, 3.0]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    assert_eq!(
        malformed
            .two_port_transducer_gain_power(&[c(50.0, 0.0); 3], &[c(50.0, 0.0); 3])
            .unwrap_err(),
        Error::TwoPortTransducerGainPowerFrequencyLengthMismatch {
            expected: 3,
            actual: 2,
        }
    );

    let mut malformed = serde_json::to_value(&source).unwrap();
    malformed["s"]["dim"] = json!([2, 1, 2]);
    let scalar = malformed["s"]["data"][0].clone();
    malformed["s"]["data"] = json!([scalar.clone(), scalar.clone(), scalar.clone(), scalar]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    assert!(matches!(
        malformed.two_port_transducer_gain_power(&[c(50.0, 0.0); 2], &[c(50.0, 0.0); 2]),
        Err(Error::InvalidTwoPortTransducerGainPowerSShape { .. })
    ));
}

#[test]
fn every_network_and_table_validation_branch_has_context() {
    let source_impedance = [c(50.0, 0.0)];
    let load_impedance = [c(50.0, 0.0)];
    let valid_s = vec![c(0.0, 0.0); 4];
    let valid_z0 = vec![c(50.0, 0.0); 2];

    for nports in [1, 3] {
        let wrong_ports = Network::new(
            Frequency::from_hz(vec![1.0]).unwrap(),
            Array3::from_elem((1, nports, nports), c(0.0, 0.0)),
            Array2::from_elem((1, nports), c(50.0, 0.0)),
        )
        .unwrap();
        assert!(matches!(
            wrong_ports.two_port_transducer_gain_power(&source_impedance, &load_impedance),
            Err(Error::InvalidTwoPortTransducerGainPowerSShape { shape })
                if shape == vec![1, nports, nports]
        ));
    }

    let valid = network(vec![1.0], valid_s.clone(), valid_z0.clone());
    let mut malformed = serde_json::to_value(&valid).unwrap();
    malformed["z0"]["dim"] = json!([1, 1]);
    malformed["z0"]["data"] = json!([malformed["z0"]["data"][0].clone()]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    assert!(matches!(
        malformed.two_port_transducer_gain_power(&source_impedance, &load_impedance),
        Err(Error::InvalidTwoPortTransducerGainPowerZ0Shape { shape })
            if shape == vec![1, 1]
    ));

    let nonfinite_frequency = Network::new(
        Frequency::from_hz(vec![f64::NAN]).unwrap(),
        Array3::from_shape_vec((1, 2, 2), valid_s.clone()).unwrap(),
        Array2::from_shape_vec((1, 2), valid_z0.clone()).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        nonfinite_frequency.two_port_transducer_gain_power(&source_impedance, &load_impedance),
        Err(Error::NonFiniteTwoPortTransducerGainPowerFrequency { index: 0, .. })
    ));

    let mut nonfinite_s = Array3::from_shape_vec((1, 2, 2), valid_s.clone()).unwrap();
    nonfinite_s[[0, 1, 0]] = c(f64::NAN, 0.0);
    let nonfinite_s = Network::new(
        Frequency::from_hz(vec![1.0]).unwrap(),
        nonfinite_s,
        Array2::from_shape_vec((1, 2), valid_z0.clone()).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        nonfinite_s.two_port_transducer_gain_power(&source_impedance, &load_impedance),
        Err(Error::NonFiniteTwoPortTransducerGainPowerS {
            frequency: 0,
            row: 1,
            column: 0,
        })
    ));

    for stored_reference in [c(0.0, 2.0), c(-1.0, 2.0)] {
        let invalid_z0 = Network::new(
            Frequency::from_hz(vec![1.0]).unwrap(),
            Array3::from_shape_vec((1, 2, 2), valid_s.clone()).unwrap(),
            Array2::from_shape_vec((1, 2), vec![stored_reference, c(50.0, 0.0)]).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            invalid_z0.two_port_transducer_gain_power(&source_impedance, &load_impedance),
            Err(Error::NonPositiveRealTwoPortTransducerGainPowerZ0 {
                frequency: 0,
                port: 0,
                ..
            })
        ));
    }

    let nonfinite_z0 = Network::new(
        Frequency::from_hz(vec![1.0]).unwrap(),
        Array3::from_shape_vec((1, 2, 2), valid_s.clone()).unwrap(),
        Array2::from_shape_vec((1, 2), vec![c(f64::INFINITY, 0.0), c(50.0, 0.0)]).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        nonfinite_z0.two_port_transducer_gain_power(&source_impedance, &load_impedance),
        Err(Error::NonFiniteTwoPortTransducerGainPowerZ0 {
            frequency: 0,
            port: 0,
        })
    ));

    let valid_network = network(vec![1.0], valid_s, valid_z0);
    for invalid_source in [c(0.0, 1.0), c(-1.0, 1.0)] {
        assert!(matches!(
            valid_network.two_port_transducer_gain_power(&[invalid_source], &load_impedance),
            Err(
                Error::NonPositiveRealTwoPortTransducerGainPowerSourceImpedance {
                    frequency: 0,
                    ..
                }
            )
        ));
    }
    for invalid_load in [c(0.0, 1.0), c(-1.0, 1.0)] {
        assert!(matches!(
            valid_network.two_port_transducer_gain_power(&source_impedance, &[invalid_load]),
            Err(Error::NonPositiveRealTwoPortTransducerGainPowerLoadImpedance { frequency: 0, .. })
        ));
    }
    assert!(matches!(
        valid_network.two_port_transducer_gain_power(&[c(f64::NAN, 0.0)], &load_impedance),
        Err(Error::NonFiniteTwoPortTransducerGainPowerSourceImpedance { frequency: 0 })
    ));
    assert!(matches!(
        valid_network.two_port_transducer_gain_power(&source_impedance, &[c(50.0, f64::INFINITY)]),
        Err(Error::NonFiniteTwoPortTransducerGainPowerLoadImpedance { frequency: 0 })
    ));
}

#[test]
fn output_zero_overflow_underflow_and_positive_subnormal_follow_exact_policy() {
    let zero = network(vec![1.0], vec![c(0.0, 0.0); 4], vec![c(50.0, 0.0); 2]);
    assert_eq!(
        zero.two_port_transducer_gain_power(&[c(50.0, 0.0)], &[c(50.0, 0.0)])
            .unwrap(),
        vec![0.0]
    );

    let overflow = network(
        vec![1.0],
        vec![c(0.0, 0.0), c(0.0, 0.0), c(f64::MAX, 0.0), c(0.0, 0.0)],
        vec![c(50.0, 0.0); 2],
    );
    assert!(matches!(
        overflow.two_port_transducer_gain_power(&[c(50.0, 0.0)], &[c(50.0, 0.0)]),
        Err(Error::NonFiniteTwoPortTransducerGainPowerComputation {
            frequency: 0,
            stage: TwoPortTransducerGainArithmetic::PowerRatio,
        })
    ));

    let underflow = network(
        vec![1.0],
        vec![
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(f64::from_bits(1), 0.0),
            c(0.0, 0.0),
        ],
        vec![c(50.0, 0.0); 2],
    );
    assert!(matches!(
        underflow.two_port_transducer_gain_power(&[c(50.0, 0.0)], &[c(50.0, 0.0)]),
        Err(Error::NonFiniteTwoPortTransducerGainPowerComputation {
            frequency: 0,
            stage: TwoPortTransducerGainArithmetic::PowerRatio,
        })
    ));

    let subnormal = network(
        vec![1.0],
        vec![
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(2.0_f64.powi(-537), 0.0),
            c(0.0, 0.0),
        ],
        vec![c(50.0, 0.0); 2],
    );
    let gain = subnormal
        .two_port_transducer_gain_power(&[c(50.0, 0.0)], &[c(50.0, 0.0)])
        .unwrap()[0];
    assert_eq!(gain, f64::from_bits(1));
}
