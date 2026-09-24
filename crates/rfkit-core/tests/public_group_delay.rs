use approx::assert_relative_eq;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, GroupDelayArithmetic, Network};
use serde_json::{Value, json};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn trace_network(
    frequency_hz: &[f64],
    phases: &[f64],
    amplitudes: &[f64],
    nports: usize,
    port_out: usize,
    port_in: usize,
    z0: Array2<Complex64>,
) -> Network {
    assert_eq!(frequency_hz.len(), phases.len());
    assert_eq!(phases.len(), amplitudes.len());
    let mut s = Array3::from_elem((frequency_hz.len(), nports, nports), c(0.0, 0.0));
    for frequency in 0..frequency_hz.len() {
        s[[frequency, port_out, port_in]] =
            Complex64::from_polar(amplitudes[frequency], phases[frequency]);
    }
    Network::new(Frequency::from_hz(frequency_hz.to_vec()).unwrap(), s, z0).unwrap()
}

fn real_z0(nfreq: usize, nports: usize) -> Array2<Complex64> {
    Array2::from_elem((nfreq, nports), c(50.0, 0.0))
}

fn assert_delays(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_relative_eq!(actual, expected, epsilon = 2.0e-20, max_relative = 2.0e-12);
        assert!(actual.is_finite(), "delay[{index}] must be finite");
    }
}

#[test]
fn one_two_and_n_port_nonuniform_traces_match_analytical_delays() {
    let frequency_hz = [0.0, 11.0e6, 37.0e6, 92.0e6];
    let expected_phase_delay = 2.5e-9;
    let phases: Vec<_> = frequency_hz
        .iter()
        .map(|frequency| 0.37 - std::f64::consts::TAU * expected_phase_delay * frequency)
        .collect();
    let amplitudes = [0.2, 0.4, 0.3, 0.8];

    let one = trace_network(&frequency_hz, &phases, &amplitudes, 1, 0, 0, real_z0(4, 1));
    assert_delays(
        &one.group_delay_secant_power(0, 0).unwrap(),
        &[expected_phase_delay; 3],
    );

    let two = trace_network(&frequency_hz, &phases, &amplitudes, 2, 1, 0, real_z0(4, 2));
    assert_delays(
        &two.group_delay_secant_power(1, 0).unwrap(),
        &[expected_phase_delay; 3],
    );

    let nport = trace_network(&frequency_hz, &phases, &amplitudes, 5, 4, 2, real_z0(4, 5));
    assert_delays(
        &nport.group_delay_secant_power(4, 2).unwrap(),
        &[expected_phase_delay; 3],
    );
}

#[test]
fn positive_negative_and_zero_delays_and_quadratic_phase_are_supported() {
    let frequency_hz = [1.0e6, 9.0e6, 23.0e6, 51.0e6];
    for phase_delay in [3.0e-9, -2.0e-9, 0.0] {
        let phases: Vec<_> = frequency_hz
            .iter()
            .map(|frequency| 0.4 - std::f64::consts::TAU * phase_delay * frequency)
            .collect();
        let source = trace_network(
            &frequency_hz,
            &phases,
            &[1.0, 0.4, 2.0, 0.7],
            2,
            0,
            1,
            real_z0(4, 2),
        );
        assert_delays(
            &source.group_delay_secant_power(0, 1).unwrap(),
            &[phase_delay; 3],
        );
    }

    let quadratic = 0.7e-18;
    let linear = 1.1e-9;
    let phases: Vec<_> = frequency_hz
        .iter()
        .map(|frequency| {
            -0.2 - std::f64::consts::TAU * (linear * frequency + quadratic * frequency * frequency)
        })
        .collect();
    let source = trace_network(
        &frequency_hz,
        &phases,
        &[0.1, 0.3, 0.8, 1.2],
        1,
        0,
        0,
        real_z0(4, 1),
    );
    let expected: Vec<_> = frequency_hz
        .windows(2)
        .map(|window| linear + quadratic * (window[0] + window[1]))
        .collect();
    assert_delays(&source.group_delay_secant_power(0, 0).unwrap(), &expected);
}

#[test]
fn shortest_phase_increments_cover_both_branch_directions_and_alias_witness() {
    let source = trace_network(
        &[0.0, 1.0e9],
        &[2.9, -2.9],
        &[1.0, 2.0],
        1,
        0,
        0,
        real_z0(2, 1),
    );
    // +0.483... rad shortest increment -> negative delay.
    assert!(source.group_delay_secant_power(0, 0).unwrap()[0] < 0.0);

    let reverse = trace_network(
        &[0.0, 1.0e9],
        &[-2.9, 2.9],
        &[1.0, 2.0],
        1,
        0,
        0,
        real_z0(2, 1),
    );
    assert!(reverse.group_delay_secant_power(0, 0).unwrap()[0] > 0.0);

    // A true 1 ns phase slope advances by 2*pi over this aperture.  The
    // sampled secant cannot infer that winding and correctly returns the
    // principal local alias rather than promising automatic undersampling
    // detection.
    let aliased = trace_network(
        &[0.0, 1.0e9, 2.0e9],
        &[0.0, 0.0, 0.0],
        &[1.0, 1.0, 1.0],
        1,
        0,
        0,
        real_z0(3, 1),
    );
    assert_eq!(
        aliased.group_delay_secant_power(0, 0).unwrap(),
        vec![-0.0, -0.0]
    );
}

#[test]
fn complex_frequency_dependent_signed_references_are_valid_and_inputs_immutable() {
    let frequency_hz = [0.0, 7.0e6, 20.0e6];
    let phases = [0.1, -0.2, -0.8];
    let z0 = Array2::from_shape_vec(
        (3, 3),
        vec![
            c(-50.0, 2.0),
            c(61.0, -3.0),
            c(72.0, 1.0),
            c(-52.0, -1.0),
            c(63.0, 4.0),
            c(74.0, -2.0),
            c(-54.0, 3.0),
            c(65.0, -5.0),
            c(76.0, 2.0),
        ],
    )
    .unwrap();
    let source = trace_network(&frequency_hz, &phases, &[1.0, 0.8, 0.4], 3, 2, 1, z0);
    let snapshot = source.clone();
    let result = source.group_delay_secant_power(2, 1).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(source, snapshot);
}

#[test]
fn amplitude_and_constant_phase_offset_invariants_and_port_permutation_covariance() {
    let frequency_hz = [10.0e6, 31.0e6, 80.0e6, 143.0e6];
    let phases: Vec<_> = frequency_hz
        .iter()
        .map(|frequency| -0.8 - std::f64::consts::TAU * 1.3e-9 * frequency)
        .collect();
    let source = trace_network(
        &frequency_hz,
        &phases,
        &[0.1, 0.7, 2.0, 0.03],
        3,
        2,
        1,
        real_z0(4, 3),
    );
    let scaled = trace_network(
        &frequency_hz,
        &phases.iter().map(|phase| phase + 1.2).collect::<Vec<_>>(),
        &[3.0, 0.01, 8.0, 0.5],
        3,
        2,
        1,
        real_z0(4, 3),
    );
    let left = source.group_delay_secant_power(2, 1).unwrap();
    let right = scaled.group_delay_secant_power(2, 1).unwrap();
    assert_delays(&left, &right);

    let permuted = source.permute_ports(&[2, 0, 1]).unwrap();
    let covariance = permuted.group_delay_secant_power(0, 2).unwrap();
    assert_delays(&left, &covariance);
}

#[test]
fn selected_trace_zero_is_rejected_but_unselected_zero_and_singular_s_are_valid() {
    let mut s = Array3::from_elem((3, 2, 2), c(0.0, 0.0));
    for frequency in 0..3 {
        s[[frequency, 0, 0]] = c(1.0 + frequency as f64, 0.0);
        // The selected trace is nonzero while every unselected coordinate is
        // exactly zero, including the singular full S matrix.
    }
    let source = Network::new(
        Frequency::from_hz(vec![0.0, 1.0, 2.0]).unwrap(),
        s,
        real_z0(3, 2),
    )
    .unwrap();
    assert_eq!(
        source.group_delay_secant_power(0, 0).unwrap(),
        vec![-0.0, -0.0]
    );

    let mut zero_selected = source.s().clone();
    zero_selected[[1, 0, 0]] = c(-0.0, 0.0);
    let zero = Network::new(
        Frequency::from_hz(vec![0.0, 1.0, 2.0]).unwrap(),
        zero_selected,
        real_z0(3, 2),
    )
    .unwrap();
    assert_eq!(
        zero.group_delay_secant_power(0, 0).unwrap_err(),
        Error::UndefinedGroupDelayPhase {
            sample: 1,
            port_out: 0,
            port_in: 0,
        }
    );
}

#[test]
fn exact_half_turns_are_rejected_and_nearby_turns_are_accepted() {
    let plus = trace_network(
        &[0.0, 1.0],
        &[0.0, std::f64::consts::PI],
        &[1.0, 1.0],
        1,
        0,
        0,
        real_z0(2, 1),
    );
    assert!(matches!(
        plus.group_delay_secant_power(0, 0),
        Err(Error::AmbiguousGroupDelayHalfTurn { interval: 0, .. })
    ));

    let minus = Network::new(
        Frequency::from_hz(vec![0.0, 1.0]).unwrap(),
        Array3::from_shape_vec((2, 1, 1), vec![c(1.0, 0.0), c(-1.0, -0.0)]).unwrap(),
        real_z0(2, 1),
    )
    .unwrap();
    assert!(matches!(
        minus.group_delay_secant_power(0, 0),
        Err(Error::AmbiguousGroupDelayHalfTurn { interval: 0, .. })
    ));

    let nearby = trace_network(
        &[0.0, 1.0],
        &[0.0, std::f64::consts::PI - 1.0e-12],
        &[1.0, 1.0],
        1,
        0,
        0,
        real_z0(2, 1),
    );
    assert!(nearby.group_delay_secant_power(0, 0).is_ok());
}

#[test]
fn signed_zero_phase_axis_huge_subnormal_and_large_aperture_inputs_are_safe() {
    let signed = Network::new(
        Frequency::from_hz(vec![0.0, 1.0]).unwrap(),
        Array3::from_shape_vec((2, 1, 1), vec![c(-1.0, 0.0), c(-1.0, -0.0)]).unwrap(),
        real_z0(2, 1),
    )
    .unwrap();
    let signed_result = signed.group_delay_secant_power(0, 0).unwrap();
    assert_eq!(signed_result.len(), 1);
    assert!(signed_result[0].is_finite());
    assert_eq!(signed_result[0], 0.0);

    let extreme = Network::new(
        Frequency::from_hz(vec![0.0, 1.0]).unwrap(),
        Array3::from_shape_vec(
            (2, 1, 1),
            vec![
                c(f64::MAX, f64::MAX),
                c(f64::from_bits(1), -f64::from_bits(1)),
            ],
        )
        .unwrap(),
        real_z0(2, 1),
    )
    .unwrap();
    let extreme_result = extreme.group_delay_secant_power(0, 0).unwrap();
    assert!(extreme_result[0].is_finite());

    let large_aperture = Network::new(
        Frequency::from_hz(vec![0.0, f64::MAX]).unwrap(),
        Array3::from_shape_vec((2, 1, 1), vec![c(1.0, 0.0), c(1.0, 1.0)]).unwrap(),
        real_z0(2, 1),
    )
    .unwrap();
    let large = large_aperture.group_delay_secant_power(0, 0).unwrap();
    assert!(large[0].is_finite());
    assert_ne!(large[0], 0.0);

    let tiny = Network::new(
        Frequency::from_hz(vec![0.0, f64::from_bits(1)]).unwrap(),
        Array3::from_shape_vec((2, 1, 1), vec![c(1.0, 0.0), c(1.0, 1.0)]).unwrap(),
        real_z0(2, 1),
    )
    .unwrap();
    assert!(matches!(
        tiny.group_delay_secant_power(0, 0),
        Err(Error::NonFiniteGroupDelayComputation {
            interval: 0,
            port_out: 0,
            port_in: 0,
            stage: GroupDelayArithmetic::DelayOutput,
        })
    ));
}

#[test]
fn subnormal_apertures_preserve_finite_signed_one_over_tau_delays() {
    let minimum_subnormal = f64::from_bits(1);
    let apertures = [minimum_subnormal, 2.0 * minimum_subnormal];
    let expected_magnitude = 1.0 / std::f64::consts::TAU;

    for &aperture in &apertures {
        for &phase_sign in &[-1.0, 1.0] {
            let source = Network::new(
                Frequency::from_hz(vec![0.0, aperture]).unwrap(),
                Array3::from_shape_vec((2, 1, 1), vec![c(1.0, 0.0), c(1.0, phase_sign * aperture)])
                    .unwrap(),
                real_z0(2, 1),
            )
            .unwrap();

            let actual = source.group_delay_secant_power(0, 0).unwrap()[0];
            // The phase/aperture ratio is exactly +/-1.  The remaining
            // division by TAU is one correctly rounded binary64 operation,
            // so a two-epsilon relative bound is appropriate here.
            assert_relative_eq!(
                actual,
                -phase_sign * expected_magnitude,
                epsilon = 0.0,
                max_relative = 2.0 * f64::EPSILON
            );
        }
    }
}

#[test]
fn malformed_serde_networks_return_structured_errors_without_panics() {
    fn base() -> Value {
        serde_json::to_value(
            Network::new(
                Frequency::from_hz(vec![1.0, 2.0]).unwrap(),
                Array3::from_shape_vec((2, 2, 2), vec![c(0.1, 0.0); 8]).unwrap(),
                real_z0(2, 2),
            )
            .unwrap(),
        )
        .unwrap()
    }
    fn deserialize(value: Value) -> Network {
        serde_json::from_value(value).unwrap()
    }
    fn set_shape(value: &mut Value, field: &str, shape: &[usize], count: usize) {
        value[field]["dim"] = json!(shape);
        let original = value[field]["data"].as_array().unwrap().clone();
        value[field]["data"] = json!(
            (0..count)
                .map(|index| original[index % original.len()].clone())
                .collect::<Vec<_>>()
        );
    }
    fn assert_error(network: Network, expected: Error) {
        let result = catch_unwind(AssertUnwindSafe(|| network.group_delay_secant_power(0, 0)));
        assert!(
            result.is_ok(),
            "group delay panicked on malformed serde network"
        );
        assert_eq!(result.unwrap().unwrap_err(), expected);
    }

    let mut value = base();
    value["frequency"]["hz"] = json!([1.0]);
    assert_error(
        deserialize(value),
        Error::GroupDelayTooFewFrequencySamples { actual: 1 },
    );

    let mut value = base();
    value["frequency"]["hz"] = json!([1.0, 2.0, 3.0]);
    assert_error(
        deserialize(value),
        Error::GroupDelayFrequencyLengthMismatch {
            expected: 3,
            actual: 2,
        },
    );

    let mut value = base();
    set_shape(&mut value, "s", &[2, 2, 3], 12);
    assert_error(
        deserialize(value),
        Error::InvalidGroupDelaySShape {
            shape: vec![2, 2, 3],
        },
    );

    let mut value = base();
    set_shape(&mut value, "z0", &[2, 3], 6);
    assert_error(
        deserialize(value),
        Error::InvalidGroupDelayZ0Shape { shape: vec![2, 3] },
    );

    let mut value = base();
    set_shape(&mut value, "s", &[2, 0, 0], 0);
    assert_error(
        deserialize(value),
        Error::InvalidGroupDelaySShape {
            shape: vec![2, 0, 0],
        },
    );

    let source = deserialize(base());
    assert_eq!(
        source.group_delay_secant_power(2, 0).unwrap_err(),
        Error::InvalidGroupDelayOutputPort { port: 2, nports: 2 }
    );
    assert_eq!(
        source.group_delay_secant_power(0, 2).unwrap_err(),
        Error::InvalidGroupDelayInputPort { port: 2, nports: 2 }
    );
}

#[test]
fn grid_reference_and_nonfinite_validation_is_explicit() {
    fn network_with(frequency: Vec<f64>, s: Array3<Complex64>, z0: Array2<Complex64>) -> Network {
        Network::new(Frequency::from_hz(frequency).unwrap(), s, z0).unwrap()
    }
    let valid_s =
        Array3::from_shape_vec((3, 1, 1), vec![c(1.0, 0.0), c(0.9, 0.1), c(0.8, 0.2)]).unwrap();
    let valid_z0 = real_z0(3, 1);

    for (frequency, expected) in [
        (
            vec![-1.0, 0.0, 1.0],
            Error::NegativeGroupDelayFrequency {
                index: 0,
                value: -1.0,
            },
        ),
        (
            vec![0.0, 0.0, 1.0],
            Error::GroupDelayFrequencyNotStrictlyIncreasing {
                index: 1,
                previous: 0.0,
                current: 0.0,
            },
        ),
        (
            vec![0.0, 2.0, 1.0],
            Error::GroupDelayFrequencyNotStrictlyIncreasing {
                index: 2,
                previous: 2.0,
                current: 1.0,
            },
        ),
        (
            vec![0.0, f64::NAN, 1.0],
            Error::NonFiniteGroupDelayFrequency {
                index: 1,
                value: f64::NAN,
            },
        ),
    ] {
        let result = network_with(frequency, valid_s.clone(), valid_z0.clone())
            .group_delay_secant_power(0, 0)
            .unwrap_err();
        match expected {
            Error::NonFiniteGroupDelayFrequency { index, .. } => assert!(matches!(
                result,
                Error::NonFiniteGroupDelayFrequency { index: actual, .. } if actual == index
            )),
            expected => assert_eq!(result, expected),
        }
    }

    let mut nonfinite_s = valid_s.clone();
    nonfinite_s[[1, 0, 0]] = c(f64::INFINITY, 0.0);
    assert_eq!(
        network_with(vec![0.0, 1.0, 2.0], nonfinite_s, valid_z0.clone())
            .group_delay_secant_power(0, 0)
            .unwrap_err(),
        Error::NonFiniteGroupDelayS {
            frequency: 1,
            row: 0,
            column: 0,
        }
    );

    let mut nonfinite_z0 = valid_z0.clone();
    nonfinite_z0[[2, 0]] = c(50.0, f64::NAN);
    assert_eq!(
        network_with(vec![0.0, 1.0, 2.0], valid_s.clone(), nonfinite_z0)
            .group_delay_secant_power(0, 0)
            .unwrap_err(),
        Error::NonFiniteGroupDelayZ0 {
            frequency: 2,
            port: 0,
        }
    );

    for value in [c(0.0, 0.0), c(-0.0, 3.0)] {
        let mut zero_real = valid_z0.clone();
        zero_real[[0, 0]] = value;
        assert_eq!(
            network_with(vec![0.0, 1.0, 2.0], valid_s.clone(), zero_real)
                .group_delay_secant_power(0, 0)
                .unwrap_err(),
            Error::ZeroRealGroupDelayReferenceImpedance {
                frequency: 0,
                port: 0,
            }
        );
    }
}
