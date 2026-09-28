use approx::assert_relative_eq;
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, Network};
use serde_json::json;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn network(
    frequency_hz: Vec<f64>,
    nports: usize,
    values: Vec<Complex64>,
    z0: Vec<Complex64>,
) -> Network {
    let nfreq = frequency_hz.len();
    Network::new(
        Frequency::from_hz(frequency_hz).unwrap(),
        Array3::from_shape_vec((nfreq, nports, nports), values).unwrap(),
        Array2::from_shape_vec((nfreq, nports), z0).unwrap(),
    )
    .unwrap()
}

fn real_z0(nfreq: usize, nports: usize) -> Vec<Complex64> {
    vec![c(50.0, 0.0); nfreq * nports]
}

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_relative_eq!(actual.re, expected.re, epsilon = 1.0e-12);
    assert_relative_eq!(actual.im, expected.im, epsilon = 1.0e-12);
}

fn assert_some_close(
    actual: &Array2<Option<Complex64>>,
    frequency: usize,
    port: usize,
    expected: Complex64,
) {
    let value = actual[[frequency, port]].expect("active reflection should be defined");
    assert_complex_close(value, expected);
}

fn expected_ratio(
    network: &Network,
    incident: &Array2<Complex64>,
    frequency: usize,
    port: usize,
) -> Complex64 {
    let mut response = c(0.0, 0.0);
    for input_port in 0..network.nports() {
        response += network.s()[[frequency, port, input_port]] * incident[[frequency, input_port]];
    }
    response / incident[[frequency, port]]
}

#[test]
fn analytic_identities_preserve_sample_and_port_order() {
    let source = network(
        vec![3.0, -0.0, 7.0],
        1,
        vec![c(-0.3, 0.4), c(0.0, 0.0), c(1.2, -1.6)],
        vec![c(41.0, 2.0), c(42.0, -1.0), c(43.0, 3.0)],
    );
    let incident =
        Array2::from_shape_vec((3, 1), vec![c(0.2, -0.1), c(0.0, 0.0), c(-0.7, 0.4)]).unwrap();
    let source_snapshot = source.clone();
    let incident_snapshot = incident.clone();
    let actual = source.active_reflection_power(&incident).unwrap();

    assert_some_close(&actual, 0, 0, c(-0.3, 0.4));
    assert_eq!(actual[[1, 0]], None);
    assert_some_close(&actual, 2, 0, c(1.2, -1.6));
    assert_eq!(source, source_snapshot);
    assert_eq!(incident, incident_snapshot);
    assert_eq!(source.frequency().hz()[1].to_bits(), (-0.0_f64).to_bits());

    let diagonal = network(
        vec![1.0, 2.0],
        3,
        vec![
            c(0.2, 0.1),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(-0.4, 0.3),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(1.1, -0.2),
            c(0.7, -0.1),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(-0.2, -0.5),
        ],
        real_z0(2, 3),
    );
    let diagonal_incident = Array2::from_shape_vec(
        (2, 3),
        vec![
            c(0.1, 0.2),
            c(-0.4, 0.5),
            c(0.7, -0.6),
            c(-0.3, 0.8),
            c(0.2, 0.9),
            c(-0.5, -0.1),
        ],
    )
    .unwrap();
    let diagonal_actual = diagonal
        .active_reflection_power(&diagonal_incident)
        .unwrap();
    for frequency in 0..2 {
        for port in 0..3 {
            assert_some_close(
                &diagonal_actual,
                frequency,
                port,
                diagonal.s()[[frequency, port, port]],
            );
        }
    }
}

#[test]
fn coherent_n_port_rows_use_scattering_axes_and_superposition() {
    let source = network(
        vec![1.0, 2.0],
        3,
        vec![
            c(0.2, 0.1),
            c(0.4, -0.2),
            c(-0.1, 0.3),
            c(-0.3, 0.2),
            c(0.1, -0.4),
            c(0.2, 0.05),
            c(0.05, -0.1),
            c(0.3, 0.2),
            c(-0.2, 0.15),
            c(-0.4, 0.05),
            c(0.2, 0.1),
            c(0.25, -0.1),
            c(0.3, 0.0),
            c(-0.1, 0.4),
            c(0.2, -0.2),
            c(0.1, 0.1),
            c(-0.3, 0.2),
            c(0.15, -0.05),
        ],
        vec![
            c(47.0, 8.0),
            c(63.0, -5.0),
            c(81.0, 2.0),
            c(49.0, -3.0),
            c(68.0, 4.0),
            c(77.0, -6.0),
        ],
    );
    let incident = Array2::from_shape_vec(
        (2, 3),
        vec![
            c(0.3, -0.2),
            c(-0.7, 0.5),
            c(0.4, 0.9),
            c(-0.1, 0.8),
            c(0.6, -0.4),
            c(-0.5, 0.2),
        ],
    )
    .unwrap();
    let actual = source.active_reflection_power(&incident).unwrap();
    for frequency in 0..2 {
        for output_port in 0..3 {
            assert_some_close(
                &actual,
                frequency,
                output_port,
                expected_ratio(&source, &incident, frequency, output_port),
            );
        }
    }

    let first_drive =
        Array2::from_shape_fn((2, 3), |(frequency, port)| incident[[frequency, port]]);
    let second_drive = Array2::from_shape_fn((2, 3), |(frequency, port)| {
        c(
            (frequency + 2 * port + 1) as f64 / 7.0,
            -((frequency + port + 1) as f64) / 11.0,
        )
    });
    let combined_drive = &first_drive + &second_drive;
    let first_response = source.active_reflection_power(&first_drive).unwrap();
    let second_response = source.active_reflection_power(&second_drive).unwrap();
    let combined_response = source.active_reflection_power(&combined_drive).unwrap();
    for frequency in 0..2 {
        for output_port in 0..3 {
            let first_b = first_response[[frequency, output_port]].unwrap()
                * first_drive[[frequency, output_port]];
            let second_b = second_response[[frequency, output_port]].unwrap()
                * second_drive[[frequency, output_port]];
            let combined_b = combined_response[[frequency, output_port]].unwrap()
                * combined_drive[[frequency, output_port]];
            assert_complex_close(combined_b, first_b + second_b);
        }
    }
}

#[test]
fn exact_zero_signed_zero_and_undriven_coupled_ports_return_none() {
    let source = network(
        vec![1.0, 2.0],
        3,
        vec![
            c(0.2, 0.0),
            c(0.4, 0.0),
            c(0.0, 0.0),
            c(0.1, 0.0),
            c(0.3, 0.0),
            c(0.5, 0.0),
            c(0.7, 0.0),
            c(0.0, 0.0),
            c(-0.2, 0.0),
            c(0.2, 0.0),
            c(0.4, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.8, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            c(0.5, 0.0),
            c(0.1, 0.0),
        ],
        real_z0(2, 3),
    );
    let incident = Array2::from_shape_vec(
        (2, 3),
        vec![
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(-0.0, 0.0),
            c(0.0, 0.0),
            c(0.0, -0.0),
            c(0.0, 0.0),
        ],
    )
    .unwrap();
    let actual = source.active_reflection_power(&incident).unwrap();

    assert_some_close(&actual, 0, 0, c(0.2, 0.0));
    assert_eq!(actual[[0, 1]], None);
    assert_eq!(actual[[0, 2]], None);
    assert_eq!(actual.row(1).to_vec(), vec![None, None, None]);

    let nonzero_outgoing = network(
        vec![1.0],
        2,
        vec![c(0.0, 0.0), c(2.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        real_z0(1, 2),
    );
    let output = nonzero_outgoing
        .active_reflection_power(
            &Array2::from_shape_vec((1, 2), vec![c(0.0, 0.0), c(1.0, 0.0)]).unwrap(),
        )
        .unwrap();
    assert_eq!(output[[0, 0]], None);
    assert_eq!(output[[0, 1]], Some(c(0.0, 0.0)));
}

#[test]
fn finite_frequency_labels_are_pointwise_and_opaque() {
    let source = network(
        vec![-2.0, 5.0, 5.0, -0.0, 1.0],
        1,
        vec![c(0.5, 0.0); 5],
        real_z0(5, 1),
    );
    let incident = Array2::from_shape_vec((5, 1), vec![c(1.0, 0.0); 5]).unwrap();
    let actual = source.active_reflection_power(&incident).unwrap();

    assert_eq!(source.frequency().hz()[0], -2.0);
    assert_eq!(source.frequency().hz()[1], source.frequency().hz()[2]);
    assert_eq!(source.frequency().hz()[3].to_bits(), (-0.0_f64).to_bits());
    for frequency in 0..5 {
        assert_eq!(actual[[frequency, 0]], Some(c(0.5, 0.0)));
    }
}

#[test]
fn complex_reference_v_i_reconstruction_and_common_scale_invariance() {
    let source = network(
        vec![1.0, 2.0],
        2,
        vec![
            c(0.2, -0.1),
            c(0.4, 0.3),
            c(-0.25, 0.2),
            c(0.1, -0.35),
            c(-0.1, 0.2),
            c(0.3, -0.2),
            c(0.15, 0.25),
            c(-0.2, 0.1),
        ],
        vec![c(47.0, 8.0), c(63.0, -5.0), c(51.0, -4.0), c(72.0, 6.0)],
    );
    let incident = Array2::from_shape_vec(
        (2, 2),
        vec![c(0.3, -0.2), c(-0.7, 0.5), c(-0.1, 0.8), c(0.6, -0.4)],
    )
    .unwrap();
    let base = source.active_reflection_power(&incident).unwrap();

    for scale in [c(1.0e150, 0.0), c(1.0e-150, 0.0), c(0.4, -0.7)] {
        let scaled_incident = incident.mapv(|value| value * scale);
        let scaled = source.active_reflection_power(&scaled_incident).unwrap();
        for frequency in 0..2 {
            for port in 0..2 {
                assert_some_close(&scaled, frequency, port, base[[frequency, port]].unwrap());
            }
        }
    }

    for frequency in 0..2 {
        let mut outgoing = [c(0.0, 0.0); 2];
        for (row, outgoing_value) in outgoing.iter_mut().enumerate() {
            for column in 0..2 {
                *outgoing_value +=
                    source.s()[[frequency, row, column]] * incident[[frequency, column]];
            }
        }
        for port in 0..2 {
            let reference = source.z0()[[frequency, port]];
            let normalization = reference.re.sqrt();
            let current = (incident[[frequency, port]] - outgoing[port]) / normalization;
            let voltage =
                c(2.0 * normalization, 0.0) * incident[[frequency, port]] - reference * current;
            let reconstructed_incident =
                (voltage + reference * current) / c(2.0 * normalization, 0.0);
            let reconstructed_outgoing =
                (voltage - reference.conj() * current) / c(2.0 * normalization, 0.0);
            assert_complex_close(reconstructed_incident, incident[[frequency, port]]);
            assert_complex_close(reconstructed_outgoing, outgoing[port]);
            assert_some_close(
                &base,
                frequency,
                port,
                reconstructed_outgoing / reconstructed_incident,
            );
        }
    }
}

#[test]
fn passive_coupling_is_not_clipped_to_one() {
    let source = network(
        vec![1.0],
        2,
        vec![c(0.0, 0.0), c(0.8, 0.0), c(0.8, 0.0), c(0.0, 0.0)],
        real_z0(1, 2),
    );
    let incident = Array2::from_shape_vec((1, 2), vec![c(0.1, 0.0), c(1.0, 0.0)]).unwrap();
    let active = source.active_reflection_power(&incident).unwrap();
    assert_eq!(active[[0, 0]], Some(c(8.0, 0.0)));
    assert_some_close(&active, 0, 1, c(0.08, 0.0));
    assert_relative_eq!(
        source.max_singular_value_power().unwrap()[0],
        0.8,
        epsilon = 1.0e-14
    );
}

#[test]
fn port_permutation_and_nonstandard_owned_layouts_are_covariant() {
    let source_s = Array3::from_shape_fn((2, 3, 3), |(frequency, row, column)| {
        c(
            (10 * frequency + 3 * row + column) as f64 / 10.0,
            (row as f64 - column as f64) / 20.0,
        )
    });
    let source_z0 = Array2::from_shape_fn((2, 3), |(frequency, port)| {
        c(40.0 + frequency as f64 + port as f64, 1.0)
    });
    let source_incident = Array2::from_shape_fn((2, 3), |(frequency, port)| {
        c(
            0.2 + frequency as f64 + port as f64 / 10.0,
            -0.3 + port as f64 / 20.0,
        )
    });
    let source = Network::new(
        Frequency::from_hz(vec![1.0, 2.0]).unwrap(),
        source_s.clone(),
        source_z0.clone(),
    )
    .unwrap();
    let permutation = [2, 0, 1];
    let reordered_source = source.select_ports_zero_incident(&permutation).unwrap();
    let reordered_incident = Array2::from_shape_fn((2, 3), |(frequency, port)| {
        source_incident[[frequency, permutation[port]]]
    });
    let original = source.active_reflection_power(&source_incident).unwrap();
    let reordered = reordered_source
        .active_reflection_power(&reordered_incident)
        .unwrap();
    for frequency in 0..2 {
        for port in 0..3 {
            assert_some_close(
                &reordered,
                frequency,
                port,
                original[[frequency, permutation[port]]].unwrap(),
            );
        }
    }

    let transposed_s = source_s
        .view()
        .permuted_axes([0, 2, 1])
        .to_owned()
        .permuted_axes([0, 2, 1]);
    let transposed_z0 =
        Array2::from_shape_fn((3, 2), |(port, frequency)| source_z0[[frequency, port]])
            .reversed_axes();
    let transposed_incident = Array2::from_shape_fn((3, 2), |(port, frequency)| {
        source_incident[[frequency, port]]
    })
    .reversed_axes();
    let nonstandard = Network::new(
        Frequency::from_hz(vec![1.0, 2.0]).unwrap(),
        transposed_s,
        transposed_z0,
    )
    .unwrap();
    let actual = nonstandard
        .active_reflection_power(&transposed_incident)
        .unwrap();
    assert_eq!(actual, original);
    let mut owned = actual.clone();
    owned[[0, 0]] = Some(c(999.0, 0.0));
    assert_ne!(owned, actual);
}

#[test]
fn extreme_scales_tiny_nonzero_drives_and_zero_numerators_are_safe() {
    let source = network(
        vec![1.0, 2.0],
        2,
        vec![
            c(0.4, -0.2),
            c(-0.1, 0.3),
            c(0.2, 0.15),
            c(-0.35, 0.05),
            c(-0.2, 0.1),
            c(0.3, -0.2),
            c(0.15, 0.25),
            c(0.1, 0.35),
        ],
        real_z0(2, 2),
    );
    let incident = Array2::from_shape_vec(
        (2, 2),
        vec![
            c(1.0e150, -2.0e150),
            c(-3.0e150, 4.0e150),
            c(1.0e-150, -2.0e-150),
            c(-3.0e-150, 4.0e-150),
        ],
    )
    .unwrap();
    let actual = source.active_reflection_power(&incident).unwrap();
    for frequency in 0..2 {
        for port in 0..2 {
            assert_some_close(
                &actual,
                frequency,
                port,
                expected_ratio(&source, &incident, frequency, port),
            );
        }
    }

    let one_port = network(vec![1.0, 2.0], 1, vec![c(0.25, -0.4); 2], real_z0(2, 1));
    let huge_and_tiny =
        Array2::from_shape_vec((2, 1), vec![c(1.0e200, 1.0e200), c(1.0e-200, 1.0e-200)]).unwrap();
    let ratios = one_port.active_reflection_power(&huge_and_tiny).unwrap();
    assert_some_close(&ratios, 0, 0, c(0.25, -0.4));
    assert_some_close(&ratios, 1, 0, c(0.25, -0.4));

    let coupled = network(
        vec![1.0],
        2,
        vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        real_z0(1, 2),
    );
    let tiny = Array2::from_shape_vec((1, 2), vec![c(1.0e-300, 0.0), c(2.0e-300, 0.0)]).unwrap();
    let tiny_output = coupled.active_reflection_power(&tiny).unwrap();
    assert_eq!(tiny_output[[0, 0]], Some(c(2.0, 0.0)));
    assert_eq!(tiny_output[[0, 1]], Some(c(0.5, 0.0)));

    let overflowed_product = network(vec![1.0], 1, vec![c(f64::MAX, 0.0)], real_z0(1, 1));
    let finite_ratio = overflowed_product
        .active_reflection_power(&Array2::from_shape_vec((1, 1), vec![c(2.0, 0.0)]).unwrap())
        .unwrap();
    assert_eq!(finite_ratio[[0, 0]], Some(c(f64::MAX, 0.0)));

    let underflowed_product = network(vec![1.0], 1, vec![c(0.5, 0.0)], real_z0(1, 1));
    let minimum_subnormal = f64::from_bits(1);
    let finite_tiny_ratio = underflowed_product
        .active_reflection_power(
            &Array2::from_shape_vec((1, 1), vec![c(minimum_subnormal, 0.0)]).unwrap(),
        )
        .unwrap();
    assert_eq!(finite_tiny_ratio[[0, 0]], Some(c(0.5, 0.0)));

    let zero_numerator = network(vec![1.0], 1, vec![c(0.0, 0.0)], real_z0(1, 1));
    let output = zero_numerator
        .active_reflection_power(&Array2::from_shape_vec((1, 1), vec![c(2.0, -3.0)]).unwrap())
        .unwrap();
    assert_eq!(output[[0, 0]], Some(c(0.0, 0.0)));
}

#[test]
fn malformed_shapes_and_nonfinite_inputs_fail_before_indexing() {
    let valid = network(vec![1.0], 1, vec![c(0.1, 0.0)], vec![c(50.0, 0.0)]);
    let valid_incident = Array2::from_shape_vec((1, 1), vec![c(1.0, 0.0)]).unwrap();

    let mut malformed = serde_json::to_value(&valid).unwrap();
    malformed["frequency"]["hz"] = json!([]);
    malformed["s"]["dim"] = json!([0, 1, 1]);
    malformed["s"]["data"] = json!([]);
    malformed["z0"]["dim"] = json!([0, 1]);
    malformed["z0"]["data"] = json!([]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        malformed.active_reflection_power(&valid_incident)
    }));
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().unwrap_err(),
        Error::EmptyActiveReflectionPowerFrequency
    );

    let mut malformed = serde_json::to_value(&valid).unwrap();
    malformed["frequency"]["hz"] = json!([1.0, 2.0]);
    malformed["s"]["dim"] = json!([1, 1, 1]);
    let scalar = malformed["s"]["data"][0].clone();
    malformed["s"]["data"] = json!([scalar]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    assert_eq!(
        malformed
            .active_reflection_power(&valid_incident)
            .unwrap_err(),
        Error::ActiveReflectionPowerFrequencyLengthMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let mut malformed = serde_json::to_value(&valid).unwrap();
    let scalar = malformed["s"]["data"][0].clone();
    malformed["s"]["dim"] = json!([1, 1, 2]);
    malformed["s"]["data"] = json!([scalar.clone(), scalar]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    assert_eq!(
        malformed
            .active_reflection_power(&valid_incident)
            .unwrap_err(),
        Error::InvalidActiveReflectionPowerSShape {
            shape: vec![1, 1, 2],
        }
    );

    let wrong_incident = Array2::zeros((2, 1));
    assert_eq!(
        valid.active_reflection_power(&wrong_incident).unwrap_err(),
        Error::InvalidActiveReflectionPowerIncidentShape { shape: vec![2, 1] }
    );

    let mut malformed = serde_json::to_value(&valid).unwrap();
    let scalar = malformed["z0"]["data"][0].clone();
    malformed["z0"]["dim"] = json!([1, 2]);
    malformed["z0"]["data"] = json!([scalar.clone(), scalar]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        malformed.active_reflection_power(&valid_incident)
    }));
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().unwrap_err(),
        Error::InvalidActiveReflectionPowerZ0Shape { shape: vec![1, 2] }
    );

    let nonfinite_frequency = network(
        vec![f64::INFINITY],
        1,
        vec![c(0.1, 0.0)],
        vec![c(50.0, 0.0)],
    );
    assert_eq!(
        nonfinite_frequency
            .active_reflection_power(&valid_incident)
            .unwrap_err(),
        Error::NonFiniteActiveReflectionPowerFrequency {
            index: 0,
            value: f64::INFINITY,
        }
    );

    let nonfinite_s = network(vec![1.0], 1, vec![c(f64::NAN, 0.0)], vec![c(50.0, 0.0)]);
    assert_eq!(
        nonfinite_s
            .active_reflection_power(&valid_incident)
            .unwrap_err(),
        Error::NonFiniteActiveReflectionPowerS {
            frequency: 0,
            row: 0,
            column: 0,
        }
    );

    let nonfinite_z0 = network(vec![1.0], 1, vec![c(0.1, 0.0)], vec![c(f64::NAN, 0.0)]);
    assert_eq!(
        nonfinite_z0
            .active_reflection_power(&valid_incident)
            .unwrap_err(),
        Error::NonFiniteActiveReflectionPowerZ0 {
            frequency: 0,
            port: 0,
        }
    );

    let nonfinite_incident = Array2::from_shape_vec((1, 1), vec![c(f64::NAN, 0.0)]).unwrap();
    assert_eq!(
        valid
            .active_reflection_power(&nonfinite_incident)
            .unwrap_err(),
        Error::NonFiniteActiveReflectionPowerIncident {
            frequency: 0,
            port: 0,
        }
    );
}

#[test]
fn nonpositive_references_are_rejected_even_for_all_zero_rows() {
    let incident = Array2::zeros((1, 2));
    for reference in [c(0.0, 2.0), c(-1.0, 4.0)] {
        let source = network(
            vec![1.0],
            2,
            vec![c(0.0, 0.0); 4],
            vec![reference, c(50.0, 0.0)],
        );
        assert!(matches!(
            source.active_reflection_power(&incident),
            Err(
                Error::NonPositiveRealActiveReflectionPowerReferenceImpedance {
                    frequency: 0,
                    port: 0,
                    ..
                }
            )
        ));
    }
}

#[test]
fn unrepresentable_ratio_is_a_contextual_arithmetic_error() {
    let source = network(
        vec![1.0],
        2,
        vec![c(0.0, 0.0), c(f64::MAX, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        real_z0(1, 2),
    );
    let incident =
        Array2::from_shape_vec((1, 2), vec![c(f64::MIN_POSITIVE, 0.0), c(f64::MAX, 0.0)]).unwrap();
    assert_eq!(
        source.active_reflection_power(&incident).unwrap_err(),
        Error::NonFiniteActiveReflectionPowerComputation {
            frequency: 0,
            output_port: 0,
        }
    );
}
