use std::panic::{AssertUnwindSafe, catch_unwind};

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{ConnectionInput, Error, Frequency, Network};
use serde_json::json;

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn network(frequency: &[f64], s: Array3<Complex64>, z0: Array2<Complex64>) -> Network {
    Network::new(Frequency::from_hz(frequency.to_vec()).unwrap(), s, z0).unwrap()
}

fn structured_network(
    frequency: &[f64],
    nports: usize,
    seed: f64,
    z0: Array2<Complex64>,
) -> Network {
    let s = Array3::from_shape_fn((frequency.len(), nports, nports), |(f, row, column)| {
        c(
            seed + 0.07 * f as f64 + 0.11 * row as f64 - 0.05 * column as f64,
            -0.13 + 0.03 * row as f64 + 0.09 * column as f64,
        )
    });
    network(frequency, s, z0)
}

fn assert_close(actual: Complex64, expected: Complex64, tolerance: f64) {
    let error = (actual - expected).norm();
    assert!(
        error <= tolerance,
        "actual={actual:?}, expected={expected:?}, error={error:e}"
    );
}

fn assert_array3_close(actual: &Array3<Complex64>, expected: &Array3<Complex64>, tolerance: f64) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, value) in actual.indexed_iter() {
        assert_close(*value, expected[index], tolerance);
    }
}

fn assert_network_finite(network: &Network) {
    assert!(
        network
            .frequency()
            .hz()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(network.s().iter().all(|value| value.is_finite()));
    assert!(network.z0().iter().all(|value| value.is_finite()));
}

fn solve_two_by_two(matrix: [[Complex64; 2]; 2], rhs: [Complex64; 2]) -> [Complex64; 2] {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    assert_ne!(determinant, ZERO, "test system must be nonsingular");
    [
        (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
        (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
    ]
}

/// Solve one external excitation from the physical voltage/current equations
/// directly.  This is intentionally separate from `direct_reference`: the
/// returned waves are used below to check the actual V/I continuity residuals
/// and the output scattering relation.
fn physical_excitation(
    a: &Network,
    port_a: usize,
    b: &Network,
    port_b: usize,
    frequency: usize,
    external_incident: &[Complex64],
) -> (
    Vec<Complex64>,
    Vec<Complex64>,
    Vec<Complex64>,
    Vec<Complex64>,
) {
    let survivors_a: Vec<_> = (0..a.nports()).filter(|&port| port != port_a).collect();
    let survivors_b: Vec<_> = (0..b.nports()).filter(|&port| port != port_b).collect();
    assert_eq!(
        external_incident.len(),
        survivors_a.len() + survivors_b.len()
    );

    let mut incident_a = vec![ZERO; a.nports()];
    let mut incident_b = vec![ZERO; b.nports()];
    for (index, &port) in survivors_a.iter().enumerate() {
        incident_a[port] = external_incident[index];
    }
    for (index, &port) in survivors_b.iter().enumerate() {
        incident_b[port] = external_incident[survivors_a.len() + index];
    }

    let external_reflected_a = (0..a.nports())
        .map(|row| {
            survivors_a
                .iter()
                .enumerate()
                .map(|(index, &port)| a.s()[[frequency, row, port]] * external_incident[index])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let external_reflected_b = (0..b.nports())
        .map(|row| {
            survivors_b
                .iter()
                .enumerate()
                .map(|(index, &port)| {
                    b.s()[[frequency, row, port]] * external_incident[survivors_a.len() + index]
                })
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();

    let za = a.z0()[[frequency, port_a]];
    let zb = b.z0()[[frequency, port_b]];
    let qa = za.re.abs().sqrt() / za.re;
    let qb = zb.re.abs().sqrt() / zb.re;
    let saa = a.s()[[frequency, port_a, port_a]];
    let sbb = b.s()[[frequency, port_b, port_b]];
    let matrix = [
        [
            c(qa, 0.0) * (za.conj() + za * saa),
            -c(qb, 0.0) * (zb.conj() + zb * sbb),
        ],
        [
            c(qa, 0.0) * (c(1.0, 0.0) - saa),
            c(qb, 0.0) * (c(1.0, 0.0) - sbb),
        ],
    ];
    let rhs = [
        -c(qa, 0.0) * za * external_reflected_a[port_a]
            + c(qb, 0.0) * zb * external_reflected_b[port_b],
        c(qa, 0.0) * external_reflected_a[port_a] + c(qb, 0.0) * external_reflected_b[port_b],
    ];
    let internal = solve_two_by_two(matrix, rhs);
    incident_a[port_a] = internal[0];
    incident_b[port_b] = internal[1];

    let reflected_a = (0..a.nports())
        .map(|row| {
            (0..a.nports())
                .map(|column| a.s()[[frequency, row, column]] * incident_a[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let reflected_b = (0..b.nports())
        .map(|row| {
            (0..b.nports())
                .map(|column| b.s()[[frequency, row, column]] * incident_b[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();

    (incident_a, reflected_a, incident_b, reflected_b)
}

fn direct_reference(
    s_a: &Array3<Complex64>,
    z_a: &Array2<Complex64>,
    port_a: usize,
    s_b: &Array3<Complex64>,
    z_b: &Array2<Complex64>,
    port_b: usize,
) -> Array3<Complex64> {
    // Independent physical boundary reconstruction.  This evaluates the
    // network equations at an external incident excitation rather than using
    // the direct connection's elimination formula.
    let nfreq = s_a.dim().0;
    let ext_a: Vec<_> = (0..s_a.dim().1).filter(|&p| p != port_a).collect();
    let ext_b: Vec<_> = (0..s_b.dim().1).filter(|&p| p != port_b).collect();
    let ext = ext_a.len() + ext_b.len();
    let mut output = Array3::zeros((nfreq, ext, ext));

    for frequency in 0..nfreq {
        let za = z_a[[frequency, port_a]];
        let zb = z_b[[frequency, port_b]];
        let qa = za.re.abs().sqrt() / za.re;
        let qb = zb.re.abs().sqrt() / zb.re;
        let saa = s_a[[frequency, port_a, port_a]];
        let sbb = s_b[[frequency, port_b, port_b]];

        // Unknowns are [a_A, a_B].  For each external excitation solve the
        // two boundary equations explicitly with a small complex determinant.
        let c00 = c(qa, 0.0);
        let c01 = c(qb, 0.0);
        let c10 = c(qa, 0.0) * za.conj();
        let c11 = -c(qb, 0.0) * zb.conj();
        let d00 = -c(qa, 0.0);
        let d01 = -c(qb, 0.0);
        let d10 = c(qa, 0.0) * za;
        let d11 = -c(qb, 0.0) * zb;
        let a00 = c00 + d00 * saa;
        let a01 = c01 + d01 * sbb;
        let a10 = c10 + d10 * saa;
        let a11 = c11 + d11 * sbb;
        let determinant = a00 * a11 - a01 * a10;

        for (input, (input_side, input_port)) in ext_a
            .iter()
            .map(|&p| (ConnectionInput::A, p))
            .chain(ext_b.iter().map(|&p| (ConnectionInput::B, p)))
            .enumerate()
        {
            let (sie_a, sie_b) = match input_side {
                ConnectionInput::A => (s_a[[frequency, port_a, input_port]], ZERO),
                ConnectionInput::B => (ZERO, s_b[[frequency, port_b, input_port]]),
                _ => unreachable!("ConnectionInput has only A/B in this test"),
            };
            let rhs0 = -(d00 * sie_a + d01 * sie_b);
            let rhs1 = -(d10 * sie_a + d11 * sie_b);
            let ia = (rhs0 * a11 - a01 * rhs1) / determinant;
            let ib = (a00 * rhs1 - rhs0 * a10) / determinant;

            for (output_index, (output_side, output_port)) in ext_a
                .iter()
                .map(|&p| (ConnectionInput::A, p))
                .chain(ext_b.iter().map(|&p| (ConnectionInput::B, p)))
                .enumerate()
            {
                let see = match (output_side, input_side) {
                    (ConnectionInput::A, ConnectionInput::A) => {
                        s_a[[frequency, output_port, input_port]]
                    }
                    (ConnectionInput::B, ConnectionInput::B) => {
                        s_b[[frequency, output_port, input_port]]
                    }
                    _ => ZERO,
                };
                let sei_a = match output_side {
                    ConnectionInput::A => s_a[[frequency, output_port, port_a]],
                    ConnectionInput::B => ZERO,
                    _ => unreachable!("ConnectionInput has only A/B in this test"),
                };
                let sei_b = match output_side {
                    ConnectionInput::A => ZERO,
                    ConnectionInput::B => s_b[[frequency, output_port, port_b]],
                    _ => unreachable!("ConnectionInput has only A/B in this test"),
                };
                output[[frequency, output_index, input]] = see + sei_a * ia + sei_b * ib;
            }
        }
    }
    output
}

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

#[test]
fn direct_connection_matches_shared_positive_real_matched_domain() {
    let frequencies = [1.0e9, 2.0e9];
    let a_z0 = Array2::from_shape_fn((2, 3), |(f, p)| c(50.0 + f as f64 + p as f64, 0.0));
    let b_z0 = Array2::from_shape_fn((2, 4), |(f, p)| c(49.0 + f as f64 + p as f64, 0.0));
    let a = structured_network(&frequencies, 3, 0.01, a_z0);
    let b = structured_network(&frequencies, 4, -0.03, b_z0);
    let direct = a.connect_direct_power(1, &b, 2).unwrap();
    let matched = a.connect_matched_power(1, &b, 2).unwrap();
    assert_eq!(direct.frequency().hz(), &frequencies);
    assert_eq!(direct.z0(), matched.z0());
    assert_array3_close(direct.s(), matched.s(), 2.0e-12);
}

#[test]
fn direct_connection_preserves_asymmetric_survivor_order_and_inputs() {
    let frequencies = [3.0e9, -1.0, 3.0e9];
    let a_z0 = Array2::from_shape_fn((3, 3), |(f, p)| c(41.0 + 2.0 * f as f64 + p as f64, 0.3));
    let b_z0 = Array2::from_shape_fn((3, 4), |(f, p)| c(62.0 + 3.0 * f as f64 + p as f64, -0.7));
    let a = structured_network(&frequencies, 3, 0.02, a_z0);
    let b = structured_network(&frequencies, 4, -0.04, b_z0);
    let original_a = a.clone();
    let original_b = b.clone();

    let result = a.connect_direct_power(0, &b, 3).unwrap();
    assert_eq!(result.frequency().hz(), frequencies);
    assert_eq!(result.z0().dim(), (3, 5));
    for f in 0..3 {
        assert_eq!(result.z0()[[f, 0]], a.z0()[[f, 1]]);
        assert_eq!(result.z0()[[f, 1]], a.z0()[[f, 2]]);
        assert_eq!(result.z0()[[f, 2]], b.z0()[[f, 0]]);
        assert_eq!(result.z0()[[f, 3]], b.z0()[[f, 1]]);
        assert_eq!(result.z0()[[f, 4]], b.z0()[[f, 2]]);
    }
    assert_eq!(a, original_a);
    assert_eq!(b, original_b);
}

#[test]
fn direct_connection_matches_independent_boundary_reconstruction() {
    let frequencies = [1.0e9];
    let a_z0 =
        Array2::from_shape_vec((1, 3), vec![c(31.0, 4.0), c(-47.0, 3.0), c(72.0, -2.0)]).unwrap();
    let b_z0 = Array2::from_shape_vec(
        (1, 4),
        vec![c(61.0, -5.0), c(39.0, 2.0), c(-53.0, -1.0), c(81.0, 7.0)],
    )
    .unwrap();
    let a = structured_network(&frequencies, 3, 0.04, a_z0);
    let b = structured_network(&frequencies, 4, -0.02, b_z0);
    let connected = a.connect_direct_power(1, &b, 2).unwrap();
    let expected = direct_reference(a.s(), a.z0(), 1, b.s(), b.z0(), 2);
    assert_array3_close(connected.s(), &expected, 3.0e-12);
}

#[test]
fn direct_connection_satisfies_physical_vi_continuity_and_scattering_reconstruction() {
    let frequency = [1.0e9];
    let a = structured_network(
        &frequency,
        3,
        0.04,
        Array2::from_shape_vec((1, 3), vec![c(37.0, 4.0), c(53.0, -3.0), c(-61.0, 2.0)]).unwrap(),
    );
    let b = structured_network(
        &frequency,
        4,
        -0.02,
        Array2::from_shape_vec(
            (1, 4),
            vec![c(67.0, -5.0), c(43.0, 3.0), c(-79.0, 1.0), c(-47.0, -2.0)],
        )
        .unwrap(),
    );
    let connected = a.connect_direct_power(1, &b, 2).unwrap();
    assert_network_finite(&connected);

    let external_incident = vec![
        c(0.31, -0.17),
        c(-0.23, 0.29),
        c(0.19, 0.11),
        c(-0.13, -0.27),
        c(0.07, 0.35),
    ];
    let (incident_a, reflected_a, incident_b, reflected_b) =
        physical_excitation(&a, 1, &b, 2, 0, &external_incident);
    let za = a.z0()[[0, 1]];
    let zb = b.z0()[[0, 2]];
    let qa = za.re.abs().sqrt() / za.re;
    let qb = zb.re.abs().sqrt() / zb.re;
    let voltage_a = c(qa, 0.0) * (za.conj() * incident_a[1] + za * reflected_a[1]);
    let voltage_b = c(qb, 0.0) * (zb.conj() * incident_b[2] + zb * reflected_b[2]);
    let current_a = c(qa, 0.0) * (incident_a[1] - reflected_a[1]);
    let current_b = c(qb, 0.0) * (incident_b[2] - reflected_b[2]);
    assert_close(voltage_a, voltage_b, 2.0e-12);
    assert_close(current_a + current_b, ZERO, 2.0e-12);

    let reflected_external = (0..connected.nports())
        .map(|row| {
            (0..connected.nports())
                .map(|column| connected.s()[[0, row, column]] * external_incident[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    for (index, &port) in [0_usize, 2].iter().enumerate() {
        assert_close(reflected_external[index], reflected_a[port], 2.0e-12);
    }
    for (index, &port) in [0_usize, 1, 3].iter().enumerate() {
        assert_close(reflected_external[2 + index], reflected_b[port], 2.0e-12);
    }
}

#[test]
fn direct_connection_matches_independent_physical_z_elimination() {
    let frequency_values = vec![1.0e9, 1.75e9];
    let frequency = Frequency::from_hz(frequency_values.clone()).unwrap();
    let z0_a = Array2::from_shape_vec(
        (2, 3),
        vec![
            c(47.0, 3.0),
            c(61.0, -2.0),
            c(-55.0, 4.0),
            c(49.0, 3.5),
            c(64.0, -2.5),
            c(-58.0, 4.5),
        ],
    )
    .unwrap();
    let z0_b = Array2::from_shape_vec(
        (2, 4),
        vec![
            c(73.0, 1.0),
            c(39.0, -4.0),
            c(82.0, 2.0),
            c(58.0, -3.0),
            c(76.0, 1.5),
            c(42.0, -4.5),
            c(86.0, 2.5),
            c(61.0, -3.5),
        ],
    )
    .unwrap();
    let z_a = Array3::from_shape_fn((2, 3, 3), |(f, row, column)| {
        let values = [
            [c(83.0, 4.0), c(7.0, -2.0), c(-5.0, 1.0)],
            [c(-4.0, 3.0), c(111.0, -5.0), c(12.0, 2.0)],
            [c(9.0, -1.0), c(-6.0, 4.0), c(94.0, 6.0)],
        ];
        values[row][column] + c(2.0 * f as f64, -0.5 * f as f64)
    });
    let z_b = Array3::from_shape_fn((2, 4, 4), |(f, row, column)| {
        let values = [
            [c(97.0, 2.0), c(-8.0, 1.0), c(6.0, -3.0), c(4.0, 2.0)],
            [c(5.0, -2.0), c(88.0, 5.0), c(-7.0, 1.0), c(3.0, -1.0)],
            [c(-9.0, 2.0), c(11.0, -4.0), c(126.0, -6.0), c(14.0, 3.0)],
            [c(2.0, 1.0), c(-5.0, 3.0), c(8.0, -2.0), c(104.0, 4.0)],
        ];
        values[row][column] + c(1.5 * f as f64, 0.75 * f as f64)
    });
    let a = Network::from_z_power(frequency.clone(), z_a.clone(), z0_a.clone()).unwrap();
    let b = Network::from_z_power(frequency.clone(), z_b.clone(), z0_b.clone()).unwrap();
    let connected = a.connect_direct_power(1, &b, 2).unwrap();
    assert_network_finite(&connected);

    let survivors_a = [0_usize, 2];
    let survivors_b = [0_usize, 1, 3];
    let mut expected_z = Array3::zeros((2, 5, 5));
    let expected_z0 = Array2::from_shape_fn((2, 5), |(f, output)| {
        if output < survivors_a.len() {
            z0_a[[f, survivors_a[output]]]
        } else {
            z0_b[[f, survivors_b[output - survivors_a.len()]]]
        }
    });
    for f in 0..2 {
        let denominator = z_a[[f, 1, 1]] + z_b[[f, 2, 2]];
        let inverse = c(1.0, 0.0) / denominator;
        for (output_row, &row) in survivors_a.iter().enumerate() {
            for (output_column, &column) in survivors_a.iter().enumerate() {
                expected_z[[f, output_row, output_column]] =
                    z_a[[f, row, column]] - z_a[[f, row, 1]] * inverse * z_a[[f, 1, column]];
            }
            for (offset, &column) in survivors_b.iter().enumerate() {
                expected_z[[f, output_row, survivors_a.len() + offset]] =
                    z_a[[f, row, 1]] * inverse * z_b[[f, 2, column]];
            }
        }
        for (offset, &row) in survivors_b.iter().enumerate() {
            let output_row = survivors_a.len() + offset;
            for (output_column, &column) in survivors_a.iter().enumerate() {
                expected_z[[f, output_row, output_column]] =
                    z_b[[f, row, 2]] * inverse * z_a[[f, 1, column]];
            }
            for (column_offset, &column) in survivors_b.iter().enumerate() {
                expected_z[[f, output_row, survivors_a.len() + column_offset]] =
                    z_b[[f, row, column]] - z_b[[f, row, 2]] * inverse * z_b[[f, 2, column]];
            }
        }
    }
    let expected = Network::from_z_power(frequency, expected_z, expected_z0).unwrap();
    assert_network_finite(&expected);
    assert_eq!(connected.z0(), expected.z0());
    assert_array3_close(connected.s(), expected.s(), 1.0e-12);
}

fn assert_output_reindexed(actual: &Network, reference: &Network, mapping: &[usize]) {
    assert_eq!(actual.frequency().hz(), reference.frequency().hz());
    assert_eq!(actual.nports(), mapping.len());
    for (frequency, (&actual_frequency, &reference_frequency)) in actual
        .frequency()
        .hz()
        .iter()
        .zip(reference.frequency().hz())
        .enumerate()
    {
        assert_eq!(actual_frequency.to_bits(), reference_frequency.to_bits());
        for row in 0..mapping.len() {
            assert_eq!(
                actual.z0()[[frequency, row]],
                reference.z0()[[frequency, mapping[row]]]
            );
            for column in 0..mapping.len() {
                assert_close(
                    actual.s()[[frequency, row, column]],
                    reference.s()[[frequency, mapping[row], mapping[column]]],
                    3.0e-12,
                );
            }
        }
    }
}

#[test]
fn direct_connection_is_covariant_under_port_permutation_and_ab_exchange() {
    let frequency = [1.0e9, 1.5e9];
    let a = structured_network(
        &frequency,
        3,
        0.03,
        Array2::from_shape_fn((2, 3), |(f, p)| {
            c(39.0 + 4.0 * f as f64 + 7.0 * p as f64, 2.0)
        }),
    );
    let b = structured_network(
        &frequency,
        4,
        -0.01,
        Array2::from_shape_fn((2, 4), |(f, p)| {
            c(57.0 + 3.0 * f as f64 + 5.0 * p as f64, -1.0)
        }),
    );
    let original = a.connect_direct_power(1, &b, 2).unwrap();
    let order_a = [2, 0, 1];
    let order_b = [3, 1, 0, 2];
    let permuted_a = a.permute_ports(&order_a).unwrap();
    let permuted_b = b.permute_ports(&order_b).unwrap();
    let permuted = permuted_a.connect_direct_power(2, &permuted_b, 3).unwrap();
    assert_network_finite(&permuted);
    // New A survivors [old 2, old 0], then new B survivors [old 3, old 1, old 0].
    assert_output_reindexed(&permuted, &original, &[1, 0, 4, 3, 2]);

    let exchanged = b.connect_direct_power(2, &a, 1).unwrap();
    assert_network_finite(&exchanged);
    // Exchanged output is B survivors followed by A survivors.
    assert_output_reindexed(&exchanged, &original, &[2, 3, 4, 0, 1]);
}

#[test]
fn direct_connection_keeps_b_two_port_survivor_order_with_equal_complex_selected_refs() {
    let frequency = [2.0e9];
    let selected_reference = c(46.0, 7.0);
    let a = structured_network(
        &frequency,
        3,
        0.02,
        Array2::from_shape_vec(
            (1, 3),
            vec![c(35.0, 2.0), selected_reference, c(68.0, -1.0)],
        )
        .unwrap(),
    );
    let b = structured_network(
        &frequency,
        2,
        -0.03,
        Array2::from_shape_vec((1, 2), vec![selected_reference, c(-59.0, 3.0)]).unwrap(),
    );
    let connected = a.connect_direct_power(1, &b, 0).unwrap();
    assert_network_finite(&connected);
    assert_eq!(connected.nports(), 3);
    assert_eq!(connected.z0()[[0, 0]], a.z0()[[0, 0]]);
    assert_eq!(connected.z0()[[0, 1]], a.z0()[[0, 2]]);
    assert_eq!(connected.z0()[[0, 2]], b.z0()[[0, 1]]);
    assert_eq!(
        connected.frequency().hz()[0].to_bits(),
        frequency[0].to_bits()
    );
}

#[test]
fn direct_connection_accepts_zero_selected_reference_sum_and_singular_source_conversion() {
    let frequency = [1.0e9];
    // A is an ideal thru, for which the composed S->Z conversion is exactly
    // singular, but its selected S_aa=0 still permits the direct junction.
    let a = network(
        &frequency,
        Array3::from_shape_vec((1, 2, 2), vec![ZERO, c(1.0, 0.0), c(1.0, 0.0), ZERO]).unwrap(),
        Array2::from_shape_vec((1, 2), vec![c(50.0, 0.0), c(61.0, 2.0)]).unwrap(),
    );
    assert!(matches!(a.to_z_power(), Err(Error::Singular { .. })));
    let b = network(
        &frequency,
        Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.2, 0.0), c(0.04, -0.02), c(-0.03, 0.01), c(0.1, 0.0)],
        )
        .unwrap(),
        Array2::from_shape_vec((1, 2), vec![c(-50.0, 0.0), c(73.0, -1.0)]).unwrap(),
    );
    let connected = a.connect_direct_power(0, &b, 0).unwrap();
    assert_network_finite(&connected);
    assert_eq!(connected.nports(), 2);
}

#[test]
fn direct_connection_supports_one_port_input_on_either_side_and_two_port_orientations() {
    let frequency = [1.0e9];
    let one = network(
        &frequency,
        Array3::from_shape_vec((1, 1, 1), vec![c(0.13, -0.02)]).unwrap(),
        Array2::from_shape_vec((1, 1), vec![c(37.0, 2.0)]).unwrap(),
    );
    let two = structured_network(
        &frequency,
        2,
        -0.03,
        Array2::from_shape_vec((1, 2), vec![c(51.0, -2.0), c(73.0, 3.0)]).unwrap(),
    );
    let left = one.connect_direct_power(0, &two, 0).unwrap();
    assert_eq!(left.nports(), 1);
    assert_eq!(left.z0()[[0, 0]], two.z0()[[0, 1]]);
    let right = two.connect_direct_power(1, &one, 0).unwrap();
    assert_eq!(right.nports(), 1);
    assert_eq!(right.z0()[[0, 0]], two.z0()[[0, 0]]);
}

#[test]
fn direct_connection_agrees_with_physical_one_port_termination() {
    let frequency = [1.0e9, 2.0e9];
    let load = [c(37.0, 11.0), c(83.0, -9.0)];
    let source_z0 =
        Array2::from_shape_fn((2, 2), |(f, p)| c(41.0 + 7.0 * p as f64 + f as f64, 2.0));
    let source = structured_network(&frequency, 2, 0.02, source_z0);
    let load_z0 = Array2::from_shape_fn((2, 1), |(f, _)| c(63.0 + 3.0 * f as f64, -4.0));
    let load_s = Array3::from_shape_fn((2, 1, 1), |(f, _, _)| {
        let z = load[f];
        let reference = load_z0[[f, 0]];
        // A one-port network with V = Z_load I (current into the load)
        // represents the physical load at the junction; the direct junction
        // then imposes I_A + I_B = 0, exactly as termination does.
        (z - reference.conj()) / (z + reference)
    });
    let load_network = network(&frequency, load_s, load_z0);

    let connected = source.connect_direct_power(0, &load_network, 0).unwrap();
    let terminated = source.terminate_port_impedance_power(0, &load).unwrap();
    assert_eq!(connected.z0(), terminated.z0());
    assert_array3_close(connected.s(), terminated.s(), 3.0e-12);
}

#[test]
fn direct_connection_accepts_negative_and_complex_selected_references() {
    let frequency = [1.0e9];
    let a = structured_network(
        &frequency,
        2,
        0.01,
        Array2::from_shape_vec((1, 2), vec![c(-43.0, 5.0), c(71.0, -3.0)]).unwrap(),
    );
    let b = structured_network(
        &frequency,
        2,
        -0.02,
        Array2::from_shape_vec((1, 2), vec![c(59.0, 4.0), c(-67.0, -2.0)]).unwrap(),
    );
    let result = a.connect_direct_power(0, &b, 1).unwrap();
    assert_eq!(result.z0()[[0, 0]], a.z0()[[0, 1]]);
    assert_eq!(result.z0()[[0, 1]], b.z0()[[0, 0]]);
    assert!(
        result
            .s()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn direct_connection_distinguishes_exact_singularity_near_singularity_and_overflow() {
    // With both selected one-port S values equal to +1, the direct junction
    // system is exactly singular even though no external coupling exists.
    let frequency = [1.0];
    let exact_a = network(
        &frequency,
        Array3::from_shape_vec((1, 2, 2), vec![c(1.0, 0.0), ZERO, ZERO, ZERO]).unwrap(),
        Array2::from_shape_vec((1, 2), vec![c(50.0, 0.0), c(65.0, 0.0)]).unwrap(),
    );
    let exact_b = network(
        &frequency,
        Array3::from_shape_vec((1, 2, 2), vec![c(1.0, 0.0), ZERO, ZERO, ZERO]).unwrap(),
        Array2::from_shape_vec((1, 2), vec![c(75.0, 0.0), c(85.0, 0.0)]).unwrap(),
    );
    assert!(matches!(
        exact_a.connect_direct_power(0, &exact_b, 0),
        Err(Error::SingularDirectConnection {
            frequency: 0,
            port_a: 0,
            port_b: 0,
            ..
        })
    ));

    let near_a = network(
        &frequency,
        Array3::from_shape_vec((1, 2, 2), vec![c(1.0 - 1.0e-15, 0.0), ZERO, ZERO, ZERO]).unwrap(),
        Array2::from_shape_vec((1, 2), vec![c(50.0, 0.0), c(65.0, 0.0)]).unwrap(),
    );
    let near_b = exact_b.clone();
    let near = near_a.connect_direct_power(0, &near_b, 0).unwrap();
    assert!(
        near.s()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let overflow_a = network(
        &frequency,
        Array3::from_shape_vec((1, 1, 1), vec![c(f64::MAX, 0.0)]).unwrap(),
        Array2::from_shape_vec((1, 1), vec![c(f64::MIN_POSITIVE, 0.0)]).unwrap(),
    );
    let overflow_b = exact_b;
    assert_eq!(
        overflow_a
            .connect_direct_power(0, &overflow_b, 0)
            .unwrap_err(),
        Error::NonFiniteDirectConnectionComputation {
            frequency: 0,
            port_a: 0,
            port_b: 0,
            row: 0,
            column: 0,
        }
    );
}

#[test]
fn direct_connection_reports_validation_without_panicking() {
    let valid = structured_network(&[1.0e9], 2, 0.0, Array2::from_elem((1, 2), c(50.0, 0.0)));
    let one = structured_network(&[1.0e9], 1, 0.0, Array2::from_elem((1, 1), c(50.0, 0.0)));
    assert_eq!(
        valid.connect_direct_power(2, &one, 0).unwrap_err(),
        Error::InvalidDirectConnectionPort {
            input: ConnectionInput::A,
            port: 2,
            nports: 2,
        }
    );
    assert_eq!(
        one.connect_direct_power(0, &one, 0).unwrap_err(),
        Error::NoExternalDirectConnectionPorts
    );
    let mismatch = structured_network(&[1.1e9], 1, 0.0, Array2::from_elem((1, 1), c(50.0, 0.0)));
    assert!(matches!(
        valid.connect_direct_power(0, &mismatch, 0),
        Err(Error::DirectConnectionFrequencyMismatch { index: 0, .. })
    ));

    let length_mismatch = structured_network(
        &[1.0e9, 2.0e9],
        1,
        0.0,
        Array2::from_elem((2, 1), c(50.0, 0.0)),
    );
    assert_eq!(
        valid
            .connect_direct_power(0, &length_mismatch, 0)
            .unwrap_err(),
        Error::DirectConnectionFrequencyLengthMismatch { a: 1, b: 2 }
    );

    let mut malformed_value = serde_json::to_value(&one).unwrap();
    malformed_value["s"]["dim"] = json!([1, 1, 2]);
    let scalar = malformed_value["s"]["data"][0].clone();
    malformed_value["s"]["data"] = json!([scalar.clone(), scalar,]);
    let malformed: Network = serde_json::from_value(malformed_value).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        malformed.connect_direct_power(0, &one, 0)
    }));
    assert!(result.is_ok());
    assert!(matches!(
        result.unwrap(),
        Err(Error::InvalidDirectConnectionSShape { .. })
    ));

    let mut empty_frequency_value = serde_json::to_value(&one).unwrap();
    empty_frequency_value["frequency"]["hz"] = json!([]);
    let empty_frequency: Network = serde_json::from_value(empty_frequency_value).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        empty_frequency.connect_direct_power(0, &valid, 0)
    }));
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().unwrap_err(),
        Error::EmptyDirectConnectionFrequency {
            input: ConnectionInput::A
        }
    );

    let mut frequency_shape_value = serde_json::to_value(&one).unwrap();
    frequency_shape_value["frequency"]["hz"] = json!([1.0e9, 2.0e9]);
    let frequency_shape: Network = serde_json::from_value(frequency_shape_value).unwrap();
    assert_eq!(
        frequency_shape
            .connect_direct_power(0, &valid, 0)
            .unwrap_err(),
        Error::DirectConnectionFrequencyShape {
            input: ConnectionInput::A,
            expected: 1,
            actual: 2,
        }
    );

    let mut b_z0_shape_value = serde_json::to_value(&valid).unwrap();
    let scalar = b_z0_shape_value["z0"]["data"][0].clone();
    b_z0_shape_value["z0"]["dim"] = json!([1, 1]);
    b_z0_shape_value["z0"]["data"] = json!([scalar]);
    let malformed_b_z0: Network = serde_json::from_value(b_z0_shape_value).unwrap();
    assert_eq!(
        one.connect_direct_power(0, &malformed_b_z0, 0).unwrap_err(),
        Error::InvalidDirectConnectionZ0Shape {
            input: ConnectionInput::B,
            shape: vec![1, 1],
        }
    );
}

#[test]
fn direct_connection_rejects_nonfinite_values_and_zero_real_references_on_both_inputs() {
    let frequency = [1.0e9];
    let valid = structured_network(&frequency, 2, 0.0, Array2::from_elem((1, 2), c(50.0, 0.0)));

    let mut nonfinite_s = valid.s().clone();
    nonfinite_s[[0, 1, 0]] = c(f64::NAN, 0.0);
    let nonfinite_s = network(&frequency, nonfinite_s, valid.z0().clone());
    assert_eq!(
        nonfinite_s.connect_direct_power(0, &valid, 0).unwrap_err(),
        Error::NonFiniteDirectConnectionS {
            input: ConnectionInput::A,
            frequency: 0,
            row: 1,
            column: 0,
        }
    );

    let nonfinite_z0 = network(
        &frequency,
        valid.s().clone(),
        Array2::from_shape_vec((1, 2), vec![c(50.0, 0.0), c(f64::INFINITY, 0.0)]).unwrap(),
    );
    assert_eq!(
        nonfinite_z0.connect_direct_power(0, &valid, 0).unwrap_err(),
        Error::NonFiniteDirectConnectionZ0 {
            input: ConnectionInput::A,
            frequency: 0,
            port: 1,
        }
    );

    let zero_real = network(
        &frequency,
        valid.s().clone(),
        Array2::from_shape_vec((1, 2), vec![c(0.0, 25.0), c(50.0, 0.0)]).unwrap(),
    );
    assert_eq!(
        zero_real.connect_direct_power(0, &valid, 0).unwrap_err(),
        Error::ZeroRealDirectConnectionReferenceImpedance {
            input: ConnectionInput::A,
            frequency: 0,
            port: 0,
        }
    );

    let nonfinite_frequency = network(
        &[f64::INFINITY],
        Array3::zeros((1, 2, 2)),
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    );
    assert!(matches!(
        nonfinite_frequency.connect_direct_power(0, &valid, 0),
        Err(Error::NonFiniteDirectConnectionFrequency {
            input: ConnectionInput::A,
            index: 0,
            ..
        })
    ));

    let mut nonfinite_b_s = valid.s().clone();
    nonfinite_b_s[[0, 0, 1]] = c(f64::INFINITY, 0.0);
    let nonfinite_b_s = network(&frequency, nonfinite_b_s, valid.z0().clone());
    assert_eq!(
        valid
            .connect_direct_power(0, &nonfinite_b_s, 0)
            .unwrap_err(),
        Error::NonFiniteDirectConnectionS {
            input: ConnectionInput::B,
            frequency: 0,
            row: 0,
            column: 1,
        }
    );

    let nonfinite_b_z0 = network(
        &frequency,
        valid.s().clone(),
        Array2::from_shape_vec((1, 2), vec![c(50.0, 0.0), c(f64::NAN, 0.0)]).unwrap(),
    );
    assert_eq!(
        valid
            .connect_direct_power(0, &nonfinite_b_z0, 0)
            .unwrap_err(),
        Error::NonFiniteDirectConnectionZ0 {
            input: ConnectionInput::B,
            frequency: 0,
            port: 1,
        }
    );

    // The selected B port is 0; the zero-real survivor at B port 1 must still
    // be rejected because the direct operation validates every reference.
    let zero_real_b = network(
        &frequency,
        valid.s().clone(),
        Array2::from_shape_vec((1, 2), vec![c(50.0, 0.0), c(-0.0, 12.0)]).unwrap(),
    );
    assert_eq!(
        valid.connect_direct_power(0, &zero_real_b, 0).unwrap_err(),
        Error::ZeroRealDirectConnectionReferenceImpedance {
            input: ConnectionInput::B,
            frequency: 0,
            port: 1,
        }
    );
}
