use std::panic::{AssertUnwindSafe, catch_unwind};

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{ConversionStage, Error, Frequency, Network};
use serde_json::json;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn network(frequency: Vec<f64>, s: Array3<Complex64>, z0: Array2<Complex64>) -> Network {
    Network::new(Frequency::from_hz(frequency).unwrap(), s, z0).unwrap()
}

fn structured_network(frequency: Vec<f64>, nports: usize) -> Network {
    let nfreq = frequency.len();
    let s = Array3::from_shape_fn((nfreq, nports, nports), |(f, row, column)| {
        c(
            0.021 + 0.013 * f as f64 + 0.007 * row as f64 - 0.003 * column as f64,
            -0.017 + 0.004 * row as f64 + 0.005 * column as f64,
        )
    });
    let z0 = Array2::from_shape_fn((nfreq, nports), |(f, port)| {
        let sign = if (f + port) % 3 == 0 { -1.0 } else { 1.0 };
        c(
            sign * (35.0 + 7.0 * f as f64 + 3.0 * port as f64),
            -2.0 + 0.75 * f as f64 - 0.5 * port as f64,
        )
    });
    network(frequency, s, z0)
}

fn assert_close(actual: Complex64, expected: Complex64, rtol: f64, atol: f64) {
    let difference = (actual - expected).norm();
    let bound = atol + rtol * expected.norm();
    assert!(
        difference <= bound,
        "actual={actual:?}, expected={expected:?}, difference={difference:e}, bound={bound:e}"
    );
}

fn assert_array3_close(
    actual: &Array3<Complex64>,
    expected: &Array3<Complex64>,
    rtol: f64,
    atol: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual_value - expected_value).norm();
        let bound = atol + rtol * expected_value.norm();
        assert!(
            difference <= bound,
            "flattened index {index}: actual={actual_value:?}, expected={expected_value:?}, difference={difference:e}, bound={bound:e}"
        );
    }
}

fn solve_two_by_two(matrix: [[Complex64; 2]; 2], rhs: [Complex64; 2]) -> [Complex64; 2] {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    assert_ne!(
        determinant, ZERO,
        "test reference system must be nonsingular"
    );
    [
        (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
        (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
    ]
}

/// An independently written full-block evaluation of the Issue #90
/// equations.  In particular, this does not call the production solver and
/// retains S_ab and S_ba in S_ii.
fn direct_reference(
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    port_a: usize,
    port_b: usize,
) -> Array3<Complex64> {
    let (nfreq, nports, _) = s.dim();
    let external: Vec<_> = (0..nports)
        .filter(|&port| port != port_a && port != port_b)
        .collect();
    let mut output = Array3::zeros((nfreq, external.len(), external.len()));

    for f in 0..nfreq {
        let za = z0[[f, port_a]];
        let zb = z0[[f, port_b]];
        let qa = c(za.re.abs().sqrt() / za.re, 0.0);
        let qb = c(zb.re.abs().sqrt() / zb.re, 0.0);
        let c_matrix = [[qa, qb], [qa * za.conj(), -qb * zb.conj()]];
        let d_matrix = [[-qa, -qb], [qa * za, -qb * zb]];
        let s_ii = [
            s[[f, port_a, port_a]],
            s[[f, port_a, port_b]],
            s[[f, port_b, port_a]],
            s[[f, port_b, port_b]],
        ];
        let system = [
            [
                c_matrix[0][0] + d_matrix[0][0] * s_ii[0] + d_matrix[0][1] * s_ii[2],
                c_matrix[0][1] + d_matrix[0][0] * s_ii[1] + d_matrix[0][1] * s_ii[3],
            ],
            [
                c_matrix[1][0] + d_matrix[1][0] * s_ii[0] + d_matrix[1][1] * s_ii[2],
                c_matrix[1][1] + d_matrix[1][0] * s_ii[1] + d_matrix[1][1] * s_ii[3],
            ],
        ];

        for (output_column, &input_column) in external.iter().enumerate() {
            let s_ie = [s[[f, port_a, input_column]], s[[f, port_b, input_column]]];
            let rhs = [
                -(d_matrix[0][0] * s_ie[0] + d_matrix[0][1] * s_ie[1]),
                -(d_matrix[1][0] * s_ie[0] + d_matrix[1][1] * s_ie[1]),
            ];
            let t = solve_two_by_two(system, rhs);
            for (output_row, &input_row) in external.iter().enumerate() {
                let correction =
                    s[[f, input_row, port_a]] * t[0] + s[[f, input_row, port_b]] * t[1];
                output[[f, output_row, output_column]] =
                    s[[f, input_row, input_column]] + correction;
            }
        }
    }
    output
}

/// Independently reduce an invertible physical Z matrix at the selected
/// junction.  The convention is the same one used by the public network
/// methods: every current points into the source network.  Let
/// `j = I_a = -I_b`; then voltage continuity gives
///
/// ```text
/// (Z_aa - Z_ab - Z_ba + Z_bb) j = (Z_be - Z_ae) I_e
/// V_e = Z_ee I_e + (Z_ea - Z_eb) j.
/// ```
///
/// Thus the returned matrix is the physical Schur-style port elimination
/// `Z_ee + (Z_ea-Z_eb)(Z_be-Z_ae)/den`, with rows and columns in the original
/// survivor order.  This helper intentionally does not call the direct-inner
/// production operation.
fn z_domain_inner_reference(
    z: &Array3<Complex64>,
    port_a: usize,
    port_b: usize,
) -> Array3<Complex64> {
    let (nfreq, nports, _) = z.dim();
    let external: Vec<_> = (0..nports)
        .filter(|&port| port != port_a && port != port_b)
        .collect();
    let mut output = Array3::zeros((nfreq, external.len(), external.len()));

    for f in 0..nfreq {
        let denominator = z[[f, port_a, port_a]] - z[[f, port_a, port_b]] - z[[f, port_b, port_a]]
            + z[[f, port_b, port_b]];
        assert_ne!(
            denominator, ZERO,
            "Z-domain junction denominator must be nonsingular in this test"
        );
        for (row, &physical_row) in external.iter().enumerate() {
            let left = z[[f, physical_row, port_a]] - z[[f, physical_row, port_b]];
            for (column, &physical_column) in external.iter().enumerate() {
                let right = z[[f, port_b, physical_column]] - z[[f, port_a, physical_column]];
                output[[f, row, column]] =
                    z[[f, physical_row, physical_column]] + left * right / denominator;
            }
        }
    }
    output
}

fn block_diagonal(left: &Array3<Complex64>, right: &Array3<Complex64>) -> Array3<Complex64> {
    let (nfreq, left_ports, _) = left.dim();
    let (_, right_ports, _) = right.dim();
    Array3::from_shape_fn(
        (nfreq, left_ports + right_ports, left_ports + right_ports),
        |(f, row, column)| {
            if row < left_ports && column < left_ports {
                left[[f, row, column]]
            } else if row >= left_ports && column >= left_ports {
                right[[f, row - left_ports, column - left_ports]]
            } else {
                ZERO
            }
        },
    )
}

#[test]
fn direct_inner_uses_full_internal_block_and_preserves_source_and_order() {
    let frequency = vec![-0.0, 1.25e9, 2.5e9];
    let source = structured_network(frequency.clone(), 5);
    let source_s = source.s().clone();
    let source_z0 = source.z0().clone();
    let reduced = source.inner_connect_direct_power(1, 3).unwrap();

    let expected_s = direct_reference(source.s(), source.z0(), 1, 3);
    assert_array3_close(reduced.s(), &expected_s, 2.0e-12, 2.0e-12);
    assert_eq!(reduced.s().dim(), (3, 3, 3));
    assert_eq!(
        reduced.z0(),
        &Array2::from_shape_vec(
            (3, 3),
            vec![
                source.z0()[[0, 0]],
                source.z0()[[0, 2]],
                source.z0()[[0, 4]],
                source.z0()[[1, 0]],
                source.z0()[[1, 2]],
                source.z0()[[1, 4]],
                source.z0()[[2, 0]],
                source.z0()[[2, 2]],
                source.z0()[[2, 4]],
            ],
        )
        .unwrap()
    );
    for (actual, expected) in reduced.frequency().hz().iter().zip(&frequency) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(source.frequency().hz(), frequency.as_slice());
    assert_eq!(source.s(), &source_s);
    assert_eq!(source.z0(), &source_z0);
}

#[test]
fn direct_inner_matches_independent_z_domain_port_elimination() {
    let frequency = vec![0.9e9, 1.7e9];
    let mut source = structured_network(frequency, 4);
    let z0 = Array2::from_shape_fn((2, 4), |(f, port)| {
        c(
            43.0 + 8.0 * f as f64 + 4.0 * port as f64,
            -3.0 + 0.75 * port as f64,
        )
    });
    source = network(source.frequency().hz().to_vec(), source.s().clone(), z0);

    let source_z = source
        .to_z_power()
        .expect("the independent Z-domain source must be invertible");
    let expected_z = z_domain_inner_reference(&source_z, 1, 2);
    let reduced = source.inner_connect_direct_power(1, 2).unwrap();
    let reduced_z = reduced
        .to_z_power()
        .expect("the reduced Z-domain result must be invertible");
    assert_array3_close(&reduced_z, &expected_z, 3.0e-11, 3.0e-11);
}

#[test]
fn direct_inner_succeeds_when_whole_network_to_z_is_singular() {
    let mut s = Array3::zeros((1, 3, 3));
    // The external survivor (port 2) is an ideal open, making the whole
    // I-S system singular.  The selected internal block [1, 0.2] still gives
    // a finite two-coordinate direct junction system.
    s[[0, 0, 0]] = c(1.0, 0.0);
    s[[0, 1, 1]] = c(0.2, 0.0);
    s[[0, 2, 2]] = c(1.0, 0.0);
    let source = network(
        vec![1.0e9],
        s,
        Array2::from_shape_vec((1, 3), vec![c(50.0, 0.0), c(63.0, 0.0), c(77.0, 0.0)]).unwrap(),
    );
    assert!(matches!(
        source.to_z_power(),
        Err(Error::Singular {
            stage: ConversionStage::SToZ,
            frequency: 0,
            ..
        })
    ));

    let reduced = source.inner_connect_direct_power(0, 1).unwrap();
    assert_eq!(reduced.z0(), &Array2::from_elem((1, 1), c(77.0, 0.0)));
    assert_eq!(reduced.s()[[0, 0, 0]], c(1.0, 0.0));
}

#[test]
fn direct_inner_accepts_equal_complex_first_last_selected_refs_and_preserves_order() {
    let source = structured_network(vec![1.0e9, 1.9e9], 4);
    let mut z0 = source.z0().clone();
    for f in 0..2 {
        let selected = c(61.0 + f as f64, 7.0 - 0.5 * f as f64);
        z0[[f, 0]] = selected;
        z0[[f, 3]] = selected;
        z0[[f, 1]] = c(42.0 + f as f64, -2.0);
        z0[[f, 2]] = c(84.0 + f as f64, 3.0);
    }
    let source = network(
        source.frequency().hz().to_vec(),
        source.s().clone(),
        z0.clone(),
    );
    let reduced = source.inner_connect_direct_power(0, 3).unwrap();
    let expected_s = direct_reference(source.s(), source.z0(), 0, 3);
    assert_array3_close(reduced.s(), &expected_s, 3.0e-12, 3.0e-12);
    assert_eq!(
        reduced.z0(),
        &Array2::from_shape_vec((2, 2), vec![z0[[0, 1]], z0[[0, 2]], z0[[1, 1]], z0[[1, 2]]],)
            .unwrap()
    );
}

#[test]
fn direct_inner_is_invariant_under_selected_exchange_and_port_permutation() {
    let source = structured_network(vec![1.0e9, 1.8e9], 5);
    let direct = source.inner_connect_direct_power(1, 3).unwrap();
    let exchanged = source.inner_connect_direct_power(3, 1).unwrap();
    assert_eq!(direct.z0(), exchanged.z0());
    assert_array3_close(direct.s(), exchanged.s(), 3.0e-12, 3.0e-12);

    // New order is [old 4, old 1, old 0, old 3, old 2].  The selected
    // coordinates are therefore [1, 3], while the output order maps to
    // original survivors [4, 0, 2].
    let order = [4, 1, 0, 3, 2];
    let permuted = source.permute_ports(&order).unwrap();
    let permuted_reduced = permuted.inner_connect_direct_power(1, 3).unwrap();
    let survivor_order = [4, 0, 2];
    for f in 0..source.frequency().len() {
        for (row, &old_row) in survivor_order.iter().enumerate() {
            assert_eq!(permuted_reduced.z0()[[f, row]], source.z0()[[f, old_row]]);
            for (column, &old_column) in survivor_order.iter().enumerate() {
                assert_close(
                    permuted_reduced.s()[[f, row, column]],
                    direct.s()[[
                        f,
                        [0, 2, 4].iter().position(|&p| p == old_row).unwrap(),
                        [0, 2, 4].iter().position(|&p| p == old_column).unwrap(),
                    ]],
                    3.0e-12,
                    3.0e-12,
                );
            }
        }
    }
}

#[test]
fn direct_inner_satisfies_physical_vi_boundary_and_external_scattering() {
    let mut s = Array3::zeros((1, 3, 3));
    for row in 0..3 {
        for column in 0..3 {
            s[[0, row, column]] = c(
                0.04 + 0.011 * row as f64 - 0.007 * column as f64,
                -0.03 + 0.005 * row as f64 + 0.009 * column as f64,
            );
        }
    }
    let z0 =
        Array2::from_shape_vec((1, 3), vec![c(50.0, 4.0), c(-65.0, 3.0), c(72.0, -1.0)]).unwrap();
    let source = network(vec![1.0e9], s, z0);
    let reduced = source.inner_connect_direct_power(0, 1).unwrap();
    let external_incident = c(0.27, -0.19);

    let za = source.z0()[[0, 0]];
    let zb = source.z0()[[0, 1]];
    let qa = c(za.re.abs().sqrt() / za.re, 0.0);
    let qb = c(zb.re.abs().sqrt() / zb.re, 0.0);
    let c_matrix = [[qa, qb], [qa * za.conj(), -qb * zb.conj()]];
    let d_matrix = [[-qa, -qb], [qa * za, -qb * zb]];
    let s_ii = [
        source.s()[[0, 0, 0]],
        source.s()[[0, 0, 1]],
        source.s()[[0, 1, 0]],
        source.s()[[0, 1, 1]],
    ];
    let system = [
        [
            c_matrix[0][0] + d_matrix[0][0] * s_ii[0] + d_matrix[0][1] * s_ii[2],
            c_matrix[0][1] + d_matrix[0][0] * s_ii[1] + d_matrix[0][1] * s_ii[3],
        ],
        [
            c_matrix[1][0] + d_matrix[1][0] * s_ii[0] + d_matrix[1][1] * s_ii[2],
            c_matrix[1][1] + d_matrix[1][0] * s_ii[1] + d_matrix[1][1] * s_ii[3],
        ],
    ];
    let s_ie = [source.s()[[0, 0, 2]], source.s()[[0, 1, 2]]];
    let internal = solve_two_by_two(
        system,
        [
            -(d_matrix[0][0] * s_ie[0] + d_matrix[0][1] * s_ie[1]) * external_incident,
            -(d_matrix[1][0] * s_ie[0] + d_matrix[1][1] * s_ie[1]) * external_incident,
        ],
    );

    let incident = [internal[0], internal[1], external_incident];
    let reflected = (0..3)
        .map(|row| {
            (0..3)
                .map(|column| source.s()[[0, row, column]] * incident[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let current_a = qa * (incident[0] - reflected[0]);
    let current_b = qb * (incident[1] - reflected[1]);
    let voltage_a = qa * (za.conj() * incident[0] + za * reflected[0]);
    let voltage_b = qb * (zb.conj() * incident[1] + zb * reflected[1]);
    assert_close(voltage_a, voltage_b, 2.0e-12, 2.0e-12);
    assert_close(current_a, -current_b, 2.0e-12, 2.0e-12);

    let expected_external_reflected = reduced.s()[[0, 0, 0]] * external_incident;
    assert_close(reflected[2], expected_external_reflected, 2.0e-12, 2.0e-12);
}

#[test]
fn direct_inner_matches_direct_inter_network_on_block_diagonal_models() {
    let frequency = vec![1.0e9, 1.3e9];
    let left = structured_network(frequency.clone(), 3);
    let right = structured_network(frequency.clone(), 4);
    let combined = network(
        frequency.clone(),
        block_diagonal(left.s(), right.s()),
        Array2::from_shape_fn((frequency.len(), 7), |(f, port)| {
            if port < 3 {
                left.z0()[[f, port]]
            } else {
                right.z0()[[f, port - 3]]
            }
        }),
    );

    let separate = left.connect_direct_power(1, &right, 2).unwrap();
    let internal = combined.inner_connect_direct_power(1, 3 + 2).unwrap();
    assert_eq!(internal.z0(), separate.z0());
    assert_eq!(internal.frequency(), separate.frequency());
    assert_array3_close(internal.s(), separate.s(), 3.0e-12, 3.0e-12);
}

#[test]
fn direct_inner_agrees_with_matched_inner_on_shared_domain() {
    let frequency = vec![1.0e9, 1.6e9];
    let mut source = structured_network(frequency, 5);
    let mut z0 = source.z0().clone();
    for f in 0..z0.dim().0 {
        z0[[f, 1]] = c(73.5 + f as f64, 0.0);
        z0[[f, 4]] = c(73.5 + f as f64, 0.0);
    }
    source = network(source.frequency().hz().to_vec(), source.s().clone(), z0);
    let direct = source.inner_connect_direct_power(1, 4).unwrap();
    let matched = source.inner_connect_matched_power(1, 4).unwrap();
    assert_eq!(direct.z0(), matched.z0());
    assert_array3_close(direct.s(), matched.s(), 4.0e-12, 4.0e-12);
}

#[test]
fn direct_inner_handles_zero_selected_reference_sum_without_dividing_by_it() {
    let frequency = vec![1.0e9];
    let mut s = Array3::zeros((1, 4, 4));
    s[[0, 0, 1]] = c(0.12, -0.03);
    s[[0, 1, 0]] = c(-0.08, 0.02);
    s[[0, 1, 1]] = c(0.19, 0.01);
    s[[0, 2, 0]] = c(0.07, 0.04);
    s[[0, 3, 1]] = c(-0.06, -0.02);
    let z0 = Array2::from_shape_vec(
        (1, 4),
        vec![c(50.0, 0.0), c(50.0, 4.0), c(-50.0, -4.0), c(67.0, 0.0)],
    )
    .unwrap();
    let source = network(frequency, s, z0);
    let selected_sum = source.z0()[[0, 1]] + source.z0()[[0, 2]];
    assert_eq!(selected_sum, ZERO);
    let reduced = source.inner_connect_direct_power(1, 2).unwrap();
    assert!(reduced.s().iter().all(|value| value.is_finite()));
    assert_eq!(reduced.z0().dim(), (1, 2));
}

#[test]
fn direct_inner_reports_exact_singularity_and_accepts_near_singularity() {
    let frequency = vec![1.0e9];
    let z0 = Array2::from_elem((1, 3), c(50.0, 0.0));
    let mut exact_s = Array3::zeros((1, 3, 3));
    exact_s[[0, 0, 0]] = c(1.0, 0.0);
    exact_s[[0, 1, 1]] = c(1.0, 0.0);
    let exact = network(frequency.clone(), exact_s, z0.clone());
    assert!(matches!(
        exact.inner_connect_direct_power(0, 1),
        Err(Error::SingularDirectInnerConnection {
            frequency: 0,
            port_a: 0,
            port_b: 1,
            ..
        })
    ));

    let mut near_s = Array3::zeros((1, 3, 3));
    near_s[[0, 0, 0]] = c(1.0 - 2.0_f64.powi(-40), 0.0);
    near_s[[0, 1, 1]] = c(1.0 - 2.0_f64.powi(-41), 0.0);
    near_s[[0, 2, 0]] = c(1.0e-7, 0.0);
    near_s[[0, 2, 1]] = c(-2.0e-7, 0.0);
    let near = network(frequency, near_s, z0);
    let reduced = near.inner_connect_direct_power(0, 1).unwrap();
    assert!(reduced.s().iter().all(|value| value.is_finite()));
}

#[test]
fn direct_inner_rejects_invalid_inputs_without_panicking() {
    let valid = structured_network(vec![1.0e9], 3);
    assert_eq!(
        valid.inner_connect_direct_power(3, 1).unwrap_err(),
        Error::InvalidDirectInnerConnectionPort { port: 3, nports: 3 }
    );
    assert_eq!(
        valid.inner_connect_direct_power(1, 1).unwrap_err(),
        Error::IdenticalDirectInnerConnectionPorts {
            port_a: 1,
            port_b: 1,
        }
    );
    let two_port = structured_network(vec![1.0e9], 2);
    assert_eq!(
        two_port.inner_connect_direct_power(0, 1).unwrap_err(),
        Error::NoExternalDirectInnerConnectionPorts
    );

    let mut malformed_s = serde_json::to_value(&valid).unwrap();
    malformed_s["s"]["dim"] = json!([1, 3, 2]);
    let data = malformed_s["s"]["data"].as_array().unwrap().clone();
    malformed_s["s"]["data"] = json!(data.into_iter().take(6).collect::<Vec<_>>());
    let malformed_s: Network = serde_json::from_value(malformed_s).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        malformed_s.inner_connect_direct_power(0, 1)
    }));
    assert!(result.is_ok());
    assert!(matches!(
        result.unwrap(),
        Err(Error::InvalidDirectInnerConnectionSShape { .. })
    ));

    let mut malformed_z0 = serde_json::to_value(&valid).unwrap();
    malformed_z0["z0"]["dim"] = json!([1, 2]);
    let data = malformed_z0["z0"]["data"].as_array().unwrap().clone();
    malformed_z0["z0"]["data"] = json!(data.into_iter().take(2).collect::<Vec<_>>());
    let malformed_z0: Network = serde_json::from_value(malformed_z0).unwrap();
    assert_eq!(
        malformed_z0.inner_connect_direct_power(0, 1).unwrap_err(),
        Error::InvalidDirectInnerConnectionZ0Shape { shape: vec![1, 2] }
    );

    let mut malformed_frequency = serde_json::to_value(&valid).unwrap();
    malformed_frequency["frequency"]["hz"] = json!([1.0e9, 2.0e9]);
    let malformed_frequency: Network = serde_json::from_value(malformed_frequency).unwrap();
    assert_eq!(
        malformed_frequency
            .inner_connect_direct_power(0, 1)
            .unwrap_err(),
        Error::DirectInnerConnectionFrequencyShape {
            expected: 1,
            actual: 2,
        }
    );
}

#[test]
fn direct_inner_rejects_nonfinite_and_zero_real_values_on_any_port() {
    let source = structured_network(vec![1.0e9], 3);

    let mut bad_frequency = source.frequency().hz().to_vec();
    bad_frequency[0] = f64::INFINITY;
    let bad_frequency = network(bad_frequency, source.s().clone(), source.z0().clone());
    assert!(matches!(
        bad_frequency.inner_connect_direct_power(0, 1),
        Err(Error::NonFiniteDirectInnerConnectionFrequency { index: 0, .. })
    ));

    let mut bad_s = source.s().clone();
    bad_s[[0, 2, 0]] = c(f64::NAN, 0.0);
    let bad_s = network(source.frequency().hz().to_vec(), bad_s, source.z0().clone());
    assert_eq!(
        bad_s.inner_connect_direct_power(0, 1).unwrap_err(),
        Error::NonFiniteDirectInnerConnectionS {
            frequency: 0,
            row: 2,
            column: 0,
        }
    );

    let mut bad_z0 = source.z0().clone();
    bad_z0[[0, 2]] = c(f64::INFINITY, 0.0);
    let bad_z0 = network(source.frequency().hz().to_vec(), source.s().clone(), bad_z0);
    assert_eq!(
        bad_z0.inner_connect_direct_power(0, 1).unwrap_err(),
        Error::NonFiniteDirectInnerConnectionZ0 {
            frequency: 0,
            port: 2,
        }
    );

    let mut zero_real = source.z0().clone();
    zero_real[[0, 1]] = c(0.0, 17.0);
    let zero_real = network(
        source.frequency().hz().to_vec(),
        source.s().clone(),
        zero_real,
    );
    assert_eq!(
        zero_real.inner_connect_direct_power(0, 1).unwrap_err(),
        Error::ZeroRealDirectInnerConnectionReferenceImpedance {
            frequency: 0,
            port: 1,
        }
    );

    let mut zero_real_survivor = source.z0().clone();
    zero_real_survivor[[0, 2]] = c(-0.0, 17.0);
    let zero_real_survivor = network(
        source.frequency().hz().to_vec(),
        source.s().clone(),
        zero_real_survivor,
    );
    assert_eq!(
        zero_real_survivor
            .inner_connect_direct_power(0, 1)
            .unwrap_err(),
        Error::ZeroRealDirectInnerConnectionReferenceImpedance {
            frequency: 0,
            port: 2,
        }
    );
}

#[test]
fn direct_inner_reports_checked_overflow_with_selected_port_context() {
    let mut s = Array3::zeros((1, 3, 3));
    s[[0, 0, 2]] = c(f64::MAX, 0.0);
    let source = network(vec![1.0e9], s, Array2::from_elem((1, 3), c(50.0, 0.0)));
    assert!(matches!(
        source.inner_connect_direct_power(0, 1),
        Err(Error::NonFiniteDirectInnerConnectionComputation {
            frequency: 0,
            port_a: 0,
            port_b: 1,
            ..
        })
    ));
}
