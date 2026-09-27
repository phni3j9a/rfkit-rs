use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, Network, PortLoad};
use serde_json::json;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn finite_loads(loads: &[Complex64]) -> Vec<PortLoad> {
    loads.iter().copied().map(PortLoad::ImpedanceOhm).collect()
}

fn network(frequency_hz: Vec<f64>, s: Array3<Complex64>, z0: Array2<Complex64>) -> Network {
    Network::new(Frequency::from_hz(frequency_hz).unwrap(), s, z0).unwrap()
}

fn assert_array3_close(
    actual: &Array3<Complex64>,
    expected: &Array3<Complex64>,
    rtol: f64,
    atol: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual - expected).norm();
        let bound = atol + rtol * expected.norm();
        assert!(
            difference <= bound,
            "index {index}: actual={actual:?}, expected={expected:?}, difference={difference:e}, bound={bound:e}"
        );
    }
}

fn matrix_vector(
    matrix: &Array3<Complex64>,
    frequency: usize,
    vector: &Array1<Complex64>,
) -> Array1<Complex64> {
    Array1::from_shape_fn(vector.len(), |row| {
        (0..vector.len())
            .map(|column| matrix[[frequency, row, column]] * vector[column])
            .sum()
    })
}

fn expected_termination(
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    port: usize,
    load: &[Complex64],
) -> (Array3<Complex64>, Array2<Complex64>) {
    let (nfreq, nports, _) = s.dim();
    let survivors: Vec<_> = (0..nports).filter(|&candidate| candidate != port).collect();
    let mut output = Array3::zeros((nfreq, nports - 1, nports - 1));
    let mut output_z0 = Array2::zeros((nfreq, nports - 1));
    for frequency in 0..nfreq {
        let z = z0[[frequency, port]];
        let c = load[frequency] - z;
        let d = load[frequency] + z.conj();
        let feedback = c / (d - c * s[[frequency, port, port]]);
        for (row, &source_row) in survivors.iter().enumerate() {
            output_z0[[frequency, row]] = z0[[frequency, source_row]];
            for (column, &source_column) in survivors.iter().enumerate() {
                output[[frequency, row, column]] = s[[frequency, source_row, source_column]]
                    + s[[frequency, source_row, port]]
                        * feedback
                        * s[[frequency, port, source_column]];
            }
        }
    }
    (output, output_z0)
}

fn expected_profile_termination(
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    port: usize,
    loads: &[PortLoad],
) -> (Array3<Complex64>, Array2<Complex64>) {
    let (nfreq, nports, _) = s.dim();
    let survivors: Vec<_> = (0..nports).filter(|&candidate| candidate != port).collect();
    let mut output = Array3::zeros((nfreq, nports - 1, nports - 1));
    let mut output_z0 = Array2::zeros((nfreq, nports - 1));
    for frequency in 0..nfreq {
        let selected_s = s[[frequency, port, port]];
        let feedback = match loads[frequency] {
            PortLoad::ImpedanceOhm(load) => {
                let reference = z0[[frequency, port]];
                let c = load - reference;
                let d = load + reference.conj();
                c / (d - c * selected_s)
            }
            PortLoad::Open => c(1.0, 0.0) / (c(1.0, 0.0) - selected_s),
            _ => unreachable!("unknown future PortLoad variant in test"),
        };
        for (row, &source_row) in survivors.iter().enumerate() {
            output_z0[[frequency, row]] = z0[[frequency, source_row]];
            for (column, &source_column) in survivors.iter().enumerate() {
                output[[frequency, row, column]] = s[[frequency, source_row, source_column]]
                    + s[[frequency, source_row, port]]
                        * feedback
                        * s[[frequency, port, source_column]];
            }
        }
    }
    (output, output_z0)
}

fn asymmetric_case() -> (
    Vec<f64>,
    Array3<Complex64>,
    Array2<Complex64>,
    Vec<Complex64>,
) {
    let frequency = vec![f64::NAN, -0.0, 2.75e9];
    let s = Array3::from_shape_fn((3, 4, 4), |(frequency, row, column)| {
        c(
            0.01 * (frequency + 1) as f64 + 0.007 * (row + 1) as f64 - 0.003 * (column + 1) as f64,
            0.004 * (row + 1) as f64 - 0.002 * (column + 1) as f64 + 0.001 * frequency as f64,
        )
    });
    let z0 = Array2::from_shape_fn((3, 4), |(frequency, port)| {
        c(
            [42.0, -57.0, 71.0, 88.0][port] + 1.5 * frequency as f64,
            [2.0, -1.5, 3.25, -4.0][port] + 0.2 * frequency as f64,
        )
    });
    let load = vec![c(0.0, 0.0), c(38.0, 12.0), c(73.0, -9.0)];
    (frequency, s, z0, load)
}

#[test]
fn termination_matches_real_reference_analytical_values_and_survivor_order() {
    let frequency = vec![1.0e9];
    let s = Array3::from_shape_vec(
        (1, 2, 2),
        vec![c(0.1, 0.02), c(0.3, -0.1), c(-0.2, 0.15), c(0.4, -0.03)],
    )
    .unwrap();
    let z0 = Array2::from_shape_vec((1, 2), vec![c(50.0, 0.0), c(75.0, 0.0)]).unwrap();
    let source = network(frequency.clone(), s.clone(), z0.clone());
    let load = [c(125.0, 0.0)];
    let actual = source
        .terminate_port_power(0, &finite_loads(&load))
        .unwrap();
    let reflection =
        c(125.0 - 50.0, 0.0) / (c(125.0 + 50.0, 0.0) - c(125.0 - 50.0, 0.0) * c(0.1, 0.02));
    let expected = c(0.4, -0.03) + c(-0.2, 0.15) * reflection * c(0.3, -0.1);
    assert_eq!(actual.frequency().hz(), frequency.as_slice());
    assert_eq!(actual.z0(), &Array2::from_elem((1, 1), c(75.0, 0.0)));
    assert!(
        (actual.s()[[0, 0, 0]] - expected).norm() < 1.0e-15,
        "actual={:?}, expected={:?}, diff={:?}",
        actual.s()[[0, 0, 0]],
        expected,
        (actual.s()[[0, 0, 0]] - expected).norm()
    );

    // Selecting the last port leaves the first port in place and uses the
    // same physical formula in the opposite survivor direction.
    let actual = source
        .terminate_port_power(1, &finite_loads(&load))
        .unwrap();
    let reflection = (c(125.0, 0.0) - c(75.0, 0.0))
        / (c(125.0 + 75.0, 0.0) - (c(125.0, 0.0) - c(75.0, 0.0)) * c(0.4, -0.03));
    let expected = c(0.1, 0.02) + c(0.3, -0.1) * reflection * c(-0.2, 0.15);
    assert_eq!(actual.z0(), &Array2::from_elem((1, 1), c(50.0, 0.0)));
    assert!(
        (actual.s()[[0, 0, 0]] - expected).norm() < 1.0e-15,
        "actual={:?}, expected={:?}, diff={:?}",
        actual.s()[[0, 0, 0]],
        expected,
        (actual.s()[[0, 0, 0]] - expected).norm()
    );
}

#[test]
fn open_termination_matches_analytical_two_port_and_preserves_inputs() {
    let frequency = vec![1.0e9];
    let s = Array3::from_shape_vec(
        (1, 2, 2),
        vec![
            c(0.17, -0.04),
            c(0.31, 0.09),
            c(-0.22, 0.13),
            c(0.42, -0.02),
        ],
    )
    .unwrap();
    let z0 = Array2::from_shape_vec((1, 2), vec![c(47.0, 8.0), c(-63.0, 2.5)]).unwrap();
    let source = network(frequency.clone(), s.clone(), z0.clone());
    let source_snapshot = source.clone();
    let loads = [PortLoad::Open];
    let loads_snapshot = loads;

    let actual = source.terminate_port_power(0, &loads).unwrap();
    let expected =
        s[[0, 1, 1]] + s[[0, 1, 0]] * (c(1.0, 0.0) / (c(1.0, 0.0) - s[[0, 0, 0]])) * s[[0, 0, 1]];
    assert_eq!(actual.frequency().hz(), frequency.as_slice());
    assert_eq!(actual.z0(), &Array2::from_elem((1, 1), z0[[0, 1]]));
    assert!((actual.s()[[0, 0, 0]] - expected).norm() < 1.0e-15);
    assert_eq!(source.s(), source_snapshot.s());
    assert_eq!(source.z0(), source_snapshot.z0());
    for (actual, expected) in source
        .frequency()
        .hz()
        .iter()
        .zip(source_snapshot.frequency().hz())
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(loads, loads_snapshot);
}

#[test]
fn open_termination_supports_mixed_profiles_asymmetric_middle_port_and_exact_metadata() {
    let (frequency, s, z0, finite) = asymmetric_case();
    let source = network(frequency.clone(), s.clone(), z0.clone());
    let source_snapshot = source.clone();
    let loads = [
        PortLoad::ImpedanceOhm(finite[0]),
        PortLoad::Open,
        PortLoad::ImpedanceOhm(finite[2]),
    ];
    let loads_snapshot = loads;
    let actual = source.terminate_port_power(2, &loads).unwrap();
    let (expected_s, expected_z0) = expected_profile_termination(&s, &z0, 2, &loads);

    assert_eq!(actual.z0(), &expected_z0);
    assert_array3_close(actual.s(), &expected_s, 3.0e-15, 3.0e-15);
    for (actual, expected) in actual.frequency().hz().iter().zip(frequency.iter()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(source.s(), source_snapshot.s());
    assert_eq!(source.z0(), source_snapshot.z0());
    for (actual, expected) in source
        .frequency()
        .hz()
        .iter()
        .zip(source_snapshot.frequency().hz())
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(loads, loads_snapshot);
}

#[test]
fn termination_supports_n_ports_complex_references_and_load_boundaries() {
    let (frequency, s, z0, load) = asymmetric_case();
    let source = network(frequency.clone(), s.clone(), z0.clone());
    let source_snapshot = source.clone();
    let load_snapshot = load.clone();
    let actual = source
        .terminate_port_power(2, &finite_loads(&load))
        .unwrap();
    let (expected_s, expected_z0) = expected_termination(&s, &z0, 2, &load);
    for (actual, expected) in actual.frequency().hz().iter().zip(frequency.iter()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(actual.z0(), &expected_z0);
    assert_array3_close(actual.s(), &expected_s, 3.0e-15, 3.0e-15);
    for (actual, expected) in source
        .frequency()
        .hz()
        .iter()
        .zip(source_snapshot.frequency().hz())
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(source.s(), source_snapshot.s());
    assert_eq!(source.z0(), source_snapshot.z0());
    assert_eq!(load, load_snapshot);

    // Negative-resistance references and loads are accepted as finite
    // algebraic values; zero and purely reactive loads are also in-domain.
    let frequency = vec![1.0, 2.0, 3.0];
    let s = Array3::from_shape_fn((3, 3, 3), |(f, row, column)| {
        c(0.02 * (f + row + 1) as f64, -0.01 * (column + 1) as f64)
    });
    let z0 = Array2::from_shape_fn((3, 3), |(f, port)| {
        c([-40.0, 55.0, 80.0][port] + f as f64, [2.0, -3.0, 4.0][port])
    });
    let loads = [c(0.0, 0.0), c(0.0, 15.0), c(-12.0, -7.0)];
    let source = network(frequency, s.clone(), z0.clone());
    // Port 0 itself has a finite negative-real complex source reference.  An
    // `abs(Re(z0))` shortcut or a real-only reflection formula would change
    // this direct boundary result, so compare the full c/d/den expression.
    let actual = source
        .terminate_port_power(0, &finite_loads(&loads))
        .unwrap();
    let (expected_s, expected_z0) = expected_termination(&s, &z0, 0, &loads);
    assert_array3_close(actual.s(), &expected_s, 3.0e-15, 3.0e-15);
    assert_eq!(actual.z0(), &expected_z0);
}

#[test]
fn termination_zl_equal_reference_is_submatrix_and_complex_short_is_oriented() {
    let frequency = vec![1.0e9, 2.0e9];
    let s = Array3::from_shape_fn((2, 3, 3), |(f, row, column)| {
        c(
            0.03 * (f + row + 1) as f64 - 0.01 * column as f64,
            0.02 * (column + 1) as f64 - 0.005 * row as f64,
        )
    });
    let z0 = Array2::from_shape_vec(
        (2, 3),
        vec![
            c(42.0, 7.0),
            c(61.0, -4.0),
            c(-73.0, 3.0),
            c(44.0, 6.0),
            c(63.0, -5.0),
            c(-75.0, 2.0),
        ],
    )
    .unwrap();
    let source = network(frequency, s.clone(), z0.clone());
    let equal_reference = [z0[[0, 1]], z0[[1, 1]]];
    let reduced = source
        .terminate_port_power(1, &finite_loads(&equal_reference))
        .unwrap();
    let expected = Array3::from_shape_fn((2, 2, 2), |(f, row, column)| {
        let source_row = [0, 2][row];
        let source_column = [0, 2][column];
        s[[f, source_row, source_column]]
    });
    assert_array3_close(reduced.s(), &expected, 1.0e-15, 1.0e-15);

    let source_reference = c(50.0, 13.0);
    let source = network(
        vec![1.0],
        Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.1, 0.0), c(0.25, -0.1), c(-0.2, 0.3), c(0.05, 0.0)],
        )
        .unwrap(),
        Array2::from_shape_vec((1, 2), vec![source_reference, c(72.0, 1.0)]).unwrap(),
    );
    let reduced = source
        .terminate_port_power(0, &finite_loads(&[c(0.0, 0.0)]))
        .unwrap();
    let reflection = -source_reference / (source_reference.conj() + source_reference * c(0.1, 0.0));
    let expected = c(0.05, 0.0) + c(-0.2, 0.3) * reflection * c(0.25, -0.1);
    assert!(
        (reduced.s()[[0, 0, 0]] - expected).norm() < 2.0e-15,
        "actual={:?}, expected={:?}, diff={:?}",
        reduced.s()[[0, 0, 0]],
        expected,
        (reduced.s()[[0, 0, 0]] - expected).norm()
    );
}

#[test]
fn termination_matches_physical_vi_boundary_and_scattering_relation() {
    let z0 =
        Array2::from_shape_vec((1, 3), vec![c(50.0, 4.0), c(62.0, -3.0), c(-71.0, 5.0)]).unwrap();
    let s = Array3::from_shape_vec(
        (1, 3, 3),
        vec![
            c(0.05, 0.01),
            c(0.12, -0.04),
            c(-0.08, 0.03),
            c(-0.2, 0.07),
            c(0.02, -0.01),
            c(0.15, 0.04),
            c(0.11, -0.02),
            c(-0.09, 0.08),
            c(0.04, 0.03),
        ],
    )
    .unwrap();
    let source = network(vec![1.0e9], s.clone(), z0.clone());
    let load = [c(38.0, 12.0)];
    let selected = 1;
    let reduced = source
        .terminate_port_power(selected, &finite_loads(&load))
        .unwrap();
    let external_a = Array1::from_vec(vec![c(0.3, -0.2), c(-0.17, 0.25)]);
    let survivors = [0, 2];
    let z = z0[[0, selected]];
    let c_boundary = load[0] - z;
    let d_boundary = load[0] + z.conj();
    let feedback = c_boundary / (d_boundary - c_boundary * s[[0, selected, selected]]);
    let selected_a = feedback
        * (0..2)
            .map(|index| s[[0, selected, survivors[index]]] * external_a[index])
            .sum::<Complex64>();
    let mut full_a = Array1::zeros(3);
    full_a[selected] = selected_a;
    for (index, &survivor) in survivors.iter().enumerate() {
        full_a[survivor] = external_a[index];
    }
    let full_b = matrix_vector(&s, 0, &full_a);
    let selected_b = full_b[selected];
    assert!((d_boundary * selected_a - c_boundary * selected_b).norm() < 4.0e-15);
    let normalization = z.re.abs().sqrt();
    let selected_i = normalization / z.re * (selected_a - selected_b);
    let selected_v =
        normalization * (selected_a + selected_b) - Complex64::new(0.0, z.im) * selected_i;
    assert!((selected_v + load[0] * selected_i).norm() < 4.0e-15);

    let reduced_b = matrix_vector(reduced.s(), 0, &external_a);
    for (index, &survivor) in survivors.iter().enumerate() {
        assert!((reduced_b[index] - full_b[survivor]).norm() < 5.0e-15);
    }
}

#[test]
fn open_termination_reconstructs_zero_selected_current_and_external_response() {
    let z0 =
        Array2::from_shape_vec((1, 3), vec![c(44.0, 3.0), c(-62.0, -5.0), c(79.0, 7.0)]).unwrap();
    let s = Array3::from_shape_vec(
        (1, 3, 3),
        vec![
            c(0.06, -0.02),
            c(0.14, 0.03),
            c(-0.11, 0.07),
            c(-0.21, 0.04),
            c(0.18, -0.06),
            c(0.16, 0.02),
            c(0.09, 0.05),
            c(-0.13, 0.08),
            c(0.03, -0.01),
        ],
    )
    .unwrap();
    let source = network(vec![1.0e9], s.clone(), z0.clone());
    let reduced = source.terminate_port_power(1, &[PortLoad::Open]).unwrap();
    let external_a = Array1::from_vec(vec![c(0.27, -0.19), c(-0.16, 0.23)]);
    let survivors = [0, 2];
    let selected_a = (c(1.0, 0.0) / (c(1.0, 0.0) - s[[0, 1, 1]]))
        * (0..2)
            .map(|index| s[[0, 1, survivors[index]]] * external_a[index])
            .sum::<Complex64>();
    let mut full_a = Array1::zeros(3);
    full_a[1] = selected_a;
    for (index, &survivor) in survivors.iter().enumerate() {
        full_a[survivor] = external_a[index];
    }
    let full_b = matrix_vector(&s, 0, &full_a);
    assert!((full_a[1] - full_b[1]).norm() < 5.0e-15);

    let selected_reference = z0[[0, 1]];
    let selected_i =
        selected_reference.re.abs().sqrt() / selected_reference.re * (full_a[1] - full_b[1]);
    let selected_v = selected_reference.re.abs().sqrt() * (full_a[1] + full_b[1])
        - c(0.0, selected_reference.im) * selected_i;
    assert!(selected_i.norm() < 5.0e-15);
    assert!(selected_v.norm() > 0.0);

    let reduced_b = matrix_vector(reduced.s(), 0, &external_a);
    for (index, &survivor) in survivors.iter().enumerate() {
        assert!((reduced_b[index] - full_b[survivor]).norm() < 5.0e-15);
    }
}

#[test]
fn termination_matches_z_schur_and_existing_matched_connection_on_shared_domain()
-> rfkit_core::Result<()> {
    let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let z0 =
        Array2::from_shape_vec((1, 3), vec![c(50.0, 0.0), c(60.0, 0.0), c(70.0, 0.0)]).unwrap();
    let z = Array3::from_shape_vec(
        (1, 3, 3),
        vec![
            c(80.0, 2.0),
            c(4.0, -1.0),
            c(7.0, 0.5),
            c(3.0, 1.5),
            c(91.0, -3.0),
            c(5.0, 2.0),
            c(6.0, -0.25),
            c(8.0, 1.0),
            c(104.0, 4.0),
        ],
    )
    .unwrap();
    let source = Network::from_z_power(frequency.clone(), z.clone(), z0.clone()).unwrap();
    let load = c(37.0, 11.0);
    let selected = 1;
    let reduced = source
        .terminate_port_power(selected, &finite_loads(&[load]))
        .unwrap();
    let mut z_reduced = Array3::zeros((1, 2, 2));
    let survivors = [0, 2];
    for (row, &source_row) in survivors.iter().enumerate() {
        for (column, &source_column) in survivors.iter().enumerate() {
            z_reduced[[0, row, column]] = z[[0, source_row, source_column]]
                - z[[0, source_row, selected]]
                    * (c(1.0, 0.0) / (z[[0, selected, selected]] + load))
                    * z[[0, selected, source_column]];
        }
    }
    let expected = Network::from_z_power(
        frequency.clone(),
        z_reduced,
        Array2::from_shape_vec((1, 2), vec![z0[[0, 0]], z0[[0, 2]]]).unwrap(),
    )?;
    assert_array3_close(reduced.s(), expected.s(), 4.0e-13, 4.0e-13);

    let mut load_s = Array3::zeros((1, 1, 1));
    load_s[[0, 0, 0]] = (load - c(60.0, 0.0)) / (load + c(60.0, 0.0));
    let load_network = network(vec![1.0e9], load_s, Array2::from_elem((1, 1), c(60.0, 0.0)));
    let connected = source.connect_power(selected, &load_network, 0).unwrap();
    assert_array3_close(reduced.s(), connected.s(), 5.0e-13, 5.0e-13);
    assert_eq!(reduced.z0(), connected.z0());
    Ok(())
}

#[test]
fn termination_is_covariant_under_port_permutation_and_handles_direct_singular_source() {
    let frequency = vec![1.0e9];
    let s = Array3::from_shape_fn((1, 4, 4), |(_, row, column)| {
        c(
            0.01 * (10 * row + column + 1) as f64,
            -0.002 * (row + column) as f64,
        )
    });
    let z0 = Array2::from_shape_vec(
        (1, 4),
        vec![c(42.0, 1.0), c(-55.0, 2.0), c(71.0, -3.0), c(88.0, 4.0)],
    )
    .unwrap();
    let source = network(frequency, s, z0);
    let direct = source
        .terminate_port_power(2, &finite_loads(&[c(33.0, -6.0)]))
        .unwrap();
    let permuted = source.permute_ports(&[2, 0, 3, 1]).unwrap();
    let permuted_reduced = permuted
        .terminate_port_power(0, &finite_loads(&[c(33.0, -6.0)]))
        .unwrap()
        .permute_ports(&[0, 2, 1])
        .unwrap();
    assert_array3_close(direct.s(), permuted_reduced.s(), 2.0e-15, 2.0e-15);
    assert_eq!(direct.z0(), permuted_reduced.z0());

    let direct_open = source.terminate_port_power(2, &[PortLoad::Open]).unwrap();
    let permuted_open = permuted
        .terminate_port_power(0, &[PortLoad::Open])
        .unwrap()
        .permute_ports(&[0, 2, 1])
        .unwrap();
    assert_array3_close(direct_open.s(), permuted_open.s(), 2.0e-15, 2.0e-15);
    assert_eq!(direct_open.z0(), permuted_open.z0());

    // The ideal thru has singular whole-network Z/Y conversion, while direct
    // finite-load elimination remains well-defined.
    let thru = network(
        vec![1.0],
        Array3::from_shape_vec(
            (1, 2, 2),
            vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        )
        .unwrap(),
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    );
    assert!(matches!(thru.to_z_power(), Err(Error::Singular { .. })));
    let terminated = thru
        .terminate_port_power(0, &finite_loads(&[c(75.0, 0.0)]))
        .unwrap();
    assert!((terminated.s()[[0, 0, 0]] - c(0.2, 0.0)).norm() < 1.0e-15);
}

#[test]
fn termination_accepts_d_zero_and_exact_frequency_labels() {
    let source_reference = c(50.0, 13.0);
    let frequency = vec![f64::NAN, -0.0];
    let s = Array3::from_shape_fn((2, 2, 2), |(f, row, column)| {
        c(
            0.03 + 0.01 * f as f64 + 0.02 * row as f64,
            -0.01 * column as f64,
        )
    });
    let z0 = Array2::from_shape_vec(
        (2, 2),
        vec![
            source_reference,
            c(72.0, 0.0),
            source_reference,
            c(72.0, 0.0),
        ],
    )
    .unwrap();
    let source = network(frequency.clone(), s, z0);
    let load = [
        -source_reference.conj(),
        c(-source_reference.re, source_reference.im),
    ];
    let result = source
        .terminate_port_power(0, &finite_loads(&load))
        .unwrap();
    assert_eq!(result.frequency().hz()[0].to_bits(), frequency[0].to_bits());
    assert_eq!(result.frequency().hz()[1].to_bits(), frequency[1].to_bits());
}

#[test]
fn termination_reports_validation_errors_without_panicking() {
    let valid = network(
        vec![1.0e9, 2.0e9],
        Array3::zeros((2, 2, 2)),
        Array2::from_elem((2, 2), c(50.0, 0.0)),
    );
    assert_eq!(
        valid
            .terminate_port_power(0, &finite_loads(&[c(1.0, 0.0)]))
            .unwrap_err(),
        Error::TerminationLoadLengthMismatch {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(
        valid
            .terminate_port_power(2, &finite_loads(&[c(1.0, 0.0), c(1.0, 0.0)]))
            .unwrap_err(),
        Error::InvalidTerminationPort { port: 2, nports: 2 }
    );
    let one_port = network(
        vec![1.0],
        Array3::zeros((1, 1, 1)),
        Array2::from_elem((1, 1), c(50.0, 0.0)),
    );
    assert_eq!(
        one_port
            .terminate_port_power(0, &finite_loads(&[c(1.0, 0.0)]))
            .unwrap_err(),
        Error::NoTerminationSurvivors { nports: 1 }
    );

    let nonfinite_s = network(
        vec![1.0],
        Array3::from_elem((1, 2, 2), c(f64::NAN, 0.0)),
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    );
    assert_eq!(
        nonfinite_s
            .terminate_port_power(0, &finite_loads(&[c(1.0, 0.0)]))
            .unwrap_err(),
        Error::NonFiniteTerminationS {
            frequency: 0,
            row: 0,
            column: 0,
        }
    );
    let nonfinite_z0 = network(
        vec![1.0],
        Array3::zeros((1, 2, 2)),
        Array2::from_shape_vec((1, 2), vec![c(50.0, 0.0), c(f64::INFINITY, 0.0)]).unwrap(),
    );
    assert_eq!(
        nonfinite_z0
            .terminate_port_power(0, &finite_loads(&[c(1.0, 0.0)]))
            .unwrap_err(),
        Error::NonFiniteTerminationZ0 {
            frequency: 0,
            port: 1,
        }
    );
    let zero_real = network(
        vec![1.0],
        Array3::zeros((1, 2, 2)),
        Array2::from_elem((1, 2), c(0.0, 2.0)),
    );
    assert_eq!(
        zero_real
            .terminate_port_power(0, &finite_loads(&[c(1.0, 0.0)]))
            .unwrap_err(),
        Error::ZeroRealTerminationReferenceImpedance {
            frequency: 0,
            port: 0,
        }
    );
    let nonfinite_load =
        valid.terminate_port_power(0, &finite_loads(&[c(f64::NAN, 0.0), c(1.0, 0.0)]));
    assert_eq!(
        nonfinite_load.unwrap_err(),
        Error::NonFiniteTerminationLoad { frequency: 0 }
    );
    let open_sentinel =
        valid.terminate_port_power(0, &finite_loads(&[c(f64::INFINITY, 0.0), c(1.0, 0.0)]));
    assert_eq!(
        open_sentinel.unwrap_err(),
        Error::NonFiniteTerminationLoad { frequency: 0 }
    );
}

#[test]
fn termination_rejects_malformed_serde_sources_without_panicking() {
    let valid = network(
        vec![1.0e9],
        Array3::zeros((1, 2, 2)),
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    );
    let mut empty_frequency = serde_json::to_value(&valid).unwrap();
    empty_frequency["frequency"]["hz"] = json!([]);
    let empty_frequency: Network = serde_json::from_value(empty_frequency).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        empty_frequency.terminate_port_power(0, &[PortLoad::Open])
    }));
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().unwrap_err(),
        Error::EmptyTerminationFrequency
    );

    let mut frequency_mismatch = serde_json::to_value(&valid).unwrap();
    frequency_mismatch["frequency"]["hz"] = json!([1.0e9, 2.0e9]);
    let frequency_mismatch: Network = serde_json::from_value(frequency_mismatch).unwrap();
    assert_eq!(
        frequency_mismatch
            .terminate_port_power(0, &finite_loads(&[c(1.0, 0.0), c(1.0, 0.0)]))
            .unwrap_err(),
        Error::TerminationFrequencyLengthMismatch {
            expected: 1,
            actual: 2,
        }
    );

    let mut malformed_s = serde_json::to_value(&valid).unwrap();
    let scalar = malformed_s["s"]["data"][0].clone();
    malformed_s["s"]["dim"] = json!([1, 1, 2]);
    malformed_s["s"]["data"] = json!([scalar.clone(), scalar]);
    let malformed_s: Network = serde_json::from_value(malformed_s).unwrap();
    assert_eq!(
        malformed_s
            .terminate_port_power(0, &finite_loads(&[c(1.0, 0.0)]))
            .unwrap_err(),
        Error::InvalidTerminationSShape {
            shape: vec![1, 1, 2]
        }
    );

    let mut malformed_z0 = serde_json::to_value(&valid).unwrap();
    malformed_z0["z0"]["dim"] = json!([1, 1]);
    malformed_z0["z0"]["data"] = json!([malformed_z0["z0"]["data"][0].clone()]);
    let malformed_z0: Network = serde_json::from_value(malformed_z0).unwrap();
    assert_eq!(
        malformed_z0
            .terminate_port_power(0, &finite_loads(&[c(1.0, 0.0)]))
            .unwrap_err(),
        Error::InvalidTerminationZ0Shape { shape: vec![1, 1] }
    );
}

#[test]
fn termination_rejects_exact_singularity_and_reports_finite_arithmetic_overflow() {
    let mut s = Array3::zeros((1, 2, 2));
    s[[0, 0, 0]] = c(-1.0, 0.0);
    let singular = network(
        vec![1.0],
        s.clone(),
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    );
    assert_eq!(
        singular
            .terminate_port_power(0, &finite_loads(&[c(0.0, 0.0)]))
            .unwrap_err(),
        Error::SingularTermination {
            frequency: 0,
            port: 0,
        }
    );
    s[[0, 0, 0]] = c(-1.0 + 1.0e-15, 0.0);
    let near = network(vec![1.0], s, Array2::from_elem((1, 2), c(50.0, 0.0)));
    assert!(
        near.terminate_port_power(0, &finite_loads(&[c(0.0, 0.0)]))
            .is_ok()
    );

    let overflow = network(
        vec![1.0],
        Array3::zeros((1, 2, 2)),
        Array2::from_shape_vec((1, 2), vec![c(-f64::MAX, 0.0), c(50.0, 0.0)]).unwrap(),
    );
    assert_eq!(
        overflow
            .terminate_port_power(0, &finite_loads(&[c(f64::MAX, 0.0)]))
            .unwrap_err(),
        Error::NonFiniteTerminationComputation {
            frequency: 0,
            port: 0,
            row: 0,
            column: 0,
        }
    );
}

#[test]
fn open_termination_distinguishes_exact_and_adjacent_singular_boundaries() {
    // The selected port is completely decoupled.  Open still checks its own
    // exact 1-Skk denominator rather than accepting the zero external result.
    let mut exact_s = Array3::zeros((1, 3, 3));
    exact_s[[0, 1, 1]] = c(1.0, 0.0);
    let exact = network(
        vec![1.0],
        exact_s,
        Array2::from_shape_vec((1, 3), vec![c(47.0, 1.0), c(62.0, -4.0), c(79.0, 2.0)]).unwrap(),
    );
    assert_eq!(
        exact.terminate_port_power(1, &[PortLoad::Open]),
        Err(Error::SingularTermination {
            frequency: 0,
            port: 1,
        })
    );

    let adjacent_to_one = f64::from_bits(1.0_f64.to_bits() - 1);
    let mut near_s = Array3::zeros((1, 2, 2));
    near_s[[0, 0, 0]] = c(adjacent_to_one, 0.0);
    near_s[[0, 1, 0]] = c(1.0e-300, 0.0);
    near_s[[0, 0, 1]] = c(1.0, 0.0);
    let near = network(vec![1.0], near_s, Array2::from_elem((1, 2), c(50.0, 0.0)));
    let reduced = near
        .terminate_port_power(0, &[PortLoad::Open])
        .expect("adjacent representable open denominator remains in domain");
    assert!(reduced.s().iter().all(|value| value.is_finite()));

    let mut overflow_s = Array3::zeros((1, 2, 2));
    overflow_s[[0, 0, 0]] = c(adjacent_to_one, 0.0);
    overflow_s[[0, 1, 0]] = c(f64::MAX, 0.0);
    overflow_s[[0, 0, 1]] = c(1.0, 0.0);
    let overflow = network(
        vec![1.0],
        overflow_s,
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    );
    assert_eq!(
        overflow.terminate_port_power(0, &[PortLoad::Open]),
        Err(Error::NonFiniteTerminationComputation {
            frequency: 0,
            port: 0,
            row: 0,
            column: 0,
        })
    );
}
