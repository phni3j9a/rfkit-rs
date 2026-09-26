use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{ConnectionInput, Error, Frequency, Network};
use serde_json::json;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn constant_z0(nfreq: usize, nports: usize, value: Complex64) -> Array2<Complex64> {
    Array2::from_elem((nfreq, nports), value)
}

fn two_port(real: f64, imag: f64) -> Network {
    let frequency = Frequency::from_hz(vec![1.0e9, 2.0e9]).unwrap();
    let s = Array3::from_shape_fn((2, 2, 2), |(frequency, row, column)| {
        c(
            real + 0.01 * frequency as f64 + 0.02 * row as f64,
            imag + 0.03 * column as f64,
        )
    });
    Network::new(frequency, s, constant_z0(2, 2, c(50.0, 0.0))).unwrap()
}

fn coupled_four_port_fixture() -> Network {
    let frequency = Frequency::from_hz(vec![0.8e9, 1.6e9]).unwrap();
    let s = Array3::from_shape_fn((2, 4, 4), |(f, row, column)| {
        let row_group = row / 2;
        let column_group = column / 2;
        let local_row = row % 2;
        let local_column = column % 2;
        let scale = 1.0 + 0.02 * f as f64;
        match (row_group, column_group) {
            (0, 0) | (1, 1) => {
                if local_row == local_column {
                    c(0.04 * scale, -0.01 + 0.002 * f as f64)
                } else {
                    c(0.013 + 0.002 * local_row as f64, 0.004)
                }
            }
            (0, 1) => {
                if local_row == local_column {
                    c(0.54 * scale, 0.03 + 0.004 * local_row as f64)
                } else {
                    c(0.037, -0.011)
                }
            }
            (1, 0) => {
                if local_row == local_column {
                    c(0.47 * scale, -0.025 - 0.003 * local_column as f64)
                } else {
                    c(-0.029, 0.009)
                }
            }
            _ => unreachable!(),
        }
    });
    let z0 = Array2::from_shape_fn((2, 4), |(f, port)| {
        c(
            [41.0, 53.0, 67.0, 79.0][port] + f as f64,
            [3.0, -4.0, 5.0, -6.0][port],
        )
    });
    Network::new(frequency, s, z0).unwrap()
}

fn physical_through(frequency: &Frequency, z0: Array2<Complex64>) -> Network {
    let nports = z0.dim().1;
    let group_size = nports / 2;
    let s = Array3::from_shape_fn((frequency.len(), nports, nports), |(_, row, column)| {
        if (row < group_size && column == row + group_size)
            || (row >= group_size && column == row - group_size)
        {
            c(1.0, 0.0)
        } else {
            c(0.0, 0.0)
        }
    });
    Network::new(frequency.clone(), s, z0).unwrap()
}

fn assert_array_close(actual: &Array3<Complex64>, expected: &Array3<Complex64>, tolerance: f64) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).norm() <= tolerance,
            "index {index}: actual={actual:?}, expected={expected:?}"
        );
    }
}

fn dense_solve(mut matrix: Vec<Vec<Complex64>>, mut rhs: Vec<Complex64>) -> Vec<Complex64> {
    let dimension = matrix.len();
    for pivot in 0..dimension {
        let pivot_row = (pivot..dimension)
            .max_by(|&left, &right| {
                matrix[left][pivot]
                    .norm_sqr()
                    .partial_cmp(&matrix[right][pivot].norm_sqr())
                    .unwrap()
            })
            .unwrap();
        matrix.swap(pivot, pivot_row);
        rhs.swap(pivot, pivot_row);
        let pivot_values = matrix[pivot].clone();
        let pivot_rhs = rhs[pivot];
        for row in (pivot + 1)..dimension {
            let factor = matrix[row][pivot] / pivot_values[pivot];
            for column in pivot..dimension {
                matrix[row][column] -= factor * pivot_values[column];
            }
            rhs[row] -= factor * pivot_rhs;
        }
    }
    let mut solution = vec![c(0.0, 0.0); dimension];
    for row in (0..dimension).rev() {
        let mut value = rhs[row];
        for column in (row + 1)..dimension {
            value -= matrix[row][column] * solution[column];
        }
        solution[row] = value / matrix[row][row];
    }
    solution
}

struct IndependentResponse {
    external_reflected: [Complex64; 4],
    a_voltage: [Complex64; 2],
    b_voltage: [Complex64; 2],
    a_current: [Complex64; 2],
    b_current: [Complex64; 2],
}

fn independent_wave_response(
    a: &Network,
    b: &Network,
    frequency: usize,
    external_incident: &[Complex64; 4],
) -> IndependentResponse {
    let mut c_matrix = vec![vec![c(0.0, 0.0); 4]; 4];
    let mut d_matrix = vec![vec![c(0.0, 0.0); 4]; 4];
    for index in 0..2 {
        let z_a = a.z0()[[frequency, index + 2]];
        let z_b = b.z0()[[frequency, index]];
        let q_a = z_a.re.abs().sqrt() / z_a.re;
        let q_b = z_b.re.abs().sqrt() / z_b.re;
        c_matrix[index][index] = c(q_a, 0.0);
        c_matrix[index][index + 2] = c(q_b, 0.0);
        c_matrix[index + 2][index] = c(q_a, 0.0) * z_a.conj();
        c_matrix[index + 2][index + 2] = -c(q_b, 0.0) * z_b.conj();
        d_matrix[index][index] = c(-q_a, 0.0);
        d_matrix[index][index + 2] = c(-q_b, 0.0);
        d_matrix[index + 2][index] = c(q_a, 0.0) * z_a;
        d_matrix[index + 2][index + 2] = -c(q_b, 0.0) * z_b;
    }
    let mut s_ii = vec![vec![c(0.0, 0.0); 4]; 4];
    let mut s_ie = vec![vec![c(0.0, 0.0); 4]; 4];
    let mut s_ei = vec![vec![c(0.0, 0.0); 4]; 4];
    let mut s_ee = vec![vec![c(0.0, 0.0); 4]; 4];
    for row in 0..2 {
        for column in 0..2 {
            s_ee[row][column] = a.s()[[frequency, row, column]];
            s_ei[row][column] = a.s()[[frequency, row, column + 2]];
            s_ie[row][column] = a.s()[[frequency, row + 2, column]];
            s_ii[row][column] = a.s()[[frequency, row + 2, column + 2]];
            s_ee[row + 2][column + 2] = b.s()[[frequency, row + 2, column + 2]];
            s_ei[row + 2][column + 2] = b.s()[[frequency, row + 2, column]];
            s_ie[row + 2][column + 2] = b.s()[[frequency, row, column + 2]];
            s_ii[row + 2][column + 2] = b.s()[[frequency, row, column]];
        }
    }
    let mut system = vec![vec![c(0.0, 0.0); 4]; 4];
    let mut right = vec![c(0.0, 0.0); 4];
    for row in 0..4 {
        for column in 0..4 {
            system[row][column] = c_matrix[row][column]
                + (0..4)
                    .map(|index| d_matrix[row][index] * s_ii[index][column])
                    .sum::<Complex64>();
        }
        right[row] = -(0..4)
            .map(|index| {
                d_matrix[row][index]
                    * (0..4)
                        .map(|column| s_ie[index][column] * external_incident[column])
                        .sum::<Complex64>()
            })
            .sum::<Complex64>();
    }
    let internal_incident = dense_solve(system, right);
    let internal_reflected = (0..4)
        .map(|row| {
            (0..4)
                .map(|column| s_ii[row][column] * internal_incident[column])
                .sum::<Complex64>()
                + (0..4)
                    .map(|column| s_ie[row][column] * external_incident[column])
                    .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let external_reflected = (0..4)
        .map(|row| {
            (0..4)
                .map(|column| s_ee[row][column] * external_incident[column])
                .sum::<Complex64>()
                + (0..4)
                    .map(|column| s_ei[row][column] * internal_incident[column])
                    .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    let mut a_voltage = [c(0.0, 0.0); 2];
    let mut b_voltage = [c(0.0, 0.0); 2];
    let mut a_current = [c(0.0, 0.0); 2];
    let mut b_current = [c(0.0, 0.0); 2];
    for index in 0..2 {
        let z_a = a.z0()[[frequency, index + 2]];
        let z_b = b.z0()[[frequency, index]];
        let q_a = z_a.re.abs().sqrt() / z_a.re;
        let q_b = z_b.re.abs().sqrt() / z_b.re;
        a_current[index] = c(q_a, 0.0) * (internal_incident[index] - internal_reflected[index]);
        b_current[index] =
            c(q_b, 0.0) * (internal_incident[index + 2] - internal_reflected[index + 2]);
        a_voltage[index] =
            c(q_a, 0.0) * (z_a.conj() * internal_incident[index] + z_a * internal_reflected[index]);
        b_voltage[index] = c(q_b, 0.0)
            * (z_b.conj() * internal_incident[index + 2] + z_b * internal_reflected[index + 2]);
    }
    IndependentResponse {
        external_reflected: [
            external_reflected[0],
            external_reflected[1],
            external_reflected[2],
            external_reflected[3],
        ],
        a_voltage,
        b_voltage,
        a_current,
        b_current,
    }
}

#[test]
fn two_port_cascade_agrees_with_single_direct_connection() {
    let a = two_port(0.07, -0.03);
    let b = two_port(-0.04, 0.05);
    let simultaneous = a.cascade_direct_power(&b).unwrap();
    let sequential = a.connect_power(1, &b, 0).unwrap();
    assert_eq!(simultaneous.frequency(), sequential.frequency());
    assert_eq!(simultaneous.z0(), sequential.z0());
    for (actual, expected) in simultaneous.s().iter().zip(sequential.s()) {
        assert!((*actual - *expected).norm() < 1.0e-14);
    }
}

#[test]
fn reconstructs_complex_signed_reference_v_i_boundary_for_nonzero_excitation() {
    let base_a = coupled_four_port_fixture();
    let base_b = coupled_four_port_fixture();
    let mut a_z0 = base_a.z0().clone();
    let mut b_z0 = base_b.z0().clone();
    // Exercise both signs on the connected pair and retain nonzero imaginary
    // parts.  The algebraic power-wave definition permits these references;
    // no passive interpretation is made here.
    a_z0[[0, 2]] = c(-67.0, 5.0);
    a_z0[[0, 3]] = c(79.0, -6.0);
    b_z0[[0, 0]] = c(-43.0, -2.0);
    b_z0[[0, 1]] = c(53.0, 4.0);
    let a = Network::new(base_a.frequency().clone(), base_a.s().clone(), a_z0).unwrap();
    let b = Network::new(base_b.frequency().clone(), base_b.s().clone(), b_z0).unwrap();
    let result = a.cascade_direct_power(&b).unwrap();
    let external_incident = [
        c(0.37, -0.21),
        c(-0.16, 0.44),
        c(0.29, 0.18),
        c(-0.33, -0.07),
    ];
    let response = independent_wave_response(&a, &b, 0, &external_incident);
    for index in 0..2 {
        assert!((response.a_voltage[index] - response.b_voltage[index]).norm() < 2.0e-12);
        assert!((response.a_current[index] + response.b_current[index]).norm() < 2.0e-12);
    }
    let actual_reflected = (0..4)
        .map(|row| {
            (0..4)
                .map(|column| result.s()[[0, row, column]] * external_incident[column])
                .sum::<Complex64>()
        })
        .collect::<Vec<_>>();
    for (actual, expected) in actual_reflected.iter().zip(response.external_reflected) {
        assert!((*actual - expected).norm() < 2.0e-12);
    }
}

#[test]
fn cascade_composes_with_inverse_in_both_orientations_for_coupled_groups() {
    let fixture = coupled_four_port_fixture();
    let inverse = fixture.inverse_cascade_power().unwrap();

    let forward = fixture.cascade_direct_power(&inverse).unwrap();
    let forward_z0 = Array2::from_shape_fn((2, 4), |(frequency, port)| {
        if port < 2 {
            fixture.z0()[[frequency, port]]
        } else {
            fixture.z0()[[frequency, port - 2]].conj()
        }
    });
    let expected_forward = physical_through(fixture.frequency(), forward_z0);
    assert_eq!(forward.z0(), expected_forward.z0());
    assert_array_close(forward.s(), expected_forward.s(), 2.0e-11);

    let reverse = inverse.cascade_direct_power(&fixture).unwrap();
    let reverse_z0 = Array2::from_shape_fn((2, 4), |(frequency, port)| {
        if port < 2 {
            fixture.z0()[[frequency, port + 2]].conj()
        } else {
            fixture.z0()[[frequency, port]]
        }
    });
    let expected_reverse = physical_through(fixture.frequency(), reverse_z0);
    assert_eq!(reverse.z0(), expected_reverse.z0());
    assert_array_close(reverse.s(), expected_reverse.s(), 2.0e-11);
}

#[test]
fn supports_a_larger_six_port_coupled_group_case() {
    const GROUP: usize = 3;
    const NPORTS: usize = 2 * GROUP;
    let frequency = Frequency::from_hz(vec![0.75e9, 1.25e9]).unwrap();
    let a_s = Array3::from_shape_fn((2, NPORTS, NPORTS), |(f, row, column)| {
        let row_group = row / GROUP;
        let column_group = column / GROUP;
        let local_row = row % GROUP;
        let local_column = column % GROUP;
        let scale = 1.0 + 0.015 * f as f64;
        match (row_group, column_group) {
            (0, 0) | (1, 1) => {
                if local_row == local_column {
                    c(0.025 * scale, -0.006)
                } else {
                    c(0.004 + 0.001 * local_row as f64, 0.002)
                }
            }
            (0, 1) => c(
                if local_row == local_column {
                    0.31 * scale
                } else {
                    0.017 + 0.002 * local_row as f64
                },
                0.008 + 0.001 * local_column as f64,
            ),
            (1, 0) => c(
                if local_row == local_column {
                    0.27 * scale
                } else {
                    0.014 + 0.001 * local_column as f64
                },
                -0.006 - 0.001 * local_row as f64,
            ),
            _ => unreachable!(),
        }
    });
    let b_s = Array3::from_shape_fn((2, NPORTS, NPORTS), |(f, row, column)| {
        let row_group = row / GROUP;
        let column_group = column / GROUP;
        let local_row = row % GROUP;
        let local_column = column % GROUP;
        let scale = 1.0 + 0.012 * f as f64;
        match (row_group, column_group) {
            (0, 0) | (1, 1) => {
                if local_row == local_column {
                    c(0.031 * scale, 0.004)
                } else {
                    c(-0.003 - 0.001 * local_column as f64, 0.001)
                }
            }
            (0, 1) => c(
                if local_row == local_column {
                    0.29 * scale
                } else {
                    0.013 + 0.001 * local_row as f64
                },
                -0.007 + 0.001 * local_column as f64,
            ),
            (1, 0) => c(
                if local_row == local_column {
                    0.34 * scale
                } else {
                    0.011 + 0.001 * local_column as f64
                },
                0.005 + 0.001 * local_row as f64,
            ),
            _ => unreachable!(),
        }
    });
    let a_z0 = Array2::from_shape_fn((2, NPORTS), |(f, port)| {
        c(
            [39.0, 47.0, 55.0, 64.0, 72.0, 83.0][port] + f as f64,
            [2.0, -3.0, 4.0, -5.0, 6.0, -7.0][port],
        )
    });
    let b_z0 = Array2::from_shape_fn((2, NPORTS), |(f, port)| {
        c(
            [43.0, 52.0, 61.0, 68.0, 77.0, 89.0][port] + 0.5 * f as f64,
            [-2.0, 3.0, -4.0, 5.0, -6.0, 7.0][port],
        )
    });
    let a = Network::new(frequency.clone(), a_s, a_z0).unwrap();
    let b = Network::new(frequency, b_s, b_z0).unwrap();
    let result = a.cascade_direct_power(&b).unwrap();
    assert_eq!(result.nports(), NPORTS);
    assert!(
        result
            .s()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert_eq!(
        result.z0().slice(ndarray::s![.., 0..GROUP]),
        a.z0().slice(ndarray::s![.., 0..GROUP])
    );
    assert_eq!(
        result.z0().slice(ndarray::s![.., GROUP..]),
        b.z0().slice(ndarray::s![.., GROUP..])
    );
}

#[test]
fn coupled_partial_singular_witness_succeeds_only_jointly() {
    let zero = c(0.0, 0.0);
    let one = c(1.0, 0.0);
    let r = [[one, one], [one, zero]];
    let mut a = Array3::from_elem((1, 4, 4), zero);
    let mut b = Array3::from_elem((1, 4, 4), zero);
    for index in 0..2 {
        a[[0, index, index + 2]] = one;
        a[[0, index + 2, index]] = one;
        b[[0, index, index]] = one;
        b[[0, index + 2, index]] = one;
        b[[0, index, index + 2]] = one;
    }
    for row in 0..2 {
        for column in 0..2 {
            a[[0, row + 2, column + 2]] = r[row][column];
        }
    }
    let references = constant_z0(1, 4, c(50.0, 0.0));
    let a = Network::new(
        Frequency::from_hz(vec![1.0e9]).unwrap(),
        a,
        references.clone(),
    )
    .unwrap();
    let b = Network::new(Frequency::from_hz(vec![1.0e9]).unwrap(), b, references).unwrap();

    assert!(matches!(
        a.connect_power(2, &b, 0),
        Err(Error::SingularConnection { .. })
    ));
    let result = a.cascade_direct_power(&b).unwrap();
    let m = [[-one, -one], [-one, zero]];
    let mut expected = Array3::from_elem((1, 4, 4), zero);
    for row in 0..2 {
        for column in 0..2 {
            expected[[0, row, column]] = m[row][column];
            expected[[0, row, column + 2]] = m[row][column];
            expected[[0, row + 2, column]] = m[row][column];
            expected[[0, row + 2, column + 2]] =
                m[row][column] - if row == column { one } else { zero };
        }
    }
    for (actual, expected) in result.s().iter().zip(expected.iter()) {
        assert!(
            (*actual - *expected).norm() < 1.0e-12,
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn accepts_zero_s_and_preserves_exact_signed_frequency_and_survivor_references() {
    let frequency = Frequency::from_hz(vec![-0.0, -2.0e9, 1.0e9]).unwrap();
    let s = Array3::zeros((3, 2, 2));
    let a_z0 = Array2::from_shape_vec(
        (3, 2),
        vec![
            c(-41.0, 3.0),
            c(52.0, -4.0),
            c(-42.0, 3.5),
            c(53.0, -4.5),
            c(-43.0, 4.0),
            c(54.0, -5.0),
        ],
    )
    .unwrap();
    let b_z0 = Array2::from_shape_vec(
        (3, 2),
        vec![
            c(61.0, 6.0),
            c(-72.0, -7.0),
            c(62.0, 6.5),
            c(-73.0, -7.5),
            c(63.0, 7.0),
            c(-74.0, -8.0),
        ],
    )
    .unwrap();
    let a = Network::new(frequency.clone(), s.clone(), a_z0.clone()).unwrap();
    let b = Network::new(frequency.clone(), s, b_z0.clone()).unwrap();
    let a_snapshot = a.clone();
    let b_snapshot = b.clone();
    let result = a.cascade_direct_power(&b).unwrap();
    assert_eq!(result.frequency(), &frequency);
    assert_eq!(result.frequency().hz()[0].to_bits(), (-0.0f64).to_bits());
    assert_eq!(result.z0()[[0, 0]], a_z0[[0, 0]]);
    assert_eq!(result.z0()[[0, 1]], b_z0[[0, 1]]);
    assert!(result.s().iter().all(|value| *value == c(0.0, 0.0)));
    assert_eq!(a, a_snapshot);
    assert_eq!(b, b_snapshot);
}

#[test]
fn rejects_joint_singularity_but_accepts_finite_near_singularity() {
    let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let mut a_s = Array3::zeros((1, 2, 2));
    let mut b_s = Array3::zeros((1, 2, 2));
    a_s[[0, 1, 1]] = c(1.0, 0.0);
    b_s[[0, 0, 0]] = c(1.0, 0.0);
    let z0 = constant_z0(1, 2, c(50.0, 0.0));
    let a = Network::new(frequency.clone(), a_s.clone(), z0.clone()).unwrap();
    let b = Network::new(frequency.clone(), b_s.clone(), z0.clone()).unwrap();
    assert!(matches!(
        a.cascade_direct_power(&b),
        Err(Error::SingularCascade { frequency: 0, .. })
    ));

    b_s[[0, 0, 0]] = c(1.0 - 1.0e-12, 0.0);
    let b = Network::new(frequency, b_s, z0).unwrap();
    let near = a.cascade_direct_power(&b).unwrap();
    assert!(
        near.s()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn rejects_malformed_inputs_before_group_indexing_and_never_panics() {
    let valid = two_port(0.03, -0.01);
    let mut malformed = serde_json::to_value(&valid).unwrap();
    malformed["s"]["dim"] = json!([1, 0, 0]);
    malformed["s"]["data"] = json!([]);
    malformed["z0"]["dim"] = json!([1, 0]);
    malformed["z0"]["data"] = json!([]);
    malformed["frequency"]["hz"] = json!([1.0e9]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    let valid_snapshot = valid.clone();
    let error = catch_unwind(AssertUnwindSafe(|| malformed.cascade_direct_power(&valid)))
        .expect("malformed cascade must not panic")
        .unwrap_err();
    assert_eq!(
        error,
        Error::InvalidCascadeSShape {
            input: ConnectionInput::A,
            shape: vec![1, 0, 0]
        }
    );
    assert_eq!(valid, valid_snapshot);

    let mut malformed_z0 = serde_json::to_value(&valid).unwrap();
    let z0_scalar = malformed_z0["z0"]["data"][0].clone();
    malformed_z0["z0"]["dim"] = json!([2, 1]);
    malformed_z0["z0"]["data"] = json!([z0_scalar.clone(), z0_scalar]);
    let malformed_z0: Network = serde_json::from_value(malformed_z0).unwrap();
    assert_eq!(
        malformed_z0.cascade_direct_power(&valid).unwrap_err(),
        Error::InvalidCascadeZ0Shape {
            input: ConnectionInput::A,
            shape: vec![2, 1]
        }
    );
}

#[test]
fn rejects_empty_and_malformed_frequency_axes_before_cascade_indexing() {
    let valid = two_port(0.03, -0.01);

    let mut equal_length_mismatch = serde_json::to_value(&valid).unwrap();
    equal_length_mismatch["frequency"]["hz"] = json!([1.0e9, 3.0e9]);
    let equal_length_mismatch: Network = serde_json::from_value(equal_length_mismatch).unwrap();
    assert_eq!(
        valid
            .cascade_direct_power(&equal_length_mismatch)
            .unwrap_err(),
        Error::CascadeFrequencyMismatch {
            index: 1,
            a: 2.0e9,
            b: 3.0e9,
        }
    );

    let mut frequency_shape = serde_json::to_value(&valid).unwrap();
    frequency_shape["frequency"]["hz"] = json!([1.0e9]);
    let frequency_shape: Network = serde_json::from_value(frequency_shape).unwrap();
    assert_eq!(
        valid.cascade_direct_power(&frequency_shape).unwrap_err(),
        Error::CascadeFrequencyShape {
            input: ConnectionInput::B,
            expected: 2,
            actual: 1,
        }
    );

    let mut empty = serde_json::to_value(&valid).unwrap();
    empty["frequency"]["hz"] = json!([]);
    empty["s"]["dim"] = json!([0, 2, 2]);
    empty["s"]["data"] = json!([]);
    empty["z0"]["dim"] = json!([0, 2]);
    empty["z0"]["data"] = json!([]);
    let empty: Network = serde_json::from_value(empty).unwrap();
    assert_eq!(
        valid.cascade_direct_power(&empty).unwrap_err(),
        Error::EmptyCascadeFrequency {
            input: ConnectionInput::B,
        }
    );
}

#[test]
fn rejects_port_grid_finiteness_and_reference_contract_errors() {
    let valid = two_port(0.03, -0.01);
    let frequency = Frequency::from_hz(vec![1.0e9, 2.0e9]).unwrap();
    let odd = Network::new(
        frequency.clone(),
        Array3::zeros((2, 3, 3)),
        constant_z0(2, 3, c(50.0, 0.0)),
    )
    .unwrap();
    assert_eq!(
        valid.cascade_direct_power(&odd).unwrap_err(),
        Error::InvalidCascadePortCount {
            input: ConnectionInput::B,
            nports: 3
        }
    );
    let unequal = Network::new(
        frequency.clone(),
        Array3::zeros((2, 4, 4)),
        constant_z0(2, 4, c(50.0, 0.0)),
    )
    .unwrap();
    assert_eq!(
        valid.cascade_direct_power(&unequal).unwrap_err(),
        Error::CascadePortCountMismatch { a: 2, b: 4 }
    );

    let b = Network::new(
        Frequency::from_hz(vec![1.0e9]).unwrap(),
        Array3::zeros((1, 2, 2)),
        constant_z0(1, 2, c(50.0, 0.0)),
    )
    .unwrap();
    assert!(matches!(
        valid.cascade_direct_power(&b),
        Err(Error::CascadeFrequencyLengthMismatch { a: 2, b: 1 })
    ));

    let nonfinite = Network::new(
        Frequency::from_hz(vec![f64::NAN]).unwrap(),
        Array3::zeros((1, 2, 2)),
        constant_z0(1, 2, c(50.0, 0.0)),
    )
    .unwrap();
    assert!(matches!(
        valid.cascade_direct_power(&nonfinite),
        Err(Error::NonFiniteCascadeFrequency {
            input: ConnectionInput::B,
            index: 0,
            ..
        })
    ));

    let zero_real = Network::new(
        frequency,
        Array3::zeros((2, 2, 2)),
        Array2::from_shape_vec(
            (2, 2),
            vec![c(0.0, 1.0), c(50.0, 0.0), c(0.0, 1.0), c(50.0, 0.0)],
        )
        .unwrap(),
    )
    .unwrap();
    assert!(matches!(
        valid.cascade_direct_power(&zero_real),
        Err(Error::ZeroRealCascadeReferenceImpedance {
            input: ConnectionInput::B,
            frequency: 0,
            port: 0
        })
    ));
}

#[test]
fn reports_nonfinite_s_z0_and_q_arithmetic_with_operation_context() {
    let valid = two_port(0.03, -0.01);
    let mut nonfinite_s = valid.s().clone();
    nonfinite_s[[0, 1, 0]] = c(f64::NAN, 0.0);
    let nonfinite_s =
        Network::new(valid.frequency().clone(), nonfinite_s, valid.z0().clone()).unwrap();
    assert_eq!(
        valid.cascade_direct_power(&nonfinite_s).unwrap_err(),
        Error::NonFiniteCascadeS {
            input: ConnectionInput::B,
            frequency: 0,
            row: 1,
            column: 0,
        }
    );

    let mut nonfinite_z0 = valid.z0().clone();
    nonfinite_z0[[0, 1]] = c(f64::INFINITY, 0.0);
    let nonfinite_z0 =
        Network::new(valid.frequency().clone(), valid.s().clone(), nonfinite_z0).unwrap();
    assert_eq!(
        valid.cascade_direct_power(&nonfinite_z0).unwrap_err(),
        Error::NonFiniteCascadeZ0 {
            input: ConnectionInput::B,
            frequency: 0,
            port: 1,
        }
    );

    let mut q_overflow_z0 = valid.z0().clone();
    q_overflow_z0[[0, 0]] = c(f64::from_bits(1), f64::MAX);
    let q_overflow =
        Network::new(valid.frequency().clone(), valid.s().clone(), q_overflow_z0).unwrap();
    assert_eq!(
        valid.cascade_direct_power(&q_overflow).unwrap_err(),
        Error::NonFiniteCascadeComputation {
            frequency: 0,
            row: 1,
            column: 1,
        }
    );
}

#[test]
fn reports_finite_input_arithmetic_overflow() {
    let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let mut a_s = Array3::zeros((1, 2, 2));
    let mut b_s = Array3::zeros((1, 2, 2));
    a_s[[0, 1, 1]] = c(f64::MAX, 0.0);
    b_s[[0, 0, 0]] = c(f64::MAX, 0.0);
    let z0 = constant_z0(1, 2, c(f64::MAX, 0.0));
    let a = Network::new(frequency.clone(), a_s, z0.clone()).unwrap();
    let b = Network::new(frequency, b_s, z0).unwrap();
    assert!(matches!(
        a.cascade_direct_power(&b),
        Err(Error::NonFiniteCascadeComputation { frequency: 0, .. })
    ));
}
