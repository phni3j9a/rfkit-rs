use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

const INPUT: &str = r#"# Hz S RI R 50
1000000000 0.11 0.01 0.12 0.02 0.13 0.03 0.14 0.04
0.15 0.05
0.21 0.01 0.22 0.02 0.23 0.03 0.24 0.04
0.25 0.05
0.31 0.01 0.32 0.02 0.33 0.03 0.34 0.04
0.35 0.05
0.41 0.01 0.42 0.02 0.43 0.03 0.44 0.04
0.45 0.05
0.51 0.01 0.52 0.02 0.53 0.03 0.54 0.04
0.55 0.05
2000000000 0.16 0.06 0.17 0.07 0.18 0.08 0.19 0.09
0.20 0.10
0.26 0.06 0.27 0.07 0.28 0.08 0.29 0.09
0.30 0.10
0.36 0.06 0.37 0.07 0.38 0.08 0.39 0.09
0.40 0.10
0.46 0.06 0.47 0.07 0.48 0.08 0.49 0.09
0.50 0.10
0.56 0.06 0.57 0.07 0.58 0.08 0.59 0.09
0.60 0.10
3000000000 0.21 0.11 0.22 0.12 0.23 0.13 0.24 0.14
0.25 0.15
0.31 0.11 0.32 0.12 0.33 0.13 0.34 0.14
0.35 0.15
0.41 0.11 0.42 0.12 0.43 0.13 0.44 0.14
0.45 0.15
0.51 0.11 0.52 0.12 0.53 0.13 0.54 0.14
0.55 0.15
0.61 0.11 0.62 0.12 0.63 0.13 0.64 0.14
0.65 0.15
"#;

const PORT_A: usize = 1;
const PORT_B: usize = 3;
const SURVIVORS: [usize; 3] = [0, 2, 4];

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn selected_references() -> Array2<Complex64> {
    Array2::from_shape_vec(
        (3, 5),
        vec![
            c(58.0, 3.0),
            c(43.0, 8.0),
            c(66.0, -4.0),
            c(71.0, -9.0),
            c(52.0, 6.0),
            c(59.0, 4.0),
            c(44.0, 9.0),
            c(67.0, -5.0),
            c(72.0, -10.0),
            c(53.0, 7.0),
            c(60.0, 5.0),
            c(45.0, 10.0),
            c(68.0, -6.0),
            c(73.0, -11.0),
            c(54.0, 8.0),
        ],
    )
    .expect("reference shape is fixed")
}

fn solve_two_by_two(matrix: [[Complex64; 2]; 2], rhs: [Complex64; 2]) -> [Complex64; 2] {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    assert!(
        determinant != c(0.0, 0.0),
        "physical junction is nonsingular"
    );
    [
        (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
        (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
    ]
}

/// Reconstruct the reduced scattering response from the full internal block
/// and the physical Kurokawa V/I boundary.  This deliberately does not call
/// any rfkit connection or matched-junction helper.
fn independently_expected(source: &rfkit_core::Network) -> Array3<Complex64> {
    let nfreq = source.frequency().hz().len();
    Array3::from_shape_fn(
        (nfreq, SURVIVORS.len(), SURVIVORS.len()),
        |(frequency, row, column)| {
            let z_a = source.z0()[[frequency, PORT_A]];
            let z_b = source.z0()[[frequency, PORT_B]];
            let q_a = z_a.re.abs().sqrt() / z_a.re;
            let q_b = z_b.re.abs().sqrt() / z_b.re;

            // Both off-diagonal entries are intentionally retained.  The source
            // has a full, non-reciprocal selected block, so a diagonal shortcut
            // would produce a different physical junction response.
            let s_ii = [
                [
                    source.s()[[frequency, PORT_A, PORT_A]],
                    source.s()[[frequency, PORT_A, PORT_B]],
                ],
                [
                    source.s()[[frequency, PORT_B, PORT_A]],
                    source.s()[[frequency, PORT_B, PORT_B]],
                ],
            ];
            let c_matrix = [
                [c(q_a, 0.0), c(q_b, 0.0)],
                [q_a * z_a.conj(), -q_b * z_b.conj()],
            ];
            let d_matrix = [[c(-q_a, 0.0), c(-q_b, 0.0)], [q_a * z_a, -q_b * z_b]];
            let system = [
                [
                    c_matrix[0][0] + d_matrix[0][0] * s_ii[0][0] + d_matrix[0][1] * s_ii[1][0],
                    c_matrix[0][1] + d_matrix[0][0] * s_ii[0][1] + d_matrix[0][1] * s_ii[1][1],
                ],
                [
                    c_matrix[1][0] + d_matrix[1][0] * s_ii[0][0] + d_matrix[1][1] * s_ii[1][0],
                    c_matrix[1][1] + d_matrix[1][0] * s_ii[0][1] + d_matrix[1][1] * s_ii[1][1],
                ],
            ];

            let external_input = SURVIVORS[column];
            let s_ie = [
                source.s()[[frequency, PORT_A, external_input]],
                source.s()[[frequency, PORT_B, external_input]],
            ];
            let rhs = [
                -(d_matrix[0][0] * s_ie[0] + d_matrix[0][1] * s_ie[1]),
                -(d_matrix[1][0] * s_ie[0] + d_matrix[1][1] * s_ie[1]),
            ];
            let internal_incident = solve_two_by_two(system, rhs);
            let internal_reflected = [
                s_ie[0] + s_ii[0][0] * internal_incident[0] + s_ii[0][1] * internal_incident[1],
                s_ie[1] + s_ii[1][0] * internal_incident[0] + s_ii[1][1] * internal_incident[1],
            ];

            // Check the two physical junction equations for this nonzero external
            // excitation, independently of the returned S matrix.
            let voltage_a = q_a * (z_a.conj() * internal_incident[0] + z_a * internal_reflected[0]);
            let voltage_b = q_b * (z_b.conj() * internal_incident[1] + z_b * internal_reflected[1]);
            let current_a = q_a * (internal_incident[0] - internal_reflected[0]);
            let current_b = q_b * (internal_incident[1] - internal_reflected[1]);
            assert!((voltage_a - voltage_b).norm() <= 1.0e-12);
            assert!((current_a + current_b).norm() <= 1.0e-12);

            let external_output = SURVIVORS[row];
            source.s()[[frequency, external_output, external_input]]
                + source.s()[[frequency, external_output, PORT_A]] * internal_incident[0]
                + source.s()[[frequency, external_output, PORT_B]] * internal_incident[1]
        },
    )
}

#[test]
fn touchstone_inner_connect_power_checks_full_physical_junction_and_writer_roundtrip() {
    let parsed = parse_touchstone_v1_0_s(INPUT, 5).expect("asymmetric 5-port input parses");
    let source_frequency = parsed.frequency().clone();
    let source_s = parsed.s().clone();
    let source_z0 = parsed.z0().clone();
    let target_z0 = selected_references();

    assert_ne!(target_z0[[0, PORT_A]], target_z0[[0, PORT_B]]);
    assert_ne!(target_z0[[0, PORT_A]].im, 0.0);
    assert_ne!(target_z0[[0, PORT_B]].im, 0.0);

    let direct = parsed
        .renormalize_direct_power(target_z0.clone())
        .expect("selected complex references are valid");
    assert_eq!(direct.frequency(), &source_frequency);
    assert_eq!(direct.z0(), &target_z0);

    let expected = independently_expected(&direct);
    let reduced = direct
        .inner_connect_power(PORT_A, PORT_B)
        .expect("direct inner physical junction succeeds");

    assert_eq!(reduced.frequency(), &source_frequency);
    assert_eq!(
        reduced.z0(),
        &Array2::from_shape_fn((3, SURVIVORS.len()), |(frequency, port)| {
            target_z0[[frequency, SURVIVORS[port]]]
        })
    );
    assert_eq!(reduced.s().dim(), (3, SURVIVORS.len(), SURVIVORS.len()));
    for (actual, expected) in reduced.s().iter().zip(expected.iter()) {
        assert!(
            (*actual - *expected).norm() <= 1.0e-12,
            "full-block physical junction mismatch: actual={actual:?}, expected={expected:?}"
        );
    }

    // The direct operation borrows its source and retains exact frequency and
    // survivor coordinates.  The original parsed source must remain untouched.
    assert_eq!(parsed.frequency(), &source_frequency);
    assert_eq!(parsed.s(), &source_s);
    assert_eq!(parsed.z0(), &source_z0);

    // Touchstone v1.0 has one common finite positive-real reference.  Choose
    // that contract explicitly after the physical inner junction; neither
    // operation nor writer silently repairs the heterogeneous references.
    let writer_z0 = Array2::from_elem((3, SURVIVORS.len()), c(60.0, 0.0));
    assert!(write_touchstone_v1_0_s_ri_hz(&reduced).is_err());
    let writer_ready = reduced
        .renormalize_direct_power(writer_z0.clone())
        .expect("explicit writer-compatible direct renormalization succeeds");
    assert_eq!(writer_ready.z0(), &writer_z0);

    let text = write_touchstone_v1_0_s_ri_hz(&writer_ready).expect("writer accepts common z0");
    let reread = parse_touchstone_v1_0_s(&text, SURVIVORS.len()).expect("writer output parses");
    assert_eq!(reread.frequency(), writer_ready.frequency());
    assert_eq!(reread.z0(), writer_ready.z0());
    assert_eq!(reread.s(), writer_ready.s());
}
