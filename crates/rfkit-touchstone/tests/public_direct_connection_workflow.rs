use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

const INPUT_A: &str = r#"# Hz S RI R 50
1000000000 0.08 0.01 0.02 -0.01 0.03 0.015
0.04 0.00 0.12 -0.01 0.05 0.01
0.06 -0.005 0.07 0.02 0.10 0.00
2000000000 0.09 0.015 0.025 -0.012 0.035 0.018
0.045 0.002 0.13 -0.012 0.055 0.012
0.065 -0.006 0.075 0.024 0.11 0.001
3000000000 0.10 0.02 0.03 -0.014 0.04 0.021
0.05 0.004 0.14 -0.014 0.06 0.014
0.07 -0.007 0.08 0.028 0.12 0.002
"#;

const INPUT_B: &str = r#"# Hz S RI R 75
1000000000 0.07 0.01 0.012 -0.004 0.018 0.008 0.025 -0.006
0.031 0.007 0.11 -0.012 0.022 0.004 0.016 0.009
0.014 -0.003 0.027 0.011 0.13 0.006 0.019 -0.008
0.021 0.005 0.017 -0.007 0.024 0.010 0.09 -0.004
2000000000 0.075 0.011 0.014 -0.005 0.020 0.009 0.027 -0.007
0.033 0.008 0.12 -0.013 0.024 0.005 0.018 0.010
0.016 -0.004 0.029 0.012 0.14 0.007 0.021 -0.009
0.023 0.006 0.019 -0.008 0.026 0.011 0.095 -0.005
3000000000 0.08 0.012 0.016 -0.006 0.022 0.010 0.029 -0.008
0.035 0.009 0.13 -0.014 0.026 0.006 0.020 0.011
0.018 -0.005 0.031 0.013 0.15 0.008 0.023 -0.010
0.025 0.007 0.021 -0.009 0.028 0.012 0.10 -0.006
"#;

const SURVIVORS_A: [usize; 2] = [0, 2];
const SURVIVORS_B: [usize; 3] = [0, 1, 3];
const SELECTED_A: usize = 1;
const SELECTED_B: usize = 2;

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn solve_two_by_two(matrix: [[Complex64; 2]; 2], rhs: [Complex64; 2]) -> [Complex64; 2] {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    assert!(
        determinant != c(0.0, 0.0),
        "physical junction must be nonsingular"
    );
    [
        (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
        (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
    ]
}

/// Independently derive S_out from the Kurokawa V/I equations and the
/// physical conditions V_A=V_B and I_A+I_B=0.  This is intentionally local to
/// the workflow test; it does not call a core connection or matched helper.
fn independently_expected(a: &rfkit_core::Network, b: &rfkit_core::Network) -> Array3<Complex64> {
    let nfreq = a.frequency().hz().len();
    Array3::from_shape_fn((nfreq, 5, 5), |(frequency, output, input)| {
        let z_a = a.z0()[[frequency, SELECTED_A]];
        let z_b = b.z0()[[frequency, SELECTED_B]];
        let q_a = z_a.re.abs().sqrt() / z_a.re;
        let q_b = z_b.re.abs().sqrt() / z_b.re;
        let s_ii = [
            [a.s()[[frequency, SELECTED_A, SELECTED_A]], c(0.0, 0.0)],
            [c(0.0, 0.0), b.s()[[frequency, SELECTED_B, SELECTED_B]]],
        ];
        let c_matrix = [
            [c(q_a, 0.0), c(q_b, 0.0)],
            [q_a * z_a.conj(), -q_b * z_b.conj()],
        ];
        let d_matrix = [[c(-q_a, 0.0), c(-q_b, 0.0)], [q_a * z_a, -q_b * z_b]];
        let system = [
            [
                c_matrix[0][0] + d_matrix[0][0] * s_ii[0][0],
                c_matrix[0][1] + d_matrix[0][1] * s_ii[1][1],
            ],
            [
                c_matrix[1][0] + d_matrix[1][0] * s_ii[0][0],
                c_matrix[1][1] + d_matrix[1][1] * s_ii[1][1],
            ],
        ];

        let s_ie = [
            if input < SURVIVORS_A.len() {
                a.s()[[frequency, SELECTED_A, SURVIVORS_A[input]]]
            } else {
                c(0.0, 0.0)
            },
            if input < SURVIVORS_A.len() {
                c(0.0, 0.0)
            } else {
                b.s()[[
                    frequency,
                    SELECTED_B,
                    SURVIVORS_B[input - SURVIVORS_A.len()],
                ]]
            },
        ];
        let rhs = [
            -(d_matrix[0][0] * s_ie[0] + d_matrix[0][1] * s_ie[1]),
            -(d_matrix[1][0] * s_ie[0] + d_matrix[1][1] * s_ie[1]),
        ];
        let internal = solve_two_by_two(system, rhs);

        let s_ee = if output < SURVIVORS_A.len() {
            if input < SURVIVORS_A.len() {
                a.s()[[frequency, SURVIVORS_A[output], SURVIVORS_A[input]]]
            } else {
                c(0.0, 0.0)
            }
        } else if input < SURVIVORS_A.len() {
            c(0.0, 0.0)
        } else {
            b.s()[[
                frequency,
                SURVIVORS_B[output - SURVIVORS_A.len()],
                SURVIVORS_B[input - SURVIVORS_A.len()],
            ]]
        };
        let s_ei = if output < SURVIVORS_A.len() {
            [
                a.s()[[frequency, SURVIVORS_A[output], SELECTED_A]],
                c(0.0, 0.0),
            ]
        } else {
            [
                c(0.0, 0.0),
                b.s()[[
                    frequency,
                    SURVIVORS_B[output - SURVIVORS_A.len()],
                    SELECTED_B,
                ]],
            ]
        };
        s_ee + s_ei[0] * internal[0] + s_ei[1] * internal[1]
    })
}

#[test]
fn touchstone_direct_connection_checks_physical_junction_and_explicit_writer_renormalization() {
    let a = parse_touchstone_v1_0_s(INPUT_A, 3).expect("A Touchstone input parses");
    let b = parse_touchstone_v1_0_s(INPUT_B, 4).expect("B Touchstone input parses");
    assert_eq!(a.z0(), &Array2::from_elem((3, 3), c(50.0, 0.0)));
    assert_eq!(b.z0(), &Array2::from_elem((3, 4), c(75.0, 0.0)));

    let a_frequency = a.frequency().clone();
    let a_s = a.s().clone();
    let a_z0 = a.z0().clone();
    let b_frequency = b.frequency().clone();
    let b_s = b.s().clone();
    let b_z0 = b.z0().clone();
    let expected = independently_expected(&a, &b);

    let joined = a
        .connect_direct_power(SELECTED_A, &b, SELECTED_B)
        .expect("unequal real references connect directly");
    assert_eq!(joined.frequency(), &a_frequency);
    assert_eq!(
        joined.z0(),
        &Array2::from_shape_fn((3, 5), |(frequency, port)| {
            if port < SURVIVORS_A.len() {
                a_z0[[frequency, SURVIVORS_A[port]]]
            } else {
                b_z0[[frequency, SURVIVORS_B[port - SURVIVORS_A.len()]]]
            }
        })
    );
    assert_eq!(joined.s().dim(), (3, 5, 5));
    for (actual, expected) in joined.s().iter().zip(expected.iter()) {
        assert!(
            (*actual - *expected).norm() <= 1.0e-12,
            "physical junction mismatch: actual={actual:?}, expected={expected:?}"
        );
    }

    // The writer must reject the direct result's heterogeneous references; it
    // must not silently repair them.  Renormalization is an explicit caller
    // operation and chooses the common writer reference deliberately.
    assert!(write_touchstone_v1_0_s_ri_hz(&joined).is_err());
    let common_z0 = Array2::from_elem((3, 5), c(60.0, 0.0));
    let writer_ready = joined
        .renormalize_direct_power(common_z0.clone())
        .expect("explicit writer-compatible direct renormalization");
    assert_eq!(writer_ready.z0(), &common_z0);
    let text = write_touchstone_v1_0_s_ri_hz(&writer_ready).expect("writer accepts common z0");
    let reread = parse_touchstone_v1_0_s(&text, 5).expect("writer output parses");
    assert_eq!(reread.frequency(), writer_ready.frequency());
    assert_eq!(reread.z0(), writer_ready.z0());
    assert_eq!(reread.s(), writer_ready.s());

    // Both parsed inputs are borrowed and remain byte-for-byte equivalent in
    // their owned arrays after direct connection and export.
    assert_eq!(a.frequency(), &a_frequency);
    assert_eq!(a.s(), &a_s);
    assert_eq!(a.z0(), &a_z0);
    assert_eq!(b.frequency(), &b_frequency);
    assert_eq!(b.s(), &b_s);
    assert_eq!(b.z0(), &b_z0);
}
