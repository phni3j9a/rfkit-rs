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

const LOAD_OHM: [Complex64; 3] = [
    Complex64::new(0.0, 0.0),
    Complex64::new(38.0, 12.0),
    Complex64::new(73.0, -9.0),
];

fn independently_expected(source_s: &Array3<Complex64>) -> Array3<Complex64> {
    let survivors = [0_usize, 1, 3, 4];
    Array3::from_shape_fn(
        (3, survivors.len(), survivors.len()),
        |(frequency, row, column)| {
            let selected = 2;
            let source_reference = Complex64::new(50.0, 0.0);
            let c = LOAD_OHM[frequency] - source_reference;
            let d = LOAD_OHM[frequency] + source_reference.conj();
            let denominator = d - c * source_s[[frequency, selected, selected]];
            let feedback = c / denominator;
            source_s[[frequency, survivors[row], survivors[column]]]
                + source_s[[frequency, survivors[row], selected]]
                    * feedback
                    * source_s[[frequency, selected, survivors[column]]]
        },
    )
}

#[test]
fn touchstone_ingress_finite_load_termination_and_writer_roundtrip() {
    let source = parse_touchstone_v1_0_s(INPUT, 5).expect("asymmetric 5-port input parses");
    let source_frequency = source.frequency().clone();
    let source_s = source.s().clone();
    let source_z0 = source.z0().clone();
    let expected = independently_expected(&source_s);

    let terminated = source
        .terminate_port_impedance_power(2, &LOAD_OHM)
        .expect("finite physical load termination succeeds");
    assert_eq!(terminated.frequency(), &source_frequency);
    assert_eq!(terminated.z0().dim(), (3, 4));
    assert_eq!(
        terminated.z0(),
        &Array2::from_shape_fn((3, 4), |(frequency, port)| {
            source_z0[[frequency, [0_usize, 1, 3, 4][port]]]
        })
    );
    for (actual, expected) in terminated.s().iter().zip(expected.iter()) {
        assert!((*actual - *expected).norm() <= 1.0e-14);
    }

    let text = write_touchstone_v1_0_s_ri_hz(&terminated).expect("survivors are writer-compatible");
    let reread = parse_touchstone_v1_0_s(&text, 4).expect("writer output parses");
    assert_eq!(reread.frequency(), terminated.frequency());
    assert_eq!(reread.z0(), terminated.z0());
    assert_eq!(reread.s(), terminated.s());

    // The source and load are borrowed inputs; this workflow must not mutate
    // either while producing the reduced owned network.
    assert_eq!(source.frequency(), &source_frequency);
    assert_eq!(source.s(), &source_s);
    assert_eq!(source.z0(), &source_z0);
}
