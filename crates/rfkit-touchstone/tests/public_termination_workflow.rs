use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::PortLoad;
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

const LOADS: [PortLoad; 3] = [
    PortLoad::Open,
    PortLoad::ImpedanceOhm(Complex64::new(38.0, 12.0)),
    PortLoad::ImpedanceOhm(Complex64::new(0.0, 0.0)),
];

fn feedback(
    source_s: &Array3<Complex64>,
    source_z0: Complex64,
    frequency: usize,
    selected: usize,
    load: &PortLoad,
) -> Complex64 {
    let s_kk = source_s[[frequency, selected, selected]];
    match load {
        PortLoad::Open => Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) - s_kk),
        PortLoad::ImpedanceOhm(load_ohm) => {
            let c = *load_ohm - source_z0;
            let d = *load_ohm + source_z0.conj();
            c / (d - c * s_kk)
        }
        _ => unreachable!("the workflow must handle every supported load variant"),
    }
}

fn independently_expected(
    source_s: &Array3<Complex64>,
    source_z0: &Array2<Complex64>,
    loads: &[PortLoad],
) -> Array3<Complex64> {
    let survivors = [0_usize, 1, 3, 4];
    Array3::from_shape_fn(
        (3, survivors.len(), survivors.len()),
        |(frequency, row, column)| {
            let selected = 2;
            let feedback = feedback(
                source_s,
                source_z0[[frequency, selected]],
                frequency,
                selected,
                &loads[frequency],
            );
            source_s[[frequency, survivors[row], survivors[column]]]
                + source_s[[frequency, survivors[row], selected]]
                    * feedback
                    * source_s[[frequency, selected, survivors[column]]]
        },
    )
}

fn assert_physical_boundary(
    source_s: &Array3<Complex64>,
    source_z0: &Array2<Complex64>,
    reduced_s: &Array3<Complex64>,
    loads: &[PortLoad],
) {
    let survivors = [0_usize, 1, 3, 4];
    let external_incident = [
        Complex64::new(0.20, 0.10),
        Complex64::new(-0.30, 0.05),
        Complex64::new(0.15, -0.08),
        Complex64::new(0.05, 0.02),
    ];
    let zero = Complex64::new(0.0, 0.0);

    for frequency in 0..source_s.dim().0 {
        let selected = 2;
        let source_reference = source_z0[[frequency, selected]];
        let selected_feedback = feedback(
            source_s,
            source_reference,
            frequency,
            selected,
            &loads[frequency],
        );
        let selected_from_survivors = survivors
            .iter()
            .zip(external_incident.iter())
            .map(|(&port, &incident)| source_s[[frequency, selected, port]] * incident)
            .fold(zero, |sum, value| sum + value);

        let mut incident = [zero; 5];
        for (&port, &value) in survivors.iter().zip(external_incident.iter()) {
            incident[port] = value;
        }
        incident[selected] = selected_feedback * selected_from_survivors;

        let mut response = [zero; 5];
        for row in 0..5 {
            response[row] = (0..5)
                .map(|column| source_s[[frequency, row, column]] * incident[column])
                .fold(zero, |sum, value| sum + value);
        }

        for (output_row, &row) in survivors.iter().enumerate() {
            let expected = external_incident
                .iter()
                .enumerate()
                .map(|(output_column, &value)| {
                    reduced_s[[frequency, output_row, output_column]] * value
                })
                .fold(zero, |sum, value| sum + value);
            assert!((response[row] - expected).norm() <= 1.0e-14);
        }

        let q = (source_reference.re.abs()).sqrt() / source_reference.re;
        let selected_current = q * (incident[selected] - response[selected]);
        let selected_voltage = q
            * (source_reference.conj() * incident[selected]
                + source_reference * response[selected]);
        match &loads[frequency] {
            PortLoad::Open => assert!(selected_current.norm() <= 1.0e-14),
            PortLoad::ImpedanceOhm(load_ohm) => {
                assert!((selected_voltage + *load_ohm * selected_current).norm() <= 1.0e-14)
            }
            _ => unreachable!("the workflow must handle every supported load variant"),
        }
    }
}

#[test]
fn touchstone_ingress_mixed_load_termination_and_writer_roundtrip() {
    let source = parse_touchstone_v1_0_s(INPUT, 5).expect("asymmetric 5-port input parses");
    let source_frequency = source.frequency().clone();
    let source_s = source.s().clone();
    let source_z0 = source.z0().clone();
    let expected = independently_expected(&source_s, &source_z0, &LOADS);

    let terminated = source
        .terminate_port_power(2, &LOADS)
        .expect("mixed open/finite/short physical termination succeeds");
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
    assert_physical_boundary(&source_s, &source_z0, terminated.s(), &LOADS);

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
    assert!(matches!(&LOADS[0], PortLoad::Open));
    assert!(matches!(
        &LOADS[1],
        PortLoad::ImpedanceOhm(load) if *load == Complex64::new(38.0, 12.0)
    ));
    assert!(matches!(
        &LOADS[2],
        PortLoad::ImpedanceOhm(load) if *load == Complex64::new(0.0, 0.0)
    ));
}
