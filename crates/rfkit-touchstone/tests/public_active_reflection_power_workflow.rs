use ndarray::Array2;
use num_complex::Complex64;
use rfkit_touchstone::parse_touchstone_v1_0_s;

const INPUT: &str = "# GHz S RI R 50\n\
1 0.10 0.00 0.25 0.00 0.00 0.00\n\
0.40 0.00 0.10 0.00 0.20 0.00\n\
0.15 0.00 0.35 0.00 0.05 0.00\n\
2 0.12 0.00 0.30 0.00 0.05 0.00\n\
0.45 0.00 0.08 0.00 0.25 0.00\n\
0.18 0.00 0.32 0.00 0.06 0.00\n";

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn drives(second_port: Complex64) -> Array2<Complex64> {
    Array2::from_shape_vec(
        (2, 3),
        vec![
            c(1.0, 0.0),
            second_port,
            c(0.0, 0.0),
            c(1.0, 0.0),
            second_port,
            c(0.0, 0.0),
        ],
    )
    .expect("drive shape")
}

#[test]
fn touchstone_load_shows_phase_dependent_active_reflection_and_undefined_zero_port() {
    let network = parse_touchstone_v1_0_s(INPUT, 3).expect("three-port Touchstone input");
    let snapshot = network.clone();
    let in_phase = drives(c(1.0, 0.0));
    let phase_opposed = drives(c(-1.0, 0.0));
    let in_phase_snapshot = in_phase.clone();
    let phase_opposed_snapshot = phase_opposed.clone();

    let in_phase_result = network
        .active_reflection_power(&in_phase)
        .expect("in-phase drive");
    let phase_opposed_result = network
        .active_reflection_power(&phase_opposed)
        .expect("phase-opposed drive");

    assert_eq!(in_phase, in_phase_snapshot);
    assert_eq!(phase_opposed, phase_opposed_snapshot);
    assert_eq!(network, snapshot);

    assert_eq!(in_phase_result[[0, 2]], None);
    assert_eq!(phase_opposed_result[[0, 2]], None);
    assert!((in_phase_result[[0, 0]].unwrap() - c(0.35, 0.0)).norm() < 1.0e-14);
    assert!((phase_opposed_result[[0, 0]].unwrap() - c(-0.15, 0.0)).norm() < 1.0e-14);
    assert_ne!(in_phase_result[[0, 0]], phase_opposed_result[[0, 0]]);
    assert!((in_phase_result[[0, 0]].unwrap() - c(0.10, 0.0)).norm() > 1.0e-3);
    assert!((phase_opposed_result[[0, 0]].unwrap() - c(0.10, 0.0)).norm() > 1.0e-3);

    assert!((in_phase_result[[1, 0]].unwrap() - c(0.42, 0.0)).norm() < 1.0e-14);
    assert!((phase_opposed_result[[1, 0]].unwrap() - c(-0.18, 0.0)).norm() < 1.0e-14);
    assert_eq!(in_phase_result[[1, 2]], None);
    assert_eq!(phase_opposed_result[[1, 2]], None);
}

#[test]
fn touchstone_active_reflection_all_zero_rows_are_all_undefined() {
    let network = parse_touchstone_v1_0_s(INPUT, 3).expect("three-port Touchstone input");
    let zero_drive = Array2::zeros((2, 3));
    let result = network
        .active_reflection_power(&zero_drive)
        .expect("zero rows remain structurally valid");
    assert!(result.iter().all(Option::is_none));
}
