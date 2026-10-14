//! Touchstone ingress -> phase-dependent coherent active reflection.
//!
//! A zero incident wave at port 2 leaves that coordinate undefined even
//! though coupling produces a nonzero outgoing wave there, so the public
//! result reports `None` only at that coordinate.

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

fn main() -> rfkit_touchstone::Result<()> {
    let network = parse_touchstone_v1_0_s(INPUT, 3)?;
    let in_phase = Array2::from_shape_vec(
        (2, 3),
        vec![
            c(1.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 0.0),
        ],
    )
    .expect("in-phase drive shape");
    let phase_opposed = Array2::from_shape_vec(
        (2, 3),
        vec![
            c(1.0, 0.0),
            c(-1.0, 0.0),
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(-1.0, 0.0),
            c(0.0, 0.0),
        ],
    )
    .expect("phase-opposed drive shape");

    let in_phase_result = network.active_reflection_power(&in_phase)?;
    let phase_opposed_result = network.active_reflection_power(&phase_opposed)?;

    for frequency in 0..2 {
        println!(
            "{} Hz in-phase: {:?}",
            network.frequency().hz()[frequency],
            in_phase_result.row(frequency).to_vec()
        );
        println!(
            "{} Hz phase-opposed: {:?}",
            network.frequency().hz()[frequency],
            phase_opposed_result.row(frequency).to_vec()
        );
    }

    assert_eq!(in_phase_result[[0, 2]], None);
    assert_eq!(phase_opposed_result[[0, 2]], None);
    assert!((in_phase_result[[0, 0]].unwrap() - c(0.35, 0.0)).norm() < 1.0e-14);
    assert!((phase_opposed_result[[0, 0]].unwrap() - c(-0.15, 0.0)).norm() < 1.0e-14);
    assert_ne!(in_phase_result[[0, 0]], phase_opposed_result[[0, 0]]);

    let zero_drive = Array2::zeros((2, 3));
    let zero_result = network.active_reflection_power(&zero_drive)?;
    assert!(zero_result.iter().all(Option::is_none));
    Ok(())
}
