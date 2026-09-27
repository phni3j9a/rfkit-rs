//! Touchstone v1 ingress -> explicit lossless reference-plane shift -> analysis -> v1 round trip.

use ndarray::Array2;
use num_complex::Complex64;
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

const INPUT: &str = "# Hz S RI R 50\n\
1000000000 0.2 0.0 0.4 0.0 0.05 0.0 0.15 0.0\n\
2000000000 0.2 0.0 0.36842439760115409 -0.15576733692346023 0.05 0.0 0.15 0.0\n\
3000000000 0.2 0.0 0.27868268373886618 -0.28694243635980915 0.05 0.0 0.15 0.0\n";

const ONE_WAY_PHASE_RAD: [[f64; 2]; 3] = [[-0.02, -0.03], [-0.04, -0.06], [-0.06, -0.09]];

fn main() -> rfkit_touchstone::Result<()> {
    let source = parse_touchstone_v1_0_s(INPUT, 2)?;
    let source_snapshot = source.clone();
    let phase = Array2::from_shape_vec(
        (3, 2),
        ONE_WAY_PHASE_RAD
            .into_iter()
            .flat_map(|row| row.into_iter())
            .collect(),
    )
    .expect("phase table has the declared shape");
    let shifted = source.shift_reference_planes_lossless_power(&phase)?;

    for sample in 0..3 {
        let expected_reflection = Complex64::from_polar(0.2, 0.04 * (sample + 1) as f64);
        let expected_transmission =
            Complex64::from_polar(0.4, -0.4 * sample as f64 + 0.05 * (sample + 1) as f64);
        assert!((shifted.s()[[sample, 0, 0]] - expected_reflection).norm() < 1.0e-14);
        assert!((shifted.s()[[sample, 1, 0]] - expected_transmission).norm() < 1.0e-14);
    }

    let group_delay = shifted.group_delay_secant_power(1, 0)?;
    let expected_group_delay = 0.35 / (std::f64::consts::TAU * 1.0e9);
    assert_eq!(group_delay.len(), 2);
    for delay in group_delay {
        assert!((delay - expected_group_delay).abs() < 1.0e-13);
    }

    let text = write_touchstone_v1_0_s_ri_hz(&shifted)?;
    assert!(text.starts_with("# Hz S RI R 50\n"));
    let reread = parse_touchstone_v1_0_s(&text, 2)?;
    assert_eq!(reread, shifted);
    assert_eq!(source, source_snapshot);
    Ok(())
}
