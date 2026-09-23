//! Touchstone ingress → simultaneous coupled-group cascade → inverse removal.
//!
//! The output is explicitly renormalized to the writer's one common positive
//! real reference before the write/read boundary is crossed.

use ndarray::Array2;
use num_complex::Complex64;
use rfkit_core::Network;
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

const INPUT_A: &str = r#"# Hz S RI R 50
1000000000 0.05 0 0.01 0.002 0.36 0.01 0.02 -0.003
0.012 -0.001 0.045 0.005 0.018 0.004 0.33 -0.008
0.42 -0.012 0.025 0.003 0.04 -0.004 0.01 0.001
0.021 0.002 0.39 0.009 0.013 -0.002 0.048 0.003
2000000000 0.055 0.001 0.012 0.002 0.37 0.011 0.021 -0.003
0.013 -0.001 0.047 0.005 0.019 0.004 0.34 -0.008
0.43 -0.012 0.026 0.003 0.042 -0.004 0.011 0.001
0.022 0.002 0.40 0.009 0.014 -0.002 0.050 0.003
"#;

const INPUT_B: &str = r#"# Hz S RI R 50
1000000000 0.04 -0.002 0.009 0.001 0.31 -0.006 0.018 0.002
0.011 0.001 0.052 -0.004 0.015 -0.002 0.28 0.007
0.35 0.008 0.022 -0.003 0.035 0.002 0.012 -0.001
0.017 -0.002 0.32 0.006 0.010 0.001 0.046 -0.003
2000000000 0.043 -0.002 0.010 0.001 0.32 -0.006 0.019 0.002
0.012 0.001 0.054 -0.004 0.016 -0.002 0.29 0.007
0.36 0.008 0.023 -0.003 0.037 0.002 0.013 -0.001
0.018 -0.002 0.33 0.006 0.011 0.001 0.048 -0.003
"#;

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn close(actual: &Network, expected: &Network) -> bool {
    actual.frequency() == expected.frequency()
        && actual.z0() == expected.z0()
        && actual
            .s()
            .iter()
            .zip(expected.s())
            .all(|(actual, expected)| (*actual - *expected).norm() <= 2.0e-10)
}

fn main() -> rfkit_touchstone::Result<()> {
    let a = parse_touchstone_v1_0_s(INPUT_A, 4)?;
    let b = parse_touchstone_v1_0_s(INPUT_B, 4)?;
    let measured = a.cascade_direct_power(&b)?;

    let recovered = a.inverse_cascade_power()?.cascade_direct_power(&measured)?;
    let writer_ready =
        recovered.renormalize_direct_power(Array2::from_elem((2, 4), c(50.0, 0.0)))?;
    assert!(close(&writer_ready, &b));

    let text = write_touchstone_v1_0_s_ri_hz(&writer_ready)?;
    let reread = parse_touchstone_v1_0_s(&text, 4)?;
    assert!(close(&reread, &writer_ready));
    print!("{text}");
    Ok(())
}
