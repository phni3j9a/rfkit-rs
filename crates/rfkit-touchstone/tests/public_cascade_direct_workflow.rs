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

fn assert_close(actual: &Network, expected: &Network, tolerance: f64) {
    assert_eq!(actual.frequency(), expected.frequency());
    assert_eq!(actual.z0(), expected.z0());
    assert_eq!(actual.s().dim(), expected.s().dim());
    for (index, (&actual, &expected)) in actual.s().iter().zip(expected.s()).enumerate() {
        let difference = (actual - expected).norm();
        assert!(
            difference <= tolerance,
            "index {index}: actual={actual:?}, expected={expected:?}, difference={difference:e}"
        );
    }
}

fn load(text: &str) -> rfkit_touchstone::Result<Network> {
    parse_touchstone_v1_0_s(text, 4)
}

#[test]
fn touchstone_load_simultaneous_cascade_inverse_remove_and_write_read() {
    let a = load(INPUT_A).unwrap();
    let b = load(INPUT_B).unwrap();
    let a_snapshot = a.clone();
    let b_snapshot = b.clone();
    let measured = a.cascade_direct_power(&b).unwrap();
    assert_eq!(measured.z0(), &Array2::from_elem((2, 4), c(50.0, 0.0)));

    // Remove A from the measured group cascade with the existing inverse
    // operation.  The direct cascade's output ordering makes the recovered
    // network [B.left..., B.right...] without a hidden permutation.
    let recovered = a
        .inverse_cascade_power()
        .unwrap()
        .cascade_direct_power(&measured)
        .unwrap();
    let common = Array2::from_elem((2, 4), c(50.0, 0.0));
    let writer_ready = recovered.renormalize_direct_power(common).unwrap();
    assert_close(&writer_ready, &b, 2.0e-10);

    let text = write_touchstone_v1_0_s_ri_hz(&writer_ready).unwrap();
    let reread = load(&text).unwrap();
    assert_close(&reread, &writer_ready, 2.0e-12);
    assert_eq!(a_snapshot, load(INPUT_A).unwrap());
    assert_eq!(b_snapshot, b);
}
