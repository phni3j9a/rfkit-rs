//! Select caller-identified observations from two unequal-grid measurements
//! before connecting them at a shared physical junction.

use ndarray::Array3;
use num_complex::Complex64;
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

const INPUT_A: &str = r#"# GHz S RI R 50
1 0.10 0.00 0.20 0.00 0.30 0.00 0.10 0.00
2 0.11 0.00 0.21 0.00 0.31 0.00 0.11 0.00
3 0.12 0.00 0.22 0.00 0.32 0.00 0.12 0.00
4 0.13 0.00 0.23 0.00 0.33 0.00 0.13 0.00
"#;

const INPUT_B: &str = r#"# GHz S RI R 50
0.5 0.40 0.00 0.50 0.00 0.60 0.00 0.70 0.00
2 0.41 0.00 0.51 0.00 0.61 0.00 0.71 0.00
3 0.42 0.00 0.52 0.00 0.62 0.00 0.72 0.00
4 0.43 0.00 0.53 0.00 0.63 0.00 0.73 0.00
5 0.44 0.00 0.54 0.00 0.64 0.00 0.74 0.00
"#;

const A_SAMPLES: [usize; 2] = [1, 3];
const B_SAMPLES: [usize; 2] = [1, 3];
const CONNECTED_A_PORT: usize = 1;
const CONNECTED_B_PORT: usize = 0;

fn c(real: f64) -> Complex64 {
    Complex64::new(real, 0.0)
}

/// Independent matched-junction equation for the selected A port 1 and B
/// port 0.  The result order is A port 0 followed by B port 1.
fn independently_expected(a: &rfkit_core::Network, b: &rfkit_core::Network) -> Array3<Complex64> {
    Array3::from_shape_fn((a.frequency().len(), 2, 2), |(frequency, output, input)| {
        let a_ee = a.s()[[frequency, 0, 0]];
        let a_ei = a.s()[[frequency, 0, CONNECTED_A_PORT]];
        let a_ie = a.s()[[frequency, CONNECTED_A_PORT, 0]];
        let a_ii = a.s()[[frequency, CONNECTED_A_PORT, CONNECTED_A_PORT]];
        let b_ee = b.s()[[frequency, 1, 1]];
        let b_ei = b.s()[[frequency, 1, CONNECTED_B_PORT]];
        let b_ie = b.s()[[frequency, CONNECTED_B_PORT, 1]];
        let b_ii = b.s()[[frequency, CONNECTED_B_PORT, CONNECTED_B_PORT]];
        let denominator = c(1.0) - a_ii * b_ii;
        match (output, input) {
            (0, 0) => a_ee + a_ei * b_ii * a_ie / denominator,
            (0, 1) => a_ei * b_ie / denominator,
            (1, 0) => b_ei * a_ie / denominator,
            (1, 1) => b_ee + b_ei * a_ii * b_ie / denominator,
            _ => unreachable!(),
        }
    })
}

fn main() -> rfkit_touchstone::Result<()> {
    let source_a = parse_touchstone_v1_0_s(INPUT_A, 2)?;
    let source_b = parse_touchstone_v1_0_s(INPUT_B, 2)?;
    assert_ne!(source_a.frequency(), source_b.frequency());

    // The caller explicitly identifies the observations to retain.  No
    // frequency intersection or nearest-label policy is inferred here.
    let a = source_a.select_frequency_samples(&A_SAMPLES)?;
    let b = source_b.select_frequency_samples(&B_SAMPLES)?;
    assert_eq!(a.frequency(), b.frequency());
    let expected = independently_expected(&a, &b);
    let connected = a.connect_power(CONNECTED_A_PORT, &b, CONNECTED_B_PORT)?;

    assert_eq!(connected.frequency().hz(), &[2.0e9, 4.0e9]);
    for (actual, expected) in connected.s().iter().zip(expected.iter()) {
        assert!((*actual - *expected).norm() <= 1.0e-12);
    }
    let text = write_touchstone_v1_0_s_ri_hz(&connected)?;
    let reread = parse_touchstone_v1_0_s(&text, connected.nports())?;
    assert_eq!(reread.frequency(), connected.frequency());
    assert_eq!(reread.s(), connected.s());
    assert_eq!(reread.z0(), connected.z0());
    print!("{text}");
    Ok(())
}
