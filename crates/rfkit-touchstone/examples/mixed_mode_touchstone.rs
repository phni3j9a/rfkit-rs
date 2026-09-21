//! Touchstone ingress → explicit physical pairing → mixed mode → restore/export.
//!
//! The input intentionally uses a physical order that is not pair-adjacent.
//! Mixed-mode metadata is not written to Touchstone; the inverse conversion
//! restores ordinary single-ended coordinates before using the existing writer.

use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

const INPUT: &str = r#"# Hz S RI R 50
1000000000 0.11 0.01 0.12 0.02 0.13 0.03 0.14 0.04
0.15 0.05
0.21 0.06 0.22 0.07 0.23 0.08 0.24 0.09
0.25 0.10
0.31 0.11 0.32 0.12 0.37 0.17 0.34 0.14
0.35 0.15
0.41 0.16 0.42 0.17 0.43 0.18 0.48 0.23
0.45 0.20
0.51 0.21 0.52 0.22 0.53 0.23 0.54 0.24
0.58 0.28
2000000000 -0.11 0.21 -0.12 0.22 -0.13 0.23 -0.14 0.24
-0.15 0.25
-0.21 0.26 -0.22 0.27 -0.23 0.28 -0.24 0.29
-0.25 0.30
-0.31 0.31 -0.32 0.32 -0.37 0.37 -0.34 0.34
-0.35 0.35
-0.41 0.36 -0.42 0.37 -0.43 0.38 -0.48 0.43
-0.45 0.40
-0.51 0.41 -0.52 0.42 -0.53 0.43 -0.54 0.44
-0.58 0.48
"#;

fn main() -> rfkit_touchstone::Result<()> {
    let source = parse_touchstone_v1_0_s(INPUT, 5)?;
    // New-port -> old-port: pair (old 2, old 0), pair (old 3, old 1), old 4.
    let pair_ordered = source.permute_ports(&[2, 0, 3, 1, 4])?;
    let mixed = pair_ordered.to_mixed_mode_equal_pair_power(2)?;
    assert_eq!(mixed.z0()[[0, 0]], num_complex::Complex64::new(100.0, 0.0));
    assert_eq!(mixed.z0()[[0, 2]], num_complex::Complex64::new(25.0, 0.0));
    // Independent checks of one differential response, one common response,
    // and a differential/common mode-conversion response for the literal
    // first-frequency input.
    assert!((mixed.s()[[0, 0, 0]] - num_complex::Complex64::new(0.02, 0.02)).norm() < 1e-12);
    assert!((mixed.s()[[0, 2, 2]] - num_complex::Complex64::new(0.46, 0.16)).norm() < 1e-12);
    assert!((mixed.s()[[0, 0, 2]] - num_complex::Complex64::new(0.22, 0.12)).norm() < 1e-12);

    // Mixed mode is a declared coordinate layout, not a Touchstone extension.
    // Restore it and undo the physical permutation before writing.
    let restored = mixed
        .to_single_ended_equal_pair_power(2)?
        .permute_ports(&[1, 3, 0, 2, 4])?;
    let text = write_touchstone_v1_0_s_ri_hz(&restored)?;
    let reread = parse_touchstone_v1_0_s(&text, 5)?;
    assert_eq!(reread.frequency(), restored.frequency());
    assert_eq!(reread.z0(), restored.z0());
    for (actual, expected) in reread.s().iter().zip(restored.s().iter()) {
        assert!((*actual - *expected).norm() <= 1.0e-12);
    }
    print!("{text}");
    Ok(())
}
