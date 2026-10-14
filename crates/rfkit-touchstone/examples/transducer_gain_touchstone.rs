//! Touchstone ingress -> explicit sample selection -> two-port transducer
//! power gain.
//!
//! Source/load tables are caller data.  The workflow keeps the matched and
//! mismatched load cases beside the analytical ideal-through result instead
//! of inferring a default 50-ohm source or load.

use num_complex::Complex64;
use rfkit_touchstone::parse_touchstone_v1_0_s;

const INPUT: &str = r#"# GHz S RI R 50
1 0 0 1 0 1 0 0 0
2 0 0 1 0 1 0 0 0
3 0 0 1 0 1 0 0 0
"#;

fn main() -> rfkit_touchstone::Result<()> {
    let source = parse_touchstone_v1_0_s(INPUT, 2)?;
    let selected = source.select_frequency_samples(&[0, 2])?;
    let source_impedance = [Complex64::new(50.0, 0.0); 2];
    let load_impedance = [Complex64::new(50.0, 0.0), Complex64::new(83.0, -9.0)];
    let gain = selected.two_port_transducer_gain_power(&source_impedance, &load_impedance)?;
    let expected_matched = 1.0;
    let expected_mismatched = 4.0 * source_impedance[1].re * load_impedance[1].re
        / (source_impedance[1] + load_impedance[1]).norm_sqr();
    assert!((gain[0] - expected_matched).abs() <= 1.0e-14);
    assert!((gain[1] - expected_mismatched).abs() <= 1.0e-14);
    for (frequency, value) in selected.frequency().hz().iter().zip(gain.iter()) {
        println!("{frequency:.0} Hz: transducer power gain={value:.12}");
    }
    Ok(())
}
