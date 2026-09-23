//! Touchstone ingress → sampled maximum power-wave singular-value inspection.
//!
//! The returned scalar is the strongest simultaneous reflected-wave amplitude
//! ratio for each source sample.  Its square is the corresponding power ratio;
//! this example intentionally does not classify the sweep as passive or
//! active.

use rfkit_touchstone::parse_touchstone_v1_0_s;

const INPUT: &str = r#"# Hz S RI R 50
1000000000 0.2 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.3 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.1 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.25 0.0
2000000000 0.6 0.0 0.6 0.0 0.0 0.0 0.0 0.0
0.6 0.0 0.6 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
"#;

fn main() -> rfkit_touchstone::Result<()> {
    let network = parse_touchstone_v1_0_s(INPUT, 4)?;
    let sigma_max = network.max_singular_value_power()?;
    for (frequency, sigma) in network.frequency().hz().iter().zip(sigma_max.iter()) {
        println!(
            "{frequency:.0} Hz: sigma_max={sigma:.6}, reflected/incident power ratio={:.6}",
            sigma * sigma
        );
    }
    Ok(())
}
