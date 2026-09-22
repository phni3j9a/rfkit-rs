//! Touchstone ingress → sampled two-port power-wave stability metrics.
//!
//! The display keeps each metric beside its source frequency and makes an
//! exactly unilateral sample explicit as `undefined`; it does not classify
//! the sweep as stable or unstable. The usual linear two-port interpretation
//! requires both `K > 1` and `|delta| < 1`, together with its auxiliary
//! provisos. These sampled external S-parameters cannot certify internal
//! poles, unsampled frequencies, nonlinear or large-signal behavior, or
//! overall circuit stability.

use rfkit_touchstone::parse_touchstone_v1_0_s;

const INPUT: &str = r#"# Hz S RI R 50
1000000000 0.0 0.0 2.0 0.0 0.1 0.0 0.0 0.0
2000000000 0.1 0.0 0.5 0.0 0.0 0.0 0.2 0.0
3000000000 0.2 0.1 -0.4 0.25 0.3 -0.2 0.1 -0.3
"#;

fn main() -> rfkit_touchstone::Result<()> {
    let network = parse_touchstone_v1_0_s(INPUT, 2)?;
    let metrics = network.two_port_stability_power()?;
    for (frequency, metric) in network.frequency().hz().iter().zip(metrics.iter()) {
        match metric.rollet_k {
            Some(k) => println!(
                "{frequency:.0} Hz: K={k:.6}, |delta|={:.6}",
                metric.delta.norm()
            ),
            None => println!(
                "{frequency:.0} Hz: K=undefined, |delta|={:.6}",
                metric.delta.norm()
            ),
        }
    }
    Ok(())
}
