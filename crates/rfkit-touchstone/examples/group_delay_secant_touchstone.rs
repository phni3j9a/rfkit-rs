//! Touchstone ingress -> selected S21 -> adjacent interval group-delay seconds.
//!
//! The returned values are attached to the displayed frequency apertures;
//! they are not sample-aligned gradient values.

use rfkit_touchstone::parse_touchstone_v1_0_s;

const INPUT: &str = "# Hz S RI R 50\n\
100000000 0.10 0.00 0.247213595499958 -0.760845213036123 0.03 0.00 0.10 0.00\n\
130000000 0.10 0.00 -0.050232415623451 -0.798421382742617 0.03 0.00 0.10 0.00\n\
180000000 0.10 0.00 -0.509939191798952 -0.616410594220632 0.03 0.00 0.10 0.00\n";

fn main() -> rfkit_touchstone::Result<()> {
    let network = parse_touchstone_v1_0_s(INPUT, 2)?;
    let delay_seconds = network.group_delay_secant_power(1, 0)?;
    let frequencies = network.frequency().hz();

    assert_eq!(delay_seconds.len(), frequencies.len() - 1);
    for (interval, delay) in delay_seconds.iter().enumerate() {
        let lower = frequencies[interval];
        let upper = frequencies[interval + 1];
        assert!((lower, upper) == (100.0e6, 130.0e6) || (lower, upper) == (130.0e6, 180.0e6));
        assert!((*delay - 2.0e-9).abs() < 5.0e-12);
        println!("S21 interval [{lower:.0}, {upper:.0}] Hz: group delay = {delay:.12e} s");
    }
    Ok(())
}
