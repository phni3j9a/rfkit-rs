//! Touchstone ingress -> uncoupled direct-sum bank -> explicit port reorder ->
//! Touchstone 2.0 Full export/readback.

use num_complex::Complex64;
use rfkit_touchstone::{
    TouchstoneParameter, parse_touchstone_v1_0_s, parse_touchstone_v2_0_power,
    write_touchstone_v2_0_power_full_ri_hz,
};

const INPUT_A: &str = r#"# Hz S RI R 40
1000000000 0.10 0.01 0.20 0.02 0.30 0.03 0.40 0.04
2000000000 0.11 0.01 0.21 0.02 0.31 0.03 0.41 0.04
"#;

const INPUT_B: &str = r#"# Hz S RI R 70
1000000000 0.11 0.01 0.12 0.02 0.13 0.03
0.21 0.01 0.22 0.02 0.23 0.03
0.31 0.01 0.32 0.02 0.33 0.03
2000000000 0.41 0.04 0.42 0.05 0.43 0.06
0.51 0.04 0.52 0.05 0.53 0.06
0.61 0.04 0.62 0.05 0.63 0.06
"#;

fn main() -> rfkit_touchstone::Result<()> {
    let a = parse_touchstone_v1_0_s(INPUT_A, 2)?;
    let b = parse_touchstone_v1_0_s(INPUT_B, 3)?;
    let bank = a.combine_uncoupled(&b)?;
    assert_eq!(bank.nports(), 5);

    // New-port -> old-port mapping: B0, A0, A1, B2, B1.  This explicit
    // reorder demonstrates that the direct sum remains usable by the normal
    // physical port-selection API.
    let reordered = bank.select_ports_zero_incident(&[2, 0, 1, 4, 3])?;
    let mapping = [("B", 0), ("A", 0), ("A", 1), ("B", 2), ("B", 1)];
    for frequency in 0..reordered.frequency().len() {
        for row in 0..mapping.len() {
            for column in 0..mapping.len() {
                let expected = match (mapping[row].0, mapping[column].0) {
                    ("A", "A") => a.s()[[frequency, mapping[row].1, mapping[column].1]],
                    ("B", "B") => b.s()[[frequency, mapping[row].1, mapping[column].1]],
                    _ => Complex64::new(0.0, 0.0),
                };
                assert_eq!(reordered.s()[[frequency, row, column]], expected);
            }
        }
    }
    assert_eq!(reordered.z0()[[0, 0]], Complex64::new(70.0, 0.0));
    assert_eq!(reordered.z0()[[0, 1]], Complex64::new(40.0, 0.0));
    assert_eq!(reordered.z0()[[0, 2]], Complex64::new(40.0, 0.0));
    assert_eq!(reordered.z0()[[0, 3]], Complex64::new(70.0, 0.0));
    assert_eq!(reordered.z0()[[0, 4]], Complex64::new(70.0, 0.0));

    let text = write_touchstone_v2_0_power_full_ri_hz(&reordered, TouchstoneParameter::S)
        .expect("unequal real per-port references are writer-supported");
    let reread = parse_touchstone_v2_0_power(&text)?;
    assert_eq!(reread.frequency(), reordered.frequency());
    assert_eq!(reread.s(), reordered.s());
    assert_eq!(reread.z0(), reordered.z0());
    print!("{text}");
    Ok(())
}
