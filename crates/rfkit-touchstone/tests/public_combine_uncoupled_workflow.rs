use num_complex::Complex64;
use rfkit_touchstone::{
    parse_touchstone_v1_0_s, parse_touchstone_v2_0_power, write_touchstone_v2_0_s_full_ri_hz,
};

const INPUT_A: &str = r#"# Hz S RI R 40
1000000000 0.10 0.01 0.20 0.02 0.30 0.03 0.40 0.04
"#;

const INPUT_B: &str = r#"# Hz S RI R 70
1000000000 0.11 0.01 0.12 0.02 0.13 0.03
0.21 0.01 0.22 0.02 0.23 0.03
0.31 0.01 0.32 0.02 0.33 0.03
"#;

#[test]
fn touchstone_load_combine_reorder_and_v2_round_trip_preserves_blocks_and_zeros() {
    let a = parse_touchstone_v1_0_s(INPUT_A, 2).unwrap();
    let b = parse_touchstone_v1_0_s(INPUT_B, 3).unwrap();
    let bank = a.combine_uncoupled(&b).unwrap();
    let reordered = bank.select_ports_zero_incident(&[2, 0, 1, 4, 3]).unwrap();
    let mapping = [("B", 0), ("A", 0), ("A", 1), ("B", 2), ("B", 1)];

    for row in 0..mapping.len() {
        for column in 0..mapping.len() {
            let expected = match (mapping[row].0, mapping[column].0) {
                ("A", "A") => a.s()[[0, mapping[row].1, mapping[column].1]],
                ("B", "B") => b.s()[[0, mapping[row].1, mapping[column].1]],
                _ => Complex64::new(0.0, 0.0),
            };
            assert_eq!(reordered.s()[[0, row, column]], expected);
        }
    }
    assert_eq!(
        reordered.z0().row(0).to_vec(),
        vec![
            Complex64::new(70.0, 0.0),
            Complex64::new(40.0, 0.0),
            Complex64::new(40.0, 0.0),
            Complex64::new(70.0, 0.0),
            Complex64::new(70.0, 0.0),
        ]
    );

    let text = write_touchstone_v2_0_s_full_ri_hz(&reordered).unwrap();
    assert!(text.contains("[Reference] 70 40 40 70 70\n"));
    let reread = parse_touchstone_v2_0_power(&text).unwrap();
    assert_eq!(reread.s(), reordered.s());
    assert_eq!(reread.z0(), reordered.z0());
    assert_eq!(reread.frequency(), reordered.frequency());
}
