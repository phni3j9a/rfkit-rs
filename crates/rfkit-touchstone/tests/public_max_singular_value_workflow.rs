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

#[test]
fn touchstone_load_to_maximum_singular_value_inspection_keeps_alignment() {
    let source = parse_touchstone_v1_0_s(INPUT, 4).expect("four-port sweep parses");
    let source_snapshot = source.clone();
    let sigma_max = source
        .max_singular_value_power()
        .expect("positive-real Touchstone references support the diagnostic");

    assert_eq!(sigma_max.len(), 2);
    assert!((sigma_max[0] - 0.3).abs() <= 1.0e-14);
    assert!((sigma_max[1] - 1.2).abs() <= 1.0e-14);
    assert_eq!(source.frequency().hz(), &[1.0e9, 2.0e9]);
    assert_eq!(source, source_snapshot);
}
