use num_complex::Complex64;
use rfkit_touchstone::parse_touchstone_v1_0_s;

const INPUT: &str = r#"# GHz S RI R 50
1 0 0 1 0 1 0 0 0
2 0 0 1 0 1 0 0 0
3 0 0 1 0 1 0 0 0
"#;

#[test]
fn touchstone_ingress_selection_and_transducer_gain_cover_matched_and_mismatched_loads() {
    let source = parse_touchstone_v1_0_s(INPUT, 2).expect("Touchstone two-port parses");
    let snapshot = source.clone();
    let selected = source
        .select_frequency_samples(&[0, 2])
        .expect("caller-selected observations are valid");
    assert_eq!(selected.frequency().hz(), &[1.0e9, 3.0e9]);

    let source_impedance = [Complex64::new(50.0, 0.0); 2];
    let load_impedance = [Complex64::new(50.0, 0.0), Complex64::new(83.0, -9.0)];
    let gain = selected
        .two_port_transducer_gain_power(&source_impedance, &load_impedance)
        .expect("positive-real source/load tables are valid");
    assert!((gain[0] - 1.0).abs() <= 1.0e-14);
    let expected_mismatched = 4.0 * source_impedance[1].re * load_impedance[1].re
        / (source_impedance[1] + load_impedance[1]).norm_sqr();
    assert!((gain[1] - expected_mismatched).abs() <= 1.0e-14);
    assert!(gain[1] < gain[0]);
    assert_eq!(source, snapshot);
}
