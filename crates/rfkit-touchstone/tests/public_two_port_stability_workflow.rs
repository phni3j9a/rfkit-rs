use num_complex::Complex64;
use rfkit_touchstone::parse_touchstone_v1_0_s;

const INPUT: &str = r#"# Hz S RI R 50
1000000000 0.0 0.0 2.0 0.0 0.1 0.0 0.0 0.0
2000000000 0.1 0.0 0.5 0.0 0.0 0.0 0.2 0.0
3000000000 0.2 0.1 -0.4 0.25 0.3 -0.2 0.1 -0.3
"#;

#[test]
fn touchstone_ingress_two_port_stability_keeps_alignment_and_source_unchanged() {
    let source = parse_touchstone_v1_0_s(INPUT, 2).expect("two-port sweep parses");
    let source_snapshot = source.clone();
    let metrics = source
        .two_port_stability_power()
        .expect("positive-real Touchstone references support stability metrics");

    assert_eq!(metrics.len(), source.frequency().len());
    assert!((metrics[0].delta - Complex64::new(-0.2, 0.0)).norm() <= 1.0e-14);
    assert!((metrics[0].rollet_k.unwrap() - 2.6).abs() <= 1.0e-14);
    assert!((metrics[1].delta - Complex64::new(0.02, 0.0)).norm() <= 1.0e-14);
    assert_eq!(metrics[1].rollet_k, None);
    assert!((metrics[2].delta - Complex64::new(0.12, -0.205)).norm() <= 1.0e-14);
    assert!(metrics[2].rollet_k.is_some());

    let aligned = source
        .frequency()
        .hz()
        .iter()
        .zip(metrics.iter())
        .map(|(frequency, metric)| {
            let k = metric
                .rollet_k
                .map_or_else(|| "undefined".to_owned(), |value| format!("{value:.6}"));
            format!(
                "{frequency:.0} Hz: K={k}, |delta|={:.6}",
                metric.delta.norm()
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(aligned[0], "1000000000 Hz: K=2.600000, |delta|=0.200000");
    assert_eq!(aligned[1], "2000000000 Hz: K=undefined, |delta|=0.020000");

    assert_eq!(source, source_snapshot);
}
