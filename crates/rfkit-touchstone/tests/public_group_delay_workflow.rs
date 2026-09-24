use rfkit_touchstone::parse_touchstone_v1_0_s;

const INPUT: &str = "# Hz S RI R 50\n\
100000000 0.10 0.00 0.247213595499958 -0.760845213036123 0.03 0.00 0.10 0.00\n\
130000000 0.10 0.00 -0.050232415623451 -0.798421382742617 0.03 0.00 0.10 0.00\n\
180000000 0.10 0.00 -0.509939191798952 -0.616410594220632 0.03 0.00 0.10 0.00\n";

#[test]
fn touchstone_load_selected_s21_returns_interval_seconds_and_bounds() {
    let network = parse_touchstone_v1_0_s(INPUT, 2).expect("Touchstone sweep parses");
    let snapshot = network.clone();
    let result = network
        .group_delay_secant_power(1, 0)
        .expect("nonzero S21 trace has defined adjacent phase");
    assert_eq!(network, snapshot);
    assert_eq!(network.frequency().hz(), &[100.0e6, 130.0e6, 180.0e6]);
    assert_eq!(result.len(), 2);
    for (interval, delay) in result.iter().enumerate() {
        let (lower, upper) = (
            network.frequency().hz()[interval],
            network.frequency().hz()[interval + 1],
        );
        assert!(lower < upper);
        assert!(matches!(
            (lower, upper),
            (100.0e6, 130.0e6) | (130.0e6, 180.0e6)
        ));
        assert!((*delay - 2.0e-9).abs() < 5.0e-12);
    }
}
