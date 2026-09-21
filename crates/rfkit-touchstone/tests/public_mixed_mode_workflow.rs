use ndarray::Array2;
use num_complex::Complex64;
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

const INPUT: &str = r#"# Hz S RI R 50
1000000000 0.11 0.01 0.12 0.02 0.13 0.03 0.14 0.04
0.15 0.05
0.21 0.06 0.22 0.07 0.23 0.08 0.24 0.09
0.25 0.10
0.31 0.11 0.32 0.12 0.37 0.17 0.34 0.14
0.35 0.15
0.41 0.16 0.42 0.17 0.43 0.18 0.48 0.23
0.45 0.20
0.51 0.21 0.52 0.22 0.53 0.23 0.54 0.24
0.58 0.28
2000000000 -0.11 0.21 -0.12 0.22 -0.13 0.23 -0.14 0.24
-0.15 0.25
-0.21 0.26 -0.22 0.27 -0.23 0.28 -0.24 0.29
-0.25 0.30
-0.31 0.31 -0.32 0.32 -0.37 0.37 -0.34 0.34
-0.35 0.35
-0.41 0.36 -0.42 0.37 -0.43 0.38 -0.48 0.43
-0.45 0.40
-0.51 0.41 -0.52 0.42 -0.53 0.43 -0.54 0.44
-0.58 0.48
"#;

const PHYSICAL_TO_PAIR_ORDER: &[usize] = &[2, 0, 3, 1, 4];
const PAIR_TO_PHYSICAL_ORDER: &[usize] = &[1, 3, 0, 2, 4];
const MODE_TOLERANCE: f64 = 1.0e-12;

fn assert_close(actual: Complex64, expected: Complex64, context: &str) {
    let difference = (actual - expected).norm();
    assert!(
        difference <= MODE_TOLERANCE,
        "{context}: actual={actual:?}, expected={expected:?}, difference={difference}"
    );
}

fn assert_network_close(actual: &rfkit_core::Network, expected: &rfkit_core::Network) {
    assert_eq!(actual.frequency(), expected.frequency());
    assert_eq!(actual.z0(), expected.z0());
    assert_eq!(actual.s().dim(), expected.s().dim());
    for (index, (left, right)) in actual.s().iter().zip(expected.s().iter()).enumerate() {
        assert_close(*left, *right, &format!("restored S scalar {index}"));
    }
}

fn mode_response(
    network: &rfkit_core::Network,
    frequency: usize,
    output: usize,
    input: usize,
) -> Complex64 {
    let coefficient = |mode: usize, port: usize| -> f64 {
        match mode {
            0 if port == 0 => 1.0 / 2.0_f64.sqrt(),
            0 if port == 1 => -1.0 / 2.0_f64.sqrt(),
            1 if port == 2 => 1.0 / 2.0_f64.sqrt(),
            1 if port == 3 => -1.0 / 2.0_f64.sqrt(),
            2 if port == 0 || port == 1 => 1.0 / 2.0_f64.sqrt(),
            3 if port == 2 || port == 3 => 1.0 / 2.0_f64.sqrt(),
            4 if port == 4 => 1.0,
            _ => 0.0,
        }
    };
    let mut value = Complex64::new(0.0, 0.0);
    for row in 0..5 {
        for column in 0..5 {
            value += network.s()[[frequency, row, column]]
                * coefficient(output, row)
                * coefficient(input, column);
        }
    }
    value
}

#[test]
fn touchstone_ingress_physical_permutation_mixed_mode_roundtrip_and_export() {
    let source = parse_touchstone_v1_0_s(INPUT, 5).expect("asymmetric 5-port input parses");
    let source_snapshot = source.clone();

    // The measurement labels do not arrive in the pair-adjacent layout.  The
    // physical map is explicit and is applied before any mode arithmetic:
    // new pair coordinates [0,1,2,3,4] are old [2,0,3,1,4].
    let pair_ordered = source
        .permute_ports(PHYSICAL_TO_PAIR_ORDER)
        .expect("complete physical port permutation succeeds");
    assert_eq!(
        pair_ordered.s()[[0, 0, 0]],
        source.s()[[0, 2, 2]],
        "permutation maps both S axes"
    );
    assert_eq!(source, source_snapshot, "ingress remains immutable");

    let mixed = pair_ordered
        .to_mixed_mode_equal_pair_power(2)
        .expect("two adjacent 50-ohm pairs convert to natural modal references");
    assert_eq!(
        mixed.z0(),
        &Array2::from_shape_vec(
            (2, 5),
            vec![
                Complex64::new(100.0, 0.0),
                Complex64::new(100.0, 0.0),
                Complex64::new(25.0, 0.0),
                Complex64::new(25.0, 0.0),
                Complex64::new(50.0, 0.0),
                Complex64::new(100.0, 0.0),
                Complex64::new(100.0, 0.0),
                Complex64::new(25.0, 0.0),
                Complex64::new(25.0, 0.0),
                Complex64::new(50.0, 0.0),
            ],
        )
        .unwrap()
    );

    // Check representative differential, common, and mode-conversion paths
    // against an independently written U S U^T sum.  This does not consume
    // mixed.s as its own expected value, so a matching forward/inverse bug
    // cannot hide a wrong coordinate order or polarity.
    for frequency in 0..2 {
        for (output, input, label) in [
            (0, 0, "d0<-d0"),
            (2, 2, "c0<-c0"),
            (0, 2, "d0<-c0"),
            (2, 0, "c0<-d0"),
            (1, 3, "d1<-c1"),
            (3, 1, "c1<-d1"),
            (0, 4, "d0<-se4"),
        ] {
            assert_close(
                mixed.s()[[frequency, output, input]],
                mode_response(&pair_ordered, frequency, output, input),
                &format!("frequency {frequency} {label}"),
            );
        }
    }

    let restored_pair_order = mixed
        .to_single_ended_equal_pair_power(2)
        .expect("natural modal references restore single-ended coordinates");
    let restored = restored_pair_order
        .permute_ports(PAIR_TO_PHYSICAL_ORDER)
        .expect("inverse physical permutation restores ingress order");
    assert_network_close(&restored, &source_snapshot);

    // The existing writer sees the restored single-ended, common 50-ohm
    // references.  No mixed-mode file extension or writer-side renormalization
    // is involved in this workflow.
    let text =
        write_touchstone_v1_0_s_ri_hz(&restored).expect("restored network is writer-compatible");
    let reread = parse_touchstone_v1_0_s(&text, 5).expect("writer output parses");
    assert_network_close(&reread, &source_snapshot);
}
