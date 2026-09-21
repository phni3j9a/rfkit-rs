use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

const INPUT: &str = "# Hz S RI R 50\n\
1000000000 0.11 0.01 0.12 0.02 0.13 0.03\n\
0.21 0.01 0.22 0.02 0.23 0.03\n\
0.31 0.01 0.32 0.02 0.33 0.03\n\
2000000000 0.41 0.01 0.42 0.02 0.43 0.03\n\
0.51 0.01 0.52 0.02 0.53 0.03\n\
0.61 0.01 0.62 0.02 0.63 0.03\n";

const EXPECTED_OUTPUT: &str = "# Hz S RI R 50\n\
1000000000 0.33 0.03 0.31 0.01 0.32 0.02\n\
0.13 0.03 0.11 0.01 0.12 0.02\n\
0.23 0.03 0.21 0.01 0.22 0.02\n\
2000000000 0.63 0.03 0.61 0.01 0.62 0.02\n\
0.43 0.03 0.41 0.01 0.42 0.02\n\
0.53 0.03 0.51 0.01 0.52 0.02\n";

#[test]
fn touchstone_ingress_permute_export_reload_preserves_physical_order() {
    let source = parse_touchstone_v1_0_s(INPUT, 3).expect("asymmetric 3-port input parses");
    let source_frequency = source.frequency().clone();
    let source_s = source.s().clone();
    let source_z0 = source.z0().clone();

    // `order[new_port] = old_port`: new port 0 is old port 2, new port 1 is
    // old port 0, and new port 2 is old port 1.  These values are written out
    // independently instead of being obtained from `permute_ports`, so both
    // S axes and the mapping direction are observable in the assertion.
    let expected_s = Array3::from_shape_vec(
        (2, 3, 3),
        vec![
            Complex64::new(0.33, 0.03),
            Complex64::new(0.31, 0.01),
            Complex64::new(0.32, 0.02),
            Complex64::new(0.13, 0.03),
            Complex64::new(0.11, 0.01),
            Complex64::new(0.12, 0.02),
            Complex64::new(0.23, 0.03),
            Complex64::new(0.21, 0.01),
            Complex64::new(0.22, 0.02),
            Complex64::new(0.63, 0.03),
            Complex64::new(0.61, 0.01),
            Complex64::new(0.62, 0.02),
            Complex64::new(0.43, 0.03),
            Complex64::new(0.41, 0.01),
            Complex64::new(0.42, 0.02),
            Complex64::new(0.53, 0.03),
            Complex64::new(0.51, 0.01),
            Complex64::new(0.52, 0.02),
        ],
    )
    .expect("expected S shape");
    let expected_z0 = Array2::from_elem((2, 3), Complex64::new(50.0, 0.0));

    let permuted = source
        .permute_ports(&[2, 0, 1])
        .expect("a complete non-involutive permutation succeeds");
    assert_eq!(permuted.frequency(), &source_frequency);
    assert_eq!(permuted.s(), &expected_s);
    assert_eq!(permuted.z0(), &expected_z0);

    // The operation is borrowing/pure.  In particular, preserving the source
    // here catches an in-place implementation that would make repeated
    // workflow use depend on call order.
    assert_eq!(source.frequency(), &source_frequency);
    assert_eq!(source.s(), &source_s);
    assert_eq!(source.z0(), &source_z0);

    // Check the physical export itself, not only a writer/parser round trip.
    // This keeps a matching pair of inverse layout bugs from hiding a wrong
    // port direction.
    let text = write_touchstone_v1_0_s_ri_hz(&permuted).expect("writer accepts common 50 ohm z0");
    assert_eq!(text, EXPECTED_OUTPUT);

    let reloaded = parse_touchstone_v1_0_s(&text, 3).expect("exported text parses");
    assert_eq!(reloaded.frequency(), &source_frequency);
    assert_eq!(reloaded.s(), &expected_s);
    assert_eq!(reloaded.z0(), &expected_z0);
}
