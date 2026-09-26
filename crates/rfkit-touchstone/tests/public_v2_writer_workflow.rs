use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_touchstone::{parse_touchstone_v2_0_s, write_touchstone_v2_0_s_full_ri_hz};

const INPUT: &str = "[Version] 2.0\n\
# Hz S RI R 37\n\
[Number of Ports] 3\n\
[Number of Frequencies] 2\n\
[Reference]\n\
37\n\
61 83\n\
[Matrix Format] Lower\n\
[Network Data]\n\
1000000000 0.11 0.01 0.21 0.04 0.22 0.02 0.31 0.05 0.32 0.06 0.33 0.03\n\
2000000000 0.41 0.04 0.51 0.07 0.52 0.05 0.61 0.08 0.62 0.09 0.63 0.06\n\
[End]\n";

#[test]
fn v2_parse_permute_write_read_preserves_physical_s_and_references() {
    let source = parse_touchstone_v2_0_s(INPUT).expect("unequal-reference v2 input parses");
    let analysis = source
        .max_singular_value_power()
        .expect("triangular input reaches existing public analysis");
    assert_eq!(analysis.len(), 2);
    assert!(analysis.iter().all(|value| value.is_finite()));
    let expected_s = Array3::from_shape_vec(
        (2, 3, 3),
        vec![
            Complex64::new(0.33, 0.03),
            Complex64::new(0.31, 0.05),
            Complex64::new(0.32, 0.06),
            Complex64::new(0.31, 0.05),
            Complex64::new(0.11, 0.01),
            Complex64::new(0.21, 0.04),
            Complex64::new(0.32, 0.06),
            Complex64::new(0.21, 0.04),
            Complex64::new(0.22, 0.02),
            Complex64::new(0.63, 0.06),
            Complex64::new(0.61, 0.08),
            Complex64::new(0.62, 0.09),
            Complex64::new(0.61, 0.08),
            Complex64::new(0.41, 0.04),
            Complex64::new(0.51, 0.07),
            Complex64::new(0.62, 0.09),
            Complex64::new(0.51, 0.07),
            Complex64::new(0.52, 0.05),
        ],
    )
    .expect("expected S shape");
    let expected_z0 = Array2::from_shape_vec(
        (2, 3),
        vec![
            Complex64::new(83.0, 0.0),
            Complex64::new(37.0, 0.0),
            Complex64::new(61.0, 0.0),
            Complex64::new(83.0, 0.0),
            Complex64::new(37.0, 0.0),
            Complex64::new(61.0, 0.0),
        ],
    )
    .expect("expected z0 shape");

    let reordered = source
        .permute_ports(&[2, 0, 1])
        .expect("physical permutation succeeds");
    assert_eq!(reordered.s(), &expected_s);
    assert_eq!(reordered.z0(), &expected_z0);

    // This is intentionally a direct v2 export: no renormalization is used,
    // so the unequal physical references and the reordered S coordinates are
    // retained exactly.
    let text =
        write_touchstone_v2_0_s_full_ri_hz(&reordered).expect("v2 writer accepts references");
    assert!(text.contains("[Reference] 83 37 61\n"));
    assert!(text.contains("[Matrix Format] Full\n"));
    let reread = parse_touchstone_v2_0_s(&text).expect("v2 output parses");
    assert_eq!(reread.frequency(), source.frequency());
    assert_eq!(reread.s(), &expected_s);
    assert_eq!(reread.z0(), &expected_z0);
}
