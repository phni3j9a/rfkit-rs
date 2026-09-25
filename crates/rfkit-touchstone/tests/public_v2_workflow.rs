use ndarray::Array2;
use num_complex::Complex64;
use rfkit_touchstone::{
    parse_touchstone_v1_0_s, parse_touchstone_v2_0_s_full, write_touchstone_v1_0_s_ri_hz,
};

const INPUT: &str = "[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 2\n[Reference] 37 83\n[Matrix Format] Full\n[Network Data]\n1 0.10 0.02 0.20 -0.03 0.30 0.04 0.40 -0.05\n2 -0.11 0.06 0.21 0.07 -0.31 -0.08 0.41 0.09\n[End]\n";

#[test]
fn touchstone_v2_load_inspect_renormalize_and_v1_writer_readback() {
    let source = parse_touchstone_v2_0_s_full(INPUT).expect("v2 input parses");
    assert_eq!(source.z0()[[0, 0]], Complex64::new(37.0, 0.0));
    assert_eq!(source.z0()[[0, 1]], Complex64::new(83.0, 0.0));
    let singular_values = source
        .max_singular_value_power()
        .expect("existing public inspection works");
    assert_eq!(singular_values.len(), 2);
    assert!(singular_values.iter().all(|value| value.is_finite()));

    let target = Array2::from_elem((2, 2), Complex64::new(50.0, 0.0));
    let transformed = source
        .renormalize_direct_power(target)
        .expect("explicit direct renormalization works");
    assert!(
        transformed
            .z0()
            .iter()
            .all(|value| *value == Complex64::new(50.0, 0.0))
    );

    let text = write_touchstone_v1_0_s_ri_hz(&transformed).expect("v1 writer accepts common z0");
    let reread = parse_touchstone_v1_0_s(&text, 2).expect("v1 readback parses");
    assert_eq!(reread.frequency(), transformed.frequency());
    assert_eq!(reread.z0(), transformed.z0());
    for (actual, expected) in reread.s().iter().zip(transformed.s().iter()) {
        assert!((*actual - *expected).norm() <= 1.0e-14);
    }
}
