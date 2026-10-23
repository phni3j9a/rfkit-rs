use num_complex::Complex64;
use rfkit_touchstone::{parse_touchstone_v2_0_power, write_touchstone_v2_0_s_full_ri_hz};

const INPUT: &str = "[Version] 2.0\n\
# GHz Z RI R 50\n\
[Number of Ports] 1\n\
[Number of Frequencies] 2\n\
[Reference] 25\n\
[Matrix Format] Full\n\
[Network Data]\n\
1 100 0\n\
2 110 0\n\
[End]\n";

#[test]
fn physical_z_ingress_inspects_writes_full_s_and_reads_back() {
    let source = parse_touchstone_v2_0_power(INPUT).expect("physical Z parses");
    let recovered_z = source.to_z_power().expect("existing Z inspection succeeds");
    assert!((recovered_z[[0, 0, 0]] - Complex64::new(100.0, 0.0)).norm() < 1.0e-12);
    assert!((recovered_z[[1, 0, 0]] - Complex64::new(110.0, 0.0)).norm() < 1.0e-12);

    let text = write_touchstone_v2_0_s_full_ri_hz(&source).expect("Full v2 S write succeeds");
    let reread = parse_touchstone_v2_0_power(&text).expect("written S rereads");
    assert_eq!(reread.frequency(), source.frequency());
    assert_eq!(reread.z0(), source.z0());
    assert_eq!(reread.s(), source.s());
}
