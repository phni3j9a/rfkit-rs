//! Read unequal-reference Touchstone 2.0 S data, permute physical ports, and
//! export/read it again without renormalizing the S coordinates.

use num_complex::Complex64;
use rfkit_touchstone::{parse_touchstone_v2_0_s_full, write_touchstone_v2_0_s_full_ri_hz};

const INPUT: &str = "[Version] 2.0\n\
# Hz S RI R 37\n\
[Number of Ports] 3\n\
[Number of Frequencies] 1\n\
[Reference] 37 61 83\n\
[Matrix Format] Full\n\
[Network Data]\n\
1000000000 0.11 0.01 0.12 0.02 0.13 0.03 0.21 0.01 0.22 0.02 0.23 0.03 0.31 0.01 0.32 0.02 0.33 0.03\n\
[End]\n";

fn main() -> rfkit_touchstone::Result<()> {
    let source = parse_touchstone_v2_0_s_full(INPUT)?;
    let reordered = source.permute_ports(&[2, 0, 1])?;

    // The mapping is new-port -> old-port.  These independent checks make
    // both S axes and the reference-port alignment observable.
    assert_eq!(reordered.s()[[0, 0, 0]], Complex64::new(0.33, 0.03));
    assert_eq!(reordered.s()[[0, 0, 1]], Complex64::new(0.31, 0.01));
    assert_eq!(reordered.s()[[0, 1, 0]], Complex64::new(0.13, 0.03));
    assert_eq!(reordered.z0()[[0, 0]], Complex64::new(83.0, 0.0));
    assert_eq!(reordered.z0()[[0, 1]], Complex64::new(37.0, 0.0));
    assert_eq!(reordered.z0()[[0, 2]], Complex64::new(61.0, 0.0));

    let text = write_touchstone_v2_0_s_full_ri_hz(&reordered)?;
    let reread = parse_touchstone_v2_0_s_full(&text)?;
    assert_eq!(reread.s(), reordered.s());
    assert_eq!(reread.z0(), reordered.z0());
    print!("{text}");
    Ok(())
}
