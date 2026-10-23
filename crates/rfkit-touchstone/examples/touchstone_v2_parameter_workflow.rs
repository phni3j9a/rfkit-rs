//! Load physical Touchstone 2.0 Z data, inspect it through the existing core
//! API, export the resulting power-wave S as Full v2 text, and read it back.
//!
//! The `to_z_power` inspection here is intentionally limited to the
//! nonsingular Z input in this example. Y input uses the same parser and the
//! direct `Network::from_y_direct_power` domain, including its zero-Y case.

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

fn main() -> rfkit_touchstone::Result<()> {
    let source = parse_touchstone_v2_0_power(INPUT)?;
    let recovered_z = source.to_z_power()?;
    assert!((recovered_z[[0, 0, 0]] - Complex64::new(100.0, 0.0)).norm() < 1.0e-12);
    assert!((recovered_z[[1, 0, 0]] - Complex64::new(110.0, 0.0)).norm() < 1.0e-12);

    let text = write_touchstone_v2_0_s_full_ri_hz(&source)?;
    let reread = parse_touchstone_v2_0_power(&text)?;
    assert_eq!(reread.frequency(), source.frequency());
    assert_eq!(reread.z0(), source.z0());
    for (actual, expected) in reread.s().iter().zip(source.s().iter()) {
        assert!((*actual - *expected).norm() < 1.0e-14);
    }
    println!(
        "loaded physical Z, recovered {} samples, and read back Full v2 S",
        recovered_z.dim().0
    );
    Ok(())
}
