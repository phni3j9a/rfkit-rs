//! Load a Touchstone 2.0 Full-matrix input with unequal real references,
//! inspect it through the existing core API, explicitly renormalize to the
//! common reference required by the v1 writer, and read the result back.

use ndarray::Array2;
use num_complex::Complex64;
use rfkit_touchstone::{
    parse_touchstone_v1_0_s, parse_touchstone_v2_0_s, write_touchstone_v1_0_s_ri_hz,
};

const INPUT: &str = "[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 2\n[Reference] 37 83\n[Matrix Format] Full\n[Network Data]\n1 0.10 0.02 0.20 -0.03 0.30 0.04 0.40 -0.05\n2 -0.11 0.06 0.21 0.07 -0.31 -0.08 0.41 0.09\n[End]\n";

fn main() -> rfkit_touchstone::Result<()> {
    let source = parse_touchstone_v2_0_s(INPUT)?;
    let inspection = source.max_singular_value_power()?;
    assert_eq!(inspection.len(), source.frequency().len());
    assert!(inspection.iter().all(|value| value.is_finite()));

    let common_z0 = Array2::from_elem(
        (source.frequency().len(), source.nports()),
        Complex64::new(50.0, 0.0),
    );
    let writer_ready = source.renormalize_direct_power(common_z0)?;
    let text = write_touchstone_v1_0_s_ri_hz(&writer_ready)?;
    let reread = parse_touchstone_v1_0_s(&text, 2)?;
    assert_eq!(reread.frequency(), writer_ready.frequency());
    assert_eq!(reread.z0(), writer_ready.z0());
    for (actual, expected) in reread.s().iter().zip(writer_ready.s().iter()) {
        assert!((*actual - *expected).norm() <= 1.0e-14);
    }
    println!(
        "loaded unequal z0, inspected {} samples, and read back v1 text",
        inspection.len()
    );
    Ok(())
}
