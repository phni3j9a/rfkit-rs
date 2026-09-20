//! Load an asymmetric Touchstone multiport, put its physical ports in an
//! explicit order, and export the result through the supported v1.0 writer.

use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

const INPUT: &str = "# Hz S RI R 50\n\
1000000000 0.11 0.01 0.12 0.02 0.13 0.03\n\
0.21 0.01 0.22 0.02 0.23 0.03\n\
0.31 0.01 0.32 0.02 0.33 0.03\n\
2000000000 0.41 0.01 0.42 0.02 0.43 0.03\n\
0.51 0.01 0.52 0.02 0.53 0.03\n\
0.61 0.01 0.62 0.02 0.63 0.03\n";

fn main() -> rfkit_touchstone::Result<()> {
    let source = parse_touchstone_v1_0_s(INPUT, 3)?;

    // The mapping is new-port -> old-port.  Thus [2, 0, 1] puts the source's
    // physical port 2 first, followed by ports 0 and 1.
    let reordered = source.permute_ports(&[2, 0, 1])?;
    assert_eq!(
        source.s()[[0, 0, 0]],
        num_complex::Complex64::new(0.11, 0.01)
    );
    assert_eq!(
        reordered.s()[[0, 0, 0]],
        num_complex::Complex64::new(0.33, 0.03)
    );
    assert_eq!(
        reordered.s()[[0, 0, 1]],
        num_complex::Complex64::new(0.31, 0.01)
    );
    assert_eq!(
        reordered.s()[[0, 1, 0]],
        num_complex::Complex64::new(0.13, 0.03)
    );
    assert_eq!(
        reordered.z0()[[0, 0]],
        num_complex::Complex64::new(50.0, 0.0)
    );

    let text = write_touchstone_v1_0_s_ri_hz(&reordered)?;
    let reread = parse_touchstone_v1_0_s(&text, 3)?;
    assert_eq!(reread.s(), reordered.s());
    assert_eq!(reread.z0(), reordered.z0());
    print!("{text}");
    Ok(())
}
