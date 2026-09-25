//! Touchstone ingress → direct physical fixture cascade → inverse removal.
//!
//! The two removal paths use the existing public direct-connection method in
//! opposite orders.  References are made writer-compatible explicitly only
//! after the physical removal, and the recovered DUT is then written/read.

use ndarray::Array2;
use num_complex::Complex64;
use rfkit_core::Network;
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

const LEFT: &str = "# Hz S RI R 50\n\
1000000000 0.10 0.00 0.30 0.01 0.40 -0.02 0.15 0.02\n\
2000000000 0.12 0.01 0.28 0.02 0.38 -0.01 0.16 0.03\n";
const DUT: &str = "# Hz S RI R 50\n\
1000000000 0.06 -0.01 0.42 0.03 0.47 -0.02 0.08 0.01\n\
2000000000 0.07 -0.02 0.39 0.02 0.44 -0.01 0.09 0.02\n";
const RIGHT: &str = "# Hz S RI R 50\n\
1000000000 0.11 0.02 0.36 -0.01 0.33 0.04 0.13 -0.02\n\
2000000000 0.13 0.01 0.34 -0.02 0.31 0.03 0.14 -0.01\n";

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn cascade(left: &Network, dut: &Network, right: &Network) -> rfkit_core::Result<Network> {
    let left_dut = left.connect_power(1, dut, 0)?;
    left_dut.connect_power(1, right, 0)
}

fn remove_left_then_right(
    measured: &Network,
    left: &Network,
    right: &Network,
) -> rfkit_core::Result<Network> {
    let left_inverse = left.inverse_cascade_power()?;
    let right_inverse = right.inverse_cascade_power()?;
    let without_left = left_inverse.connect_power(1, measured, 0)?;
    without_left.connect_power(1, &right_inverse, 0)
}

fn remove_right_then_left(
    measured: &Network,
    left: &Network,
    right: &Network,
) -> rfkit_core::Result<Network> {
    let left_inverse = left.inverse_cascade_power()?;
    let right_inverse = right.inverse_cascade_power()?;
    let without_right = measured.connect_power(1, &right_inverse, 0)?;
    left_inverse.connect_power(1, &without_right, 0)
}

fn assert_close(actual: &Network, expected: &Network, tolerance: f64) {
    assert_eq!(actual.frequency(), expected.frequency());
    assert_eq!(actual.z0(), expected.z0());
    for (actual, expected) in actual.s().iter().zip(expected.s()) {
        assert!((*actual - *expected).norm() <= tolerance);
    }
}

fn main() -> rfkit_touchstone::Result<()> {
    let left = parse_touchstone_v1_0_s(LEFT, 2)?;
    let dut = parse_touchstone_v1_0_s(DUT, 2)?;
    let right = parse_touchstone_v1_0_s(RIGHT, 2)?;
    let measured = cascade(&left, &dut, &right)?;

    // Direct connection's survivor order is A survivors followed by B
    // survivors.  The selected ports are therefore left[1]→dut[0], then the
    // intermediate port[1]→right[0].
    let common = Array2::from_elem((2, 2), c(50.0, 0.0));
    let recovered_left_right = remove_left_then_right(&measured, &left, &right)?
        .renormalize_direct_power(common.clone())?;
    let recovered_right_left = remove_right_then_left(&measured, &left, &right)?
        .renormalize_direct_power(common.clone())?;
    assert_close(&recovered_left_right, &dut, 2.0e-11);
    assert_close(&recovered_right_left, &dut, 2.0e-11);

    let text = write_touchstone_v1_0_s_ri_hz(&recovered_left_right)?;
    let reread = parse_touchstone_v1_0_s(&text, 2)?;
    assert_close(&reread, &recovered_left_right, 2.0e-12);
    print!("{text}");
    Ok(())
}
