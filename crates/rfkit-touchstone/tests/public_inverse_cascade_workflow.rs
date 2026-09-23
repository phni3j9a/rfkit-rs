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

fn assert_s_close(actual: &Network, expected: &Network, tolerance: f64) {
    assert_eq!(actual.frequency(), expected.frequency());
    assert_eq!(actual.z0(), expected.z0());
    assert_eq!(actual.s().dim(), expected.s().dim());
    for (index, (&actual, &expected)) in actual.s().iter().zip(expected.s()).enumerate() {
        let difference = (actual - expected).norm();
        assert!(
            difference <= tolerance,
            "index {index}: actual={actual:?}, expected={expected:?}, difference={difference:e}"
        );
    }
}

fn load_fixture(text: &str) -> rfkit_touchstone::Result<Network> {
    parse_touchstone_v1_0_s(text, 2)
}

fn cascade(left: &Network, dut: &Network, right: &Network) -> rfkit_core::Result<Network> {
    let left_dut = left.connect_direct_power(1, dut, 0)?;
    left_dut.connect_direct_power(1, right, 0)
}

fn remove_left_then_right(
    measured: &Network,
    left: &Network,
    right: &Network,
) -> rfkit_core::Result<Network> {
    let left_inverse = left.inverse_cascade_power()?;
    let right_inverse = right.inverse_cascade_power()?;
    let without_left = left_inverse.connect_direct_power(1, measured, 0)?;
    without_left.connect_direct_power(1, &right_inverse, 0)
}

fn remove_right_then_left(
    measured: &Network,
    left: &Network,
    right: &Network,
) -> rfkit_core::Result<Network> {
    let left_inverse = left.inverse_cascade_power()?;
    let right_inverse = right.inverse_cascade_power()?;
    let without_right = measured.connect_direct_power(1, &right_inverse, 0)?;
    left_inverse.connect_direct_power(1, &without_right, 0)
}

#[test]
fn touchstone_load_cascade_remove_both_orientations_and_write_read() {
    let left = load_fixture(LEFT).unwrap();
    let dut = load_fixture(DUT).unwrap();
    let right = load_fixture(RIGHT).unwrap();
    let left_snapshot = left.clone();
    let dut_snapshot = dut.clone();
    let right_snapshot = right.clone();

    // A two-port cascade is built explicitly as left port 1 -> DUT port 0,
    // then the resulting port 1 -> right port 0.  Direct connection keeps
    // the A-survivor then B-survivor order at every step.
    let measured = cascade(&left, &dut, &right).unwrap();
    assert_eq!(measured.nports(), 2);
    let common = Array2::from_elem((2, 2), c(50.0, 0.0));
    assert_eq!(measured.z0(), &common);

    let recovered_left_right = remove_left_then_right(&measured, &left, &right).unwrap();
    let recovered_left_right = recovered_left_right
        .renormalize_direct_power(common.clone())
        .unwrap();
    assert_s_close(&recovered_left_right, &dut, 2.0e-11);

    // The same measured cascade is also removable from the right first.  The
    // direct connection survivor ordering already returns the DUT as
    // [left,right]; no implicit port permutation is hidden in this workflow.
    let recovered_right_left = remove_right_then_left(&measured, &left, &right)
        .unwrap()
        .renormalize_direct_power(common.clone())
        .unwrap();
    assert_s_close(&recovered_right_left, &dut, 2.0e-11);

    // The recovered DUT is writer-ready only after the caller's explicit
    // renormalization/orientation choice.  A write/read round trip then keeps
    // the canonical frequency, common reference, and S data.
    let text = write_touchstone_v1_0_s_ri_hz(&recovered_left_right).unwrap();
    let reread = parse_touchstone_v1_0_s(&text, 2).unwrap();
    assert_s_close(&reread, &recovered_left_right, 2.0e-12);

    assert_eq!(left, left_snapshot);
    assert_eq!(dut, dut_snapshot);
    assert_eq!(right, right_snapshot);
}
