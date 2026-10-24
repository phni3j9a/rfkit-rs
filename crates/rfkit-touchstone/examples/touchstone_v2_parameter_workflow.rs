//! Load physical Touchstone 2.0 Z/Y data, explicitly reorder ports, export the
//! selected physical quantity as Full v2 text, and read it back.
use ndarray::Array3;
use num_complex::Complex64;
use rfkit_touchstone::{
    TouchstoneParameter, parse_touchstone_v2_0_power, write_touchstone_v2_0_power_full_ri_hz,
};

const Z_INPUT: &str = "[Version] 2.0\n\
# GHz Z RI R 50\n\
[Number of Ports] 3\n\
[Number of Frequencies] 1\n\
[Reference] 25 61 97\n\
[Matrix Format] Full\n\
[Network Data]\n\
1 10 1 20 2 30 3 40 4 50 5 60 6 70 7 80 8 90 9\n\
[End]\n";

const Y_INPUT: &str = "[Version] 2.0\n\
# GHz Y RI R 50\n\
[Number of Ports] 3\n\
[Number of Frequencies] 1\n\
[Reference] 25 61 97\n\
[Matrix Format] Full\n\
[Network Data]\n\
1 0.01 0.001 0.002 -0.001 0.003 0.0005 0.004 0.002 0.02 -0.001 0.005 0.0007 0.006 0.001 0.007 -0.0005 0.03 0.002\n\
[End]\n";

fn main() -> rfkit_touchstone::Result<()> {
    let source_z = parse_touchstone_v2_0_power(Z_INPUT)?;
    let reordered_z = source_z.select_ports_zero_incident(&[2, 0, 1])?;
    let recovered_z = reordered_z.to_z_power()?;
    let expected_z = Array3::from_shape_vec(
        (1, 3, 3),
        vec![
            Complex64::new(90.0, 9.0),
            Complex64::new(70.0, 7.0),
            Complex64::new(80.0, 8.0),
            Complex64::new(30.0, 3.0),
            Complex64::new(10.0, 1.0),
            Complex64::new(20.0, 2.0),
            Complex64::new(60.0, 6.0),
            Complex64::new(40.0, 4.0),
            Complex64::new(50.0, 5.0),
        ],
    )
    .expect("expected reordered Z shape");
    assert_eq!(reordered_z.z0()[[0, 0]], Complex64::new(97.0, 0.0));
    for (actual, expected) in recovered_z.iter().zip(expected_z.iter()) {
        assert!((*actual - *expected).norm() < 1.0e-12);
    }
    let z_text = write_touchstone_v2_0_power_full_ri_hz(&reordered_z, TouchstoneParameter::Z)?;
    let z_reread = parse_touchstone_v2_0_power(&z_text)?;
    for (actual, expected) in z_reread.to_z_power()?.iter().zip(recovered_z.iter()) {
        assert!((*actual - *expected).norm() < 1.0e-11);
    }

    let source_y = parse_touchstone_v2_0_power(Y_INPUT)?;
    let reordered_y = source_y.select_ports_zero_incident(&[2, 0, 1])?;
    let expected_y = Array3::from_shape_vec(
        (1, 3, 3),
        vec![
            Complex64::new(0.03, 0.002),
            Complex64::new(0.006, 0.001),
            Complex64::new(0.007, -0.0005),
            Complex64::new(0.003, 0.0005),
            Complex64::new(0.01, 0.001),
            Complex64::new(0.002, -0.001),
            Complex64::new(0.005, 0.0007),
            Complex64::new(0.004, 0.002),
            Complex64::new(0.02, -0.001),
        ],
    )
    .expect("expected reordered Y shape");
    for (actual, expected) in reordered_y
        .to_y_direct_power()?
        .iter()
        .zip(expected_y.iter())
    {
        assert!((*actual - *expected).norm() < 1.0e-12);
    }
    let y_text = write_touchstone_v2_0_power_full_ri_hz(&reordered_y, TouchstoneParameter::Y)?;
    let y_reread = parse_touchstone_v2_0_power(&y_text)?;
    for (actual, expected) in y_reread.to_y_direct_power()?.iter().zip(expected_y.iter()) {
        assert!((*actual - *expected).norm() < 1.0e-12);
    }
    println!(
        "loaded physical Z/Y, reordered {} ports, and read back Full v2 physical matrices",
        reordered_z.s().dim().1
    );
    Ok(())
}
