//! Model a floating series admittance, convert it to S, and recover Y directly
//! after crossing the rfkit-touchstone writer/reader boundary.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frequency = Frequency::from_hz(vec![1.0e9, 2.0e9])?;
    let y = Array3::from_shape_vec(
        (2, 2, 2),
        vec![
            Complex64::new(0.01, 0.0),
            Complex64::new(-0.01, 0.0),
            Complex64::new(-0.01, 0.0),
            Complex64::new(0.01, 0.0),
            Complex64::new(0.01, 0.0),
            Complex64::new(-0.01, 0.0),
            Complex64::new(-0.01, 0.0),
            Complex64::new(0.01, 0.0),
        ],
    )?;
    let source_z0 = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(43.0, 7.0),
            Complex64::new(68.0, -11.0),
            Complex64::new(47.0, 5.0),
            Complex64::new(71.0, -9.0),
        ],
    )?;
    let network = Network::from_y_direct_power(frequency, y, source_z0)?;
    let common_z0 = Array2::from_elem((2, 2), Complex64::new(75.0, 0.0));
    let transformed = network.renormalize_direct_power(common_z0)?;

    let text = write_touchstone_v1_0_s_ri_hz(&transformed)?;
    let reread = parse_touchstone_v1_0_s(&text, 2)?;
    assert_eq!(reread.frequency(), transformed.frequency());
    assert_eq!(reread.z0(), transformed.z0());

    let expected_y = Array3::from_shape_fn((2, 2, 2), |(_, row, column)| {
        if row == column {
            Complex64::new(0.01, 0.0)
        } else {
            Complex64::new(-0.01, 0.0)
        }
    });
    let extracted_y = reread.to_y_direct_power()?;
    for (actual, expected) in extracted_y.iter().zip(expected_y.iter()) {
        assert!((actual - expected).norm() < 1.0e-14);
    }

    print!("{text}");
    Ok(())
}
