//! Model a real one-port impedance, convert it to S, and cross the
//! rfkit-touchstone writer/reader boundary.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frequency = Frequency::from_hz(vec![1.0e9, 2.0e9])?;
    let z = Array3::from_shape_vec(
        (2, 1, 1),
        vec![Complex64::new(100.0, 0.0), Complex64::new(100.0, 0.0)],
    )?;
    let z0 = Array2::from_elem((2, 1), Complex64::new(50.0, 0.0));
    let network = Network::from_z_power(frequency, z, z0)?;

    let text = write_touchstone_v1_0_s_ri_hz(&network)?;
    let reread = parse_touchstone_v1_0_s(&text, 1)?;
    assert_eq!(reread.frequency(), network.frequency());
    assert_eq!(reread.z0(), network.z0());
    // For z=100 ohm and a 50-ohm real reference, s=(z-z0)/(z+z0)=1/3.
    for value in reread.s().iter() {
        assert!((value.re - 1.0 / 3.0).abs() < 1.0e-14);
        assert_eq!(value.im, 0.0);
    }
    print!("{text}");
    Ok(())
}
