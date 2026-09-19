//! Construct a network from a modelled impedance matrix and use the ordinary
//! public Network operations.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frequency = Frequency::from_hz(vec![1.0e9, 2.0e9])?;
    let z = Array3::from_shape_fn((2, 2, 2), |(frequency_index, row, column)| {
        if row == column {
            Complex64::new(
                75.0 + 5.0 * frequency_index as f64 + 10.0 * row as f64,
                2.0 * (row + 1) as f64,
            )
        } else {
            Complex64::new(1.5 + row as f64, -0.5 * (column + 1) as f64)
        }
    });
    let z0 = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(50.0, 0.0),
            Complex64::new(62.5, 2.0),
            Complex64::new(53.0, 0.0),
            Complex64::new(65.0, 2.0),
        ],
    )?;

    let network = Network::from_z_power(frequency, z, z0)?;
    let admittance_siemens = network.to_y_power()?;
    let target = network.renormalize_power(Array2::from_elem((2, 2), Complex64::new(75.0, 0.0)))?;

    println!(
        "{} ports, {} frequency points; Y[0,0,0]={:?}; target z0={:?}",
        target.nports(),
        target.frequency().len(),
        admittance_siemens[[0, 0, 0]],
        target.z0()[[0, 0]],
    );
    Ok(())
}
