use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use rfkit_touchstone::write_touchstone_v1_0_s_ri_hz;

fn network(nports: usize, frequencies: &[f64], reference: f64) -> Network {
    let frequency = Frequency::from_hz(frequencies.to_vec()).expect("fixed frequency grid");
    let s = Array3::from_shape_fn((frequencies.len(), nports, nports), |(f, row, column)| {
        let value = (f * 100 + row * 10 + column + 1) as f64;
        Complex64::new(0.005 * value, -0.003 * value)
    });
    let z0 = Array2::from_elem((frequencies.len(), nports), Complex64::new(reference, 0.0));
    Network::new(frequency, s, z0).expect("fixed Network dimensions")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let case = std::env::args()
        .nth(1)
        .ok_or("missing oracle case; choose `two-port` or `five-port`")?;
    let network = match case.as_str() {
        "two-port" => network(2, &[1.0e6, 2.0e6], 73.5),
        "five-port" => network(5, &[1.0e6, 2.0e6], 88.25),
        _ => return Err(format!("unknown oracle case {case:?}").into()),
    };
    print!("{}", write_touchstone_v1_0_s_ri_hz(&network)?);
    Ok(())
}
