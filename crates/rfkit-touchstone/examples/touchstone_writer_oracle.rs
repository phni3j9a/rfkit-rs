use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use rfkit_touchstone::{
    TouchstoneParameter, write_touchstone_v1_0_s_ri_hz, write_touchstone_v2_0_power_full_ri_hz,
};

fn network(nports: usize, frequencies: &[f64], reference: f64) -> Network {
    let frequency = Frequency::from_hz(frequencies.to_vec()).expect("fixed frequency grid");
    let s = Array3::from_shape_fn((frequencies.len(), nports, nports), |(f, row, column)| {
        let value = (f * 100 + row * 10 + column + 1) as f64;
        Complex64::new(0.005 * value, -0.003 * value)
    });
    let z0 = Array2::from_elem((frequencies.len(), nports), Complex64::new(reference, 0.0));
    Network::new(frequency, s, z0).expect("fixed Network dimensions")
}

fn v2_network(nports: usize, frequencies: &[f64]) -> Network {
    let frequency = Frequency::from_hz(frequencies.to_vec()).expect("fixed frequency grid");
    let s = Array3::from_shape_fn((frequencies.len(), nports, nports), |(f, row, column)| {
        let value = (f * 100 + row * 10 + column + 1) as f64;
        Complex64::new(0.005 * value, -0.003 * value)
    });
    let z0 = Array2::from_shape_fn((frequencies.len(), nports), |(_, port)| {
        Complex64::new(37.0 + 11.0 * port as f64, 0.0)
    });
    Network::new(frequency, s, z0).expect("fixed v2 Network dimensions")
}

fn physical_source_network() -> Network {
    let frequency = Frequency::from_hz(vec![1.0e6, 2.0e6]).expect("fixed frequency grid");
    let z0 = Array2::from_shape_fn((2, 3), |(_, port)| {
        Complex64::new([31.0, 47.0, 73.0][port], 0.0)
    });
    let s = Array3::from_shape_fn((2, 3, 3), |(sample, row, column)| {
        Complex64::new(
            0.025 + 0.006 * sample as f64 + 0.004 * row as f64 + 0.001 * column as f64,
            -0.012 + 0.002 * sample as f64 + 0.001 * row as f64 - 0.0007 * column as f64,
        )
    });
    Network::new(frequency, s, z0).expect("well-conditioned source S construction")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let case = std::env::args().nth(1).ok_or(
        "missing oracle case; choose `two-port`, `five-port`, `v2-two-port`, or `v2-five-port`",
    )?;
    let text = match case.as_str() {
        "two-port" => write_touchstone_v1_0_s_ri_hz(&network(2, &[1.0e6, 2.0e6], 73.5))?,
        "five-port" => write_touchstone_v1_0_s_ri_hz(&network(5, &[1.0e6, 2.0e6], 88.25))?,
        "v2-two-port" => write_touchstone_v2_0_power_full_ri_hz(
            &v2_network(2, &[1.0e6, 2.0e6]),
            TouchstoneParameter::S,
        )?,
        "v2-five-port" => write_touchstone_v2_0_power_full_ri_hz(
            &v2_network(5, &[1.0e6, 2.0e6]),
            TouchstoneParameter::S,
        )?,
        "v2-z-three-port" => write_touchstone_v2_0_power_full_ri_hz(
            &physical_source_network(),
            TouchstoneParameter::Z,
        )?,
        "v2-y-three-port" => write_touchstone_v2_0_power_full_ri_hz(
            &physical_source_network(),
            TouchstoneParameter::Y,
        )?,
        _ => return Err(format!("unknown oracle case {case:?}").into()),
    };
    print!("{text}");
    Ok(())
}
