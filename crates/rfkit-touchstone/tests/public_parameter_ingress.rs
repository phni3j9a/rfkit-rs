use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

#[test]
fn modelled_real_common_reference_crosses_writer_reader_boundary() {
    let frequency = Frequency::from_hz(vec![1.0e9, 2.0e9]).unwrap();
    let z = Array3::from_shape_vec(
        (2, 1, 1),
        vec![Complex64::new(100.0, 0.0), Complex64::new(100.0, 0.0)],
    )
    .unwrap();
    let z0 = Array2::from_elem((2, 1), Complex64::new(50.0, 0.0));
    let network = Network::from_z_power(frequency, z, z0).unwrap();

    let text = write_touchstone_v1_0_s_ri_hz(&network).unwrap();
    assert_eq!(
        text,
        "# Hz S RI R 50\n1000000000 0.3333333333333333 0\n2000000000 0.3333333333333333 0\n"
    );
    let reread = parse_touchstone_v1_0_s(&text, 1).unwrap();
    assert_eq!(reread.frequency(), network.frequency());
    assert_eq!(reread.z0(), network.z0());
    assert_eq!(reread.s(), network.s());
}

#[test]
fn direct_floating_series_admittance_crosses_touchstone_writer_reader_boundary() {
    let frequency = Frequency::from_hz(vec![1.0e9, 2.0e9]).unwrap();
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
    )
    .unwrap();
    let z0 = Array2::from_elem((2, 2), Complex64::new(50.0, 0.0));
    let network = Network::from_y_direct_power(frequency, y, z0).unwrap();
    let text = write_touchstone_v1_0_s_ri_hz(&network).unwrap();
    let reread = parse_touchstone_v1_0_s(&text, 2).unwrap();

    assert_eq!(reread.frequency(), network.frequency());
    assert_eq!(reread.z0(), network.z0());
    assert_eq!(reread.s(), network.s());
    let expected = 100.0 / (100.0 + 2.0 * 50.0);
    for row in 0..2 {
        for column in 0..2 {
            assert!((reread.s()[[0, row, column]].re - expected).abs() < 1.0e-14);
            assert_eq!(reread.s()[[0, row, column]].im, 0.0);
        }
    }
}
