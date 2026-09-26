use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, Network};

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn network(s: Array3<Complex64>, z0: Array2<Complex64>) -> Network {
    Network::new(Frequency::from_hz(vec![1.0e9]).unwrap(), s, z0).unwrap()
}

#[test]
fn matched_selector_accepts_zero_real_external_survivor_references() {
    let left = network(
        Array3::zeros((1, 3, 3)),
        Array2::from_shape_vec((1, 3), vec![c(0.0, 12.0), c(50.0, 0.0), c(0.0, -7.0)]).unwrap(),
    );
    let right = network(
        Array3::zeros((1, 3, 3)),
        Array2::from_shape_vec((1, 3), vec![c(0.0, 4.0), c(50.0, 0.0), c(0.0, -9.0)]).unwrap(),
    );

    let connected = left.connect_power(1, &right, 1).unwrap();
    assert_eq!(
        connected.z0().to_owned(),
        Array2::from_shape_vec(
            (1, 4),
            vec![c(0.0, 12.0), c(0.0, -7.0), c(0.0, 4.0), c(0.0, -9.0)],
        )
        .unwrap()
    );
    assert!(connected.s().iter().all(|value| *value == c(0.0, 0.0)));
}

#[test]
fn matched_selector_keeps_exact_singularity_on_matched_kernel() {
    let mut s_left = Array3::zeros((1, 2, 2));
    let mut s_right = Array3::zeros((1, 2, 2));
    s_left[[0, 0, 0]] = c(1.0, 0.0);
    s_right[[0, 0, 0]] = c(1.0, 0.0);
    let left = network(s_left, Array2::from_elem((1, 2), c(50.0, 0.0)));
    let right = network(s_right, Array2::from_elem((1, 2), c(50.0, 0.0)));

    assert_eq!(
        left.connect_power(0, &right, 0),
        Err(Error::SingularConnection { frequency: 0 })
    );
}

#[test]
fn unequal_junction_references_use_direct_physical_domain() {
    let left = network(
        Array3::zeros((1, 2, 2)),
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    );
    let right = network(
        Array3::zeros((1, 2, 2)),
        Array2::from_elem((1, 2), c(75.0, 0.0)),
    );

    let connected = left.connect_power(0, &right, 0).unwrap();
    assert_eq!(connected.nports(), 2);
    assert!(connected.s().iter().all(|value| value.is_finite()));
}

#[test]
fn inner_matched_selector_preserves_zero_real_survivor_reference() {
    let network = network(
        Array3::zeros((1, 4, 4)),
        Array2::from_shape_vec(
            (1, 4),
            vec![c(50.0, 0.0), c(0.0, 2.0), c(50.0, 0.0), c(0.0, -3.0)],
        )
        .unwrap(),
    );
    let reduced = network.inner_connect_power(0, 2).unwrap();
    assert_eq!(
        reduced.z0().to_owned(),
        Array2::from_shape_vec((1, 2), vec![c(0.0, 2.0), c(0.0, -3.0)]).unwrap()
    );
}
