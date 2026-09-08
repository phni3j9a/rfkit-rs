use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{ConnectionInput, Error, Frequency, Network};
use serde::Deserialize;

const CONNECT_FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/power_wave_connect_matched_three_to_four_port_real_frequency_dependent_z0.json"
);

#[derive(Debug, Deserialize)]
struct FixtureDocument {
    data: FixtureData,
    metadata: FixtureMetadata,
}

#[derive(Debug, Deserialize)]
struct FixtureData {
    frequency_hz: Vec<f64>,
    s_a: Vec<Vec<Vec<ComplexValue>>>,
    s_b: Vec<Vec<Vec<ComplexValue>>>,
    s_connected: Vec<Vec<Vec<ComplexValue>>>,
    z0_a_ohm: Vec<Vec<ComplexValue>>,
    z0_b_ohm: Vec<Vec<ComplexValue>>,
    z0_connected_ohm: Vec<Vec<ComplexValue>>,
}

#[derive(Debug, Deserialize)]
struct ComplexValue {
    real: f64,
    imag: f64,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    case_id: String,
    operation: String,
    junction_ports: JunctionPorts,
    port_order: PortOrder,
    tolerance_policy: TolerancePolicy,
}

#[derive(Debug, Deserialize)]
struct JunctionPorts {
    a: usize,
    b: usize,
}

#[derive(Debug, Deserialize)]
struct PortOrder {
    a_survivors: Vec<usize>,
    b_survivors: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct TolerancePolicy {
    atol: f64,
    rtol: f64,
}

fn c(real: f64, imag: f64) -> Complex64 {
    Complex64::new(real, imag)
}

fn array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
    let nfreq = values.len();
    let nports = values[0].len();
    Array3::from_shape_fn((nfreq, nports, nports), |(frequency, row, column)| {
        let value = &values[frequency][row][column];
        c(value.real, value.imag)
    })
}

fn array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
    let nfreq = values.len();
    let nports = values[0].len();
    Array2::from_shape_fn((nfreq, nports), |(frequency, port)| {
        let value = &values[frequency][port];
        c(value.real, value.imag)
    })
}

fn network(frequency_hz: Vec<f64>, s: Array3<Complex64>, z0: Array2<Complex64>) -> Network {
    Network::new(Frequency::from_hz(frequency_hz).unwrap(), s, z0).unwrap()
}

fn zero_network(frequency_hz: Vec<f64>, nports: usize, z0: Complex64) -> Network {
    let nfreq = frequency_hz.len();
    network(
        frequency_hz,
        Array3::zeros((nfreq, nports, nports)),
        Array2::from_elem((nfreq, nports), z0),
    )
}

fn assert_array3_close(
    actual: &Array3<Complex64>,
    expected: &Array3<Complex64>,
    rtol: f64,
    atol: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected).enumerate() {
        let difference = (actual_value - expected_value).norm();
        let bound = atol + rtol * expected_value.norm();
        assert!(
            difference <= bound,
            "flattened index {index}: actual={actual_value:?}, expected={expected_value:?}, difference={difference:e}, bound={bound:e}"
        );
    }
}

fn ideal_tee(nfreq: usize) -> Array3<Complex64> {
    Array3::from_shape_fn((nfreq, 3, 3), |(_, row, column)| {
        if row == column {
            c(-1.0 / 3.0, 0.0)
        } else {
            c(2.0 / 3.0, 0.0)
        }
    })
}

#[test]
fn public_connection_matches_analytic_ideal_tees_and_preserves_inputs() {
    let frequency_hz = vec![1.0e9, 1.7e9];
    let s_a = ideal_tee(frequency_hz.len());
    let s_b = ideal_tee(frequency_hz.len());
    let z0_a = Array2::from_shape_vec(
        (2, 3),
        vec![
            c(41.0, 1.0),
            c(61.25, 0.0),
            c(83.0, -2.0),
            c(44.0, 1.5),
            c(77.5, 0.0),
            c(97.0, -2.5),
        ],
    )
    .unwrap();
    let z0_b = Array2::from_shape_vec(
        (2, 3),
        vec![
            c(101.0, 3.0),
            c(107.0, -1.0),
            c(61.25, 0.0),
            c(111.0, 3.5),
            c(117.0, -1.5),
            c(77.5, 0.0),
        ],
    )
    .unwrap();
    let a = network(frequency_hz.clone(), s_a.clone(), z0_a.clone());
    let b = network(frequency_hz.clone(), s_b.clone(), z0_b.clone());

    let connected = a.connect_matched_power(1, &b, 2).unwrap();

    assert_eq!(connected.frequency().hz(), frequency_hz.as_slice());
    assert_eq!(connected.s().dim(), (2, 4, 4));
    for frequency in 0..2 {
        for row in 0..4 {
            for column in 0..4 {
                let expected = if row == column { -0.5 } else { 0.5 };
                assert_eq!(connected.s()[[frequency, row, column]], c(expected, 0.0));
            }
        }
    }
    assert_eq!(
        connected.z0(),
        &Array2::from_shape_vec(
            (2, 4),
            vec![
                c(41.0, 1.0),
                c(83.0, -2.0),
                c(101.0, 3.0),
                c(107.0, -1.0),
                c(44.0, 1.5),
                c(97.0, -2.5),
                c(111.0, 3.5),
                c(117.0, -1.5),
            ],
        )
        .unwrap()
    );

    // The method returns newly allocated arrays and does not mutate either
    // borrowed input network.
    assert_ne!(connected.s().as_ptr(), a.s().as_ptr());
    assert_ne!(connected.z0().as_ptr(), a.z0().as_ptr());
    assert_eq!(a.frequency().hz(), frequency_hz.as_slice());
    assert_eq!(a.s(), &s_a);
    assert_eq!(a.z0(), &z0_a);
    assert_eq!(b.frequency().hz(), frequency_hz.as_slice());
    assert_eq!(b.s(), &s_b);
    assert_eq!(b.z0(), &z0_b);
}

#[test]
fn public_connection_matches_recorded_scikit_rf_fixture_and_order() {
    let fixture: FixtureDocument = serde_json::from_str(CONNECT_FIXTURE_JSON).unwrap();
    assert_eq!(
        fixture.metadata.case_id,
        "power_wave_connect_matched_three_to_four_port_real_frequency_dependent_z0"
    );
    assert_eq!(fixture.metadata.operation, "connect_matched");
    assert_eq!(fixture.metadata.junction_ports.a, 1);
    assert_eq!(fixture.metadata.junction_ports.b, 2);
    assert_eq!(fixture.metadata.port_order.a_survivors, vec![0, 2]);
    assert_eq!(fixture.metadata.port_order.b_survivors, vec![0, 1, 3]);

    let a = network(
        fixture.data.frequency_hz.clone(),
        array3(&fixture.data.s_a),
        array2(&fixture.data.z0_a_ohm),
    );
    let b = network(
        fixture.data.frequency_hz.clone(),
        array3(&fixture.data.s_b),
        array2(&fixture.data.z0_b_ohm),
    );
    let expected_s = array3(&fixture.data.s_connected);
    let expected_z0 = array2(&fixture.data.z0_connected_ohm);
    let connected = a
        .connect_matched_power(
            fixture.metadata.junction_ports.a,
            &b,
            fixture.metadata.junction_ports.b,
        )
        .unwrap();

    assert_eq!(
        connected.frequency().hz(),
        fixture.data.frequency_hz.as_slice()
    );
    assert_eq!(connected.z0(), &expected_z0);
    assert_array3_close(
        connected.s(),
        &expected_s,
        fixture.metadata.tolerance_policy.rtol,
        fixture.metadata.tolerance_policy.atol,
    );
}

#[test]
fn public_connection_accepts_exact_signed_zero_duplicate_descending_grid() {
    let frequency_a = vec![-0.0, 2.0, 2.0, -4.0];
    // `+0.0` is equal to A's `-0.0`, as required by the kernel's exact f64
    // comparison.  The output still copies A's signed bit pattern.
    let frequency_b = vec![0.0, 2.0, 2.0, -4.0];
    let mut z0_a = Array2::from_elem((4, 2), c(73.5, 0.0));
    let mut z0_b = Array2::from_elem((4, 2), c(73.5, 0.0));
    for frequency in 0..4 {
        z0_a[[frequency, 0]] = c(0.0, -0.0);
        z0_a[[frequency, 1]] = c(73.5, 0.0);
        z0_b[[frequency, 0]] = c(-0.0, 0.0);
        z0_b[[frequency, 1]] = c(73.5, 0.0);
    }
    let a = zero_network(frequency_a.clone(), 2, c(73.5, 0.0));
    let a = network(frequency_a.clone(), a.s().clone(), z0_a.clone());
    let b = zero_network(frequency_b.clone(), 2, c(73.5, 0.0));
    let b = network(frequency_b, b.s().clone(), z0_b);

    let connected = a.connect_matched_power(1, &b, 1).unwrap();
    assert_eq!(connected.frequency().len(), 4);
    for (actual, expected) in connected.frequency().hz().iter().zip(frequency_a.iter()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    assert_eq!(connected.z0()[[0, 0]].re.to_bits(), 0.0_f64.to_bits());
    assert_eq!(connected.z0()[[0, 0]].im.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(connected.z0()[[0, 1]].re.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(connected.z0()[[0, 1]].im.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn public_connection_supports_one_port_termination_and_complex_external_z0() {
    let frequency = vec![2.4e9];
    let mut s_a = Array3::zeros((1, 1, 1));
    s_a[[0, 0, 0]] = c(0.2, 0.0);
    let a = network(
        frequency.clone(),
        s_a,
        Array2::from_elem((1, 1), c(73.5, 0.0)),
    );

    let mut s_b = Array3::zeros((1, 2, 2));
    s_b[[0, 0, 0]] = c(0.05, -0.01);
    s_b[[0, 0, 1]] = c(0.4, 0.0);
    s_b[[0, 1, 0]] = c(0.3, 0.0);
    s_b[[0, 1, 1]] = c(0.1, 0.0);
    let b_z0 = Array2::from_shape_vec((1, 2), vec![c(41.0, -3.0), c(73.5, 0.0)]).unwrap();
    let b = network(frequency, s_b, b_z0);

    let connected = a.connect_matched_power(0, &b, 1).unwrap();
    let expected = c(0.05, -0.01)
        + c(0.4, 0.0) * c(0.2, 0.0) * c(0.3, 0.0) / (c(1.0, 0.0) - c(0.2, 0.0) * c(0.1, 0.0));
    assert_eq!(connected.s().dim(), (1, 1, 1));
    assert!((connected.s()[[0, 0, 0]] - expected).norm() <= 1.0e-15);
    assert_eq!(connected.z0(), &Array2::from_elem((1, 1), c(41.0, -3.0)));
}

#[test]
fn public_connection_reports_structured_port_grid_and_input_failures() {
    let a = zero_network(vec![1.0, 2.0], 2, c(50.0, 0.0));
    let b = zero_network(vec![1.0, 2.0], 2, c(50.0, 0.0));

    assert_eq!(
        a.connect_matched_power(2, &b, 0).unwrap_err(),
        Error::InvalidConnectionPort {
            input: ConnectionInput::A,
            port: 2,
            nports: 2,
        }
    );
    assert_eq!(
        a.connect_matched_power(0, &b, 2).unwrap_err(),
        Error::InvalidConnectionPort {
            input: ConnectionInput::B,
            port: 2,
            nports: 2,
        }
    );

    let b_short = zero_network(vec![1.0], 2, c(50.0, 0.0));
    assert_eq!(
        a.connect_matched_power(0, &b_short, 0).unwrap_err(),
        Error::ConnectionFrequencyLengthMismatch { a: 2, b: 1 }
    );
    let b_different = zero_network(vec![1.0, 2.5], 2, c(50.0, 0.0));
    assert_eq!(
        a.connect_matched_power(0, &b_different, 0).unwrap_err(),
        Error::ConnectionFrequencyMismatch {
            index: 1,
            a: 2.0,
            b: 2.5,
        }
    );

    let mut a_bad_frequency = a.frequency().hz().to_vec();
    a_bad_frequency[0] = f64::NAN;
    let a_bad_frequency = network(a_bad_frequency, a.s().clone(), a.z0().clone());
    assert!(matches!(
        a_bad_frequency.connect_matched_power(0, &b, 0),
        Err(Error::NonFiniteConnectionFrequency {
            input: ConnectionInput::A,
            index: 0,
            value,
        }) if value.is_nan()
    ));

    let mut b_bad_s = b.s().clone();
    b_bad_s[[1, 1, 1]] = c(f64::INFINITY, 0.0);
    let b_bad_s = network(b.frequency().hz().to_vec(), b_bad_s, b.z0().clone());
    assert_eq!(
        a.connect_matched_power(0, &b_bad_s, 0).unwrap_err(),
        Error::NonFiniteConnectionS {
            input: ConnectionInput::B,
            frequency: 1,
            row: 1,
            column: 1,
        }
    );

    let mut a_bad_z0 = a.z0().clone();
    a_bad_z0[[1, 1]] = c(50.0, f64::NAN);
    let a_bad_z0 = network(a.frequency().hz().to_vec(), a.s().clone(), a_bad_z0);
    assert_eq!(
        a_bad_z0.connect_matched_power(0, &b, 0).unwrap_err(),
        Error::NonFiniteConnectionZ0 {
            input: ConnectionInput::A,
            frequency: 1,
            port: 1,
        }
    );
}

#[test]
fn public_connection_reports_junction_contract_and_survivor_failures() {
    let a = zero_network(vec![1.0, 2.0], 2, c(50.0, 0.0));
    let b = zero_network(vec![1.0, 2.0], 2, c(50.0, 0.0));

    let mut a_imaginary = a.z0().clone();
    a_imaginary[[0, 0]] = c(50.0, 1.0);
    let a_imaginary = network(a.frequency().hz().to_vec(), a.s().clone(), a_imaginary);
    assert!(matches!(
        a_imaginary.connect_matched_power(0, &b, 0),
        Err(Error::InvalidConnectionJunctionZ0 {
            input: ConnectionInput::A,
            frequency: 0,
            port: 0,
            value,
        }) if value == c(50.0, 1.0)
    ));

    let mut b_negative = b.z0().clone();
    b_negative[[1, 0]] = c(-50.0, 0.0);
    let b_negative = network(b.frequency().hz().to_vec(), b.s().clone(), b_negative);
    assert!(matches!(
        a.connect_matched_power(0, &b_negative, 0),
        Err(Error::InvalidConnectionJunctionZ0 {
            input: ConnectionInput::B,
            frequency: 1,
            port: 0,
            value,
        }) if value == c(-50.0, 0.0)
    ));

    let mut b_unequal = b.z0().clone();
    b_unequal[[1, 0]] = c(51.0, 0.0);
    let b_unequal = network(b.frequency().hz().to_vec(), b.s().clone(), b_unequal);
    assert_eq!(
        a.connect_matched_power(0, &b_unequal, 0).unwrap_err(),
        Error::MismatchedConnectionJunctionZ0 {
            frequency: 1,
            a: c(50.0, 0.0),
            b: c(51.0, 0.0),
        }
    );

    let one_port_a = zero_network(vec![1.0], 1, c(50.0, 0.0));
    let one_port_b = zero_network(vec![1.0], 1, c(50.0, 0.0));
    assert_eq!(
        one_port_a
            .connect_matched_power(0, &one_port_b, 0)
            .unwrap_err(),
        Error::NoExternalConnectionPorts
    );
}

#[test]
fn public_connection_distinguishes_exact_singularity_from_nonzero_near_singularity() {
    let frequency = vec![1.0e9];
    let mut s_a = Array3::zeros((1, 2, 2));
    let mut s_b = Array3::zeros((1, 2, 2));
    s_a[[0, 0, 0]] = c(1.0, 0.0);
    s_b[[0, 0, 0]] = c(1.0, 0.0);
    let a = network(
        frequency.clone(),
        s_a.clone(),
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    );
    let b = network(
        frequency.clone(),
        s_b.clone(),
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    );
    assert_eq!(
        a.connect_matched_power(0, &b, 0).unwrap_err(),
        Error::SingularConnection { frequency: 0 }
    );

    s_a[[0, 0, 0]] = c(1.0 - 2.0_f64.powi(-40), 0.0);
    s_a[[0, 1, 0]] = c(1.0e-6, 0.0);
    s_a[[0, 0, 1]] = c(1.0e-6, 0.0);
    s_b[[0, 1, 0]] = c(1.0e-6, 0.0);
    s_b[[0, 0, 1]] = c(1.0e-6, 0.0);
    let a = network(
        frequency.clone(),
        s_a,
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    );
    let b = network(frequency, s_b, Array2::from_elem((1, 2), c(50.0, 0.0)));
    let connected = a.connect_matched_power(0, &b, 0).unwrap();
    assert!(
        connected
            .s()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert_ne!(connected.s()[[0, 0, 0]], c(0.0, 0.0));
}

#[test]
fn public_connection_maps_checked_arithmetic_overflow() {
    let frequency = vec![1.0e9];
    let mut s_a = Array3::zeros((1, 2, 2));
    let mut s_b = Array3::zeros((1, 2, 2));
    s_a[[0, 0, 0]] = c(f64::MAX, 0.0);
    s_b[[0, 0, 0]] = c(2.0, 0.0);
    let a = network(
        frequency.clone(),
        s_a,
        Array2::from_elem((1, 2), c(50.0, 0.0)),
    );
    let b = network(frequency, s_b, Array2::from_elem((1, 2), c(50.0, 0.0)));
    assert_eq!(
        a.connect_matched_power(0, &b, 0).unwrap_err(),
        Error::NonFiniteConnectionComputation {
            frequency: 0,
            row: 0,
            column: 0,
        }
    );
}

#[test]
fn connection_input_display_explains_public_to_kernel_attribution() {
    assert_eq!(ConnectionInput::A.to_string(), "A (self)");
    assert_eq!(ConnectionInput::B.to_string(), "B (other)");
}
