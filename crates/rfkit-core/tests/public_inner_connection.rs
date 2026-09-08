use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Error, Frequency, Network};
use serde::Deserialize;

const INNER_CONNECT_FIXTURE_JSON: &str = include_str!(
    "../../../tools/oracle/fixtures/power_wave_inner_connect_matched_five_port_real_frequency_dependent_z0.json"
);

#[derive(Debug, Deserialize)]
struct FixtureDocument {
    data: FixtureData,
    metadata: FixtureMetadata,
}

#[derive(Debug, Deserialize)]
struct FixtureData {
    frequency_hz: Vec<f64>,
    s: Vec<Vec<Vec<ComplexValue>>>,
    s_inner_connected: Vec<Vec<Vec<ComplexValue>>>,
    z0_ohm: Vec<Vec<ComplexValue>>,
    z0_inner_connected_ohm: Vec<Vec<ComplexValue>>,
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
    k: usize,
    l: usize,
}

#[derive(Debug, Deserialize)]
struct PortOrder {
    output: Vec<usize>,
    survivors: Vec<usize>,
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

fn block_diagonal_tees(nfreq: usize) -> Array3<Complex64> {
    Array3::from_shape_fn((nfreq, 6, 6), |(_, row, column)| {
        if row / 3 != column / 3 {
            c(0.0, 0.0)
        } else if row == column {
            c(-1.0 / 3.0, 0.0)
        } else {
            c(2.0 / 3.0, 0.0)
        }
    })
}

#[test]
fn public_inner_connection_matches_analytic_block_diagonal_tees_and_two_network_connection() {
    let frequency_hz = vec![1.0e9, 1.7e9];
    let s = block_diagonal_tees(frequency_hz.len());
    let z0 = Array2::from_shape_vec(
        (2, 6),
        vec![
            c(41.0, 1.0),
            c(73.5, 0.0),
            c(83.0, -2.0),
            c(101.0, 3.0),
            c(73.5, 0.0),
            c(107.0, -1.0),
            c(44.0, 1.5),
            c(86.25, 0.0),
            c(97.0, -2.5),
            c(111.0, 3.5),
            c(86.25, 0.0),
            c(117.0, -1.5),
        ],
    )
    .unwrap();
    let source = network(frequency_hz.clone(), s.clone(), z0.clone());
    let source_frequency = source.frequency().hz().to_vec();
    let source_s = source.s().clone();
    let source_z0 = source.z0().clone();

    let reduced = source.inner_connect_matched_power(1, 4).unwrap();

    assert_eq!(reduced.frequency().hz(), frequency_hz.as_slice());
    assert_eq!(reduced.s().dim(), (2, 4, 4));
    for frequency in 0..2 {
        for row in 0..4 {
            for column in 0..4 {
                let expected = if row == column { -0.5 } else { 0.5 };
                assert_eq!(reduced.s()[[frequency, row, column]], c(expected, 0.0));
            }
        }
    }
    assert_eq!(
        reduced.z0(),
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

    let left = network(
        frequency_hz.clone(),
        s.slice(ndarray::s![.., 0..3, 0..3]).to_owned(),
        z0.slice(ndarray::s![.., 0..3]).to_owned(),
    );
    let right = network(
        frequency_hz,
        s.slice(ndarray::s![.., 3..6, 3..6]).to_owned(),
        z0.slice(ndarray::s![.., 3..6]).to_owned(),
    );
    let two_network = left.connect_matched_power(1, &right, 1).unwrap();
    assert_eq!(reduced.z0(), two_network.z0());
    assert_eq!(reduced.frequency(), two_network.frequency());
    assert_array3_close(reduced.s(), two_network.s(), 1.0e-14, 1.0e-14);

    // The operation returns owned arrays and leaves every source field intact.
    assert_ne!(reduced.s().as_ptr(), source.s().as_ptr());
    assert_ne!(reduced.z0().as_ptr(), source.z0().as_ptr());
    assert_eq!(source.frequency().hz(), source_frequency.as_slice());
    assert_eq!(source.s(), &source_s);
    assert_eq!(source.z0(), &source_z0);
}

#[test]
fn public_inner_connection_matches_recorded_fixture_and_survivor_order() {
    let fixture: FixtureDocument = serde_json::from_str(INNER_CONNECT_FIXTURE_JSON).unwrap();
    assert_eq!(
        fixture.metadata.case_id,
        "power_wave_inner_connect_matched_five_port_real_frequency_dependent_z0"
    );
    assert_eq!(fixture.metadata.operation, "inner_connect_matched");
    assert_eq!(fixture.metadata.junction_ports.k, 1);
    assert_eq!(fixture.metadata.junction_ports.l, 3);
    assert_eq!(fixture.metadata.port_order.output, vec![0, 2, 4]);
    assert_eq!(fixture.metadata.port_order.survivors, vec![0, 2, 4]);

    let source = network(
        fixture.data.frequency_hz.clone(),
        array3(&fixture.data.s),
        array2(&fixture.data.z0_ohm),
    );
    let expected_s = array3(&fixture.data.s_inner_connected);
    let expected_z0 = array2(&fixture.data.z0_inner_connected_ohm);
    let reduced = source
        .inner_connect_matched_power(
            fixture.metadata.junction_ports.k,
            fixture.metadata.junction_ports.l,
        )
        .unwrap();

    assert_eq!(
        reduced.frequency().hz(),
        fixture.data.frequency_hz.as_slice()
    );
    assert_eq!(reduced.z0(), &expected_z0);
    assert_array3_close(
        reduced.s(),
        &expected_s,
        fixture.metadata.tolerance_policy.rtol,
        fixture.metadata.tolerance_policy.atol,
    );
}

#[test]
fn public_inner_connection_supports_single_survivor_non50_junction_and_complex_external_z0() {
    let frequency_hz = vec![2.4e9];
    let mut s = Array3::zeros((1, 3, 3));
    s[[0, 1, 1]] = c(0.1, -0.02);
    s[[0, 1, 0]] = c(0.2, 0.0);
    s[[0, 2, 1]] = c(0.3, 0.0);
    let z0 =
        Array2::from_shape_vec((1, 3), vec![c(61.25, 0.0), c(41.0, -3.0), c(61.25, 0.0)]).unwrap();
    let source = network(frequency_hz.clone(), s, z0);
    let reduced = source.inner_connect_matched_power(0, 2).unwrap();

    assert_eq!(reduced.frequency().hz(), frequency_hz.as_slice());
    assert_eq!(reduced.s().dim(), (1, 1, 1));
    assert_eq!(reduced.s()[[0, 0, 0]], c(0.16, -0.02));
    assert_eq!(reduced.z0(), &Array2::from_elem((1, 1), c(41.0, -3.0)));
}

#[test]
fn public_inner_connection_preserves_irregular_frequency_and_survivor_z0_bits() {
    let frequency_hz = vec![-0.0, -1.0, -1.0, -2.0];
    let mut s = Array3::zeros((4, 3, 3));
    for frequency in 0..4 {
        s[[frequency, 1, 1]] = c(0.1 + frequency as f64 * 0.01, -0.02);
    }
    let z0 = Array2::from_shape_vec(
        (4, 3),
        vec![
            c(73.5, 0.0),
            c(0.0, -0.0),
            c(73.5, 0.0),
            c(81.0, 0.0),
            c(-0.0, 2.0),
            c(81.0, 0.0),
            c(67.0, 0.0),
            c(3.0, -4.0),
            c(67.0, 0.0),
            c(91.0, 0.0),
            c(-5.0, 6.0),
            c(91.0, 0.0),
        ],
    )
    .unwrap();
    let source = network(frequency_hz.clone(), s.clone(), z0.clone());
    let reduced = source.inner_connect_matched_power(0, 2).unwrap();

    assert_eq!(reduced.frequency().len(), frequency_hz.len());
    for (actual, expected) in reduced.frequency().hz().iter().zip(&frequency_hz) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    for frequency in 0..frequency_hz.len() {
        assert_eq!(
            reduced.z0()[[frequency, 0]].re.to_bits(),
            z0[[frequency, 1]].re.to_bits()
        );
        assert_eq!(
            reduced.z0()[[frequency, 0]].im.to_bits(),
            z0[[frequency, 1]].im.to_bits()
        );
    }
    assert_eq!(source.frequency().hz(), frequency_hz.as_slice());
    assert_eq!(source.s(), &s);
    assert_eq!(source.z0(), &z0);
}

#[test]
fn public_inner_connection_reports_port_selection_and_survivor_errors() {
    let source = zero_network(vec![1.0e9], 3, c(50.0, 0.0));
    assert_eq!(
        source.inner_connect_matched_power(3, 1).unwrap_err(),
        Error::InvalidInnerConnectionPort { port: 3, nports: 3 }
    );
    assert_eq!(
        source.inner_connect_matched_power(1, 3).unwrap_err(),
        Error::InvalidInnerConnectionPort { port: 3, nports: 3 }
    );
    assert_eq!(
        source.inner_connect_matched_power(1, 1).unwrap_err(),
        Error::IdenticalInnerConnectionPorts {
            port_a: 1,
            port_b: 1,
        }
    );

    let two_port = zero_network(vec![1.0e9], 2, c(50.0, 0.0));
    assert_eq!(
        two_port.inner_connect_matched_power(0, 1).unwrap_err(),
        Error::NoExternalInnerConnectionPorts
    );
}

#[test]
fn public_inner_connection_reports_nonfinite_source_data() {
    let source = zero_network(vec![1.0e9, 2.0e9], 3, c(50.0, 0.0));

    let mut bad_frequency = source.frequency().hz().to_vec();
    bad_frequency[1] = f64::NAN;
    let bad_frequency = network(bad_frequency, source.s().clone(), source.z0().clone());
    assert!(matches!(
        bad_frequency.inner_connect_matched_power(0, 2),
        Err(Error::NonFiniteInnerConnectionFrequency {
            index: 1,
            value,
        }) if value.is_nan()
    ));

    let mut bad_s = source.s().clone();
    bad_s[[1, 1, 0]] = c(f64::INFINITY, 0.0);
    let bad_s = network(source.frequency().hz().to_vec(), bad_s, source.z0().clone());
    assert_eq!(
        bad_s.inner_connect_matched_power(0, 2).unwrap_err(),
        Error::NonFiniteInnerConnectionS {
            frequency: 1,
            row: 1,
            column: 0,
        }
    );

    let mut bad_z0 = source.z0().clone();
    bad_z0[[1, 1]] = c(50.0, f64::NAN);
    let bad_z0 = network(source.frequency().hz().to_vec(), source.s().clone(), bad_z0);
    assert_eq!(
        bad_z0.inner_connect_matched_power(0, 2).unwrap_err(),
        Error::NonFiniteInnerConnectionZ0 {
            frequency: 1,
            port: 1,
        }
    );
}

#[test]
fn public_inner_connection_reports_both_selected_junction_ports_and_values() {
    let source = zero_network(vec![1.0e9, 1.7e9], 3, c(61.25, 0.0));

    let mut invalid_second = source.z0().clone();
    invalid_second[[0, 2]] = c(61.25, 1.0);
    let invalid_second = network(
        source.frequency().hz().to_vec(),
        source.s().clone(),
        invalid_second,
    );
    assert_eq!(
        invalid_second
            .inner_connect_matched_power(0, 2)
            .unwrap_err(),
        Error::InvalidInnerConnectionJunctionZ0 {
            frequency: 0,
            port: 2,
            value: c(61.25, 1.0),
        }
    );

    let mut mismatched = source.z0().clone();
    mismatched[[1, 2]] = c(62.0, 0.0);
    let mismatched = network(
        source.frequency().hz().to_vec(),
        source.s().clone(),
        mismatched,
    );
    assert_eq!(
        mismatched.inner_connect_matched_power(2, 0).unwrap_err(),
        Error::MismatchedInnerConnectionJunctionZ0 {
            frequency: 1,
            port_a: 2,
            z0_a: c(62.0, 0.0),
            port_b: 0,
            z0_b: c(61.25, 0.0),
        }
    );
}

#[test]
fn public_inner_connection_distinguishes_exact_singularity_from_nonzero_near_singularity() {
    let frequency = vec![1.0e9];
    let z0 = Array2::from_elem((1, 3), c(50.0, 0.0));
    let mut singular_s = Array3::zeros((1, 3, 3));
    singular_s[[0, 2, 0]] = c(1.0, 0.0);
    let singular = network(frequency.clone(), singular_s, z0.clone());
    assert_eq!(
        singular.inner_connect_matched_power(0, 2).unwrap_err(),
        Error::SingularInnerConnection { frequency: 0 }
    );

    let mut near_singular_s = Array3::zeros((1, 3, 3));
    near_singular_s[[0, 2, 0]] = c(1.0 - 2.0_f64.powi(-40), 0.0);
    near_singular_s[[0, 1, 0]] = c(1.0e-6, 0.0);
    near_singular_s[[0, 2, 1]] = c(1.0e-6, 0.0);
    let near_singular = network(frequency, near_singular_s, z0);
    let reduced = near_singular.inner_connect_matched_power(0, 2).unwrap();
    assert!(
        reduced
            .s()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert_ne!(reduced.s()[[0, 0, 0]], c(0.0, 0.0));
}

#[test]
fn public_inner_connection_maps_checked_arithmetic_overflow() {
    let frequency = vec![1.0e9];
    let mut s = Array3::zeros((1, 3, 3));
    s[[0, 1, 0]] = c(f64::MAX, 0.0);
    s[[0, 2, 1]] = c(2.0, 0.0);
    let source = network(frequency, s, Array2::from_elem((1, 3), c(50.0, 0.0)));
    assert_eq!(
        source.inner_connect_matched_power(0, 2).unwrap_err(),
        Error::NonFiniteInnerConnectionComputation {
            frequency: 0,
            row: 0,
            column: 0,
        }
    );
}
