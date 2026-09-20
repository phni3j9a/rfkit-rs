use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{DirectRenormalizationReference, Error, Frequency, Network};
use serde_json::json;
use std::panic::{AssertUnwindSafe, catch_unwind};

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
            "index {index}: actual={actual_value:?}, expected={expected_value:?}, difference={difference:e}, bound={bound:e}"
        );
    }
}

fn identity(nport: usize) -> Array3<Complex64> {
    Array3::from_shape_fn((1, nport, nport), |(_, row, column)| {
        if row == column {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })
}

fn one_port_network(s: Complex64, z0: Complex64) -> Network {
    Network::new(
        Frequency::from_hz(vec![1.0e9]).unwrap(),
        Array3::from_elem((1, 1, 1), s),
        Array2::from_elem((1, 1), z0),
    )
    .unwrap()
}

fn asymmetric_case() -> (Frequency, Array3<Complex64>, Array2<Complex64>) {
    let frequency = Frequency::from_hz(vec![2.0e9, 1.0e9, 2.0e9]).unwrap();
    let s = Array3::from_shape_fn((3, 3, 3), |(f, row, column)| {
        let real = if row == column {
            0.08 + 0.01 * f as f64 + 0.006 * row as f64
        } else {
            0.011 * (row + 1) as f64 - 0.004 * (column + 1) as f64
        };
        let imag = if row == column {
            -0.02 + 0.004 * row as f64
        } else {
            0.003 * (f + row + 1) as f64 - 0.002 * column as f64
        };
        Complex64::new(real, imag)
    });
    let z0 = Array2::from_shape_fn((3, 3), |(f, port)| {
        Complex64::new(
            [42.0, -57.0, 71.0][port] + 1.5 * f as f64,
            [2.0, -1.5, 3.25][port] + 0.2 * f as f64,
        )
    });
    (frequency, s, z0)
}

fn matrix_vector(
    matrix: &Array3<Complex64>,
    frequency: usize,
    vector: &Array1<Complex64>,
) -> Array1<Complex64> {
    let nport = vector.len();
    Array1::from_shape_fn(nport, |row| {
        (0..nport)
            .map(|column| matrix[[frequency, row, column]] * vector[column])
            .sum()
    })
}

fn wave_pair(
    voltage: &Array1<Complex64>,
    current: &Array1<Complex64>,
    z0: &Array1<Complex64>,
) -> (Array1<Complex64>, Array1<Complex64>) {
    let a = Array1::from_shape_fn(voltage.len(), |port| {
        let f = 1.0 / (2.0 * z0[port].re.abs().sqrt());
        (voltage[port] + z0[port] * current[port]) * f
    });
    let b = Array1::from_shape_fn(voltage.len(), |port| {
        let f = 1.0 / (2.0 * z0[port].re.abs().sqrt());
        (voltage[port] - z0[port].conj() * current[port]) * f
    });
    (a, b)
}

#[test]
fn direct_renormalization_supports_ideal_open_and_complex_reference_short() {
    let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let source_z0 = Array2::from_shape_vec(
        (1, 2),
        vec![Complex64::new(50.0, 2.0), Complex64::new(-60.0, 3.0)],
    )
    .unwrap();
    let target_z0 = Array2::from_shape_vec(
        (1, 2),
        vec![Complex64::new(75.0, -4.0), Complex64::new(90.0, 6.0)],
    )
    .unwrap();
    let open = Network::new(frequency, identity(2), source_z0).unwrap();
    let transformed = open.renormalize_direct_power(target_z0.clone()).unwrap();
    assert_array3_close(transformed.s(), &identity(2), 1.0e-13, 1.0e-13);
    assert_eq!(transformed.z0(), &target_z0);
    let equal_reference = open
        .renormalize_direct_power(open.z0().clone())
        .expect("equal-reference ideal open still uses the direct validation and solve path");
    assert_array3_close(equal_reference.s(), open.s(), 1.0e-13, 1.0e-13);
    assert!(matches!(
        open.renormalize_power(target_z0),
        Err(Error::Singular { .. })
    ));

    let source = Complex64::new(50.0, 10.0);
    let target = Complex64::new(-75.0, 12.0);
    let short = one_port_network(-source.conj() / source, source);
    let transformed = short
        .renormalize_direct_power(Array2::from_elem((1, 1), target))
        .unwrap();
    assert!((transformed.s()[[0, 0, 0]] + target.conj() / target).norm() < 1.0e-13);
}

#[test]
fn direct_renormalization_round_trips_and_agrees_across_composition_paths() {
    let (frequency, s, source_z0) = asymmetric_case();
    let source = Network::new(frequency.clone(), s.clone(), source_z0.clone()).unwrap();
    let target_a = Array2::from_shape_fn((3, 3), |(f, port)| {
        Complex64::new(
            [63.0, 88.0, 54.0][port] + f as f64,
            [4.0, -2.0, 7.0][port] - 0.25 * f as f64,
        )
    });
    let target_b = Array2::from_shape_fn((3, 3), |(f, port)| {
        Complex64::new(
            [79.0, 96.0, 61.0][port] - 0.5 * f as f64,
            [-3.0, 5.0, 2.5][port] + 0.1 * f as f64,
        )
    });
    let target_c = Array2::from_shape_fn((3, 3), |(f, port)| {
        Complex64::new(
            [111.0, 47.0, 83.0][port] + 0.75 * f as f64,
            [1.5, -6.0, 4.0][port] + 0.2 * f as f64,
        )
    });

    let atob = source.renormalize_direct_power(target_a.clone()).unwrap();
    let back = atob.renormalize_direct_power(source_z0.clone()).unwrap();
    assert_array3_close(back.s(), &s, 2.0e-12, 2.0e-12);
    let atoc = source.renormalize_direct_power(target_c.clone()).unwrap();
    let via_b = atob
        .renormalize_direct_power(target_b.clone())
        .unwrap()
        .renormalize_direct_power(target_c.clone())
        .unwrap();
    assert_array3_close(via_b.s(), atoc.s(), 3.0e-12, 3.0e-12);
    assert_eq!(atoc.frequency(), &frequency);
    assert_eq!(atoc.z0(), &target_c);
    assert_eq!(source.frequency(), &frequency);
    assert_eq!(source.s(), &s);
    assert_eq!(source.z0(), &source_z0);

    // Independently check the V/I wave relation.  For every frequency the
    // source and target S matrices must describe the same physical V and I,
    // rather than merely round-tripping through another representation.
    for f in 0..3 {
        let old = source_z0.slice(ndarray::s![f, ..]).to_owned();
        let new = target_c.slice(ndarray::s![f, ..]).to_owned();
        let a_old = Array1::from_vec(vec![
            Complex64::new(0.3 + 0.02 * f as f64, -0.1),
            Complex64::new(-0.2, 0.4 + 0.03 * f as f64),
            Complex64::new(0.5, 0.1 - 0.01 * f as f64),
        ]);
        let b_old = matrix_vector(source.s(), f, &a_old);
        let current = Array1::from_shape_fn(3, |port| {
            let normalization = 1.0 / (2.0 * old[port].re.abs().sqrt());
            (a_old[port] - b_old[port]) / (normalization * Complex64::new(2.0 * old[port].re, 0.0))
        });
        let voltage = Array1::from_shape_fn(3, |port| {
            let normalization = 1.0 / (2.0 * old[port].re.abs().sqrt());
            a_old[port] / normalization - old[port] * current[port]
        });
        let (a_old_from_vi, b_old_from_vi) = wave_pair(&voltage, &current, &old);
        let (a_new, b_new) = wave_pair(&voltage, &current, &new);
        let b_from_source = matrix_vector(source.s(), f, &a_old);
        let b_from_target = matrix_vector(atoc.s(), f, &a_new);
        for port in 0..3 {
            assert!((a_old_from_vi[port] - a_old[port]).norm() < 2.0e-13);
            assert!((b_old_from_vi[port] - b_old[port]).norm() < 2.0e-13);
            assert!((b_from_source[port] - b_old[port]).norm() < 2.0e-13);
            assert!((b_from_target[port] - b_new[port]).norm() < 2.0e-13);
        }
    }
}

#[test]
fn direct_renormalization_reuses_direct_y_domain_for_floating_series() {
    let frequency = Frequency::from_hz(vec![1.0e9, 2.0e9]).unwrap();
    let y = Array3::from_shape_vec(
        (2, 2, 2),
        vec![
            Complex64::new(0.01, 0.0),
            Complex64::new(-0.01, 0.0),
            Complex64::new(-0.01, 0.0),
            Complex64::new(0.01, 0.0),
            Complex64::new(0.005, 0.0),
            Complex64::new(-0.005, 0.0),
            Complex64::new(-0.005, 0.0),
            Complex64::new(0.005, 0.0),
        ],
    )
    .unwrap();
    let source_z0 = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(43.0, 7.0),
            Complex64::new(68.0, -11.0),
            Complex64::new(47.0, 5.0),
            Complex64::new(71.0, -9.0),
        ],
    )
    .unwrap();
    let source = Network::from_y_direct_power(frequency, y.clone(), source_z0).unwrap();
    let target_z0 = Array2::from_elem((2, 2), Complex64::new(75.0, 0.0));
    let target = source.renormalize_direct_power(target_z0.clone()).unwrap();
    assert_eq!(target.z0(), &target_z0);
    assert_array3_close(&target.to_y_direct_power().unwrap(), &y, 2.0e-12, 2.0e-12);
    // The direct S generated from unequal complex references need not retain
    // an exactly binary-singular I-S matrix after the independent direct
    // solve.  The composed method's singular-stage regression is covered by
    // the exact ideal-open case above and the existing public tests.
}

#[test]
fn direct_renormalization_reports_operation_specific_validation_without_panicking() {
    let valid = one_port_network(Complex64::new(0.2, -0.1), Complex64::new(50.0, 0.0));

    assert_eq!(
        valid
            .renormalize_direct_power(Array2::from_elem((1, 2), Complex64::new(75.0, 0.0)))
            .unwrap_err(),
        Error::InvalidDirectRenormalizationZ0Shape {
            reference: DirectRenormalizationReference::Target,
            shape: vec![1, 2],
        }
    );
    assert_eq!(
        one_port_network(Complex64::new(0.2, -0.1), Complex64::new(0.0, 50.0))
            .renormalize_direct_power(Array2::from_elem((1, 1), Complex64::new(75.0, 0.0)))
            .unwrap_err(),
        Error::ZeroRealDirectRenormalizationReferenceImpedance {
            reference: DirectRenormalizationReference::Source,
            frequency: 0,
            port: 0,
        }
    );
    assert_eq!(
        valid
            .renormalize_direct_power(Array2::from_elem((1, 1), Complex64::new(0.0, -75.0)))
            .unwrap_err(),
        Error::ZeroRealDirectRenormalizationReferenceImpedance {
            reference: DirectRenormalizationReference::Target,
            frequency: 0,
            port: 0,
        }
    );
    let nonfinite_s = one_port_network(Complex64::new(f64::NAN, 0.0), Complex64::new(50.0, 0.0));
    assert_eq!(
        nonfinite_s
            .renormalize_direct_power(Array2::from_elem((1, 1), Complex64::new(75.0, 0.0)))
            .unwrap_err(),
        Error::NonFiniteDirectRenormalizationS {
            frequency: 0,
            row: 0,
            column: 0,
        }
    );
    let nonfinite_target = valid
        .renormalize_direct_power(Array2::from_elem(
            (1, 1),
            Complex64::new(f64::INFINITY, 0.0),
        ))
        .unwrap_err();
    assert_eq!(
        nonfinite_target,
        Error::NonFiniteDirectRenormalizationZ0 {
            reference: DirectRenormalizationReference::Target,
            frequency: 0,
            port: 0,
        }
    );

    let mut malformed = serde_json::to_value(&valid).unwrap();
    malformed["frequency"]["hz"] = json!([]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        malformed.renormalize_direct_power(Array2::from_elem((1, 1), Complex64::new(75.0, 0.0)))
    }));
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().unwrap_err(),
        Error::EmptyDirectRenormalizationFrequency
    );

    let mut malformed = serde_json::to_value(&valid).unwrap();
    malformed["frequency"]["hz"] = json!([1.0e9, 2.0e9]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        malformed.renormalize_direct_power(Array2::from_elem((1, 1), Complex64::new(75.0, 0.0)))
    }));
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().unwrap_err(),
        Error::DirectRenormalizationFrequencyLengthMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let mut malformed = serde_json::to_value(&valid).unwrap();
    malformed["s"]["dim"] = json!([1, 0, 0]);
    malformed["s"]["data"] = json!([]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    assert_eq!(
        malformed
            .renormalize_direct_power(Array2::zeros((1, 0)))
            .unwrap_err(),
        Error::InvalidDirectRenormalizationSShape {
            shape: vec![1, 0, 0],
        }
    );

    let mut malformed = serde_json::to_value(&valid).unwrap();
    let scalar = malformed["s"]["data"][0].clone();
    malformed["s"]["dim"] = json!([1, 1, 2]);
    malformed["s"]["data"] = json!([scalar.clone(), scalar]);
    let malformed: Network = serde_json::from_value(malformed).unwrap();
    assert_eq!(
        malformed
            .renormalize_direct_power(Array2::from_elem((1, 1), Complex64::new(75.0, 0.0)))
            .unwrap_err(),
        Error::InvalidDirectRenormalizationSShape {
            shape: vec![1, 1, 2],
        }
    );

    let nonfinite_source =
        one_port_network(Complex64::new(0.2, -0.1), Complex64::new(f64::NAN, 0.0));
    assert_eq!(
        nonfinite_source
            .renormalize_direct_power(Array2::from_elem((1, 1), Complex64::new(75.0, 0.0)))
            .unwrap_err(),
        Error::NonFiniteDirectRenormalizationZ0 {
            reference: DirectRenormalizationReference::Source,
            frequency: 0,
            port: 0,
        }
    );

    let exact_singular = one_port_network(Complex64::new(-3.0, 0.0), Complex64::new(50.0, 0.0));
    assert_eq!(
        exact_singular
            .renormalize_direct_power(Array2::from_elem((1, 1), Complex64::new(25.0, 0.0)))
            .unwrap_err(),
        Error::SingularDirectRenormalization {
            frequency: 0,
            pivot: 0,
        }
    );
    let near_singular = one_port_network(
        Complex64::new(-3.0 + 2.0_f64.powi(-20), 0.0),
        Complex64::new(50.0, 0.0),
    );
    let near = near_singular
        .renormalize_direct_power(Array2::from_elem((1, 1), Complex64::new(25.0, 0.0)))
        .unwrap();
    assert!(near.s().iter().all(|value| value.is_finite()));

    let overflow = one_port_network(Complex64::new(f64::MAX, 0.0), Complex64::new(50.0, 0.0));
    assert!(matches!(
        overflow.renormalize_direct_power(Array2::from_elem((1, 1), Complex64::new(75.0, 0.0))),
        Err(Error::NonFiniteDirectRenormalizationComputation { frequency: 0, .. })
    ));

    // Finite references can overflow while forming one of the direct D/E/C/J
    // coefficient sums. The public boundary must return structured context,
    // not panic inside the finite-only scalar divide helper.
    let extreme_source = one_port_network(Complex64::new(0.1, 0.0), Complex64::new(f64::MAX, 0.0));
    let extreme_result = catch_unwind(AssertUnwindSafe(|| {
        extreme_source.renormalize_direct_power(Array2::from_elem(
            (1, 1),
            Complex64::new(f64::MAX / 2.0, 0.0),
        ))
    }));
    assert!(extreme_result.is_ok());
    assert_eq!(
        extreme_result.unwrap().unwrap_err(),
        Error::NonFiniteDirectRenormalizationComputation {
            frequency: 0,
            row: 0,
            column: 0,
        }
    );

    let pointwise_frequency = Network::new(
        Frequency::from_hz(vec![f64::NAN]).unwrap(),
        Array3::from_elem((1, 1, 1), Complex64::new(0.1, 0.0)),
        Array2::from_elem((1, 1), Complex64::new(50.0, 0.0)),
    )
    .unwrap();
    assert!(
        pointwise_frequency
            .renormalize_direct_power(Array2::from_elem((1, 1), Complex64::new(75.0, 0.0)))
            .is_ok()
    );

    // Equal finite references remain on the ordinary solve path even when
    // the scalar 1/(2 Re(G)) itself is outside f64; coefficient factoring in
    // the kernel preserves the finite identity transformation.
    let tiny_reference = Complex64::new(f64::from_bits(1), 0.0);
    let tiny = one_port_network(Complex64::new(0.2, -0.1), tiny_reference);
    let tiny_identity = tiny
        .renormalize_direct_power(Array2::from_elem((1, 1), tiny_reference))
        .unwrap();
    assert_array3_close(tiny_identity.s(), tiny.s(), 1.0e-13, 1.0e-13);
}
