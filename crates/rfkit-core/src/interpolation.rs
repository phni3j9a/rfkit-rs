//! Internal Cartesian interpolation for frequency-major N-port data.
//!
//! This module deliberately contains only a raw-array kernel.  It does not
//! select a frequency grid, extrapolate, or expose a public `Network` API.
//! Every real and imaginary component is interpolated independently with a
//! linear Cartesian rule, and the reference impedance is interpolated along
//! with the S-parameter stack.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use thiserror::Error;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Failure modes for the private Cartesian interpolation kernel.
#[derive(Debug, Error, PartialEq)]
pub(crate) enum InterpolationError {
    #[error("S-parameter shape must be (nfreq, nport, nport), got {shape:?}")]
    InvalidSShape { shape: (usize, usize, usize) },

    #[error(
        "source frequency axis length does not match S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    SourceFrequencyShape { expected: usize, actual: usize },

    #[error("reference-impedance shape must be (nfreq, nport), got {shape:?}")]
    InvalidZ0Shape { shape: (usize, usize) },

    #[error("source frequency axis must contain at least two samples, got {actual}")]
    TooFewSourceSamples { actual: usize },

    #[error("target frequency axis must not be empty")]
    EmptyTarget,

    #[error("source frequency is non-finite at index {index}: {value:?}")]
    NonFiniteSourceFrequency { index: usize, value: f64 },

    #[error(
        "source frequency axis must be strictly increasing at index {index}: previous={previous:?}, current={current:?}"
    )]
    SourceFrequencyNotStrictlyIncreasing {
        index: usize,
        previous: f64,
        current: f64,
    },

    #[error("target frequency is non-finite at index {index}: {value:?}")]
    NonFiniteTargetFrequency { index: usize, value: f64 },

    #[error(
        "target frequency axis must be strictly increasing at index {index}: previous={previous:?}, current={current:?}"
    )]
    TargetFrequencyNotStrictlyIncreasing {
        index: usize,
        previous: f64,
        current: f64,
    },

    #[error(
        "target frequency is outside the source span at index {index}: value={value:?}, span=[{lower:?}, {upper:?}]"
    )]
    TargetFrequencyOutOfRange {
        index: usize,
        value: f64,
        lower: f64,
        upper: f64,
    },

    #[error("S-parameter is non-finite at frequency {frequency}, row {row}, column {column}")]
    NonFiniteS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("reference impedance is non-finite at frequency {frequency}, port {port}")]
    NonFiniteZ0 { frequency: usize, port: usize },

    #[error(
        "non-finite interpolation weight at target {target} between source indices {lower_source} and {upper_source}"
    )]
    NonFiniteWeight {
        target: usize,
        lower_source: usize,
        upper_source: usize,
    },

    #[error(
        "non-finite interpolation computation for {quantity:?} at target {target}, row {row}, column {column}"
    )]
    NonFiniteComputation {
        quantity: InterpolationQuantity,
        target: usize,
        row: usize,
        column: usize,
    },
}

/// Identifies the array involved in a failed computed-value check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum InterpolationQuantity {
    S,
    Z0,
}

/// Result of the private Cartesian interpolation operation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InterpolatedNetwork {
    /// Caller-provided target frequencies, copied without arithmetic.
    pub(crate) frequency_hz: Vec<f64>,
    /// Interpolated frequency-major S-parameter stack.
    pub(crate) s: Array3<Complex64>,
    /// Interpolated frequency-major reference impedances.
    pub(crate) z0: Array2<Complex64>,
}

#[derive(Debug, Clone, Copy)]
struct InputShape {
    nfreq: usize,
    nports: usize,
}

/// Interpolate frequency-major S and z0 arrays in Cartesian coordinates.
///
/// The source axis must contain at least two finite, strictly increasing
/// samples and agree with the first dimension of `s`.  The target axis must
/// be non-empty, finite, strictly increasing, and wholly inside the inclusive
/// source span.  No extrapolation is performed.  A target that is exactly a
/// source sample copies the complete source slice, preserving that slice
/// without interpolation arithmetic.
#[allow(dead_code)] // Internal kernel is staged for a future Network call site.
pub(crate) fn interpolate_cartesian_linear(
    source_frequency_hz: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    target_frequency_hz: &[f64],
) -> Result<InterpolatedNetwork, InterpolationError> {
    let shape = validate_inputs(source_frequency_hz, s, z0, target_frequency_hz)?;

    let mut output_s = Array3::from_elem(
        (target_frequency_hz.len(), shape.nports, shape.nports),
        ZERO,
    );
    let mut output_z0 = Array2::from_elem((target_frequency_hz.len(), shape.nports), ZERO);

    for (target_index, &target) in target_frequency_hz.iter().enumerate() {
        let source_index = match source_frequency_hz.binary_search_by(|&source| {
            // Validation has already established that both axes contain
            // only finite values. `partial_cmp` therefore cannot return
            // `None`, and its IEEE equality semantics treat signed zero
            // as the same exact frequency knot.
            source
                .partial_cmp(&target)
                .expect("validated source and target frequencies are finite")
        }) {
            Ok(index) => {
                copy_source_slice(&mut output_s, &mut output_z0, target_index, s, z0, index);
                continue;
            }
            Err(insertion) => {
                // Validation establishes that every target lies in the
                // inclusive source span.  Thus a non-knot target always has
                // one source sample on either side.
                debug_assert!(insertion > 0 && insertion < shape.nfreq);
                (insertion - 1, insertion)
            }
        };

        let (lower_index, upper_index) = source_index;
        let weight = interpolation_weight(
            source_frequency_hz[lower_index],
            source_frequency_hz[upper_index],
            target,
        )
        .ok_or(InterpolationError::NonFiniteWeight {
            target: target_index,
            lower_source: lower_index,
            upper_source: upper_index,
        })?;
        let lower_weight = 1.0 - weight;
        if !lower_weight.is_finite() {
            return Err(InterpolationError::NonFiniteWeight {
                target: target_index,
                lower_source: lower_index,
                upper_source: upper_index,
            });
        }

        for row in 0..shape.nports {
            for column in 0..shape.nports {
                let value = interpolate_complex(
                    s[[lower_index, row, column]],
                    s[[upper_index, row, column]],
                    lower_weight,
                    weight,
                )
                .ok_or(InterpolationError::NonFiniteComputation {
                    quantity: InterpolationQuantity::S,
                    target: target_index,
                    row,
                    column,
                })?;
                output_s[[target_index, row, column]] = value;
            }
        }

        for port in 0..shape.nports {
            let value = interpolate_complex(
                z0[[lower_index, port]],
                z0[[upper_index, port]],
                lower_weight,
                weight,
            )
            .ok_or(InterpolationError::NonFiniteComputation {
                quantity: InterpolationQuantity::Z0,
                target: target_index,
                row: port,
                column: 0,
            })?;
            output_z0[[target_index, port]] = value;
        }
    }

    Ok(InterpolatedNetwork {
        frequency_hz: target_frequency_hz.to_vec(),
        s: output_s,
        z0: output_z0,
    })
}

fn validate_inputs(
    source_frequency_hz: &[f64],
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    target_frequency_hz: &[f64],
) -> Result<InputShape, InterpolationError> {
    let s_shape = s.dim();
    if s_shape.1 == 0 || s_shape.1 != s_shape.2 {
        return Err(InterpolationError::InvalidSShape { shape: s_shape });
    }
    if source_frequency_hz.len() != s_shape.0 {
        return Err(InterpolationError::SourceFrequencyShape {
            expected: s_shape.0,
            actual: source_frequency_hz.len(),
        });
    }
    let z0_shape = z0.dim();
    if z0_shape != (s_shape.0, s_shape.1) {
        return Err(InterpolationError::InvalidZ0Shape { shape: z0_shape });
    }
    if source_frequency_hz.len() < 2 {
        return Err(InterpolationError::TooFewSourceSamples {
            actual: source_frequency_hz.len(),
        });
    }
    if target_frequency_hz.is_empty() {
        return Err(InterpolationError::EmptyTarget);
    }

    for (index, &value) in source_frequency_hz.iter().enumerate() {
        if !value.is_finite() {
            return Err(InterpolationError::NonFiniteSourceFrequency { index, value });
        }
        if index > 0 {
            let previous = source_frequency_hz[index - 1];
            if value <= previous {
                return Err(InterpolationError::SourceFrequencyNotStrictlyIncreasing {
                    index,
                    previous,
                    current: value,
                });
            }
        }
    }

    for (index, &value) in target_frequency_hz.iter().enumerate() {
        if !value.is_finite() {
            return Err(InterpolationError::NonFiniteTargetFrequency { index, value });
        }
        if index > 0 {
            let previous = target_frequency_hz[index - 1];
            if value <= previous {
                return Err(InterpolationError::TargetFrequencyNotStrictlyIncreasing {
                    index,
                    previous,
                    current: value,
                });
            }
        }
        if value < source_frequency_hz[0] || value > source_frequency_hz[s_shape.0 - 1] {
            return Err(InterpolationError::TargetFrequencyOutOfRange {
                index,
                value,
                lower: source_frequency_hz[0],
                upper: source_frequency_hz[s_shape.0 - 1],
            });
        }
    }

    for (index, &value) in s.indexed_iter() {
        if !is_finite(value) {
            return Err(InterpolationError::NonFiniteS {
                frequency: index.0,
                row: index.1,
                column: index.2,
            });
        }
    }
    for (index, &value) in z0.indexed_iter() {
        if !is_finite(value) {
            return Err(InterpolationError::NonFiniteZ0 {
                frequency: index.0,
                port: index.1,
            });
        }
    }

    Ok(InputShape {
        nfreq: s_shape.0,
        nports: s_shape.1,
    })
}

fn copy_source_slice(
    output_s: &mut Array3<Complex64>,
    output_z0: &mut Array2<Complex64>,
    target_index: usize,
    s: &Array3<Complex64>,
    z0: &Array2<Complex64>,
    source_index: usize,
) {
    for row in 0..s.dim().1 {
        for column in 0..s.dim().2 {
            output_s[[target_index, row, column]] = s[[source_index, row, column]];
        }
    }
    for port in 0..z0.dim().1 {
        output_z0[[target_index, port]] = z0[[source_index, port]];
    }
}

/// Compute `(target-lower)/(upper-lower)` without overflowing a cross-zero
/// frequency interval.  The caller has already established `lower < target <
/// upper`, so the returned value is a finite convex weight.
fn interpolation_weight(lower: f64, upper: f64, target: f64) -> Option<f64> {
    let weight = if lower >= 0.0 || upper <= 0.0 {
        // In a same-sign interval, subtraction cannot exceed the finite
        // binary64 range because both endpoints are on the same side of zero.
        (target - lower) / (upper - lower)
    } else {
        // For a cross-zero interval, `upper - lower` can overflow even though
        // the ratio is perfectly representable.  Scale both sides by the
        // larger magnitude before adding them.
        let negative_magnitude = -lower;
        let scale = negative_magnitude.max(upper);
        let denominator = negative_magnitude / scale + upper / scale;
        let numerator = if target < 0.0 {
            (target - lower) / scale
        } else {
            negative_magnitude / scale + target / scale
        };
        numerator / denominator
    };

    if !weight.is_finite() {
        return None;
    }
    // A correctly ordered finite interval yields a value in [0, 1].  Clamp a
    // possible one-ulp boundary overshoot from the scaled arithmetic instead
    // of turning a mathematically valid convex interpolation into needless
    // extrapolation.
    Some(weight.clamp(0.0, 1.0))
}

/// Evaluate a convex combination without forming `upper-lower`, which can
/// overflow for finite opposite-sign values.  Scaling by the largest input
/// magnitude keeps all intermediate normalized values in [-1, 1].
fn interpolate_complex(
    lower: Complex64,
    upper: Complex64,
    lower_weight: f64,
    upper_weight: f64,
) -> Option<Complex64> {
    let real = interpolate_component(lower.re, upper.re, lower_weight, upper_weight)?;
    let imag = interpolate_component(lower.im, upper.im, lower_weight, upper_weight)?;
    let value = Complex64::new(real, imag);
    is_finite(value).then_some(value)
}

fn interpolate_component(
    lower: f64,
    upper: f64,
    lower_weight: f64,
    upper_weight: f64,
) -> Option<f64> {
    if !lower.is_finite()
        || !upper.is_finite()
        || !lower_weight.is_finite()
        || !upper_weight.is_finite()
    {
        return None;
    }

    let scale = lower.abs().max(upper.abs());
    if scale == 0.0 {
        return Some(0.0);
    }

    let normalized = (lower / scale).mul_add(lower_weight, (upper / scale) * upper_weight);
    if !normalized.is_finite() {
        return None;
    }
    // A convex combination of finite values cannot mathematically exceed the
    // largest magnitude input.  Bound the result if a rounded normalized sum
    // lands just outside [-1, 1], then perform the final finite multiplication.
    let normalized = normalized.clamp(-1.0, 1.0);
    let result = normalized * scale;
    result.is_finite().then_some(result)
}

fn is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};
    use serde::Deserialize;

    const INTERPOLATION_FIXTURE_JSON: &str = include_str!(
        "../../../tools/oracle/fixtures/interpolation_cartesian_linear_three_port_complex_z0.json"
    );

    #[derive(Debug, Deserialize)]
    struct FixtureDocument {
        data: FixtureData,
        metadata: FixtureMetadata,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureData {
        s: Vec<Vec<Vec<ComplexValue>>>,
        s_input: Vec<Vec<Vec<ComplexValue>>>,
        source_frequency_hz: Vec<f64>,
        target_frequency_hz: Vec<f64>,
        z0_input_ohm: Vec<Vec<ComplexValue>>,
        z0_ohm: Vec<Vec<ComplexValue>>,
    }

    #[derive(Debug, Deserialize)]
    struct ComplexValue {
        imag: f64,
        real: f64,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureMetadata {
        basis: String,
        case_id: String,
        coords: String,
        kind: String,
        numpy_version: String,
        operation: String,
        random_seed: u64,
        schema: String,
        schema_version: u32,
        scikit_rf_version: String,
        scipy_version: String,
        shape: FixtureShape,
        tolerance_policy: FixtureTolerance,
        wave_definition: String,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureShape {
        input_s: Vec<usize>,
        input_z0: Vec<usize>,
        output_s: Vec<usize>,
        output_z0: Vec<usize>,
        source_frequency: Vec<usize>,
        target_frequency: Vec<usize>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureTolerance {
        atol: f64,
        rtol: f64,
    }

    fn complex(real: f64, imag: f64) -> Complex64 {
        Complex64::new(real, imag)
    }

    fn affine_s(frequency: f64, row: usize, column: usize) -> Complex64 {
        let row = row as f64;
        let column = column as f64;
        complex(
            0.125 * frequency + 0.75 * row - 0.5 * column,
            -0.375 * frequency + 0.25 * row + 1.25 * column,
        )
    }

    fn affine_z0(frequency: f64, port: usize) -> Complex64 {
        complex(
            37.0 + 1.75 * frequency + 4.5 * port as f64,
            2.0 - 0.625 * frequency + 0.8 * port as f64,
        )
    }

    fn fixture_array3(values: &[Vec<Vec<ComplexValue>>]) -> Array3<Complex64> {
        assert!(!values.is_empty());
        let nfreq = values.len();
        let nrows = values[0].len();
        assert!(nrows > 0);
        let ncolumns = values[0][0].len();
        assert!(ncolumns > 0);
        let flattened = values.iter().flat_map(|frequency| {
            frequency.iter().flat_map(|row| {
                row.iter()
                    .map(|value| Complex64::new(value.real, value.imag))
            })
        });
        Array3::from_shape_vec((nfreq, nrows, ncolumns), flattened.collect())
            .expect("fixture S dimensions must agree")
    }

    fn fixture_array2(values: &[Vec<ComplexValue>]) -> Array2<Complex64> {
        assert!(!values.is_empty());
        let nfreq = values.len();
        let nports = values[0].len();
        assert!(nports > 0);
        let flattened = values.iter().flat_map(|frequency| {
            frequency
                .iter()
                .map(|value| Complex64::new(value.real, value.imag))
        });
        Array2::from_shape_vec((nfreq, nports), flattened.collect())
            .expect("fixture z0 dimensions must agree")
    }

    fn assert_recorded_output(
        actual: impl IntoIterator<Item = Complex64>,
        expected: impl IntoIterator<Item = Complex64>,
        rtol: f64,
        atol: f64,
    ) {
        for (index, (actual_value, expected_value)) in actual.into_iter().zip(expected).enumerate()
        {
            assert!(
                is_finite(actual_value),
                "actual output at flattened index {index} is non-finite"
            );
            let difference = (actual_value - expected_value).norm();
            let bound = atol + rtol * expected_value.norm();
            assert!(
                difference <= bound,
                "output at flattened index {index} differs by {difference:?}, bound {bound:?}; actual {actual_value:?}, expected {expected_value:?}"
            );
        }
    }

    #[test]
    fn interpolates_irregular_affine_three_port_data_and_preserves_knots() {
        let source = [1.0, 2.5, 6.0, 10.0];
        let target = [1.0, 1.75, 2.5, 4.25, 6.0, 8.0, 10.0];
        let s = Array3::from_shape_fn((source.len(), 3, 3), |(frequency, row, column)| {
            affine_s(source[frequency], row, column)
        });
        let z0 = Array2::from_shape_fn((source.len(), 3), |(frequency, port)| {
            affine_z0(source[frequency], port)
        });

        let result = interpolate_cartesian_linear(&source, &s, &z0, &target).unwrap();

        assert_eq!(result.frequency_hz, target);
        assert_eq!(result.s.dim(), (target.len(), 3, 3));
        assert_eq!(result.z0.dim(), (target.len(), 3));
        for (target_index, &frequency) in target.iter().enumerate() {
            for row in 0..3 {
                for column in 0..3 {
                    let actual = result.s[[target_index, row, column]];
                    let expected = affine_s(frequency, row, column);
                    assert!((actual - expected).norm() <= 1.0e-12);
                }
            }
            for port in 0..3 {
                let actual = result.z0[[target_index, port]];
                let expected = affine_z0(frequency, port);
                assert!((actual - expected).norm() <= 1.0e-12);
            }
        }

        // Exact source frequencies copy complete slices, including values that
        // do not participate in any arithmetic at the corresponding target.
        for (target_index, source_index) in [(0, 0), (2, 1), (4, 2), (6, 3)] {
            assert_eq!(
                result.s.slice(ndarray::s![target_index, .., ..]),
                s.slice(ndarray::s![source_index, .., ..])
            );
            assert_eq!(
                result.z0.slice(ndarray::s![target_index, ..]),
                z0.slice(ndarray::s![source_index, ..])
            );
        }
    }

    #[test]
    fn finite_extreme_cross_zero_frequencies_and_values_do_not_overflow() {
        let source = [-f64::MAX, 0.0, f64::MAX];
        let target = [-f64::MAX / 2.0, 0.0, f64::MAX / 2.0];
        let s = Array3::from_shape_fn((source.len(), 1, 1), |(frequency, _, _)| {
            complex(source[frequency], -source[frequency])
        });
        let z0 = Array2::from_shape_fn((source.len(), 1), |(frequency, _)| {
            complex(-source[frequency], source[frequency])
        });

        let result = interpolate_cartesian_linear(&source, &s, &z0, &target).unwrap();

        assert_eq!(
            result.s[[0, 0, 0]],
            complex(-f64::MAX / 2.0, f64::MAX / 2.0)
        );
        assert_eq!(result.s[[1, 0, 0]], complex(0.0, 0.0));
        assert_eq!(
            result.s[[2, 0, 0]],
            complex(f64::MAX / 2.0, -f64::MAX / 2.0)
        );
        assert_eq!(result.z0[[0, 0]], complex(f64::MAX / 2.0, -f64::MAX / 2.0));
        assert_eq!(result.z0[[2, 0]], complex(-f64::MAX / 2.0, f64::MAX / 2.0));
    }

    #[test]
    fn matches_direct_scikit_rf_interpolation_fixture_with_recorded_tolerance() {
        let fixture: FixtureDocument =
            serde_json::from_str(INTERPOLATION_FIXTURE_JSON).expect("fixture must parse");
        let metadata = &fixture.metadata;
        assert_eq!(
            metadata.case_id,
            "interpolation_cartesian_linear_three_port_complex_z0"
        );
        assert_eq!(metadata.operation, "interpolate_s");
        assert_eq!(metadata.basis, "s");
        assert_eq!(metadata.coords, "cart");
        assert_eq!(metadata.kind, "linear");
        assert_eq!(metadata.schema, "rfkit-rs.oracle.fixture");
        assert_eq!(metadata.schema_version, 1);
        assert_eq!(metadata.numpy_version, "2.5.1");
        assert_eq!(metadata.scikit_rf_version, "2.0.1");
        assert_eq!(metadata.scipy_version, "1.18.1");
        assert_eq!(metadata.random_seed, 20_260_942);
        assert_eq!(metadata.wave_definition, "power");
        assert_eq!(metadata.shape.source_frequency, vec![5]);
        assert_eq!(metadata.shape.target_frequency, vec![6]);
        assert_eq!(metadata.shape.input_s, vec![5, 3, 3]);
        assert_eq!(metadata.shape.input_z0, vec![5, 3]);
        assert_eq!(metadata.shape.output_s, vec![6, 3, 3]);
        assert_eq!(metadata.shape.output_z0, vec![6, 3]);

        let source_s = fixture_array3(&fixture.data.s_input);
        let source_z0 = fixture_array2(&fixture.data.z0_input_ohm);
        let expected_s = fixture_array3(&fixture.data.s);
        let expected_z0 = fixture_array2(&fixture.data.z0_ohm);
        let result = interpolate_cartesian_linear(
            &fixture.data.source_frequency_hz,
            &source_s,
            &source_z0,
            &fixture.data.target_frequency_hz,
        )
        .expect("fixture interpolation must succeed");

        assert_eq!(result.frequency_hz, fixture.data.target_frequency_hz);
        assert_eq!(result.s.dim(), expected_s.dim());
        assert_eq!(result.z0.dim(), expected_z0.dim());
        assert_recorded_output(
            result.s.iter().copied(),
            expected_s.iter().copied(),
            metadata.tolerance_policy.rtol,
            metadata.tolerance_policy.atol,
        );
        assert_recorded_output(
            result.z0.iter().copied(),
            expected_z0.iter().copied(),
            metadata.tolerance_policy.rtol,
            metadata.tolerance_policy.atol,
        );
    }

    #[test]
    fn rejects_invalid_shapes_and_frequency_axes_deterministically() {
        let valid_source = [1.0, 2.0];
        let valid_target = [1.5];
        let valid_s = Array3::from_elem((2, 2, 2), complex(1.0, 0.0));
        let valid_z0 = Array2::from_elem((2, 2), complex(50.0, 0.0));

        assert_eq!(
            interpolate_cartesian_linear(
                &valid_source,
                &Array3::from_elem((2, 0, 0), ZERO),
                &Array2::zeros((2, 0)),
                &valid_target,
            )
            .unwrap_err(),
            InterpolationError::InvalidSShape { shape: (2, 0, 0) }
        );
        assert_eq!(
            interpolate_cartesian_linear(
                &valid_source,
                &Array3::from_elem((2, 2, 3), ZERO),
                &Array2::zeros((2, 2)),
                &valid_target,
            )
            .unwrap_err(),
            InterpolationError::InvalidSShape { shape: (2, 2, 3) }
        );
        assert_eq!(
            interpolate_cartesian_linear(&[1.0], &valid_s, &valid_z0, &valid_target,).unwrap_err(),
            InterpolationError::SourceFrequencyShape {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            interpolate_cartesian_linear(
                &valid_source,
                &valid_s,
                &Array2::zeros((2, 1)),
                &valid_target,
            )
            .unwrap_err(),
            InterpolationError::InvalidZ0Shape { shape: (2, 1) }
        );
        let one_source = Array3::from_elem((1, 2, 2), ZERO);
        assert_eq!(
            interpolate_cartesian_linear(&[1.0], &one_source, &Array2::zeros((1, 2)), &[1.0])
                .unwrap_err(),
            InterpolationError::TooFewSourceSamples { actual: 1 }
        );
        assert_eq!(
            interpolate_cartesian_linear(&valid_source, &valid_s, &valid_z0, &[]).unwrap_err(),
            InterpolationError::EmptyTarget
        );

        assert!(matches!(
            interpolate_cartesian_linear(&[1.0, f64::NAN], &valid_s, &valid_z0, &valid_target),
            Err(InterpolationError::NonFiniteSourceFrequency { index: 1, .. })
        ));
        assert!(matches!(
            interpolate_cartesian_linear(&valid_source, &valid_s, &valid_z0, &[f64::INFINITY]),
            Err(InterpolationError::NonFiniteTargetFrequency { index: 0, .. })
        ));
        assert!(matches!(
            interpolate_cartesian_linear(&[2.0, 1.0], &valid_s, &valid_z0, &valid_target),
            Err(InterpolationError::SourceFrequencyNotStrictlyIncreasing { index: 1, .. })
        ));
        assert!(matches!(
            interpolate_cartesian_linear(&[1.0, 1.0], &valid_s, &valid_z0, &valid_target),
            Err(InterpolationError::SourceFrequencyNotStrictlyIncreasing { index: 1, .. })
        ));
        assert!(matches!(
            interpolate_cartesian_linear(&valid_source, &valid_s, &valid_z0, &[2.0, 1.5]),
            Err(InterpolationError::TargetFrequencyNotStrictlyIncreasing { index: 1, .. })
        ));
        assert!(matches!(
            interpolate_cartesian_linear(&valid_source, &valid_s, &valid_z0, &[1.5, 1.5]),
            Err(InterpolationError::TargetFrequencyNotStrictlyIncreasing { index: 1, .. })
        ));
        assert!(matches!(
            interpolate_cartesian_linear(&valid_source, &valid_s, &valid_z0, &[0.5]),
            Err(InterpolationError::TargetFrequencyOutOfRange { index: 0, .. })
        ));
        assert!(matches!(
            interpolate_cartesian_linear(&valid_source, &valid_s, &valid_z0, &[2.5]),
            Err(InterpolationError::TargetFrequencyOutOfRange { index: 0, .. })
        ));
    }

    #[test]
    fn rejects_non_finite_data_with_source_locations() {
        let source = [1.0, 2.0];
        let target = [1.5];
        let mut s = Array3::from_elem((2, 2, 2), ZERO);
        let z0 = Array2::from_elem((2, 2), complex(50.0, 0.0));
        s[[1, 0, 1]] = complex(f64::NAN, 0.0);
        assert_eq!(
            interpolate_cartesian_linear(&source, &s, &z0, &target).unwrap_err(),
            InterpolationError::NonFiniteS {
                frequency: 1,
                row: 0,
                column: 1,
            }
        );

        let s = Array3::from_elem((2, 2, 2), ZERO);
        let mut z0 = Array2::from_elem((2, 2), complex(50.0, 0.0));
        z0[[0, 1]] = complex(50.0, f64::INFINITY);
        assert_eq!(
            interpolate_cartesian_linear(&source, &s, &z0, &target).unwrap_err(),
            InterpolationError::NonFiniteZ0 {
                frequency: 0,
                port: 1,
            }
        );
    }

    #[test]
    fn non_finite_component_guard_is_deterministic() {
        assert_eq!(
            interpolate_complex(complex(f64::INFINITY, 0.0), ZERO, 0.5, 0.5),
            None
        );
        assert_eq!(interpolate_complex(ZERO, ZERO, f64::NAN, 0.5), None);
        assert_eq!(interpolation_weight(-f64::MAX, f64::MAX, 0.0), Some(0.5));
    }
}
