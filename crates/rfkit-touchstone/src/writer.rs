use num_complex::Complex64;
use rfkit_core::Network;
use std::fmt::Write as _;

use crate::{Error, Result};

struct ValidatedNetwork<'a> {
    network: &'a Network,
    nports: usize,
    reference: f64,
}

struct ValidatedV2Network<'a> {
    network: &'a Network,
    nports: usize,
}

/// Validate all data before allocating or appending any successful output.
///
/// `Network` constructors enforce the ordinary shape invariants, but serde
/// can deserialize a value without running those constructors.  The writer is
/// therefore deliberately defensive at this boundary and never indexes an
/// assumed square array before checking its dimensions.
fn validate(network: &Network) -> Result<ValidatedNetwork<'_>> {
    let frequency = network.frequency().hz();
    if frequency.is_empty() {
        return Err(Error::WriterEmptyFrequency);
    }

    let s_shape = network.s().dim();
    let nports = s_shape.1;
    if s_shape.0 != frequency.len() || nports == 0 || s_shape.1 != s_shape.2 {
        return Err(Error::WriterInvalidSShape {
            shape: vec![s_shape.0, s_shape.1, s_shape.2],
            frequency_length: frequency.len(),
        });
    }

    let z0_shape = network.z0().dim();
    if z0_shape != (frequency.len(), nports) {
        return Err(Error::WriterInvalidZ0Shape {
            shape: vec![z0_shape.0, z0_shape.1],
            frequency_length: frequency.len(),
            nports,
        });
    }

    for (index, &current) in frequency.iter().enumerate() {
        if !current.is_finite() || current < 0.0 {
            return Err(Error::WriterInvalidFrequency {
                index,
                value: current,
            });
        }
        if let Some(&previous) = index.checked_sub(1).and_then(|i| frequency.get(i))
            && current <= previous
        {
            return Err(Error::WriterFrequencyNotStrictlyIncreasing {
                index,
                previous,
                current,
            });
        }
    }

    for ((frequency_index, row, column), &value) in network.s().indexed_iter() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(Error::WriterNonFiniteS {
                frequency: frequency_index,
                row,
                column,
                value,
            });
        }
    }

    let mut reference = None;
    for ((frequency_index, port), &value) in network.z0().indexed_iter() {
        if !value.re.is_finite() || !value.im.is_finite() || value.im != 0.0 || value.re <= 0.0 {
            return Err(Error::WriterInvalidZ0 {
                frequency: frequency_index,
                port,
                value,
            });
        }
        if let Some(expected) = reference {
            if value.re != expected {
                return Err(Error::WriterMismatchedZ0 {
                    frequency: frequency_index,
                    port,
                    expected,
                    actual: value,
                });
            }
        } else {
            reference = Some(value.re);
        }
    }

    // The shape and non-empty checks above guarantee that at least one z0
    // element exists.  Keep the fallback defensive in case ndarray changes
    // its dimension/iteration behavior in a future release.
    let reference = reference.ok_or(Error::WriterInvalidZ0Shape {
        shape: vec![z0_shape.0, z0_shape.1],
        frequency_length: frequency.len(),
        nports,
    })?;

    Ok(ValidatedNetwork {
        network,
        nports,
        reference,
    })
}

fn push_number(output: &mut String, value: f64) {
    // `f64::to_string` uses the standard shortest representation that rounds
    // back to the same finite binary64 value.  The writer has already checked
    // finiteness, so it cannot emit NaN or an infinity token.
    let _ = write!(output, "{value}");
}

fn push_pair(output: &mut String, value: Complex64, first: &mut bool) {
    if !*first {
        output.push(' ');
    }
    push_number(output, value.re);
    output.push(' ');
    push_number(output, value.im);
    *first = false;
}

fn push_line(
    output: &mut String,
    frequency: Option<f64>,
    s: &ndarray::Array3<Complex64>,
    frequency_index: usize,
    row: usize,
    column_start: usize,
    column_end: usize,
) {
    let mut first = true;
    if let Some(frequency) = frequency {
        push_number(output, frequency);
        first = false;
    }
    for column in column_start..column_end {
        push_pair(output, s[[frequency_index, row, column]], &mut first);
    }
    output.push('\n');
}

fn write_small_records(validated: &ValidatedNetwork<'_>, output: &mut String) {
    let network = validated.network;
    for (frequency_index, &frequency) in network.frequency().hz().iter().enumerate() {
        push_number(output, frequency);
        let mut first = false;
        if validated.nports == 1 {
            push_pair(output, network.s()[[frequency_index, 0, 0]], &mut first);
        } else {
            // Touchstone v1.0's historical two-port physical order is
            // S11, S21, S12, S22, while Network stores row-major S11, S12,
            // S21, S22.
            for (row, column) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                push_pair(
                    output,
                    network.s()[[frequency_index, row, column]],
                    &mut first,
                );
            }
        }
        output.push('\n');
    }
}

fn write_matrix_records(validated: &ValidatedNetwork<'_>, output: &mut String) {
    let network = validated.network;
    for (frequency_index, &frequency) in network.frequency().hz().iter().enumerate() {
        for row in 0..validated.nports {
            let mut column_start = 0;
            while column_start < validated.nports {
                let column_end = column_start.saturating_add(4).min(validated.nports);
                let line_frequency = (row == 0 && column_start == 0).then_some(frequency);
                push_line(
                    output,
                    line_frequency,
                    network.s(),
                    frequency_index,
                    row,
                    column_start,
                    column_end,
                );
                column_start = column_end;
            }
        }
    }
}

pub(crate) fn write_touchstone_v1_0_s_ri_hz(network: &Network) -> Result<String> {
    let validated = validate(network)?;
    let mut output = String::new();
    output.push_str("# Hz S RI R ");
    push_number(&mut output, validated.reference);
    output.push('\n');

    if validated.nports <= 2 {
        write_small_records(&validated, &mut output);
    } else {
        write_matrix_records(&validated, &mut output);
    }

    Ok(output)
}

/// Validate the bounded v2.0 writer domain before constructing any output.
///
/// The v2 writer deliberately has a separate validator from the v1 writer:
/// v1 requires one common reference, while v2 permits unequal references as
/// long as each physical port is exactly constant over the frequency axis.
fn validate_v2(network: &Network) -> Result<ValidatedV2Network<'_>> {
    let frequency = network.frequency().hz();
    if frequency.is_empty() {
        return Err(Error::WriterEmptyFrequency);
    }

    let s_shape = network.s().dim();
    let nports = s_shape.1;
    if s_shape.0 != frequency.len() || nports == 0 || s_shape.1 != s_shape.2 {
        return Err(Error::WriterV2InvalidSShape {
            shape: vec![s_shape.0, s_shape.1, s_shape.2],
            frequency_length: frequency.len(),
        });
    }

    // These checks do not allocate.  They keep malformed serde-created
    // dimensions from reaching any derived record arithmetic or an output
    // loop whose count cannot be represented by this platform.
    let matrix_elements = nports
        .checked_mul(nports)
        .ok_or(Error::WriterV2SizeOverflow {
            frequency_length: frequency.len(),
            nports,
            quantity: crate::SizeQuantity::MatrixElements,
        })?;
    let record_values = matrix_elements
        .checked_mul(2)
        .ok_or(Error::WriterV2SizeOverflow {
            frequency_length: frequency.len(),
            nports,
            quantity: crate::SizeQuantity::RecordValues,
        })?;
    frequency
        .len()
        .checked_mul(record_values)
        .ok_or(Error::WriterV2SizeOverflow {
            frequency_length: frequency.len(),
            nports,
            quantity: crate::SizeQuantity::RecordValues,
        })?;

    let z0_shape = network.z0().dim();
    if z0_shape != (frequency.len(), nports) {
        return Err(Error::WriterV2InvalidZ0Shape {
            shape: vec![z0_shape.0, z0_shape.1],
            frequency_length: frequency.len(),
            nports,
        });
    }

    for (index, &current) in frequency.iter().enumerate() {
        if !current.is_finite() || current < 0.0 {
            return Err(Error::WriterV2InvalidFrequency {
                index,
                value: current,
            });
        }
        if let Some(&previous) = index.checked_sub(1).and_then(|i| frequency.get(i))
            && current <= previous
        {
            return Err(Error::WriterV2FrequencyNotStrictlyIncreasing {
                index,
                previous,
                current,
            });
        }
    }

    for ((frequency_index, row, column), &value) in network.s().indexed_iter() {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(Error::WriterV2NonFiniteS {
                frequency: frequency_index,
                row,
                column,
                value,
            });
        }
    }

    // Validate each port independently against its first sample.  In
    // particular, do not compare a port's reference with another port's
    // reference: unequal physical references are the reason for this v2
    // writer's existence.
    for frequency_index in 0..frequency.len() {
        for port in 0..nports {
            let value = network.z0()[[frequency_index, port]];
            if !value.re.is_finite() || !value.im.is_finite() || value.im != 0.0 || value.re <= 0.0
            {
                return Err(Error::WriterV2InvalidZ0 {
                    frequency: frequency_index,
                    port,
                    value,
                });
            }
            if frequency_index > 0 {
                let expected = network.z0()[[0, port]].re;
                if value.re != expected {
                    return Err(Error::WriterV2MismatchedZ0 {
                        frequency: frequency_index,
                        port,
                        expected,
                        actual: value,
                    });
                }
            }
        }
    }

    Ok(ValidatedV2Network { network, nports })
}

/// Serialize the bounded Touchstone 2.0 Full S/RI/Hz subset.
pub(crate) fn write_touchstone_v2_0_s_full_ri_hz(network: &Network) -> Result<String> {
    let validated = validate_v2(network)?;
    let network = validated.network;
    let frequency = network.frequency().hz();

    let mut output = String::new();
    output.push_str("[Version] 2.0\n");
    output.push_str("# Hz S RI R ");
    push_number(&mut output, network.z0()[[0, 0]].re);
    output.push('\n');

    let _ = writeln!(output, "[Number of Ports] {}", validated.nports);
    if validated.nports == 2 {
        output.push_str("[Two-Port Data Order] 12_21\n");
    }
    let _ = writeln!(output, "[Number of Frequencies] {}", frequency.len());

    output.push_str("[Reference]");
    for port in 0..validated.nports {
        output.push(' ');
        push_number(&mut output, network.z0()[[0, port]].re);
    }
    output.push('\n');
    output.push_str("[Matrix Format] Full\n");
    output.push_str("[Network Data]\n");

    // Touchstone 2.0 allows an arbitrary number of pairs per physical line.
    // Keeping one complete frequency record per line makes the emitted
    // ordering auditable and avoids v1's historical four-pair wrapping.
    for (frequency_index, &sample) in frequency.iter().enumerate() {
        push_number(&mut output, sample);
        let mut first = false;
        for row in 0..validated.nports {
            for column in 0..validated.nports {
                push_pair(
                    &mut output,
                    network.s()[[frequency_index, row, column]],
                    &mut first,
                );
            }
        }
        output.push('\n');
    }
    output.push_str("[End]\n");

    Ok(output)
}
