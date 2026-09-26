//! Touchstone 2.0 single-ended Full/Lower/Upper S-parameter ingress.
//!
//! This module intentionally has its own document state machine.  Touchstone
//! 1.0's row/continuation rules are sufficiently different that trying to
//! widen the v1 parser would make both contracts harder to audit.  The module
//! reuses only the crate's scalar option, comment, frequency, and scalar
//! conversion helpers; it does not reuse the v1 record state machine.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};

use super::{
    DataFormat, Error, Options, Result, SizeQuantity, Token, check_frequency_order, finite_complex,
    parse_frequency, parse_number, parse_option_line, reject_recognized_extension, split_comment,
    split_lines, tokenize, validate_character_set,
};

#[derive(Clone, Copy)]
struct RawLine<'a> {
    number: usize,
    text: &'a str,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TwoPortDataOrder {
    Legacy21_12,
    Natural12_21,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MatrixFormat {
    Full,
    Lower,
    Upper,
}

struct KeywordLine<'a> {
    number: usize,
    name: String,
    args: Vec<Token<'a>>,
}

#[derive(Clone, Copy)]
struct ScalarValue {
    value: f64,
    line: usize,
    column: usize,
}

/// Parse the supported Touchstone 2.0 document subset.
pub(crate) fn parse_touchstone_v2_0_s(input: &str) -> Result<Network> {
    let lines = collect_lines(input)?;

    let mut version_seen = false;
    let mut option: Option<Options> = None;
    let mut nports: Option<usize> = None;
    let mut full_matrix_elements: Option<usize> = None;
    let mut record_elements: Option<usize> = None;
    let mut two_port_order: Option<TwoPortDataOrder> = None;
    let mut reference_seen = false;
    let mut references = Vec::new();
    let mut reference_pending = false;
    let mut nfrequencies: Option<usize> = None;
    let mut matrix_format = MatrixFormat::Full;
    let mut matrix_format_seen = false;
    let mut network_data_seen = false;
    let mut ended = false;

    let mut frequencies = Vec::new();
    let mut s_values = Vec::new();
    let mut current_frequency = None;
    let mut current_record_line = 0usize;
    let mut current_scalars = Vec::new();
    let mut current_matrix = Vec::new();

    let mut index = 0usize;
    while index < lines.len() {
        let raw = lines[index];

        if ended {
            return Err(Error::V2Structural {
                line: raw.number,
                message: "non-comment content appears after [End]".to_owned(),
            });
        }

        // Once the first option line has been accepted, all later `#` lines
        // are non-authoritative.  Check this before consuming a continued
        // [Reference] vector so that an ignored option line cannot be
        // mistaken for a reference value.
        if reference_pending && raw.text.trim_start().starts_with('#') {
            index += 1;
            continue;
        }

        if reference_pending {
            consume_reference_line(
                raw,
                nports.expect("reference requires [Number of Ports]"),
                &mut references,
                &mut reference_pending,
            )?;
            index += 1;
            continue;
        }

        if let Some(keyword) = parse_keyword_line(raw)? {
            if !version_seen {
                if keyword.name != "version" {
                    return Err(Error::V2Structural {
                        line: keyword.number,
                        message: "[Version] 2.0 must be the first non-comment directive".to_owned(),
                    });
                }
                parse_version(&keyword, &mut version_seen)?;
                index += 1;
                continue;
            }

            if keyword.name == "version" {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "duplicate [Version] directive".to_owned(),
                });
            }

            if !network_data_seen {
                if option.is_none() {
                    return Err(Error::V2Structural {
                        line: keyword.number,
                        message: "the option line must immediately follow [Version] 2.0".to_owned(),
                    });
                }
                parse_pre_network_keyword(
                    &keyword,
                    &mut option,
                    &mut nports,
                    &mut full_matrix_elements,
                    &mut record_elements,
                    &mut two_port_order,
                    &mut reference_seen,
                    &mut references,
                    &mut reference_pending,
                    &mut nfrequencies,
                    &mut matrix_format,
                    &mut matrix_format_seen,
                    &mut network_data_seen,
                )?;

                if network_data_seen {
                    // `[Network Data]` is only a marker; the next physical
                    // data line starts the first complete record.
                    if option.is_none() {
                        return Err(Error::V2Structural {
                            line: keyword.number,
                            message: "missing option line before [Network Data]".to_owned(),
                        });
                    }
                    if nfrequencies.is_none() {
                        return Err(Error::V2Structural {
                            line: keyword.number,
                            message: "missing [Number of Frequencies] before [Network Data]"
                                .to_owned(),
                        });
                    }
                    if nports == Some(2) && two_port_order.is_none() {
                        return Err(Error::V2Structural {
                            line: keyword.number,
                            message: "two-port data requires [Two-Port Data Order]".to_owned(),
                        });
                    }
                    if nports != Some(2) && two_port_order.is_some() {
                        return Err(Error::V2Structural {
                            line: keyword.number,
                            message: "[Two-Port Data Order] is only valid for two ports".to_owned(),
                        });
                    }
                }
            } else if keyword.name == "end" {
                if !keyword.args.is_empty() {
                    return Err(Error::V2Structural {
                        line: keyword.number,
                        message: "[End] does not accept arguments".to_owned(),
                    });
                }
                finish_end(
                    keyword.number,
                    nfrequencies.expect("network data requires a frequency count"),
                    record_elements.expect("network data requires a matrix size"),
                    current_frequency,
                    &current_scalars,
                    &current_matrix,
                    frequencies.len(),
                    current_record_line,
                )?;
                ended = true;
            } else {
                if !is_supported_keyword(&keyword.name) {
                    return Err(Error::V2Unsupported {
                        line: keyword.number,
                        feature: format!("keyword [{}]", keyword.name),
                    });
                }
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: format!(
                        "directive [{}] is not permitted inside network data",
                        keyword.name
                    ),
                });
            }

            index += 1;
            continue;
        }

        let trimmed = raw.text.trim_start();
        if trimmed.starts_with('#') {
            if !version_seen {
                return Err(Error::V2Structural {
                    line: raw.number,
                    message: "[Version] 2.0 must precede the option line".to_owned(),
                });
            }
            // Touchstone 2.0, like v1.0, ignores every option line after the
            // first one. In particular, do not parse a later line: the
            // standard makes even a malformed duplicate non-authoritative.
            if option.is_some() {
                index += 1;
                continue;
            }
            if network_data_seen {
                return Err(Error::V2Structural {
                    line: raw.number,
                    message: "option lines are not permitted after [Network Data]".to_owned(),
                });
            }
            if nports.is_some() {
                return Err(Error::V2Structural {
                    line: raw.number,
                    message: "the option line must precede [Number of Ports]".to_owned(),
                });
            }
            option = Some(parse_option_line(trimmed, raw.number)?);
            index += 1;
            continue;
        }

        if !version_seen {
            return Err(Error::V2Structural {
                line: raw.number,
                message: "[Version] 2.0 must be the first non-comment directive".to_owned(),
            });
        }
        if !network_data_seen {
            return Err(Error::V2Structural {
                line: raw.number,
                message: "network data appeared before [Network Data]".to_owned(),
            });
        }

        feed_network_data_line(
            raw,
            nports.expect("network data requires [Number of Ports]"),
            record_elements.expect("network data requires a matrix size"),
            nfrequencies.expect("network data requires a frequency count"),
            two_port_order.unwrap_or(TwoPortDataOrder::Natural12_21),
            matrix_format,
            option.expect("network data requires an option line"),
            &mut current_frequency,
            &mut current_record_line,
            &mut current_scalars,
            &mut current_matrix,
            &mut frequencies,
            &mut s_values,
        )?;
        index += 1;
    }

    if !version_seen {
        return Err(Error::V2Structural {
            line: 1,
            message: "missing required [Version] 2.0 directive".to_owned(),
        });
    }
    if nports.is_none() {
        return Err(Error::V2Structural {
            line: last_line_number(&lines),
            message: "missing required [Number of Ports] directive".to_owned(),
        });
    }
    if option.is_none() {
        return Err(Error::V2Structural {
            line: last_line_number(&lines),
            message: "missing required option line".to_owned(),
        });
    }
    if reference_pending {
        return Err(Error::V2Count {
            line: last_line_number(&lines),
            what: "[Reference] entries".to_owned(),
            expected: nports.expect("reference requires ports"),
            actual: references.len(),
        });
    }
    if !network_data_seen {
        return Err(Error::V2Structural {
            line: last_line_number(&lines),
            message: "missing required [Network Data] directive".to_owned(),
        });
    }
    if !ended {
        return Err(Error::V2Structural {
            line: last_line_number(&lines),
            message: "missing required [End] directive".to_owned(),
        });
    }

    let nports = nports.expect("checked above");
    let full_matrix_elements = full_matrix_elements.expect("checked above");
    let record_elements = record_elements.expect("checked by network data");
    let nfrequencies = nfrequencies.expect("checked by network data");
    let options = option.expect("checked above");
    if frequencies.len() != nfrequencies {
        return Err(Error::V2Count {
            line: last_line_number(&lines),
            what: "network frequency records".to_owned(),
            expected: nfrequencies,
            actual: frequencies.len(),
        });
    }
    if references.len() > nports {
        return Err(Error::V2Count {
            line: last_line_number(&lines),
            what: "[Reference] entries".to_owned(),
            expected: nports,
            actual: references.len(),
        });
    }
    nfrequencies
        .checked_mul(record_elements)
        .ok_or_else(|| Error::V2SizeOverflow {
            line: last_line_number(&lines),
            quantity: SizeQuantity::MatrixElements,
            detail: format!(
                "{nfrequencies} frequency records × {record_elements} compact matrix elements"
            ),
        })?;
    let z0_values_len = nfrequencies
        .checked_mul(nports)
        .ok_or_else(|| Error::V2SizeOverflow {
            line: last_line_number(&lines),
            quantity: SizeQuantity::ReferenceImpedanceElements,
            detail: format!("{nfrequencies} frequency records × {nports} ports"),
        })?;
    let s_values_len = nfrequencies
        .checked_mul(full_matrix_elements)
        .ok_or_else(|| Error::V2SizeOverflow {
            line: last_line_number(&lines),
            quantity: SizeQuantity::MatrixElements,
            detail: format!(
                "{nfrequencies} frequency records × {full_matrix_elements} expanded matrix elements"
            ),
        })?;

    let z0_profile = if references.is_empty() {
        vec![options.resistance; nports]
    } else {
        references
    };
    if z0_profile.len() != nports {
        return Err(Error::V2Count {
            line: last_line_number(&lines),
            what: "[Reference] entries".to_owned(),
            expected: nports,
            actual: z0_profile.len(),
        });
    }
    let mut z0_values = Vec::new();
    // The declared dimensions were checked only after all actual records and
    // the [End] marker were validated.  This avoids allocating an enormous
    // output for a tiny malformed document.
    for _ in 0..nfrequencies {
        for &reference in &z0_profile {
            z0_values.push(Complex64::new(reference, 0.0));
        }
    }
    debug_assert_eq!(z0_values.len(), z0_values_len);

    let frequency = Frequency::from_hz(frequencies).map_err(Error::Core)?;
    let s = Array3::from_shape_vec((nfrequencies, nports, nports), s_values).map_err(|_| {
        Error::ConstructionShape {
            quantity: "Touchstone 2.0 S-parameter array",
            shape: vec![nfrequencies, nports, nports],
        }
    })?;
    debug_assert_eq!(s_values_len, s.len());
    let z0 = Array2::from_shape_vec((nfrequencies, nports), z0_values).map_err(|_| {
        Error::ConstructionShape {
            quantity: "Touchstone 2.0 reference-impedance array",
            shape: vec![nfrequencies, nports],
        }
    })?;
    debug_assert_eq!(
        record_elements,
        if matrix_format == MatrixFormat::Full {
            full_matrix_elements
        } else {
            triangular_count(nports).expect("checked at declaration")
        }
    );
    Network::new(frequency, s, z0).map_err(Error::Core)
}

fn collect_lines<'a>(input: &'a str) -> Result<Vec<RawLine<'a>>> {
    let mut lines = Vec::new();
    for (text, number) in split_lines(input) {
        validate_character_set(text, number)?;
        let (before_comment, comment) = split_comment(text);
        if let Some(comment) = comment {
            // Keep the v1 HFSS/Ansys semantic-comment boundary for this v2
            // entrypoint.  The parser never silently changes wave/reference
            // semantics based on a vendor annotation.
            reject_recognized_extension(comment, number)?;
        }
        if !before_comment.trim().is_empty() {
            lines.push(RawLine {
                number,
                text: before_comment,
            });
        }
    }
    Ok(lines)
}

fn last_line_number(lines: &[RawLine<'_>]) -> usize {
    lines.last().map_or(1, |line| line.number)
}

fn parse_keyword_line<'a>(raw: RawLine<'a>) -> Result<Option<KeywordLine<'a>>> {
    let trimmed = raw.text.trim_start();
    if !trimmed.starts_with('[') {
        return Ok(None);
    }
    if !raw.text.starts_with('[') {
        return Err(Error::V2Structural {
            line: raw.number,
            message: "Touchstone 2.0 keywords must begin in column 1".to_owned(),
        });
    }

    let closing = raw.text.find(']').ok_or_else(|| Error::V2Structural {
        line: raw.number,
        message: "keyword is missing its closing ']'".to_owned(),
    })?;
    let keyword_text = &raw.text[1..closing];
    if keyword_text.is_empty()
        || keyword_text
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_whitespace)
        || keyword_text
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_whitespace)
    {
        return Err(Error::V2Structural {
            line: raw.number,
            message: "keyword brackets may not contain leading/trailing whitespace".to_owned(),
        });
    }
    let mut separator = false;
    for character in keyword_text.chars() {
        if character.is_ascii_alphanumeric() {
            separator = false;
        } else if matches!(character, ' ' | '-') {
            if separator {
                return Err(Error::V2Structural {
                    line: raw.number,
                    message: "keyword words must have exactly one separator".to_owned(),
                });
            }
            separator = true;
        } else {
            return Err(Error::V2Structural {
                line: raw.number,
                message: format!("invalid character {character:?} in keyword"),
            });
        }
    }
    if separator {
        return Err(Error::V2Structural {
            line: raw.number,
            message: "keyword may not end with a separator".to_owned(),
        });
    }
    if closing + 1 < raw.text.len() && !matches!(raw.text.as_bytes()[closing + 1], b' ' | b'\t') {
        return Err(Error::V2Structural {
            line: raw.number,
            message: "keyword arguments must be separated by whitespace".to_owned(),
        });
    }
    // Keyword spelling is part of the v2 grammar.  Keep the documented dash
    // in `[Two-Port Data Order]` instead of normalizing it to a space; the
    // specification permits case variation, not alternate keyword spellings.
    let name = keyword_text.to_ascii_lowercase();
    let argument_offset = closing + 1;
    let mut args = tokenize(&raw.text[argument_offset..]);
    for token in &mut args {
        token.column += argument_offset;
    }
    Ok(Some(KeywordLine {
        number: raw.number,
        name,
        args,
    }))
}

fn parse_version(keyword: &KeywordLine<'_>, seen: &mut bool) -> Result<()> {
    if *seen {
        return Err(Error::V2Structural {
            line: keyword.number,
            message: "duplicate [Version] directive".to_owned(),
        });
    }
    if keyword.args.len() != 1 {
        return Err(Error::V2Structural {
            line: keyword.number,
            message: "[Version] requires exactly one argument".to_owned(),
        });
    }
    if keyword.args[0].text != "2.0" {
        return Err(Error::V2Unsupported {
            line: keyword.number,
            feature: format!("Touchstone version {:?}", keyword.args[0].text),
        });
    }
    *seen = true;
    Ok(())
}

fn triangular_count(nports: usize) -> Option<usize> {
    if nports % 2 == 0 {
        nports
            .checked_div(2)
            .and_then(|half| half.checked_mul(nports.checked_add(1)?))
    } else {
        nports.checked_mul(nports.checked_add(1)?.checked_div(2)?)
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_pre_network_keyword(
    keyword: &KeywordLine<'_>,
    option: &mut Option<Options>,
    nports: &mut Option<usize>,
    full_matrix_elements: &mut Option<usize>,
    record_elements: &mut Option<usize>,
    two_port_order: &mut Option<TwoPortDataOrder>,
    reference_seen: &mut bool,
    references: &mut Vec<f64>,
    reference_pending: &mut bool,
    nfrequencies: &mut Option<usize>,
    matrix_format: &mut MatrixFormat,
    matrix_format_seen: &mut bool,
    network_data_seen: &mut bool,
) -> Result<()> {
    if nports.is_none() && keyword.name != "number of ports" {
        return Err(Error::V2Structural {
            line: keyword.number,
            message: "[Number of Ports] must precede all other Touchstone 2.0 keywords".to_owned(),
        });
    }
    match keyword.name.as_str() {
        "number of ports" => {
            if nports.is_some() {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "duplicate [Number of Ports] directive".to_owned(),
                });
            }
            if keyword.args.len() != 1 {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "[Number of Ports] requires exactly one positive integer".to_owned(),
                });
            }
            let ports = keyword.args[0]
                .text
                .parse::<usize>()
                .map_err(|_| Error::V2Structural {
                    line: keyword.number,
                    message: format!(
                        "[Number of Ports] argument {:?} is not a positive integer",
                        keyword.args[0].text
                    ),
                })?;
            if ports == 0 {
                return Err(Error::InvalidPortCount { nports: 0 });
            }
            let full_elements = ports
                .checked_mul(ports)
                .ok_or_else(|| Error::V2SizeOverflow {
                    line: keyword.number,
                    quantity: SizeQuantity::MatrixElements,
                    detail: format!("{ports} ports × {ports} ports"),
                })?;
            let compact_elements =
                triangular_count(ports).ok_or_else(|| Error::V2SizeOverflow {
                    line: keyword.number,
                    quantity: SizeQuantity::MatrixElements,
                    detail: format!("{ports} ports × ({ports} + 1) / 2 compact elements"),
                })?;
            full_elements
                .checked_mul(2)
                .ok_or_else(|| Error::V2SizeOverflow {
                    line: keyword.number,
                    quantity: SizeQuantity::RecordValues,
                    detail: format!("{full_elements} expanded complex matrix elements × 2 scalars"),
                })?;
            compact_elements
                .checked_mul(2)
                .ok_or_else(|| Error::V2SizeOverflow {
                    line: keyword.number,
                    quantity: SizeQuantity::RecordValues,
                    detail: format!(
                        "{compact_elements} compact complex matrix elements × 2 scalars"
                    ),
                })?;
            *nports = Some(ports);
            *full_matrix_elements = Some(full_elements);
            *record_elements = Some(full_elements);
        }
        "two-port data order" => {
            if two_port_order.is_some() {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "duplicate [Two-Port Data Order] directive".to_owned(),
                });
            }
            if nports != &Some(2) {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "[Two-Port Data Order] is only permitted for two ports".to_owned(),
                });
            }
            if keyword.args.len() != 1 {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "[Two-Port Data Order] requires 12_21 or 21_12".to_owned(),
                });
            }
            *two_port_order = Some(match keyword.args[0].text.to_ascii_lowercase().as_str() {
                "21_12" => TwoPortDataOrder::Legacy21_12,
                "12_21" => TwoPortDataOrder::Natural12_21,
                _ => {
                    return Err(Error::V2Structural {
                        line: keyword.number,
                        message: format!(
                            "unsupported [Two-Port Data Order] argument {:?}",
                            keyword.args[0].text
                        ),
                    });
                }
            });
        }
        "reference" => {
            if *reference_seen {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "duplicate [Reference] directive".to_owned(),
                });
            }
            let ports = nports.ok_or_else(|| Error::V2Structural {
                line: keyword.number,
                message: "[Reference] requires [Number of Ports] first".to_owned(),
            })?;
            *reference_seen = true;
            consume_reference_tokens(
                keyword.number,
                &keyword.args,
                ports,
                references,
                reference_pending,
            )?;
        }
        "number of frequencies" => {
            if nfrequencies.is_some() {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "duplicate [Number of Frequencies] directive".to_owned(),
                });
            }
            if keyword.args.len() != 1 {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "[Number of Frequencies] requires exactly one positive integer"
                        .to_owned(),
                });
            }
            let count = keyword.args[0]
                .text
                .parse::<usize>()
                .map_err(|_| Error::V2Structural {
                    line: keyword.number,
                    message: format!(
                        "[Number of Frequencies] argument {:?} is not a positive integer",
                        keyword.args[0].text
                    ),
                })?;
            if count == 0 {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "[Number of Frequencies] must be greater than zero".to_owned(),
                });
            }
            *nfrequencies = Some(count);
        }
        "matrix format" => {
            if *matrix_format_seen {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "duplicate [Matrix Format] directive".to_owned(),
                });
            }
            if keyword.args.len() != 1 {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "[Matrix Format] requires Full, Lower, or Upper".to_owned(),
                });
            }
            let format = match keyword.args[0].text.to_ascii_lowercase().as_str() {
                "full" => MatrixFormat::Full,
                "lower" => MatrixFormat::Lower,
                "upper" => MatrixFormat::Upper,
                _ => {
                    return Err(Error::V2Structural {
                        line: keyword.number,
                        message: format!(
                            "unsupported [Matrix Format] argument {:?}",
                            keyword.args[0].text
                        ),
                    });
                }
            };
            let ports = nports.expect("matrix format requires ports");
            let elements = if format == MatrixFormat::Full {
                *full_matrix_elements
                    .as_ref()
                    .expect("number of ports computes full matrix size")
            } else {
                triangular_count(ports).ok_or_else(|| Error::V2SizeOverflow {
                    line: keyword.number,
                    quantity: SizeQuantity::MatrixElements,
                    detail: format!("{ports} ports × ({ports} + 1) / 2 compact elements"),
                })?
            };
            elements
                .checked_mul(2)
                .ok_or_else(|| Error::V2SizeOverflow {
                    line: keyword.number,
                    quantity: SizeQuantity::RecordValues,
                    detail: format!("{elements} complex matrix elements × 2 scalars"),
                })?;
            *matrix_format = format;
            *record_elements = Some(elements);
            *matrix_format_seen = true;
        }
        "network data" => {
            if *network_data_seen {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "duplicate [Network Data] directive".to_owned(),
                });
            }
            if !keyword.args.is_empty() {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "[Network Data] does not accept arguments".to_owned(),
                });
            }
            if nports.is_none() {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "[Network Data] requires [Number of Ports] first".to_owned(),
                });
            }
            if nfrequencies.is_none() {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "[Network Data] requires [Number of Frequencies] first".to_owned(),
                });
            }
            if option.is_none() {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "[Network Data] requires an option line first".to_owned(),
                });
            }
            if nports == &Some(2) && two_port_order.is_none() {
                return Err(Error::V2Structural {
                    line: keyword.number,
                    message: "two-port data requires [Two-Port Data Order]".to_owned(),
                });
            }
            *network_data_seen = true;
        }
        "end" => {
            return Err(Error::V2Structural {
                line: keyword.number,
                message: "[End] must follow [Network Data]".to_owned(),
            });
        }
        unsupported => {
            return Err(Error::V2Unsupported {
                line: keyword.number,
                feature: format!("keyword [{unsupported}]"),
            });
        }
    }
    Ok(())
}

fn consume_reference_tokens(
    line: usize,
    tokens: &[Token<'_>],
    nports: usize,
    references: &mut Vec<f64>,
    pending: &mut bool,
) -> Result<()> {
    for token in tokens {
        if references.len() >= nports {
            return Err(Error::V2Count {
                line,
                what: "[Reference] entries".to_owned(),
                expected: nports,
                actual: references.len().saturating_add(1),
            });
        }
        let value = parse_number(*token, line)?;
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::InvalidReferenceResistance { line, value });
        }
        references.push(value);
    }
    *pending = references.len() < nports;
    Ok(())
}

fn consume_reference_line(
    raw: RawLine<'_>,
    nports: usize,
    references: &mut Vec<f64>,
    pending: &mut bool,
) -> Result<()> {
    let trimmed = raw.text.trim_start();
    if trimmed.starts_with('[') || trimmed.starts_with('#') {
        return Err(Error::V2Count {
            line: raw.number,
            what: "[Reference] entries".to_owned(),
            expected: nports,
            actual: references.len(),
        });
    }
    let tokens = tokenize(raw.text);
    consume_reference_tokens(raw.number, &tokens, nports, references, pending)
}

#[allow(clippy::too_many_arguments)]
fn feed_network_data_line(
    raw: RawLine<'_>,
    nports: usize,
    record_elements: usize,
    nfrequencies: usize,
    two_port_order: TwoPortDataOrder,
    matrix_format: MatrixFormat,
    options: Options,
    current_frequency: &mut Option<f64>,
    current_record_line: &mut usize,
    current_scalars: &mut Vec<ScalarValue>,
    current_matrix: &mut Vec<Complex64>,
    frequencies: &mut Vec<f64>,
    s_values: &mut Vec<Complex64>,
) -> Result<()> {
    let tokens = tokenize(raw.text);
    if tokens.is_empty() {
        return Ok(());
    }

    if current_frequency.is_none() {
        if raw
            .text
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_whitespace)
        {
            return Err(Error::V2Structural {
                line: raw.number,
                message: "each frequency record must begin in column 1".to_owned(),
            });
        }
        if frequencies.len() >= nfrequencies {
            return Err(Error::V2Count {
                line: raw.number,
                what: "network frequency records".to_owned(),
                expected: nfrequencies,
                actual: frequencies.len().saturating_add(1),
            });
        }
        let frequency = parse_frequency(tokens[0], raw.number, options.unit)?;
        check_frequency_order(frequencies, frequency, raw.number)?;
        *current_frequency = Some(frequency);
        *current_record_line = raw.number;
        decode_network_scalars(
            &tokens[1..],
            raw.number,
            options.format,
            record_elements,
            current_scalars,
            current_matrix,
        )?;
    } else {
        decode_network_scalars(
            &tokens,
            raw.number,
            options.format,
            record_elements,
            current_scalars,
            current_matrix,
        )?;
    }

    if current_matrix.len() == record_elements {
        if !current_scalars.is_empty() {
            return Err(Error::V2Structural {
                line: raw.number,
                message:
                    "internal record state retained a scalar after completing the matrix record"
                        .to_owned(),
            });
        }
        let frequency = current_frequency
            .take()
            .ok_or_else(|| Error::V2Structural {
                line: raw.number,
                message: "internal record state lost its frequency".to_owned(),
            })?;
        let matrix = std::mem::take(current_matrix);
        append_matrix(
            matrix,
            nports,
            two_port_order,
            matrix_format,
            frequencies,
            s_values,
            frequency,
            raw.number,
        )?;
    }
    Ok(())
}

fn decode_network_scalars(
    tokens: &[Token<'_>],
    line: usize,
    format: DataFormat,
    record_elements: usize,
    current_scalars: &mut Vec<ScalarValue>,
    current_matrix: &mut Vec<Complex64>,
) -> Result<()> {
    let record_values = record_elements
        .checked_mul(2)
        .ok_or_else(|| Error::V2SizeOverflow {
            line,
            quantity: SizeQuantity::RecordValues,
            detail: format!("{record_elements} complex matrix elements × 2 scalars"),
        })?;
    let decoded_values = current_matrix
        .len()
        .checked_mul(2)
        .and_then(|count| count.checked_add(current_scalars.len()))
        .ok_or_else(|| Error::V2SizeOverflow {
            line,
            quantity: SizeQuantity::RecordValues,
            detail: "decoded scalar count".to_owned(),
        })?;
    if decoded_values > record_values {
        return Err(Error::V2Count {
            line,
            what: "scalar values in the current matrix record".to_owned(),
            expected: record_values,
            actual: decoded_values,
        });
    }
    let remaining = record_values - decoded_values;
    if tokens.len() > remaining {
        return Err(Error::V2Count {
            line,
            what: "scalar values in the current matrix record".to_owned(),
            expected: remaining,
            actual: tokens.len(),
        });
    }
    for token in tokens {
        current_scalars.push(ScalarValue {
            value: parse_number(*token, line)?,
            line,
            column: token.column,
        });
        if current_scalars.len() == 2 {
            let first = current_scalars[0];
            let second = current_scalars[1];
            current_matrix.push(decode_scalar_pair(first, second, format)?);
            current_scalars.clear();
        }
    }
    Ok(())
}

fn decode_scalar_pair(
    first: ScalarValue,
    second: ScalarValue,
    format: DataFormat,
) -> Result<Complex64> {
    match format {
        DataFormat::Ri => finite_complex(
            Complex64::new(first.value, second.value),
            first.line,
            first.column,
            format,
        ),
        DataFormat::Ma => {
            if first.value < 0.0 {
                return Err(Error::NegativeMagnitude {
                    line: first.line,
                    column: first.column,
                    value: first.value,
                });
            }
            let radians = second.value * (std::f64::consts::PI / 180.0);
            finite_complex(
                Complex64::new(first.value * radians.cos(), first.value * radians.sin()),
                first.line,
                first.column,
                format,
            )
        }
        DataFormat::Db => {
            let magnitude = 10.0_f64.powf(first.value / 20.0);
            let radians = second.value * (std::f64::consts::PI / 180.0);
            finite_complex(
                Complex64::new(magnitude * radians.cos(), magnitude * radians.sin()),
                first.line,
                first.column,
                format,
            )
        }
    }
}

fn triangular_row_major_index(
    format: MatrixFormat,
    nports: usize,
    row: usize,
    column: usize,
) -> Option<usize> {
    match format {
        MatrixFormat::Lower if row >= column => triangular_count(row)?.checked_add(column),
        MatrixFormat::Upper if row <= column => {
            let prior_rows = match row.checked_sub(1) {
                Some(prior) => triangular_count(prior)?,
                None => 0,
            };
            row.checked_mul(nports)?
                .checked_sub(prior_rows)?
                .checked_add(column.checked_sub(row)?)
        }
        MatrixFormat::Full | MatrixFormat::Lower | MatrixFormat::Upper => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn append_matrix(
    matrix: Vec<Complex64>,
    nports: usize,
    order: TwoPortDataOrder,
    matrix_format: MatrixFormat,
    frequencies: &mut Vec<f64>,
    s_values: &mut Vec<Complex64>,
    frequency: f64,
    line: usize,
) -> Result<()> {
    let full_elements = nports
        .checked_mul(nports)
        .ok_or_else(|| Error::V2SizeOverflow {
            line,
            quantity: SizeQuantity::MatrixElements,
            detail: format!("{nports} ports × {nports} expanded elements"),
        })?;
    let record_elements = if matrix_format == MatrixFormat::Full {
        full_elements
    } else {
        triangular_count(nports).ok_or_else(|| Error::V2SizeOverflow {
            line,
            quantity: SizeQuantity::MatrixElements,
            detail: format!("{nports} ports × ({nports} + 1) / 2 compact elements"),
        })?
    };
    if matrix.len() != record_elements {
        return Err(Error::V2Structural {
            line,
            message: "completed matrix record has an invalid number of pairs".to_owned(),
        });
    }
    frequencies.push(frequency);
    if matrix_format == MatrixFormat::Full && nports == 2 && order == TwoPortDataOrder::Legacy21_12
    {
        // Physical order N11, N21, N12, N22; Network is row-major N11,
        // N12, N21, N22.
        s_values.extend([matrix[0], matrix[2], matrix[1], matrix[3]]);
    } else if matrix_format == MatrixFormat::Full {
        s_values.extend(matrix);
    } else {
        // Triangular Touchstone records declare symmetry.  Expand the missing
        // half with a plain transpose, preserving complex values without
        // conjugation.  The two-port rule is deliberately special: both
        // Lower and Upper records carry N11, N21, N22 in that order.
        let storage_format = if nports == 2 {
            MatrixFormat::Lower
        } else {
            matrix_format
        };
        for row in 0..nports {
            for column in 0..nports {
                let (stored_row, stored_column) = match storage_format {
                    MatrixFormat::Lower if row >= column => (row, column),
                    MatrixFormat::Lower => (column, row),
                    MatrixFormat::Upper if row <= column => (row, column),
                    MatrixFormat::Upper => (column, row),
                    MatrixFormat::Full => unreachable!("triangular expansion uses Lower/Upper"),
                };
                let index =
                    triangular_row_major_index(storage_format, nports, stored_row, stored_column)
                        .ok_or_else(|| Error::V2SizeOverflow {
                        line,
                        quantity: SizeQuantity::MatrixElements,
                        detail: "triangular matrix index".to_owned(),
                    })?;
                s_values.push(matrix[index]);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn finish_end(
    line: usize,
    nfrequencies: usize,
    record_elements: usize,
    current_frequency: Option<f64>,
    current_scalars: &[ScalarValue],
    current_matrix: &[Complex64],
    actual_frequencies: usize,
    record_line: usize,
) -> Result<()> {
    if current_frequency.is_some() {
        let expected = record_elements
            .checked_mul(2)
            .ok_or_else(|| Error::V2SizeOverflow {
                line,
                quantity: SizeQuantity::RecordValues,
                detail: format!("{record_elements} complex matrix elements × 2 scalars"),
            })?;
        let actual = current_matrix
            .len()
            .checked_mul(2)
            .and_then(|count| count.checked_add(current_scalars.len()))
            .ok_or_else(|| Error::V2SizeOverflow {
                line,
                quantity: SizeQuantity::RecordValues,
                detail: "decoded scalar count".to_owned(),
            })?;
        return Err(Error::V2Count {
            line,
            what: format!("scalar values in matrix record starting at line {record_line}"),
            expected,
            actual,
        });
    }
    if actual_frequencies != nfrequencies {
        return Err(Error::V2Count {
            line,
            what: "network frequency records".to_owned(),
            expected: nfrequencies,
            actual: actual_frequencies,
        });
    }
    Ok(())
}

fn is_supported_keyword(name: &str) -> bool {
    matches!(
        name,
        "version"
            | "number of ports"
            | "two-port data order"
            | "reference"
            | "number of frequencies"
            | "matrix format"
            | "network data"
            | "end"
    )
}
