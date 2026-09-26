use num_complex::Complex64;
use rfkit_touchstone::{Error, parse_touchstone_v2_0_s};

fn one_port_document(option: &str, header: &str, data: &str, end: &str) -> String {
    format!(
        "[Version] 2.0\n{option}\n[Number of Ports] 1\n{header}[Number of Frequencies] 1\n[Network Data]\n{data}\n{end}\n"
    )
}

fn triangular_document(
    nports: usize,
    format: &str,
    order: Option<&str>,
    frequencies: &[f64],
    references: &[f64],
) -> String {
    let effective_format = if format.is_empty() { "Full" } else { format };
    let mut input = format!("[Version] 2.0\n# Hz S RI\n[Number of Ports] {nports}\n");
    if let Some(order) = order {
        input.push_str(&format!("[Two-Port Data Order] {order}\n"));
    }
    input.push_str(&format!("[Number of Frequencies] {}\n", frequencies.len()));
    if !references.is_empty() {
        input.push_str("[Reference] ");
        for (index, reference) in references.iter().enumerate() {
            if index != 0 {
                input.push(' ');
            }
            input.push_str(&reference.to_string());
        }
        input.push('\n');
    }
    if !format.is_empty() {
        input.push_str(&format!("[Matrix Format] {format}\n"));
    }
    input.push_str("[Network Data]\n");
    for (sample, frequency) in frequencies.iter().enumerate() {
        input.push_str(&frequency.to_string());
        for row in 0..nports {
            let columns: Box<dyn Iterator<Item = usize>> = match effective_format {
                "Lower" if nports == 2 => Box::new(0..=row),
                "Lower" => Box::new(0..=row),
                "Upper" => {
                    if nports == 2 {
                        Box::new(0..=row)
                    } else {
                        Box::new(row..nports)
                    }
                }
                "Full" => Box::new(0..nports),
                _ => panic!("test helper only supports Full/Lower/Upper"),
            };
            for column in columns {
                let (canonical_row, canonical_column) = if effective_format == "Full" {
                    (row.max(column), row.min(column))
                } else if effective_format == "Upper" && nports != 2 {
                    (column.max(row), column.min(row))
                } else {
                    (row, column)
                };
                let value =
                    10.0 * sample as f64 + (canonical_row * nports + canonical_column + 1) as f64;
                input.push_str(&format!(" {value} -{}", value / 10.0));
            }
        }
        input.push('\n');
    }
    input.push_str("[End]\n");
    input
}

#[test]
fn parses_one_port_with_full_record_continuation_and_explicit_reference() {
    let input = "[Version] 2.0\n# MHz S RI R 75\n[Number of Ports] 1\n[Number of Frequencies] 2\n[Reference]\n61\n[Matrix Format] Full\n[Network Data]\n1 0.1 0.2\n2\n0.3 0.4\n[End]\n";
    let network = parse_touchstone_v2_0_s(input).unwrap();
    assert_eq!(network.frequency().hz(), &[1.0e6, 2.0e6]);
    assert_eq!(network.z0().shape(), &[2, 1]);
    assert_eq!(network.z0()[[0, 0]], Complex64::new(61.0, 0.0));
    assert_eq!(network.s()[[0, 0, 0]], Complex64::new(0.1, 0.2));
    assert_eq!(network.s()[[1, 0, 0]], Complex64::new(0.3, 0.4));
}

#[test]
fn preserves_asymmetric_two_port_orders_and_n_port_row_major_values() {
    let legacy = "[Version] 2.0\n# Hz S RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] 21_12\n[Number of Frequencies] 1\n[Network Data]\n100 1 0 2 0\n3 0 4 0\n[End]\n";
    let network = parse_touchstone_v2_0_s(legacy).unwrap();
    assert_eq!(
        network.s().as_slice().unwrap(),
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]
    );

    let natural = legacy.replace("21_12", "12_21");
    let network = parse_touchstone_v2_0_s(&natural).unwrap();
    assert_eq!(
        network.s().as_slice().unwrap(),
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]
    );

    let three = "[Version] 2.0\n# GHz S RI R 50\n[Number of Ports] 3\n[Number of Frequencies] 1\n[Network Data]\n1 11 0 12 0 13 0 21 0 22 0\n23 0 31 0 32 0 33 0\n[End]\n";
    let network = parse_touchstone_v2_0_s(three).unwrap();
    assert_eq!(network.s()[[0, 0, 2]], Complex64::new(13.0, 0.0));
    assert_eq!(network.s()[[0, 1, 0]], Complex64::new(21.0, 0.0));
    assert_eq!(network.s()[[0, 2, 1]], Complex64::new(32.0, 0.0));
}

#[test]
fn decodes_ma_db_units_and_reference_override() {
    let ma = "[Version] 2.0\n# kHz S MA R 11\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Reference] 73\n[Network Data]\n2 0.5 90\n[End]\n";
    let network = parse_touchstone_v2_0_s(ma).unwrap();
    assert_eq!(network.frequency().hz(), &[2.0e3]);
    assert_eq!(network.z0()[[0, 0]], Complex64::new(73.0, 0.0));
    assert!((network.s()[[0, 0, 0]].re).abs() < 1.0e-15);
    assert!((network.s()[[0, 0, 0]].im - 0.5).abs() < 1.0e-15);

    let db = "[Version] 2.0\n# GHz S DB R 50\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 -6.020599913279624 0\n[End]\n";
    let network = parse_touchstone_v2_0_s(db).unwrap();
    assert!((network.s()[[0, 0, 0]].re - 0.5).abs() < 1.0e-14);
}

#[test]
fn decodes_pairs_split_across_physical_lines_for_ri_ma_and_db() {
    // The second scalar of the RI pair is on a continuation line.
    let ri = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0.25\n-0.5\n[End]\n";
    let network = parse_touchstone_v2_0_s(ri).unwrap();
    assert_eq!(network.s()[[0, 0, 0]], Complex64::new(0.25, -0.5));

    // The first continuation line leaves a pending scalar, and the next
    // continuation line supplies its mate before starting another pair.
    let ma = "[Version] 2.0\n# Hz S MA\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n[Network Data]\n1 0.5 0\n0.5\n90 0.25\n180 1 0\n[End]\n";
    let network = parse_touchstone_v2_0_s(ma).unwrap();
    assert!((network.s()[[0, 0, 0]].re - 0.5).abs() < 1.0e-15);
    assert!(network.s()[[0, 0, 0]].im.abs() < 1.0e-15);
    assert!(network.s()[[0, 0, 1]].re.abs() < 1.0e-15);
    assert!((network.s()[[0, 0, 1]].im - 0.5).abs() < 1.0e-15);
    assert!((network.s()[[0, 1, 0]].re + 0.25).abs() < 1.0e-15);
    assert!(network.s()[[0, 1, 0]].im.abs() < 1.0e-15);
    assert!((network.s()[[0, 1, 1]].re - 1.0).abs() < 1.0e-15);
    assert!(network.s()[[0, 1, 1]].im.abs() < 1.0e-15);

    let db = "[Version] 2.0\n# Hz S DB\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 -6.020599913279624\n0\n[End]\n";
    let network = parse_touchstone_v2_0_s(db).unwrap();
    assert!((network.s()[[0, 0, 0]].re - 0.5).abs() < 1.0e-14);
    assert!(network.s()[[0, 0, 0]].im.abs() < 1.0e-15);
}

#[test]
fn reports_numeric_error_from_split_pair_on_source_line() {
    let input = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0.25\nNaN\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(input),
        Err(Error::NonFiniteNumber {
            line: 7,
            column: 1,
            ..
        })
    ));
}

#[test]
fn accepts_defaults_all_frequency_units_option_reference_and_ordinary_comments() {
    let defaults = "! ordinary comment\r\n[Version] 2.0\r\n#\r\n[Number of Ports] 1\r\n! Port[1] = input\r\n[Number of Frequencies] 1\r\n[Network Data]\r\n2 1 0\r\n[End]\r\n";
    let network = parse_touchstone_v2_0_s(defaults).unwrap();
    assert_eq!(network.frequency().hz(), &[2.0e9]);
    assert_eq!(network.z0()[[0, 0]], Complex64::new(50.0, 0.0));
    assert_eq!(network.s()[[0, 0, 0]], Complex64::new(1.0, 0.0));

    for (unit, multiplier) in [("Hz", 1.0), ("kHz", 1.0e3), ("MHz", 1.0e6), ("GHz", 1.0e9)] {
        let option = format!("# {unit} S RI R 72.5");
        let input = one_port_document(&option, "", "2 0.25 -0.5", "[End]");
        let network = parse_touchstone_v2_0_s(&input).unwrap();
        assert_eq!(network.frequency().hz(), &[2.0 * multiplier]);
        assert_eq!(network.z0()[[0, 0]], Complex64::new(72.5, 0.0));
        assert_eq!(network.s()[[0, 0, 0]], Complex64::new(0.25, -0.5));
    }
}

#[test]
fn accepts_five_port_full_record_with_more_than_four_pairs_on_one_line() {
    let mut input = String::from(
        "[Version] 2.0\n# Hz S RI R 63\n[Number of Ports] 5\n[Number of Frequencies] 1\n[Network Data]\n1",
    );
    for value in 1..=25 {
        input.push_str(&format!(" {value} 0"));
    }
    input.push_str("\n[End]\n");

    let network = parse_touchstone_v2_0_s(&input).unwrap();
    assert_eq!(network.s().dim(), (1, 5, 5));
    assert_eq!(network.s()[[0, 0, 4]], Complex64::new(5.0, 0.0));
    assert_eq!(network.s()[[0, 4, 0]], Complex64::new(21.0, 0.0));
    assert_eq!(network.s()[[0, 4, 4]], Complex64::new(25.0, 0.0));
}

#[test]
fn rejects_missing_duplicate_wrong_and_misplaced_header_directives() {
    let cases = [
        "# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n",
        "[Version] 2.0\n[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n",
        "[Version] 2.0\n[Number of Ports] 1\n# Hz S RI\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n",
        "[Version] 2.0\n# Hz S RI\n[Number_of_Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n",
    ];
    for input in cases {
        assert!(
            matches!(
                parse_touchstone_v2_0_s(input),
                Err(Error::V2Structural { .. })
            ),
            "accepted malformed header: {input:?}"
        );
    }

    let wrong_version = "[Version] 2.1\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(wrong_version),
        Err(Error::V2Unsupported { .. })
    ));
}

#[test]
fn ignores_later_option_lines_even_when_they_are_malformed() {
    let before_ports = "[Version] 2.0\n# Hz S RI R 61\n# not-an-option\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n";
    let network = parse_touchstone_v2_0_s(before_ports).unwrap();
    assert_eq!(network.frequency().hz(), &[1.0]);
    assert_eq!(network.z0()[[0, 0]], Complex64::new(61.0, 0.0));

    let after_data = "[Version] 2.0\n# Hz S RI R 61\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n# also ignored\n[End]\n";
    let network = parse_touchstone_v2_0_s(after_data).unwrap();
    assert_eq!(network.frequency().hz(), &[1.0]);
    assert_eq!(network.s()[[0, 0, 0]], Complex64::new(0.0, 0.0));
}

#[test]
fn ignores_later_options_while_reference_vector_is_pending() {
    for later_option in ["# GHz S DB R 999", "# Hz S XY R nope"] {
        let before_first = format!(
            "[Version] 2.0\n# MHz S MA R 61\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Reference]\n{later_option}\n61 73\n[Number of Frequencies] 1\n[Network Data]\n1 0.5 90 0 0 0 0 0 0\n[End]\n"
        );
        let network = parse_touchstone_v2_0_s(&before_first).unwrap();
        assert_eq!(network.frequency().hz(), &[1.0e6]);
        assert_eq!(network.z0()[[0, 0]], Complex64::new(61.0, 0.0));
        assert_eq!(network.z0()[[0, 1]], Complex64::new(73.0, 0.0));
        assert!(network.s()[[0, 0, 0]].re.abs() < 1.0e-15);
        assert!((network.s()[[0, 0, 0]].im - 0.5).abs() < 1.0e-15);

        let between = format!(
            "[Version] 2.0\n# MHz S MA R 61\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Reference] 61\n{later_option}\n73\n[Number of Frequencies] 1\n[Network Data]\n1 0.5 90 0 0 0 0 0 0\n[End]\n"
        );
        let network = parse_touchstone_v2_0_s(&between).unwrap();
        assert_eq!(network.frequency().hz(), &[1.0e6]);
        assert_eq!(network.z0()[[0, 0]], Complex64::new(61.0, 0.0));
        assert_eq!(network.z0()[[0, 1]], Complex64::new(73.0, 0.0));
        assert!(network.s()[[0, 0, 0]].re.abs() < 1.0e-15);
        assert!((network.s()[[0, 0, 0]].im - 0.5).abs() < 1.0e-15);
    }
}

#[test]
fn reports_source_columns_for_inline_and_continued_reference_arguments() {
    let malformed = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Reference]   50+j\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(malformed),
        Err(Error::MalformedNumber {
            line: 4,
            column: 15,
            ref token,
        }) if token == "50+j"
    ));

    let nonfinite = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Reference] \t NaN\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(nonfinite),
        Err(Error::NonFiniteNumber {
            line: 4,
            column: 15,
            ref token,
        }) if token == "NaN"
    ));

    let continuation = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 2\n[Reference] 37\n   NaN\n[Number of Frequencies] 1\n[Two-Port Data Order] 12_21\n[Network Data]\n1 0 0 0 0 0 0 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(continuation),
        Err(Error::NonFiniteNumber {
            line: 5,
            column: 4,
            ref token,
        }) if token == "NaN"
    ));
}

#[test]
fn rejects_invalid_option_lines_and_reference_vectors() {
    let non_s = one_port_document("# Hz Z RI R 50", "", "1 0 0", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s(&non_s),
        Err(Error::UnsupportedParameter { .. })
    ));

    for option in ["# THz S RI R 50", "# Hz S XY R 50", "# Hz S RI R nope"] {
        let input = one_port_document(option, "", "1 0 0", "[End]");
        assert!(matches!(
            parse_touchstone_v2_0_s(&input),
            Err(Error::MalformedOption { .. })
        ));
    }
    for option in ["# Hz S RI R 0", "# Hz S RI R -1", "# Hz S RI R 1e999"] {
        let input = one_port_document(option, "", "1 0 0", "[End]");
        assert!(matches!(
            parse_touchstone_v2_0_s(&input),
            Err(Error::InvalidReferenceResistance { .. })
        ));
    }

    let reference_cases = [
        ("[Reference] 0\n", "zero"),
        ("[Reference] -1\n", "negative"),
        ("[Reference] NaN\n", "non-finite"),
        ("[Reference] 50+j\n", "complex"),
        ("[Reference] 50 60\n", "surplus"),
    ];
    for (header, label) in reference_cases {
        let input = one_port_document("# Hz S RI", header, "1 0 0", "[End]");
        assert!(
            parse_touchstone_v2_0_s(&input).is_err(),
            "accepted {label} reference"
        );
    }

    let incomplete = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Reference] 50\n[Number of Frequencies] 1\n[Network Data]\n1 0 0 0 0 0 0 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(incomplete),
        Err(Error::V2Count { .. })
    ));
}

#[test]
fn rejects_duplicate_misplaced_and_unsupported_subset_keywords() {
    let cases = [
        "[Number of Ports] 1\n",
        "[Number of Frequencies] 1\n[Number of Frequencies] 1\n",
        "[Reference] 50\n[Reference] 50\n",
        "[Matrix Format] Full\n[Matrix Format] Full\n",
    ];
    for header in cases {
        let input = one_port_document("# Hz S RI", header, "1 0 0", "[End]");
        assert!(matches!(
            parse_touchstone_v2_0_s(&input),
            Err(Error::V2Structural { .. })
        ));
    }

    let misplaced_reference = one_port_document("# Hz S RI", "", "1 0 0\n[Reference] 50", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s(&misplaced_reference),
        Err(Error::V2Structural { .. })
    ));

    for keyword in [
        "[Mixed-Mode Order] S1",
        "[Number of Noise Frequencies] 1",
        "[Begin Information]",
        "[Unknown Keyword] value",
        "[Two Port Data Order] 12_21",
    ] {
        let header = format!("{keyword}\n");
        let input = one_port_document("# Hz S RI", &header, "1 0 0", "[End]");
        assert!(
            matches!(
                parse_touchstone_v2_0_s(&input),
                Err(Error::V2Unsupported { .. })
            ),
            "did not classify {keyword:?} as unsupported"
        );
    }
}

#[test]
fn rejects_malformed_nonfinite_and_out_of_order_network_records() {
    let structural = ["1 0", "1 0 0 1 0", " 1 0 0"];
    for data in structural {
        let input = one_port_document("# Hz S RI", "", data, "[End]");
        assert!(parse_touchstone_v2_0_s(&input).is_err());
    }

    let negative = one_port_document("# Hz S RI", "", "-1 0 0", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s(&negative),
        Err(Error::NegativeFrequency { .. })
    ));
    let nonfinite_frequency = one_port_document("# Hz S RI", "", "NaN 0 0", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s(&nonfinite_frequency),
        Err(Error::NonFiniteFrequency { .. })
    ));
    let nonfinite_s = one_port_document("# Hz S RI", "", "1 NaN 0", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s(&nonfinite_s),
        Err(Error::NonFiniteNumber { .. })
    ));
    let negative_magnitude = one_port_document("# Hz S MA", "", "1 -1 0", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s(&negative_magnitude),
        Err(Error::NegativeMagnitude { .. })
    ));
    let db_overflow = one_port_document("# Hz S DB", "", "1 10000 0", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s(&db_overflow),
        Err(Error::ConversionOverflow { .. })
    ));

    let non_increasing = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 2\n[Network Data]\n2 0 0\n1 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(non_increasing),
        Err(Error::FrequencyNotStrictlyIncreasing { .. })
    ));
}

#[test]
fn rejects_end_failures_unknown_characters_and_truncated_two_port_records() {
    let missing_end = one_port_document("# Hz S RI", "", "1 0 0", "");
    assert!(matches!(
        parse_touchstone_v2_0_s(&missing_end),
        Err(Error::V2Structural { .. })
    ));
    let end_args = one_port_document("# Hz S RI", "", "1 0 0", "[End] extra");
    assert!(matches!(
        parse_touchstone_v2_0_s(&end_args),
        Err(Error::V2Structural { .. })
    ));
    let duplicate_end = one_port_document("# Hz S RI", "", "1 0 0", "[End]\n[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s(&duplicate_end),
        Err(Error::V2Structural { .. })
    ));
    let non_ascii = one_port_document("# Hz S RI", "", "1 0 0 é", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s(&non_ascii),
        Err(Error::UnsupportedCharacter { .. })
    ));

    let truncated = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n[Network Data]\n1 0 0 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(truncated),
        Err(Error::V2Count { .. })
    ));
}

#[test]
fn rejects_v2_exclusions_order_count_and_trailing_content() {
    let missing_order = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 2\n[Number of Frequencies] 1\n[Network Data]\n1 0 0 0 0 0 0 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(missing_order),
        Err(Error::V2Structural { .. })
    ));

    let lower = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Matrix Format] Lower\n[Network Data]\n1 0 0\n[End]\n";
    assert_eq!(
        parse_touchstone_v2_0_s(lower).unwrap().s()[[0, 0, 0]],
        Complex64::new(0.0, 0.0)
    );

    let noise = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Number of Noise Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(noise),
        Err(Error::V2Unsupported { .. })
    ));

    let wrong_count = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 2\n[Network Data]\n1 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(wrong_count),
        Err(Error::V2Count { .. })
    ));

    let trailing = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n1 0 0\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(trailing),
        Err(Error::V2Structural { .. })
    ));
}

#[test]
fn rejects_hfss_semantic_comments_and_pathological_dimensions_without_panic() {
    let hfss = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0 0 ! Port Impedance 50 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(hfss),
        Err(Error::UnsupportedExtension { .. })
    ));

    let overflowing_ports = format!(
        "[Version] 2.0\n# Hz S RI\n[Number of Ports] {}\n[Number of Frequencies] 1\n[Network Data]\n1\n[End]\n",
        usize::MAX
    );
    let overflow_result = std::panic::catch_unwind(|| parse_touchstone_v2_0_s(&overflowing_ports));
    assert!(matches!(
        overflow_result,
        Ok(Err(Error::V2SizeOverflow { .. }))
    ));

    let large_but_representable_ports = ((usize::MAX as f64).sqrt() as usize) / 2;
    let large_ports = format!(
        "[Version] 2.0\n# Hz S RI\n[Number of Ports] {large_but_representable_ports}\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n"
    );
    let large_result = std::panic::catch_unwind(|| parse_touchstone_v2_0_s(&large_ports));
    assert!(matches!(large_result, Ok(Err(Error::V2Count { .. }))));

    let huge_frequency_count = format!(
        "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] {}\n[Network Data]\n1 0 0\n[End]\n",
        usize::MAX
    );
    let frequency_result =
        std::panic::catch_unwind(|| parse_touchstone_v2_0_s(&huge_frequency_count));
    assert!(matches!(frequency_result, Ok(Err(Error::V2Count { .. }))));
}

#[test]
fn expands_lower_and_upper_complex_three_port_records_without_conjugation() {
    let lower = triangular_document(3, "Lower", None, &[1.0, 2.0], &[25.0, 50.0, 75.0]);
    let upper = triangular_document(3, "Upper", None, &[1.0, 2.0], &[25.0, 50.0, 75.0]);
    let lower_network = parse_touchstone_v2_0_s(&lower).unwrap();
    let upper_network = parse_touchstone_v2_0_s(&upper).unwrap();
    assert_eq!(lower_network.frequency().hz(), &[1.0, 2.0]);
    assert_eq!(lower_network.z0()[[0, 0]], Complex64::new(25.0, 0.0));
    assert_eq!(lower_network.z0()[[0, 2]], Complex64::new(75.0, 0.0));
    assert_eq!(lower_network.s(), upper_network.s());

    let expected = Complex64::new(4.0, -0.4);
    assert_eq!(lower_network.s()[[0, 0, 1]], expected);
    assert_eq!(lower_network.s()[[0, 1, 0]], expected);
    assert_eq!(lower_network.s()[[0, 1, 2]], Complex64::new(8.0, -0.8));
    assert_eq!(lower_network.s()[[0, 2, 1]], Complex64::new(8.0, -0.8));
    assert_eq!(lower_network.s()[[1, 2, 2]], Complex64::new(19.0, -1.9));
}

#[test]
fn triangular_one_port_and_larger_n_records_preserve_samples_and_references() {
    for format in ["Lower", "Upper"] {
        let input = triangular_document(1, format, None, &[10.0], &[91.0]);
        let network = parse_touchstone_v2_0_s(&input).unwrap();
        assert_eq!(network.s()[[0, 0, 0]], Complex64::new(1.0, -0.1));
        assert_eq!(network.z0()[[0, 0]], Complex64::new(91.0, 0.0));
    }

    let input = triangular_document(
        5,
        "Upper",
        None,
        &[100.0, 200.0],
        &[10.0, 20.0, 30.0, 40.0, 50.0],
    );
    let network = parse_touchstone_v2_0_s(&input).unwrap();
    assert_eq!(network.s().dim(), (2, 5, 5));
    assert_eq!(network.s()[[0, 0, 4]], Complex64::new(21.0, -2.1));
    assert_eq!(network.s()[[0, 4, 0]], Complex64::new(21.0, -2.1));
    assert_eq!(network.s()[[1, 3, 4]], Complex64::new(34.0, -3.4));
    assert_eq!(network.s()[[1, 4, 3]], Complex64::new(34.0, -3.4));
    assert_eq!(network.z0()[[1, 4]], Complex64::new(50.0, 0.0));
}

#[test]
fn triangular_two_port_orders_follow_standards_n11_n21_n22_rule() {
    let expected = [
        Complex64::new(0.1, 0.2),
        Complex64::new(0.3, 0.4),
        Complex64::new(0.3, 0.4),
        Complex64::new(0.5, 0.6),
    ];
    for format in ["Lower", "Upper"] {
        for order in ["21_12", "12_21"] {
            let input = format!(
                "[Version] 2.0\n# Hz S RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] {order}\n[Number of Frequencies] 1\n[Matrix Format] {format}\n[Network Data]\n1 0.1 0.2 0.3 0.4 0.5 0.6\n[End]\n"
            );
            let network = parse_touchstone_v2_0_s(&input).unwrap();
            assert_eq!(network.s().as_slice().unwrap(), &expected);
        }
    }
}

#[test]
fn full_lower_upper_invariant_holds_for_symmetric_complex_coordinates() {
    let frequencies = [1.0, 2.0];
    let references = [31.0, 47.0, 83.0];
    let full = parse_touchstone_v2_0_s(&triangular_document(
        3,
        "Full",
        None,
        &frequencies,
        &references,
    ))
    .unwrap();
    let lower = parse_touchstone_v2_0_s(&triangular_document(
        3,
        "Lower",
        None,
        &frequencies,
        &references,
    ))
    .unwrap();
    let upper = parse_touchstone_v2_0_s(&triangular_document(
        3,
        "Upper",
        None,
        &frequencies,
        &references,
    ))
    .unwrap();
    assert_eq!(full.s(), lower.s());
    assert_eq!(full.s(), upper.s());
    assert_eq!(full.frequency(), lower.frequency());
    assert_eq!(full.z0(), lower.z0());

    // The absent keyword is Full by definition.
    let default_full =
        parse_touchstone_v2_0_s(&triangular_document(3, "", None, &frequencies, &references))
            .unwrap();
    assert_eq!(default_full.s(), full.s());
}

#[test]
fn triangular_ma_and_db_decode_with_the_same_mirroring_rules() {
    let ma = "[Version] 2.0\n# MHz S MA R 50\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n[Matrix Format] Upper\n[Network Data]\n2 0.5 0 0.25 90 0.8 -45\n[End]\n";
    let ma_network = parse_touchstone_v2_0_s(ma).unwrap();
    assert!((ma_network.s()[[0, 0, 1]].re).abs() < 1.0e-15);
    assert!((ma_network.s()[[0, 1, 0]].im - 0.25).abs() < 1.0e-15);
    assert!((ma_network.s()[[0, 1, 1]].re - 0.8 / 2.0_f64.sqrt()).abs() < 1.0e-15);

    let db = "[Version] 2.0\n# GHz S DB R 50\n[Number of Ports] 2\n[Two-Port Data Order] 21_12\n[Number of Frequencies] 1\n[Matrix Format] Lower\n[Network Data]\n1 -6.020599913279624 0 -12.041199826559248 90 0 0\n[End]\n";
    let db_network = parse_touchstone_v2_0_s(db).unwrap();
    assert!((db_network.s()[[0, 0, 0]].re - 0.5).abs() < 1.0e-14);
    assert!((db_network.s()[[0, 0, 1]].im - 0.25).abs() < 1.0e-14);
    assert!((db_network.s()[[0, 1, 0]].im - 0.25).abs() < 1.0e-14);
}

#[test]
fn triangular_continuations_can_split_pairs_and_rows_at_frequency_boundaries() {
    let input = "[Version] 2.0\n# Hz S RI R 50\n[Number of Ports] 3\n[Number of Frequencies] 2\n[Matrix Format] Lower\n[Network Data]\n1 1\n0.1 2 0.2\n3 -0.3 4 0.4\n5\n-0.5 6 -0.6\n2 7 0.7 8 0.8\n9 -0.9 10 1.0\n11 -1.1 12 1.2\n[End]\n";
    let network = parse_touchstone_v2_0_s(input).unwrap();
    assert_eq!(network.frequency().hz(), &[1.0, 2.0]);
    assert_eq!(network.s()[[0, 0, 0]], Complex64::new(1.0, 0.1));
    assert_eq!(network.s()[[0, 2, 1]], Complex64::new(5.0, -0.5));
    assert_eq!(network.s()[[1, 0, 2]], Complex64::new(10.0, 1.0));
    assert_eq!(network.s()[[1, 2, 2]], Complex64::new(12.0, 1.2));
}

#[test]
fn triangular_counts_and_format_validation_remain_structured_and_bounded() {
    let incomplete = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 3\n[Number of Frequencies] 1\n[Matrix Format] Lower\n[Network Data]\n1 1 0 2 0 3 0 4 0 5 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(incomplete),
        Err(Error::V2Count { .. })
    ));

    let excess = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 3\n[Number of Frequencies] 1\n[Matrix Format] Upper\n[Network Data]\n1 1 0 2 0 3 0 4 0 5 0 6 0 7 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(excess),
        Err(Error::V2Count { .. })
    ));

    let malformed = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 3\n[Number of Frequencies] 1\n[Matrix Format] Lower\n[Network Data]\n1 NaN 0 2 0 3 0 4 0 5 0 6 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s(malformed),
        Err(Error::NonFiniteNumber { .. })
    ));

    for format in ["Diagonal", "Lower Upper"] {
        let input = format!(
            "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Matrix Format] {format}\n[Network Data]\n1 0 0\n[End]\n"
        );
        assert!(matches!(
            parse_touchstone_v2_0_s(&input),
            Err(Error::V2Structural { .. })
        ));
    }

    let duplicate = one_port_document(
        "# Hz S RI",
        "[Matrix Format] Lower\n[Matrix Format] Upper\n",
        "1 0 0",
        "[End]",
    );
    assert!(matches!(
        parse_touchstone_v2_0_s(&duplicate),
        Err(Error::V2Structural { .. })
    ));

    let huge = format!(
        "[Version] 2.0\n# Hz S RI\n[Number of Ports] {}\n[Number of Frequencies] 1\n[Matrix Format] Lower\n[Network Data]\n1 0 0\n[End]\n",
        ((usize::MAX as f64).sqrt() as usize) / 2
    );
    let result = std::panic::catch_unwind(|| parse_touchstone_v2_0_s(&huge));
    assert!(matches!(result, Ok(Err(Error::V2Count { .. }))));
}
