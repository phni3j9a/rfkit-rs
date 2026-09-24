use num_complex::Complex64;
use rfkit_touchstone::{Error, parse_touchstone_v2_0_s_full};

fn one_port_document(option: &str, header: &str, data: &str, end: &str) -> String {
    format!(
        "[Version] 2.0\n{option}\n[Number of Ports] 1\n{header}[Number of Frequencies] 1\n[Network Data]\n{data}\n{end}\n"
    )
}

#[test]
fn parses_one_port_with_full_record_continuation_and_explicit_reference() {
    let input = "[Version] 2.0\n# MHz S RI R 75\n[Number of Ports] 1\n[Number of Frequencies] 2\n[Reference]\n61\n[Matrix Format] Full\n[Network Data]\n1 0.1 0.2\n2\n0.3 0.4\n[End]\n";
    let network = parse_touchstone_v2_0_s_full(input).unwrap();
    assert_eq!(network.frequency().hz(), &[1.0e6, 2.0e6]);
    assert_eq!(network.z0().shape(), &[2, 1]);
    assert_eq!(network.z0()[[0, 0]], Complex64::new(61.0, 0.0));
    assert_eq!(network.s()[[0, 0, 0]], Complex64::new(0.1, 0.2));
    assert_eq!(network.s()[[1, 0, 0]], Complex64::new(0.3, 0.4));
}

#[test]
fn preserves_asymmetric_two_port_orders_and_n_port_row_major_values() {
    let legacy = "[Version] 2.0\n# Hz S RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] 21_12\n[Number of Frequencies] 1\n[Network Data]\n100 1 0 2 0\n3 0 4 0\n[End]\n";
    let network = parse_touchstone_v2_0_s_full(legacy).unwrap();
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
    let network = parse_touchstone_v2_0_s_full(&natural).unwrap();
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
    let network = parse_touchstone_v2_0_s_full(three).unwrap();
    assert_eq!(network.s()[[0, 0, 2]], Complex64::new(13.0, 0.0));
    assert_eq!(network.s()[[0, 1, 0]], Complex64::new(21.0, 0.0));
    assert_eq!(network.s()[[0, 2, 1]], Complex64::new(32.0, 0.0));
}

#[test]
fn decodes_ma_db_units_and_reference_override() {
    let ma = "[Version] 2.0\n# kHz S MA R 11\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Reference] 73\n[Network Data]\n2 0.5 90\n[End]\n";
    let network = parse_touchstone_v2_0_s_full(ma).unwrap();
    assert_eq!(network.frequency().hz(), &[2.0e3]);
    assert_eq!(network.z0()[[0, 0]], Complex64::new(73.0, 0.0));
    assert!((network.s()[[0, 0, 0]].re).abs() < 1.0e-15);
    assert!((network.s()[[0, 0, 0]].im - 0.5).abs() < 1.0e-15);

    let db = "[Version] 2.0\n# GHz S DB R 50\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 -6.020599913279624 0\n[End]\n";
    let network = parse_touchstone_v2_0_s_full(db).unwrap();
    assert!((network.s()[[0, 0, 0]].re - 0.5).abs() < 1.0e-14);
}

#[test]
fn decodes_pairs_split_across_physical_lines_for_ri_ma_and_db() {
    // The second scalar of the RI pair is on a continuation line.
    let ri = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0.25\n-0.5\n[End]\n";
    let network = parse_touchstone_v2_0_s_full(ri).unwrap();
    assert_eq!(network.s()[[0, 0, 0]], Complex64::new(0.25, -0.5));

    // The first continuation line leaves a pending scalar, and the next
    // continuation line supplies its mate before starting another pair.
    let ma = "[Version] 2.0\n# Hz S MA\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n[Network Data]\n1 0.5 0\n0.5\n90 0.25\n180 1 0\n[End]\n";
    let network = parse_touchstone_v2_0_s_full(ma).unwrap();
    assert!((network.s()[[0, 0, 0]].re - 0.5).abs() < 1.0e-15);
    assert!(network.s()[[0, 0, 0]].im.abs() < 1.0e-15);
    assert!(network.s()[[0, 0, 1]].re.abs() < 1.0e-15);
    assert!((network.s()[[0, 0, 1]].im - 0.5).abs() < 1.0e-15);
    assert!((network.s()[[0, 1, 0]].re + 0.25).abs() < 1.0e-15);
    assert!(network.s()[[0, 1, 0]].im.abs() < 1.0e-15);
    assert!((network.s()[[0, 1, 1]].re - 1.0).abs() < 1.0e-15);
    assert!(network.s()[[0, 1, 1]].im.abs() < 1.0e-15);

    let db = "[Version] 2.0\n# Hz S DB\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 -6.020599913279624\n0\n[End]\n";
    let network = parse_touchstone_v2_0_s_full(db).unwrap();
    assert!((network.s()[[0, 0, 0]].re - 0.5).abs() < 1.0e-14);
    assert!(network.s()[[0, 0, 0]].im.abs() < 1.0e-15);
}

#[test]
fn reports_numeric_error_from_split_pair_on_source_line() {
    let input = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0.25\nNaN\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s_full(input),
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
    let network = parse_touchstone_v2_0_s_full(defaults).unwrap();
    assert_eq!(network.frequency().hz(), &[2.0e9]);
    assert_eq!(network.z0()[[0, 0]], Complex64::new(50.0, 0.0));
    assert_eq!(network.s()[[0, 0, 0]], Complex64::new(1.0, 0.0));

    for (unit, multiplier) in [("Hz", 1.0), ("kHz", 1.0e3), ("MHz", 1.0e6), ("GHz", 1.0e9)] {
        let option = format!("# {unit} S RI R 72.5");
        let input = one_port_document(&option, "", "2 0.25 -0.5", "[End]");
        let network = parse_touchstone_v2_0_s_full(&input).unwrap();
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

    let network = parse_touchstone_v2_0_s_full(&input).unwrap();
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
                parse_touchstone_v2_0_s_full(input),
                Err(Error::V2Structural { .. })
            ),
            "accepted malformed header: {input:?}"
        );
    }

    let wrong_version = "[Version] 2.1\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s_full(wrong_version),
        Err(Error::V2Unsupported { .. })
    ));
}

#[test]
fn ignores_later_option_lines_even_when_they_are_malformed() {
    let before_ports = "[Version] 2.0\n# Hz S RI R 61\n# not-an-option\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n";
    let network = parse_touchstone_v2_0_s_full(before_ports).unwrap();
    assert_eq!(network.frequency().hz(), &[1.0]);
    assert_eq!(network.z0()[[0, 0]], Complex64::new(61.0, 0.0));

    let after_data = "[Version] 2.0\n# Hz S RI R 61\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n# also ignored\n[End]\n";
    let network = parse_touchstone_v2_0_s_full(after_data).unwrap();
    assert_eq!(network.frequency().hz(), &[1.0]);
    assert_eq!(network.s()[[0, 0, 0]], Complex64::new(0.0, 0.0));
}

#[test]
fn rejects_invalid_option_lines_and_reference_vectors() {
    let non_s = one_port_document("# Hz Z RI R 50", "", "1 0 0", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s_full(&non_s),
        Err(Error::UnsupportedParameter { .. })
    ));

    for option in ["# THz S RI R 50", "# Hz S XY R 50", "# Hz S RI R nope"] {
        let input = one_port_document(option, "", "1 0 0", "[End]");
        assert!(matches!(
            parse_touchstone_v2_0_s_full(&input),
            Err(Error::MalformedOption { .. })
        ));
    }
    for option in ["# Hz S RI R 0", "# Hz S RI R -1", "# Hz S RI R 1e999"] {
        let input = one_port_document(option, "", "1 0 0", "[End]");
        assert!(matches!(
            parse_touchstone_v2_0_s_full(&input),
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
            parse_touchstone_v2_0_s_full(&input).is_err(),
            "accepted {label} reference"
        );
    }

    let incomplete = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Reference] 50\n[Number of Frequencies] 1\n[Network Data]\n1 0 0 0 0 0 0 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s_full(incomplete),
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
            parse_touchstone_v2_0_s_full(&input),
            Err(Error::V2Structural { .. })
        ));
    }

    let misplaced_reference = one_port_document("# Hz S RI", "", "1 0 0\n[Reference] 50", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s_full(&misplaced_reference),
        Err(Error::V2Structural { .. })
    ));

    for keyword in [
        "[Matrix Format] Upper",
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
                parse_touchstone_v2_0_s_full(&input),
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
        assert!(parse_touchstone_v2_0_s_full(&input).is_err());
    }

    let negative = one_port_document("# Hz S RI", "", "-1 0 0", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s_full(&negative),
        Err(Error::NegativeFrequency { .. })
    ));
    let nonfinite_frequency = one_port_document("# Hz S RI", "", "NaN 0 0", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s_full(&nonfinite_frequency),
        Err(Error::NonFiniteFrequency { .. })
    ));
    let nonfinite_s = one_port_document("# Hz S RI", "", "1 NaN 0", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s_full(&nonfinite_s),
        Err(Error::NonFiniteNumber { .. })
    ));
    let negative_magnitude = one_port_document("# Hz S MA", "", "1 -1 0", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s_full(&negative_magnitude),
        Err(Error::NegativeMagnitude { .. })
    ));
    let db_overflow = one_port_document("# Hz S DB", "", "1 10000 0", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s_full(&db_overflow),
        Err(Error::ConversionOverflow { .. })
    ));

    let non_increasing = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 2\n[Network Data]\n2 0 0\n1 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s_full(non_increasing),
        Err(Error::FrequencyNotStrictlyIncreasing { .. })
    ));
}

#[test]
fn rejects_end_failures_unknown_characters_and_truncated_two_port_records() {
    let missing_end = one_port_document("# Hz S RI", "", "1 0 0", "");
    assert!(matches!(
        parse_touchstone_v2_0_s_full(&missing_end),
        Err(Error::V2Structural { .. })
    ));
    let end_args = one_port_document("# Hz S RI", "", "1 0 0", "[End] extra");
    assert!(matches!(
        parse_touchstone_v2_0_s_full(&end_args),
        Err(Error::V2Structural { .. })
    ));
    let duplicate_end = one_port_document("# Hz S RI", "", "1 0 0", "[End]\n[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s_full(&duplicate_end),
        Err(Error::V2Structural { .. })
    ));
    let non_ascii = one_port_document("# Hz S RI", "", "1 0 0 é", "[End]");
    assert!(matches!(
        parse_touchstone_v2_0_s_full(&non_ascii),
        Err(Error::UnsupportedCharacter { .. })
    ));

    let truncated = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n[Network Data]\n1 0 0 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s_full(truncated),
        Err(Error::V2Count { .. })
    ));
}

#[test]
fn rejects_v2_exclusions_order_count_and_trailing_content() {
    let missing_order = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 2\n[Number of Frequencies] 1\n[Network Data]\n1 0 0 0 0 0 0 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s_full(missing_order),
        Err(Error::V2Structural { .. })
    ));

    let lower = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Matrix Format] Lower\n[Network Data]\n1 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s_full(lower),
        Err(Error::V2Unsupported { .. })
    ));

    let noise = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Number of Noise Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s_full(noise),
        Err(Error::V2Unsupported { .. })
    ));

    let wrong_count = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 2\n[Network Data]\n1 0 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s_full(wrong_count),
        Err(Error::V2Count { .. })
    ));

    let trailing = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n1 0 0\n";
    assert!(matches!(
        parse_touchstone_v2_0_s_full(trailing),
        Err(Error::V2Structural { .. })
    ));
}

#[test]
fn rejects_hfss_semantic_comments_and_pathological_dimensions_without_panic() {
    let hfss = "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 0 0 ! Port Impedance 50 0\n[End]\n";
    assert!(matches!(
        parse_touchstone_v2_0_s_full(hfss),
        Err(Error::UnsupportedExtension { .. })
    ));

    let overflowing_ports = format!(
        "[Version] 2.0\n# Hz S RI\n[Number of Ports] {}\n[Number of Frequencies] 1\n[Network Data]\n1\n[End]\n",
        usize::MAX
    );
    let overflow_result =
        std::panic::catch_unwind(|| parse_touchstone_v2_0_s_full(&overflowing_ports));
    assert!(matches!(
        overflow_result,
        Ok(Err(Error::V2SizeOverflow { .. }))
    ));

    let large_but_representable_ports = ((usize::MAX as f64).sqrt() as usize) / 2;
    let large_ports = format!(
        "[Version] 2.0\n# Hz S RI\n[Number of Ports] {large_but_representable_ports}\n[Number of Frequencies] 1\n[Network Data]\n1 0 0\n[End]\n"
    );
    let large_result = std::panic::catch_unwind(|| parse_touchstone_v2_0_s_full(&large_ports));
    assert!(matches!(large_result, Ok(Err(Error::V2Count { .. }))));

    let huge_frequency_count = format!(
        "[Version] 2.0\n# Hz S RI\n[Number of Ports] 1\n[Number of Frequencies] {}\n[Network Data]\n1 0 0\n[End]\n",
        usize::MAX
    );
    let frequency_result =
        std::panic::catch_unwind(|| parse_touchstone_v2_0_s_full(&huge_frequency_count));
    assert!(matches!(frequency_result, Ok(Err(Error::V2Count { .. }))));
}
