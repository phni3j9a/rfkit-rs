use num_complex::Complex64;
use rfkit_touchstone::{Error, ExtensionKind, parse_touchstone_v1_0_s};

fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
    assert!(
        (actual - expected).norm() <= tolerance,
        "actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn parses_units_formats_comments_defaults_and_line_endings() {
    let ri = "! header\r# RI s R 73.5 MHz\r10 0.25 -0.5 ! inline\r20 0.5 0.25\r";
    let network = parse_touchstone_v1_0_s(ri, 1).unwrap();
    assert_eq!(network.frequency().hz(), &[10.0e6, 20.0e6]);
    assert_eq!(network.z0()[[0, 0]], Complex64::new(73.5, 0.0));
    assert_eq!(network.s()[[0, 0, 0]], Complex64::new(0.25, -0.5));

    // The option-line tokens are case-insensitive, can be reordered, and
    // omitted tokens take the v1 defaults (GHz/S/MA/50 ohm).
    let defaults = parse_touchstone_v1_0_s("#\n1 0.5 0\n", 1).unwrap();
    assert_eq!(defaults.frequency().hz(), &[1.0e9]);
    assert_eq!(defaults.z0()[[0, 0]], Complex64::new(50.0, 0.0));
    assert_eq!(defaults.s()[[0, 0, 0]], Complex64::new(0.5, 0.0));

    for (unit, multiplier) in [("Hz", 1.0), ("kHz", 1.0e3), ("MHz", 1.0e6), ("GHz", 1.0e9)] {
        let network = parse_touchstone_v1_0_s(&format!("# {unit} S RI\n2 0.1 0\n",), 1).unwrap();
        assert_eq!(network.frequency().hz(), &[2.0 * multiplier]);
    }

    let line_ending_input = "# MHz S RI\n10 0.25 -0.5\n20 0.5 0.25\n";
    for line_ending in ["\n", "\r\n", "\r"] {
        let text = line_ending_input.replace('\n', line_ending);
        let network = parse_touchstone_v1_0_s(&text, 1).unwrap();
        assert_eq!(network.frequency().hz(), &[10.0e6, 20.0e6]);
    }
}

#[test]
fn equivalent_ri_ma_db_representations_match() {
    let ri = parse_touchstone_v1_0_s("# GHz S RI R 61\n1 0.3 0.4\n", 1).unwrap();
    let ma = parse_touchstone_v1_0_s("# GHz S MA R 61\n1 0.5 53.130102354\n", 1).unwrap();
    let db = parse_touchstone_v1_0_s("# GHz S DB R 61\n1 -6.020599913 53.130102354\n", 1).unwrap();
    let expected = ri.s()[[0, 0, 0]];
    assert_complex_close(ma.s()[[0, 0, 0]], expected, 1.0e-10);
    assert_complex_close(db.s()[[0, 0, 0]], expected, 1.0e-10);
}

#[test]
fn parses_legacy_two_port_and_row_major_four_port_order() {
    let two = parse_touchstone_v1_0_s("# Hz S RI R 90\n100 1 0 2 0 3 0 4 0\n", 2).unwrap();
    assert_eq!(two.s()[[0, 0, 0]], Complex64::new(1.0, 0.0));
    assert_eq!(two.s()[[0, 0, 1]], Complex64::new(3.0, 0.0));
    assert_eq!(two.s()[[0, 1, 0]], Complex64::new(2.0, 0.0));
    assert_eq!(two.s()[[0, 1, 1]], Complex64::new(4.0, 0.0));

    let four = parse_touchstone_v1_0_s(
        "# GHz S RI R 61\n1 11 0 12 0 13 0 14 0\n21 0 22 0 23 0 24 0\n31 0 32 0 33 0 34 0\n41 0 42 0 43 0 44 0\n",
        4,
    )
    .unwrap();
    assert_eq!(four.s()[[0, 0, 3]], Complex64::new(14.0, 0.0));
    assert_eq!(four.s()[[0, 2, 1]], Complex64::new(32.0, 0.0));
    assert_eq!(four.s()[[0, 3, 0]], Complex64::new(41.0, 0.0));
}

#[test]
fn parses_five_port_v1_continuations_and_preserves_asymmetry() {
    let text = "# GHz S RI R 77\n\
        1 11 0 12 0 13 0 14 0\n\
        15 0\n\
        21 0 22 0 23 0 24 0\n\
        25 0\n\
        31 0 32 0 33 0 34 0\n\
        35 0\n\
        41 0 42 0 43 0 44 0\n\
        45 0\n\
        51 0 52 0 53 0 54 0\n\
        55 0\n";
    let network = parse_touchstone_v1_0_s(text, 5).unwrap();
    assert_eq!(network.s().dim(), (1, 5, 5));
    assert_eq!(network.s()[[0, 0, 4]], Complex64::new(15.0, 0.0));
    assert_eq!(network.s()[[0, 4, 0]], Complex64::new(51.0, 0.0));
    assert_eq!(network.s()[[0, 4, 3]], Complex64::new(54.0, 0.0));

    // v1 limits each physical line to four pairs but does not require a
    // row-start line to use all four; a 2+3 split is also a valid row.
    let split = parse_touchstone_v1_0_s(
        "# GHz S RI\n1 11 0 12 0\n13 0 14 0 15 0\n21 0 22 0 23 0 24 0\n25 0\n31 0 32 0 33 0 34 0\n35 0\n41 0 42 0 43 0 44 0\n45 0\n51 0 52 0 53 0 54 0\n55 0\n",
        5,
    )
    .unwrap();
    assert_eq!(split.s()[[0, 0, 1]], Complex64::new(12.0, 0.0));
    assert_eq!(split.s()[[0, 0, 4]], Complex64::new(15.0, 0.0));
}

#[test]
fn parsed_network_flows_into_existing_power_analysis() {
    let network = parse_touchstone_v1_0_s("# MHz S RI R 75\n10 0.2 0\n", 1).unwrap();
    let z = network.to_z_power().unwrap();
    assert_complex_close(z[[0, 0, 0]], Complex64::new(112.5, 0.0), 1.0e-12);
}

#[test]
fn rejects_malformed_unsupported_and_ambiguous_domain() {
    assert!(matches!(
        parse_touchstone_v1_0_s("1 0 0\n", 1),
        Err(Error::MissingOptionLine { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz Y RI R 50\n1 0 0\n", 1),
        Err(Error::UnsupportedParameter { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI R 50 75\n1 0 0\n", 1),
        Err(Error::MultipleReferenceValues { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI R 50 R 75\n1 0 0\n", 1),
        Err(Error::MultipleReferenceValues { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S MA R 50\n1 -0.1 0\n", 1),
        Err(Error::NegativeMagnitude { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI R 50\n1 NaN 0\n", 1),
        Err(Error::NonFiniteNumber { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI R 50\n1 not-a-number 0\n", 1),
        Err(Error::MalformedNumber { .. })
    ));
    match parse_touchstone_v1_0_s("# GHz S RI R 50\n    1 bad 0\n", 1) {
        Err(Error::MalformedNumber {
            line: 2,
            column: 7,
            token,
        }) => assert_eq!(token, "bad"),
        other => panic!("expected physical-line column 7, got {other:?}"),
    }
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI R 50\n1 1e999 0\n", 1),
        Err(Error::NumericOverflow { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S DB R 50\n1 1e999 0\n", 1),
        Err(Error::NumericOverflow { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S DB R 50\n1 10000 0\n", 1),
        Err(Error::ConversionOverflow {
            format: rfkit_touchstone::DataFormat::Db,
            ..
        })
    ));
    for reference in ["0", "-50", "NaN", "inf"] {
        assert!(matches!(
            parse_touchstone_v1_0_s(&format!("# GHz S RI R {reference}\n1 0 0\n"), 1,),
            Err(Error::InvalidReferenceResistance { .. })
        ));
    }
    for option in [
        "# GHz S RI R",
        "# THz S RI",
        "# GHz GHz S RI",
        "# GHz S RI MA",
        "# GHz S RI unknown",
    ] {
        assert!(matches!(
            parse_touchstone_v1_0_s(&format!("{option}\n1 0 0\n"), 1),
            Err(Error::MalformedOption { .. })
        ));
    }
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI S\n1 0 0\n", 1),
        Err(Error::MalformedOption { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI\n1 0 0 0 0\n", 1),
        Err(Error::SurplusRecord { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI\n! only a comment\n", 1),
        Err(Error::EmptyData)
    ));
    let scientific = parse_touchstone_v1_0_s("# MHz S RI\n1.0e1 2.5e-1 -3.0e-1\n", 1).unwrap();
    assert_eq!(scientific.frequency().hz(), &[10.0e6]);
    assert_eq!(scientific.s()[[0, 0, 0]], Complex64::new(2.5e-1, -3.0e-1));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI R 50\n1 0\n", 1),
        Err(Error::IncompleteRecord { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI R 50\n1 0 0\n! Terminal Data Exported\n", 1),
        Err(Error::UnsupportedExtension {
            extension: ExtensionKind::ReferenceImpedance,
            ..
        })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI R 50\n1 0 0 0 0 0 0 0 0\n2 0 0 0 0\n", 2),
        Err(Error::NoiseData { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI R 50\n1 0 0\n[Network Data]\n", 1),
        Err(Error::UnsupportedKeyword { .. })
    ));
}

#[test]
fn first_option_line_wins_and_port_count_is_checked() {
    let network = parse_touchstone_v1_0_s(
        "! comment\n# MHz S RI R 75\n# this is not a valid option line\n1 0.2 0\n",
        1,
    )
    .unwrap();
    assert_eq!(network.frequency().hz(), &[1.0e6]);
    assert_eq!(network.z0()[[0, 0]], Complex64::new(75.0, 0.0));

    assert!(matches!(
        parse_touchstone_v1_0_s("#\n1 0 0\n", 0),
        Err(Error::InvalidPortCount { nports: 0 })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("#\n1 0 0\n", usize::MAX),
        Err(Error::SizeOverflow { .. })
    ));
}

#[test]
fn rejects_all_documented_vendor_semantic_markers_but_keeps_port_name_comments() {
    let reference_markers = [
        "! Gamma 1 0",
        "! Port Impedance 50 0",
        "! Port Impedance0 50 0",
        "! Terminal data exported",
        "! Modal data exported",
    ];
    for marker in reference_markers {
        assert!(matches!(
            parse_touchstone_v1_0_s(&format!("# GHz S RI\n{marker}\n1 0 0\n"), 1,),
            Err(Error::UnsupportedExtension { .. })
        ));
    }

    for definition in ["power", "pseudo", "traveling"] {
        assert!(matches!(
            parse_touchstone_v1_0_s(
                &format!("# GHz S RI\n! S-parameter uses the {definition} definition\n1 0 0\n"),
                1,
            ),
            Err(Error::UnsupportedExtension {
                extension: ExtensionKind::WaveDefinition,
                ..
            })
        ));
    }

    let network = parse_touchstone_v1_0_s(
        "! Port[1] = input reference impedance\n# GHz S RI\n1 0 0\n",
        1,
    )
    .unwrap();
    assert_eq!(network.s()[[0, 0, 0]], Complex64::new(0.0, 0.0));

    let ordinary = parse_touchstone_v1_0_s(
        "# GHz S RI\n! Our S-parameter uses the power amplifier output\n1 0 0\n",
        1,
    )
    .unwrap();
    assert_eq!(ordinary.s()[[0, 0, 0]], Complex64::new(0.0, 0.0));
}

#[test]
fn rejects_invalid_matrix_line_boundaries_without_flattening_records() {
    // A 5-port row cannot put five pairs on one physical line.
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI\n1 11 0 12 0 13 0 14 0 15 0\n", 5,),
        Err(Error::SurplusRecord { .. })
    ));

    // Blank/comment lines are ignored while the row remains incomplete, but
    // the next data line is still required to finish that same row.
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI\n1 11 0\n! gap\n", 5),
        Err(Error::IncompleteRecord { .. })
    ));

    // 3-port rows are physical-line records and cannot be continued onto a
    // second line.
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI\n1 11 0 12 0\n13 0\n", 3),
        Err(Error::IncompleteRecord { .. })
    ));
}

#[test]
fn rejects_invalid_frequency_axes_and_scaled_overflow() {
    assert!(matches!(
        parse_touchstone_v1_0_s("# Hz S RI\n-1 0 0\n", 1),
        Err(Error::NegativeFrequency { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# Hz S RI\nNaN 0 0\n", 1),
        Err(Error::NonFiniteFrequency { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI\n1 0 0\n1 0 0\n", 1),
        Err(Error::FrequencyNotStrictlyIncreasing { .. })
    ));
    assert!(matches!(
        parse_touchstone_v1_0_s("# GHz S RI\n1e308 0 0\n", 1),
        Err(Error::FrequencyOverflow { .. })
    ));
}
