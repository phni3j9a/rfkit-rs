use num_complex::Complex64;
use rfkit_touchstone::{
    Error, TouchstoneParameter, parse_touchstone_v2_0_power, write_touchstone_v2_0_power_full_ri_hz,
};

const INPUT_A: &str = "[Version] 2.0\n\
# Hz S RI R 35\n\
[Number of Ports] 3\n\
[Number of Frequencies] 2\n\
[Reference] 35 50 72\n\
[Matrix Format] Full\n\
[Network Data]\n\
1000000000 0.10 0.01 0.20 0.02 0.30 0.03 0.40 0.04 0.50 0.05 0.60 0.06 0.70 0.07 0.80 0.08 0.90 0.09\n\
2000000000 0.11 0.02 0.21 0.03 0.31 0.04 0.41 0.05 0.51 0.06 0.61 0.07 0.71 0.08 0.81 0.09 0.91 0.10\n\
[End]\n";

const INPUT_B: &str = "[Version] 2.0\n\
# Hz S RI R 35\n\
[Number of Ports] 3\n\
[Number of Frequencies] 3\n\
[Reference] 35 50 72\n\
[Matrix Format] Full\n\
[Network Data]\n\
3000000000 0.40 0.04 0.50 0.05 0.60 0.06 0.70 0.07 0.80 0.08 0.90 0.09 1.00 0.10 1.10 0.11 1.20 0.12\n\
4000000000 0.41 0.05 0.51 0.06 0.61 0.07 0.71 0.08 0.81 0.09 0.91 0.10 1.01 0.11 1.11 0.12 1.21 0.13\n\
5000000000 0.42 0.06 0.52 0.07 0.62 0.08 0.72 0.09 0.82 0.10 0.92 0.11 1.02 0.12 1.12 0.13 1.22 0.14\n\
[End]\n";

const INPUT_B_DUPLICATE: &str = "[Version] 2.0\n\
# Hz S RI R 35\n\
[Number of Ports] 3\n\
[Number of Frequencies] 2\n\
[Reference] 35 50 72\n\
[Matrix Format] Full\n\
[Network Data]\n\
2000000000 0.40 0.04 0.50 0.05 0.60 0.06 0.70 0.07 0.80 0.08 0.90 0.09 1.00 0.10 1.10 0.11 1.20 0.12\n\
3000000000 0.41 0.05 0.51 0.06 0.61 0.07 0.71 0.08 0.81 0.09 0.91 0.10 1.01 0.11 1.11 0.12 1.21 0.13\n\
[End]\n";

const INPUT_B_NONMONOTONIC: &str = "[Version] 2.0\n\
# Hz S RI R 35\n\
[Number of Ports] 3\n\
[Number of Frequencies] 1\n\
[Reference] 35 50 72\n\
[Matrix Format] Full\n\
[Network Data]\n\
1500000000 0.40 0.04 0.50 0.05 0.60 0.06 0.70 0.07 0.80 0.08 0.90 0.09 1.00 0.10 1.10 0.11 1.20 0.12\n\
[End]\n";

const INPUT_B_VARYING_REFERENCE: &str = "[Version] 2.0\n\
# Hz S RI R 35\n\
[Number of Ports] 3\n\
[Number of Frequencies] 2\n\
[Reference] 35 51 72\n\
[Matrix Format] Full\n\
[Network Data]\n\
3000000000 0.40 0.04 0.50 0.05 0.60 0.06 0.70 0.07 0.80 0.08 0.90 0.09 1.00 0.10 1.10 0.11 1.20 0.12\n\
4000000000 0.41 0.05 0.51 0.06 0.61 0.07 0.71 0.08 0.81 0.09 0.91 0.10 1.01 0.11 1.11 0.12 1.21 0.13\n\
[End]\n";

fn assert_writer_frequency_error(result: rfkit_touchstone::Result<String>) {
    assert!(matches!(
        result,
        Err(Error::WriterV2FrequencyNotStrictlyIncreasing { .. })
    ));
}

#[test]
fn measured_segments_concatenate_select_analyze_and_round_trip_as_v2_full() {
    let source_a = parse_touchstone_v2_0_power(INPUT_A).expect("A segment parses");
    let source_b = parse_touchstone_v2_0_power(INPUT_B).expect("B segment parses");
    assert_eq!(source_a.nports(), 3);
    assert_eq!(source_b.nports(), 3);
    assert_eq!(
        source_a.z0().row(0).to_vec(),
        vec![
            Complex64::new(35.0, 0.0),
            Complex64::new(50.0, 0.0),
            Complex64::new(72.0, 0.0),
        ]
    );
    assert_eq!(source_b.z0().row(2), source_a.z0().row(0));

    let assembled = source_a
        .concatenate_frequency_samples(&source_b)
        .expect("same-port-count measured segments concatenate");
    assert_eq!(
        assembled.frequency().hz(),
        &[1.0e9, 2.0e9, 3.0e9, 4.0e9, 5.0e9]
    );
    let expected_s00 = [0.10, 0.11, 0.40, 0.41, 0.42];
    let expected_s12 = [0.60, 0.61, 0.90, 0.91, 0.92];
    let expected_s22 = [0.90, 0.91, 1.20, 1.21, 1.22];
    let expected_s00_imag = [0.01, 0.02, 0.04, 0.05, 0.06];
    let expected_s12_imag = [0.06, 0.07, 0.09, 0.10, 0.11];
    let expected_s22_imag = [0.09, 0.10, 0.12, 0.13, 0.14];
    for frequency in 0..5 {
        assert_eq!(
            assembled.s()[[frequency, 0, 0]],
            Complex64::new(expected_s00[frequency], expected_s00_imag[frequency])
        );
        assert_eq!(
            assembled.s()[[frequency, 1, 2]],
            Complex64::new(expected_s12[frequency], expected_s12_imag[frequency])
        );
        assert_eq!(
            assembled.s()[[frequency, 2, 2]],
            Complex64::new(expected_s22[frequency], expected_s22_imag[frequency])
        );
        assert_eq!(
            assembled.z0().row(frequency).to_vec(),
            vec![
                Complex64::new(35.0, 0.0),
                Complex64::new(50.0, 0.0),
                Complex64::new(72.0, 0.0),
            ]
        );
    }

    // Selection is explicit: it chooses stored observations rather than
    // asking concatenation to infer a target grid.  Analysis consumes the
    // selected network as a three-sample frequency-aligned result.
    let selected = assembled
        .select_frequency_samples(&[0, 2, 4])
        .expect("selected observations are valid");
    assert_eq!(selected.frequency().hz(), &[1.0e9, 3.0e9, 5.0e9]);
    let sigma_max = selected
        .max_singular_value_power()
        .expect("selected network reaches existing analysis");
    assert_eq!(sigma_max.len(), 3);
    assert!(sigma_max.iter().all(|value| value.is_finite()));

    let text = write_touchstone_v2_0_power_full_ri_hz(&assembled, TouchstoneParameter::S)
        .expect("separated finite bands with constant per-port references write");
    assert!(text.contains("[Matrix Format] Full\n"));
    assert!(text.contains("[Reference] 35 50 72\n"));
    let reread = parse_touchstone_v2_0_power(&text).expect("written v2 Full data rereads");
    assert_eq!(reread.frequency(), assembled.frequency());
    assert_eq!(reread.s(), assembled.s());
    assert_eq!(reread.z0(), assembled.z0());
}

#[test]
fn concatenation_does_not_repair_writer_grid_or_reference_diagnostics() {
    let source_a = parse_touchstone_v2_0_power(INPUT_A).expect("A segment parses");

    let duplicate = source_a
        .concatenate_frequency_samples(
            &parse_touchstone_v2_0_power(INPUT_B_DUPLICATE).expect("duplicate segment parses"),
        )
        .expect("structural concatenation accepts duplicate labels");
    assert_writer_frequency_error(write_touchstone_v2_0_power_full_ri_hz(
        &duplicate,
        TouchstoneParameter::S,
    ));

    let nonmonotonic = source_a
        .concatenate_frequency_samples(
            &parse_touchstone_v2_0_power(INPUT_B_NONMONOTONIC)
                .expect("nonmonotonic segment parses"),
        )
        .expect("structural concatenation accepts nonmonotonic labels");
    assert_writer_frequency_error(write_touchstone_v2_0_power_full_ri_hz(
        &nonmonotonic,
        TouchstoneParameter::S,
    ));

    let varying_reference = source_a
        .concatenate_frequency_samples(
            &parse_touchstone_v2_0_power(INPUT_B_VARYING_REFERENCE)
                .expect("varying-reference segment parses"),
        )
        .expect("structural concatenation accepts changing references");
    assert!(matches!(
        write_touchstone_v2_0_power_full_ri_hz(&varying_reference, TouchstoneParameter::S),
        Err(Error::WriterV2MismatchedZ0 { .. })
    ));
}
