//! Concatenate two measured Touchstone 2.0 Full segments, analyze explicit
//! stored observations, and exercise the existing writer domain.

use num_complex::Complex64;
use rfkit_touchstone::{Error, parse_touchstone_v2_0_s, write_touchstone_v2_0_s_full_ri_hz};

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

fn expect_writer_frequency_error(result: rfkit_touchstone::Result<String>) {
    assert!(matches!(
        result,
        Err(Error::WriterV2FrequencyNotStrictlyIncreasing { .. })
    ));
}

fn main() -> rfkit_touchstone::Result<()> {
    let source_a = parse_touchstone_v2_0_s(INPUT_A)?;
    let source_b = parse_touchstone_v2_0_s(INPUT_B)?;
    let assembled = source_a.concatenate_frequency_samples(&source_b)?;
    assert_eq!(assembled.nports(), 3);
    assert_eq!(
        assembled.frequency().hz(),
        &[1.0e9, 2.0e9, 3.0e9, 4.0e9, 5.0e9]
    );
    assert_eq!(
        assembled.z0().row(0).to_vec(),
        vec![
            Complex64::new(35.0, 0.0),
            Complex64::new(50.0, 0.0),
            Complex64::new(72.0, 0.0),
        ]
    );
    assert_eq!(assembled.s()[[0, 0, 0]], Complex64::new(0.10, 0.01));
    assert_eq!(assembled.s()[[2, 0, 0]], Complex64::new(0.40, 0.04));
    assert_eq!(assembled.s()[[4, 2, 2]], Complex64::new(1.22, 0.14));

    let selected = assembled.select_frequency_samples(&[0, 2, 4])?;
    assert_eq!(selected.frequency().hz(), &[1.0e9, 3.0e9, 5.0e9]);
    let sigma_max = selected.max_singular_value_power()?;
    assert_eq!(sigma_max.len(), 3);
    assert!(sigma_max.iter().all(|value| value.is_finite()));

    let text = write_touchstone_v2_0_s_full_ri_hz(&assembled)?;
    assert!(text.contains("[Reference] 35 50 72\n"));
    let reread = parse_touchstone_v2_0_s(&text)?;
    assert_eq!(reread.frequency(), assembled.frequency());
    assert_eq!(reread.s(), assembled.s());
    assert_eq!(reread.z0(), assembled.z0());

    let duplicate =
        source_a.concatenate_frequency_samples(&parse_touchstone_v2_0_s(INPUT_B_DUPLICATE)?)?;
    expect_writer_frequency_error(write_touchstone_v2_0_s_full_ri_hz(&duplicate));
    let nonmonotonic =
        source_a.concatenate_frequency_samples(&parse_touchstone_v2_0_s(INPUT_B_NONMONOTONIC)?)?;
    expect_writer_frequency_error(write_touchstone_v2_0_s_full_ri_hz(&nonmonotonic));
    let varying_reference = source_a
        .concatenate_frequency_samples(&parse_touchstone_v2_0_s(INPUT_B_VARYING_REFERENCE)?)?;
    assert!(matches!(
        write_touchstone_v2_0_s_full_ri_hz(&varying_reference),
        Err(Error::WriterV2MismatchedZ0 { .. })
    ));

    print!("{text}");
    Ok(())
}
