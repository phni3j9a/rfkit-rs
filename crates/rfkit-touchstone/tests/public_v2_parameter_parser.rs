use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{ConversionStage, Error as CoreError, Frequency, Network};
use rfkit_touchstone::{Error as TouchstoneError, parse_touchstone_v2_0_power};

fn one_port(kind: &str, reference: f64, data: &str, format: &str) -> String {
    one_port_with_reference(kind, 50.0, Some(reference), data, format)
}

fn one_port_with_reference(
    kind: &str,
    option_reference: f64,
    reference: Option<f64>,
    data: &str,
    format: &str,
) -> String {
    let reference_line =
        reference.map_or_else(String::new, |value| format!("[Reference] {value}\n"));
    format!(
        "[Version] 2.0\n# GHz {kind} {format} R {option_reference}\n[Number of Ports] 1\n[Number of Frequencies] 1\n{reference_line}[Network Data]\n1 {data}\n[End]\n"
    )
}

#[test]
fn parses_analytical_one_port_z_and_y_at_declared_non_50_references() {
    let z_network = parse_touchstone_v2_0_power(&one_port("Z", 25.0, "100 0", "RI"))
        .expect("physical Z input parses");
    assert_eq!(z_network.z0()[[0, 0]], Complex64::new(25.0, 0.0));
    assert!((z_network.s()[[0, 0, 0]].re - 0.6).abs() < 1.0e-14);
    assert!(z_network.s()[[0, 0, 0]].im.abs() < 1.0e-14);
    assert!((z_network.to_z_power().unwrap()[[0, 0, 0]].re - 100.0).abs() < 1.0e-12);

    let y_network = parse_touchstone_v2_0_power(&one_port("Y", 75.0, "0.01 0", "RI"))
        .expect("physical Y input parses");
    assert_eq!(y_network.z0()[[0, 0]], Complex64::new(75.0, 0.0));
    assert!((y_network.s()[[0, 0, 0]].re - (1.0 / 7.0)).abs() < 1.0e-14);
    assert!(y_network.s()[[0, 0, 0]].im.abs() < 1.0e-14);
    assert!((y_network.to_y_direct_power().unwrap()[[0, 0, 0]].re - 0.01).abs() < 1.0e-14);
}

#[test]
fn declared_reference_changes_output_coordinates_without_rescaling_physical_z_or_y() {
    let z_at_25 = parse_touchstone_v2_0_power(&one_port("Z", 25.0, "100 0", "RI"))
        .expect("Z at 25 ohm parses");
    let z_at_75 = parse_touchstone_v2_0_power(&one_port("Z", 75.0, "100 0", "RI"))
        .expect("Z at 75 ohm parses");
    assert_ne!(z_at_25.s()[[0, 0, 0]], z_at_75.s()[[0, 0, 0]]);
    assert!(
        (z_at_25.to_z_power().unwrap()[[0, 0, 0]] - z_at_75.to_z_power().unwrap()[[0, 0, 0]])
            .norm()
            < 1.0e-12
    );

    let y_at_25 = parse_touchstone_v2_0_power(&one_port("Y", 25.0, "0.01 0", "RI"))
        .expect("Y at 25 ohm parses");
    let y_at_75 = parse_touchstone_v2_0_power(&one_port("Y", 75.0, "0.01 0", "RI"))
        .expect("Y at 75 ohm parses");
    assert_ne!(y_at_25.s()[[0, 0, 0]], y_at_75.s()[[0, 0, 0]]);
    assert!(
        (y_at_25.to_y_direct_power().unwrap()[[0, 0, 0]]
            - y_at_75.to_y_direct_power().unwrap()[[0, 0, 0]])
        .norm()
            < 1.0e-12
    );
}

#[test]
fn reference_vector_overrides_option_r_and_option_r_only_sets_output_coordinates() {
    let z_at_25 =
        parse_touchstone_v2_0_power(&one_port_with_reference("Z", 25.0, None, "100 0", "RI"))
            .expect("Z with option R parses");
    let z_at_75 =
        parse_touchstone_v2_0_power(&one_port_with_reference("Z", 75.0, None, "100 0", "RI"))
            .expect("Z with changed option R parses");
    let z_override = parse_touchstone_v2_0_power(&one_port_with_reference(
        "Z",
        25.0,
        Some(75.0),
        "100 0",
        "RI",
    ))
    .expect("Z with [Reference] override parses");
    assert_eq!(z_at_25.z0()[[0, 0]], Complex64::new(25.0, 0.0));
    assert_eq!(z_at_75.z0()[[0, 0]], Complex64::new(75.0, 0.0));
    assert_eq!(z_override.z0()[[0, 0]], Complex64::new(75.0, 0.0));
    assert_ne!(z_at_25.s()[[0, 0, 0]], z_at_75.s()[[0, 0, 0]]);
    assert_eq!(z_override.s()[[0, 0, 0]], z_at_75.s()[[0, 0, 0]]);
    assert!((z_at_25.to_z_power().unwrap()[[0, 0, 0]] - 100.0).norm() < 1.0e-12);
    assert!((z_at_75.to_z_power().unwrap()[[0, 0, 0]] - 100.0).norm() < 1.0e-12);

    let y_at_25 =
        parse_touchstone_v2_0_power(&one_port_with_reference("Y", 25.0, None, "0.01 0", "RI"))
            .expect("Y with option R parses");
    let y_at_75 =
        parse_touchstone_v2_0_power(&one_port_with_reference("Y", 75.0, None, "0.01 0", "RI"))
            .expect("Y with changed option R parses");
    let y_override = parse_touchstone_v2_0_power(&one_port_with_reference(
        "Y",
        25.0,
        Some(75.0),
        "0.01 0",
        "RI",
    ))
    .expect("Y with [Reference] override parses");
    assert_eq!(y_at_25.z0()[[0, 0]], Complex64::new(25.0, 0.0));
    assert_eq!(y_at_75.z0()[[0, 0]], Complex64::new(75.0, 0.0));
    assert_eq!(y_override.z0()[[0, 0]], Complex64::new(75.0, 0.0));
    assert_ne!(y_at_25.s()[[0, 0, 0]], y_at_75.s()[[0, 0, 0]]);
    assert_eq!(y_override.s()[[0, 0, 0]], y_at_75.s()[[0, 0, 0]]);
    assert!((y_at_25.to_y_direct_power().unwrap()[[0, 0, 0]] - 0.01).norm() < 1.0e-14);
    assert!((y_at_75.to_y_direct_power().unwrap()[[0, 0, 0]] - 0.01).norm() < 1.0e-14);
}

#[test]
fn accepts_asymmetric_three_port_full_z_with_unequal_references() {
    let input = "[Version] 2.0\n# GHz Z RI R 50\n[Number of Ports] 3\n[Number of Frequencies] 1\n[Reference] 25 50 75\n[Matrix Format] Full\n[Network Data]\n1 10 1 2 -3 4 5 6 -7 8 9 10 -11 12 13 14 -15 16 -17\n[End]\n";
    let actual = parse_touchstone_v2_0_power(input).expect("asymmetric Z input parses");

    let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
    let z = Array3::from_shape_vec(
        (1, 3, 3),
        vec![
            Complex64::new(10.0, 1.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(4.0, 5.0),
            Complex64::new(6.0, -7.0),
            Complex64::new(8.0, 9.0),
            Complex64::new(10.0, -11.0),
            Complex64::new(12.0, 13.0),
            Complex64::new(14.0, -15.0),
            Complex64::new(16.0, -17.0),
        ],
    )
    .unwrap();
    let z0 = Array2::from_shape_vec(
        (1, 3),
        vec![
            Complex64::new(25.0, 0.0),
            Complex64::new(50.0, 0.0),
            Complex64::new(75.0, 0.0),
        ],
    )
    .unwrap();
    let expected = Network::from_z_power(frequency, z, z0).unwrap();
    assert_eq!(actual.frequency(), expected.frequency());
    assert_eq!(actual.z0(), expected.z0());
    for (actual, expected) in actual.s().iter().zip(expected.s().iter()) {
        assert!((*actual - *expected).norm() < 1.0e-13);
    }
    for (actual, expected) in actual.to_z_power().unwrap().iter().zip([
        Complex64::new(10.0, 1.0),
        Complex64::new(2.0, -3.0),
        Complex64::new(4.0, 5.0),
        Complex64::new(6.0, -7.0),
        Complex64::new(8.0, 9.0),
        Complex64::new(10.0, -11.0),
        Complex64::new(12.0, 13.0),
        Complex64::new(14.0, -15.0),
        Complex64::new(16.0, -17.0),
    ]) {
        assert!((*actual - expected).norm() < 1.0e-12);
    }
}

#[test]
fn lower_and_upper_complex_physical_matrices_share_plain_transpose_expansion() {
    let lower = "[Version] 2.0\n# GHz Z RI\n[Number of Ports] 3\n[Number of Frequencies] 1\n[Reference] 25 50 75\n[Matrix Format] Lower\n[Network Data]\n1 10 1 2 3 20 2 4 5 6 7 30 3\n[End]\n";
    let upper = lower
        .replace("[Matrix Format] Lower", "[Matrix Format] Upper")
        .replace(
            "1 10 1 2 3 20 2 4 5 6 7 30 3",
            "1 10 1 2 3 4 5 20 2 6 7 30 3",
        );
    let lower = parse_touchstone_v2_0_power(lower).expect("lower Z input parses");
    let upper = parse_touchstone_v2_0_power(&upper).expect("upper Z input parses");
    assert_eq!(lower.frequency(), upper.frequency());
    assert_eq!(lower.z0(), upper.z0());
    assert_eq!(lower.s(), upper.s());
    assert_eq!(lower.to_z_power().unwrap(), upper.to_z_power().unwrap());
    let recovered = lower.to_z_power().unwrap();
    assert!((recovered[[0, 0, 1]] - Complex64::new(2.0, 3.0)).norm() < 1.0e-12);
    assert!((recovered[[0, 1, 0]] - Complex64::new(2.0, 3.0)).norm() < 1.0e-12);
}

#[test]
fn direct_y_ingress_accepts_zero_admittance_and_zero_z_ingress_is_finite() {
    let zero_y = parse_touchstone_v2_0_power(&one_port("Y", 75.0, "0 0", "RI"))
        .expect("zero Y is accepted by the direct path");
    assert_eq!(zero_y.s()[[0, 0, 0]], Complex64::new(1.0, 0.0));

    let zero_z =
        parse_touchstone_v2_0_power(&one_port("Z", 75.0, "0 0", "RI")).expect("zero Z is accepted");
    assert_eq!(zero_z.s()[[0, 0, 0]], Complex64::new(-1.0, 0.0));
}

#[test]
fn direct_y_ingress_accepts_nonzero_rank_deficient_matrix() {
    let input = "[Version] 2.0\n# GHz Y RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n[Reference] 50 75\n[Matrix Format] Full\n[Network Data]\n1 0.01 0.001 0.02 0.002 0.002 0.0002 0.004 0.0004\n[End]\n";
    let network = parse_touchstone_v2_0_power(input).expect("rank-deficient Y parses directly");
    let recovered = network
        .to_y_direct_power()
        .expect("direct Y extraction succeeds");
    assert!((recovered[[0, 0, 0]] - Complex64::new(0.01, 0.001)).norm() < 1.0e-12);
    assert!((recovered[[0, 0, 1]] - Complex64::new(0.02, 0.002)).norm() < 1.0e-12);
    assert!((recovered[[0, 1, 0]] - Complex64::new(0.002, 0.0002)).norm() < 1.0e-12);
    assert!((recovered[[0, 1, 1]] - Complex64::new(0.004, 0.0004)).norm() < 1.0e-12);
}

#[test]
fn physical_parameter_two_port_orders_decode_to_the_same_matrix() {
    for (kind, values) in [
        ("Z", ["1 0", "2 0", "3 0", "4 0"]),
        ("Y", ["0.01 0", "0.02 0", "0.03 0", "0.04 0"]),
    ] {
        let natural = format!(
            "[Version] 2.0\n# GHz {kind} RI\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n[Reference] 50 75\n[Matrix Format] Full\n[Network Data]\n1 {} {} {} {}\n[End]\n",
            values[0], values[1], values[2], values[3]
        );
        let legacy = format!(
            "[Version] 2.0\n# GHz {kind} RI\n[Number of Ports] 2\n[Two-Port Data Order] 21_12\n[Number of Frequencies] 1\n[Reference] 50 75\n[Matrix Format] Full\n[Network Data]\n1 {} {} {} {}\n[End]\n",
            values[0], values[2], values[1], values[3]
        );
        let natural = parse_touchstone_v2_0_power(&natural).expect("natural order parses");
        let legacy = parse_touchstone_v2_0_power(&legacy).expect("legacy order parses");
        assert_eq!(natural.s(), legacy.s());
        let (natural_parameter, legacy_parameter) = if kind == "Z" {
            (natural.to_z_power().unwrap(), legacy.to_z_power().unwrap())
        } else {
            (
                natural.to_y_direct_power().unwrap(),
                legacy.to_y_direct_power().unwrap(),
            )
        };
        assert_eq!(natural_parameter, legacy_parameter);
    }
}

#[test]
fn y_triangular_records_use_plain_transpose_before_direct_conversion() {
    let lower = "[Version] 2.0\n# GHz Y RI\n[Number of Ports] 3\n[Number of Frequencies] 1\n[Reference] 25 50 75\n[Matrix Format] Lower\n[Network Data]\n1 0.01 0.001 0.002 0.003 0.02 0.002 0.004 0.005 0.006 0.007 0.03 0.003\n[End]\n";
    let upper = "[Version] 2.0\n# GHz Y RI\n[Number of Ports] 3\n[Number of Frequencies] 1\n[Reference] 25 50 75\n[Matrix Format] Upper\n[Network Data]\n1 0.01 0.001 0.002 0.003 0.004 0.005 0.02 0.002 0.006 0.007 0.03 0.003\n[End]\n";
    let lower = parse_touchstone_v2_0_power(lower).expect("lower Y input parses");
    let upper = parse_touchstone_v2_0_power(upper).expect("upper Y input parses");
    assert_eq!(lower.s(), upper.s());
    let lower_y = lower
        .to_y_direct_power()
        .expect("lower Y extraction succeeds");
    let upper_y = upper
        .to_y_direct_power()
        .expect("upper Y extraction succeeds");
    assert_eq!(lower_y, upper_y);
    assert!((lower_y[[0, 0, 1]] - Complex64::new(0.002, 0.003)).norm() < 1.0e-12);
    assert!((lower_y[[0, 1, 0]] - Complex64::new(0.002, 0.003)).norm() < 1.0e-12);
    assert!((lower_y[[0, 2, 1]] - Complex64::new(0.006, 0.007)).norm() < 1.0e-12);
}

#[test]
fn parser_preserves_core_singularity_and_nonfinite_arithmetic_context() {
    let z_singular =
        parse_touchstone_v2_0_power(&one_port_with_reference("Z", 50.0, None, "-50 0", "RI"));
    assert!(matches!(
        z_singular,
        Err(TouchstoneError::Core(CoreError::Singular {
            stage: ConversionStage::ZToS,
            frequency: 0,
            pivot: 0,
        }))
    ));

    let y_singular =
        parse_touchstone_v2_0_power(&one_port_with_reference("Y", 50.0, None, "-0.02 0", "RI"));
    assert!(matches!(
        y_singular,
        Err(TouchstoneError::Core(CoreError::Singular {
            stage: ConversionStage::YToS,
            frequency: 0,
            pivot: 0,
        }))
    ));

    let z_near = parse_touchstone_v2_0_power(&one_port_with_reference(
        "Z",
        50.0,
        None,
        "-49.9999999999 0",
        "RI",
    ))
    .expect("finite near-singular Z remains in the exact-pivot domain");
    assert!(z_near.s()[[0, 0, 0]].re.is_finite());
    assert!(z_near.s()[[0, 0, 0]].norm() > 1.0e8);

    let y_near = parse_touchstone_v2_0_power(&one_port_with_reference(
        "Y",
        50.0,
        None,
        "-0.0199999999 0",
        "RI",
    ))
    .expect("finite near-singular Y remains in the exact-pivot domain");
    assert!(y_near.s()[[0, 0, 0]].re.is_finite());
    assert!(y_near.s()[[0, 0, 0]].norm() > 1.0e8);

    let max = "1.7976931348623157e308";
    let z_overflow = parse_touchstone_v2_0_power(&format!(
        "[Version] 2.0\n# GHz Z RI R {max}\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 {max} 0\n[End]\n"
    ));
    assert!(matches!(
        z_overflow,
        Err(TouchstoneError::Core(CoreError::NonFiniteComputation {
            stage: ConversionStage::ZToS,
            frequency: 0,
            ..
        }))
    ));

    let y_overflow = parse_touchstone_v2_0_power(&format!(
        "[Version] 2.0\n# GHz Y RI R {max}\n[Number of Ports] 1\n[Number of Frequencies] 1\n[Network Data]\n1 {max} 0\n[End]\n"
    ));
    assert!(matches!(
        y_overflow,
        Err(TouchstoneError::Core(CoreError::NonFiniteComputation {
            stage: ConversionStage::YToS,
            frequency: 0,
            ..
        }))
    ));
}

#[test]
fn physical_parameters_use_the_same_ri_ma_db_decoder() {
    let z_ma = parse_touchstone_v2_0_power(&one_port("Z", 25.0, "100 0", "MA"))
        .expect("MA Z input parses");
    assert!((z_ma.s()[[0, 0, 0]].re - 0.6).abs() < 1.0e-14);

    let y_db = parse_touchstone_v2_0_power(&one_port("Y", 50.0, "-40 0", "DB"))
        .expect("DB Y input parses");
    assert!((y_db.to_y_direct_power().unwrap()[[0, 0, 0]].re - 0.01).abs() < 1.0e-14);
}
