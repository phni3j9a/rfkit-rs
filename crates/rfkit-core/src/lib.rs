//! Rust-native RF and microwave network analysis primitives.
//!
//! The project starts deliberately small: a trustworthy core data model first,
//! then numerical operations backed by differential tests against scikit-rf.
//!
//! The canonical [`Network`] owns frequency-major S-parameter data.  The
//! provisional parameter-ingress constructors [`Network::from_z_power`],
//! [`Network::from_y_via_z_power`], and [`Network::from_y_direct_power`]
//! accept owned frequency-major matrices in ohms and siemens respectively,
//! plus explicit complex reference impedances in ohms, and return that
//! canonical S representation.  They use Kurokawa
//! power-wave semantics; S itself is dimensionless.  The supplied frequency
//! samples and reference values are pointwise data and are retained exactly.
//! The constructors require a nonempty axis with a matching first dimension,
//! but do not impose finite, nonnegative, sorted, or unique sample policy.
//! Those restrictions belong to consuming formats or operations such as the
//! Touchstone writer.  No 50-ohm default, broadcasting, regularization,
//! pseudoinverse, cutoff, or fallback is applied.
//!
//! `from_y_via_z_power` is intentionally named for its composed Y→Z→S path:
//! singular or zero Y is rejected while forming the intermediate impedance.
//! `from_y_direct_power` instead solves the explicit Y→S wave equations and
//! accepts singular or zero Y whenever its conversion system is nonsingular.
//! `to_y_power` likewise remains the composed S→Z→Y operation, while
//! [`Network::to_y_direct_power`] exposes the direct S→Y equation for networks
//! whose `I-S` system is singular but whose direct system is nonsingular.
//! These constructors and conversion methods return structured crate-level
//! errors that retain Z/Y input kind, conversion stage, and matrix location
//! where the existing kernels provide that context.  Inputs are borrowed
//! during computation and are not mutated; the returned arrays are newly
//! owned.
//!
//! [`Network::renormalize_direct_power`] is the additive direct Kurokawa
//! wave-change path between explicit source and target references.  It solves
//! the direct right system without an intermediate Z/Y matrix, so singular
//! ideal-open and floating-network domains can be exported after changing to
//! a writer-compatible common reference.  [`Network::renormalize_power`]
//! remains the composed S→Z→S operation with its existing domain and errors.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

mod composition;
mod connection;
mod direct_connection;
mod impedance_admittance;
mod interpolation;
mod inverse_cascade;
mod linalg;
mod mixed_mode;
mod power_wave_admittance;
mod power_waves;
mod stability;
mod termination;

/// Errors produced while constructing or manipulating RF network data.
#[non_exhaustive]
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("frequency axis must not be empty")]
    EmptyFrequency,

    #[error("{stage} conversion frequency axis must not be empty")]
    EmptyConversionFrequency { stage: ConversionStage },

    #[error(
        "{stage} conversion frequency length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    ConversionFrequencyLengthMismatch {
        stage: ConversionStage,
        expected: usize,
        actual: usize,
    },

    #[error("{parameter} constructor frequency axis must not be empty")]
    EmptyParameterFrequency { parameter: ParameterKind },

    #[error(
        "{parameter} constructor frequency length does not match the parameter first axis: expected {expected}, got {actual}"
    )]
    ParameterFrequencyLengthMismatch {
        parameter: ParameterKind,
        expected: usize,
        actual: usize,
    },

    #[error("S-parameter shape must be (nfreq, nport, nport)")]
    InvalidSShape,

    #[error("reference-impedance shape must be (nfreq, nport)")]
    InvalidZ0Shape,

    #[error("port permutation frequency axis must not be empty")]
    EmptyPortPermutationFrequency,

    #[error(
        "port permutation frequency length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    PortPermutationFrequencyLengthMismatch { expected: usize, actual: usize },

    #[error("port permutation received an invalid S-parameter shape {shape:?}")]
    InvalidPortPermutationSShape { shape: Vec<usize> },

    #[error("port permutation received an invalid reference-impedance shape {shape:?}")]
    InvalidPortPermutationZ0Shape { shape: Vec<usize> },

    #[error(
        "port permutation length does not match the network port count: expected {expected}, got {actual}"
    )]
    PortPermutationLengthMismatch { expected: usize, actual: usize },

    #[error(
        "port permutation entry at position {position} refers to old port {port}, out of range for {nports} ports"
    )]
    PortPermutationOutOfRange {
        position: usize,
        port: usize,
        nports: usize,
    },

    #[error(
        "port permutation repeats old port {port} at positions {first_position} and {second_position}"
    )]
    PortPermutationDuplicate {
        port: usize,
        first_position: usize,
        second_position: usize,
    },

    #[error("{stage} conversion received an invalid {parameter} shape {shape:?}")]
    InvalidShape {
        stage: ConversionStage,
        parameter: ParameterKind,
        shape: Vec<usize>,
    },

    #[error(
        "{stage} conversion received a non-finite S-parameter at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteS {
        stage: ConversionStage,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "{stage} conversion received a non-finite Z-parameter at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteZ {
        stage: ConversionStage,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "{stage} conversion received a non-finite Y-parameter at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteY {
        stage: ConversionStage,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "{stage} conversion received a non-finite reference impedance at frequency {frequency}, port {port}"
    )]
    NonFiniteZ0 {
        stage: ConversionStage,
        frequency: usize,
        port: usize,
    },

    #[error(
        "{stage} conversion has a zero-real reference impedance at frequency {frequency}, port {port}"
    )]
    ZeroRealReferenceImpedance {
        stage: ConversionStage,
        frequency: usize,
        port: usize,
    },

    #[error(
        "{stage} conversion system is exactly singular at frequency {frequency}, pivot {pivot}"
    )]
    Singular {
        stage: ConversionStage,
        frequency: usize,
        pivot: usize,
    },

    #[error(
        "{stage} conversion produced a non-finite computation at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteComputation {
        stage: ConversionStage,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("direct power-wave renormalization frequency axis must not be empty")]
    EmptyDirectRenormalizationFrequency,

    #[error(
        "direct power-wave renormalization frequency length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    DirectRenormalizationFrequencyLengthMismatch { expected: usize, actual: usize },

    #[error("direct power-wave renormalization received an invalid S-parameter shape {shape:?}")]
    InvalidDirectRenormalizationSShape { shape: Vec<usize> },

    #[error(
        "direct power-wave renormalization received an invalid {reference} reference-impedance shape {shape:?}"
    )]
    InvalidDirectRenormalizationZ0Shape {
        reference: DirectRenormalizationReference,
        shape: Vec<usize>,
    },

    #[error(
        "direct power-wave renormalization received a non-finite S-parameter at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteDirectRenormalizationS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "direct power-wave renormalization received a non-finite {reference} reference impedance at frequency {frequency}, port {port}"
    )]
    NonFiniteDirectRenormalizationZ0 {
        reference: DirectRenormalizationReference,
        frequency: usize,
        port: usize,
    },

    #[error(
        "direct power-wave renormalization received a zero-real {reference} reference impedance at frequency {frequency}, port {port}"
    )]
    ZeroRealDirectRenormalizationReferenceImpedance {
        reference: DirectRenormalizationReference,
        frequency: usize,
        port: usize,
    },

    #[error(
        "direct power-wave renormalization system is exactly singular at frequency {frequency}, pivot {pivot}"
    )]
    SingularDirectRenormalization { frequency: usize, pivot: usize },

    #[error(
        "direct power-wave renormalization produced a non-finite computation at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteDirectRenormalizationComputation {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("inverse-cascade frequency axis must not be empty")]
    EmptyInverseCascadeFrequency,

    #[error(
        "inverse-cascade frequency length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    InverseCascadeFrequencyLengthMismatch { expected: usize, actual: usize },

    #[error(
        "inverse-cascade requires a square positive even-port S-parameter shape, got {shape:?}"
    )]
    InvalidInverseCascadeSShape { shape: Vec<usize> },

    #[error("inverse-cascade reference-impedance shape must be (nfreq, nport), got {shape:?}")]
    InvalidInverseCascadeZ0Shape { shape: Vec<usize> },

    #[error("inverse-cascade requires an even positive port count, got {nports}")]
    InvalidInverseCascadePortCount { nports: usize },

    #[error("inverse-cascade frequency is non-finite at index {index}: {value:?}")]
    NonFiniteInverseCascadeFrequency { index: usize, value: f64 },

    #[error(
        "inverse-cascade {stage} input S-parameter is non-finite at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteInverseCascadeS {
        stage: InverseCascadeStage,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "inverse-cascade reference impedance is non-finite at frequency {frequency}, port {port}"
    )]
    NonFiniteInverseCascadeZ0 { frequency: usize, port: usize },

    #[error(
        "inverse-cascade reference impedance has a zero real part at frequency {frequency}, port {port}"
    )]
    ZeroRealInverseCascadeReferenceImpedance { frequency: usize, port: usize },

    #[error(
        "inverse-cascade {stage} system is exactly singular at frequency {frequency}, pivot {pivot}"
    )]
    SingularInverseCascade {
        stage: InverseCascadeStage,
        frequency: usize,
        pivot: usize,
    },

    #[error(
        "inverse-cascade {stage} computation became non-finite at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteInverseCascadeComputation {
        stage: InverseCascadeStage,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("{direction} mixed-mode conversion frequency axis must not be empty")]
    EmptyMixedModeFrequency { direction: MixedModeDirection },

    #[error(
        "{direction} mixed-mode conversion frequency length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    MixedModeFrequencyLengthMismatch {
        direction: MixedModeDirection,
        expected: usize,
        actual: usize,
    },

    #[error("{direction} mixed-mode conversion received an invalid S-parameter shape {shape:?}")]
    InvalidMixedModeSShape {
        direction: MixedModeDirection,
        shape: Vec<usize>,
    },

    #[error(
        "{direction} mixed-mode conversion received an invalid reference-impedance shape {shape:?}"
    )]
    InvalidMixedModeZ0Shape {
        direction: MixedModeDirection,
        shape: Vec<usize>,
    },

    #[error(
        "{direction} mixed-mode conversion pair_count must be at least one and no greater than floor(nports/2): pair_count={pair_count}, nports={nports}"
    )]
    MixedModePairCountOutOfRange {
        direction: MixedModeDirection,
        pair_count: usize,
        nports: usize,
    },

    #[error(
        "{direction} mixed-mode conversion received a non-finite S-parameter at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteMixedModeS {
        direction: MixedModeDirection,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "{direction} mixed-mode conversion received a non-finite reference impedance at frequency {frequency}, port {port}"
    )]
    NonFiniteMixedModeZ0 {
        direction: MixedModeDirection,
        frequency: usize,
        port: usize,
    },

    #[error(
        "{direction} mixed-mode conversion has a zero-real reference impedance at frequency {frequency}, port {port}"
    )]
    ZeroRealMixedModeReferenceImpedance {
        direction: MixedModeDirection,
        frequency: usize,
        port: usize,
    },

    #[error(
        "{direction} mixed-mode pair {pair} has unequal single-ended references at frequency {frequency}: positive={positive:?}, negative={negative:?}"
    )]
    UnequalMixedModePairReferences {
        direction: MixedModeDirection,
        frequency: usize,
        pair: usize,
        positive: Complex64,
        negative: Complex64,
    },

    #[error(
        "{direction} mixed-mode {mode:?} reference scaling at frequency {frequency}, pair {pair} with {scaling:?} of {value:?} loses finite information or is not finite"
    )]
    MixedModeReferenceScalingLoss {
        direction: MixedModeDirection,
        frequency: usize,
        pair: usize,
        mode: MixedModeMode,
        scaling: MixedModeScaling,
        value: Complex64,
    },

    #[error(
        "{direction} mixed-mode references at frequency {frequency}, pair {pair} are not exactly natural: differential={differential:?}, common={common:?}, zd/2={from_differential:?}, 2*zc={from_common:?}"
    )]
    MixedModeInverseReferenceMismatch {
        direction: MixedModeDirection,
        frequency: usize,
        pair: usize,
        differential: Complex64,
        common: Complex64,
        from_differential: Complex64,
        from_common: Complex64,
    },

    #[error(
        "{direction} mixed-mode conversion produced a non-finite computation at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteMixedModeComputation {
        direction: MixedModeDirection,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error("interpolation received an invalid {quantity} shape {shape:?}")]
    InvalidInterpolationShape {
        quantity: InterpolationQuantity,
        shape: Vec<usize>,
    },

    #[error(
        "interpolation source frequency axis length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    InterpolationSourceFrequencyLengthMismatch { expected: usize, actual: usize },

    #[error("interpolation source frequency axis must contain at least two samples, got {actual}")]
    InterpolationTooFewSourceSamples { actual: usize },

    #[error("interpolation target frequency axis must not be empty")]
    InterpolationEmptyTarget,

    #[error("interpolation {axis} frequency is non-finite at index {index}: {value:?}")]
    NonFiniteInterpolationFrequency {
        axis: InterpolationAxis,
        index: usize,
        value: f64,
    },

    #[error(
        "interpolation {axis} frequency axis must be strictly increasing at index {index}: previous={previous:?}, current={current:?}"
    )]
    InterpolationFrequencyNotStrictlyIncreasing {
        axis: InterpolationAxis,
        index: usize,
        previous: f64,
        current: f64,
    },

    #[error(
        "interpolation target frequency is outside the source span at index {index}: value={value:?}, span=[{lower:?}, {upper:?}]"
    )]
    InterpolationTargetOutOfRange {
        index: usize,
        value: f64,
        lower: f64,
        upper: f64,
    },

    #[error(
        "interpolation received a non-finite S-parameter at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteInterpolationS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "interpolation received a non-finite reference impedance at frequency {frequency}, port {port}"
    )]
    NonFiniteInterpolationZ0 { frequency: usize, port: usize },

    #[error(
        "interpolation weight is non-finite at target {target} between source indices {lower_source} and {upper_source}"
    )]
    NonFiniteInterpolationWeight {
        target: usize,
        lower_source: usize,
        upper_source: usize,
    },

    #[error(
        "interpolation computation is non-finite for {quantity} at target {target}, row {row}, column {column}"
    )]
    NonFiniteInterpolationComputation {
        quantity: InterpolationQuantity,
        target: usize,
        row: usize,
        column: usize,
    },

    #[error("{input} connection S-parameter shape must be (nfreq, nport, nport), got {shape:?}")]
    InvalidConnectionSShape {
        input: ConnectionInput,
        shape: Vec<usize>,
    },

    #[error("{input} connection reference-impedance shape must be (nfreq, nport), got {shape:?}")]
    InvalidConnectionZ0Shape {
        input: ConnectionInput,
        shape: Vec<usize>,
    },

    #[error("{input} connection frequency axis must not be empty")]
    EmptyConnectionFrequency { input: ConnectionInput },

    #[error(
        "{input} connection frequency axis length does not match S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    ConnectionFrequencyShape {
        input: ConnectionInput,
        expected: usize,
        actual: usize,
    },

    #[error("A and B connection frequency axes have different lengths: A={a}, B={b}")]
    ConnectionFrequencyLengthMismatch { a: usize, b: usize },

    #[error("connection frequency axes differ at index {index}: A={a:?}, B={b:?}")]
    ConnectionFrequencyMismatch { index: usize, a: f64, b: f64 },

    #[error("{input} connection frequency is non-finite at index {index}: {value:?}")]
    NonFiniteConnectionFrequency {
        input: ConnectionInput,
        index: usize,
        value: f64,
    },

    #[error(
        "{input} connection S-parameter is non-finite at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteConnectionS {
        input: ConnectionInput,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "{input} connection reference impedance is non-finite at frequency {frequency}, port {port}"
    )]
    NonFiniteConnectionZ0 {
        input: ConnectionInput,
        frequency: usize,
        port: usize,
    },

    #[error("{input} connection port {port} is out of range for {nports} ports")]
    InvalidConnectionPort {
        input: ConnectionInput,
        port: usize,
        nports: usize,
    },

    #[error(
        "{input} connection junction reference impedance is not finite, real, and strictly positive at frequency {frequency}, port {port}: {value:?}"
    )]
    InvalidConnectionJunctionZ0 {
        input: ConnectionInput,
        frequency: usize,
        port: usize,
        value: Complex64,
    },

    #[error("connected reference impedances differ at frequency {frequency}: A={a:?}, B={b:?}")]
    MismatchedConnectionJunctionZ0 {
        frequency: usize,
        a: Complex64,
        b: Complex64,
    },

    #[error("connecting the selected ports leaves no external ports")]
    NoExternalConnectionPorts,

    #[error("matched-junction connection is exactly singular at frequency {frequency}")]
    SingularConnection { frequency: usize },

    #[error(
        "non-finite value while evaluating matched-junction connection at frequency {frequency}, output row {row}, column {column}"
    )]
    NonFiniteConnectionComputation {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "{input} direct connection S-parameter shape must be (nfreq, nport, nport), got {shape:?}"
    )]
    InvalidDirectConnectionSShape {
        input: ConnectionInput,
        shape: Vec<usize>,
    },

    #[error(
        "{input} direct connection reference-impedance shape must be (nfreq, nport), got {shape:?}"
    )]
    InvalidDirectConnectionZ0Shape {
        input: ConnectionInput,
        shape: Vec<usize>,
    },

    #[error("{input} direct connection frequency axis must not be empty")]
    EmptyDirectConnectionFrequency { input: ConnectionInput },

    #[error(
        "{input} direct connection frequency axis length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    DirectConnectionFrequencyShape {
        input: ConnectionInput,
        expected: usize,
        actual: usize,
    },

    #[error("A and B direct connection frequency axes have different lengths: A={a}, B={b}")]
    DirectConnectionFrequencyLengthMismatch { a: usize, b: usize },

    #[error("direct connection frequency axes differ at index {index}: A={a:?}, B={b:?}")]
    DirectConnectionFrequencyMismatch { index: usize, a: f64, b: f64 },

    #[error("{input} direct connection frequency is non-finite at index {index}: {value:?}")]
    NonFiniteDirectConnectionFrequency {
        input: ConnectionInput,
        index: usize,
        value: f64,
    },

    #[error(
        "{input} direct connection S-parameter is non-finite at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteDirectConnectionS {
        input: ConnectionInput,
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "{input} direct connection reference impedance is non-finite at frequency {frequency}, port {port}"
    )]
    NonFiniteDirectConnectionZ0 {
        input: ConnectionInput,
        frequency: usize,
        port: usize,
    },

    #[error("{input} direct connection port {port} is out of range for {nports} ports")]
    InvalidDirectConnectionPort {
        input: ConnectionInput,
        port: usize,
        nports: usize,
    },

    #[error(
        "{input} direct connection reference impedance has a zero real part at frequency {frequency}, port {port}"
    )]
    ZeroRealDirectConnectionReferenceImpedance {
        input: ConnectionInput,
        frequency: usize,
        port: usize,
    },

    #[error("direct connection leaves no external ports")]
    NoExternalDirectConnectionPorts,

    #[error(
        "direct power-wave connection is exactly singular at frequency {frequency}, selected ports A={port_a}, B={port_b}, pivot {pivot}"
    )]
    SingularDirectConnection {
        frequency: usize,
        port_a: usize,
        port_b: usize,
        pivot: usize,
    },

    #[error(
        "non-finite value while evaluating direct power-wave connection at frequency {frequency}, selected ports A={port_a}, B={port_b}, row {row}, column {column}"
    )]
    NonFiniteDirectConnectionComputation {
        frequency: usize,
        port_a: usize,
        port_b: usize,
        row: usize,
        column: usize,
    },

    #[error("explicit-grid matched connection {stage} stage failed: {source}")]
    GridConnection {
        stage: GridConnectionStage,
        #[source]
        source: Box<Error>,
    },

    #[error("inner connection S-parameter shape must be (nfreq, nport, nport), got {shape:?}")]
    InvalidInnerConnectionSShape { shape: Vec<usize> },

    #[error("inner connection reference-impedance shape must be (nfreq, nport), got {shape:?}")]
    InvalidInnerConnectionZ0Shape { shape: Vec<usize> },

    #[error("inner connection frequency axis must not be empty")]
    EmptyInnerConnectionFrequency,

    #[error(
        "inner connection frequency axis length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    InnerConnectionFrequencyShape { expected: usize, actual: usize },

    #[error("inner connection frequency is non-finite at index {index}: {value:?}")]
    NonFiniteInnerConnectionFrequency { index: usize, value: f64 },

    #[error(
        "inner connection S-parameter is non-finite at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteInnerConnectionS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "inner connection reference impedance is non-finite at frequency {frequency}, port {port}"
    )]
    NonFiniteInnerConnectionZ0 { frequency: usize, port: usize },

    #[error("inner connection port {port} is out of range for {nports} ports")]
    InvalidInnerConnectionPort { port: usize, nports: usize },

    #[error("inner connection requires two distinct ports, got {port_a} and {port_b}")]
    IdenticalInnerConnectionPorts { port_a: usize, port_b: usize },

    #[error(
        "inner connection junction reference impedance is not finite, real, and strictly positive at frequency {frequency}, port {port}: {value:?}"
    )]
    InvalidInnerConnectionJunctionZ0 {
        frequency: usize,
        port: usize,
        value: Complex64,
    },

    #[error(
        "inner connection junction reference impedances differ at frequency {frequency}: port {port_a}={z0_a:?}, port {port_b}={z0_b:?}"
    )]
    MismatchedInnerConnectionJunctionZ0 {
        frequency: usize,
        port_a: usize,
        z0_a: Complex64,
        port_b: usize,
        z0_b: Complex64,
    },

    #[error("inner connection leaves no external ports")]
    NoExternalInnerConnectionPorts,

    #[error("matched-junction inner connection is exactly singular at frequency {frequency}")]
    SingularInnerConnection { frequency: usize },

    #[error(
        "non-finite value while evaluating matched-junction inner connection at frequency {frequency}, computation row {row}, computation column {column}"
    )]
    NonFiniteInnerConnectionComputation {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "direct inner connection S-parameter shape must be (nfreq, nport, nport), got {shape:?}"
    )]
    InvalidDirectInnerConnectionSShape { shape: Vec<usize> },

    #[error(
        "direct inner connection reference-impedance shape must be (nfreq, nport), got {shape:?}"
    )]
    InvalidDirectInnerConnectionZ0Shape { shape: Vec<usize> },

    #[error("direct inner connection frequency axis must not be empty")]
    EmptyDirectInnerConnectionFrequency,

    #[error(
        "direct inner connection frequency axis length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    DirectInnerConnectionFrequencyShape { expected: usize, actual: usize },

    #[error("direct inner connection frequency is non-finite at index {index}: {value:?}")]
    NonFiniteDirectInnerConnectionFrequency { index: usize, value: f64 },

    #[error(
        "direct inner connection S-parameter is non-finite at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteDirectInnerConnectionS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "direct inner connection reference impedance is non-finite at frequency {frequency}, port {port}"
    )]
    NonFiniteDirectInnerConnectionZ0 { frequency: usize, port: usize },

    #[error("direct inner connection port {port} is out of range for {nports} ports")]
    InvalidDirectInnerConnectionPort { port: usize, nports: usize },

    #[error("direct inner connection requires two distinct ports, got {port_a} and {port_b}")]
    IdenticalDirectInnerConnectionPorts { port_a: usize, port_b: usize },

    #[error(
        "direct inner connection reference impedance has a zero real part at frequency {frequency}, port {port}"
    )]
    ZeroRealDirectInnerConnectionReferenceImpedance { frequency: usize, port: usize },

    #[error("direct inner connection leaves no external ports")]
    NoExternalDirectInnerConnectionPorts,

    #[error(
        "direct power-wave inner connection is exactly singular at frequency {frequency}, selected ports A={port_a}, B={port_b}, pivot {pivot}"
    )]
    SingularDirectInnerConnection {
        frequency: usize,
        port_a: usize,
        port_b: usize,
        pivot: usize,
    },

    #[error(
        "non-finite value while evaluating direct power-wave inner connection at frequency {frequency}, selected ports A={port_a}, B={port_b}, row {row}, column {column}"
    )]
    NonFiniteDirectInnerConnectionComputation {
        frequency: usize,
        port_a: usize,
        port_b: usize,
        row: usize,
        column: usize,
    },

    #[error("termination source frequency axis must not be empty")]
    EmptyTerminationFrequency,

    #[error(
        "termination source frequency length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    TerminationFrequencyLengthMismatch { expected: usize, actual: usize },

    #[error("termination received an invalid S-parameter shape {shape:?}")]
    InvalidTerminationSShape { shape: Vec<usize> },

    #[error("termination received an invalid reference-impedance shape {shape:?}")]
    InvalidTerminationZ0Shape { shape: Vec<usize> },

    #[error(
        "termination load length does not match the source frequency dimension: expected {expected}, got {actual}"
    )]
    TerminationLoadLengthMismatch { expected: usize, actual: usize },

    #[error("termination requires at least two source ports, got {nports}")]
    NoTerminationSurvivors { nports: usize },

    #[error("termination port {port} is out of range for {nports} source ports")]
    InvalidTerminationPort { port: usize, nports: usize },

    #[error(
        "termination received a non-finite S-parameter at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteTerminationS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "termination received a non-finite reference impedance at frequency {frequency}, port {port}"
    )]
    NonFiniteTerminationZ0 { frequency: usize, port: usize },

    #[error(
        "termination received a zero-real reference impedance at frequency {frequency}, port {port}"
    )]
    ZeroRealTerminationReferenceImpedance { frequency: usize, port: usize },

    #[error("termination load is non-finite at frequency {frequency}")]
    NonFiniteTerminationLoad { frequency: usize },

    #[error("termination denominator is exactly singular at frequency {frequency}, port {port}")]
    SingularTermination { frequency: usize, port: usize },

    #[error(
        "termination produced a non-finite computation at frequency {frequency}, selected port {port}, row {row}, column {column}"
    )]
    NonFiniteTerminationComputation {
        frequency: usize,
        port: usize,
        row: usize,
        column: usize,
    },

    #[error("two-port stability frequency axis must not be empty")]
    EmptyTwoPortStabilityFrequency,

    #[error(
        "two-port stability frequency length does not match the S-parameter frequency dimension: expected {expected}, got {actual}"
    )]
    TwoPortStabilityFrequencyLengthMismatch { expected: usize, actual: usize },

    #[error("two-port stability requires S-parameter shape (nfreq, 2, 2), got {shape:?}")]
    InvalidTwoPortStabilitySShape { shape: Vec<usize> },

    #[error("two-port stability requires reference-impedance shape (nfreq, 2), got {shape:?}")]
    InvalidTwoPortStabilityZ0Shape { shape: Vec<usize> },

    #[error("two-port stability frequency is non-finite at index {index}: {value:?}")]
    NonFiniteTwoPortStabilityFrequency { index: usize, value: f64 },

    #[error(
        "two-port stability received a non-finite S-parameter at frequency {frequency}, row {row}, column {column}"
    )]
    NonFiniteTwoPortStabilityS {
        frequency: usize,
        row: usize,
        column: usize,
    },

    #[error(
        "two-port stability received a non-finite reference impedance at frequency {frequency}, port {port}"
    )]
    NonFiniteTwoPortStabilityZ0 { frequency: usize, port: usize },

    #[error(
        "two-port stability reference impedance must have a strictly positive real part at frequency {frequency}, port {port}: {value:?}"
    )]
    NonPositiveRealTwoPortStabilityReferenceImpedance {
        frequency: usize,
        port: usize,
        value: Complex64,
    },

    #[error(
        "two-port stability arithmetic became non-finite or unrepresentable at frequency {frequency} while evaluating {stage}"
    )]
    NonFiniteTwoPortStabilityComputation {
        frequency: usize,
        stage: TwoPortStabilityArithmetic,
    },
}

/// Identifies which input supplied a connection value or triggered a
/// connection validation error.
///
/// `A` denotes the receiver (`self`) and the private kernel's input A; `B`
/// denotes the `other` network and the private kernel's input B.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionInput {
    /// The receiver network (`self`, kernel input A).
    A,
    /// The network passed as `other` (kernel input B).
    B,
}

/// Identifies the ordered stage that failed during an explicit-grid matched
/// connection.
///
/// The operation always interpolates the receiver (A) first, then `other`
/// (B), and only then evaluates the matched connection.  Keeping this stage
/// outside the nested [`Error`] preserves deterministic failure attribution
/// while the nested error retains the detailed shape, axis, index, value, or
/// numerical context from the underlying public operation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridConnectionStage {
    /// Cartesian interpolation of the receiver (`self`, kernel input A).
    AInterpolation,
    /// Cartesian interpolation of `other` (kernel input B).
    BInterpolation,
    /// Matched power-wave connection after both interpolations complete.
    Connection,
}

impl fmt::Display for GridConnectionStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::AInterpolation => "A interpolation (receiver)",
            Self::BInterpolation => "B interpolation (other)",
            Self::Connection => "matched connection",
        })
    }
}

impl fmt::Display for ConnectionInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::A => "A (self)",
            Self::B => "B (other)",
        })
    }
}

/// Axis associated with a structured interpolation error.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationAxis {
    /// The source network's frequency axis.
    Source,
    /// The caller-provided target frequency axis.
    Target,
}

impl fmt::Display for InterpolationAxis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Source => "source",
            Self::Target => "target",
        })
    }
}

/// Quantity associated with a structured interpolation shape or computation
/// error.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationQuantity {
    /// Scattering-parameter data.
    S,
    /// Reference-impedance data.
    Z0,
}

impl fmt::Display for InterpolationQuantity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::S => "S-parameter",
            Self::Z0 => "reference-impedance",
        })
    }
}

/// High-level conversion stage associated with a public conversion error.
///
/// `to_y_power` is intentionally a composed `S→Z→Y` operation, while
/// [`Network::to_y_direct_power`] uses the separate direct `S→Y` stage.  Both
/// methods preserve their distinct conversion domains.  The parameter-ingress
/// method [`Network::from_y_via_z_power`] is intentionally a composed `Y→Z→S`
/// operation. [`Network::from_y_direct_power`] uses the separate direct
/// `Y→S` stage. Preserving these stages lets callers distinguish a singular or
/// invalid intermediate impedance conversion from a failure while inverting
/// that impedance or evaluating a direct wave equation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionStage {
    /// Conversion from scattering parameters to impedance parameters.
    SToZ,
    /// Direct conversion from scattering parameters to admittance parameters.
    SToY,
    /// Conversion from admittance parameters to impedance parameters.
    YToZ,
    /// Conversion from impedance parameters to admittance parameters.
    ZToY,
    /// Conversion from impedance parameters to scattering parameters.
    ZToS,
    /// Direct conversion from admittance parameters to scattering parameters.
    YToS,
}

impl fmt::Display for ConversionStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::SToZ => "S→Z",
            Self::SToY => "S→Y",
            Self::YToZ => "Y→Z",
            Self::ZToY => "Z→Y",
            Self::ZToS => "Z→S",
            Self::YToS => "Y→S",
        })
    }
}

/// Kind of RF data reported by a structured conversion error.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterKind {
    /// Scattering-parameter matrix.
    S,
    /// Impedance-parameter matrix.
    Z,
    /// Admittance-parameter matrix.
    Y,
    /// Reference-impedance data.
    Z0,
}

impl fmt::Display for ParameterKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::S => "S-parameter",
            Self::Z => "Z-parameter",
            Self::Y => "Y-parameter",
            Self::Z0 => "reference-impedance",
        })
    }
}

/// Identifies which reference-impedance array supplied a value to direct
/// power-wave renormalization.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectRenormalizationReference {
    /// The network's stored source reference array.
    Source,
    /// The caller-supplied target reference array.
    Target,
}

impl fmt::Display for DirectRenormalizationReference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Source => "source",
            Self::Target => "target",
        })
    }
}

/// Stage associated with a power-wave inverse-cascade validation or solve
/// failure.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InverseCascadeStage {
    /// The complete source S matrix `S`.
    FullS,
    /// The forward transmission block `S[right, left]`.
    ForwardTransmission,
    /// The reverse transmission block `S[left, right]`.
    ReverseTransmission,
}

impl fmt::Display for InverseCascadeStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::FullS => "full S",
            Self::ForwardTransmission => "forward transmission",
            Self::ReverseTransmission => "reverse transmission",
        })
    }
}

/// Direction of an equal-pair mixed-mode coordinate conversion.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedModeDirection {
    /// Convert adjacent single-ended pairs to differential/common coordinates.
    ToMixedMode,
    /// Convert `[d..., c..., unpaired]` coordinates back to adjacent pairs.
    ToSingleEnded,
}

impl fmt::Display for MixedModeDirection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ToMixedMode => "single-ended to mixed-mode",
            Self::ToSingleEnded => "mixed-mode to single-ended",
        })
    }
}

/// Modal reference kind used in mixed-mode reference-scaling diagnostics.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedModeMode {
    /// Differential mode, with natural reference `2*z`.
    Differential,
    /// Common mode, with natural reference `z/2`.
    Common,
}

/// Exact scalar operation used to derive a natural modal reference.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedModeScaling {
    /// Multiply a reference component by two.
    Double,
    /// Divide a reference component by two.
    Half,
}

/// Arithmetic quantity used to identify an unrepresentable sampled
/// two-port stability calculation.
///
/// The operation reports undefined K only for an exactly zero transmission
/// coefficient. A finite, nonzero coefficient whose magnitude product
/// underflows, overflows, or otherwise produces a non-finite intermediate is
/// an arithmetic error identified by this stage instead.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoPortStabilityArithmetic {
    /// The `S11*S22` determinant product.
    DeltaS11S22Product,
    /// The `S12*S21` determinant product.
    DeltaS12S21Product,
    /// The determinant subtraction `S11*S22-S12*S21`.
    DeltaSubtraction,
    /// The magnitude of `S11`.
    S11Magnitude,
    /// The magnitude of `S22`.
    S22Magnitude,
    /// The magnitude of `delta`.
    DeltaMagnitude,
    /// The squared magnitude of `S11`.
    S11MagnitudeSquared,
    /// The squared magnitude of `S22`.
    S22MagnitudeSquared,
    /// The squared magnitude of `delta`.
    DeltaMagnitudeSquared,
    /// The first subtraction in the Rollett numerator.
    NumeratorS11Subtraction,
    /// The second subtraction in the Rollett numerator.
    NumeratorS22Subtraction,
    /// The determinant-magnitude addition in the Rollett numerator.
    NumeratorDeltaAddition,
    /// The magnitude of `S12`.
    S12Magnitude,
    /// The magnitude of `S21`.
    S21Magnitude,
    /// The product of the nonzero transmission magnitudes.
    TransmissionMagnitudeProduct,
    /// The factor-of-two denominator scaling.
    TransmissionDenominatorScaling,
    /// The final Rollett K division.
    RolletKDivision,
}

impl fmt::Display for TwoPortStabilityArithmetic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = match self {
            Self::DeltaS11S22Product => "delta S11*S22 product",
            Self::DeltaS12S21Product => "delta S12*S21 product",
            Self::DeltaSubtraction => "delta subtraction",
            Self::S11Magnitude => "|S11|",
            Self::S22Magnitude => "|S22|",
            Self::DeltaMagnitude => "|delta|",
            Self::S11MagnitudeSquared => "|S11|²",
            Self::S22MagnitudeSquared => "|S22|²",
            Self::DeltaMagnitudeSquared => "|delta|²",
            Self::NumeratorS11Subtraction => "Rollett numerator 1-|S11|²",
            Self::NumeratorS22Subtraction => "Rollett numerator subtraction of |S22|²",
            Self::NumeratorDeltaAddition => "Rollett numerator addition of |delta|²",
            Self::S12Magnitude => "|S12|",
            Self::S21Magnitude => "|S21|",
            Self::TransmissionMagnitudeProduct => "|S12|*|S21|",
            Self::TransmissionDenominatorScaling => "2*|S12|*|S21|",
            Self::RolletKDivision => "Rollett K division",
        };
        formatter.write_str(description)
    }
}

/// The crate-wide public error boundary.
pub type Result<T> = std::result::Result<T, Error>;

/// Frequency axis in hertz.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Frequency {
    hz: Vec<f64>,
}

impl Frequency {
    /// Creates a frequency axis in hertz.
    pub fn from_hz(hz: Vec<f64>) -> Result<Self> {
        if hz.is_empty() {
            return Err(Error::EmptyFrequency);
        }
        Ok(Self { hz })
    }

    /// Returns the frequency samples in hertz.
    #[must_use]
    pub fn hz(&self) -> &[f64] {
        &self.hz
    }

    /// Number of frequency points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hz.len()
    }

    /// Whether the axis contains no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hz.is_empty()
    }
}

/// Sampled two-port power-wave stability metrics.
///
/// `delta` is the dimensionless complex determinant
/// `S11*S22-S12*S21`. `rollet_k` is the dimensionless Rollett factor when
/// both transmission coefficients are exactly nonzero, and is `None` when
/// either `S12` or `S21` is exactly complex zero. The latter is an explicit
/// undefined value, not a stability verdict and not a numerical-failure
/// catch-all.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TwoPortStability {
    /// The sampled determinant `S11*S22-S12*S21`.
    pub delta: Complex64,
    /// The sampled Rollett factor, undefined for an exactly unilateral or
    /// isolated transmission coefficient.
    pub rollet_k: Option<f64>,
}

/// An N-port network represented by scattering parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Network {
    frequency: Frequency,
    s: Array3<Complex64>,
    z0: Array2<Complex64>,
}

impl Network {
    /// Constructs a network after validating the core dimensional invariants.
    pub fn new(frequency: Frequency, s: Array3<Complex64>, z0: Array2<Complex64>) -> Result<Self> {
        let (nfreq, nport_a, nport_b) = s.dim();
        if nfreq != frequency.len() || nport_a == 0 || nport_a != nport_b {
            return Err(Error::InvalidSShape);
        }
        if z0.dim() != (nfreq, nport_a) {
            return Err(Error::InvalidZ0Shape);
        }

        Ok(Self { frequency, s, z0 })
    }

    #[must_use]
    pub fn frequency(&self) -> &Frequency {
        &self.frequency
    }

    #[must_use]
    pub fn s(&self) -> &Array3<Complex64> {
        &self.s
    }

    #[must_use]
    pub fn z0(&self) -> &Array2<Complex64> {
        &self.z0
    }

    #[must_use]
    pub fn nports(&self) -> usize {
        self.s.dim().1
    }

    /// Computes sampled two-port Rollett stability metrics from the stored
    /// Kurokawa power-wave S-parameters.
    ///
    /// The returned vector has exactly one [`TwoPortStability`] record per
    /// source-frequency sample, in the same order. For each sample,
    /// `delta = S11*S22-S12*S21` and, when both transmission coefficients are
    /// exactly nonzero,
    /// `K = (1-|S11|²-|S22|²+|delta|²)/(2|S12||S21|)`. If `S12` or `S21` is
    /// exactly complex zero, `delta` is still evaluated and `rollet_k` is
    /// `None`; this represents an undefined denominator and is not an
    /// unconditional-stability verdict. Finite nonzero transmissions are
    /// never classified as undefined by a tolerance or cutoff.
    ///
    /// This operation borrows the network and does not convert through Z/Y,
    /// renormalize, sort, interpolate, clip, or otherwise mutate its data.
    /// It requires exactly two ports, finite frequency labels, finite S and
    /// reference values, and strictly positive real parts for both stored
    /// references. Complex, unequal, per-port, and frequency-dependent
    /// positive-real-part references are supported; no 50-ohm default is
    /// assumed. Frequency labels remain opaque pointwise samples, so negative,
    /// duplicate, descending, and signed-zero values are retained.
    ///
    /// The familiar linear two-port interpretation requires both `K > 1` and
    /// `|delta| < 1`, together with the usual auxiliary/proviso conditions;
    /// this method intentionally returns only sampled metrics. Sampled
    /// external S-parameters cannot certify internal poles, unsampled
    /// frequencies, nonlinear or large-signal behavior, or overall circuit
    /// stability.
    ///
    /// This additive operation is provisional during the `0.x` series; its
    /// name and signature are not a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns a structured [`Error`] for malformed serde-created axes or
    /// shapes, non-finite labels/data, non-positive-real references, and any
    /// non-finite or unrepresentable intermediate or output. In particular,
    /// an underflow or overflow in the magnitude product of finite nonzero
    /// transmissions is an arithmetic error rather than `None`.
    pub fn two_port_stability_power(&self) -> Result<Vec<TwoPortStability>> {
        stability::two_port_stability_power(&self.frequency.hz, &self.s, &self.z0)
            .map_err(map_two_port_stability_error)
    }

    /// Returns an owned network with its ports in an explicitly requested
    /// order.
    ///
    /// `order[new_port] = old_port` uses zero-based port indices.  For
    /// example, `[2, 0, 1]` places the original port 2 first, the original
    /// port 0 second, and the original port 1 third.  The argument must be a
    /// complete bijection of all ports; it is not a port-selection or partial
    /// renumbering operation.
    ///
    /// For every frequency `f`, the returned values are copied according to
    /// `out.s[f, i, j] = self.s[f, order[i], order[j]]` and
    /// `out.z0[f, i] = self.z0[f, order[i]]`.  Frequency samples and every
    /// copied S/z0 scalar are retained exactly, including non-finite values,
    /// signed zero, and complex references that other RF operations may
    /// reject.  No RF arithmetic, normalization, interpolation, tolerance,
    /// or implicit reference selection is performed.
    ///
    /// The result owns independent copies of the frequency, S-parameter, and
    /// reference arrays.  Neither this network nor `order` is modified.
    /// Downstream operations continue to apply their own validation rules.
    ///
    /// This additive operation is provisional while `rfkit-core` is in the
    /// `0.x` series; its name and signature are not a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns a structured [`enum@Error`] when a serde-created network has
    /// an empty or mismatched frequency axis, a non-square/zero-port S array,
    /// or a mismatched z0 array.  A permutation with the wrong length, an
    /// out-of-range old-port index, or a duplicate old-port index is rejected
    /// before any array indexing occurs.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array2, Array3};
    /// use num_complex::Complex64;
    /// use rfkit_core::{Frequency, Network};
    ///
    /// # fn example() -> rfkit_core::Result<()> {
    /// let network = Network::new(
    ///     Frequency::from_hz(vec![1.0e9])?,
    ///     Array3::from_shape_fn((1, 3, 3), |(_, row, column)| {
    ///         Complex64::new((10 * row + column) as f64, 0.0)
    ///     }),
    ///     Array2::from_shape_fn((1, 3), |(_, port)| {
    ///         Complex64::new([50.0, 60.0, 70.0][port], 0.0)
    ///     }),
    /// )?;
    ///
    /// // New port 0 is old port 2; new port 1 is old port 0; new port 2 is old port 1.
    /// let reordered = network.permute_ports(&[2, 0, 1])?;
    /// assert_eq!(reordered.s()[[0, 0, 0]], network.s()[[0, 2, 2]]);
    /// assert_eq!(reordered.z0()[[0, 0]], network.z0()[[0, 2]]);
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn permute_ports(&self, order: &[usize]) -> Result<Network> {
        if self.frequency.is_empty() {
            return Err(Error::EmptyPortPermutationFrequency);
        }

        let (parameter_frequency_length, nport_rows, nport_columns) = self.s.dim();
        if parameter_frequency_length != self.frequency.len() {
            return Err(Error::PortPermutationFrequencyLengthMismatch {
                expected: self.frequency.len(),
                actual: parameter_frequency_length,
            });
        }
        if nport_rows == 0 || nport_rows != nport_columns {
            return Err(Error::InvalidPortPermutationSShape {
                shape: vec![parameter_frequency_length, nport_rows, nport_columns],
            });
        }
        if self.z0.dim() != (parameter_frequency_length, nport_rows) {
            let (z0_frequencies, z0_ports) = self.z0.dim();
            return Err(Error::InvalidPortPermutationZ0Shape {
                shape: vec![z0_frequencies, z0_ports],
            });
        }

        if order.len() != nport_rows {
            return Err(Error::PortPermutationLengthMismatch {
                expected: nport_rows,
                actual: order.len(),
            });
        }

        let mut first_positions = vec![None; nport_rows];
        for (position, &port) in order.iter().enumerate() {
            if port >= nport_rows {
                return Err(Error::PortPermutationOutOfRange {
                    position,
                    port,
                    nports: nport_rows,
                });
            }
            if let Some(first_position) = first_positions[port] {
                return Err(Error::PortPermutationDuplicate {
                    port,
                    first_position,
                    second_position: position,
                });
            }
            first_positions[port] = Some(position);
        }

        let output_s = Array3::from_shape_fn(
            (parameter_frequency_length, nport_rows, nport_rows),
            |(frequency, row, column)| self.s[[frequency, order[row], order[column]]],
        );
        let output_z0 = Array2::from_shape_fn(
            (parameter_frequency_length, nport_rows),
            |(frequency, port)| self.z0[[frequency, order[port]]],
        );

        Ok(Network {
            frequency: self.frequency.clone(),
            s: output_s,
            z0: output_z0,
        })
    }

    /// Converts adjacent equal-reference single-ended pairs to differential
    /// and common-mode coordinates using Kurokawa power waves.
    ///
    /// `pair_count` selects the adjacent positive/negative pairs
    /// `(0,1), (2,3), ...`, with the first port positive.  The returned
    /// coordinate order is `[d0, ..., d(p-1), c0, ..., c(p-1), unpaired]`,
    /// where `p == pair_count`; ports from `2*p` onward are copied in their
    /// original order.  Other physical pairings or polarity choices are not
    /// inferred.  Call [`Network::permute_ports`] explicitly before this
    /// method when a different pairing or positive-port polarity is wanted.
    ///
    /// For each selected pair whose equal single-ended reference is `z`, the
    /// natural modal references are `zd = 2*z` and `zc = z/2`.  This follows
    /// directly from the repository's Kurokawa definitions with currents into
    /// the network: both incident and reflected modal waves are respectively
    /// `(u-v)/sqrt(2)` and `(u+v)/sqrt(2)`.  References can vary by pair and
    /// frequency, can be complex, and can have negative real parts.  Equality
    /// within a pair is exact complex equality; no reference averaging,
    /// tolerance, or hidden renormalization is applied.
    ///
    /// At every frequency the S-parameter transform is `Smm = U*Sse*U^T`,
    /// where the rows of real orthogonal `U` are the difference/sum rows just
    /// described followed by identity rows for unpaired ports.  The operation
    /// is a coordinate transformation, not a Z/Y conversion, so singular S
    /// matrices, ideal opens, shorts, and thru networks remain in-domain when
    /// their finite floating-point transform remains finite.
    ///
    /// The method returns the existing owned [`Network`] representation.  It
    /// does not add mode metadata: after conversion, `Network::s` and
    /// `Network::z0` describe the declared modal coordinate order, and callers
    /// must retain `pair_count` and the pairing/polarity convention when using
    /// the inverse or exporting the result.  Frequencies are opaque pointwise
    /// labels and are copied exactly; no sorting, interpolation, or frequency
    /// restriction is introduced.  The input network and all caller-owned
    /// arrays remain unchanged.
    ///
    /// Natural-reference scaling is deliberately a representability boundary.
    /// The method rejects non-finite doubled/halved components, a nonzero
    /// component collapsing to zero, or a scale followed by its inverse not
    /// reproducing the original component exactly.  This prevents a later
    /// inverse from silently inventing or losing a reference value.
    ///
    /// This additive operation is provisional while `rfkit-core` is in the
    /// `0.x` series; its name and signature are not a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns a structured [`enum@Error`] for empty or frequency-mismatched
    /// serde-created data, non-square/zero-port S arrays, mismatched z0
    /// arrays, invalid pair counts (including a huge `usize`), non-finite S or
    /// references, zero-real references, unequal pair references, scaling
    /// overflow/underflow/information loss, or non-finite transform arithmetic.
    pub fn to_mixed_mode_equal_pair_power(&self, pair_count: usize) -> Result<Network> {
        let (s, z0) = mixed_mode::to_mixed_mode_equal_pair_power(
            self.frequency.len(),
            &self.s,
            &self.z0,
            pair_count,
        )
        .map_err(map_mixed_mode_error)?;

        Network::new(self.frequency.clone(), s, z0)
    }

    /// Converts `[d..., c..., unpaired]` equal-pair mixed-mode coordinates
    /// back to adjacent positive/negative single-ended pairs.
    ///
    /// The input is interpreted explicitly as differential modes first,
    /// followed by common modes, followed by unpaired coordinates.  For
    /// `pair_count == p`, output ports are adjacent pairs `(0,1), (2,3), ...`
    /// with the first member positive, followed by the unchanged unpaired
    /// ports.  This is the exact inverse coordinate convention of
    /// [`Network::to_mixed_mode_equal_pair_power`]; use
    /// [`Network::permute_ports`] when the physical source ordering or
    /// polarity differs.
    ///
    /// For each mode pair the input references must satisfy the natural
    /// relationship exactly: `zd/2 == 2*zc`, with both values finite, nonzero
    /// real, and representable without component loss.  The resulting value is
    /// assigned to both single-ended ports.  No tolerance, reference averaging,
    /// hidden renormalization, or 50-ohm assumption is used.  Complex and
    /// negative-real references are supported under the same Kurokawa
    /// `abs(Re(z0))` normalization domain.
    ///
    /// At every frequency the S-parameter transform is
    /// `Sse = U^T*Smm*U`; singular S and ideal networks are valid when the
    /// finite floating-point transform remains finite.  Frequencies are opaque
    /// labels copied exactly.  The returned `Network` owns independent arrays,
    /// and this method does not attach mode metadata: callers must retain the
    /// declared coordinate order and `pair_count` themselves.
    ///
    /// This additive operation is provisional while `rfkit-core` is in the
    /// `0.x` series; its name and signature are not a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns a structured [`enum@Error`] for the same shape, finite-value,
    /// zero-real, pair-count, and arithmetic domains as the forward method,
    /// plus an exact differential/common natural-reference mismatch.
    pub fn to_single_ended_equal_pair_power(&self, pair_count: usize) -> Result<Network> {
        let (s, z0) = mixed_mode::to_single_ended_equal_pair_power(
            self.frequency.len(),
            &self.s,
            &self.z0,
            pair_count,
        )
        .map_err(map_mixed_mode_error)?;

        Network::new(self.frequency.clone(), s, z0)
    }

    /// Constructs a scattering [`Network`] from frequency-major impedance
    /// parameters using the Kurokawa power-wave convention.
    ///
    /// `z` contains physical impedance matrices in ohms with shape
    /// `(nfreq, nport, nport)`, and `z0` contains the explicit reference
    /// impedances in ohms with shape `(nfreq, nport)`. The returned S-parameter
    /// matrices are dimensionless and retain the supplied frequency samples,
    /// frequency order, port order, and `z0` values exactly. Both arrays are
    /// owned at this boundary; the conversion borrows them while evaluating
    /// and does not mutate them.
    ///
    /// The conversion uses power waves, with
    /// `G = diag(z0)` and `F = diag(1/(2*sqrt(abs(Re(z0)))))`, and delegates
    /// the numerical work to the existing `Z→S` kernel. References may be
    /// finite complex values, may vary by frequency and port, and may have a
    /// negative real part because the kernel uses `abs(Re(z0))`. A zero real
    /// part or non-finite reference is outside this normalization domain.
    /// `z` itself need not be invertible; for example, a zero impedance matrix
    /// succeeds whenever the `Z + G` conversion system is nonsingular.
    ///
    /// Frequency samples are pointwise labels for this constructor. The axis
    /// must be non-empty and its length must equal the first axis of `z`; the
    /// constructor does not require samples to be finite, non-negative,
    /// sorted, or unique. Those policies belong to operations such as the
    /// Touchstone writer. No frequency sorting, resampling, broadcasting,
    /// 50-ohm default, regularization, pseudoinverse, cutoff, fallback, or
    /// identity shortcut is applied.
    ///
    /// This is a provisional additive 0.x API. Its name and signature are not
    /// a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns a structured [`enum@Error`] identifying an empty or mismatched Z
    /// frequency axis before kernel execution. Kernel diagnostics preserve
    /// the `Z→S` stage and identify Z, z0, non-finite input, exact singularity,
    /// and non-finite computation failures.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array2, Array3};
    /// use num_complex::Complex64;
    /// use rfkit_core::{Frequency, Network};
    ///
    /// # fn example() -> rfkit_core::Result<()> {
    /// let frequency = Frequency::from_hz(vec![1.0e9])?;
    /// let z = Array3::from_elem((1, 1, 1), Complex64::new(100.0, 0.0));
    /// let z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
    /// let network = Network::from_z_power(frequency, z, z0)?;
    ///
    /// assert!((network.s()[[0, 0, 0]].re - 1.0 / 3.0).abs() < 1.0e-14);
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn from_z_power(
        frequency: Frequency,
        z: Array3<Complex64>,
        z0: Array2<Complex64>,
    ) -> Result<Network> {
        validate_parameter_frequency(&frequency, z.dim().0, ParameterKind::Z)?;

        let s = power_waves::z_to_s_power(&z, &z0)
            .map_err(|error| map_power_wave_error(ConversionStage::ZToS, error))?;
        Network::new(frequency, s, z0)
    }

    /// Constructs a scattering [`Network`] from frequency-major admittance
    /// parameters through the explicit composed `Y→Z→S` power-wave path.
    ///
    /// `y` contains physical admittance matrices in siemens with shape
    /// `(nfreq, nport, nport)`, and `z0` contains explicit reference
    /// impedances in ohms with shape `(nfreq, nport)`. The result stores
    /// dimensionless S-parameters and exact owned copies of the supplied
    /// frequency labels and references. The public name intentionally states
    /// the composed path: the existing kernel first inverts `Y` to form `Z`,
    /// then applies the verified power-wave `Z→S` conversion.
    ///
    /// Consequently, singular or zero Y is rejected in the `Y→Z` stage,
    /// even where [`Network::from_y_direct_power`] can represent an ideal
    /// open. This constructor does not add a direct Y→S equation or broaden
    /// the inverse-domain policy. References otherwise follow
    /// [`Network::from_z_power`]: finite complex, per-port,
    /// frequency-dependent, and negative-real values are accepted under the
    /// existing `abs(Re(z0))` normalization, while zero-real and non-finite
    /// values are rejected.
    ///
    /// Frequency samples are pointwise labels. The axis must be non-empty and
    /// have the same length as the first axis of `y`; samples need not be
    /// finite, non-negative, sorted, or unique. No sorting, resampling,
    /// broadcasting, 50-ohm default, regularization, pseudoinverse, cutoff,
    /// fallback, or identity shortcut is used.
    ///
    /// This is a provisional additive 0.x API. Its name and signature are not
    /// a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns a structured [`enum@Error`] identifying an empty or mismatched Y
    /// frequency axis before kernel execution. Kernel diagnostics preserve
    /// the `Y→Z` versus `Z→S` stage and identify Y, intermediate Z, z0,
    /// non-finite input, exact singularity, and non-finite computation
    /// failures.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array2, Array3};
    /// use num_complex::Complex64;
    /// use rfkit_core::{Frequency, Network};
    ///
    /// # fn example() -> rfkit_core::Result<()> {
    /// let frequency = Frequency::from_hz(vec![1.0e9])?;
    /// let y = Array3::from_elem((1, 1, 1), Complex64::new(0.02, 0.0));
    /// let z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
    /// let network = Network::from_y_via_z_power(frequency, y, z0)?;
    ///
    /// assert!(network.s()[[0, 0, 0]].norm() < 1.0e-14);
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn from_y_via_z_power(
        frequency: Frequency,
        y: Array3<Complex64>,
        z0: Array2<Complex64>,
    ) -> Result<Network> {
        validate_parameter_frequency(&frequency, y.dim().0, ParameterKind::Y)?;

        let s = power_wave_admittance::y_to_s_power(&y, &z0)
            .map_err(map_power_wave_admittance_error)?;
        Network::new(frequency, s, z0)
    }

    /// Constructs a scattering [`Network`] directly from frequency-major
    /// admittance parameters using Kurokawa power waves.
    ///
    /// `y` contains physical admittance matrices in siemens with shape
    /// `(nfreq, nport, nport)`, and `z0` contains explicit reference
    /// impedances in ohms with shape `(nfreq, nport)`. The result stores
    /// dimensionless S-parameters and preserves the supplied frequency
    /// samples, their order, port order, and `z0` values exactly.
    ///
    /// Unlike [`Network::from_y_via_z_power`], this constructor does not form
    /// `Y⁻¹`. At each frequency it forms
    /// `A = F (I + G Y)` and `B = F (I - conj(G) Y)` and solves `S A = B`,
    /// where `G = diag(z0)` and
    /// `F = diag(1/(2*sqrt(abs(Re(z0)))))`. Consequently a singular or zero
    /// Y is supported whenever the direct conversion system `A` is nonsingular
    /// and its arithmetic remains finite. Zero Y follows the ordinary
    /// validation and solve path and produces identity S for valid references;
    /// it is not an identity shortcut.
    ///
    /// References may be finite complex values, may vary by frequency and
    /// port, and may have a negative real part because the normalization uses
    /// `abs(Re(z0))`. A zero real part or non-finite reference is outside this
    /// wave normalization domain. The direct solver uses exact-zero pivot
    /// detection only: it does not use an explicit inverse, pseudoinverse,
    /// rank cutoff, regularization, nudge, clipping, or fallback. A finite
    /// near-singular `A` is therefore not rejected merely for being near
    /// singular.
    ///
    /// This is a provisional additive 0.x API. Its name and signature are not
    /// a `1.0` stability promise. Existing downstream conversions retain
    /// their own domains: a network constructed from singular Y is not
    /// promised to succeed through `to_z_power`, `to_y_power`, or any other
    /// operation whose intermediate conversion is singular.
    ///
    /// # Errors
    ///
    /// Returns a structured [`enum@Error`] identifying an empty or mismatched
    /// Y frequency axis before kernel execution. Kernel diagnostics use the
    /// direct `Y→S` stage and preserve Y versus z0 attribution, exact
    /// singularity, and non-finite computation context.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array2, Array3};
    /// use num_complex::Complex64;
    /// use rfkit_core::{Frequency, Network};
    ///
    /// # fn example() -> rfkit_core::Result<()> {
    /// let frequency = Frequency::from_hz(vec![1.0e9])?;
    /// let series_y = Array3::from_shape_vec(
    ///     (1, 2, 2),
    ///     vec![
    ///         Complex64::new(0.01, 0.0),
    ///         Complex64::new(-0.01, 0.0),
    ///         Complex64::new(-0.01, 0.0),
    ///         Complex64::new(0.01, 0.0),
    ///     ],
    /// )
    /// .expect("series admittance shape is valid");
    /// let z0 = Array2::from_elem((1, 2), Complex64::new(50.0, 0.0));
    /// let network = Network::from_y_direct_power(frequency, series_y, z0)?;
    ///
    /// assert!((network.s()[[0, 0, 0]].re - 0.5).abs() < 1.0e-14);
    /// assert!((network.s()[[0, 0, 1]].re - 0.5).abs() < 1.0e-14);
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn from_y_direct_power(
        frequency: Frequency,
        y: Array3<Complex64>,
        z0: Array2<Complex64>,
    ) -> Result<Network> {
        validate_parameter_frequency(&frequency, y.dim().0, ParameterKind::Y)?;

        let s = power_waves::y_to_s_power_direct(&y, &z0)
            .map_err(|error| map_power_wave_error(ConversionStage::YToS, error))?;
        Network::new(frequency, s, z0)
    }

    /// Converts this network's S-parameters to impedance parameters using
    /// Kurokawa power-wave semantics.
    ///
    /// The returned owned array is frequency-major with shape
    /// `(nfreq, nport, nport)`: `[[frequency, port_out, port_in]]`.  Values are
    /// expressed in ohms and retain the input frequency and port ordering.
    /// The network and its input arrays are not modified.  The conversion uses
    /// each stored reference impedance directly, including complex,
    /// per-port, and frequency-dependent values, with the existing
    /// `abs(Re(z0))` normalization.  It does not assume 50 ohms or
    /// renormalize the network.
    ///
    /// This method is provisional while `rfkit-core` is in the `0.x` series;
    /// the name and signature are not a `1.0` stability promise.  A zero real
    /// part or non-finite reference impedance is outside the Kurokawa
    /// normalization domain.  Exact singular systems are reported as errors;
    /// no near-singular tolerance or regularization is applied.
    ///
    /// # Errors
    ///
    /// Returns a structured [`Error`] preserving the `S→Z` stage and the
    /// frequency/port or matrix location of invalid data and numerical
    /// failures.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array2, Array3};
    /// use num_complex::Complex64;
    /// use rfkit_core::{Frequency, Network};
    ///
    /// # fn example() -> rfkit_core::Result<()> {
    /// let frequency = Frequency::from_hz(vec![1.0e9])?;
    /// let s = Array3::from_elem((1, 1, 1), Complex64::new(0.0, 0.0));
    /// let z0 = Array2::from_elem((1, 1), Complex64::new(75.0, 10.0));
    /// let network = Network::new(frequency, s, z0)?;
    ///
    /// let z_ohm = network.to_z_power()?;
    /// let y_siemens = network.to_y_power()?;
    /// assert_eq!(z_ohm.dim(), (1, 1, 1));
    /// assert_eq!(y_siemens.dim(), (1, 1, 1));
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn to_z_power(&self) -> Result<Array3<Complex64>> {
        power_waves::s_to_z_power(&self.s, &self.z0)
            .map_err(|error| map_power_wave_error(ConversionStage::SToZ, error))
    }

    /// Converts this network's S-parameters to admittance parameters using
    /// composed Kurokawa power-wave `S→Z→Y` semantics.
    ///
    /// The returned owned array is frequency-major with shape
    /// `(nfreq, nport, nport)`: `[[frequency, port_out, port_in]]`.  Values are
    /// expressed in siemens and retain the input frequency and port ordering.
    /// The network and its input arrays are not modified.  The conversion uses
    /// each stored reference impedance directly, including complex,
    /// per-port, and frequency-dependent values, with the existing
    /// `abs(Re(z0))` normalization.  It does not assume 50 ohms or
    /// renormalize the network.
    ///
    /// This method intentionally performs `S→Z` followed by `Z→Y`.  Therefore,
    /// an exact singularity in either stage is an error even if another direct
    /// S-to-Y formula could produce a value, and the public error identifies
    /// which stage failed.  This method is provisional while `rfkit-core` is
    /// in the `0.x` series; the name and signature are not a `1.0` stability
    /// promise.  A zero real part or non-finite reference impedance is outside
    /// the Kurokawa normalization domain.  No near-singular tolerance or
    /// regularization is applied.
    ///
    /// # Errors
    ///
    /// Returns a structured [`Error`] preserving either the `S→Z` or `Z→Y`
    /// stage and the frequency/port, matrix, pivot, or computation location of
    /// invalid data and numerical failures.
    pub fn to_y_power(&self) -> Result<Array3<Complex64>> {
        power_wave_admittance::s_to_y_power(&self.s, &self.z0)
            .map_err(map_power_wave_admittance_error)
    }

    /// Converts this network's S-parameters to admittance parameters by
    /// solving the direct Kurokawa power-wave `S→Y` equation.
    ///
    /// The returned owned array is frequency-major with shape
    /// `(nfreq, nport, nport)`: `[[frequency, port_out, port_in]]`.  Values are
    /// expressed in siemens and retain the input frequency and port ordering.
    /// The network and its input arrays are not modified.  The conversion uses
    /// each stored reference impedance directly, including complex,
    /// per-port, and frequency-dependent values, with the existing
    /// `abs(Re(z0))` normalization.  It does not assume 50 ohms or
    /// renormalize the network.
    ///
    /// For every frequency the direct path forms
    /// `A = (S G + conj(G)) F` and `B = (I - S) F`, then solves `A Y = B`,
    /// where `G = diag(z0)` and
    /// `F = diag(1/(2*sqrt(abs(Re(z0)))))`.  Unlike
    /// [`Network::to_y_power`], this method does not form the intermediate Z
    /// matrix.  An exact singularity in `I-S` is therefore not by itself an
    /// error: an ideal open produces zero Y when the direct A system is
    /// nonsingular.  Conversely, an exact singularity in A is reported.
    ///
    /// The solve uses exact-zero pivot detection only.  No explicit inverse,
    /// pseudoinverse, rank cutoff, regularization, nudge, clipping, or
    /// fallback is used.  A finite near-singular direct system remains in the
    /// domain unless its arithmetic becomes non-finite.  This method is
    /// provisional while `rfkit-core` is in the `0.x` series; its name and
    /// signature are not a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns a structured [`enum@Error`] preserving the direct `S→Y` stage and
    /// the frequency-axis, shape, reference, input, pivot, and computation
    /// context available from validation and the numerical kernel.  A
    /// serde-created empty or frequency-length-mismatched network is rejected
    /// before kernel execution.  This method's direct domain does not broaden
    /// the existing composed `to_y_power`, `to_z_power`, or renormalization
    /// domains.
    pub fn to_y_direct_power(&self) -> Result<Array3<Complex64>> {
        const STAGE: ConversionStage = ConversionStage::SToY;

        if self.frequency.is_empty() {
            return Err(Error::EmptyConversionFrequency { stage: STAGE });
        }
        let parameter_frequency_length = self.s.dim().0;
        if parameter_frequency_length != self.frequency.len() {
            return Err(Error::ConversionFrequencyLengthMismatch {
                stage: STAGE,
                expected: self.frequency.len(),
                actual: parameter_frequency_length,
            });
        }

        power_waves::s_to_y_power_direct(&self.s, &self.z0)
            .map_err(|error| map_power_wave_error(STAGE, error))
    }

    /// Re-expresses this network at explicit reference impedances using
    /// Kurokawa power-wave renormalization.
    ///
    /// The operation is composed as `S→Z` using this network's stored source
    /// references, followed by `Z→S` using `new_z0` as the target references.
    /// It therefore preserves the underlying physical impedance network while
    /// changing only the representation of the scattering parameters.  The
    /// target array must have shape `(nfreq, nport)`, where `nfreq` is this
    /// network's frequency count and `nport` is its port count.  Complex,
    /// per-port, and frequency-dependent references are accepted whenever the
    /// Kurokawa normalization domain is defined, including finite references
    /// with negative real parts; normalization uses the existing
    /// `abs(Re(z0))` rule.  Zero-real and non-finite references are rejected.
    ///
    /// The returned [`Network`] is newly owned.  Its frequency axis and port
    /// ordering are exact clones of this network, its S-parameters are the
    /// target-referenced result, and its reference impedances are exactly the
    /// supplied `new_z0`.  This method does not modify the source network.
    ///
    /// Because this is the verified two-stage conversion path, an exact
    /// singularity in either stage is an error.  Equal source and target
    /// references do not bypass a singular `S→Z` stage, and no near-singular
    /// tolerance, regularization, or identity shortcut is applied.  The
    /// public error preserves whether a failure occurred in the source `S→Z`
    /// or target `Z→S` stage together with available frequency, port, row,
    /// column, and pivot context.
    ///
    /// This method is provisional while `rfkit-core` is in the `0.x` series;
    /// its name and signature are not a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns a structured [`Error`] if `new_z0` has the wrong shape, either
    /// reference array contains invalid values, either conversion stage is
    /// exactly singular, or a non-finite intermediate/result is produced.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array2, Array3};
    /// use num_complex::Complex64;
    /// use rfkit_core::{Frequency, Network};
    ///
    /// # fn example() -> rfkit_core::Result<()> {
    /// let frequency = Frequency::from_hz(vec![1.0e9])?;
    /// let source_z0 = Array2::from_elem((1, 1), Complex64::new(50.0, 0.0));
    /// let s = Array3::from_elem((1, 1, 1), Complex64::new(0.2, -0.1));
    /// let source = Network::new(frequency, s, source_z0)?;
    /// let source_z = source.to_z_power()?;
    ///
    /// let target_z0 = Array2::from_elem((1, 1), Complex64::new(75.0, 0.0));
    /// let target = source.renormalize_power(target_z0)?;
    /// assert_eq!(target.z0()[[0, 0]], Complex64::new(75.0, 0.0));
    /// let target_z = target.to_z_power()?;
    /// assert!(target_z
    ///     .iter()
    ///     .zip(source_z.iter())
    ///     .all(|(actual, expected)| (*actual - *expected).norm() <= 1.0e-12));
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn renormalize_power(&self, new_z0: Array2<Complex64>) -> Result<Network> {
        let z = power_waves::s_to_z_power(&self.s, &self.z0)
            .map_err(|error| map_power_wave_error(ConversionStage::SToZ, error))?;
        let s = power_waves::z_to_s_power(&z, &new_z0)
            .map_err(|error| map_power_wave_error(ConversionStage::ZToS, error))?;

        // The target shape has already been checked by z_to_s_power.  Given
        // the source Network invariants, constructing the owned result cannot
        // fail; retain the public constructor as the single invariant gate.
        Network::new(self.frequency.clone(), s, new_z0)
    }

    /// Re-express this network at explicit reference impedances by changing
    /// Kurokawa power waves directly.
    ///
    /// Unlike [`Network::renormalize_power`], this operation does not form an
    /// intermediate Z or Y matrix.  With source references `G`, target
    /// references `H`, and their Kurokawa normalization matrices `F` and
    /// `F_new`, it forms the diagonal wave-change factors
    /// `K = F_new F⁻¹ (2 Re(G))⁻¹` and solves
    ///
    /// ```text
    /// S_new (D + E S) = C + J S
    /// D = K (conj(G) + H)       E = K (G - H)
    /// C = K (conj(G) - conj(H)) J = K (G + conj(H)).
    /// ```
    ///
    /// The right system is solved with the existing exact-pivot
    /// multiple-right-hand-side solver after a plain transpose.  Complex,
    /// per-port, frequency-dependent, and negative-real source/target
    /// references are supported whenever both real parts are finite and
    /// nonzero.  A singular `I-S` or `I+S` by itself is not a failure; only an
    /// exactly singular direct wave-change system or non-finite arithmetic is
    /// rejected.  No Z/Y intermediate, explicit inverse, pseudoinverse,
    /// cutoff, regularization, identity shortcut, or fallback is used.
    ///
    /// The returned network is newly owned.  It preserves the source
    /// frequency and port order exactly and stores an exact owned copy of
    /// `new_z0`; the source network is not modified.  Frequency samples are
    /// pointwise labels and only need to have a nonempty axis matching the S
    /// first dimension.  They need not be finite, sorted, unique, or ordered.
    ///
    /// This is an additive, provisional 0.x API.  It intentionally coexists
    /// with [`Network::renormalize_power`], whose composed S→Z→S validation
    /// and singular-stage errors remain unchanged.
    ///
    /// # Errors
    ///
    /// Returns a direct-renormalization-specific [`enum@Error`] for malformed
    /// serde-created shapes, empty or mismatched axes, non-finite S or
    /// references, zero-real references, exact direct-system singularities,
    /// and non-finite arithmetic.  Errors identify source versus target
    /// references where applicable and never report a fictitious S→Z or Z→S
    /// stage.
    pub fn renormalize_direct_power(&self, new_z0: Array2<Complex64>) -> Result<Network> {
        if self.frequency.is_empty() {
            return Err(Error::EmptyDirectRenormalizationFrequency);
        }
        let parameter_frequency_length = self.s.dim().0;
        if parameter_frequency_length != self.frequency.len() {
            return Err(Error::DirectRenormalizationFrequencyLengthMismatch {
                expected: self.frequency.len(),
                actual: parameter_frequency_length,
            });
        }

        let s = power_waves::renormalize_s_power_direct(&self.s, &self.z0, &new_z0)
            .map_err(map_direct_renormalization_error)?;

        Network::new(self.frequency.clone(), s, new_z0)
    }

    /// Returns the power-wave inverse of an ordered even-port cascade.
    ///
    /// The source ports are interpreted as two equal ordered groups,
    /// `[left_0..left_(N-1), right_0..right_(N-1)]`.  The returned network
    /// uses the fixed reversed group order `[old right, old left]`; use
    /// [`Network::permute_ports`] first when the physical fixture uses another
    /// ordering.  With `P` the group-exchange matrix, the S-parameters are
    /// evaluated as `S_inverse = P S⁻¹ P`, and the references are exactly
    /// `P conj(z0)`.
    ///
    /// This is a wave-reversal operation, not an elementwise reciprocal and
    /// not a port permutation of the source S matrix.  The implementation
    /// computes and stores the full `S⁻¹` by solving `S X = I` with the core
    /// checked exact-pivot solver.  It does not use an elementwise reciprocal
    /// or silently convert through Z/Y.  In addition to the full S system,
    /// both directional N-by-N transmission blocks are solved and must be
    /// nonsingular: `S[right,left]` is the forward stage and
    /// `S[left,right]` is the reverse stage.  Only exact evaluated zero pivots
    /// are singular, so finite near-singular systems remain eligible when all
    /// arithmetic stays finite.
    ///
    /// The operation uses Kurokawa power-wave reversal with currents directed
    /// into the source network.  It requires finite frequency labels and S/z0
    /// values, and finite references with nonzero real parts.  Frequencies are
    /// pointwise labels: their original order and bit patterns (including
    /// signed zero, negative, duplicate, and descending samples) are copied
    /// exactly.  Unequal, per-port, frequency-dependent, complex, and
    /// negative-real references are supported under the repository's
    /// algebraic `abs(Re(z0))` normalization; negative-real values do not
    /// imply a passive-power interpretation.
    ///
    /// Inverse networks can be active or noncausal mathematical removal
    /// operators rather than realizable passive devices.  Cascading one with a
    /// fixture cancels that fixture only for the declared orientation, paired
    /// physical ports, compatible frequency grids, and nonsingular connection
    /// conditions.  No noise de-embedding, automatic calibration, pole or
    /// stability claim, or measurement-error correction is implied.  For
    /// complex references, the conjugated swapped references are intentional;
    /// callers should explicitly renormalize when comparing the recovered DUT
    /// at another reference.
    ///
    /// This additive operation is provisional during the `0.x` series.  Its
    /// name and signature are not a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns inverse-cascade-specific structured diagnostics for malformed
    /// serde-created axes/shapes, odd or zero port counts, non-finite labels or
    /// data, zero-real references, exact singular full-S or transmission
    /// systems, and non-finite arithmetic.  Numerical diagnostics retain the
    /// failing [`InverseCascadeStage`] plus frequency, row/column, or pivot
    /// context where available.
    pub fn inverse_cascade_power(&self) -> Result<Network> {
        let (s, z0) =
            inverse_cascade::inverse_cascade_power(self.frequency.hz(), &self.s, &self.z0)
                .map_err(map_inverse_cascade_error)?;

        let frequency = self.frequency.clone();
        Network::new(frequency, s, z0)
    }

    /// Resamples this network onto an explicit frequency grid using
    /// component-wise Cartesian linear interpolation.
    ///
    /// Every real and imaginary component of the frequency-major S-parameter
    /// stack and of the frequency-major reference-impedance array is
    /// interpolated independently.  The source axis must contain at least two
    /// finite, strictly increasing samples.  `target` must be non-empty,
    /// finite, strictly increasing, and wholly within the inclusive source
    /// span.  No sorting, deduplication, extrapolation, automatic grid
    /// alignment, or reference-impedance renormalization is performed.
    ///
    /// A target frequency that exactly equals a source knot, including either
    /// endpoint, copies the complete source S and z0 slices without
    /// interpolation arithmetic.  Finite complex reference impedances are
    /// accepted as stored, including non-50-ohm, zero-real, and negative-real
    /// values; interpolation does not apply the domain checks used by
    /// power-wave conversions.  Finite negative frequencies and the existing
    /// signed-zero knot equality behavior are retained.
    ///
    /// The returned network is newly owned, preserves the source port count
    /// and ordering, and uses the target frequency values exactly.  Neither
    /// this network nor `target` is modified.  Interpolation is a component-
    /// wise resampling operation only; this method makes no claim about
    /// preserving physical Z-parameters, passivity, or causality.
    ///
    /// This method is provisional while `rfkit-core` is in the `0.x` series;
    /// its name and signature are not a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns a structured [`Error`] when the source or target axis violates
    /// the restrictions above, an S/z0 shape is invalid, source data is
    /// non-finite, a computed weight is non-finite, or a computed component
    /// is non-finite.  Interpolation failures remain distinct from the
    /// power-conversion error stages.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array2, Array3};
    /// use num_complex::Complex64;
    /// use rfkit_core::{Frequency, Network};
    ///
    /// # fn example() -> rfkit_core::Result<()> {
    /// let source_frequency = Frequency::from_hz(vec![1.0e9, 2.0e9])?;
    /// let source = Network::new(
    ///     source_frequency,
    ///     Array3::from_shape_fn((2, 1, 1), |(frequency, _, _)| {
    ///         Complex64::new(0.1 + 0.2 * frequency as f64, -0.05)
    ///     }),
    ///     Array2::from_shape_fn((2, 1), |(frequency, _)| {
    ///         Complex64::new(50.0 + 10.0 * frequency as f64, 2.0)
    ///     }),
    /// )?;
    /// let target_frequency = Frequency::from_hz(vec![1.25e9, 1.75e9])?;
    ///
    /// let resampled = source.interpolate_cartesian_linear(&target_frequency)?;
    /// assert_eq!(resampled.frequency(), &target_frequency);
    /// assert_eq!(resampled.s().dim(), (2, 1, 1));
    /// assert_eq!(resampled.z0().dim(), (2, 1));
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn interpolate_cartesian_linear(&self, target: &Frequency) -> Result<Network> {
        let interpolated = interpolation::interpolate_cartesian_linear(
            self.frequency.hz(),
            &self.s,
            &self.z0,
            target.hz(),
        )
        .map_err(map_interpolation_error)?;

        let frequency = Frequency::from_hz(interpolated.frequency_hz)?;
        Network::new(frequency, interpolated.s, interpolated.z0)
    }

    /// Connects one port of this network to one port of `other` through a
    /// matched Kurokawa power-wave junction.
    ///
    /// The operation uses the existing exact-grid matched-connection kernel.
    /// The two frequency axes must both be non-empty and finite, have equal
    /// lengths, and contain exactly equal values at corresponding indices.
    /// Equality is Rust's `f64` equality: finite negative, duplicate, and
    /// descending samples are accepted, and `-0.0` equals `+0.0`.  A
    /// single-frequency grid is valid.  No grid is selected or inferred; this
    /// method does not sort, deduplicate, interpolate, extrapolate, or
    /// otherwise resample either input.  A's frequency values
    /// (`self.frequency()`) are copied into the result exactly.
    ///
    /// The selected reference impedances must be finite, real, strictly
    /// positive, and exactly equal at every frequency.  They are the matched
    /// junction impedance, and may be non-50-ohm and frequency-dependent.
    /// This requirement is distinct from the surviving external references:
    /// those values may be any finite complex impedances admitted by the
    /// connection kernel and are copied without conversion-specific
    /// positive-real restrictions.
    ///
    /// If `k` and `l` are the selected ports of A and B, and `EA` and `EB` are
    /// the surviving ports in their original order, the elimination equation
    /// is
    ///
    /// ```text
    /// D     = 1 - S_A[k,k] S_B[l,l]
    /// C_AA = S_A[EA,EA] + S_A[EA,k] S_B[l,l] S_A[k,EA] / D
    /// C_AB = S_A[EA,k] S_B[l,EB] / D
    /// C_BA = S_B[EB,l] S_A[k,EA] / D
    /// C_BB = S_B[EB,EB] + S_B[EB,l] S_A[k,k] S_B[l,EB] / D.
    /// ```
    ///
    /// The denominator is classified as singular only when its computed
    /// complex value is exactly zero.  No conditioning threshold,
    /// regularization, or mismatch renormalization is applied.  The returned
    /// network is newly owned, has `self.nports() + other.nports() - 2`
    /// ports, and orders them as A survivors followed by B survivors.  Its
    /// survivor `z0` values are copied in that same order.  Neither input,
    /// their arrays, nor their frequency axes is modified.  Ports are
    /// zero-based.
    ///
    /// This operation is provisional while `rfkit-core` is in the `0.x`
    /// series; its name and signature are not a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns a structured connection-specific [`Error`].  `A`
    /// ([`ConnectionInput::A`]) identifies `self` and kernel input A; `B`
    /// ([`ConnectionInput::B`]) identifies `other` and kernel input B.  The
    /// error preserves invalid port/count, exact-grid, source-data location,
    /// junction impedance, no-survivor, exact-singularity, and non-finite
    /// computation context.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array2, Array3};
    /// use num_complex::Complex64;
    /// use rfkit_core::{Frequency, Network};
    ///
    /// # fn example() -> rfkit_core::Result<()> {
    /// let frequency = Frequency::from_hz(vec![1.0e9])?;
    /// let left = Network::new(
    ///     frequency.clone(),
    ///     Array3::zeros((1, 2, 2)),
    ///     Array2::from_elem((1, 2), Complex64::new(50.0, 0.0)),
    /// )?;
    /// let right = Network::new(
    ///     frequency,
    ///     Array3::zeros((1, 2, 2)),
    ///     Array2::from_elem((1, 2), Complex64::new(50.0, 0.0)),
    /// )?;
    ///
    /// let connected = left.connect_matched_power(0, &right, 0)?;
    /// assert_eq!(connected.nports(), 2);
    /// assert_eq!(connected.s().dim(), (1, 2, 2));
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn connect_matched_power(
        &self,
        port: usize,
        other: &Network,
        other_port: usize,
    ) -> Result<Network> {
        let connected = connection::connect_matched(
            self.frequency.hz(),
            &self.s,
            &self.z0,
            port,
            other.frequency.hz(),
            &other.s,
            &other.z0,
            other_port,
        )
        .map_err(map_connection_error)?;

        let frequency = Frequency::from_hz(connected.frequency_hz)?;
        Network::new(frequency, connected.s, connected.z0)
    }

    /// Connects one port of this network directly to one port of `other` by
    /// physical voltage continuity and current conservation.
    ///
    /// The selected ports use currents directed into their respective
    /// networks, and the junction conditions are `V_A = V_B` and
    /// `I_A + I_B = 0`.  The operation uses the repository's Kurokawa
    /// power-wave equations with
    /// `q = sqrt(abs(Re(z0))) / Re(z0)`, retaining the signed real part of
    /// each reference.  This is a direct two-coordinate elimination; it does
    /// not convert through Z/Y, insert a mismatch network, renormalize either
    /// input, or choose a fixed reference impedance.
    ///
    /// `port_a` and `port_b` are zero-based selected coordinates.  The
    /// result contains the survivors of `self` in their original order,
    /// followed by the survivors of `other` in their original order.  Thus a
    /// one-port input is valid when the other input contributes at least one
    /// survivor, and no special two-port ordering or insertion rule is used.
    /// Surviving references are copied exactly in that same order.  The result
    /// owns a bit-for-bit copy of this network's frequency axis; both inputs
    /// and all caller-owned data remain unchanged.
    ///
    /// Both frequency axes must be non-empty, finite, equal in length, and
    /// equal at corresponding positions by ordinary `f64` equality.  Finite
    /// negative, duplicate, descending, and signed-zero samples are valid.
    /// Every S and reference value must be finite, and every reference in both
    /// networks (not only selected ports) must have a nonzero real part.
    /// Unequal, complex, per-port, frequency-dependent, and negative-real
    /// references are all in-domain.  Only exact evaluated zero pivots in the
    /// two-by-two direct junction system are singular; finite near-singular
    /// systems remain valid.  Checked arithmetic reports non-finite
    /// intermediate values rather than returning a non-finite network.
    ///
    /// This additive operation is provisional while `rfkit-core` is in the
    /// `0.x` series; its name and signature are not a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns direct-connection-specific structured errors for malformed
    /// serde-created shapes, axis mismatches, invalid ports or survivor count,
    /// non-finite inputs, zero-real references, exact junction singularity,
    /// and non-finite arithmetic.  The selected-port and pivot coordinates
    /// are retained for numerical failures.
    pub fn connect_direct_power(
        &self,
        port_a: usize,
        other: &Network,
        port_b: usize,
    ) -> Result<Network> {
        let connected = direct_connection::connect_direct(
            self.frequency.hz(),
            &self.s,
            &self.z0,
            port_a,
            other.frequency.hz(),
            &other.s,
            &other.z0,
            port_b,
        )
        .map_err(map_direct_connection_error)?;

        let frequency = Frequency::from_hz(connected.frequency_hz)?;
        Network::new(frequency, connected.s, connected.z0)
    }

    /// Connects one port of this network to one port of `other` through a
    /// matched Kurokawa power-wave junction after interpolating both networks
    /// onto an explicit target frequency grid.
    ///
    /// The operation stages are deliberately ordered: all of A (`self`) is
    /// interpolated first, all of B (`other`) is interpolated second, and the
    /// existing exact-grid matched connection is evaluated only after both
    /// stages succeed.  The target grid is never inferred, intersected,
    /// sorted, subset, or extrapolated.  Even when a source and target grid
    /// have equal values, each source still goes through the interpolation
    /// contract; in particular, a source with fewer than two samples is not
    /// accepted as a same-grid shortcut.
    ///
    /// Cartesian linear interpolation is applied independently to the real
    /// and imaginary components of both S and `z0`.  Each source axis must
    /// contain at least two finite, strictly increasing samples.  `target`
    /// must be non-empty, finite, strictly increasing, and within both
    /// inclusive source spans.  A one-sample target is valid.  Exact source
    /// knots copy the complete source S/`z0` slices, and the returned
    /// frequency axis copies `target` exactly, including signed zero.
    ///
    /// The final connection uses the existing Kurokawa power-wave
    /// matched-junction contract.  Selected junction references must be
    /// finite, real, strictly positive, and exactly equal at every target
    /// frequency after interpolation; no tolerance-based matching or hidden
    /// renormalization is performed.  Non-50-ohm and frequency-dependent
    /// matched junctions are valid, and surviving ports may retain finite
    /// complex reference impedances.  Output ports are the original A
    /// survivors followed by the original B survivors.
    ///
    /// A connection denominator is singular only when its computed complex
    /// value is exactly zero.  No conditioning threshold or regularization is
    /// used.  The result is newly owned and neither input network nor `target`
    /// is modified.  This operation is provisional while `rfkit-core` is in
    /// the `0.x` series; its name and signature are not a `1.0` stability
    /// promise.
    ///
    /// # Errors
    ///
    /// Errors are wrapped in [`Error::GridConnection`] with a
    /// [`GridConnectionStage`] identifying A interpolation, B interpolation,
    /// or the final matched connection.  The nested error preserves the
    /// detailed shape/axis/index/value, selected-port, junction,
    /// no-survivor, singularity, and checked-computation context from the
    /// underlying operation.  Because interpolation precedes port validation,
    /// an invalid port is reported only after both source interpolations have
    /// succeeded.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array2, Array3};
    /// use num_complex::Complex64;
    /// use rfkit_core::{Frequency, Network};
    ///
    /// # fn example() -> rfkit_core::Result<()> {
    /// let left = Network::new(
    ///     Frequency::from_hz(vec![1.0e9, 2.0e9])?,
    ///     Array3::zeros((2, 2, 2)),
    ///     Array2::from_elem((2, 2), Complex64::new(50.0, 0.0)),
    /// )?;
    /// let right = Network::new(
    ///     Frequency::from_hz(vec![1.0e9, 1.5e9, 2.0e9])?,
    ///     Array3::zeros((3, 2, 2)),
    ///     Array2::from_elem((3, 2), Complex64::new(50.0, 0.0)),
    /// )?;
    /// let target = Frequency::from_hz(vec![1.0e9, 1.25e9, 2.0e9])?;
    /// let connected = left.connect_matched_power_on_grid(0, &right, 0, &target)?;
    /// assert_eq!(connected.frequency(), &target);
    /// assert_eq!(connected.nports(), 2);
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn connect_matched_power_on_grid(
        &self,
        port: usize,
        other: &Network,
        other_port: usize,
        target: &Frequency,
    ) -> Result<Network> {
        let connected = composition::connect_matched_on_grid(
            self.frequency.hz(),
            &self.s,
            &self.z0,
            port,
            other.frequency.hz(),
            &other.s,
            &other.z0,
            other_port,
            target.hz(),
        )
        .map_err(map_composition_error)?;

        let frequency = Frequency::from_hz(connected.frequency_hz)?;
        Network::new(frequency, connected.s, connected.z0)
    }

    /// Connects two distinct ports of this network through a matched
    /// Kurokawa power-wave junction and returns the remaining network.
    ///
    /// For selected ports in internal order `[port_a, port_b]`, the matched
    /// junction exchanges the two incident waves with
    /// `P = [[0, 1], [1, 0]]`.  Eliminating the selected internal waves uses
    /// the equation
    ///
    /// ```text
    /// S_out = S_EE + S_EI (I - P S_II)^-1 P S_IE.
    /// ```
    ///
    /// `port_a` and `port_b` are zero-based and must be distinct and in range.
    /// At least one survivor must remain.  The output contains the survivors
    /// in their original order, with their reference impedances copied
    /// exactly.  The result, including its frequency axis and arrays, is newly
    /// owned; this network and its arrays are not modified.
    ///
    /// The source frequency axis must be non-empty and finite.  A single
    /// sample is valid, as are finite negative, duplicate, and descending
    /// samples and signed zero.  The axis is copied exactly, without sorting,
    /// deduplication, interpolation, or resampling.  The selected junction
    /// reference impedances must be finite, real, strictly positive, and
    /// exactly equal at every frequency.  They may be non-50-ohm and
    /// frequency-dependent.  External survivor reference impedances may be
    /// any finite complex values admitted by the matched connection kernel and
    /// are copied without conversion-specific positive-real restrictions.
    ///
    /// The private kernel classifies only exact-zero pivots as singular.  No
    /// near-singular threshold, regularization, determinant/rank fallback, or
    /// mismatch renormalization is applied.  Checked arithmetic reports
    /// non-finite computations instead of returning non-finite output.
    ///
    /// This operation is provisional while `rfkit-core` is in the `0.x`
    /// series; its name and signature are not a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns a structured [`Error`] preserving invalid source-data shapes,
    /// frequency and S/z0 coordinates, selected-port validation, junction
    /// identities and values, no-survivor, exact-singularity, and checked
    /// computation context.  Computation row and column fields refer to the
    /// kernel's generic arithmetic coordinates (including its internal 2×2
    /// solve), not to output-network coordinates.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array2, Array3};
    /// use num_complex::Complex64;
    /// use rfkit_core::{Frequency, Network};
    ///
    /// # fn example() -> rfkit_core::Result<()> {
    /// let frequency = Frequency::from_hz(vec![1.0e9])?;
    /// let network = Network::new(
    ///     frequency,
    ///     Array3::zeros((1, 3, 3)),
    ///     Array2::from_elem((1, 3), Complex64::new(50.0, 0.0)),
    /// )?;
    /// let reduced = network.inner_connect_matched_power(0, 1)?;
    /// assert_eq!(reduced.nports(), 1);
    /// assert_eq!(reduced.s().dim(), (1, 1, 1));
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn inner_connect_matched_power(&self, port_a: usize, port_b: usize) -> Result<Network> {
        let connected = connection::inner_connect_matched(
            self.frequency.hz(),
            &self.s,
            &self.z0,
            port_a,
            port_b,
        )
        .map_err(|error| map_inner_connection_error(error, port_a, port_b))?;

        let frequency = Frequency::from_hz(connected.frequency_hz)?;
        Network::new(frequency, connected.s, connected.z0)
    }

    /// Connects two distinct ports of this network by direct physical voltage
    /// continuity and current conservation under Kurokawa power waves.
    ///
    /// For selected internal coordinates `i = [port_a, port_b]` and the
    /// remaining external coordinates `e`, the source relation is partitioned
    /// as `b_i = S_ii a_i + S_ie a_e` and
    /// `b_e = S_ei a_i + S_ee a_e`.  The junction equations use currents
    /// directed into this network:
    ///
    /// ```text
    /// a = (V + z I)/(2 sqrt(abs(Re(z))))
    /// b = (V - conj(z) I)/(2 sqrt(abs(Re(z))))
    /// q = sqrt(abs(Re(z))) / Re(z)
    /// I = q (a - b)
    /// V = q (conj(z) a + z b)
    ///
    /// C = [[ qa,             qb           ],
    ///      [ qa*conj(za), -qb*conj(zb)  ]]
    /// D = [[-qa,            -qb           ],
    ///      [ qa*za,         -qb*zb      ]]
    ///
    /// (C + D*S_ii) T = -D*S_ie
    /// S_out = S_ee + S_ei*T
    /// ```
    ///
    /// The complete two-by-two internal block is used, including both
    /// off-diagonal couplings.  This is a direct two-coordinate elimination:
    /// it does not convert through S/Z/Y, divide by `za + zb`, insert a
    /// mismatch network, renormalize, regularize, or choose a fixed reference.
    /// The selected reference impedances may be unequal or equal, complex,
    /// frequency-dependent, or have negative real parts, provided every
    /// reference is finite with a nonzero real part.  The signed real part is
    /// retained in `q`.
    ///
    /// The result contains every non-selected port in the source's original
    /// order, with each surviving reference copied exactly.  The source and
    /// all caller-owned data remain unchanged.  Frequencies are finite
    /// pointwise labels and are copied bit-for-bit, including signed zero;
    /// negative, duplicate, and descending samples are valid.  Only exact
    /// zero pivots in the evaluated two-by-two system are singular.  Finite
    /// near-singular systems remain in-domain, while checked arithmetic
    /// rejects non-finite intermediate or output values.
    ///
    /// This additive operation is provisional while `rfkit-core` is in the
    /// `0.x` series; its name and signature are not a `1.0` stability promise.
    ///
    /// # Errors
    ///
    /// Returns direct-inner-connection-specific structured errors for malformed
    /// serde-created shapes, invalid frequency/S/z0 data, invalid or equal
    /// ports, no survivors, zero-real references, exact junction singularity,
    /// and non-finite arithmetic.  Numerical errors retain selected-port,
    /// pivot, frequency, and computation row/column context.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::{Array2, Array3};
    /// use num_complex::Complex64;
    /// use rfkit_core::{Frequency, Network};
    ///
    /// # fn example() -> rfkit_core::Result<()> {
    /// let network = Network::new(
    ///     Frequency::from_hz(vec![1.0e9])?,
    ///     Array3::zeros((1, 3, 3)),
    ///     Array2::from_elem((1, 3), Complex64::new(50.0, 0.0)),
    /// )?;
    /// let reduced = network.inner_connect_direct_power(0, 1)?;
    /// assert_eq!(reduced.nports(), 1);
    /// assert_eq!(reduced.s().dim(), (1, 1, 1));
    /// # Ok(())
    /// # }
    /// # example().unwrap();
    /// ```
    pub fn inner_connect_direct_power(&self, port_a: usize, port_b: usize) -> Result<Network> {
        let connected = direct_connection::inner_connect_direct(
            self.frequency.hz(),
            &self.s,
            &self.z0,
            port_a,
            port_b,
        )
        .map_err(map_direct_inner_connection_error)?;

        let frequency = Frequency::from_hz(connected.frequency_hz)?;
        Network::new(frequency, connected.s, connected.z0)
    }

    /// Applies one finite physical impedance to a selected source port and
    /// returns the reduced network with that port removed.
    ///
    /// The operation uses currents directed into this network and Kurokawa
    /// power waves.  For selected port `k`, source reference `z_k`, and load
    /// `ZL`, the physical boundary is eliminated directly with
    /// `c = ZL - z_k`, `d = ZL + conj(z_k)`, and
    /// `den = d - c*S[k,k]`:
    ///
    /// ```text
    /// S_out = S[E,E] + S[E,k] * (c/den) * S[k,E]
    /// ```
    ///
    /// `load_ohm` supplies exactly one finite complex impedance in ohms per
    /// source frequency.  Zero, purely reactive, and negative-resistance
    /// loads are valid; an open-circuit sentinel is not part of this finite
    /// impedance operation.  Source references may be complex or have
    /// negative real parts, but every source reference must be finite with a
    /// nonzero real part for the Kurokawa normalization domain.
    ///
    /// The source frequency samples are opaque pointwise labels and are
    /// copied exactly.  Survivors retain their original order and exact
    /// source references.  The source network and `load_ohm` are borrowed and
    /// unchanged.  No S/Z/Y conversion, renormalization, tolerance cutoff,
    /// pseudoinverse, or fallback is used.  Only an exactly zero evaluated
    /// `den` is singular; a finite nonzero near-singular denominator remains
    /// valid.  In particular, `d == 0` is valid when `den != 0` because the
    /// implementation never forms `c/d`.
    ///
    /// This additive operation is provisional while `rfkit-core` is in the
    /// `0.x` series; its name and signature are not a `1.0` stability
    /// promise.
    ///
    /// # Errors
    ///
    /// Returns a structured [`Error`] for malformed serde-created source
    /// axes/shapes, a one-port source or invalid selected port, a mismatched
    /// load length, non-finite S/references/load values, zero-real source
    /// references, exact denominator singularity, or non-finite arithmetic.
    pub fn terminate_port_impedance_power(
        &self,
        port: usize,
        load_ohm: &[Complex64],
    ) -> Result<Network> {
        let terminated = termination::terminate_port_impedance_power(
            self.frequency.hz(),
            &self.s,
            &self.z0,
            port,
            load_ohm,
        )
        .map_err(map_termination_error)?;

        let frequency = Frequency::from_hz(terminated.frequency_hz)?;
        Network::new(frequency, terminated.s, terminated.z0)
    }
}

fn validate_parameter_frequency(
    frequency: &Frequency,
    parameter_frequency_length: usize,
    parameter: ParameterKind,
) -> Result<()> {
    if frequency.is_empty() {
        return Err(Error::EmptyParameterFrequency { parameter });
    }
    if parameter_frequency_length != frequency.len() {
        return Err(Error::ParameterFrequencyLengthMismatch {
            parameter,
            expected: frequency.len(),
            actual: parameter_frequency_length,
        });
    }
    Ok(())
}

fn map_power_wave_admittance_error(
    error: power_wave_admittance::PowerWaveAdmittanceError,
) -> Error {
    match error {
        power_wave_admittance::PowerWaveAdmittanceError::SToZ(error) => {
            map_power_wave_error(ConversionStage::SToZ, error)
        }
        power_wave_admittance::PowerWaveAdmittanceError::ZToY(error) => {
            map_impedance_admittance_error(ConversionStage::ZToY, error)
        }
        power_wave_admittance::PowerWaveAdmittanceError::YToZ(error) => {
            map_impedance_admittance_error(ConversionStage::YToZ, error)
        }
        power_wave_admittance::PowerWaveAdmittanceError::ZToS(error) => {
            map_power_wave_error(ConversionStage::ZToS, error)
        }
    }
}

fn map_interpolation_error(error: interpolation::InterpolationError) -> Error {
    match error {
        interpolation::InterpolationError::InvalidSShape { shape } => {
            Error::InvalidInterpolationShape {
                quantity: InterpolationQuantity::S,
                shape: vec![shape.0, shape.1, shape.2],
            }
        }
        interpolation::InterpolationError::SourceFrequencyShape { expected, actual } => {
            Error::InterpolationSourceFrequencyLengthMismatch { expected, actual }
        }
        interpolation::InterpolationError::InvalidZ0Shape { shape } => {
            Error::InvalidInterpolationShape {
                quantity: InterpolationQuantity::Z0,
                shape: vec![shape.0, shape.1],
            }
        }
        interpolation::InterpolationError::TooFewSourceSamples { actual } => {
            Error::InterpolationTooFewSourceSamples { actual }
        }
        interpolation::InterpolationError::EmptyTarget => Error::InterpolationEmptyTarget,
        interpolation::InterpolationError::NonFiniteSourceFrequency { index, value } => {
            Error::NonFiniteInterpolationFrequency {
                axis: InterpolationAxis::Source,
                index,
                value,
            }
        }
        interpolation::InterpolationError::SourceFrequencyNotStrictlyIncreasing {
            index,
            previous,
            current,
        } => Error::InterpolationFrequencyNotStrictlyIncreasing {
            axis: InterpolationAxis::Source,
            index,
            previous,
            current,
        },
        interpolation::InterpolationError::NonFiniteTargetFrequency { index, value } => {
            Error::NonFiniteInterpolationFrequency {
                axis: InterpolationAxis::Target,
                index,
                value,
            }
        }
        interpolation::InterpolationError::TargetFrequencyNotStrictlyIncreasing {
            index,
            previous,
            current,
        } => Error::InterpolationFrequencyNotStrictlyIncreasing {
            axis: InterpolationAxis::Target,
            index,
            previous,
            current,
        },
        interpolation::InterpolationError::TargetFrequencyOutOfRange {
            index,
            value,
            lower,
            upper,
        } => Error::InterpolationTargetOutOfRange {
            index,
            value,
            lower,
            upper,
        },
        interpolation::InterpolationError::NonFiniteS {
            frequency,
            row,
            column,
        } => Error::NonFiniteInterpolationS {
            frequency,
            row,
            column,
        },
        interpolation::InterpolationError::NonFiniteZ0 { frequency, port } => {
            Error::NonFiniteInterpolationZ0 { frequency, port }
        }
        interpolation::InterpolationError::NonFiniteWeight {
            target,
            lower_source,
            upper_source,
        } => Error::NonFiniteInterpolationWeight {
            target,
            lower_source,
            upper_source,
        },
        interpolation::InterpolationError::NonFiniteComputation {
            quantity,
            target,
            row,
            column,
        } => Error::NonFiniteInterpolationComputation {
            quantity: match quantity {
                interpolation::InterpolationQuantity::S => InterpolationQuantity::S,
                interpolation::InterpolationQuantity::Z0 => InterpolationQuantity::Z0,
            },
            target,
            row,
            column,
        },
    }
}

fn map_composition_error(error: composition::CompositionError) -> Error {
    match error {
        composition::CompositionError::AInterpolation(error) => Error::GridConnection {
            stage: GridConnectionStage::AInterpolation,
            source: Box::new(map_interpolation_error(error)),
        },
        composition::CompositionError::BInterpolation(error) => Error::GridConnection {
            stage: GridConnectionStage::BInterpolation,
            source: Box::new(map_interpolation_error(error)),
        },
        composition::CompositionError::Connection(error) => Error::GridConnection {
            stage: GridConnectionStage::Connection,
            source: Box::new(map_connection_error(error)),
        },
    }
}

fn map_connection_input(input: connection::NetworkSide) -> ConnectionInput {
    match input {
        connection::NetworkSide::A => ConnectionInput::A,
        connection::NetworkSide::B => ConnectionInput::B,
    }
}

fn map_connection_error(error: connection::ConnectionError) -> Error {
    match error {
        connection::ConnectionError::InvalidSShape { network, shape } => {
            Error::InvalidConnectionSShape {
                input: map_connection_input(network),
                shape: vec![shape.0, shape.1, shape.2],
            }
        }
        connection::ConnectionError::InvalidZ0Shape { network, shape } => {
            Error::InvalidConnectionZ0Shape {
                input: map_connection_input(network),
                shape: vec![shape.0, shape.1],
            }
        }
        connection::ConnectionError::EmptyFrequency { network } => {
            Error::EmptyConnectionFrequency {
                input: map_connection_input(network),
            }
        }
        connection::ConnectionError::FrequencyShape {
            network,
            expected,
            actual,
        } => Error::ConnectionFrequencyShape {
            input: map_connection_input(network),
            expected,
            actual,
        },
        connection::ConnectionError::FrequencyLengthMismatch { a, b } => {
            Error::ConnectionFrequencyLengthMismatch { a, b }
        }
        connection::ConnectionError::FrequencyMismatch { index, a, b } => {
            Error::ConnectionFrequencyMismatch { index, a, b }
        }
        connection::ConnectionError::NonFiniteFrequency {
            network,
            index,
            value,
        } => Error::NonFiniteConnectionFrequency {
            input: map_connection_input(network),
            index,
            value,
        },
        connection::ConnectionError::NonFiniteS {
            network,
            frequency,
            row,
            column,
        } => Error::NonFiniteConnectionS {
            input: map_connection_input(network),
            frequency,
            row,
            column,
        },
        connection::ConnectionError::NonFiniteZ0 {
            network,
            frequency,
            port,
        } => Error::NonFiniteConnectionZ0 {
            input: map_connection_input(network),
            frequency,
            port,
        },
        connection::ConnectionError::InvalidPort {
            network,
            port,
            nports,
        } => Error::InvalidConnectionPort {
            input: map_connection_input(network),
            port,
            nports,
        },
        connection::ConnectionError::InvalidInnerPort { .. }
        | connection::ConnectionError::IdenticalInnerPorts { .. } => {
            unreachable!("two-network connection kernel returned an inner-connect error")
        }
        connection::ConnectionError::InvalidJunctionZ0 {
            network,
            frequency,
            port,
            value,
        } => Error::InvalidConnectionJunctionZ0 {
            input: map_connection_input(network),
            frequency,
            port,
            value,
        },
        connection::ConnectionError::MismatchedJunctionZ0 { frequency, a, b } => {
            Error::MismatchedConnectionJunctionZ0 { frequency, a, b }
        }
        connection::ConnectionError::NoExternalPorts => Error::NoExternalConnectionPorts,
        connection::ConnectionError::Singular { frequency } => {
            Error::SingularConnection { frequency }
        }
        connection::ConnectionError::NonFiniteComputation {
            frequency,
            row,
            column,
        } => Error::NonFiniteConnectionComputation {
            frequency,
            row,
            column,
        },
    }
}

fn map_direct_connection_input(input: direct_connection::NetworkSide) -> ConnectionInput {
    match input {
        direct_connection::NetworkSide::A => ConnectionInput::A,
        direct_connection::NetworkSide::B => ConnectionInput::B,
    }
}

fn map_direct_connection_error(error: direct_connection::DirectConnectionError) -> Error {
    match error {
        direct_connection::DirectConnectionError::InvalidSShape { network, shape } => {
            Error::InvalidDirectConnectionSShape {
                input: map_direct_connection_input(network),
                shape: vec![shape.0, shape.1, shape.2],
            }
        }
        direct_connection::DirectConnectionError::InvalidZ0Shape { network, shape } => {
            Error::InvalidDirectConnectionZ0Shape {
                input: map_direct_connection_input(network),
                shape: vec![shape.0, shape.1],
            }
        }
        direct_connection::DirectConnectionError::EmptyFrequency { network } => {
            Error::EmptyDirectConnectionFrequency {
                input: map_direct_connection_input(network),
            }
        }
        direct_connection::DirectConnectionError::FrequencyShape {
            network,
            expected,
            actual,
        } => Error::DirectConnectionFrequencyShape {
            input: map_direct_connection_input(network),
            expected,
            actual,
        },
        direct_connection::DirectConnectionError::FrequencyLengthMismatch { a, b } => {
            Error::DirectConnectionFrequencyLengthMismatch { a, b }
        }
        direct_connection::DirectConnectionError::FrequencyMismatch { index, a, b } => {
            Error::DirectConnectionFrequencyMismatch { index, a, b }
        }
        direct_connection::DirectConnectionError::NonFiniteFrequency {
            network,
            index,
            value,
        } => Error::NonFiniteDirectConnectionFrequency {
            input: map_direct_connection_input(network),
            index,
            value,
        },
        direct_connection::DirectConnectionError::NonFiniteS {
            network,
            frequency,
            row,
            column,
        } => Error::NonFiniteDirectConnectionS {
            input: map_direct_connection_input(network),
            frequency,
            row,
            column,
        },
        direct_connection::DirectConnectionError::NonFiniteZ0 {
            network,
            frequency,
            port,
        } => Error::NonFiniteDirectConnectionZ0 {
            input: map_direct_connection_input(network),
            frequency,
            port,
        },
        direct_connection::DirectConnectionError::InvalidPort {
            network,
            port,
            nports,
        } => Error::InvalidDirectConnectionPort {
            input: map_direct_connection_input(network),
            port,
            nports,
        },
        direct_connection::DirectConnectionError::ZeroRealReferenceImpedance {
            network,
            frequency,
            port,
        } => Error::ZeroRealDirectConnectionReferenceImpedance {
            input: map_direct_connection_input(network),
            frequency,
            port,
        },
        direct_connection::DirectConnectionError::NoExternalPorts => {
            Error::NoExternalDirectConnectionPorts
        }
        direct_connection::DirectConnectionError::Singular {
            frequency,
            port_a,
            port_b,
            pivot,
        } => Error::SingularDirectConnection {
            frequency,
            port_a,
            port_b,
            pivot,
        },
        direct_connection::DirectConnectionError::NonFiniteComputation {
            frequency,
            port_a,
            port_b,
            row,
            column,
        } => Error::NonFiniteDirectConnectionComputation {
            frequency,
            port_a,
            port_b,
            row,
            column,
        },
    }
}

fn map_direct_inner_connection_error(
    error: direct_connection::DirectInnerConnectionError,
) -> Error {
    match error {
        direct_connection::DirectInnerConnectionError::InvalidSShape { shape } => {
            Error::InvalidDirectInnerConnectionSShape {
                shape: vec![shape.0, shape.1, shape.2],
            }
        }
        direct_connection::DirectInnerConnectionError::InvalidZ0Shape { shape } => {
            Error::InvalidDirectInnerConnectionZ0Shape {
                shape: vec![shape.0, shape.1],
            }
        }
        direct_connection::DirectInnerConnectionError::EmptyFrequency => {
            Error::EmptyDirectInnerConnectionFrequency
        }
        direct_connection::DirectInnerConnectionError::FrequencyShape { expected, actual } => {
            Error::DirectInnerConnectionFrequencyShape { expected, actual }
        }
        direct_connection::DirectInnerConnectionError::NonFiniteFrequency { index, value } => {
            Error::NonFiniteDirectInnerConnectionFrequency { index, value }
        }
        direct_connection::DirectInnerConnectionError::NonFiniteS {
            frequency,
            row,
            column,
        } => Error::NonFiniteDirectInnerConnectionS {
            frequency,
            row,
            column,
        },
        direct_connection::DirectInnerConnectionError::NonFiniteZ0 { frequency, port } => {
            Error::NonFiniteDirectInnerConnectionZ0 { frequency, port }
        }
        direct_connection::DirectInnerConnectionError::InvalidPort { port, nports } => {
            Error::InvalidDirectInnerConnectionPort { port, nports }
        }
        direct_connection::DirectInnerConnectionError::IdenticalPorts { port_a, port_b } => {
            Error::IdenticalDirectInnerConnectionPorts { port_a, port_b }
        }
        direct_connection::DirectInnerConnectionError::ZeroRealReferenceImpedance {
            frequency,
            port,
        } => Error::ZeroRealDirectInnerConnectionReferenceImpedance { frequency, port },
        direct_connection::DirectInnerConnectionError::NoExternalPorts => {
            Error::NoExternalDirectInnerConnectionPorts
        }
        direct_connection::DirectInnerConnectionError::Singular {
            frequency,
            port_a,
            port_b,
            pivot,
        } => Error::SingularDirectInnerConnection {
            frequency,
            port_a,
            port_b,
            pivot,
        },
        direct_connection::DirectInnerConnectionError::NonFiniteComputation {
            frequency,
            port_a,
            port_b,
            row,
            column,
        } => Error::NonFiniteDirectInnerConnectionComputation {
            frequency,
            port_a,
            port_b,
            row,
            column,
        },
    }
}

fn map_inner_connection_error(
    error: connection::ConnectionError,
    port_a: usize,
    port_b: usize,
) -> Error {
    match error {
        connection::ConnectionError::InvalidSShape { shape, .. } => {
            Error::InvalidInnerConnectionSShape {
                shape: vec![shape.0, shape.1, shape.2],
            }
        }
        connection::ConnectionError::InvalidZ0Shape { shape, .. } => {
            Error::InvalidInnerConnectionZ0Shape {
                shape: vec![shape.0, shape.1],
            }
        }
        connection::ConnectionError::EmptyFrequency { .. } => Error::EmptyInnerConnectionFrequency,
        connection::ConnectionError::FrequencyShape {
            expected, actual, ..
        } => Error::InnerConnectionFrequencyShape { expected, actual },
        connection::ConnectionError::FrequencyLengthMismatch { .. }
        | connection::ConnectionError::FrequencyMismatch { .. }
        | connection::ConnectionError::InvalidPort { .. } => {
            unreachable!("same-network inner-connect kernel returned a two-network error")
        }
        connection::ConnectionError::NonFiniteFrequency { index, value, .. } => {
            Error::NonFiniteInnerConnectionFrequency { index, value }
        }
        connection::ConnectionError::NonFiniteS {
            frequency,
            row,
            column,
            ..
        } => Error::NonFiniteInnerConnectionS {
            frequency,
            row,
            column,
        },
        connection::ConnectionError::NonFiniteZ0 {
            frequency, port, ..
        } => Error::NonFiniteInnerConnectionZ0 { frequency, port },
        connection::ConnectionError::InvalidInnerPort { port, nports } => {
            Error::InvalidInnerConnectionPort { port, nports }
        }
        connection::ConnectionError::IdenticalInnerPorts { port_a, port_b } => {
            Error::IdenticalInnerConnectionPorts { port_a, port_b }
        }
        connection::ConnectionError::InvalidJunctionZ0 {
            frequency,
            port,
            value,
            ..
        } => Error::InvalidInnerConnectionJunctionZ0 {
            frequency,
            port,
            value,
        },
        connection::ConnectionError::MismatchedJunctionZ0 { frequency, a, b } => {
            Error::MismatchedInnerConnectionJunctionZ0 {
                frequency,
                port_a,
                z0_a: a,
                port_b,
                z0_b: b,
            }
        }
        connection::ConnectionError::NoExternalPorts => Error::NoExternalInnerConnectionPorts,
        connection::ConnectionError::Singular { frequency } => {
            Error::SingularInnerConnection { frequency }
        }
        connection::ConnectionError::NonFiniteComputation {
            frequency,
            row,
            column,
        } => Error::NonFiniteInnerConnectionComputation {
            frequency,
            row,
            column,
        },
    }
}

fn map_termination_error(error: termination::TerminationError) -> Error {
    match error {
        termination::TerminationError::InvalidSShape { shape } => Error::InvalidTerminationSShape {
            shape: vec![shape.0, shape.1, shape.2],
        },
        termination::TerminationError::EmptyFrequency => Error::EmptyTerminationFrequency,
        termination::TerminationError::FrequencyShape { expected, actual } => {
            Error::TerminationFrequencyLengthMismatch { expected, actual }
        }
        termination::TerminationError::InvalidZ0Shape { shape } => {
            Error::InvalidTerminationZ0Shape {
                shape: vec![shape.0, shape.1],
            }
        }
        termination::TerminationError::LoadLengthMismatch { expected, actual } => {
            Error::TerminationLoadLengthMismatch { expected, actual }
        }
        termination::TerminationError::NoExternalPorts { nports } => {
            Error::NoTerminationSurvivors { nports }
        }
        termination::TerminationError::InvalidPort { port, nports } => {
            Error::InvalidTerminationPort { port, nports }
        }
        termination::TerminationError::NonFiniteS {
            frequency,
            row,
            column,
        } => Error::NonFiniteTerminationS {
            frequency,
            row,
            column,
        },
        termination::TerminationError::NonFiniteZ0 { frequency, port } => {
            Error::NonFiniteTerminationZ0 { frequency, port }
        }
        termination::TerminationError::ZeroRealReferenceImpedance { frequency, port } => {
            Error::ZeroRealTerminationReferenceImpedance { frequency, port }
        }
        termination::TerminationError::NonFiniteLoad { frequency } => {
            Error::NonFiniteTerminationLoad { frequency }
        }
        termination::TerminationError::Singular { frequency, port } => {
            Error::SingularTermination { frequency, port }
        }
        termination::TerminationError::NonFiniteComputation {
            frequency,
            port,
            row,
            column,
        } => Error::NonFiniteTerminationComputation {
            frequency,
            port,
            row,
            column,
        },
    }
}

fn map_two_port_stability_error(error: stability::TwoPortStabilityError) -> Error {
    match error {
        stability::TwoPortStabilityError::EmptyFrequency => Error::EmptyTwoPortStabilityFrequency,
        stability::TwoPortStabilityError::FrequencyLengthMismatch { expected, actual } => {
            Error::TwoPortStabilityFrequencyLengthMismatch { expected, actual }
        }
        stability::TwoPortStabilityError::InvalidSShape { shape } => {
            Error::InvalidTwoPortStabilitySShape {
                shape: vec![shape.0, shape.1, shape.2],
            }
        }
        stability::TwoPortStabilityError::InvalidZ0Shape { shape } => {
            Error::InvalidTwoPortStabilityZ0Shape {
                shape: vec![shape.0, shape.1],
            }
        }
        stability::TwoPortStabilityError::NonFiniteFrequency { index, value } => {
            Error::NonFiniteTwoPortStabilityFrequency { index, value }
        }
        stability::TwoPortStabilityError::NonFiniteS {
            frequency,
            row,
            column,
        } => Error::NonFiniteTwoPortStabilityS {
            frequency,
            row,
            column,
        },
        stability::TwoPortStabilityError::NonFiniteZ0 { frequency, port } => {
            Error::NonFiniteTwoPortStabilityZ0 { frequency, port }
        }
        stability::TwoPortStabilityError::NonPositiveRealZ0 {
            frequency,
            port,
            value,
        } => Error::NonPositiveRealTwoPortStabilityReferenceImpedance {
            frequency,
            port,
            value,
        },
        stability::TwoPortStabilityError::Arithmetic { frequency, stage } => {
            Error::NonFiniteTwoPortStabilityComputation { frequency, stage }
        }
    }
}

fn map_power_wave_error(stage: ConversionStage, error: power_waves::PowerWaveError) -> Error {
    match error {
        power_waves::PowerWaveError::InvalidSShape { shape } => Error::InvalidShape {
            stage,
            parameter: ParameterKind::S,
            shape: vec![shape.0, shape.1, shape.2],
        },
        power_waves::PowerWaveError::InvalidZShape { shape } => Error::InvalidShape {
            stage,
            parameter: ParameterKind::Z,
            shape: vec![shape.0, shape.1, shape.2],
        },
        power_waves::PowerWaveError::InvalidYShape { shape } => Error::InvalidShape {
            stage,
            parameter: ParameterKind::Y,
            shape: vec![shape.0, shape.1, shape.2],
        },
        power_waves::PowerWaveError::InvalidZ0Shape { shape } => Error::InvalidShape {
            stage,
            parameter: ParameterKind::Z0,
            shape: vec![shape.0, shape.1],
        },
        power_waves::PowerWaveError::ZeroRealReferenceImpedance { frequency, port } => {
            Error::ZeroRealReferenceImpedance {
                stage,
                frequency,
                port,
            }
        }
        power_waves::PowerWaveError::Singular { frequency, pivot } => Error::Singular {
            stage,
            frequency,
            pivot,
        },
        power_waves::PowerWaveError::NonFiniteS {
            frequency,
            row,
            column,
        } => Error::NonFiniteS {
            stage,
            frequency,
            row,
            column,
        },
        power_waves::PowerWaveError::NonFiniteZ {
            frequency,
            row,
            column,
        } => Error::NonFiniteZ {
            stage,
            frequency,
            row,
            column,
        },
        power_waves::PowerWaveError::NonFiniteY {
            frequency,
            row,
            column,
        } => Error::NonFiniteY {
            stage,
            frequency,
            row,
            column,
        },
        power_waves::PowerWaveError::NonFiniteZ0 { frequency, port } => Error::NonFiniteZ0 {
            stage,
            frequency,
            port,
        },
        power_waves::PowerWaveError::NonFiniteComputation {
            frequency,
            row,
            column,
        } => Error::NonFiniteComputation {
            stage,
            frequency,
            row,
            column,
        },
    }
}

fn map_direct_renormalization_error(error: power_waves::DirectRenormalizationError) -> Error {
    match error {
        power_waves::DirectRenormalizationError::InvalidSShape { shape } => {
            Error::InvalidDirectRenormalizationSShape {
                shape: vec![shape.0, shape.1, shape.2],
            }
        }
        power_waves::DirectRenormalizationError::InvalidZ0Shape { reference, shape } => {
            Error::InvalidDirectRenormalizationZ0Shape {
                reference: map_direct_renormalization_reference(reference),
                shape: vec![shape.0, shape.1],
            }
        }
        power_waves::DirectRenormalizationError::NonFiniteS {
            frequency,
            row,
            column,
        } => Error::NonFiniteDirectRenormalizationS {
            frequency,
            row,
            column,
        },
        power_waves::DirectRenormalizationError::NonFiniteZ0 {
            reference,
            frequency,
            port,
        } => Error::NonFiniteDirectRenormalizationZ0 {
            reference: map_direct_renormalization_reference(reference),
            frequency,
            port,
        },
        power_waves::DirectRenormalizationError::ZeroRealReferenceImpedance {
            reference,
            frequency,
            port,
        } => Error::ZeroRealDirectRenormalizationReferenceImpedance {
            reference: map_direct_renormalization_reference(reference),
            frequency,
            port,
        },
        power_waves::DirectRenormalizationError::Singular { frequency, pivot } => {
            Error::SingularDirectRenormalization { frequency, pivot }
        }
        power_waves::DirectRenormalizationError::NonFiniteComputation {
            frequency,
            row,
            column,
        } => Error::NonFiniteDirectRenormalizationComputation {
            frequency,
            row,
            column,
        },
    }
}

fn map_inverse_cascade_error(error: inverse_cascade::InverseCascadeError) -> Error {
    match error {
        inverse_cascade::InverseCascadeError::EmptyFrequency => Error::EmptyInverseCascadeFrequency,
        inverse_cascade::InverseCascadeError::FrequencyLengthMismatch { expected, actual } => {
            Error::InverseCascadeFrequencyLengthMismatch { expected, actual }
        }
        inverse_cascade::InverseCascadeError::InvalidSShape { shape } => {
            Error::InvalidInverseCascadeSShape {
                shape: vec![shape.0, shape.1, shape.2],
            }
        }
        inverse_cascade::InverseCascadeError::InvalidZ0Shape { shape } => {
            Error::InvalidInverseCascadeZ0Shape {
                shape: vec![shape.0, shape.1],
            }
        }
        inverse_cascade::InverseCascadeError::InvalidPortCount { nports } => {
            Error::InvalidInverseCascadePortCount { nports }
        }
        inverse_cascade::InverseCascadeError::NonFiniteFrequency { index, value } => {
            Error::NonFiniteInverseCascadeFrequency { index, value }
        }
        inverse_cascade::InverseCascadeError::NonFiniteS {
            stage,
            frequency,
            row,
            column,
        } => Error::NonFiniteInverseCascadeS {
            stage,
            frequency,
            row,
            column,
        },
        inverse_cascade::InverseCascadeError::NonFiniteZ0 { frequency, port } => {
            Error::NonFiniteInverseCascadeZ0 { frequency, port }
        }
        inverse_cascade::InverseCascadeError::ZeroRealReferenceImpedance { frequency, port } => {
            Error::ZeroRealInverseCascadeReferenceImpedance { frequency, port }
        }
        inverse_cascade::InverseCascadeError::Singular {
            stage,
            frequency,
            pivot,
        } => Error::SingularInverseCascade {
            stage,
            frequency,
            pivot,
        },
        inverse_cascade::InverseCascadeError::NonFiniteComputation {
            stage,
            frequency,
            row,
            column,
        } => Error::NonFiniteInverseCascadeComputation {
            stage,
            frequency,
            row,
            column,
        },
    }
}

fn map_direct_renormalization_reference(
    reference: power_waves::DirectRenormalizationReference,
) -> DirectRenormalizationReference {
    match reference {
        power_waves::DirectRenormalizationReference::Source => {
            DirectRenormalizationReference::Source
        }
        power_waves::DirectRenormalizationReference::Target => {
            DirectRenormalizationReference::Target
        }
    }
}

fn map_mixed_mode_error(error: mixed_mode::MixedModeError) -> Error {
    match error {
        mixed_mode::MixedModeError::EmptyFrequency { direction } => {
            Error::EmptyMixedModeFrequency { direction }
        }
        mixed_mode::MixedModeError::FrequencyLengthMismatch {
            direction,
            expected,
            actual,
        } => Error::MixedModeFrequencyLengthMismatch {
            direction,
            expected,
            actual,
        },
        mixed_mode::MixedModeError::InvalidSShape { direction, shape } => {
            Error::InvalidMixedModeSShape {
                direction,
                shape: vec![shape.0, shape.1, shape.2],
            }
        }
        mixed_mode::MixedModeError::InvalidZ0Shape { direction, shape } => {
            Error::InvalidMixedModeZ0Shape {
                direction,
                shape: vec![shape.0, shape.1],
            }
        }
        mixed_mode::MixedModeError::PairCountOutOfRange {
            direction,
            pair_count,
            nports,
        } => Error::MixedModePairCountOutOfRange {
            direction,
            pair_count,
            nports,
        },
        mixed_mode::MixedModeError::NonFiniteS {
            direction,
            frequency,
            row,
            column,
        } => Error::NonFiniteMixedModeS {
            direction,
            frequency,
            row,
            column,
        },
        mixed_mode::MixedModeError::NonFiniteZ0 {
            direction,
            frequency,
            port,
        } => Error::NonFiniteMixedModeZ0 {
            direction,
            frequency,
            port,
        },
        mixed_mode::MixedModeError::ZeroRealReferenceImpedance {
            direction,
            frequency,
            port,
        } => Error::ZeroRealMixedModeReferenceImpedance {
            direction,
            frequency,
            port,
        },
        mixed_mode::MixedModeError::UnequalPairReferences {
            direction,
            frequency,
            pair,
            positive,
            negative,
        } => Error::UnequalMixedModePairReferences {
            direction,
            frequency,
            pair,
            positive,
            negative,
        },
        mixed_mode::MixedModeError::ReferenceScalingLoss {
            direction,
            frequency,
            pair,
            mode,
            scaling,
            value,
        } => Error::MixedModeReferenceScalingLoss {
            direction,
            frequency,
            pair,
            mode,
            scaling,
            value,
        },
        mixed_mode::MixedModeError::InverseReferenceMismatch {
            direction,
            frequency,
            pair,
            differential,
            common,
            from_differential,
            from_common,
        } => Error::MixedModeInverseReferenceMismatch {
            direction,
            frequency,
            pair,
            differential,
            common,
            from_differential,
            from_common,
        },
        mixed_mode::MixedModeError::NonFiniteComputation {
            direction,
            frequency,
            row,
            column,
        } => Error::NonFiniteMixedModeComputation {
            direction,
            frequency,
            row,
            column,
        },
    }
}

fn map_impedance_admittance_error(
    stage: ConversionStage,
    error: impedance_admittance::ImpedanceAdmittanceError,
) -> Error {
    match error {
        impedance_admittance::ImpedanceAdmittanceError::InvalidZShape { shape } => {
            Error::InvalidShape {
                stage,
                parameter: ParameterKind::Z,
                shape: vec![shape.0, shape.1, shape.2],
            }
        }
        impedance_admittance::ImpedanceAdmittanceError::InvalidYShape { shape } => {
            Error::InvalidShape {
                stage,
                parameter: ParameterKind::Y,
                shape: vec![shape.0, shape.1, shape.2],
            }
        }
        impedance_admittance::ImpedanceAdmittanceError::NonFiniteZ {
            frequency,
            row,
            column,
        } => Error::NonFiniteZ {
            stage,
            frequency,
            row,
            column,
        },
        impedance_admittance::ImpedanceAdmittanceError::NonFiniteY {
            frequency,
            row,
            column,
        } => Error::NonFiniteY {
            stage,
            frequency,
            row,
            column,
        },
        impedance_admittance::ImpedanceAdmittanceError::NonFiniteComputation {
            frequency,
            row,
            column,
        } => Error::NonFiniteComputation {
            stage,
            frequency,
            row,
            column,
        },
        impedance_admittance::ImpedanceAdmittanceError::Singular { frequency, pivot } => {
            Error::Singular {
                stage,
                frequency,
                pivot,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    #[test]
    fn constructs_a_two_port_network() {
        let frequency = Frequency::from_hz(vec![1.0e9, 2.0e9]).unwrap();
        let s = Array3::zeros((2, 2, 2));
        let z0 = Array2::from_elem((2, 2), Complex64::new(50.0, 0.0));

        let network = Network::new(frequency, s, z0).unwrap();
        assert_eq!(network.nports(), 2);
        assert_eq!(network.frequency().len(), 2);
    }

    #[test]
    fn rejects_non_square_s_matrices() {
        let frequency = Frequency::from_hz(vec![1.0e9]).unwrap();
        let s = Array3::zeros((1, 2, 3));
        let z0 = Array2::from_elem((1, 2), Complex64::new(50.0, 0.0));

        assert_eq!(
            Network::new(frequency, s, z0).unwrap_err(),
            Error::InvalidSShape
        );
    }

    #[test]
    fn maps_private_shape_errors_to_structured_public_context() {
        let s_shape = map_power_wave_error(
            ConversionStage::SToZ,
            power_waves::PowerWaveError::InvalidSShape { shape: (2, 3, 4) },
        );
        assert_eq!(
            s_shape,
            Error::InvalidShape {
                stage: ConversionStage::SToZ,
                parameter: ParameterKind::S,
                shape: vec![2, 3, 4],
            }
        );

        let z_shape = map_impedance_admittance_error(
            ConversionStage::ZToY,
            impedance_admittance::ImpedanceAdmittanceError::InvalidZShape { shape: (1, 2, 3) },
        );
        assert_eq!(
            z_shape,
            Error::InvalidShape {
                stage: ConversionStage::ZToY,
                parameter: ParameterKind::Z,
                shape: vec![1, 2, 3],
            }
        );

        let z0_shape = map_power_wave_error(
            ConversionStage::SToZ,
            power_waves::PowerWaveError::InvalidZ0Shape { shape: (4, 5) },
        );
        assert_eq!(
            z0_shape,
            Error::InvalidShape {
                stage: ConversionStage::SToZ,
                parameter: ParameterKind::Z0,
                shape: vec![4, 5],
            }
        );
    }

    #[test]
    fn maps_private_stage_and_location_errors_without_flattening() {
        let singular =
            map_power_wave_admittance_error(power_wave_admittance::PowerWaveAdmittanceError::SToZ(
                power_waves::PowerWaveError::Singular {
                    frequency: 3,
                    pivot: 2,
                },
            ));
        assert_eq!(
            singular,
            Error::Singular {
                stage: ConversionStage::SToZ,
                frequency: 3,
                pivot: 2,
            }
        );

        let z_to_y_computation =
            map_power_wave_admittance_error(power_wave_admittance::PowerWaveAdmittanceError::ZToY(
                impedance_admittance::ImpedanceAdmittanceError::NonFiniteComputation {
                    frequency: 1,
                    row: 4,
                    column: 5,
                },
            ));
        assert_eq!(
            z_to_y_computation,
            Error::NonFiniteComputation {
                stage: ConversionStage::ZToY,
                frequency: 1,
                row: 4,
                column: 5,
            }
        );
    }

    #[test]
    fn maps_private_interpolation_shape_and_axis_errors_without_flattening() {
        assert_eq!(
            map_interpolation_error(interpolation::InterpolationError::InvalidSShape {
                shape: (2, 3, 4),
            }),
            Error::InvalidInterpolationShape {
                quantity: InterpolationQuantity::S,
                shape: vec![2, 3, 4],
            }
        );
        assert_eq!(
            map_interpolation_error(interpolation::InterpolationError::InvalidZ0Shape {
                shape: (2, 1),
            }),
            Error::InvalidInterpolationShape {
                quantity: InterpolationQuantity::Z0,
                shape: vec![2, 1],
            }
        );
        assert_eq!(
            map_interpolation_error(interpolation::InterpolationError::SourceFrequencyShape {
                expected: 4,
                actual: 3,
            }),
            Error::InterpolationSourceFrequencyLengthMismatch {
                expected: 4,
                actual: 3,
            }
        );
        assert_eq!(
            map_interpolation_error(interpolation::InterpolationError::EmptyTarget),
            Error::InterpolationEmptyTarget
        );
        assert_eq!(
            map_interpolation_error(interpolation::InterpolationError::TooFewSourceSamples {
                actual: 1,
            }),
            Error::InterpolationTooFewSourceSamples { actual: 1 }
        );
        assert_eq!(
            map_interpolation_error(
                interpolation::InterpolationError::SourceFrequencyNotStrictlyIncreasing {
                    index: 2,
                    previous: 2.0,
                    current: 2.0,
                },
            ),
            Error::InterpolationFrequencyNotStrictlyIncreasing {
                axis: InterpolationAxis::Source,
                index: 2,
                previous: 2.0,
                current: 2.0,
            }
        );
        assert!(matches!(
            map_interpolation_error(
                interpolation::InterpolationError::NonFiniteTargetFrequency {
                    index: 1,
                    value: f64::NAN,
                }
            ),
            Error::NonFiniteInterpolationFrequency {
                axis: InterpolationAxis::Target,
                index: 1,
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn maps_private_interpolation_numeric_context_without_flattening() {
        assert_eq!(
            map_interpolation_error(interpolation::InterpolationError::NonFiniteWeight {
                target: 5,
                lower_source: 2,
                upper_source: 3,
            }),
            Error::NonFiniteInterpolationWeight {
                target: 5,
                lower_source: 2,
                upper_source: 3,
            }
        );
        assert_eq!(
            map_interpolation_error(interpolation::InterpolationError::NonFiniteComputation {
                quantity: interpolation::InterpolationQuantity::Z0,
                target: 4,
                row: 2,
                column: 0,
            }),
            Error::NonFiniteInterpolationComputation {
                quantity: InterpolationQuantity::Z0,
                target: 4,
                row: 2,
                column: 0,
            }
        );
        assert_eq!(
            map_interpolation_error(
                interpolation::InterpolationError::TargetFrequencyOutOfRange {
                    index: 0,
                    value: 0.5,
                    lower: 1.0,
                    upper: 2.0,
                }
            ),
            Error::InterpolationTargetOutOfRange {
                index: 0,
                value: 0.5,
                lower: 1.0,
                upper: 2.0,
            }
        );
        assert_eq!(
            map_interpolation_error(interpolation::InterpolationError::NonFiniteS {
                frequency: 3,
                row: 1,
                column: 2,
            }),
            Error::NonFiniteInterpolationS {
                frequency: 3,
                row: 1,
                column: 2,
            }
        );
        assert_eq!(
            map_interpolation_error(interpolation::InterpolationError::NonFiniteZ0 {
                frequency: 2,
                port: 1,
            }),
            Error::NonFiniteInterpolationZ0 {
                frequency: 2,
                port: 1,
            }
        );
    }

    #[test]
    fn maps_private_composition_errors_with_ordered_public_stage_context() {
        assert_eq!(
            map_composition_error(composition::CompositionError::AInterpolation(
                interpolation::InterpolationError::InvalidSShape { shape: (2, 3, 4) },
            )),
            Error::GridConnection {
                stage: GridConnectionStage::AInterpolation,
                source: Box::new(Error::InvalidInterpolationShape {
                    quantity: InterpolationQuantity::S,
                    shape: vec![2, 3, 4],
                }),
            }
        );

        assert_eq!(
            map_composition_error(composition::CompositionError::BInterpolation(
                interpolation::InterpolationError::TargetFrequencyOutOfRange {
                    index: 2,
                    value: 4.0,
                    lower: 1.0,
                    upper: 3.0,
                },
            )),
            Error::GridConnection {
                stage: GridConnectionStage::BInterpolation,
                source: Box::new(Error::InterpolationTargetOutOfRange {
                    index: 2,
                    value: 4.0,
                    lower: 1.0,
                    upper: 3.0,
                }),
            }
        );

        assert_eq!(
            map_composition_error(composition::CompositionError::Connection(
                connection::ConnectionError::MismatchedJunctionZ0 {
                    frequency: 1,
                    a: Complex64::new(73.5, 0.0),
                    b: Complex64::new(74.0, 0.0),
                },
            )),
            Error::GridConnection {
                stage: GridConnectionStage::Connection,
                source: Box::new(Error::MismatchedConnectionJunctionZ0 {
                    frequency: 1,
                    a: Complex64::new(73.5, 0.0),
                    b: Complex64::new(74.0, 0.0),
                }),
            }
        );
    }

    #[test]
    fn grid_connection_stage_display_is_actionable() {
        assert_eq!(
            GridConnectionStage::AInterpolation.to_string(),
            "A interpolation (receiver)"
        );
        assert_eq!(
            GridConnectionStage::BInterpolation.to_string(),
            "B interpolation (other)"
        );
        assert_eq!(
            GridConnectionStage::Connection.to_string(),
            "matched connection"
        );
    }

    #[test]
    fn maps_every_private_connection_error_with_input_context() {
        use connection::{ConnectionError, NetworkSide};

        assert_eq!(
            map_connection_error(ConnectionError::InvalidSShape {
                network: NetworkSide::B,
                shape: (2, 3, 4),
            }),
            Error::InvalidConnectionSShape {
                input: ConnectionInput::B,
                shape: vec![2, 3, 4],
            }
        );
        assert_eq!(
            map_connection_error(ConnectionError::InvalidZ0Shape {
                network: NetworkSide::A,
                shape: (2, 3),
            }),
            Error::InvalidConnectionZ0Shape {
                input: ConnectionInput::A,
                shape: vec![2, 3],
            }
        );
        assert_eq!(
            map_connection_error(ConnectionError::EmptyFrequency {
                network: NetworkSide::A,
            }),
            Error::EmptyConnectionFrequency {
                input: ConnectionInput::A,
            }
        );
        assert_eq!(
            map_connection_error(ConnectionError::FrequencyShape {
                network: NetworkSide::B,
                expected: 4,
                actual: 3,
            }),
            Error::ConnectionFrequencyShape {
                input: ConnectionInput::B,
                expected: 4,
                actual: 3,
            }
        );
        assert_eq!(
            map_connection_error(ConnectionError::FrequencyLengthMismatch { a: 2, b: 3 }),
            Error::ConnectionFrequencyLengthMismatch { a: 2, b: 3 }
        );
        assert_eq!(
            map_connection_error(ConnectionError::FrequencyMismatch {
                index: 1,
                a: 2.0,
                b: 2.5,
            }),
            Error::ConnectionFrequencyMismatch {
                index: 1,
                a: 2.0,
                b: 2.5,
            }
        );
        assert!(matches!(
            map_connection_error(ConnectionError::NonFiniteFrequency {
                network: NetworkSide::B,
                index: 2,
                value: f64::NAN,
            }),
            Error::NonFiniteConnectionFrequency {
                input: ConnectionInput::B,
                index: 2,
                value,
            } if value.is_nan()
        ));
        assert_eq!(
            map_connection_error(ConnectionError::NonFiniteS {
                network: NetworkSide::A,
                frequency: 3,
                row: 1,
                column: 2,
            }),
            Error::NonFiniteConnectionS {
                input: ConnectionInput::A,
                frequency: 3,
                row: 1,
                column: 2,
            }
        );
        assert_eq!(
            map_connection_error(ConnectionError::NonFiniteZ0 {
                network: NetworkSide::B,
                frequency: 4,
                port: 5,
            }),
            Error::NonFiniteConnectionZ0 {
                input: ConnectionInput::B,
                frequency: 4,
                port: 5,
            }
        );
        assert_eq!(
            map_connection_error(ConnectionError::InvalidPort {
                network: NetworkSide::A,
                port: 4,
                nports: 3,
            }),
            Error::InvalidConnectionPort {
                input: ConnectionInput::A,
                port: 4,
                nports: 3,
            }
        );
        assert_eq!(
            map_connection_error(ConnectionError::InvalidJunctionZ0 {
                network: NetworkSide::B,
                frequency: 2,
                port: 1,
                value: Complex64::new(0.0, 1.0),
            }),
            Error::InvalidConnectionJunctionZ0 {
                input: ConnectionInput::B,
                frequency: 2,
                port: 1,
                value: Complex64::new(0.0, 1.0),
            }
        );
        assert_eq!(
            map_connection_error(ConnectionError::MismatchedJunctionZ0 {
                frequency: 2,
                a: Complex64::new(50.0, 0.0),
                b: Complex64::new(75.0, 0.0),
            }),
            Error::MismatchedConnectionJunctionZ0 {
                frequency: 2,
                a: Complex64::new(50.0, 0.0),
                b: Complex64::new(75.0, 0.0),
            }
        );
        assert_eq!(
            map_connection_error(ConnectionError::NoExternalPorts),
            Error::NoExternalConnectionPorts
        );
        assert_eq!(
            map_connection_error(ConnectionError::Singular { frequency: 2 }),
            Error::SingularConnection { frequency: 2 }
        );
        assert_eq!(
            map_connection_error(ConnectionError::NonFiniteComputation {
                frequency: 2,
                row: 3,
                column: 4,
            }),
            Error::NonFiniteConnectionComputation {
                frequency: 2,
                row: 3,
                column: 4,
            }
        );
    }

    #[test]
    fn maps_every_private_inner_connection_error_without_two_network_context() {
        use connection::{ConnectionError, NetworkSide};

        assert_eq!(
            map_inner_connection_error(
                ConnectionError::InvalidSShape {
                    network: NetworkSide::A,
                    shape: (2, 3, 4),
                },
                5,
                7,
            ),
            Error::InvalidInnerConnectionSShape {
                shape: vec![2, 3, 4],
            }
        );
        assert_eq!(
            map_inner_connection_error(
                ConnectionError::InvalidZ0Shape {
                    network: NetworkSide::A,
                    shape: (2, 3),
                },
                5,
                7,
            ),
            Error::InvalidInnerConnectionZ0Shape { shape: vec![2, 3] }
        );
        assert_eq!(
            map_inner_connection_error(
                ConnectionError::EmptyFrequency {
                    network: NetworkSide::A,
                },
                5,
                7,
            ),
            Error::EmptyInnerConnectionFrequency
        );
        assert_eq!(
            map_inner_connection_error(
                ConnectionError::FrequencyShape {
                    network: NetworkSide::A,
                    expected: 4,
                    actual: 3,
                },
                5,
                7,
            ),
            Error::InnerConnectionFrequencyShape {
                expected: 4,
                actual: 3,
            }
        );
        assert!(matches!(
            map_inner_connection_error(
                ConnectionError::NonFiniteFrequency {
                    network: NetworkSide::A,
                    index: 2,
                    value: f64::NAN,
                },
                5,
                7,
            ),
            Error::NonFiniteInnerConnectionFrequency { index: 2, value } if value.is_nan()
        ));
        assert_eq!(
            map_inner_connection_error(
                ConnectionError::NonFiniteS {
                    network: NetworkSide::A,
                    frequency: 3,
                    row: 1,
                    column: 2,
                },
                5,
                7,
            ),
            Error::NonFiniteInnerConnectionS {
                frequency: 3,
                row: 1,
                column: 2,
            }
        );
        assert_eq!(
            map_inner_connection_error(
                ConnectionError::NonFiniteZ0 {
                    network: NetworkSide::A,
                    frequency: 4,
                    port: 5,
                },
                5,
                7,
            ),
            Error::NonFiniteInnerConnectionZ0 {
                frequency: 4,
                port: 5,
            }
        );
        assert_eq!(
            map_inner_connection_error(
                ConnectionError::InvalidInnerPort { port: 8, nports: 6 },
                5,
                7,
            ),
            Error::InvalidInnerConnectionPort { port: 8, nports: 6 }
        );
        assert_eq!(
            map_inner_connection_error(
                ConnectionError::IdenticalInnerPorts {
                    port_a: 5,
                    port_b: 5,
                },
                5,
                5,
            ),
            Error::IdenticalInnerConnectionPorts {
                port_a: 5,
                port_b: 5,
            }
        );
        assert_eq!(
            map_inner_connection_error(
                ConnectionError::InvalidJunctionZ0 {
                    network: NetworkSide::A,
                    frequency: 2,
                    port: 7,
                    value: Complex64::new(50.0, 1.0),
                },
                5,
                7,
            ),
            Error::InvalidInnerConnectionJunctionZ0 {
                frequency: 2,
                port: 7,
                value: Complex64::new(50.0, 1.0),
            }
        );
        assert_eq!(
            map_inner_connection_error(
                ConnectionError::MismatchedJunctionZ0 {
                    frequency: 2,
                    a: Complex64::new(50.0, 0.0),
                    b: Complex64::new(75.0, 0.0),
                },
                5,
                7,
            ),
            Error::MismatchedInnerConnectionJunctionZ0 {
                frequency: 2,
                port_a: 5,
                z0_a: Complex64::new(50.0, 0.0),
                port_b: 7,
                z0_b: Complex64::new(75.0, 0.0),
            }
        );
        assert_eq!(
            map_inner_connection_error(ConnectionError::NoExternalPorts, 5, 7),
            Error::NoExternalInnerConnectionPorts
        );
        assert_eq!(
            map_inner_connection_error(ConnectionError::Singular { frequency: 2 }, 5, 7),
            Error::SingularInnerConnection { frequency: 2 }
        );
        assert_eq!(
            map_inner_connection_error(
                ConnectionError::NonFiniteComputation {
                    frequency: 2,
                    row: 0,
                    column: 1,
                },
                5,
                7,
            ),
            Error::NonFiniteInnerConnectionComputation {
                frequency: 2,
                row: 0,
                column: 1,
            }
        );
    }
}
