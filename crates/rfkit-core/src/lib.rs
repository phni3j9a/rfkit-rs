//! Rust-native RF and microwave network analysis primitives.
//!
//! The project starts deliberately small: a trustworthy core data model first,
//! then numerical operations backed by differential tests against scikit-rf.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

mod composition;
mod connection;
mod impedance_admittance;
mod interpolation;
mod linalg;
mod power_wave_admittance;
mod power_waves;

/// Errors produced while constructing or manipulating RF network data.
#[non_exhaustive]
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("frequency axis must not be empty")]
    EmptyFrequency,

    #[error("S-parameter shape must be (nfreq, nport, nport)")]
    InvalidSShape,

    #[error("reference-impedance shape must be (nfreq, nport)")]
    InvalidZ0Shape,

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
/// `to_y_power` is intentionally a composed `S→Z→Y` operation, so preserving
/// this stage lets callers distinguish a singular or invalid intermediate
/// impedance conversion from a failure while inverting that impedance.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionStage {
    /// Conversion from scattering parameters to impedance parameters.
    SToZ,
    /// Conversion from impedance parameters to admittance parameters.
    ZToY,
    /// Conversion from impedance parameters to scattering parameters.
    ZToS,
}

impl fmt::Display for ConversionStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::SToZ => "S→Z",
            Self::ZToY => "Z→Y",
            Self::ZToS => "Z→S",
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
        // These stages belong to the inverse Y→S kernel, which is not exposed
        // by this increment and therefore cannot be returned by
        // `s_to_y_power`.
        power_wave_admittance::PowerWaveAdmittanceError::YToZ(_)
        | power_wave_admittance::PowerWaveAdmittanceError::ZToS(_) => {
            unreachable!("S-to-Y kernel returned an inverse-direction stage")
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
}
