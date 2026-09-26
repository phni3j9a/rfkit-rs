//! Pure in-memory parsing and deterministic writing for deliberately bounded
//! Touchstone S subsets.
//!
//! [`parse_touchstone_v1_0_s`] accepts Touchstone text and an explicit positive
//! port count.  It returns the canonical frequency-major
//! [`rfkit_core::Network`] used by the numerical core.  The reader supports
//! single-ended S-parameters in RI, MA, or DB form, the four frequency units,
//! comments, and the original v1.0 row/continuation layout.  It does not
//! inspect filenames or perform I/O.
//!
//! The reference resistance is deliberately restricted to one finite,
//! positive, real scalar.  It is expanded over every frequency and port so
//! that the resulting network can be used directly with the core's explicit
//! power-wave transformations.
//!
//! [`write_touchstone_v1_0_s_ri_hz`] emits the complementary fixed S/RI/Hz
//! representation.  It borrows and validates the complete Network, uses one
//! exact common finite positive real reference scalar, and leaves filesystem
//! and metadata policy to callers.
//!
//! [`write_touchstone_v2_0_s_full_ri_hz`] is the additive v2.0 counterpart for
//! the bounded export subset.  It preserves one finite, strictly positive
//! real reference per physical port (constant across the frequency axis),
//! emits a Full matrix in row-major order, and keeps each frequency on one
//! physical line.  It does not widen the v1 writer's common-reference
//! contract.
//!
//! Comment handling has one deliberate vendor-extension boundary: matching is
//! case-insensitive; comments beginning with `Gamma` or `Port Impedance` (or
//! `Port Impedance0`), or containing `Terminal data exported` or `Modal data
//! exported`, are rejected, as are comments containing the complete
//! `S-parameter uses the power definition`, `S-parameter uses the pseudo
//! definition`, or `S-parameter uses the traveling definition` marker.
//! Ordinary comments, including `Port[n] =
//! ...` port name comments, remain ignorable. This recognizes the documented
//! HFSS/Ansys semantic markers only; universal vendor-format recognition is
//! outside this crate's scope.
//!
//! [`parse_touchstone_v2_0_s`] is the separate Touchstone 2.0 entrypoint.  It
//! accepts single-ended Full, Lower, and Upper S data, obtains the port and
//! frequency counts from the document, preserves an optional real-positive
//! per-port `[Reference]` vector, and requires complete records plus `[End]`.
//! Triangular records are expanded by plain transpose without conjugation;
//! absent `[Matrix Format]` means Full.  Mixed-mode, noise, information,
//! non-S, unknown-keyword, complex-reference, and malformed-count features
//! remain rejected explicitly; no filename inference or filesystem I/O is
//! performed.
//!
//! # Example
//!
//! ```
//! use rfkit_touchstone::parse_touchstone_v1_0_s;
//!
//! # fn example() -> rfkit_touchstone::Result<()> {
//! let text = "# MHz S RI R 75\n10 0.2 0.0\n20 0.25 0.0\n";
//! let network = parse_touchstone_v1_0_s(text, 1)?;
//! assert_eq!(network.frequency().hz(), &[10.0e6, 20.0e6]);
//! assert_eq!(network.z0()[[0, 0]].re, 75.0);
//!
//! // The parsed data is an ordinary rfkit-core Network.  Existing public
//! // analysis methods can be used without a format-specific adapter.
//! let impedance = network.to_z_power()?;
//! assert!((impedance[[0, 0, 0]].re - 112.5).abs() < 1.0e-12);
//! # Ok(())
//! # }
//! # example().unwrap();
//! ```
//!
//! A parsed network can be transformed through the existing public core API,
//! exported, and read back without a format-specific adapter:
//!
//! ```
//! use ndarray::Array2;
//! use num_complex::Complex64;
//! use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};
//!
//! # fn workflow() -> rfkit_touchstone::Result<()> {
//! let source = parse_touchstone_v1_0_s("# Hz S RI R 75\n1000000000 0.2 0\n", 1)?;
//! let transformed = source.renormalize_power(Array2::from_elem(
//!     (1, 1),
//!     Complex64::new(100.0, 0.0),
//! ))?;
//! let text = write_touchstone_v1_0_s_ri_hz(&transformed)?;
//! let reread = parse_touchstone_v1_0_s(&text, 1)?;
//!
//! assert_eq!(reread.z0()[[0, 0]], Complex64::new(100.0, 0.0));
//! // z=75*(1+0.2)/(1-0.2)=112.5 ohm, so S at 100 ohm is 1/17.
//! assert!((reread.s()[[0, 0, 0]].re - 1.0 / 17.0).abs() < 1.0e-12);
//! # Ok(())
//! # }
//! # workflow().unwrap();
//! ```

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};
use thiserror::Error;

mod v2;
mod writer;

/// The crate-wide public error boundary for bounded Touchstone v1.0/v2.0 S
/// parsing and deterministic v1 S/RI/Hz writing.
///
/// The enum is non-exhaustive so future format-specific diagnostics can be
/// added without freezing the provisional 0.x API.  Errors identify the
/// parser stage and, where applicable, the source line and token location.
#[non_exhaustive]
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    /// The caller supplied zero ports.
    #[error("Touchstone port count must be positive, got {nports}")]
    InvalidPortCount { nports: usize },

    /// A port-count-derived allocation or record size cannot be represented
    /// by the target platform's `usize`.
    #[error(
        "Touchstone port count {nports} overflows checked size arithmetic while computing {quantity}"
    )]
    SizeOverflow {
        nports: usize,
        quantity: SizeQuantity,
    },

    /// No initial v1 option line was found before the first data/keyword line.
    #[error("Touchstone v1.0 requires an initial option line beginning with '#' (line {line})")]
    MissingOptionLine { line: usize },

    /// A token or arrangement on the option line is not valid.
    #[error("malformed Touchstone option line {line}: {message}")]
    MalformedOption { line: usize, message: String },

    /// Y, Z, H, and G are intentionally outside this reader's S-only domain.
    #[error(
        "unsupported Touchstone network parameter {parameter:?} on option line {line}; only S is supported"
    )]
    UnsupportedParameter { line: usize, parameter: String },

    /// More than one reference value denotes the v1.1 option-line extension.
    #[error(
        "Touchstone option line {line} contains multiple reference resistances; v1.0 accepts exactly one scalar"
    )]
    MultipleReferenceValues { line: usize },

    /// The single v1.0 reference resistance is not finite and strictly
    /// positive.
    #[error("invalid Touchstone reference resistance on line {line}: {value:?}")]
    InvalidReferenceResistance { line: usize, value: f64 },

    /// A bracketed keyword belongs to the v2.x syntax, which this reader does
    /// not silently reinterpret as v1 data.
    #[error("unsupported Touchstone keyword on line {line}: {keyword}")]
    UnsupportedKeyword { line: usize, keyword: String },

    /// A recognized HFSS/Ansys extension would change reference impedance or
    /// wave semantics and therefore cannot be ignored.
    #[error("unsupported Touchstone vendor extension on line {line}: {extension}")]
    UnsupportedExtension {
        line: usize,
        extension: ExtensionKind,
    },

    /// Touchstone permits only the character set described by the
    /// specification.  Reporting this explicitly also prevents surprising
    /// Unicode tokenization differences.
    #[error("unsupported non-ASCII/control character at line {line}, column {column}")]
    UnsupportedCharacter { line: usize, column: usize },

    /// The option line was valid but did not precede any network records.
    #[error("Touchstone input contains no network data records")]
    EmptyData,

    /// A numeric token cannot be parsed as binary64.
    #[error("malformed Touchstone number at line {line}, column {column}: {token:?}")]
    MalformedNumber {
        line: usize,
        column: usize,
        token: String,
    },

    /// A textual NaN or infinity is not an admissible network value.
    #[error("non-finite Touchstone number at line {line}, column {column}: {token:?}")]
    NonFiniteNumber {
        line: usize,
        column: usize,
        token: String,
    },

    /// A finite-looking decimal overflowed binary64 while parsing.
    #[error("Touchstone number overflow at line {line}, column {column}: {token:?}")]
    NumericOverflow {
        line: usize,
        column: usize,
        token: String,
    },

    /// A frequency is negative (negative zero is accepted as zero).
    #[error("Touchstone frequency is negative at line {line}: {value:?}")]
    NegativeFrequency { line: usize, value: f64 },

    /// A frequency token or scaled value is non-finite.
    #[error("Touchstone frequency is non-finite at line {line}: {value:?}")]
    NonFiniteFrequency { line: usize, value: f64 },

    /// Unit scaling overflowed a finite frequency token.
    #[error("Touchstone frequency conversion overflow at line {line}: {value:?}")]
    FrequencyOverflow { line: usize, value: f64 },

    /// v1 records must be ordered strictly by increasing scaled frequency.
    #[error(
        "Touchstone frequencies must be strictly increasing; line {line} has {current:?} after {previous:?}"
    )]
    FrequencyNotStrictlyIncreasing {
        line: usize,
        previous: f64,
        current: f64,
    },

    /// A matrix row/record ended before all required values were supplied.
    #[error(
        "incomplete Touchstone record at line {line}: expected {expected_pairs} parameter pairs, got {actual_pairs}"
    )]
    IncompleteRecord {
        line: usize,
        expected_pairs: usize,
        actual_pairs: usize,
    },

    /// A row/record supplied values beyond the exact v1 shape.
    #[error(
        "surplus Touchstone record values at line {line}: expected at most {expected_pairs} parameter pairs, got {actual_pairs}"
    )]
    SurplusRecord {
        line: usize,
        expected_pairs: usize,
        actual_pairs: usize,
    },

    /// A record cannot be partitioned into frequency/pair tokens or violates
    /// the four-pairs-per-line rule.
    #[error("malformed Touchstone record at line {line}: {message}")]
    MalformedRecord { line: usize, message: String },

    /// A magnitude in MA representation is outside its specified domain.
    #[error("negative magnitude in Touchstone MA data at line {line}, column {column}: {value:?}")]
    NegativeMagnitude {
        line: usize,
        column: usize,
        value: f64,
    },

    /// MA/DB conversion produced an infinite or NaN component.
    #[error("Touchstone {format} conversion overflow at line {line}, column {column}")]
    ConversionOverflow {
        line: usize,
        column: usize,
        format: DataFormat,
    },

    /// A 2-port v1 trailing row has the shape of Touchstone noise data.
    #[error(
        "unsupported Touchstone noise data at line {line}; this reader accepts network S data only"
    )]
    NoiseData { line: usize },

    /// Construction failures from the canonical core are retained as a
    /// structured nested source rather than flattened to a string.
    #[error("rfkit-core rejected the parsed network: {0}")]
    Core(#[from] rfkit_core::Error),

    /// This is an internal invariant guard for shape construction.  It is
    /// public because the enclosing error boundary is public, but callers
    /// should not normally observe it for valid `usize` arithmetic.
    #[error("parsed {quantity} has an invalid shape {shape:?}")]
    ConstructionShape {
        quantity: &'static str,
        shape: Vec<usize>,
    },

    /// The writer cannot emit a Touchstone record for an empty frequency axis.
    #[error("Touchstone writer requires a nonempty frequency axis")]
    WriterEmptyFrequency,

    /// The serde-readable Network state has an S-parameter dimension that is
    /// not frequency-major square data with a positive port count.
    #[error(
        "Touchstone writer received invalid S-parameter shape {shape:?} for frequency length {frequency_length}"
    )]
    WriterInvalidSShape {
        shape: Vec<usize>,
        frequency_length: usize,
    },

    /// The serde-readable Network state has a reference-impedance dimension
    /// that does not match its frequency and port dimensions.
    #[error(
        "Touchstone writer received invalid reference-impedance shape {shape:?}; expected ({frequency_length}, {nports})"
    )]
    WriterInvalidZ0Shape {
        shape: Vec<usize>,
        frequency_length: usize,
        nports: usize,
    },

    /// A writer frequency is outside the finite, non-negative domain.
    #[error(
        "Touchstone writer frequency at index {index} is not finite and non-negative: {value:?}"
    )]
    WriterInvalidFrequency { index: usize, value: f64 },

    /// Writer frequencies must be strictly increasing after validation.
    #[error(
        "Touchstone writer frequencies must be strictly increasing at index {index}: previous={previous:?}, current={current:?}"
    )]
    WriterFrequencyNotStrictlyIncreasing {
        index: usize,
        previous: f64,
        current: f64,
    },

    /// A scattering component cannot be represented by finite RI text.
    #[error(
        "Touchstone writer received a non-finite S-parameter at frequency {frequency}, row {row}, column {column}: {value:?}"
    )]
    WriterNonFiniteS {
        frequency: usize,
        row: usize,
        column: usize,
        value: Complex64,
    },

    /// A reference impedance must be one finite, strictly positive real
    /// scalar shared by every frequency and port.
    #[error(
        "Touchstone writer received an invalid reference impedance at frequency {frequency}, port {port}: {value:?}; expected one finite, strictly positive real scalar"
    )]
    WriterInvalidZ0 {
        frequency: usize,
        port: usize,
        value: Complex64,
    },

    /// Every reference impedance must exactly equal the first valid scalar.
    #[error(
        "Touchstone writer reference impedance differs at frequency {frequency}, port {port}: expected {expected:?}, got {actual:?}"
    )]
    WriterMismatchedZ0 {
        frequency: usize,
        port: usize,
        expected: f64,
        actual: Complex64,
    },

    /// A v2 writer input has a malformed frequency-major square S shape.
    #[error(
        "Touchstone 2.0 writer received invalid S-parameter shape {shape:?} for frequency length {frequency_length}"
    )]
    WriterV2InvalidSShape {
        shape: Vec<usize>,
        frequency_length: usize,
    },

    /// A v2 writer input has a malformed frequency-major reference shape.
    #[error(
        "Touchstone 2.0 writer received invalid reference-impedance shape {shape:?}; expected ({frequency_length}, {nports})"
    )]
    WriterV2InvalidZ0Shape {
        shape: Vec<usize>,
        frequency_length: usize,
        nports: usize,
    },

    /// The v2 writer cannot safely represent the derived matrix/data size.
    #[error(
        "Touchstone 2.0 writer size arithmetic overflow for {quantity} with {frequency_length} frequency samples and {nports} ports"
    )]
    WriterV2SizeOverflow {
        frequency_length: usize,
        nports: usize,
        quantity: SizeQuantity,
    },

    /// A v2 writer frequency is outside the finite, non-negative domain.
    #[error(
        "Touchstone 2.0 writer frequency at index {index} is not finite and non-negative: {value:?}"
    )]
    WriterV2InvalidFrequency { index: usize, value: f64 },

    /// v2 writer frequencies must be strictly increasing.
    #[error(
        "Touchstone 2.0 writer frequencies must be strictly increasing at index {index}: previous={previous:?}, current={current:?}"
    )]
    WriterV2FrequencyNotStrictlyIncreasing {
        index: usize,
        previous: f64,
        current: f64,
    },

    /// A v2 scattering component cannot be represented by finite RI text.
    #[error(
        "Touchstone 2.0 writer received a non-finite S-parameter at frequency {frequency}, row {row}, column {column}: {value:?}"
    )]
    WriterV2NonFiniteS {
        frequency: usize,
        row: usize,
        column: usize,
        value: Complex64,
    },

    /// A v2 reference is not finite, strictly positive, and real.
    #[error(
        "Touchstone 2.0 writer received an invalid reference impedance at frequency {frequency}, port {port}: {value:?}; expected a finite, strictly positive real value"
    )]
    WriterV2InvalidZ0 {
        frequency: usize,
        port: usize,
        value: Complex64,
    },

    /// A single port's reference changed across frequency samples.
    #[error(
        "Touchstone 2.0 writer reference impedance varies across frequency for port {port} at frequency {frequency}: expected {expected:?}, got {actual:?}; each port must be exactly constant across samples"
    )]
    WriterV2MismatchedZ0 {
        frequency: usize,
        port: usize,
        expected: f64,
        actual: Complex64,
    },

    /// A Touchstone 2.0 keyword or feature is outside the deliberately small
    /// single-ended, full-matrix S subset exposed by this crate.
    #[error("unsupported Touchstone 2.0 subset feature on line {line}: {feature}")]
    V2Unsupported { line: usize, feature: String },

    /// A Touchstone 2.0 directive, section, or record has an invalid
    /// placement or shape.  The v2 parser keeps this separate from numeric
    /// conversion failures so callers can distinguish a bad document from a
    /// bad value.
    #[error("malformed Touchstone 2.0 structure on line {line}: {message}")]
    V2Structural { line: usize, message: String },

    /// A Touchstone 2.0 declared count differs from the number of values or
    /// records that the document actually contains.
    #[error(
        "Touchstone 2.0 count mismatch on line {line} for {what}: expected {expected}, got {actual}"
    )]
    V2Count {
        line: usize,
        what: String,
        expected: usize,
        actual: usize,
    },

    /// A Touchstone 2.0 declaration or value could not be represented safely
    /// using the platform's checked dimensions.
    #[error(
        "Touchstone 2.0 size arithmetic overflow on line {line} while computing {quantity}: {detail}"
    )]
    V2SizeOverflow {
        line: usize,
        quantity: SizeQuantity,
        detail: String,
    },
}

/// Quantity used to identify checked port-count arithmetic in [`Error`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeQuantity {
    /// Number of elements in one square S matrix.
    MatrixElements,
    /// Number of scalar real/imaginary values in one matrix record.
    RecordValues,
    /// Number of expanded reference-impedance values.
    ReferenceImpedanceElements,
}

impl std::fmt::Display for SizeQuantity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::MatrixElements => "matrix elements",
            Self::RecordValues => "record values",
            Self::ReferenceImpedanceElements => "reference-impedance elements",
        })
    }
}

/// Recognized vendor extension category that is rejected instead of being
/// silently treated as the option-line scalar reference.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtensionKind {
    /// HFSS/Ansys per-frequency port reference impedance comments.
    ReferenceImpedance,
    /// HFSS/Ansys Gamma or explicit S-parameter wave-definition comments.
    WaveDefinition,
}

impl std::fmt::Display for ExtensionKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ReferenceImpedance => "HFSS/Ansys reference impedance",
            Self::WaveDefinition => "HFSS/Ansys wave definition",
        })
    }
}

/// Data-pair representation selected by the option line.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    /// Real part followed by imaginary part.
    Ri,
    /// Magnitude followed by angle in degrees.
    Ma,
    /// Decibel magnitude followed by angle in degrees.
    Db,
}

impl std::fmt::Display for DataFormat {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Ri => "RI",
            Self::Ma => "MA",
            Self::Db => "DB",
        })
    }
}

/// The crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Serialize a network as deterministic Touchstone v1.0 S-parameter text in
/// RI/Hz form.
///
/// The emitted option line is always `# Hz S RI R <reference>`.  The input is
/// borrowed and never modified.  Frequencies are written in their existing
/// order and must be finite, non-negative, and strictly increasing.  The S
/// array must be a positive square frequency-major shape and contain only
/// finite components.  Every reference impedance must be finite, strictly
/// positive, real, and exactly equal to one common scalar; the writer never
/// sorts, interpolates, repairs, or renormalizes a network.
///
/// Two-port data uses the Touchstone v1.0 physical order `S11, S21, S12,
/// S22`.  Three-port and larger matrices use row-major order, with no more
/// than four parameter pairs on each physical line and continuation lines for
/// rows wider than four ports.  All output is ASCII with LF line endings and
/// one final newline.  Finite binary64 values are formatted with Rust's
/// shortest round-tripping representation so the existing public reader can
/// reconstruct their numeric values without precision-control policy.
///
/// This is a provisional additive 0.x API.  It intentionally covers only the
/// explicit Touchstone v1.0 S/RI/Hz subset; it does not write MA/DB, v2.x,
/// metadata, noise, mixed-mode, or filesystem data.
pub fn write_touchstone_v1_0_s_ri_hz(network: &Network) -> Result<String> {
    writer::write_touchstone_v1_0_s_ri_hz(network)
}

/// Serialize a network as deterministic Touchstone 2.0 Full-matrix S/RI/Hz
/// text for the bounded single-ended export subset.
///
/// The output always contains `[Version] 2.0`, `# Hz S RI R <first-port>`
/// and explicit `[Number of Ports]`, `[Number of Frequencies]`, `[Reference]`,
/// `[Matrix Format] Full`, `[Network Data]`, and `[End]` directives.  For two
/// ports it also contains `[Two-Port Data Order] 12_21`.  Every frequency is
/// emitted on one physical line followed by all S pairs in frequency-major,
/// row-major order.  The output is ASCII with LF endings and a final newline.
///
/// The writer accepts a nonempty, finite, non-negative, strictly increasing
/// frequency axis; finite S components; and a `(nfreq, nport)` reference
/// array whose values are finite, strictly positive, real numbers.  Each
/// physical port may use a different reference, but that port's value must be
/// exactly constant over all frequency samples.  The function borrows without
/// mutation and never sorts, interpolates, renormalizes, or performs file I/O.
/// It intentionally does not emit MA/DB, triangular matrices, mixed-mode or
/// noise data, vendor extensions, metadata, or complex/frequency-varying
/// references.
pub fn write_touchstone_v2_0_s_full_ri_hz(network: &Network) -> Result<String> {
    writer::write_touchstone_v2_0_s_full_ri_hz(network)
}

/// Parse the deliberately bounded Touchstone 2.0 single-ended Full/Lower/Upper
/// S-parameter subset into the canonical frequency-major [`Network`].
///
/// Unlike the v1 entrypoint, the port count and sample count come from the
/// required v2 directives.  The parser accepts RI, MA, and DB data in the
/// four standard frequency units, both explicit two-port orders, and an
/// optional real-positive per-port `[Reference]` vector (including
/// continuation lines).  It requires complete records, `[Network Data]`, the
/// declared `[Number of Frequencies]`, and a terminating `[End]`.  Full is the
/// default when `[Matrix Format]` is absent.  Lower and Upper records include
/// the diagonal in row-wise order and expand their missing half by plain
/// transpose, never conjugation.
///
/// Mixed-mode, noise, information blocks, non-S parameters, unknown keywords,
/// complex references, and vendor semantic comments are rejected explicitly.
/// Parsing is pure and in-memory; no file name, filesystem, automatic version
/// detection, sorting, interpolation, renormalization, or alternate wave
/// convention is applied.
pub fn parse_touchstone_v2_0_s(input: &str) -> Result<Network> {
    v2::parse_touchstone_v2_0_s(input)
}

#[derive(Clone, Copy)]
enum FrequencyUnit {
    Hz,
    KHz,
    MHz,
    GHz,
}

impl FrequencyUnit {
    fn multiplier(self) -> f64 {
        match self {
            Self::Hz => 1.0,
            Self::KHz => 1.0e3,
            Self::MHz => 1.0e6,
            Self::GHz => 1.0e9,
        }
    }
}

#[derive(Clone, Copy)]
struct Options {
    unit: FrequencyUnit,
    format: DataFormat,
    resistance: f64,
}

#[derive(Clone, Copy)]
struct Token<'a> {
    text: &'a str,
    column: usize,
}

struct DataLine<'a> {
    line: usize,
    tokens: Vec<Token<'a>>,
}

/// Parse a Touchstone 1.0 single-ended S-parameter text subset.
///
/// The parser is pure and in-memory.  `nports` is explicit because this
/// v1.0 entrypoint intentionally does not infer port count from a filename.
/// The option line must be the first non-comment, non-blank line; additional
/// option lines are ignored as specified by Touchstone.  Defaults are GHz,
/// S, MA, and a 50-ohm scalar reference.
///
/// For two-port input the legacy pair order is `S11, S21, S12, S22`.  For
/// three and larger ports values are row-major (`S11, S12, ...`).  In v1.0 a
/// line contains at most four parameter pairs; rows for five or more ports
/// continue on following physical lines, while each row starts on its own
/// physical line.  No sorting, interpolation, renormalization, repair,
/// filename inference, metadata extraction, or other writer behavior is
/// provided by this parser entrypoint.
/// Case-insensitive comments beginning with `Gamma` or `Port Impedance` (also
/// `Port Impedance0`), containing `Terminal data exported` or `Modal data
/// exported`, or containing the complete `S-parameter uses the power
/// definition`, `S-parameter uses the pseudo definition`, or `S-parameter
/// uses the traveling definition` marker are rejected as recognized
/// HFSS/Ansys semantic extensions.
/// Other comments, including ordinary `Port[n] = ...` names, are ignored;
/// recognizing every vendor marker is outside this parser's scope.
pub fn parse_touchstone_v1_0_s(input: &str, nports: usize) -> Result<Network> {
    let matrix_elements = nports.checked_mul(nports).ok_or(Error::SizeOverflow {
        nports,
        quantity: SizeQuantity::MatrixElements,
    })?;
    let record_values = matrix_elements.checked_mul(2).ok_or(Error::SizeOverflow {
        nports,
        quantity: SizeQuantity::RecordValues,
    })?;

    if nports == 0 {
        return Err(Error::InvalidPortCount { nports });
    }

    let mut option: Option<Options> = None;
    let mut data_lines = Vec::new();

    for (line, line_number) in split_lines(input) {
        validate_character_set(line, line_number)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let (before_comment, comment) = split_comment(line);
        if let Some(comment) = comment {
            reject_recognized_extension(comment, line_number)?;
        }
        let content = before_comment.trim();
        if content.is_empty() {
            continue;
        }

        if content.starts_with('[') {
            return Err(Error::UnsupportedKeyword {
                line: line_number,
                keyword: content.to_owned(),
            });
        }

        if content.starts_with('#') {
            // Touchstone specifies that only the first option line controls
            // parsing; deliberately ignore all later option lines, including
            // ones whose tokens would otherwise be malformed.
            if option.is_none() {
                option = Some(parse_option_line(content, line_number)?);
            }
            continue;
        }

        if option.is_none() {
            return Err(Error::MissingOptionLine { line: line_number });
        }

        // Keep the original prefix so token columns refer to the physical
        // source line rather than to the trimmed record text.
        let tokens = tokenize(before_comment);
        if tokens.is_empty() {
            continue;
        }
        data_lines.push(DataLine {
            line: line_number,
            tokens,
        });
    }

    let options = option.ok_or(Error::MissingOptionLine { line: 1 })?;
    if data_lines.is_empty() {
        return Err(Error::EmptyData);
    }

    let (frequency_hz, s_values) = parse_records(&data_lines, nports, record_values, options)?;

    let nfreq = frequency_hz.len();
    let z0_len = nfreq.checked_mul(nports).ok_or(Error::SizeOverflow {
        nports,
        quantity: SizeQuantity::ReferenceImpedanceElements,
    })?;
    let z0_values = vec![Complex64::new(options.resistance, 0.0); z0_len];

    let frequency = Frequency::from_hz(frequency_hz).map_err(Error::Core)?;
    let s = Array3::from_shape_vec((nfreq, nports, nports), s_values).map_err(|_| {
        Error::ConstructionShape {
            quantity: "S-parameter array",
            shape: vec![nfreq, nports, nports],
        }
    })?;
    let z0 = Array2::from_shape_vec((nfreq, nports), z0_values).map_err(|_| {
        Error::ConstructionShape {
            quantity: "reference-impedance array",
            shape: vec![nfreq, nports],
        }
    })?;

    Network::new(frequency, s, z0).map_err(Error::Core)
}

fn split_lines(input: &str) -> Vec<(&str, usize)> {
    let bytes = input.as_bytes();
    let mut lines = Vec::new();
    let mut start = 0usize;
    let mut line_number = 1usize;
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'\r' || bytes[index] == b'\n' {
            lines.push((&input[start..index], line_number));
            if bytes[index] == b'\r'
                && index
                    .checked_add(1)
                    .is_some_and(|next| next < bytes.len() && bytes[next] == b'\n')
            {
                index += 1;
            }
            start = index + 1;
            line_number += 1;
        }
        index += 1;
    }
    if start < input.len() {
        lines.push((&input[start..], line_number));
    }
    lines
}

fn validate_character_set(line: &str, line_number: usize) -> Result<()> {
    for (index, byte) in line.bytes().enumerate() {
        if byte != b'\t' && !(0x20..=0x7e).contains(&byte) {
            return Err(Error::UnsupportedCharacter {
                line: line_number,
                column: index + 1,
            });
        }
    }
    Ok(())
}

fn split_comment(line: &str) -> (&str, Option<&str>) {
    match line.find('!') {
        Some(index) => (&line[..index], Some(&line[index + 1..])),
        None => (line, None),
    }
}

fn reject_recognized_extension(comment: &str, line: usize) -> Result<()> {
    let lower = comment.trim().to_ascii_lowercase();
    // HFSS port-name comments use `Port[n] = ...`; preserve those ordinary
    // comments even when a user-chosen port name happens to mention an
    // impedance or wave term.
    if lower.starts_with("port[") || lower.starts_with("port [") {
        return Ok(());
    }

    let reference = lower.starts_with("port impedance")
        || lower.contains("terminal data exported")
        || lower.contains("modal data exported");
    if reference {
        return Err(Error::UnsupportedExtension {
            line,
            extension: ExtensionKind::ReferenceImpedance,
        });
    }

    let wave_definition = lower.starts_with("gamma")
        || lower.contains("s-parameter uses the power definition")
        || lower.contains("s-parameter uses the pseudo definition")
        || lower.contains("s-parameter uses the traveling definition");
    if wave_definition {
        return Err(Error::UnsupportedExtension {
            line,
            extension: ExtensionKind::WaveDefinition,
        });
    }
    Ok(())
}

fn parse_option_line(line_text: &str, line: usize) -> Result<Options> {
    let body = line_text
        .strip_prefix('#')
        .ok_or_else(|| Error::MalformedOption {
            line,
            message: "option line must begin with '#'".to_owned(),
        })?;
    let tokens = tokenize(body);
    let mut unit = None;
    let mut format = None;
    let mut resistance = None;
    let mut parameter_seen = false;
    let mut index = 0usize;

    while index < tokens.len() {
        let token = tokens[index];
        let lower = token.text.to_ascii_lowercase();
        match lower.as_str() {
            "hz" | "khz" | "mhz" | "ghz" => {
                if unit.is_some() {
                    return Err(Error::MalformedOption {
                        line,
                        message: format!("duplicate frequency unit {:?}", token.text),
                    });
                }
                unit = Some(match lower.as_str() {
                    "hz" => FrequencyUnit::Hz,
                    "khz" => FrequencyUnit::KHz,
                    "mhz" => FrequencyUnit::MHz,
                    "ghz" => FrequencyUnit::GHz,
                    _ => unreachable!(),
                });
            }
            "s" => {
                if parameter_seen {
                    return Err(Error::MalformedOption {
                        line,
                        message: "duplicate network parameter".to_owned(),
                    });
                }
                parameter_seen = true;
            }
            "y" | "z" | "h" | "g" => {
                return Err(Error::UnsupportedParameter {
                    line,
                    parameter: token.text.to_owned(),
                });
            }
            "ri" | "ma" | "db" => {
                if format.is_some() {
                    return Err(Error::MalformedOption {
                        line,
                        message: "duplicate data format".to_owned(),
                    });
                }
                format = Some(match lower.as_str() {
                    "ri" => DataFormat::Ri,
                    "ma" => DataFormat::Ma,
                    "db" => DataFormat::Db,
                    _ => unreachable!(),
                });
            }
            "r" => {
                if resistance.is_some() {
                    return Err(Error::MultipleReferenceValues { line });
                }
                let value_token = tokens
                    .get(index + 1)
                    .ok_or_else(|| Error::MalformedOption {
                        line,
                        message: "reference resistance marker requires one value".to_owned(),
                    })?;
                let value =
                    value_token
                        .text
                        .parse::<f64>()
                        .map_err(|_| Error::MalformedOption {
                            line,
                            message: format!(
                                "reference resistance value {:?} is not a number",
                                value_token.text
                            ),
                        })?;
                if !value.is_finite() || value <= 0.0 {
                    return Err(Error::InvalidReferenceResistance { line, value });
                }
                if tokens
                    .get(index + 2)
                    .is_some_and(|next| looks_like_number(next.text))
                {
                    return Err(Error::MultipleReferenceValues { line });
                }
                resistance = Some(value);
                index += 1;
            }
            _ => {
                return Err(Error::MalformedOption {
                    line,
                    message: format!("unrecognized option token {:?}", token.text),
                });
            }
        }
        index += 1;
    }

    Ok(Options {
        unit: unit.unwrap_or(FrequencyUnit::GHz),
        format: format.unwrap_or(DataFormat::Ma),
        resistance: resistance.unwrap_or(50.0),
    })
}

fn looks_like_number(text: &str) -> bool {
    text.parse::<f64>().is_ok()
}

fn tokenize<'a>(text: &'a str) -> Vec<Token<'a>> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        while index < bytes.len() && (bytes[index] == b' ' || bytes[index] == b'\t') {
            index += 1;
        }
        if index == bytes.len() {
            break;
        }
        let start = index;
        while index < bytes.len() && bytes[index] != b' ' && bytes[index] != b'\t' {
            index += 1;
        }
        tokens.push(Token {
            text: &text[start..index],
            column: start + 1,
        });
    }
    tokens
}

fn parse_number(token: Token<'_>, line: usize) -> Result<f64> {
    let value = token
        .text
        .parse::<f64>()
        .map_err(|_| Error::MalformedNumber {
            line,
            column: token.column,
            token: token.text.to_owned(),
        })?;
    if value.is_finite() {
        return Ok(value);
    }
    if value.is_infinite() && token.text.contains(['e', 'E']) {
        return Err(Error::NumericOverflow {
            line,
            column: token.column,
            token: token.text.to_owned(),
        });
    }
    Err(Error::NonFiniteNumber {
        line,
        column: token.column,
        token: token.text.to_owned(),
    })
}

fn parse_records(
    lines: &[DataLine<'_>],
    nports: usize,
    record_values: usize,
    options: Options,
) -> Result<(Vec<f64>, Vec<Complex64>)> {
    if nports <= 2 {
        parse_small_records(lines, nports, record_values, options)
    } else {
        parse_matrix_records(lines, nports, options)
    }
}

fn parse_small_records(
    lines: &[DataLine<'_>],
    nports: usize,
    record_values: usize,
    options: Options,
) -> Result<(Vec<f64>, Vec<Complex64>)> {
    let expected_tokens = record_values.checked_add(1).ok_or(Error::SizeOverflow {
        nports,
        quantity: SizeQuantity::RecordValues,
    })?;
    let expected_pairs = record_values / 2;
    let mut frequencies = Vec::new();
    let mut values = Vec::new();

    for data_line in lines {
        let actual_tokens = data_line.tokens.len();
        if nports == 2 && actual_tokens == 5 && !frequencies.is_empty() {
            return Err(Error::NoiseData {
                line: data_line.line,
            });
        }
        if actual_tokens < expected_tokens {
            return Err(Error::IncompleteRecord {
                line: data_line.line,
                expected_pairs,
                actual_pairs: actual_tokens.saturating_sub(1) / 2,
            });
        }
        if actual_tokens > expected_tokens {
            return Err(Error::SurplusRecord {
                line: data_line.line,
                expected_pairs,
                actual_pairs: actual_tokens.saturating_sub(1) / 2,
            });
        }
        let frequency = parse_frequency(data_line.tokens[0], data_line.line, options.unit)?;
        check_frequency_order(&frequencies, frequency, data_line.line)?;
        let mut record = Vec::with_capacity(expected_pairs);
        for pair in 0..expected_pairs {
            let first = data_line.tokens[1 + pair * 2];
            let second = data_line.tokens[2 + pair * 2];
            record.push(decode_pair(first, second, data_line.line, options.format)?);
        }
        frequencies.push(frequency);

        if nports == 2 {
            // Legacy v1 order is N11, N21, N12, N22; Network stores rows
            // contiguously as N11, N12, N21, N22.
            values.extend([record[0], record[2], record[1], record[3]]);
        } else {
            values.extend(record);
        }
    }
    Ok((frequencies, values))
}

fn parse_matrix_records(
    lines: &[DataLine<'_>],
    nports: usize,
    options: Options,
) -> Result<(Vec<f64>, Vec<Complex64>)> {
    let mut frequencies = Vec::new();
    let mut values = Vec::new();
    let mut index = 0usize;

    while index < lines.len() {
        let first = &lines[index];
        if first.tokens.len() < 3 {
            return Err(Error::IncompleteRecord {
                line: first.line,
                expected_pairs: nports,
                actual_pairs: first.tokens.len().saturating_sub(1) / 2,
            });
        }
        let frequency = parse_frequency(first.tokens[0], first.line, options.unit)?;
        check_frequency_order(&frequencies, frequency, first.line)?;
        let mut matrix = Vec::new();

        let first_row_pairs = checked_pair_count(&first.tokens[1..], first.line)?;
        validate_line_pair_count(first.line, first_row_pairs, nports)?;
        if nports <= 4 && first_row_pairs != nports {
            return Err(if first_row_pairs < nports {
                Error::IncompleteRecord {
                    line: first.line,
                    expected_pairs: nports,
                    actual_pairs: first_row_pairs,
                }
            } else {
                Error::SurplusRecord {
                    line: first.line,
                    expected_pairs: nports,
                    actual_pairs: first_row_pairs,
                }
            });
        }
        decode_pairs_into(&first.tokens[1..], first.line, options.format, &mut matrix)?;
        index += 1;
        finish_row(
            lines,
            &mut index,
            nports,
            first.line,
            options.format,
            &mut matrix,
            0,
        )?;

        for row in 1..nports {
            let row_line = lines.get(index).ok_or(Error::IncompleteRecord {
                line: first.line,
                expected_pairs: nports,
                actual_pairs: 0,
            })?;
            let row_pairs = checked_pair_count(&row_line.tokens, row_line.line)?;
            validate_line_pair_count(row_line.line, row_pairs, nports)?;
            if nports <= 4 && row_pairs != nports {
                return Err(if row_pairs < nports {
                    Error::IncompleteRecord {
                        line: row_line.line,
                        expected_pairs: nports,
                        actual_pairs: row_pairs,
                    }
                } else {
                    Error::SurplusRecord {
                        line: row_line.line,
                        expected_pairs: nports,
                        actual_pairs: row_pairs,
                    }
                });
            }
            decode_pairs_into(&row_line.tokens, row_line.line, options.format, &mut matrix)?;
            index += 1;
            finish_row(
                lines,
                &mut index,
                nports,
                row_line.line,
                options.format,
                &mut matrix,
                row,
            )?;
        }

        if matrix.len()
            != nports.checked_mul(nports).ok_or(Error::SizeOverflow {
                nports,
                quantity: SizeQuantity::MatrixElements,
            })?
        {
            return Err(Error::MalformedRecord {
                line: first.line,
                message: "matrix row count did not produce a square record".to_owned(),
            });
        }
        frequencies.push(frequency);
        values.extend(matrix);
    }

    Ok((frequencies, values))
}

fn checked_pair_count(tokens: &[Token<'_>], line: usize) -> Result<usize> {
    if tokens.len() % 2 != 0 {
        return Err(Error::MalformedRecord {
            line,
            message: "parameter data must contain complete real/imaginary or magnitude/angle pairs"
                .to_owned(),
        });
    }
    Ok(tokens.len() / 2)
}

fn validate_line_pair_count(line: usize, pairs: usize, nports: usize) -> Result<()> {
    if pairs == 0 {
        return Err(Error::IncompleteRecord {
            line,
            expected_pairs: nports,
            actual_pairs: 0,
        });
    }
    if pairs > 4 {
        return Err(Error::SurplusRecord {
            line,
            expected_pairs: 4,
            actual_pairs: pairs,
        });
    }
    Ok(())
}

fn finish_row(
    lines: &[DataLine<'_>],
    index: &mut usize,
    nports: usize,
    row_line: usize,
    format: DataFormat,
    matrix: &mut Vec<Complex64>,
    row: usize,
) -> Result<()> {
    if nports <= 4 || matrix.len() % nports == 0 {
        return Ok(());
    }

    let row_start = row * nports;
    while matrix.len() < row_start + nports {
        let continuation = lines.get(*index).ok_or(Error::IncompleteRecord {
            line: row_line,
            expected_pairs: nports,
            actual_pairs: matrix.len().saturating_sub(row_start),
        })?;
        let pairs = checked_pair_count(&continuation.tokens, continuation.line)?;
        validate_line_pair_count(continuation.line, pairs, nports)?;
        let remaining = nports - (matrix.len() - row_start);
        if pairs > remaining {
            return Err(Error::SurplusRecord {
                line: continuation.line,
                expected_pairs: remaining,
                actual_pairs: pairs,
            });
        }
        decode_pairs_into(&continuation.tokens, continuation.line, format, matrix)?;
        *index += 1;
    }
    Ok(())
}

fn decode_pairs_into(
    tokens: &[Token<'_>],
    line: usize,
    format: DataFormat,
    output: &mut Vec<Complex64>,
) -> Result<()> {
    for pair in tokens.chunks_exact(2) {
        output.push(decode_pair(pair[0], pair[1], line, format)?);
    }
    Ok(())
}

fn parse_frequency(token: Token<'_>, line: usize, unit: FrequencyUnit) -> Result<f64> {
    let raw = match parse_number(token, line) {
        Ok(value) => value,
        Err(Error::NonFiniteNumber { .. }) => {
            // `parse_number` has already established that this token is a
            // valid floating-point spelling.  Re-parsing only to retain the
            // domain-specific diagnostic cannot fail, but keep a NaN fallback
            // rather than introducing an input-triggered panic.
            let value = token.text.parse::<f64>().unwrap_or(f64::NAN);
            return Err(Error::NonFiniteFrequency { line, value });
        }
        Err(Error::NumericOverflow { .. }) => {
            let value = token.text.parse::<f64>().unwrap_or(f64::INFINITY);
            return Err(Error::FrequencyOverflow { line, value });
        }
        Err(error) => return Err(error),
    };
    if raw < 0.0 {
        return Err(Error::NegativeFrequency { line, value: raw });
    }
    let scaled = raw * unit.multiplier();
    if !scaled.is_finite() {
        return Err(Error::FrequencyOverflow { line, value: raw });
    }
    if scaled < 0.0 {
        return Err(Error::NegativeFrequency {
            line,
            value: scaled,
        });
    }
    Ok(scaled)
}

fn check_frequency_order(frequencies: &[f64], current: f64, line: usize) -> Result<()> {
    if let Some(previous) = frequencies.last().copied() {
        if current <= previous {
            return Err(Error::FrequencyNotStrictlyIncreasing {
                line,
                previous,
                current,
            });
        }
    }
    Ok(())
}

fn decode_pair(
    first: Token<'_>,
    second: Token<'_>,
    line: usize,
    format: DataFormat,
) -> Result<Complex64> {
    let first_value = parse_number(first, line)?;
    let second_value = parse_number(second, line)?;
    match format {
        DataFormat::Ri => finite_complex(
            Complex64::new(first_value, second_value),
            line,
            first.column,
            format,
        ),
        DataFormat::Ma => {
            if first_value < 0.0 {
                return Err(Error::NegativeMagnitude {
                    line,
                    column: first.column,
                    value: first_value,
                });
            }
            let radians = second_value * (std::f64::consts::PI / 180.0);
            finite_complex(
                Complex64::new(first_value * radians.cos(), first_value * radians.sin()),
                line,
                first.column,
                format,
            )
        }
        DataFormat::Db => {
            let magnitude = 10.0_f64.powf(first_value / 20.0);
            let radians = second_value * (std::f64::consts::PI / 180.0);
            finite_complex(
                Complex64::new(magnitude * radians.cos(), magnitude * radians.sin()),
                line,
                first.column,
                format,
            )
        }
    }
}

fn finite_complex(
    value: Complex64,
    line: usize,
    column: usize,
    format: DataFormat,
) -> Result<Complex64> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(value)
    } else {
        Err(Error::ConversionOverflow {
            line,
            column,
            format,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_defaults_and_one_port_ri() {
        let network = parse_touchstone_v1_0_s("# S RI\n1 0.2 -0.1\n2 0.3 0.4\n", 1).unwrap();
        assert_eq!(network.frequency().hz(), &[1.0e9, 2.0e9]);
        assert_eq!(network.z0()[[0, 0]], Complex64::new(50.0, 0.0));
        assert_eq!(network.s()[[1, 0, 0]], Complex64::new(0.3, 0.4));
    }

    #[test]
    fn parses_legacy_two_port_order() {
        let network =
            parse_touchstone_v1_0_s("# Hz S RI R 75\n1 11 0 21 0 12 0 22 0\n", 2).unwrap();
        assert_eq!(network.s()[[0, 0, 0]], Complex64::new(11.0, 0.0));
        assert_eq!(network.s()[[0, 0, 1]], Complex64::new(12.0, 0.0));
        assert_eq!(network.s()[[0, 1, 0]], Complex64::new(21.0, 0.0));
        assert_eq!(network.s()[[0, 1, 1]], Complex64::new(22.0, 0.0));
    }

    #[test]
    fn parses_three_port_rows_and_five_port_continuations() {
        let three = parse_touchstone_v1_0_s(
            "# GHz S RI\n1 11 0 12 0 13 0\n 21 0 22 0 23 0\n 31 0 32 0 33 0\n",
            3,
        )
        .unwrap();
        assert_eq!(three.s()[[0, 1, 0]], Complex64::new(21.0, 0.0));
        assert_eq!(three.s()[[0, 2, 1]], Complex64::new(32.0, 0.0));

        let five = parse_touchstone_v1_0_s(
            "# GHz S RI\n1 11 0 12 0 13 0 14 0\n15 0\n21 0 22 0 23 0 24 0\n25 0\n31 0 32 0 33 0 34 0\n35 0\n41 0 42 0 43 0 44 0\n45 0\n51 0 52 0 53 0 54 0\n55 0\n",
            5,
        )
        .unwrap();
        assert_eq!(five.s()[[0, 0, 4]], Complex64::new(15.0, 0.0));
        assert_eq!(five.s()[[0, 4, 0]], Complex64::new(51.0, 0.0));
    }

    #[test]
    fn rejects_v2_keyword_and_hfss_extension() {
        assert!(matches!(
            parse_touchstone_v1_0_s("[Version] 2.1\n#\n1 0 0\n", 1),
            Err(Error::UnsupportedKeyword { .. })
        ));
        assert!(matches!(
            parse_touchstone_v1_0_s("# GHz S RI\n! Port Impedance 50 0\n1 0 0\n", 1),
            Err(Error::UnsupportedExtension { .. })
        ));
    }
}
