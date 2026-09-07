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
}

impl fmt::Display for ConversionStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::SToZ => "S→Z",
            Self::ZToY => "Z→Y",
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
}
