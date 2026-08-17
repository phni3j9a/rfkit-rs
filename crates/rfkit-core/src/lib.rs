//! Rust-native RF and microwave network analysis primitives.
//!
//! The project starts deliberately small: a trustworthy core data model first,
//! then numerical operations backed by differential tests against scikit-rf.

use ndarray::{Array2, Array3};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors produced while constructing or manipulating RF network data.
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("frequency axis must not be empty")]
    EmptyFrequency,

    #[error("S-parameter shape must be (nfreq, nport, nport)")]
    InvalidSShape,

    #[error("reference-impedance shape must be (nfreq, nport)")]
    InvalidZ0Shape,
}

/// Frequency axis in hertz.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Frequency {
    hz: Vec<f64>,
}

impl Frequency {
    /// Creates a frequency axis in hertz.
    pub fn from_hz(hz: Vec<f64>) -> Result<Self, Error> {
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
    pub fn new(
        frequency: Frequency,
        s: Array3<Complex64>,
        z0: Array2<Complex64>,
    ) -> Result<Self, Error> {
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
}
