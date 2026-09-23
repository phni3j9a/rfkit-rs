# Third-party licenses

Place license notices for copied/adapted code and fixtures here.

Potential reference projects include scikit-rf, rust-rf, and rust-skrf, all of which must be evaluated at the exact revision used before importing material.

`rfkit-core` uses nalgebra `0.33.3` as a normal private dependency for its
full dense complex SVD adapter (Issue #98).  The crate is Apache-2.0 licensed;
no nalgebra source or license text is copied into this repository.  Its exact
resolved dependency graph is recorded in `Cargo.lock`, and the workspace's
declared Rust `1.85` MSRV remains unchanged.
