# rfkit-rs

A Rust-native RF and microwave network-analysis library focused on **correctness, numerical conformance, portability, and production use**.

> Working name. The repository/crate can be renamed before the first public release.

## Why this project exists

Python and scikit-rf are excellent for interactive RF analysis, but native applications benefit from a small, fast, portable core that is easy to embed in desktop, mobile, server, and WASM targets.

This project does **not** aim to blindly transliterate Python into Rust. scikit-rf is treated as a mature reference implementation and numerical oracle while the public API is designed for Rust.

## Non-negotiable principles

1. **Correctness before feature count.** A feature is not complete because it compiles.
2. **Differential verification.** Where practical, numerical behavior is compared against scikit-rf over deterministic and randomized fixtures.
3. **RF invariants.** Round trips, reciprocity, passivity, dimensional consistency, and physically meaningful edge cases are tested explicitly.
4. **N-port and complex Z0 from the start.** Do not accidentally design a 2-port/50-ohm-only API.
5. **Provenance is explicit.** Any code or fixture adapted from scikit-rf, rust-rf, rust-skrf, papers, or other projects must retain the required attribution and be recorded in `docs/PROVENANCE.md`.
6. **Rust-native architecture.** Python-specific dynamic APIs are not compatibility requirements.

## Initial scope

The first vertical slice should make these excellent before expanding broadly:

- Frequency and Network data model
- Touchstone 1.x / 2.x / 2.1
- S/Z/Y conversions
- renormalization including complex and per-port Z0
- interpolation
- connect / inner-connect / cascade
- de-embedding primitives
- time-domain transforms
- Smith-chart data generation as a presentation-independent layer

Calibration, media models, vector fitting, VNA control, and bindings come after the core conformance harness is trustworthy.

## Repository layout

```text
crates/rfkit-core/   Rust RF numerical core
tools/oracle/        scikit-rf reference/differential-test tools
docs/                architecture, conformance and provenance policy
```

## Development

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

For Codex, read `AGENTS.md` before implementation.

## License

BSD-3-Clause. See `LICENSE` and `docs/PROVENANCE.md` before incorporating third-party code or fixtures.
