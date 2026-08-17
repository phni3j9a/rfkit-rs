# AGENTS.md — rfkit-rs engineering contract

This repository is intended to be developed aggressively with AI assistance, but RF correctness must remain stronger than implementation velocity.

## Primary objective

Build a production-quality Rust-native RF/microwave network-analysis library that can eventually cover the useful functionality of scikit-rf while preserving Rust-native APIs and strong numerical verification.

## Source hierarchy

Use sources in this order when resolving behavior:

1. RF/microwave mathematics and published standards/papers.
2. scikit-rf current behavior and tests as a reference oracle.
3. Instrument/file-format specifications (for example Touchstone).
4. rust-rf and rust-skrf as prior-art implementations, never as unquestioned truth.

## Mandatory implementation workflow

For every substantive numerical feature:

1. Identify the mathematical definition and edge cases.
2. Inspect equivalent scikit-rf behavior/tests.
3. Inspect rust-rf/rust-skrf only for prior-art comparison when useful.
4. Decide explicitly whether prior-art code should be `REUSE`, `ADAPT`, or `REWRITE`.
5. Record any copied/adapted third-party code or fixture in `docs/PROVENANCE.md` and retain required license notices.
6. Implement a Rust-native API.
7. Add deterministic unit tests.
8. Add differential/conformance fixtures against scikit-rf when applicable.
9. Add property/invariant tests where mathematically meaningful.
10. Run fmt, clippy, and the full test suite before considering the task complete.

## Numerical rules

- Do not assume 50-ohm reference impedance.
- Support complex reference impedance where the underlying operation permits it.
- Prefer N-port algorithms over special-casing 2-port behavior unless the mathematics is inherently 2-port.
- Document wave definitions where relevant (power, pseudo, traveling).
- Every conversion should have round-trip tests where mathematically valid.
- Near-singular matrices and ill-conditioned operations need deliberate error/tolerance handling, not silent garbage.
- Never claim scikit-rf compatibility without a machine-executed comparison.

## Conformance strategy

`tools/oracle/` is reserved for Python/scikit-rf reference generation. Keep oracle output deterministic and versioned by:

- scikit-rf version/commit
- NumPy version where relevant
- random seed
- operation name
- tolerance policy

Prefer generated fixtures or a reproducible test runner over hand-copied expected values.

## Architecture

Keep the numerical core independent of plotting frameworks, Python, WASM, GUI frameworks, and instrument I/O. Add integrations as separate crates when they become necessary.

## Third-party provenance

BSD-3-Clause permits reuse but attribution still matters. Do not erase lineage to make code appear original. If copying or closely adapting code, preserve the applicable copyright/license terms and log the source commit/path.

## Avoid

- bulk-porting thousands of lines before the conformance harness exists
- README feature checkboxes unsupported by tests
- Python API mimicry that fights Rust's type system
- introducing plotting/UI dependencies into the numerical core
- optimizing before correctness is characterized
