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
- Touchstone ingress (the current public slice is an explicitly limited
  Touchstone 1.0 single-ended S-parameter reader; broader format support is
  future work)
- S/Z/Y conversions
- renormalization including complex and per-port Z0
- interpolation
- connect / inner-connect / cascade
- de-embedding primitives
- time-domain transforms
- Smith-chart data generation as a presentation-independent layer

Calibration, media models, vector fitting, VNA control, and bindings come after the core conformance harness is trustworthy.

The scope above is directional context rather than an ordered autonomous backlog.

## Construct a Network from modelled Z or Y parameters

`rfkit-core` exposes the verified power-wave ingress paths as explicit,
provisional 0.x constructors. `from_z_power` accepts owned frequency-major
impedance matrices in ohms; `from_y_via_z_power` accepts admittance matrices in
siemens and deliberately composes Y→Z→S; `from_y_direct_power` solves the
Kurokawa Y→S equation directly. All require an explicit `(frequency, port)`
reference-impedance array in ohms. The returned S matrices are dimensionless
and the supplied frequency labels, order, port order, and references are
preserved exactly.

Frequency samples are pointwise labels for these constructors: they must be
nonempty and match the parameter first axis, but they are not required to be
finite, non-negative, sorted, or unique. Reference values use the existing
power-wave `abs(Re(z0))` domain, so finite complex and negative-real values are
accepted; zero-real and non-finite values are rejected. A singular or zero Y
is rejected in the composed Y→Z stage, but is supported by
`from_y_direct_power` whenever the direct `A=F(I+G Y)` system is nonsingular
and finite. The direct path has no hidden fallback to the composed path,
inverse/pseudoinverse, cutoff, regularization, or identity shortcut.

The direct constructor's ingress domain does not broaden existing downstream
conversions. A network made from singular Y may not succeed through
`to_z_power`, `to_y_power`, renormalization, or another operation that needs
an invertible intermediate matrix; those restrictions remain explicit and
unchanged on the composed constructor. When a network's S representation has
singular `I-S` but the direct S→Y system is nonsingular, use the explicitly
named `to_y_direct_power` extraction path. `to_y_power` remains the composed
S→Z→Y path and retains its structured singular-stage error; it does not hide a
direct-conversion fallback.

```rust
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{Frequency, Network};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frequency = Frequency::from_hz(vec![1.0e9, 2.0e9])?;
    let z = Array3::from_shape_vec(
        (2, 1, 1),
        vec![Complex64::new(100.0, 0.0), Complex64::new(100.0, 0.0)],
    )?;
    let z0 = Array2::from_elem((2, 1), Complex64::new(50.0, 0.0));
    let network = Network::from_z_power(frequency, z, z0)?;

    // The existing analysis API is immediately available: z=100 Ω at a 50 Ω
    // real reference gives the analytical one-port S=(z-z0)/(z+z0)=1/3.
    let y_siemens = network.to_y_power()?;
    assert!((network.s()[[0, 0, 0]].re - 1.0 / 3.0).abs() < 1.0e-14);
    assert!(y_siemens[[0, 0, 0]].re > 0.0);
    Ok(())
}
```

## Read, analyze, and write Touchstone text

The `rfkit-touchstone` crate provides pure in-memory Touchstone v1.0 ingress
and an explicit S/RI/Hz egress function. The caller chooses the port count for
parsing and can then use the returned canonical `rfkit-core::Network` with
existing analysis methods:

```rust
use rfkit_touchstone::parse_touchstone_v1_0_s;

fn main() -> rfkit_touchstone::Result<()> {
    let network = parse_touchstone_v1_0_s("# MHz S RI R 75\n10 0.2 0\n", 1)?;
    let z = network.to_z_power()?;
    assert!((z[[0, 0, 0]].re - 112.5).abs() < 1e-12);
    Ok(())
}
```

To export a validated network, borrow it with the explicitly scoped writer:

```rust
use rfkit_touchstone::write_touchstone_v1_0_s_ri_hz;

fn write(network: &rfkit_core::Network) -> rfkit_touchstone::Result<String> {
    write_touchstone_v1_0_s_ri_hz(network)
}
```

The writer always emits ASCII `# Hz S RI R <reference>` text with LF line
endings and a final newline. It requires a nonempty finite, nonnegative,
strictly increasing frequency axis; finite S components; positive finite real
reference impedance; and one exact common reference scalar at every frequency
and port. It never sorts, repairs, interpolates, renormalizes, or emits a
partial result. Two-port records use the v1.0 physical order
`S11,S21,S12,S22`; three-port and larger records are row-major with at most
four parameter pairs per physical line and continuation lines for wider rows.
Rust's shortest binary64 representation is used so the existing reader
reconstructs finite values numerically, including extreme and subnormal
values. MA/DB, Touchstone v2.x, metadata, noise, mixed-mode, and filesystem
I/O remain outside this API.

The ingress, existing public transformation, and egress APIs compose directly:

```rust
use ndarray::Array2;
use num_complex::Complex64;
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

fn main() -> rfkit_touchstone::Result<()> {
    let source = parse_touchstone_v1_0_s("# Hz S RI R 75\n1000000000 0.2 0\n", 1)?;
    let transformed = source.renormalize_power(Array2::from_elem(
        (1, 1),
        Complex64::new(100.0, 0.0),
    ))?;
    let text = write_touchstone_v1_0_s_ri_hz(&transformed)?;
    let reread = parse_touchstone_v1_0_s(&text, 1)?;

    assert_eq!(reread.z0()[[0, 0]], Complex64::new(100.0, 0.0));
    // z=75*(1+0.2)/(1-0.2)=112.5 ohm, so S at 100 ohm is 1/17.
    assert!((reread.s()[[0, 0, 0]].re - 1.0 / 17.0).abs() < 1.0e-12);
    Ok(())
}
```

The direct and composed S→Y methods intentionally coexist. This makes a
singular-I-S but finite-Y case observable after a Touchstone round trip without
changing the domain or error behavior of existing conversions:

```rust
use ndarray::{Array2, Array3};
use num_complex::Complex64;
use rfkit_core::{ConversionStage, Error, Frequency, Network};
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frequency = Frequency::from_hz(vec![1.0e9, 2.0e9])?;
    let y = Array3::from_shape_vec(
        (2, 2, 2),
        vec![
            Complex64::new(0.01, 0.0),
            Complex64::new(-0.01, 0.0),
            Complex64::new(-0.01, 0.0),
            Complex64::new(0.01, 0.0),
            Complex64::new(0.01, 0.0),
            Complex64::new(-0.01, 0.0),
            Complex64::new(-0.01, 0.0),
            Complex64::new(0.01, 0.0),
        ],
    )?;
    let z0 = Array2::from_elem((2, 2), Complex64::new(50.0, 0.0));
    let source = Network::from_y_direct_power(frequency, y, z0)?;
    let text = write_touchstone_v1_0_s_ri_hz(&source)?;
    let reread = parse_touchstone_v1_0_s(&text, 2)?;

    let extracted_y = reread.to_y_direct_power()?;
    for (actual, expected) in extracted_y.iter().zip([
        Complex64::new(0.01, 0.0),
        Complex64::new(-0.01, 0.0),
        Complex64::new(-0.01, 0.0),
        Complex64::new(0.01, 0.0),
        Complex64::new(0.01, 0.0),
        Complex64::new(-0.01, 0.0),
        Complex64::new(-0.01, 0.0),
        Complex64::new(0.01, 0.0),
    ]) {
        assert!((*actual - expected).norm() < 1.0e-14);
    }

    let composed_error = reread.to_y_power().expect_err("I-S is singular");
    assert!(matches!(
        composed_error,
        Error::Singular {
            stage: ConversionStage::SToZ,
            ..
        }
    ));
    Ok(())
}
```

The supported input domain is Touchstone v1.0 single-ended S data in RI, MA,
or DB pairs, with Hz/kHz/MHz/GHz units, one finite positive real `R` scalar,
comments, and v1.0 row/continuation layout. Frequencies must be finite,
non-negative, and strictly increasing. The reader does not infer filenames,
perform I/O, sort or repair data, renormalize, preserve metadata, or accept
Touchstone v2.x keywords or mixed-mode/noise data. Its case-insensitive
HFSS/Ansys semantic-extension boundary rejects comment text beginning with
`Gamma` or `Port Impedance` (including `Port Impedance0`), comment text
containing `Terminal data exported` or `Modal data exported`, and explicit
`S-parameter uses the power definition`, `S-parameter uses the pseudo
definition`, or `S-parameter uses the traveling definition` comments.
Ordinary comments, including `Port[n] = ...` port-name comments, remain
ignorable; universal vendor-marker recognition is outside this reader's scope.

## Repository layout

```text
crates/rfkit-core/       Rust RF numerical core
crates/rfkit-touchstone/ pure Touchstone 1.0 S-parameter text ingress/egress
tools/oracle/        scikit-rf reference/differential-test tools
docs/                architecture, development, conformance and provenance policy
```

## Autonomous loop engineering

The repository supports a bounded GitHub-centered autonomous development loop:

```text
Codex Planner → Green/Yellow decision → one loop:ready Issue → Codex Worker → reviewed PR → next Planner cycle
```

The default autonomous WIP is one implementation at a time. Higher timer frequency is used to reduce idle latency, not to manufacture additional Issues.

The loop does not require human pre-approval for every provisional public method. Green work follows established semantics; Yellow work records and independently reviews a bounded reversible design choice. Human approval is reserved for Red decisions such as stabilization, unresolved authoritative RF conflicts, unresolved provenance or licensing obligations, major difficult-to-reverse architecture changes, and release or publication. A blocked decision does not stop unrelated eligible work once the WIP slot is clear.

ChatGPT is intentionally outside the normal scheduled execution path. It can act as a governor/auditor when the human owner asks whether recent autonomous work, including its Green/Yellow classifications, is producing meaningful RF capability, correctness, and leverage rather than activity for its own sake.

See:

- `docs/DEVELOPMENT_DIRECTION.md` for the project north star and meaningful-progress criteria;
- `docs/LOOP_ENGINEERING.md` for Planner/Worker roles, dispatch state, WIP, merge gates, prioritization, auditing, and escalation;
- `docs/CODEX_AUTOMATION.md` for thin scheduled `codex exec` instructions and the recommended four-cycle cadence.

GitHub repository documents are the source of truth; automation prompts should not carry a duplicated long-lived roadmap or policy.

## Development

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

For Codex, read `AGENTS.md` before implementation.

## License

BSD-3-Clause. See `LICENSE` and `docs/PROVENANCE.md` before incorporating third-party code or fixtures.
