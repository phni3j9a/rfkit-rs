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
use rfkit_core::{Frequency, Network};
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
    // Keep unequal complex references in the model, then make the explicit
    // common positive-real reference required by the Touchstone writer.
    let source_z0 = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(43.0, 7.0),
            Complex64::new(68.0, -11.0),
            Complex64::new(47.0, 5.0),
            Complex64::new(71.0, -9.0),
        ],
    )?;
    let source = Network::from_y_direct_power(frequency, y, source_z0)?;
    let common_z0 = Array2::from_elem((2, 2), Complex64::new(75.0, 0.0));
    let transformed = source.renormalize_direct_power(common_z0)?;
    let text = write_touchstone_v1_0_s_ri_hz(&transformed)?;
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

`renormalize_direct_power` is the explicit direct Kurokawa wave-change path
for this workflow. It supports singular physical Z/Y cases because it solves
the direct wave relation between the stored source references and the caller's
target references. The existing `renormalize_power` remains the composed
S→Z→S operation, including its existing singular-stage behavior; callers
choose the domain explicitly by the method name. Direct renormalization does
not silently fall back, route through Y, regularize, or choose a target
reference.

## Remove a known fixture with a power-wave inverse cascade

`Network::inverse_cascade_power` returns an owned inverse of an ordered even-port
cascade:

```rust
let fixture_inverse = fixture.inverse_cascade_power()?;
```

The source ports are two equal ordered groups `[left..., right...]`; the result
uses the fixed `[old right, old left]` order. With `P` exchanging those groups,
the operation is `S_inverse = P S⁻¹ P` and its references are exactly
`P conj(z0)`. It is a Kurokawa wave-reversal operation, not an elementwise
reciprocal or a port permutation. The implementation computes and stores the
full inverse by solving `S X = I` with the checked exact-pivot solver; it does
not use an elementwise reciprocal or silently convert through Z/Y.

The operation validates every finite frequency label, S/z0 value and
nonzero-real reference before indexing. At every sample the full S matrix and
both directional transmission blocks `S[right,left]` and `S[left,right]` must
be exactly nonsingular. Errors identify the full-S, forward-transmission, or
reverse-transmission stage and retain frequency/pivot or coordinate context;
finite near-singular systems remain eligible when their checked arithmetic
stays finite. Frequency order, signed-zero bits, and references are otherwise
preserved according to the contract, including unequal complex or negative-real
references under the repository's algebraic power-wave extension.

Known-fixture removal is explicit about physical orientation and connection
ports. For a two-port measured cascade built as `left[1] → dut[0]` followed by
`[1] → right[0]`, callers can remove either side first with the existing
`connect_direct_power` method, then explicitly renormalize the recovered DUT to
the writer's common positive-real reference. Inverse networks can be active or
noncausal mathematical removal operators; this method is not noise
de-embedding, automatic calibration, a pole/stability claim, or a general
scikit-rf compatibility promise.

The focused Touchstone load → cascade → remove → write/read workflow is
executable with:

```text
cargo run -p rfkit-touchstone --example inverse_cascade_touchstone
```

This additive Yellow 0.x decision keeps the fixed ordering and transmission
domain visible at the call site. Alternatives such as plain reference swapping,
a two-port-only API, an inferred pairing map, or a broad transfer-parameter
hierarchy were rejected. Removing the method, diagnostics, fixture, tests, and
documentation during 0.x requires no data migration; existing Network,
connection, renormalization, and Touchstone APIs remain unchanged.

## Cascade coupled even-port fixtures in one direct solve

`Network::cascade_direct_power` composes two equal ordered `2N`-port networks
through all corresponding internal pairs in one simultaneous Kurokawa
power-wave V/I solve:

```rust
let cascaded = left.cascade_direct_power(&right)?;
```

Inputs use `[left_0..left_(N-1), right_0..right_(N-1)]`; every
`self.right_k` is joined to `other.left_k`, and the result is ordered
`[self.left..., other.right...]`.  The joint equation is

```text
(C + D S_ii) X = -D S_ie
S_out = S_ee + S_ei X
```

so full within-group coupling is retained.  This is not a wrapper around
repeated one-port connections: a partial one-port junction can be singular
while the complete coupled group junction is nonsingular.  The operation
supports finite unequal/per-port/frequency-dependent complex references,
including signed negative-real references with nonzero real parts, and copies
the surviving references exactly.  It requires equal finite frequency grids,
does no interpolation or hidden renormalization, and leaves both borrowed
inputs unchanged.  Only an exactly zero evaluated joint pivot is singular;
finite near-singular arithmetic remains eligible.

The operation deliberately does not add arbitrary pair maps, topology or
calibration semantics.  Use `permute_ports` when a fixture's physical order
differs, and explicitly call `renormalize_direct_power` before Touchstone
export when the resulting references are not the writer's one common
positive-real scalar.  A focused load → simultaneous cascade → inverse removal
→ explicit writer-compatible renormalization → write/read workflow is
available as:

```text
cargo test -p rfkit-touchstone --test public_cascade_direct_workflow
cargo run -p rfkit-touchstone --example cascade_direct_power_touchstone
```

The canonical pinned differential fixture is
`tools/oracle/fixtures/power_wave_cascade_direct_four_port_complex_z0.json`;
its expected S is generated through scikit-rf 2.0.1 and explicitly restored
to power waves after `Network.cascade`.  Complex and negative-real references,
the partial-singular boundary witness, and singular/near-singular decisions
remain independently authored equation/invariant coverage rather than a broad
scikit-rf compatibility promise.

This additive Yellow 0.x decision keeps the fixed group ordering and direct
joint-solve domain visible at the call site. Caller-side sequential one-port
repetition was rejected because it loses the simultaneous partial-singular
domain; transfer/Z/Y composition was rejected because it adds avoidable
invertibility restrictions; and arbitrary pairing maps, graph/topology
cascades, calibration, and broader parameter hierarchies were rejected because
they are outside this bounded coupled-fixture workflow. Removing the method,
diagnostics, fixture, tests, and documentation during 0.x requires no data
migration; existing Network, connection, renormalization, and Touchstone APIs
remain unchanged.

## Reorder physical ports before Touchstone export

`Network::permute_ports` is the explicit, wave-definition-independent way to
put an owned network's physical ports into a caller's required order. Its
mapping is deliberately written as `order[new_port] = old_port`: for example,
`[2, 0, 1]` places original port 2 first, original port 0 second, and original
port 1 third.

```rust
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

fn reorder_for_export() -> rfkit_touchstone::Result<String> {
    let source = parse_touchstone_v1_0_s(
        "# Hz S RI R 50\n\
         1000000000 0.11 0.01 0.12 0.02 0.13 0.03\n\
         0.21 0.01 0.22 0.02 0.23 0.03\n\
         0.31 0.01 0.32 0.02 0.33 0.03\n",
        3,
    )?;
    let reordered = source.permute_ports(&[2, 0, 1])?;
    assert_eq!(
        reordered.s()[[0, 0, 0]],
        num_complex::Complex64::new(0.33, 0.03),
    );
    assert_eq!(
        reordered.s()[[0, 0, 1]],
        num_complex::Complex64::new(0.31, 0.01),
    );
    assert_eq!(
        reordered.s()[[0, 1, 0]],
        num_complex::Complex64::new(0.13, 0.03),
    );
    write_touchstone_v1_0_s_ri_hz(&reordered)
}
```

For every frequency, the returned network copies the source values according
to `out.s[f, new_row, new_column] = source.s[f, order[new_row],
order[new_column]]` and `out.z0[f, new_port] = source.z0[f, order[new_port]]`.
The frequency samples and order are retained exactly, the source and mapping
slice are not modified, and no conversion, renormalization, interpolation,
finite-value check, or reference-impedance selection is performed. A complete
mapping with exactly one occurrence of every port is required; wrong length,
out-of-range indices, and duplicates return structured errors. This is a
provisional additive 0.x operation, so its name and signature are not a 1.0
stability promise.

The Touchstone writer remains a separate format boundary. It accepts only one
finite, strictly positive, real reference scalar shared by every frequency and
port. Touchstone v1 ingress therefore starts with a common scalar, as in the
example above; a network with per-port or complex references must be explicitly
renormalized to a writer-compatible common reference before export. Port
permutation itself does not relax or silently satisfy that writer contract.

Run the complete deterministic workflow example with:

```text
cargo run -p rfkit-touchstone --example permute_ports_touchstone
```

## Convert pair-adjacent ports to mixed mode

The provisional power-wave mixed-mode slice adds two borrowing, owned-result
methods:

```rust
let mixed = pair_ordered.to_mixed_mode_equal_pair_power(2)?;
let single_ended = mixed.to_single_ended_equal_pair_power(2)?;
```

`pair_count = p` selects adjacent positive/negative pairs `(0,1)`, `(2,3)`,
and so on; the first member is positive. The forward output layout is
`[d0..d(p-1), c0..c(p-1), unpaired...]`, and the inverse requires that same
declared layout. The transform is the real orthogonal `U` with rows
`(u-v)/sqrt(2)` and `(u+v)/sqrt(2)`, so `S_mm = U S_se U^T` and
`S_se = U^T S_mm U`. Use `permute_ports` first when the physical measurement
order is different or a pair polarity must be reversed; no pair map is inferred.

For each pair the two single-ended references must be exactly equal finite
complex values with a non-zero real part. The natural modal references are
`zd = 2*z` and `zc = z/2`; references may differ between pairs and frequency
samples. Unpaired references are copied unchanged, including complex or
negative-real values under the existing `abs(Re(z0))` power-wave extension.
The methods preserve opaque frequency labels and do not sort, interpolate,
renormalize, or invert `S`; singular S is valid. Invalid shapes, pair counts,
non-finite values, zero-real references, unequal pair references, and
unrepresentable doubling/halving are reported through the structured core
error boundary.

`Network` carries no mode metadata. After conversion, its ordinary `s()` and
`z0()` accessors describe the caller-declared modal coordinates; a later
inverse call must supply the matching `pair_count`. Touchstone remains a
single-ended v1.0 format boundary: restore single-ended coordinates and a
common positive-real reference before using the existing writer. The complete
ingress → physical permutation → mixed-mode analysis → inverse → writer/read
workflow is executable with:

```text
cargo run -p rfkit-touchstone --example mixed_mode_touchstone
```

This additive API is a Yellow, reversible 0.x decision. Alternatives such as
arbitrary pair-map types, a new mode-aware `Network`, generalized unequal-pair
reference transforms, private-only math, or fixed 50-ohm formulas were
considered. The selected bounded contract makes pair order, polarity,
reference relationship, and coordinate interpretation visible at the call
site while retaining the owned canonical model and a cheap inverse. Removing
the two methods, tests, fixtures, and documentation during 0.x requires no
stored-data migration; a future mode-aware type can adapt the documented
layout. The oracle uses independently authored five-port forward and inverse
fixtures generated through pinned public scikit-rf 2.0.1 APIs, not copied
instrument material or a broad scikit-rf compatibility claim.

## Apply a finite physical load to one port

`Network::terminate_port_impedance_power` applies one finite complex physical
load impedance in ohms per source-frequency sample and removes the selected
port:

```rust
use num_complex::Complex64;
use rfkit_touchstone::parse_touchstone_v1_0_s;

fn load_port() -> rfkit_touchstone::Result<rfkit_core::Network> {
    let source = parse_touchstone_v1_0_s(
        "# Hz S RI R 50\n\
         1000000000 0.11 0.01 0.12 0.02 0.13 0.03\n\
         0.21 0.01 0.22 0.02 0.23 0.03\n\
         0.31 0.01 0.32 0.02 0.33 0.03\n",
        3,
    )?;
    let load_ohm = [Complex64::new(0.0, 0.0)]; // ideal short at the sample
    Ok(source.terminate_port_impedance_power(1, &load_ohm)?)
}
```

The method uses the source network's Kurokawa power-wave boundary directly:
`V_k=-Z_L I_k`, `den=(Z_L+conj(z_k))-(Z_L-z_k)*S_kk`, and
`S_out=S_EE+S_Ek*((Z_L-z_k)/den)*S_kE`. It does not model an independent
load excitation, invert S/Z/Y, renormalize, choose a frequency grid, or hide a
matched-connection policy. The selected port must be valid in a source with at
least two ports, and `load_ohm.len()` must equal the source frequency count.
Finite complex source references with nonzero real parts are supported,
including the repository's negative-real extension. Loads may be positive,
zero, or negative resistance; an ideal short is valid, while non-finite open
sentinels are outside this finite-only API. Exact zero `den` is rejected, but
finite nonzero near-singular values are not rejected by an arbitrary cutoff;
`d=Z_L+conj(z_k)==0` remains valid when `den` is nonzero.

The returned frequency axis, survivor S coordinates, and survivor references
retain source order exactly. The writer remains its own format boundary: a
terminated network can be written directly only when the survivors still have
the writer's common finite positive-real reference contract. The executable
Touchstone ingress → termination → independently checked reduction →
writer/read workflow is:

```text
cargo run -p rfkit-touchstone --example terminate_port_touchstone
```

This is a provisional additive Yellow decision for `0.x`, with no general
scikit-rf compatibility promise. A canonical differential fixture uses pinned
scikit-rf `2.0.1` (commit
`bd651e923cac6020de49a096e1d7e9b5f949f884`), NumPy `2.5.1`, seed `20260950`,
selected port `2`, loads `[0, 38+12j, 73-9j]` ohm, and survivor order
`[0,1,3,4]`. Its expected S is obtained through public `z2s`/`Network` load
construction followed by public `connect`; only floating S output uses the
recorded strict tolerance.

## Inspect sampled two-port power-wave stability metrics

`Network::two_port_stability_power` computes one owned
`TwoPortStability { delta, rollet_k }` record for every source-frequency
sample, preserving the source order and cardinality:

```rust
use rfkit_touchstone::parse_touchstone_v1_0_s;

fn inspect() -> rfkit_touchstone::Result<()> {
    let network = parse_touchstone_v1_0_s(
        "# Hz S RI R 50\n\
         1000000000 0.0 0.0 2.0 0.0 0.1 0.0 0.0 0.0\n",
        2,
    )?;
    let metrics = network.two_port_stability_power()?;
    assert!((metrics[0].delta.re + 0.2).abs() < 1.0e-14);
    assert!((metrics[0].rollet_k.unwrap() - 2.6).abs() < 1.0e-14);
    Ok(())
}
```

The operation uses the stored Kurokawa power-wave S data directly:
`delta=S11*S22-S12*S21` and
`K=(1-|S11|²-|S22|²+|delta|²)/(2|S12||S21|)`. It accepts only finite
two-port S/z0 data with strictly positive reference real parts, including
unequal complex and frequency-dependent references. Negative, duplicate,
descending, and signed-zero frequency labels remain opaque pointwise samples;
no sorting, conversion, interpolation, renormalization, tolerance cutoff, or
50-ohm default is applied. An exactly zero `S12` or `S21` gives a finite
`delta` and `rollet_k=None`; this is an undefined metric, not infinity, a
verdict, or a numerical-failure catch-all. A finite nonzero transmission whose
magnitude product underflows or overflows returns a structured arithmetic
error.

The familiar linear two-port interpretation requires `K > 1` *and*
`|delta| < 1`, with the usual auxiliary/proviso conditions; K alone is not a
stability verdict. Sampled external S data cannot certify internal poles,
unsampled frequencies, nonlinear or large-signal behavior, or overall circuit
stability. The focused Touchstone ingress workflow is executable with:

```text
cargo run -p rfkit-touchstone --example two_port_stability_touchstone
```

This is a provisional additive Yellow 0.x API. The selected small result type
and explicit `Option` semantics keep undefined samples inspectable without
introducing poles, circles, μ factors, gain optimization, tolerance-based
classifiers, or a broad analysis hierarchy. Removing the method, tests,
fixture, and documentation during 0.x requires no data migration.

## Connect two networks at a direct physical junction

`Network::connect_direct_power` joins one port from each network with the
physical conditions `V_A=V_B` and `I_A+I_B=0` (currents into both networks):

```rust
let connected = a.connect_direct_power(1, &b, 2)?;
```

The result keeps A's unconnected ports in original order, followed by B's
unconnected ports in original order. The method solves the two-coordinate
junction directly under the repository's Kurokawa power-wave equations. It
accepts finite complex references with nonzero real parts, including unequal,
frequency-dependent and negative-real references; the sign of `Re(z)` is
preserved in the wave equations. It does not insert a mismatch network,
convert through S/Z/Y, renormalize implicitly, or choose a frequency grid.

Both inputs must have matching finite frequency axes with exact pointwise
labels, valid selected ports, finite square S data and finite references, and
at least one surviving port. Frequencies are copied from A without sorting,
intersection, interpolation, or broadcasting. Exact singularity of the
two-coordinate junction system is reported; finite nonsingular near-singular
systems remain valid without an arbitrary condition cutoff, while non-finite
arithmetic is an explicit error. Inputs and surviving references are unchanged
and copied exactly.

The end-to-end Touchstone workflow uses two separate v1.0 S/RI/Hz inputs with
different common positive-real references, connects them without
pre-renormalizing either source, and checks the result from the independent
physical V/I boundary. It then explicitly calls `renormalize_direct_power` to
a caller-chosen common writer reference before export:

```rust
use ndarray::Array2;
use num_complex::Complex64;
use rfkit_touchstone::{parse_touchstone_v1_0_s, write_touchstone_v1_0_s_ri_hz};

fn direct_touchstone_workflow(
    a_text: &str,
    b_text: &str,
) -> rfkit_touchstone::Result<String> {
    let a = parse_touchstone_v1_0_s(a_text, 3)?; // e.g. R 50
    let b = parse_touchstone_v1_0_s(b_text, 4)?; // e.g. R 75
    let joined = a.connect_direct_power(1, &b, 2)?;
    let common = Array2::from_elem(
        joined.z0().dim(),
        Complex64::new(60.0, 0.0),
    );
    let writer_ready = joined.renormalize_direct_power(common)?;
    write_touchstone_v1_0_s_ri_hz(&writer_ready)
}
```

The Touchstone writer retains its own v1.0 contract: one finite common
positive-real reference, finite S values, and a finite non-negative strictly
increasing frequency axis. It never repairs or silently renormalizes a direct
connection. This additive Yellow API has no broad scikit-rf compatibility or
`1.0` stability promise; alternatives such as widening the matched method,
mandatory renormalization, an inserted mismatch Network, or a broad topology
type were rejected. Rollback removes the method, tests, fixture, and docs with
no persisted-data migration. Run the focused executable workflow with:

```text
cargo run -p rfkit-touchstone --example connect_direct_power_touchstone
```

## Close two ports of one network at a direct physical junction

`Network::inner_connect_direct_power` closes two distinct ports of one
network with the physical conditions `V_a = V_b` and `I_a + I_b = 0` (both
currents point into the source network):

```rust
let reduced = direct.inner_connect_direct_power(1, 3)?;
```

The operation is the same explicit Kurokawa power-wave boundary used by
`connect_direct_power`, but its internal scattering block is the full
two-by-two block for the selected coordinates:

```text
C = [[ q_a,             q_b            ],
     [ q_a*conj(z_a),  -q_b*conj(z_b)  ]]
D = [[-q_a,            -q_b           ],
     [ q_a*z_a,         -q_b*z_b       ]]

(C + D*S_ii) T = -D*S_ie
S_out = S_ee + S_ei*T
```

Both `S_ab` and `S_ba` are retained; treating the selected ports as two
independent one-port networks is not equivalent for a coupled multiport.
Each reference must be finite with a nonzero real part. Unequal or equal
complex, per-port, frequency-dependent, and negative-real references are
therefore in-domain, with the signed `Re(z)` retained in
`q = sqrt(abs(Re(z)))/Re(z)`. The frequency axis is copied bit-for-bit and
survivor ports remain in their original order with exact references. The
source is borrowed and unchanged.

The source must have a nonempty frequency axis, a positive square S matrix,
matching frequency/S and `(nfreq,nport)` z0 shapes, finite S/z0/frequency
values, distinct in-range selected ports, and at least one survivor. Only the
two-coordinate direct system is solved. An exactly zero evaluated pivot is a
structured singular-junction error; finite nonsingular near-singular systems
remain valid without a condition or rank cutoff. No S/Z/Y conversion,
pseudoinverse, regularization, hidden wave conversion, fixed reference, or
writer repair is introduced. This additive 0.x API is Green, reversible, and
provisional; it makes no broad scikit-rf compatibility or `1.0` promise.

The focused Touchstone workflow parses a five-port v1.0 input, explicitly
renormalizes it to unequal complex selected references, checks the full
physical V/I response independently, closes ports 1 and 3, then explicitly
renormalizes the survivors to one common positive-real writer reference and
performs a write/read round trip:

```text
cargo run -p rfkit-touchstone --example inner_connect_direct_power_touchstone
```

The pinned differential fixture is
`tools/oracle/fixtures/power_wave_inner_connect_direct_five_port_complex_z0.json`
(seed `20260952`). scikit-rf `2.0.1`'s public `innerconnect` path returns a
pseudo-wave result for this complex-reference case, so the oracle explicitly
calls `result.renormalize(result.z0, s_def="power")` before reading expected S.
The raw pseudo result is deliberately not treated as a power-wave oracle, and
this is not a general scikit-rf compatibility claim.

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
