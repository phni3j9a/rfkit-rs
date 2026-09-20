# Public `Network` API Direction

This document records the implemented baseline and autonomous design envelope for the provisional public RF-analysis surface of `rfkit-core`.

The first verified `Network` methods are complete. Their concrete semantics remain policy and compatibility evidence, but their original method list is no longer an allowlist that requires human approval before every subsequent public capability. New public work is classified under the Green, Yellow, and Red rules in `docs/LOOP_ENGINEERING.md` and the RF-specific boundaries below.

## Goals

The public surface should:

- expose already-verified RF capability instead of continuing to accumulate private kernels;
- be Rust-native rather than mimic the scikit-rf object model mechanically;
- keep numerical and wave semantics explicit at the call site;
- avoid implicit frequency-grid selection, hidden renormalization, or convenience defaults that would freeze policy accidentally;
- stay small enough to evolve before a deliberate stabilization milestone.

## Stability policy

`rfkit-core` is currently a `0.x` crate. The first public RF-analysis methods are intentionally **provisional**, not a promise that names or signatures are frozen for `1.0`.

That does not make churn free. Public changes still need concrete user or correctness value, but the project should not postpone useful public capability merely to avoid ever changing a pre-1.0 API.

Until an explicit stabilization milestone, autonomous work may add or evolve bounded provisional APIs under the autonomy classes below. A Yellow change may revise provisional behavior when the Issue and PR record compatibility impact, migration or rollback, alternatives considered, and why the result remains preferable and reversible.

Declaring stability, making a new compatibility guarantee, or breaking a guarantee already made is Red. Replacing explicit behavior with a vague or hidden default is also Red; an explicitly named additional behavior can usually be evaluated as Green or Yellow instead.

## Core model

Keep the existing owned, frequency-major core model as the current baseline:

- `Frequency` is the public frequency-axis type and uses hertz at the boundary;
- `Network` remains the canonical owned scattering-network type;
- `Network.s` is conceptually `(frequency, port_out, port_in)`;
- `Network.z0` is conceptually `(frequency, port)`;
- ports use zero-based `usize` indices initially;
- `ndarray` remains part of the initial public data boundary because it is already exposed by construction and accessors.

The Yellow parameter-ingress slice adds three associated constructors while
keeping this owned model:

```rust
impl Network {
    pub fn from_z_power(
        frequency: Frequency,
        z: Array3<Complex64>,
        z0: Array2<Complex64>,
    ) -> Result<Network>;

    pub fn from_y_via_z_power(
        frequency: Frequency,
        y: Array3<Complex64>,
        z0: Array2<Complex64>,
    ) -> Result<Network>;

    pub fn from_y_direct_power(
        frequency: Frequency,
        y: Array3<Complex64>,
        z0: Array2<Complex64>,
    ) -> Result<Network>;
}
```

`z` is a frequency-major `(nfreq, nport, nport)` stack in ohms, `y` is the
same shape in siemens, `z0` is `(nfreq, nport)` in ohms, and the returned S
stack is dimensionless. Both paths use Kurokawa power-wave semantics and
preserve the supplied frequency samples/order, port order, and references
exactly. The constructors require a nonempty frequency axis and exact
first-axis length agreement, but treat samples as pointwise labels: they do
not require finite/nonnegative/sorted/unique frequencies, and do not sort,
resample, broadcast, default to 50 ohms, regularize, use a pseudoinverse,
apply a cutoff, fall back, or take an identity shortcut. The Touchstone writer
retains its separate finite/nonnegative/strictly-increasing/common-positive-
reference contract.

`from_z_power` delegates to the existing `Z→S` power-wave kernel. The
explicitly named `from_y_via_z_power` delegates to the existing composed
`Y→Z→S` path; singular or zero Y therefore fails at `Y→Z`, even though
`from_y_direct_power` can support an ideal open. The direct constructor forms
`A=F(I+GY)` and `B=F(I-conj(G)Y)` and solves
`S A=B` directly, so singular or zero Y is accepted whenever `A` is
nonsingular and its arithmetic remains finite. Finite complex references,
including per-port/frequency-dependent and negative-real values, remain within
the existing `abs(Re(z0))` domain. Zero-real and non-finite references are
rejected. Errors retain the supplied Z/Y kind, direct/composed stage, and
available row/column/pivot context through the crate-level structured
boundary. These additive entrypoints are provisional during 0.x.

The direct constructor's broader Y domain does not alter downstream conversion
domains. A network created from singular Y is not promised to succeed through
`to_z_power`, `to_y_power`, renormalization, or another operation that requires
an invertible intermediate parameter matrix; callers should select the direct
entrypoint intentionally and handle those existing restrictions.

Do **not** introduce view lifetimes, generic storage traits, port-index newtypes, builders, parameter-container hierarchies, or alternate owned network representations merely to make the API look more abstract. A concrete supporting type or bounded internal architecture change may be Yellow when demonstrated usage justifies it. Replacing the canonical model or storage representation in a difficult-to-reverse way is Red.

## Operation style

Prefer methods on `Network` when one network is the natural primary object. Operations should be pure from the caller's perspective: borrow inputs and return owned results rather than mutating a network in place.

The public API should not expose a parallel free-function surface that duplicates the same operations without a demonstrated reason.

Use `Frequency` rather than raw frequency slices at public operation boundaries.

## Wave semantics

The current verified conversion, renormalization, and matched-connection kernels use Kurokawa power-wave semantics. The `Network` model does not yet carry a wave-definition field.

Keep that fact explicit in **wave-sensitive public method names** rather than adding a speculative `WaveDefinition` field before a second convention is actually supported.

An additional convention with authoritative mathematics, explicit naming, and conformance evidence may proceed as Yellow. Selecting a broad implicit wave default, or resolving an authoritative disagreement that cannot be represented through explicit side-by-side APIs, is Red.

## Direct S→Y extraction (Issue #78 Yellow decision)

Issue #78 selects one additive, explicitly named method:

```rust
impl Network {
    pub fn to_y_direct_power(&self) -> Result<Array3<Complex64>>;
}
```

The method returns an owned, frequency-major Y array in siemens and borrows the
network without mutation. It uses the stored `(frequency, port)` reference
impedances in ohms, preserves frequency and port order, and does not sort,
resample, broadcast, renormalize, or choose a default reference. It is a
provisional 0.x API: this record makes no stability or general scikit-rf
compatibility promise.

The direct equation follows the repository's Kurokawa power-wave definitions
with currents into the ports:

```text
a = F(V + G I),  b = F(V - conj(G) I),  b = S a,  I = Y V
G = diag(z0),    F = diag(1 / (2 sqrt(abs(Re(z0)))))
A = (S G + conj(G)) F,    B = (I_n - S) F,    A Y = B
```

The implementation solves this left system with the existing multiple-RHS
solver. A singular `I_n-S` is therefore not itself an error. An exact singular
`A`, non-finite input/reference, or non-finite arithmetic is an error with the
direct `S→Y` stage and available frequency/port/row/column/pivot context. The
operation requires a nonempty frequency axis, square positive-port S data, and
matching `(nfreq, nport)` references; malformed serde-created shapes must be
reported rather than panic. Frequency samples remain pointwise labels and need
not be finite, non-negative, sorted, or unique. Finite complex references,
including per-port/frequency-dependent and negative-real values, remain in the
existing `abs(Re(z0))` domain; zero-real and non-finite references are rejected.
There is no explicit inverse, pseudoinverse, rank or condition cutoff,
regularization, eigenvalue nudge, clipping, identity shortcut, or fallback.
Ideal-open `S=I` produces zero Y through the ordinary validation and solve;
floating-series models are supported when `A` is nonsingular; a real-reference
ideal short `S=-I` remains an exact singular direct system.

This is explicit coexistence with the existing `to_y_power`, not a replacement
or hidden fallback. `to_y_power` continues to mean the composed `S→Z→Y`
operation, retaining its existing name, behavior, and structured `S→Z` or
`Z→Y` stage errors. The direct method does not broaden `to_z_power`,
`to_y_power`, renormalization, or any other downstream operation that requires
an invertible intermediate matrix.

Alternatives considered were replacing `to_y_power` internally, renaming or
removing it, silently falling back to direct conversion on singularity, asking
callers to reconstruct Y, or extending renormalization and other conversions in
the same change. The selected additive method makes the conversion domain
visible at the call site and keeps existing callers and failure semantics
unchanged. During the provisional 0.x phase, rollback is reversible: remove
the additive method/kernel, its tests and documentation, with no storage or
data migration. No wave-definition field, generic parameter hierarchy, or
broader API redesign is implied.

## Direct power-wave renormalization (Issue #80 Yellow decision)

Issue #80 adds one additive, explicitly named method:

```rust
impl Network {
    pub fn renormalize_direct_power(
        &self,
        new_z0: Array2<Complex64>,
    ) -> Result<Network>;
}
```

The method directly changes the Kurokawa power-wave reference from the
network's stored source `G=diag(z_old)` to the caller-supplied target
`H=diag(z_new)`. It returns an owned frequency-major `Network`, preserving
the source frequency and port order exactly and retaining an exact owned copy
of `new_z0`. Neither the source network nor the target array is mutated.
References may be finite complex, per-port, frequency-dependent, or have
negative real parts under the existing `abs(Re(z0))` normalization. Zero-real
and non-finite references are rejected.

With `F` and `F_new` the source and target diagonal normalization matrices,
the direct equations are:

```text
K = F_new F^-1 (2 Re(G))^-1
D = K (conj(G) + H)       E = K (G - H)
C = K (conj(G) - conj(H)) J = K (G + conj(H))
S_new (D + E S) = C + J S
```

The implementation solves the right system by plain-transposing it into the
existing multiple-right-hand-side solver. `2 Re(G)` retains its signed real
value for mixed-sign references; only the wave normalization uses
`abs(Re(z0))`. Exact singular pivots and non-finite arithmetic are reported,
while finite near-singular systems remain in-domain. A singular `I-S` or
`I+S` alone is not a failure. There is no Z/Y intermediate, dense explicit
inverse, pseudoinverse, regularization, rank cutoff, identity shortcut,
fallback, grid selection, broadcasting, or implicit target reference.

Malformed serde-created Networks are validated without panicking: the
frequency axis must be nonempty and match the S first axis, S must be square
with a positive port count, and both source and target references must have
shape `(nfreq,nport)`. Frequency values remain pointwise labels and need not
be finite, sorted, unique, or ordered. Direct failures use
operation-specific structured errors, with source/target attribution for
reference failures and frequency/port/row/column/pivot context where
available; they never report a fictitious S→Z or Z→S stage.

This method intentionally coexists with `renormalize_power`. The latter
remains the existing composed S→Z→S operation, including equal-reference
validation and its source/target conversion-stage errors. The direct method
is the selected additive behavior because it removes the singular Z/Y barrier
for explicit workflows such as direct singular-Y ingress, direct wave-change
to one common positive-real reference, Touchstone writer/read, and direct Y
extraction. Replacing the existing method, silently falling back after a
composed failure, routing through direct Y and Y→S, or requiring callers to
reconstruct the wave change were rejected. The compatibility impact is
additive/provisional during 0.x; rollback removes the method/kernel,
diagnostics, tests, and documentation with no storage or data migration.
No broader wave convention, dependency, crate, default-reference, or API
architecture change is implied.

## Explicit port permutation (Issue #82 Green decision)

The provisional public surface adds one owned, borrowing transformation:

```rust
impl Network {
    pub fn permute_ports(&self, order: &[usize]) -> Result<Network>;
}
```

`order` is a complete zero-based new-to-old mapping. If there are `nport`
ports, it must contain each index in `0..nport` exactly once. Thus
`[2, 0, 1]` puts old port 2 at new port 0, old port 0 at new port 1, and old
port 1 at new port 2. For every frequency, the operation applies the same
coordinate relabeling to both S axes and the reference vector:

```text
out.s[f, i, j] = input.s[f, order[i], order[j]]
out.z0[f, i]   = input.z0[f, order[i]]
```

Equivalently, for the scattering relation `b = S a`, it copies the same
incident/reflected coordinate permutation on both sides, `S' = P S Pᵀ` and
`z0' = P z0`. This is pure data reindexing: it does not perform wave
arithmetic, select a wave convention, renormalize, interpolate, change units,
choose a default reference, or apply finite-value/writer-domain checks. The
frequency samples/order, scalar components, and port count are copied exactly;
the input network and `order` slice remain unchanged, and the result is a new
owned `Network`.

The public error boundary rejects malformed source shapes before indexing and
reports an incomplete mapping, an out-of-range port (with its position), or a
duplicate port with structured context. This includes identity mappings and
serde-created malformed `Network` values; no validation bypass is available for
the identity case. Downstream operations retain their own domains, so a
permuted network still has to satisfy the Touchstone writer's nonempty,
finite/nonnegative/strictly increasing frequency and common finite positive-real
reference contract when it is exported.

This additive operation is provisional during the `0.x` phase. It makes no
general scikit-rf API or `1.0` stability promise; rollback is limited to the
method, its diagnostics, tests, and documentation, with no data migration. The
implementation is an independent rewrite from the indexing equations rather
than a port-name or in-place renumbering API. No new wave definition,
parameter-container hierarchy, dependency, or storage representation is
implied.

## Implemented public baseline

The implemented shape is:

```rust
impl Network {
    pub fn to_z_power(&self) -> Result<Array3<Complex64>>;
    pub fn to_y_power(&self) -> Result<Array3<Complex64>>;
    pub fn to_y_direct_power(&self) -> Result<Array3<Complex64>>;

    pub fn renormalize_power(
        &self,
        new_z0: Array2<Complex64>,
    ) -> Result<Network>;

    pub fn renormalize_direct_power(
        &self,
        new_z0: Array2<Complex64>,
    ) -> Result<Network>;

    pub fn permute_ports(&self, order: &[usize]) -> Result<Network>;

    pub fn interpolate_cartesian_linear(
        &self,
        target: &Frequency,
    ) -> Result<Network>;

    pub fn connect_matched_power(
        &self,
        port: usize,
        other: &Network,
        other_port: usize,
    ) -> Result<Network>;

    pub fn connect_matched_power_on_grid(
        &self,
        port: usize,
        other: &Network,
        other_port: usize,
        target: &Frequency,
    ) -> Result<Network>;

    pub fn inner_connect_matched_power(
        &self,
        port_a: usize,
        port_b: usize,
    ) -> Result<Network>;
}
```

The exact internal delegation remains an implementation detail. The semantic distinctions represented by these names remain policy and evidence for future API consistency:

- `to_z_power`, `to_y_power`, and `to_y_direct_power` explicitly select the verified power-wave conversion convention; the latter names the direct S→Y equation while `to_y_power` retains composed S→Z→Y semantics;
- `renormalize_power` explicitly selects power-wave renormalization;
- `renormalize_direct_power` explicitly selects the direct Kurokawa wave-change equation, while `renormalize_power` retains its composed S→Z→S domain and diagnostics;
- `permute_ports` explicitly selects a complete new-to-old physical-port reindexing and carries S rows, S columns, and z0 together without wave arithmetic;
- `interpolate_cartesian_linear` does not establish a vague interpolation default that would later need reinterpretation;
- `connect_matched_power` requires the existing exactly matched real-positive junction contract and exact compatible frequency grids;
- `connect_matched_power_on_grid` requires an explicit caller-provided grid and performs interpolation-before-connection under the existing verified composition semantics;
- `inner_connect_matched_power` exposes the existing matched same-network elimination semantics.

Do not shorten these to broad names such as `connect`, `interpolate`, or `renormalize` until the library has enough supported semantics and evidence to justify what those names mean. Introducing such a default is at least Yellow and becomes Red when reasonable conventions conflict or the choice would freeze hidden policy.

## Frequency policy

Public connection APIs must not silently choose a frequency grid.

- `connect_matched_power` requires the inputs to satisfy the existing exact-grid contract.
- `connect_matched_power_on_grid` uses only the caller-provided `Frequency` target.
- No implicit intersection, subset selection, nearest-grid match, extrapolation, or automatic interpolation is part of the first public API.

A future convenience API may proceed as Green or Yellow when its frequency-selection policy is explicit in its name or required arguments and is independently verified. An unqualified API that silently chooses intersection, extrapolation, sorting, tolerance, or another grid policy is outside the envelope.

## Parameter-return policy

`Network` remains S-parameter based. Z and Y are currently alternative parameter matrices produced from a network rather than separate public `ZNetwork` / `YNetwork` types.

Therefore `to_z_power`, `to_y_power`, and `to_y_direct_power` return owned matrices. Do not create a parameter-type hierarchy solely to wrap those matrices before real usage requires it. A later typed representation may be Yellow when multiple concrete workflows demonstrate that it improves correctness or usability without replacing the canonical model implicitly.

Transformations that still produce an S-parameter network, such as renormalization, interpolation, and connection, return a new `Network`.

## Error policy

Use one crate-level public `Error` / `Result<T>` boundary as the current default.

The public error must remain structured and actionable rather than collapsing failures to strings. Preserve stable context such as invalid shapes/frequencies/ports, invalid or mismatched junction reference impedances, singularity, non-finite inputs/computation, and the failing high-level operation stage where it materially helps callers.

Internal kernel error enums may remain private and be translated at the public boundary. Mark the public error non-exhaustive before expanding it for the new operations so adding future error cases does not unnecessarily freeze the variant set.

Do not expose private implementation types merely to avoid writing a public error mapping.

## Autonomous extension envelope

The Planner may dispatch a public API increment as **Green** when all of these are true:

- it is a bounded, useful vertical slice within the current development horizon;
- RF mathematics, applicable specifications, and reference behavior do not present a material unresolved semantic conflict;
- wave convention, units, frequency behavior, tolerances, and other consequential choices are explicit;
- it follows the current core model, operation style, and error boundary;
- it is additive or a correctness-preserving implementation change and carries proportionate deterministic, invariant, and differential evidence;
- no new compatibility promise, irreversible action, or uncertain provenance is involved.

The Planner may dispatch a public API increment as **Yellow** when a material choice remains but all of these are true:

- the choice is bounded and reversible during the provisional `0.x` phase;
- the Issue records the plausible alternatives, selection criteria, compatibility impact, and intended rollback or migration;
- the selected API names its semantics rather than hiding policy in a convenience default;
- the PR preserves that decision record and the independent reviewer explicitly evaluates the RF/API choice, evidence, and reversibility;
- the change does not cross a Red boundary.

Examples of Yellow work include an explicitly named additional wave convention, a justified supporting public type, a bounded dependency or crate-boundary adjustment, and a provisional API revision with a documented migration. Yellow is not a license for speculative abstraction or weakly sourced RF behavior.

The following remain **Red** and require human approval:

- API stabilization, release policy, or a compatibility guarantee;
- an implicit broad default when multiple reasonable wave, frequency, interpolation, or connection semantics remain;
- externally visible RF behavior where authoritative sources materially disagree and explicit APIs cannot preserve the alternatives;
- a difficult-to-reverse replacement of the canonical model, storage representation, or repository architecture;
- unresolved provenance/licensing obligations or unavailable evidence required for correctness;
- release, package publication, signing, credentials, or another irreversible external action.

Implementation remains incremental. A Red question blocks only the affected Issue; it does not turn this baseline into a repository-wide stop when independent Green or Yellow work remains.
