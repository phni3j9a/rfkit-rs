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

## Equal-pair mixed-mode power waves (Issue #84 Yellow decision)

The provisional public surface adds one reversible forward/inverse slice:

```rust
impl Network {
    pub fn to_mixed_mode_equal_pair_power(&self, pair_count: usize) -> Result<Network>;
    pub fn to_single_ended_equal_pair_power(&self, pair_count: usize) -> Result<Network>;
}
```

These methods use the existing Kurokawa power-wave convention with currents
into the network. For an ordered positive/negative pair `(u,v)`,
`Vd=Vu-Vv`, `Id=(Iu-Iv)/2`, `Vc=(Vu+Vv)/2`, and `Ic=Iu+Iv`. If both
single-ended references equal `z`, the natural modal references are `zd=2z`
and `zc=z/2`. Substitution into the repository wave equations gives the same
normalized wave transform for both incident and reflected waves:

```text
Ud = (u-v)/sqrt(2)       Uc = (u+v)/sqrt(2)
S_mm = U S_se U^T        S_se = U^T S_mm U
```

`pair_count = p` requires `1 <= p <= nports/2` and selects adjacent pairs
`(0,1), (2,3), ...`, where the first member is positive. The forward output
coordinate order is `[d0..d(p-1), c0..c(p-1), unpaired...]`; the inverse
requires that declared modal order and returns adjacent single-ended pairs.
Use `permute_ports` explicitly for any other physical pairing or polarity; the
methods infer no map. The transform is N-port and permits asymmetric,
non-reciprocal, mode-converting, and singular S data.

References are frequency-major `(nfreq,nport)` data. Each selected pair must
have exactly equal finite complex references with non-zero real parts at every
frequency; equality is not tolerance-based. References may differ between
pairs and frequency samples. Unpaired references are copied unchanged,
including complex and negative-real values under the repository's
`abs(Re(z0))` normalization. The modal values are exactly `2*z` and `z/2`;
if finite binary64 arithmetic would lose information (including an
unrepresentable subnormal half), the operation rejects the reference rather
than inventing a discrepancy or silently clipping it. All derived references
must remain finite with non-zero real parts.

Both methods validate nonempty frequency/S/z0 axes and finite S/references
before indexing, including malformed serde-created `Network` values. Frequency
samples are opaque pointwise labels and are copied exactly; no sorting,
interpolation, hidden renormalization, inversion of S, pseudoinverse,
conditioning cutoff, or fallback is performed. The source network is borrowed
and remains unchanged. Rustdoc/examples should treat the returned `Network`
as the declared coordinate system: it stores no mode metadata, so ordinary
`s()` and `z0()` accessors describe differential/common/unpaired coordinates
only by the caller's explicit contract. Touchstone v1.0 remains a single-ended
format boundary; restore single-ended coordinates and a common finite
positive-real reference before using the existing writer. No mixed-mode
Touchstone extension or automatic writer renormalization is introduced.

The Yellow alternatives considered were arbitrary pair-map/supporting types, a
new mode-aware canonical `Network`, generalized unequal/complex pair
transforms with caller-selected modal references, exposing only private math,
and fixed 50-ohm/4-port formulas. The selected option makes the useful N-port
capability available now, keeps pair order/polarity/reference assumptions
inspectable, reuses explicit permutation, retains the owned core model, and
keeps the transform/inverse cheap to review and verify. It is additive and
provisional during 0.x: rollback removes only these methods, diagnostics,
tests, fixtures, and documentation, with no stored-data migration. A future
mode-aware type can adapt the documented layout, but this issue makes no such
architecture commitment and no broad scikit-rf compatibility promise.

Conformance uses independently authored asymmetric five-port, three-frequency
forward and inverse fixtures with `p=2`, two distinct complex equal pair
references per frequency, and one unpaired complex reference. The forward
fixture calls pinned public `Network.se2gmm`; the inverse input is generated
independently and calls pinned public `Network.gmm2se` with an explicit
adjacent target `z0_se`. Pinned lineage is scikit-rf `2.0.1`, commit
`bd651e923cac6020de49a096e1d7e9b5f949f884`, NumPy `2.5.1`; only floating S
outputs use the recorded strict binary64 tolerance, while metadata, shapes,
frequencies, inputs, and references are exact contract fields. The Rust
implementation is a REWRITE from the coordinate equations.

## Finite physical-load port termination (Issue #86 Yellow decision)

The provisional public surface adds one borrowing, owned-result operation:

```rust
impl Network {
    pub fn terminate_port_impedance_power(
        &self,
        port: usize,
        load_ohm: &[Complex64],
    ) -> Result<Network>;
}
```

The method applies one finite complex physical load impedance in ohms at each
source-frequency sample, removes the selected zero-based port, and returns the
surviving ports in their original order. It uses currents into the source
network, `b = S a`, and the existing Kurokawa power-wave convention. No load
excitation is modeled. The source network and the borrowed load slice are
unchanged; frequency labels and the surviving `(frequency, port)` references
are copied exactly.

For selected port `k`, source reference `z_k`, physical load `Z_L`, and source
survivors `E`, the boundary is `V_k = -Z_L I_k`. Define
`c = Z_L - z_k` and `d = Z_L + conj(z_k)` only as algebraic names. The direct
elimination is evaluated as:

```text
den = d - c*S[k,k]
S_out = S[E,E] + S[E,k] * (c/den) * S[k,E]
```

The implementation must not require `d != 0` or form `c/d` as an obligatory
intermediate: a finite load with `d == 0` is valid whenever the evaluated
`den` is nonzero. An exact complex-zero `den` is a structured singular
termination error. A finite nonzero near-singular denominator remains in the
domain; there is no arbitrary cutoff, pseudoinverse, regularization, hidden
renormalization, S/Z/Y intermediate, or fallback. Scale-safe complex division
and finite-arithmetic checks are used so reported errors are explicit rather
than silent non-finite output.

The source must have at least two ports and a nonempty, matching frequency/S
axis. `port` must be valid and the load slice length must equal the number of
source-frequency samples exactly; no scalar broadcasting, sorting,
interpolation, extrapolation, or inferred grid is performed. S and source
references must be finite. References are finite complex values with nonzero
real parts under the existing `abs(Re(z0))` power-wave domain, including the
repository's negative-real extension. Loads may have positive, zero, or
negative real parts; an ideal short (`Z_L = 0`) is supported. Infinity/NaN
open sentinels are rejected, and an open-termination enum or generic
multiport-load contract is intentionally outside this finite-only slice.

This method is additive and provisional during `0.x`; it makes no general
scikit-rf compatibility or `1.0` stability promise. The Yellow alternatives
were a feedback-coefficient argument, a one-port `Network` argument, an
impedance/admittance/open/short union, generic multiport termination, or
composition only through renormalization and matched connection. The selected
physical finite-impedance boundary keeps units, wave direction, frequency
cardinality, and survivor order visible at the call site while closing the
loaded-response workflow. During `0.x`, rollback removes the method, local
diagnostics, tests, fixture, and documentation without persisted-data
migration; a future explicit load type could adapt finite values later. No
new wave field, generic parameter hierarchy, dependency, crate boundary, or
implicit writer renormalization is implied.

Conformance uses one canonical asymmetric five-port, three-frequency fixture
with selected middle port `2`, frequency-dependent unequal complex
positive-real source references, and finite loads `[0, 38+12j, 73-9j]` ohm
(including the ideal short). Expected S comes from pinned public scikit-rf
`skrf.network.connect(source, 2, one_port_load, 0)`, where the one-port load is
constructed through public `skrf.network.z2s(..., s_def="power")` and
`Network` APIs. The pinned lineage is scikit-rf `2.0.1`, commit
`bd651e923cac6020de49a096e1d7e9b5f949f884`, NumPy `2.5.1`, seed `20260950`;
loads and references are in ohms and survivors are `[0,1,3,4]`. Only S
output uses the fixture's strict `rtol=1e-12`, `atol=1e-12` policy; metadata,
inputs, frequency labels, loads, references, and ordering are exact contract
fields. The Rust operation is a REWRITE from the physical boundary equations.

## Sampled two-port power-wave stability metrics (Issue #92 Yellow decision)

The provisional public surface adds one borrowing, owned-result operation:

```rust
impl Network {
    pub fn two_port_stability_power(&self) -> Result<Vec<TwoPortStability>>;
}

pub struct TwoPortStability {
    pub delta: Complex64,
    pub rollet_k: Option<f64>,
}
```

For every source-frequency sample, the method computes directly from the
stored Kurokawa power-wave S matrix:

```text
delta = S11*S22 - S12*S21
K     = (1 - |S11|² - |S22|² + |delta|²) / (2*|S12|*|S21|)
```

The returned vector has exactly the source cardinality and order. `delta` and
K are dimensionless. If either `S12` or `S21` is exactly complex zero, delta is
still checked and returned while `rollet_k` is `None`; `None` is an explicit
undefined denominator state, never a numerical-failure catch-all, infinity,
or stability verdict. Nonzero transmission coefficients are not classified by
a tolerance. A magnitude-product underflow/overflow or any non-finite
intermediate/output returns the operation-specific structured arithmetic
error instead.

The method validates malformed serde-created values before indexing: the
frequency axis is nonempty and finite, S has exact shape `(nfreq,2,2)`, z0
has exact shape `(nfreq,2)`, all S/z0 components are finite, and both
references have strictly positive real parts. Unequal, complex, per-port, and
frequency-dependent positive-real references are supported. Frequency labels
are opaque pointwise values, so negative, duplicate, descending, and signed
zero values are retained. The source is borrowed and unchanged; no S/Z/Y
conversion, renormalization, interpolation, sorting, clipping, default
reference, pole search, or condition cutoff is introduced.

The usual linear two-port interpretation requires `K > 1` and `|delta| < 1`
together with the familiar auxiliary/proviso conditions. This method reports
sampled metrics only: external sampled S cannot certify internal poles,
unsampled frequencies, nonlinear/large-signal behavior, or overall circuit
stability. It deliberately does not return verdict booleans, stability
circles, μ factors, gain optimization, or tolerance classifications.

This additive API is provisional during 0.x. Alternatives considered were
returning K alone with scikit-rf-style infinity, returning an error for every
zero transmission sample, and adding a broad stability-analysis hierarchy or
signed-negative reference domain. The selected result record and explicit
`Option` preserve useful whole-sweep inspection while making undefined values
honest and the physical reference domain explicit. The implementation is a
REWRITE from the determinant/Rollett equations; pinned public scikit-rf is a
finite-domain oracle only. Rollback removes this method/type, diagnostics,
tests, fixture, and documentation without storage migration or changes to the
canonical `Network` model.

The canonical four-sample fixture
`two_port_stability_power_four_frequency.json` uses public scikit-rf
`Network.stability` for finite K and NumPy determinant for delta. Its S input
is an independent base stack plus a seeded NumPy `default_rng` complex
perturbation (seed `20260954`, scale `1e-3`), with one passive and three
active/non-passive samples established by true largest singular values; it
also uses unequal real/complex positive-real references and strict
`rtol=1e-12`, `atol=1e-12` output-only comparison. The external Touchstone workflow is
`cargo run -p rfkit-touchstone --example two_port_stability_touchstone`.

## Direct physical power-wave connection (Issue #88 Yellow decision)

The provisional public surface adds one borrowing, owned-result operation:

```rust
impl Network {
    pub fn connect_direct_power(
        &self,
        port_a: usize,
        other: &Network,
        port_b: usize,
    ) -> Result<Network>;
}
```

The method connects exactly one coordinate from `self` (network A) to one
coordinate from `other` (network B) using the physical junction conditions
`V_A = V_B` and `I_A + I_B = 0`, with currents directed into each network. It
returns A's unconnected ports in their original order followed by B's
unconnected ports in their original order. One-port inputs are allowed when at
least one external survivor remains; there is no special two-port insertion or
reordering rule.

At each frequency, use the repository's Kurokawa power-wave equations,

```text
a = (V + z I)/(2 sqrt(abs(Re(z))))
b = (V - conj(z) I)/(2 sqrt(abs(Re(z))))
q = sqrt(abs(Re(z)))/Re(z)
I = q (a-b)
V = q (conj(z) a + z b)
```

and let `i=[A.port_a,B.port_b]` and `e=[A survivors,B survivors]`. With the
corresponding block partitions of the two independent S matrices, the direct
physical elimination is:

```text
C = [[ qA,              qB             ],
     [ qA*conj(zA),    -qB*conj(zB)    ]]
D = [[-qA,             -qB            ],
     [ qA*zA,          -qB*zB         ]]
(C + D*S_ii) T = -D*S_ie
S_out = S_ee + S_ei*T
```

This is an explicit two-coordinate junction solve, not a mandatory mismatch
network, S/Z/Y conversion, whole-network inverse, pseudoinverse, regularizer,
or hidden renormalization. The sign of `Re(z)` is retained in `q`; replacing
it with `abs(Re(z))` changes the documented negative-real extension. Every
reference must be finite with nonzero real part, so unequal complex,
frequency-dependent, per-port, and negative-real references are in-domain.
The direct method does not broaden any downstream conversion or writer domain.

Both networks must have nonempty square positive-port S data, matching
frequency/S axes and exact equal frequency labels; A's axis is copied exactly.
Frequencies remain pointwise labels and must be finite for this connection
operation, but need not be sorted, unique, or nonnegative. S and references
must be finite, selected ports must be valid, and at least one survivor must
remain. There is no intersection, sorting, interpolation, broadcasting, or
automatic grid selection. Exact evaluated zero in the two-coordinate junction
system is a structured singular-junction error even if external coupling is
zero; finite nonsingular near-singular systems remain valid without an
arbitrary tolerance or condition cutoff. Non-finite intermediate/output
arithmetic is reported with operation/frequency/port context when available.
Inputs remain unchanged and surviving references are copied exactly.

This is one additive provisional `0.x` method. Alternatives considered were
silently widening `connect_matched_power`, requiring explicit renormalization
followed by matched connection, inserting a mismatch `Network`, introducing
a broad wave/topology type, or exposing only a private direct kernel. The
selected explicit name makes physical continuity and the reference/grid
domain visible while retaining the owned `Network` model and existing matched
operation unchanged. Rollback removes this method, diagnostics, tests, fixture,
and documentation without persisted-data migration; no new dependency, data
model, crate boundary, wave convention, or topology engine is implied. This
decision makes no broad scikit-rf compatibility or `1.0` stability promise.

Conformance uses the independently generated
`power_wave_connect_direct_three_to_four_port_complex_z0.json` fixture: an
asymmetric non-reciprocal 3-port A plus 4-port B at three frequencies, selected
A[1] and B[2], seed `20260951`, unequal complex frequency-dependent references
with positive real parts, and explicit A-then-B survivor mapping. Expected S
comes only from pinned public scikit-rf `2.0.1` at commit
`bd651e923cac6020de49a096e1d7e9b5f949f884` through one
`skrf.network.connect` call. Metadata, source inputs, frequency grid,
references, shapes, and order are exact; only S uses `rtol=1e-12`,
`atol=1e-12`. A focused local check also exercises unequal real positive
references without adding a second canonical fixture. The implementation is a
REWRITE from the Kurokawa equations plus the physical junction conditions.

The corresponding Touchstone workflow parses two separate v1.0 inputs with
different common positive-real references, calls `connect_direct_power`
without pre-renormalizing either source, checks the response through the
physical V/I boundary independently, then explicitly calls
`renormalize_direct_power` to a caller-selected common writer-compatible
reference before writing and reading. The writer does not repair or
renormalize a direct-connection result automatically.

## Direct physical power-wave inner connection (Issue #90 Green decision)

The provisional public surface adds one borrowing, owned-result operation:

```rust
impl Network {
    pub fn inner_connect_direct_power(
        &self,
        port_a: usize,
        port_b: usize,
    ) -> Result<Network>;
}
```

The method connects two distinct coordinates of one network with the
physical conditions `V_a = V_b` and `I_a + I_b = 0`, where both currents point
into the source network. It removes both selected ports and returns all
survivors in their original order. At each frequency, let `i=[port_a,port_b]`
and let `e` be those survivors. With the Kurokawa power-wave equations

```text
a = (V + z I)/(2 sqrt(abs(Re(z))))
b = (V - conj(z) I)/(2 sqrt(abs(Re(z))))
q = sqrt(abs(Re(z)))/Re(z)
I = q(a-b)
V = q(conj(z) a + z b)
```

the internal boundary is evaluated as

```text
C = [[ q_a,             q_b            ],
     [ q_a*conj(z_a),  -q_b*conj(z_b)  ]]
D = [[-q_a,            -q_b           ],
     [ q_a*z_a,         -q_b*z_b       ]]

(C + D*S_ii) T = -D*S_ie
S_out = S_ee + S_ei*T
```

`S_ii` is the complete selected 2×2 block, including both `S_ab` and
`S_ba`; dropping either off-diagonal coupling or treating the selected ports
as independent one-port networks is incorrect. The direct solve is only for
this two-coordinate system. It does not convert through S/Z/Y, invert the
whole network, divide by `z_a+z_b`, insert a mismatch network, use a
pseudoinverse or least-squares fallback, regularize, clip, nudge an
eigenvalue, or silently change the wave definition.

The input shape is checked before selected-port indexing, including
serde-created malformed values: the frequency axis must be nonempty, S must
be positive and square, the frequency/S lengths must agree, and z0 must have
exact `(nfreq,nport)` shape. Frequencies are finite pointwise labels copied
bit-for-bit, including signed zero; negative, duplicate, and descending
finite labels remain valid. Every S and z0 component must be finite. Every
reference must have a finite nonzero real part, so equal or unequal complex,
per-port, frequency-dependent, and negative-real references are in-domain.
The sign of `Re(z)` is retained in `q`. Selected ports must be distinct and
in range, and at least one survivor must remain. Survivor order, frequency,
and references are preserved exactly; the source network is unchanged.

An exactly zero evaluated pivot is a structured singular-junction error,
including when external coupling is zero. A finite nonsingular near-singular
system remains in-domain without an arbitrary condition, rank, or tolerance
cutoff. Non-finite intermediate or output arithmetic is reported with
operation, frequency, selected-port, and row/column/pivot context where
available. Existing matched, direct inter-network, explicit-grid,
termination, and writer semantics remain unchanged.

This additive operation is Green, reversible, and provisional for the `0.x`
series. It introduces no new wave convention, storage or topology model,
compatibility promise, dependency, publication boundary, or file-format
extension. Rollback removes the method, kernel, tests, fixture, and docs
without persisted-data migration. The implementation is a REWRITE from the
Kurokawa equations plus voltage continuity/current conservation; scikit-rf is
used as a behavior oracle only, and no broad scikit-rf compatibility promise
is made.

The canonical differential case is
`tools/oracle/fixtures/power_wave_inner_connect_direct_five_port_complex_z0.json`:
an independently authored asymmetric non-reciprocal five-port, three
frequency samples, selected ports 1 and 3, seed `20260952`, and unequal
complex frequency-dependent references with positive real parts. The pinned
scikit-rf `2.0.1` public `innerconnect` call deliberately exposes a raw
pseudo-wave result for complex references; expected S is obtained only after
the explicit public restoration
`result.renormalize(result.z0, s_def="power")`. The input, grid, references,
survivor mapping, wave-definition metadata, and tolerance policy are exact;
only output S uses `rtol=1e-12`, `atol=1e-12`.

The executable and external workflow test are
`crates/rfkit-touchstone/examples/inner_connect_direct_power_touchstone.rs`
and
`crates/rfkit-touchstone/tests/public_direct_inner_connection_workflow.rs`.
They parse the multiport input, explicitly direct-renormalize selected
references, independently reconstruct the full physical V/I response, close
the pair, explicitly restore one writer-compatible positive-real reference,
and write/read the result.

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

    pub fn to_mixed_mode_equal_pair_power(&self, pair_count: usize) -> Result<Network>;

    pub fn to_single_ended_equal_pair_power(&self, pair_count: usize) -> Result<Network>;

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

    pub fn connect_direct_power(
        &self,
        port_a: usize,
        other: &Network,
        port_b: usize,
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

    pub fn inner_connect_direct_power(
        &self,
        port_a: usize,
        port_b: usize,
    ) -> Result<Network>;

    pub fn terminate_port_impedance_power(
        &self,
        port: usize,
        load_ohm: &[Complex64],
    ) -> Result<Network>;
    pub fn two_port_stability_power(&self) -> Result<Vec<TwoPortStability>>;
}
```

The exact internal delegation remains an implementation detail. The semantic distinctions represented by these names remain policy and evidence for future API consistency:

- `to_z_power`, `to_y_power`, and `to_y_direct_power` explicitly select the verified power-wave conversion convention; the latter names the direct S→Y equation while `to_y_power` retains composed S→Z→Y semantics;
- `renormalize_power` explicitly selects power-wave renormalization;
- `renormalize_direct_power` explicitly selects the direct Kurokawa wave-change equation, while `renormalize_power` retains its composed S→Z→S domain and diagnostics;
- `permute_ports` explicitly selects a complete new-to-old physical-port reindexing and carries S rows, S columns, and z0 together without wave arithmetic;
- `to_mixed_mode_equal_pair_power` and `to_single_ended_equal_pair_power` explicitly select the equal single-ended-reference adjacent-pair power-wave transform and its inverse, with modal layout `[d...,c...,unpaired...]`; they do not infer physical pairing or store mode metadata;
- `interpolate_cartesian_linear` does not establish a vague interpolation default that would later need reinterpretation;
- `connect_matched_power` requires the existing exactly matched real-positive junction contract and exact compatible frequency grids;
- `connect_matched_power_on_grid` requires an explicit caller-provided grid and performs interpolation-before-connection under the existing verified composition semantics;
- `inner_connect_matched_power` exposes the existing matched same-network elimination semantics.
- `connect_direct_power` exposes one direct physical V/I junction solve for
  unequal finite nonzero-real references and retains A-then-B survivor order;
  it does not silently change the exact-grid or matched-junction contracts of
  the other connection methods.
- `inner_connect_direct_power` exposes the same direct physical V/I boundary
  for two ports of one network, retaining the full internal 2×2 S block and
  original survivor order; it does not widen the matched inner-connect
  reference contract or make a broad scikit-rf compatibility promise.
- `terminate_port_impedance_power` applies a finite physical impedance boundary directly at one
  selected port, removes that port, and retains the original survivor order and references without
  selecting a new frequency grid or renormalizing the source.

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
