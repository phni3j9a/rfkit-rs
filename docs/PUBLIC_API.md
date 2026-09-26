# Public `Network` API Direction

This document records the implemented baseline and autonomous design envelope for the provisional public RF-analysis surface of `rfkit-core`.

The first verified `Network` methods are complete. Their concrete semantics remain policy and compatibility evidence, but their original method list is no longer an allowlist that requires human approval before every subsequent public capability. New public work is classified under the Green, Yellow, and Red rules in `docs/LOOP_ENGINEERING.md` and the RF-specific boundaries below.

## Goals

The public surface should:

- expose already-verified RF capability instead of continuing to accumulate private kernels;
- be Rust-native rather than mimic the scikit-rf object model mechanically;
- keep numerical and wave semantics explicit at the call site;
- avoid implicit frequency-grid selection, hidden renormalization, or convenience defaults that would freeze policy accidentally;
- give each RF operation one public entry point per distinct semantic choice, rather than one per internal evaluation strategy;
- stay small enough to evolve before a deliberate stabilization milestone.

## Stability policy

`rfkit-core` is currently a `0.x` crate. The first public RF-analysis methods are intentionally **provisional**, not a promise that names or signatures are frozen for `1.0`.

That does not make churn free. Public changes still need concrete user or correctness value, but the project should not postpone useful public capability merely to avoid ever changing a pre-1.0 API.

Until an explicit stabilization milestone, autonomous work may add or evolve bounded provisional APIs under the autonomy classes below. A Yellow change may revise provisional behavior when the Issue and PR record compatibility impact, migration or rollback, alternatives considered, and why the result remains preferable and reversible.

Declaring stability, making a new compatibility guarantee, or breaking a guarantee already made is Red. Replacing an explicit semantic choice with a vague or hidden default is also Red; an explicitly named additional semantic choice can usually be evaluated as Green or Yellow instead.

Selecting the internal evaluation strategy for one explicitly specified operation is not a semantic default. When a verified path already covers an existing operation's domain, generalizing that operation is preferred over adding a coexisting variant. See "Semantic qualifiers and evaluation strategy" and the overlap inventory below.

## Core model

Keep the existing owned, frequency-major core model as the current baseline:

- `Frequency` is the public frequency-axis type and uses hertz at the boundary;
- `Network` remains the canonical owned scattering-network type;
- `Network.s` is conceptually `(frequency, port_out, port_in)`;
- `Network.z0` is conceptually `(frequency, port)`;
- ports use zero-based `usize` indices initially;
- `ndarray` remains part of the initial public data boundary because it is already exposed by construction and accessors.

## Touchstone 2.0 ingress boundary (Issue #102 Yellow) — Superseded by #112

This is the historical #102 Full-only decision record. Its original reader
name and rejection boundary are retained below for provenance; the current
sole reader and its migration are recorded in the Issue #112 section that
follows.

The `rfkit-touchstone` crate exposes the additive pure in-memory function
`parse_touchstone_v2_0_s_full(input: &str) -> Result<rfkit_core::Network>`.
Its name is deliberately explicit: it is a Touchstone 2.0, single-ended,
Full-matrix S subset and does not widen or auto-detect the existing v1 reader.
The file supplies the positive port count and declared frequency count through
`[Number of Ports]` and `[Number of Frequencies]`; `[Version] 2.0`, an option
line, `[Network Data]`, and `[End]` are required. Two-port files must include
`[Two-Port Data Order]` with either `21_12` or `12_21`.

RI, MA, and DB pairs, the four frequency units, complete arbitrary physical
line continuations, and a real-positive per-port `[Reference]` vector are
supported. `[Reference]` may continue over lines and overrides the option-line
`R` value; absent it, the option-line resistance is expanded over every
frequency and port. Parsed S is dimensionless, references are stored in ohms,
and frequency/S/reference coordinate order and values are preserved exactly
apart from the documented finite scalar decoding. Each frequency starts a
physical line, and the declared record count must match exactly.

The subset rejects Lower/Upper matrices, non-S parameters, mixed-mode, noise,
information blocks, unknown/other-version keywords, complex or non-positive
references, duplicate/misplaced bracket directives, malformed or non-finite
numbers, incomplete/surplus records, missing `[End]`, trailing content, and
recognized HFSS/Ansys semantic comments. Later `#` option lines follow the
Touchstone rule and are ignored after the first option line. Errors distinguish unsupported subset features,
structural/order/count failures, numerical decoding failures, and checked size
arithmetic. The parser validates actual document content before constructing
arrays from declared dimensions and performs no filename inference, I/O,
sorting, interpolation, repair, or renormalization. The existing v1 writer
still requires one common finite positive-real reference; callers must invoke
an explicit core renormalization operation before exporting v2 data through it.
The additive 0.x boundary is provisional, reversible, and makes no general
Touchstone or scikit-rf compatibility promise.

The canonical pinned parser fixture is
`tools/oracle/fixtures/touchstone_v2_0_s_full_three_port.json`, generated from
an independently authored text through public scikit-rf `2.0.1` at commit
`bd651e923cac6020de49a096e1d7e9b5f949f884` with NumPy `2.5.1`. Only complex S
uses the recorded `rtol=1e-12`, `atol=1e-12` policy; frequency, references,
text, dimensions, and metadata remain exact contract fields. The executable
workflow is `crates/rfkit-touchstone/examples/touchstone_v2_full_workflow.rs`.

The Yellow alternatives were widening or auto-detecting the v1 entrypoint,
introducing a universal options/metadata reader, preprocessing files for
callers, or implementing all v2 sections. The selected separate function
keeps version/matrix semantics visible, preserves v1 behavior, and completes
an ingress-to-analysis workflow without a core-model or writer change.

## Touchstone 2.0 S ingress generalization (Issue #112 Yellow)

This section is the current Yellow decision record for Issue #112.
The current and sole public Version 2.0 single-ended S reader is the
provisional pure in-memory function
`parse_touchstone_v2_0_s(input: &str) -> Result<rfkit_core::Network>`. The
historical `parse_touchstone_v2_0_s_full` name is removed with no deprecated
alias. The caller asks to decode Touchstone 2.0 S data; the document's
`[Matrix Format]` selects the storage representation rather than creating a
second public operation. An absent matrix-format keyword means Full.

Full records preserve every arbitrary row-major complex value. Lower and Upper
records contain the diagonal and exactly `n(n+1)/2` complex pairs per sample;
the missing half is expanded by plain transpose, never conjugation. A two-port
document still requires `[Two-Port Data Order]` with `21_12` or `12_21`, and
both legal orders decode to the same symmetric matrix under the ratified
Touchstone 2.0 semantics. The representation is selected from the document,
never inferred from the scalar count. RI, MA, and DB pairs, all supported
frequency units, continued `[Reference]` data, option-line override, physical
frequency-line boundaries, the finite/non-negative/strictly increasing grid,
and positive real per-port references retain the prior semantics.

The reader remains pure in-memory and single-ended. It continues to reject
non-S parameters, mixed-mode/noise/information blocks, unknown or other-version
keywords, complex or non-positive references, malformed or non-finite values,
duplicate/misplaced directives, invalid count/grid/record boundaries, semantic
vendor comments, missing `[End]`, and trailing content. It does not sort,
interpolate, average, repair, renormalize, infer filenames, or add a writer or
metadata model. The existing writer remains explicitly
`write_touchstone_v2_0_s_full_ri_hz`; it emits Full/RI/Hz only and keeps its
current reference restrictions.

The pinned conformance evidence is independently authored and registered as
`touchstone_v2_0_s_full_three_port`,
`touchstone_v2_0_s_lower_three_port`, and
`touchstone_v2_0_s_upper_three_port`. The Lower and Upper fixtures are literal
deterministic N=3 RI inputs with three samples, unequal real references
`[25,50,75]`, a continued `[Reference]` vector, split compact records, and
nonzero imaginary entries. The Full fixture remains the existing canonical
input and numeric values. All three record scikit-rf `2.0.1` at commit
`bd651e923cac6020de49a096e1d7e9b5f949f884`, NumPy `2.5.1`, exact
frequency/reference/text/shape/metadata contract fields, and only decoded S
under `rtol=1e-12`, `atol=1e-12`. The Lower/Upper operation records are
`touchstone_v2_0_s_lower_parse` and `touchstone_v2_0_s_upper_parse`.

The pinned scikit-rf two-port triangular `21_12` path has a legacy
transpose-before-mirror defect: its opposite-triangle storage can be read
before it is initialized (the required Lower/`21_12` discrepancy is the
documented example, and the analogous Upper path is not canonicalized). No
uninitialized or opposite-triangle value is accepted as an expected fixture.
This is a reference-implementation defect, not a conflict between RF
standards; Rust follows the ratified two-port semantics in its deterministic
regression and makes no broad scikit-rf compatibility promise.

The Yellow decision selected one version/S-explicit reader whose document
keyword selects matrix storage. Keeping a separate triangular reader beside
the old Full reader would duplicate caller choice for one decoding operation.
Broadening the old `_full` name was rejected because it would misdescribe its
accepted domain; retaining triangular rejection and requiring caller-written
preprocessing leaves the measured ingress gap; a universal reader/options/
metadata type is larger than this bounded capability. The selected
generalization is therefore the overlap outcome: one public reader, no new
overlap-inventory row, and no change to the three existing RF-conversion
overlaps.

Compatibility is intentionally provisional because `rfkit-rs` is unpublished
and remains in `0.x`: existing callers replace
`parse_touchstone_v2_0_s_full` with `parse_touchstone_v2_0_s`, with no alias.
Successful Full inputs and results remain unchanged; Lower/Upper moves from an
explicit unsupported error to decoding or contextual validation errors.
Unaffected diagnostics retain their structure, with only compact-count
diagnostics necessarily generalized where applicable. No persisted-data or
canonical-storage migration is needed. Rollback is the bounded PR reversal:
restore the historical Full-only name and triangular rejection, revert the
migrated callers/docs/oracle registrations, and leave stored data untouched.
The choice remains reversible until an explicit stabilization milestone.

Evidence and acceptance are preserved in Issue #112 and
`docs/CONFORMANCE.md`; implementation provenance is a REWRITE from the
ratified IBIS Touchstone 2.0 specification, with pinned scikit-rf consulted
only as a behavior oracle. The focused workflow loads compact triangular data,
permutes/inspects the resulting Network, writes the existing Full v2 text,
and reads it back while retaining frequencies and unequal references without
implicit renormalization.

## Touchstone 2.0 Full S egress (Issue #104 Yellow)

The `rfkit-touchstone` crate adds the additive pure in-memory function
`write_touchstone_v2_0_s_full_ri_hz(network: &Network) -> Result<String>`.
Its explicit name fixes the output subset: Touchstone 2.0, single-ended Full
S, RI pairs, and hertz. The function borrows the canonical `Network`, validates
the complete input before constructing output, and never performs filesystem
I/O, sorting, interpolation, or implicit renormalization.

The deterministic document order is `[Version] 2.0`, `# Hz S RI R
<first-port-reference>`, `[Number of Ports]`, `[Two-Port Data Order] 12_21`
for exactly two ports, `[Number of Frequencies]`, one physical-line
`[Reference]` vector in physical port order, `[Matrix Format] Full`,
`[Network Data]`, one complete frequency record per physical line, and
`[End]`. Each record contains the frequency followed by all complex pairs in
frequency-major, row-major `S[row,column]` order. Output is ASCII, uses LF
line endings and one final newline, and uses Rust's shortest round-tripping
binary64 decimal representation.

The supported input domain is a nonempty, finite, non-negative, strictly
increasing hertz axis; finite S components in a positive square
`(nfreq,nport,nport)` array; and a `(nfreq,nport)` reference array. Every
reference must be finite, strictly positive, and real; `+0.0` and `-0.0`
imaginary components are both treated as numerically zero. References may be
unequal between physical ports, but each port's real value must be exactly
constant across all samples. No tolerance, averaging, first-sample
substitution, common-50-ohm assumption, sorting, clipping, or implicit
renormalization is applied. The operation-specific v2 writer errors retain
sample/port or row/column context, including a later-sample change to one
port's reference; malformed serde-created shapes are rejected before any
unchecked indexing or output construction.

This writer intentionally excludes MA/DB, Lower/Upper or mixed-mode matrices,
noise, vendor extensions, metadata, complex references, and frequency-varying
references. The existing v1 writer remains unchanged and still requires one
common finite positive-real reference across every sample and port. Callers
with unsupported references must choose an explicit core transformation (for
example direct renormalization) before v1 export, or use this v2 function when
its bounded contract applies.

The Yellow alternatives were widening or auto-detecting the v1 writer,
requiring all callers to renormalize before export, adding a universal
options/metadata abstraction, or emitting vendor-specific complex or
frequency-varying references. The selected additive API preserves supported
coordinates, keeps version/matrix/encoding semantics visible, and is
reversible during the provisional 0.x period by removing only the writer,
tests, oracle calls, example, and documentation; it changes no core storage or
existing v1 behavior. The focused executable
`crates/rfkit-touchstone/examples/touchstone_v2_writer_workflow.rs` parses an
unequal-reference v2 input, permutes physical ports, writes without
renormalization, and independently checks the reordered S/reference values on
readback.

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
apply a cutoff, fall back, or take an identity shortcut. Touchstone export is
deliberately separate: the v1 writer retains its finite/nonnegative/strictly-
increasing/common-positive-reference contract, while the additive v2 writer
allows unequal real references that are exactly constant per port across the
frequency axis.

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

## Semantic qualifiers and evaluation strategy

Public names and required arguments must carry **semantic qualifiers**: choices that change which RF quantity is computed or which caller-visible policy applies. Examples are the wave convention (`_power`), the frequency-grid policy (an explicit target grid), interpolation coordinates and kind (`cartesian_linear`), pairing and modal layout (`equal_pair`), sample alignment (`secant`, interval-aligned), and survivor ordering.

An **evaluation strategy** is how one semantically fixed quantity is computed: direct versus composed through Z or Y, a matched-only kernel versus a general junction solve, or which linear system is factorized. The strategy is an implementation detail. It should not become a public axis once one verified path covers the domain of the others.

A consolidated public operation must:

- keep every semantic qualifier explicit and drop strategy-only qualifiers such as `direct`, `via_z`, or `matched` when they no longer distinguish semantics;
- accept the union of the retired entry points' documented domains, so no input that succeeded through a retired entry point starts failing, unless that success is documented as a defect;
- agree with each retired entry point on their common domain within the recorded tolerance, demonstrated by running the retired entry points' existing conformance fixtures through the consolidated operation;
- use one documented evaluation per input, chosen only from properties that are checked before computing; it must never retry a different path after a numerical failure;
- keep failures structured and document any change in stage diagnostics;
- record migration from each retired name in this document and update README and examples in the same change.

`rfkit-rs` has not been published, so a consolidation may remove retired names in the same change instead of keeping deprecated aliases, unless a staged in-repository migration needs them. Historical decision records below stay as history; add a `Superseded by #N` line to the affected record rather than rewriting it.

## Public surface overlap inventory

This inventory lists public operations that currently expose the same RF quantity through more than one entry point. It is the current consolidation direction. Where it differs from the "explicit coexistence" wording in older decision records, the older records describe the decision taken at that time.

A change that adds a public operation overlapping an existing one must either consolidate them under the rules above or add an entry here stating why callers, not the implementation, need both entry points and what condition will retire the overlap. A change that consolidates an entry removes it from this list.

| Operation | Entry points | Relationship | Consolidation direction |
|---|---|---|---|
| S→Y extraction | `to_y_power`, `to_y_direct_power` | Same Y. The composed path needs invertible `I-S` and Z; the direct path needs only its `A` system, so its domain is expected to contain the composed domain. | One `to_y_power` using the direct path. |
| Y ingress | `from_y_via_z_power`, `from_y_direct_power` | Same S. The composed path fails on singular Y, which the direct path accepts. | One `from_y_power` using the direct path. |
| Power-wave renormalization | `renormalize_power`, `renormalize_direct_power` | Same S. The composed path fails at singular Z; the direct path does not need Z. | One `renormalize_power` using the direct path. First confirm whether any composed-only validation is a semantic contract rather than a domain limitation. |
The following are **not** overlaps, because their qualifiers are semantic: `cascade_direct_power`, which keeps within-group coupling in one simultaneous solve and differs from repeated connection; `inverse_cascade_power`; the `equal_pair` mixed-mode pair; and the `_power` wave suffix itself.

## Consolidated physical power-wave connection (Issue #109 Yellow decision)

Issue #109 consolidates the matched/direct two-network and same-network
physical-junction entry points into exactly these two public methods:

```rust
impl Network {
    pub fn connect_power(
        &self,
        port_a: usize,
        other: &Network,
        port_b: usize,
    ) -> Result<Network>;

    pub fn inner_connect_power(
        &self,
        port_a: usize,
        port_b: usize,
    ) -> Result<Network>;
}
```

The operation is one Kurokawa power-wave physical V/I junction. `connect_power`
returns A survivors followed by B survivors, preserving each source order and
reference exactly. `inner_connect_power` removes two selected ports and
preserves the original survivor order. Neither method changes the wave
convention, inserts a mismatch network, renormalizes, or chooses a frequency
grid.

The selector performs one pre-computation decision. It first requires the
existing exact-grid, shape, axis, finite-value, port, and survivor validity
conditions. If the selected junction references are finite, exactly real,
strictly positive, and exactly equal under Rust's `f64` value equality at
every frequency, it uses the existing matched kernel (`-0.0` and `+0.0`
therefore compare equal). Otherwise it uses the existing direct physical V/I
kernel.
The direct domain therefore retains unequal, complex, and negative-real
nonzero-real junction references, while the matched path still accepts finite
complex or zero-real *external survivor* references and copies them. A failed
matched or direct computation is never retried through the other kernel.

For the matched choice, each frequency is evaluated exactly once by a private
dyadic complex Schur evaluator. Every finite binary64 real and imaginary
component is converted to `num_rational::BigRational`; `1-S`, the matched 2x2
determinant/adjugate, the bilinear `S_EI * adjugate * P * S_IE` term, and the
final `Q = S_EE * D + N` numerator remain rational until `Q / D` is converted
at the output boundary. `num-rational` performs the final binary64
round-to-nearest-even conversion, including subnormal and overflow behavior.
The matched path has one exact evaluator: it does not materialize a binary64
inverse or internal RHS, use a pivot threshold, or retry through the direct
kernel. An exact zero determinant is a structured singularity; if the final
rational output is outside finite binary64 range, the result is a structured
non-finite computation.

Both methods require exact pointwise-compatible frequency labels and do not
sort, intersect, extrapolate, interpolate, or broadcast. Callers that need a
target grid compose the policy explicitly:

```rust
let a_on_grid = a.interpolate_cartesian_linear(&grid)?;
let b_on_grid = b.interpolate_cartesian_linear(&grid)?;
let joined = a_on_grid.connect_power(port_a, &b_on_grid, port_b)?;
```

Interpolation errors remain interpolation errors and connection errors remain
connection errors; the deleted convenience adapter and its `GridConnection`
wrapper are not part of the public boundary. Exact singularities and checked
non-finite arithmetic retain structured diagnostics. Because the selector can
change which private kernel supplies a failure, the diagnostic family may now
be matched-junction or direct-junction; the input location, selected ports,
frequency, and arithmetic context remain structured where available.

The consolidation accepts the union of the retired *physical* domains and
agrees with the matched and direct fixtures on their common domains within
their recorded tolerances. One documented compatibility exception is a
numerical defect in the retired matched Gaussian solver: for
`S_ii=[[-11,-54],[-2,-15]]`, its rounded second pivot can be `2^-49` even
though the exact determinant is zero. The consolidated exact evaluator reports
that physical singularity instead of preserving the false old success; this
is not a new semantic restriction. Evidence includes the matched and direct
public/oracle fixture suites, independent common-domain physical
reconstructions, A-then-B and original-survivor mapping tests, the explicit
interpolation-composition fixtures, the Touchstone workflows, Issue #44
bit-exact arithmetic, exact-dyadic A/B/C/D and huge-internal-block cases,
minimum-subnormal and final-rounding conversion tests, final-Q cancellation,
genuinely unrepresentable-output diagnostics, and exact/near-singular
selector tests. The singular matched-junction case is required to report its
exact singularity rather than succeeding via an alternate path.

Alternatives rejected were retaining strategy-qualified public names, exposing
a public strategy switch, using direct-only evaluation, silently interpolating
or selecting a default grid, pre-renormalizing mismatched references, or
retrying after arithmetic failure. The crate is unpublished 0.x software, so
the five retired names are removed without deprecated aliases; the migration
is:

- `connect_matched_power` and `connect_direct_power` → `connect_power`;
- `inner_connect_matched_power` and `inner_connect_direct_power` →
  `inner_connect_power`;
- `connect_matched_power_on_grid` → explicit interpolation of both inputs
  with `interpolate_cartesian_linear`, followed by `connect_power`.

Rollback is reversible by reverting the Issue #109 change, which restores the
five former public methods and the explicit-grid adapter and removes the two
consolidated methods, selector, exact matched-evaluation adaptation, and
migrated tests/docs. No serialized data, frequency grid, wave convention, or
storage representation migration is required. The canonical fixture values
remain available as conformance lineage, and no data migration is needed to
undo the API change.

## Wave semantics

The current verified conversion, renormalization, and matched-connection kernels use Kurokawa power-wave semantics. The `Network` model does not yet carry a wave-definition field.

Keep that fact explicit in **wave-sensitive public method names** rather than adding a speculative `WaveDefinition` field before a second convention is actually supported.

An additional convention with authoritative mathematics, explicit naming, and conformance evidence may proceed as Yellow. Selecting a broad implicit wave default, or resolving an authoritative disagreement that cannot be represented through explicit side-by-side APIs, is Red.

When a second convention is actually added, prefer an explicit required wave-definition argument or type on wave-sensitive operations over duplicating every wave-sensitive method name. The `_power` names would then migrate under the consolidation rules above. Making that wave argument optional, or giving it a default, is still an implicit wave default.

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

## Power-wave inverse cascade (Issue #94 Yellow decision)

The provisional public surface adds one borrowing, owned-result operation:

```rust
impl Network {
    pub fn inverse_cascade_power(&self) -> Result<Network>;
}
```

The source must have an even positive port count `2N`, interpreted as two
ordered groups `[left_0..left_(N-1), right_0..right_(N-1)]`. The result uses
the fixed `[old right, old left]` group order. If `P` exchanges those groups,
the mathematical definition is

```text
a = F (V + G I)                 b = F (V - conj(G) I)
b = S a                         V' = P V, I' = -P I
a' = P b, b' = P a             z0' = P conj(z0)
S_inverse = P S^-1 P
```

The implementation computes and stores the full `S^-1` by solving `S X = I`
through the existing checked multiple-right-hand-side exact-pivot solver. It
does not use an elementwise reciprocal or silently convert through Z/Y. The
returned `Network` owns new S/z0 arrays, copies the source frequency
axis exactly (including order and signed-zero bits), and leaves the borrowed
source unchanged.

The inverse-cascade domain is deliberately two-sided. Before indexing, the
operation validates a nonempty frequency axis matching S, square positive-even
S, z0 shape `(nfreq,2N)`, finite frequency/S/z0 values, and a nonzero real part
for every reference. At each sample it requires exact nonsingularity of the
full S, `S[right,left]` (forward transmission), and `S[left,right]` (reverse
transmission). [`InverseCascadeStage`] distinguishes these three stages in
structured errors; numerical failures retain sample and pivot or row/column
context. Only exact evaluated zero pivots are singular. Finite near-singular
inputs remain eligible when all checked arithmetic stays finite, and no rank
cutoff, regularization, pseudoinverse, nudge, or fallback is used.

Unequal, per-port, frequency-dependent, complex, and negative-real references
are supported under the existing algebraic `abs(Re(z0))` power-wave extension;
negative-real references do not imply a passive-power interpretation. Inverse
networks may be active or noncausal mathematical removal operators. A known
fixture cancels only with the declared orientation, paired ports, compatible
frequency grids, and nonsingular direct connection conditions. Callers must
explicitly renormalize when comparing a recovered DUT at another reference.
No noise de-embedding, automatic calibration, pole/stability claim,
measurement-error correction, or broad scikit-rf compatibility promise is
implied.

The selected Yellow scope is one explicit method with a fixed group convention
and a bounded nonsingular transmission domain. Plain reference swapping was
rejected for complex power waves; a two-port/common-reference-only operation,
inferred pairing metadata, and a new transfer-parameter hierarchy were also
rejected as unnecessary or less reversible. The public method is additive and
provisional during 0.x, so rollback removes the method/kernel, diagnostics,
tests, fixture, and documentation without stored-data migration or changes to
the canonical `Network` representation.

Conformance uses one independently seeded four-port, three-frequency fixture
with unequal real-positive references, generated and checked against pinned
public scikit-rf `Network.inv` 2.0.1 at commit
`bd651e923cac6020de49a096e1d7e9b5f949f884` and NumPy `2.5.1`. The generator
records random seed `20260955`, ordered groups, and the independent
`P @ solve(S,I) @ P` check with a `<=1e-12` guard; runtime floating residuals
and determinant magnitudes are not canonical metadata. Strict output-only
`rtol=1e-12`, `atol=1e-12` remains unchanged; frequencies, inputs, references,
shapes, and the remaining metadata are exact contract fields. The Rust oracle test calls only the public method. Local tests
add ideal-through, analytic/non-reciprocal two-port, complex-reference V/I,
coupled asymmetric four-port, larger six-port, double-inverse, malformed/error,
and both-order physical cancellation coverage. The Touchstone counterpart is
`crates/rfkit-touchstone/examples/inverse_cascade_touchstone.rs`.

## Simultaneous direct power-wave cascade (Issue #96 Yellow decision)

The provisional public surface adds one borrowing, owned-result operation:

```rust
impl Network {
    pub fn cascade_direct_power(&self, other: &Network) -> Result<Network>;
}
```

Both inputs must have the same positive even port count `2N`, with ordered
groups `[left_0..left_(N-1), right_0..right_(N-1)]`.  The operation connects
every `self.right_k` to `other.left_k` simultaneously and returns surviving
order `[self.left..., other.right...]`.  It does not infer a pairing or
physical ordering; callers use `permute_ports` explicitly for another layout.

For internal coordinates `i=[self.right...,other.left...]` and external
coordinates `e=[self.left...,other.right...]`, the full source blocks satisfy
`b_i=S_ie a_e+S_ii a_i` and the Kurokawa physical boundary is
`C a_i+D b_i=0`.  The kernel solves the complete joint system and evaluates

```text
(C + D S_ii) X = -D S_ie
S_out = S_ee + S_ei X
```

The `S_ii` block retains all within-group coupling.  The implementation uses
the checked exact-pivot multiple-right-hand-side solver and does not use an
inverse, pseudoinverse, least-squares fallback, regularization, condition
cutoff, transfer/Z/Y conversion, or an identity shortcut.  Full input S and
directional transmission blocks need not be invertible.  Only an exactly zero
evaluated joint pivot is singular; finite near-singular systems remain in the
domain when arithmetic stays finite.

The two frequency axes must be nonempty, finite, equal in length, and equal at
each position under ordinary `f64` equality.  The returned axis is an exact
copy of `self`, including signed-zero bits and non-monotone point labels.
Every S/reference value is finite and every reference has nonzero real part.
Unequal, per-port, frequency-dependent, complex, and signed negative-real
references use the existing algebraic `abs(Re(z0))` normalization; surviving
references are copied exactly and no common-reference or hidden
renormalization is applied.  The source networks are borrowed and unchanged.

Malformed serde-created shapes are checked before group indexing.  Structured
errors identify A/B input context for shapes, frequency axes, finite values,
and zero-real references, plus frequency and matrix row/column or pivot
context for arithmetic and exact singularity.  The method is additive and
provisional during `0.x`; rollback removes the kernel, method, diagnostics,
tests, fixture, and documentation without persisted-data migration.

The compared alternatives were retaining caller-side repeated connections,
composing through transfer/Z/Y parameters, and introducing arbitrary pair
maps, unequal-size graph/topology cascades, calibration, or a parameter
hierarchy.  They were rejected because repeated connections lose the
partial-singular/joint-nonsingular domain, intermediate parameter inverses
add unnecessary restrictions, and the broader surfaces are not required by
the coupled fixture workflow.  No release, stability, passivity, calibration,
noise, or broad scikit-rf compatibility promise is implied.

Conformance uses the independently seeded
`power_wave_cascade_direct_four_port_complex_z0.json` fixture (seed
`20260956`), with coupled four-port A/B inputs, unequal complex
frequency-dependent positive-real references, and pinned public scikit-rf
`Network.cascade` 2.0.1 at commit
`bd651e923cac6020de49a096e1d7e9b5f949f884`; NumPy `2.5.1` is recorded.  The
oracle explicitly restores the public result with
`output.renormalize(output.z0, s_def="power")` before extracting S.  Only
output S uses `rtol=1e-12`, `atol=1e-12`; inputs, surviving references, order,
frequency, shapes, and metadata are exact contract fields.  Local tests cover
two-port sequential agreement, coupled four-port and larger even-port
composition, complex/signed-reference V/I behavior, the partial-singular
witness, exact singular/near-singular systems, zero S, malformed/error and
arithmetic cases, inverse composition/removal, and source immutability.  The
Touchstone counterpart is
`crates/rfkit-touchstone/examples/cascade_direct_power_touchstone.rs`.

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

Superseded by #109

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

The corresponding current Touchstone workflow is
`crates/rfkit-touchstone/tests/public_connect_power_workflow.rs` with
`crates/rfkit-touchstone/examples/connect_power_touchstone.rs`. It parses two
separate v1.0 inputs with different common positive-real references, calls
`connect_power` without pre-renormalizing either source, checks the response
through the physical V/I boundary independently, then explicitly calls
`renormalize_direct_power` to a caller-selected common writer-compatible
reference before writing and reading. The writer does not repair or
renormalize a direct-connection result automatically.

## Direct physical power-wave inner connection (Issue #90 Green decision)

Superseded by #109

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
available. The consolidated connection methods and their explicit
interpolation-then-connection grid workflow now provide the union of the
former matched/direct domains. Termination and writer semantics remain
unchanged.

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
`crates/rfkit-touchstone/examples/inner_connect_power_touchstone.rs`
and
`crates/rfkit-touchstone/tests/public_inner_connect_power_workflow.rs`.
They parse the multiport input, explicitly direct-renormalize selected
references, independently reconstruct the full physical V/I response, close
the pair, explicitly restore one writer-compatible positive-real reference,
and write/read the result.

## Sampled maximum power-wave singular value (Issue #98 Yellow decision)

The provisional public surface adds one borrowing, owned-result diagnostic:

```rust
impl Network {
    pub fn max_singular_value_power(&self) -> Result<Vec<f64>>;
}
```

For each stored frequency sample, the method returns the dimensionless
amplitude ratio

```text
sigma_max(S) = max_{a != 0} ||S a||_2 / ||a||_2
```

under the network's Kurokawa power-wave coordinates.  The corresponding
maximum reflected/incident wave-power ratio is `sigma_max(S)^2`; the method
returns the amplitude value itself, not its square, a dB value, column norm,
Frobenius norm, spectral radius, elementwise square root, or a passivity
boolean.  The returned vector has exactly `frequency.len()` entries in source
order.  The network, frequency axis, S array, and z0 array are borrowed and
unchanged; no sorting, interpolation, aggregation, hidden renormalization, or
frequency selection is performed.

The RF basis is the existing power-wave identity for finite references with
strictly positive real parts:

```text
Re(V conj(I)) = ||a||_2^2 - ||b||_2^2,  b = S a
```

so `sigma_max <= 1` is the sampled contractivity condition for all simultaneous
incident waves.  This operation reports the measured binary64 value and does
not classify a value near one.  It is not an all-frequency passivity
certificate, an internal-pole/causality guarantee, a stability assessment,
noise or fitting operation, gain optimization, or passivity enforcement.

The full dense complex decomposition is private and uses nalgebra `0.33.3`
with `default-features = false, features = ["std"]`; no nalgebra types or
BLAS/LAPACK runtime are exposed.  The operation passes a binary64 convergence
tolerance of `5 * f64::EPSILON` and a finite total budget of 10,000 SVD
iterations per sample.  These settings govern solver convergence only and
are not an RF tolerance.  Nalgebra's dense SVD scales the input internally;
the adapter does not form `SᴴS`, use a random/single-vector power iteration,
truncate rank, regularize, invert S/Z/Y, clip around one, or fall back to a
column heuristic.  Exact zero, rank-deficient, repeated-singular-value,
one-port, unitary/lossless, and active matrices are valid.  A non-convergent
or non-finite decomposition is an operation-specific structured error rather
than a panic or NaN/Inf result.

Before indexing, the method validates malformed serde-created values.  The
frequency axis must be nonempty, have the same length as S's first axis, and
contain finite labels; negative, duplicate, descending, and signed-zero
labels remain valid pointwise samples.  S must be finite, square, and have a
positive port count.  z0 must be finite and have shape `(nfreq, nport)`, with
strictly positive real parts at every port and sample.  Unequal,
frequency-dependent, per-port, and complex positive-real references are
supported.  The numeric result depends on those wave coordinates and is not
invariant under arbitrary renormalization; port permutations and unitary
coordinate changes preserve the singular values.

The selected Yellow slice is additive and provisional during `0.x`.  A
column/Frobenius norm was rejected because coherent excitation can amplify
power even when every column norm is below one; eigenvalue magnitude was
rejected because a nonnormal matrix such as `[[0,2],[0,0]]` has zero
eigenvalues but `sigma_max=2`.  A boolean `is_passive(tol)` was rejected as a
first API because it hides the measured boundary policy.  A public SVD type,
handwritten Jacobi/QR implementation, power iteration, or FFI BLAS backend
would add types, convergence ownership, or deployment burden without helping
this bounded sampled workflow.  The selected dependency and adapter are
reversible before stabilization: rollback removes the method, diagnostics,
tests, fixture, dependency, and documentation without storage migration.

## Adjacent-interval power-wave group delay (Issue #100 Yellow decision)

The provisional public surface adds one borrowing, owned-result diagnostic:

```rust
impl Network {
    pub fn group_delay_secant_power(
        &self,
        port_out: usize,
        port_in: usize,
    ) -> Result<Vec<f64>>;
}
```

The method selects the zero-based stored coordinate `S[port_out, port_in]` and
returns exactly `nfreq - 1` seconds values.  Value `k` belongs to the actual
frequency aperture `[f[k], f[k+1]]`, including on a nonuniform grid.  For the
principal phases `phi[k] = atan2(Im(S[k]), Re(S[k]))`, it takes the shortest
adjacent increment and evaluates:

```text
d = phi[k+1] - phi[k]
if d > pi:  d -= 2*pi
if d < -pi: d += 2*pi
delay[k] = -d / (2*pi*(f[k+1] - f[k]))  // seconds, f in Hz
```

This is an explicit finite-aperture secant estimate, not scikit-rf's
sample-aligned `Network.group_delay`, which uses `gradient` on the public
unwrapped trace and returns one value per source sample.  The Rust operation
does not synthesize endpoints, midpoint frequencies, cumulative winding
counts, smoothing, fitting, or time-domain data.  A physical phase advance of
magnitude `>= pi` is not generally recoverable from adjacent samples; exact
evaluated half-turns are rejected, while other undersampling aliases remain a
caller responsibility.

Validation is operation-specific and happens before selected indexing,
including for malformed serde-created networks: at least two samples, exact
S/frequency cardinality, square positive-port S, exact `(nfreq,nport)` z0,
in-range ports, finite nonnegative strictly increasing Hz, finite all S/z0,
and nonzero z0 real parts.  Negative-real, complex, unequal, and
frequency-dependent references are accepted algebraically.  An exact-zero
selected sample returns an undefined-phase error; zero unselected entries and
singular full S matrices remain valid.  The kernel extracts phase directly
with `atan2`, avoiding magnitude/product/ratio overflow for finite huge or
subnormal nonzero selected components.  Seconds conversion retains a
large-aperture overflow-avoiding order and, for subnormal apertures, divides
the phase increment by the aperture before dividing by `2*pi` when that
intermediate quotient is finite; this avoids rounding `2*pi*df` in subnormal
space while retaining a safe fallback when the intermediate quotient
overflows.  A genuinely unrepresentable result is a structured arithmetic
error; no NaN/Inf is returned.

The Yellow alternatives were sample-aligned central/one-sided gradients,
caller-selected windows, an Option-valued full N-port output, a public phase
trace hierarchy, or leaving phase handling to callers.  The selected
per-trace adjacent result keeps alignment and undefined/ambiguous boundaries
explicit while remaining additive and reversible during 0.x.  It makes no
new scikit-rf compatibility or stability/causality/propagation certificate.
Rollback removes this method, diagnostics, tests, fixture, and docs without
data migration; a future sample-aligned estimator may coexist under a
separate explicit name.

The canonical differential fixture is
`tools/oracle/fixtures/group_delay_secant_power_three_port_branch_crossing.json`:
seed `20260958`, seven nonuniform frequencies `[0,31,80,143,225,320,429]` MHz,
an asymmetric three-port varying-amplitude S31 branch crossing, and complex
frequency-dependent references.  Expected seconds come from pinned public
scikit-rf `2.0.1` `Network.s_rad_unwrap` followed by explicit interval
differencing; only output uses `rtol=1e-12`, `atol=1e-21`.

The focused Touchstone workflow is
`crates/rfkit-touchstone/examples/group_delay_secant_touchstone.rs` and
`crates/rfkit-touchstone/tests/public_group_delay_workflow.rs`; it displays
each S21 interval with frequency bounds and checks an analytical 2 ns line
delay.

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

    pub fn inverse_cascade_power(&self) -> Result<Network>;

    pub fn cascade_direct_power(&self, other: &Network) -> Result<Network>;

    pub fn permute_ports(&self, order: &[usize]) -> Result<Network>;

    pub fn to_mixed_mode_equal_pair_power(&self, pair_count: usize) -> Result<Network>;

    pub fn to_single_ended_equal_pair_power(&self, pair_count: usize) -> Result<Network>;

    pub fn interpolate_cartesian_linear(
        &self,
        target: &Frequency,
    ) -> Result<Network>;

    pub fn connect_power(
        &self,
        port_a: usize,
        other: &Network,
        port_b: usize,
    ) -> Result<Network>;

    pub fn inner_connect_power(
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

    pub fn max_singular_value_power(&self) -> Result<Vec<f64>>;

    pub fn group_delay_secant_power(
        &self,
        port_out: usize,
        port_in: usize,
    ) -> Result<Vec<f64>>;
}
```

The exact internal delegation remains an implementation detail. The semantic distinctions represented by these names remain policy and evidence for future API consistency:

- `to_z_power`, `to_y_power`, and `to_y_direct_power` explicitly select the verified power-wave conversion convention; the latter names the direct S→Y equation while `to_y_power` retains composed S→Z→Y semantics;
- `renormalize_power` explicitly selects power-wave renormalization;
- `renormalize_direct_power` explicitly selects the direct Kurokawa wave-change equation, while `renormalize_power` retains its composed S→Z→S domain and diagnostics;
- `inverse_cascade_power` explicitly selects fixed-group Kurokawa wave reversal `P S^-1 P` with conjugated/group-exchanged references and full/forward/reverse transmission diagnostics; it does not imply a generic dense-inverse or calibration API;
- `cascade_direct_power` explicitly selects one simultaneous fixed-group
  direct V/I cascade with `[self.left...,other.right...]` survivors and full
  within-group coupling; it does not imply sequential connection semantics,
  transfer/Z/Y composition, arbitrary pair maps, topology, or calibration;
- `permute_ports` explicitly selects a complete new-to-old physical-port reindexing and carries S rows, S columns, and z0 together without wave arithmetic;
- `to_mixed_mode_equal_pair_power` and `to_single_ended_equal_pair_power` explicitly select the equal single-ended-reference adjacent-pair power-wave transform and its inverse, with modal layout `[d...,c...,unpaired...]`; they do not infer physical pairing or store mode metadata;
- `interpolate_cartesian_linear` does not establish a vague interpolation default that would later need reinterpretation;
- `connect_power` exposes one physical Kurokawa junction for the union of the
  matched and direct domains. Its pre-computation selector chooses the
  existing matched kernel only for finite, exactly real, strictly positive,
  exactly `f64`-equal selected references on an exact valid grid; otherwise
  it chooses the direct V/I solve. A's survivors precede B's survivors.
- `inner_connect_power` exposes the same physical junction inside one network,
  retaining the full selected 2×2 S block and original survivor order. It uses
  the same one-time selector and never retries through the other kernel.
- Explicit target-grid connection is caller composition:
  `interpolate_cartesian_linear` on both inputs, then `connect_power`; there is
  no hidden interpolation or renormalization.
- `terminate_port_impedance_power` applies a finite physical impedance boundary directly at one
  selected port, removes that port, and retains the original survivor order and references without
  selecting a new frequency grid or renormalizing the source.
- `max_singular_value_power` reports the full dense sampled largest singular
  value of each stored S matrix in the existing power-wave coordinates.  It
  returns an amplitude diagnostic rather than a passivity verdict and does
  not imply an all-frequency, internal-pole, stability, or scikit-rf API
  compatibility promise.
- `group_delay_secant_power` reports shortest-principal-phase secants on
  adjacent source-frequency intervals for one selected stored power-wave S
  coordinate.  It is explicitly interval-aligned and seconds-valued, unlike
  scikit-rf's sample-aligned `Network.group_delay`; it does not imply phase
  unwrapping, undersampling detection, propagation speed, causality,
  stability, or a broad scikit-rf API compatibility promise.

Do not shorten these to broad names such as `connect`, `interpolate`, or `renormalize` until the library has enough supported semantics and evidence to justify what those names mean. Introducing such a default is at least Yellow and becomes Red when reasonable conventions conflict or the choice would freeze hidden policy.

This does not prevent consolidating strategy-only variants. For example, `connect_power` still names the wave convention; dropping `matched` or `direct` removes only an evaluation-strategy qualifier. Such consolidations follow the overlap inventory and the rules in "Semantic qualifiers and evaluation strategy".

## Frequency policy

Public connection APIs must not silently choose a frequency grid.

- `connect_power` and `inner_connect_power` use the stored exact pointwise
  labels; they do not sort, intersect, subset, extrapolate, interpolate, or
  infer a target grid.
- A caller-selected grid is explicit composition: interpolate each input with
  `interpolate_cartesian_linear(&grid)?`, then call `connect_power`.
- Errors remain staged as interpolation or connection errors; there is no
  `GridConnection` convenience wrapper at the public boundary.

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
- it does not add an unrecorded overlap with an existing public operation (see the overlap inventory);
- no new compatibility promise, irreversible action, or uncertain provenance is involved.

The Planner may dispatch a public API increment as **Yellow** when a material choice remains but all of these are true:

- the choice is bounded and reversible during the provisional `0.x` phase;
- the Issue records the plausible alternatives, selection criteria, compatibility impact, and intended rollback or migration;
- the selected API names its semantics rather than hiding policy in a convenience default;
- the PR preserves that decision record and the independent reviewer explicitly evaluates the RF/API choice, evidence, and reversibility;
- the change does not cross a Red boundary.

Examples of Yellow work include an explicitly named additional wave convention, a justified supporting public type, a bounded dependency or crate-boundary adjustment, a provisional API revision with a documented migration, and consolidating an overlap-inventory entry into one entry point under the rules above. Yellow is not a license for speculative abstraction or weakly sourced RF behavior.

The following remain **Red** and require human approval:

- API stabilization, release policy, or a compatibility guarantee;
- an implicit broad default when multiple reasonable wave, frequency, interpolation, or connection semantics remain;
- externally visible RF behavior where authoritative sources materially disagree and explicit APIs cannot preserve the alternatives;
- a difficult-to-reverse replacement of the canonical model, storage representation, or repository architecture;
- unresolved provenance/licensing obligations or unavailable evidence required for correctness;
- release, package publication, signing, credentials, or another irreversible external action.

Implementation remains incremental. A Red question blocks only the affected Issue; it does not turn this baseline into a repository-wide stop when independent Green or Yellow work remains.
