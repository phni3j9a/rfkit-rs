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

Do **not** introduce view lifetimes, generic storage traits, port-index newtypes, builders, parameter-container hierarchies, or alternate owned network representations merely to make the API look more abstract. A concrete supporting type or bounded internal architecture change may be Yellow when demonstrated usage justifies it. Replacing the canonical model or storage representation in a difficult-to-reverse way is Red.

## Operation style

Prefer methods on `Network` when one network is the natural primary object. Operations should be pure from the caller's perspective: borrow inputs and return owned results rather than mutating a network in place.

The public API should not expose a parallel free-function surface that duplicates the same operations without a demonstrated reason.

Use `Frequency` rather than raw frequency slices at public operation boundaries.

## Wave semantics

The current verified conversion, renormalization, and matched-connection kernels use Kurokawa power-wave semantics. The `Network` model does not yet carry a wave-definition field.

Keep that fact explicit in **wave-sensitive public method names** rather than adding a speculative `WaveDefinition` field before a second convention is actually supported.

An additional convention with authoritative mathematics, explicit naming, and conformance evidence may proceed as Yellow. Selecting a broad implicit wave default, or resolving an authoritative disagreement that cannot be represented through explicit side-by-side APIs, is Red.

## Implemented public baseline

The implemented shape is:

```rust
impl Network {
    pub fn to_z_power(&self) -> Result<Array3<Complex64>>;
    pub fn to_y_power(&self) -> Result<Array3<Complex64>>;

    pub fn renormalize_power(
        &self,
        new_z0: Array2<Complex64>,
    ) -> Result<Network>;

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

- `to_z_power` / `to_y_power` explicitly select the verified power-wave conversion convention;
- `renormalize_power` explicitly selects power-wave renormalization;
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

Therefore `to_z_power` and `to_y_power` return owned matrices. Do not create a parameter-type hierarchy solely to wrap those matrices before real usage requires it. A later typed representation may be Yellow when multiple concrete workflows demonstrate that it improves correctness or usability without replacing the canonical model implicitly.

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
