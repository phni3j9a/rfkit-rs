# Public `Network` API Direction

This document records the human-approved direction for the first useful public RF-analysis surface of `rfkit-core`.

It resolves the public-API design checkpoint reached after the verified internal `Network` foundations cover power-wave conversion/renormalization, interpolation, matched connection/inner-connection, and explicit-grid composition. Autonomous development may implement bounded slices within this document without re-escalating the same design decision. Material deviations remain a human decision.

## Goals

The first public surface should:

- expose already-verified RF capability instead of continuing to accumulate private kernels;
- be Rust-native rather than mimic the scikit-rf object model mechanically;
- keep numerical and wave semantics explicit at the call site;
- avoid implicit frequency-grid selection, hidden renormalization, or convenience defaults that would freeze policy accidentally;
- stay small enough to evolve before a deliberate stabilization milestone.

## Stability policy

`rfkit-core` is currently a `0.x` crate. The first public RF-analysis methods are intentionally **provisional**, not a promise that names or signatures are frozen for `1.0`.

That does not make churn free. Public changes still need concrete user or correctness value, but the project should not postpone useful public capability merely to avoid ever changing a pre-1.0 API.

Autonomous work may add the approved surface below in bounded increments. It must escalate before:

- declaring the API stable or making a compatibility guarantee beyond normal `0.x` expectations;
- introducing a materially different public model or operation semantics;
- making a breaking change outside an already-approved bounded API increment;
- replacing explicit behavior below with implicit policy.

## Core model

Keep the existing owned, frequency-major core model for the first public slice:

- `Frequency` is the public frequency-axis type and uses hertz at the boundary;
- `Network` remains the canonical owned scattering-network type;
- `Network.s` is conceptually `(frequency, port_out, port_in)`;
- `Network.z0` is conceptually `(frequency, port)`;
- ports use zero-based `usize` indices initially;
- `ndarray` remains part of the initial public data boundary because it is already exposed by construction and accessors.

Do **not** introduce view lifetimes, generic storage traits, port-index newtypes, builders, parameter-container hierarchies, or alternate owned network representations merely to make the first API look more abstract. Add those only when concrete usage demonstrates value.

## Operation style

Prefer methods on `Network` when one network is the natural primary object. Operations should be pure from the caller's perspective: borrow inputs and return owned results rather than mutating a network in place.

The public API should not expose a parallel free-function surface that duplicates the same operations without a demonstrated reason.

Use `Frequency` rather than raw frequency slices at public operation boundaries.

## Wave semantics

The current verified conversion, renormalization, and matched-connection kernels use Kurokawa power-wave semantics. The `Network` model does not yet carry a wave-definition field.

For the first public slice, keep that fact explicit in **wave-sensitive public method names** rather than adding a speculative `WaveDefinition` field before a second convention is actually supported.

This leaves room for a later typed wave-definition design without silently declaring every `Network` to be power-wave data.

## Approved first public slice

The intended shape is:

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

The exact internal delegation remains an implementation detail. The semantic distinctions represented by these names are policy:

- `to_z_power` / `to_y_power` explicitly select the verified power-wave conversion convention;
- `renormalize_power` explicitly selects power-wave renormalization;
- `interpolate_cartesian_linear` does not establish a vague interpolation default that would later need reinterpretation;
- `connect_matched_power` requires the existing exactly matched real-positive junction contract and exact compatible frequency grids;
- `connect_matched_power_on_grid` requires an explicit caller-provided grid and performs interpolation-before-connection under the existing verified composition semantics;
- `inner_connect_matched_power` exposes the existing matched same-network elimination semantics.

Do not shorten these to broad names such as `connect`, `interpolate`, or `renormalize` until the library has enough supported semantics to justify what those names should mean.

## Frequency policy

Public connection APIs must not silently choose a frequency grid.

- `connect_matched_power` requires the inputs to satisfy the existing exact-grid contract.
- `connect_matched_power_on_grid` uses only the caller-provided `Frequency` target.
- No implicit intersection, subset selection, nearest-grid match, extrapolation, or automatic interpolation is part of the first public API.

A future convenience API may be considered only after its frequency-selection policy is explicitly approved.

## Parameter-return policy

`Network` remains S-parameter based. Z and Y are alternative parameter matrices produced from a network, not separate public `ZNetwork` / `YNetwork` types in the first slice.

Therefore `to_z_power` and `to_y_power` return owned matrices. Do not create a parameter-type hierarchy solely to wrap those matrices before real usage requires it.

Transformations that still produce an S-parameter network, such as renormalization, interpolation, and connection, return a new `Network`.

## Error policy

Use one crate-level public `Error` / `Result<T>` boundary for the first public surface.

The public error must remain structured and actionable rather than collapsing failures to strings. Preserve stable context such as invalid shapes/frequencies/ports, invalid or mismatched junction reference impedances, singularity, non-finite inputs/computation, and the failing high-level operation stage where it materially helps callers.

Internal kernel error enums may remain private and be translated at the public boundary. Mark the public error non-exhaustive before expanding it for the new operations so adding future error cases does not unnecessarily freeze the variant set.

Do not expose private implementation types merely to avoid writing a public error mapping.

## Autonomous-development checkpoint

Once the currently active explicit-grid composition work is merged (PR #50 or an equivalent successor), the default next horizon is **public usability of the verified `Network` core**, not another sequence of private kernels.

The Planner should prefer bounded vertical slices that expose the approved methods above with documentation and tests. It should not create additional private-only RF operations merely to postpone this public-API work unless fresh evidence shows a concrete correctness blocker or a prerequisite that materially prevents the approved surface.

Implementation should remain incremental; this document does not require all methods to land in one Issue or PR.

Escalate instead of autonomously deciding when work would require:

- a different public method family or naming semantics;
- implicit frequency alignment or automatic mismatch handling;
- another wave convention or a `WaveDefinition` model change;
- a generic/view/trait abstraction that materially changes the public model;
- a new canonical storage representation;
- a broad `connect`/`interpolate`/`renormalize` default whose semantics are not selected here;
- API stabilization or release policy.
