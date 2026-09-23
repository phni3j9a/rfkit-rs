# Conformance

scikit-rf is used as a reference oracle, not as the API specification.

## Definition of "implemented"

A numerical operation is considered implemented only when:

1. its mathematical behavior is documented,
2. deterministic Rust unit tests pass,
3. relevant RF invariants/property tests pass,
4. differential comparison against a pinned scikit-rf version passes for representative cases,
5. tolerances are justified rather than widened to hide failures.

## Case dimensions

The conformance suite should deliberately vary:

- 1, 2, 4, 8+ ports
- scalar, per-port, and frequency-dependent Z0
- real and complex Z0
- passive and active networks
- reciprocal and non-reciprocal networks
- well-conditioned and near-singular cases
- DC and high-frequency boundaries where relevant

Touchstone v1.0 S parsing and the explicit S/RI/Hz writer are covered by
independently authored text/Network contract tests. The mutually supported
oracle slice records the pinned `scikit-rf==2.0.1` parser, `numpy==2.5.1`,
deterministic input text, port count, and mixed relative/absolute tolerance.
Rust compares parsed frequency, S, and expanded real scalar `z0` values;
writer tests independently assert option/ordering/continuation layout,
full-domain validation, unchanged input, and exact writer→reader numeric
round trips. A CI oracle check invokes the actual public Rust writer for a
legacy two-port and a multiport continuation case, then feeds those emitted
texts to scikit-rf and compares frequency, S, and z0. v1.0 record boundaries,
unsupported v2/noise/vendor extensions, and deliberate domain rejections are
specified directly in Rust rather than inferred from permissive oracle
behavior.

The parameter-ingress constructors are covered by the existing pinned
scikit-rf 2.0.1 Z→S and Y→S fixture families, including the representative
complex per-port/frequency-dependent three-port case and a near-singular
three-port Z→S case. Direct `from_y_direct_power` additionally uses the pinned
`power_wave_y_to_s_three_port_rank_deficient_complex_z0` fixture: three
frequency samples, a non-reciprocal rank-deficient three-port Y, and complex
per-port/frequency-dependent references. External-crate tests additionally
cover analytical one-port values, a floating series element and ideal open,
asymmetric multi-frequency N-port data, negative-real and complex references,
exact frequency/reference/order preservation, Z/Y round trips where the
composed domain is invertible, zero-Z success, and explicit zero/singular-Y
rejection. Boundary tests exercise empty or mismatched frequency axes
(including a serde-deserialized empty `Frequency`), zero-port and nonsquare
matrices, malformed z0, non-finite values, exact reference domain failures,
finite-input overflow, finite near-singular direct systems, exact direct
singularity, and separate composed Y→Z versus direct Y→S stage errors. The
constructors' pointwise frequency-label policy is intentionally looser than
the Touchstone writer's format policy and is tested as such.

Issue #78 adds the complementary direct `Network::to_y_direct_power` path.
Its deterministic coverage includes ideal-open and floating-series networks
whose `I-S` is singular but whose direct `A Y = B` system is nonsingular,
including the public Touchstone writer→reader workflow. The floating-series
case checks the analytical `0.01[[1,-1],[-1,1]]` siemens result directly after
the reread, while the same network must still return a structured
`ConversionStage::SToZ` singular error through the existing composed
`to_y_power` method. Direct and composed extraction are compared on their
well-conditioned common domain; this does not broaden renormalization or any
other conversion domain.

Issue #80 adds the additive `Network::renormalize_direct_power` path. Its
deterministic tests cover ideal open and complex-reference ideal short cases,
floating series Y at unequal complex source references, asymmetric
multi-frequency N-port data, non-50-ohm and negative-real references, exact
and finite near-singular direct systems, A→B→A and A→B→C versus A→C
round-trips, input immutability, frequency/port/reference preservation, and
an independent V/I wave-relation check. Existing pinned renormalization
fixtures are reused on their shared non-singular domain, including the
larger N-port and complex per-port/frequency-dependent cases; the direct and
composed paths are compared without changing fixture tolerances. Boundary
tests cover malformed serde axes/shapes, zero ports, invalid old/new
references, non-finite S, finite-input overflow, equal-reference ideal-open
validation, and exact direct-system singularity. The public direct error
vocabulary identifies the operation and source versus target reference where
applicable, without pretending that a direct failure occurred in S→Z or Z→S.

The executable and external-crate Touchstone workflow now constructs the
floating singular-Y model with unequal complex references, directly
renormalizes it to a common positive-real reference, writes and rereads
Touchstone, and extracts the analytical physical Y through the direct path.
The existing composed renormalization and conversion paths remain separately
regression-tested with their documented singular-stage behavior.

Issue #82 adds the pure `Network::permute_ports` reindexing operation. The
external Touchstone workflow test in
`crates/rfkit-touchstone/tests/public_permutation_workflow.rs` parses an
asymmetric three-port, two-frequency v1.0 S/RI/Hz input, applies the explicit
non-involutive `[2, 0, 1]` new-to-old mapping, and independently spells out the
expected S values after both row and column moves and the expected port-aligned
z0 values. It also snapshots and compares the source network, fixes the exact
writer text, and reparses that text to verify the exported physical order.
The executable counterpart is
`crates/rfkit-touchstone/examples/permute_ports_touchstone.rs`.

The Touchstone fixture uses the writer-supported common 50-ohm reference, so
the workflow observes z0 alignment through an independently expected common
vector while S proves both axes and the non-involutive direction. Core-level
permutation coverage additionally exercises unequal complex and
frequency-dependent references, identity/swap/cycle and inverse round trips,
component preservation, malformed serde shapes, and structured mapping
diagnostics. These checks characterize permutation as exact coordinate copying,
not a wave conversion or renormalization, and leave the writer's separate
common finite positive-real validation unchanged.

The pinned direct S→Y differential case is independently specified as a
non-reciprocal three-port with singular `I-S` and nonsingular direct `A`, plus
complex per-port/frequency-dependent references. Its expected Y is generated
only by public `skrf.network.s2y(..., s_def="power")` from pinned
`scikit-rf==2.0.1` (`bd651e923cac6020de49a096e1d7e9b5f949f884`), with the local
seed `20260946` and fixture name
`power_wave_s_to_y_three_port_singular_i_minus_s_complex_z0.json`; the fixture
metadata records operation, units, versions, and strict tolerances. The
input is not derived from an opposite-direction fixture or a round trip.

Issue #84 adds the equal-pair mixed-mode power-wave slice. The canonical
oracle registers two small, direction-specific fixtures:

- `mixed_mode_forward_five_port_complex_z0.json` is an independently authored
  asymmetric five-port, three-frequency single-ended input with `p=2`,
  adjacent `(0+,1-)` and `(2+,3-)` pairs, two distinct complex equal pair
  references at each frequency, and one unpaired complex reference. Its
  expected output is generated through pinned public `Network.se2gmm(p=2,
  s_def="power")`.
- `mixed_mode_inverse_five_port_complex_z0.json` has an independently seeded
  modal S input and authored natural modal references. It is not the forward
  fixture's output; generation calls pinned public `Network.gmm2se` with an
  explicit adjacent `(frequency,4)` target `z0_se` array. This prevents a
  forward-only or default-reference shortcut from certifying the inverse.

Both fixtures record direction, coordinate order/polarity, pair count,
unpaired port, source/target references, input recipes, seeds, shapes, and
the pinned scikit-rf `2.0.1` commit
`bd651e923cac6020de49a096e1d7e9b5f949f884` with NumPy `2.5.1`. The canonical
checker removes only `data.s` from its exact contract projection. Thus
metadata, frequencies, S inputs, references, target shapes, and all other
fields must match exact canonical JSON; only floating S output uses the
recorded `rtol=1e-12`, `atol=1e-12` bound. Python tests defend registration,
independence, public direction-specific calls, reference relationships, and
output/contract drift. `crates/rfkit-core/tests/oracle_mixed_mode.rs` repeats
the strict metadata/input/reference checks and compares each public Rust
method against its corresponding fixture S output.

The end-to-end Touchstone test and executable parse an asymmetric five-port
v1.0 S/RI/Hz input, apply explicit physical permutation `[2,0,3,1,4]`, check
differential/common and mode-conversion responses against an independently
written `U S U^T` sum, apply the inverse and `[1,3,0,2,4]` permutation, then
write/read with the existing single-ended writer. The source uses common
positive-real 50-ohm references for the format boundary. No mixed-mode
Touchstone extension or implicit writer renormalization is involved.

The Rust methods are a REWRITE of the repository's explicit coordinate and
Kurokawa equations. Negative-real and complex references remain covered by
core mixed-mode boundary/invariant tests, but the scikit-rf differential claim
is limited to the pinned positive-real-compatible fixture domain.

Issue #86 adds the direct finite physical-load operation
`Network::terminate_port_impedance_power`. The canonical fixture
`power_wave_terminate_port_impedance_five_port_complex_z0.json` is an
independently generated asymmetric five-port, three-frequency case with the
middle source port `2` removed. It uses unequal, frequency-dependent complex
source references whose real parts are strictly positive and explicit finite
loads `[0, 38+12j, 73-9j]` ohm, including an ideal short. The expected reduced
S is produced only through pinned public scikit-rf `skrf.network.connect` with
a one-port load constructed through public `skrf.network.z2s(...,
s_def="power")` and `Network` APIs. The fixture records operation, selected
port, ohm units, source/output order, seed `20260950`, scikit-rf `2.0.1`
commit `bd651e923cac6020de49a096e1d7e9b5f949f884`, NumPy `2.5.1`, and the
strict `rtol=1e-12`, `atol=1e-12` policy. The canonical Python checker removes
only `data.s_terminated` from the exact contract projection; source S,
frequency, references, loads, survivor references, shapes, and metadata must
match exactly. Python tests defend registration, public z2s/connect call
reconstruction, finite-load/short metadata, and input/output drift.

The external Rust oracle test
`crates/rfkit-core/tests/oracle_termination.rs` repeats those strict metadata,
input, reference, load, frequency, and survivor-order checks, then compares
only the public method's reduced S against the fixture tolerance. It derives
the expected survivor references by slicing the source reference array rather
than treating output z0 as a floating oracle. Core edge coverage separately
handles negative/zero load resistance, complex and negative-real source
references, d=0 with nonzero denominator, exact singularity, finite
near-singularity, malformed shapes, non-finite inputs, and arithmetic failure.

The focused Touchstone workflow in
`crates/rfkit-touchstone/tests/public_termination_workflow.rs` and executable
`crates/rfkit-touchstone/examples/terminate_port_touchstone.rs` parse an
asymmetric five-port v1.0 S/RI/Hz input, apply the three finite loads, check
the reduced response independently from the direct boundary equation, and
write/read the four-port result with unchanged common 50-ohm survivor
references. No writer-side renormalization, open sentinel, load excitation,
or mixed-mode/file-format extension is involved.

Issue #92 adds the sampled two-port power-wave stability operation
`Network::two_port_stability_power`. Core tests cover direct analytic values
including `K=2.6, delta=-0.2`, the ideal-through boundary, `K>1` with
`|delta|>1`, negative and sub-unity finite K, complex non-reciprocal and
singular S, mixed defined/undefined sweeps, tiny nonzero transmission,
malformed serde-created shapes, finite/non-finite labels and data, strict
positive-real reference validation, input immutability, port exchange, and
representative positive-real power renormalization. Exactly zero `S12` or
`S21` produces `None` only after finite determinant arithmetic; nonzero
magnitude-product underflow/overflow remains a structured arithmetic error.

The canonical fixture
`two_port_stability_power_four_frequency.json` has four deterministic
two-port samples from an independent base S stack plus a seeded NumPy
`default_rng` complex perturbation (seed `20260954`, scale `1e-3`), with one
passive and three active/non-passive S matrices established by true largest
singular values. It also has unequal real/complex positive-real references,
exact input/metadata contracts, and finite nonzero transmissions. Pinned scikit-rf `2.0.1` (commit
`bd651e923cac6020de49a096e1d7e9b5f949f884`) public `Network.stability`
generates K, while NumPy `2.5.1` `linalg.det` independently generates delta;
the local Rust formula is a REWRITE. Only `data.delta` and `data.rollet_k`
use strict `rtol=1e-12`, `atol=1e-12` comparison. The Python generator tests
protect registration, public-output sources, numeric-only tolerance, and
metadata/input drift; `crates/rfkit-core/tests/oracle_two_port_stability.rs`
rechecks metadata, shapes, source arrays, determinant relation, active/passive
representatives, and source immutability before comparing the public method.

The focused Touchstone workflow in
`crates/rfkit-touchstone/tests/public_two_port_stability_workflow.rs` and
`crates/rfkit-touchstone/examples/two_port_stability_touchstone.rs` parses a
two-port v1.0 sweep, independently checks aligned K and `|delta|`, displays an
explicit undefined K, and leaves the source unchanged. The familiar linear
two-port interpretation requires `K>1` and `|delta|<1` with its usual
auxiliary/proviso conditions; sampled external S cannot certify internal
poles, unsampled frequencies, nonlinear/large-signal behavior, or overall
circuit stability. No verdict booleans, circles, μ factors, gain optimization,
or tolerance classifications are part of this slice.

Issue #94 adds the borrowing `Network::inverse_cascade_power` operation for
ordered even-port power-wave cascades. Core coverage includes analytic and
non-reciprocal two-port cases, an ideal through, complex/signed-reference V/I
wave reversal, a genuinely cross-mode-coupled asymmetric four-port, a larger
six-port double inverse, exact metadata/source preservation, and finite
near-singular transmission acceptance. The four-port invariant workflow
connects each group pair through the existing public `connect_direct_power` and
`inner_connect_direct_power` methods, then removes both left-first and
right-first fixtures with the same helpers; it checks the coupled DUT response
and exact output references rather than only testing a relational inverse.
Structured tests distinguish malformed zero-port/cardinality/z0 shapes,
non-finite frequency/S/z0 data, zero-real references, full-S versus forward
and reverse transmission singularity, and finite-input arithmetic overflow.

The canonical fixture
`power_wave_inverse_cascade_four_port_real_unequal_z0.json` is independently
seeded with NumPy `default_rng` seed `20260955`, has three frequency samples and
unequal real-positive frequency-dependent per-port references, and fixes input
groups as `[left_0,left_1,right_0,right_1]` with output `[old_right,old_left]`.
Pinned public scikit-rf `Network.inv` 2.0.1 (commit
`bd651e923cac6020de49a096e1d7e9b5f949f884`) supplies expected S; NumPy `2.5.1`
independently checks `P @ solve(S,I) @ P`. Only output S uses strict
`rtol=1e-12`, `atol=1e-12`; frequencies, source S/z0, swapped conjugate
references, shapes, group metadata, and recipe fields are exact. The generator
keeps the independent solve as a `<=1e-12` guard but does not serialize its
runtime residual or determinant magnitudes as canonical metadata, so BLAS
rounding does not change the contract. `crates/rfkit-core/tests/oracle_inverse_cascade.rs`
calls only the public Rust method, while Python generator/checker tests reject
registration, input/group, and metadata drift.

The focused external workflow in
`crates/rfkit-touchstone/tests/public_inverse_cascade_workflow.rs` and
`crates/rfkit-touchstone/examples/inverse_cascade_touchstone.rs` parses
independent two-port LEFT/DUT/RIGHT sweeps, constructs the measured cascade
with `connect_direct_power`, removes fixtures in both orders, explicitly
renormalizes to common 50 Ω, and verifies writer/readback. No implicit
orientation, reference conversion, noise calibration, or broad scikit-rf
compatibility claim is introduced; inverse networks remain mathematical
removal operators that may be active or noncausal.

Issue #96 adds `Network::cascade_direct_power` for one simultaneous physical
group junction between equal ordered even-port inputs.  Its internal
coordinates are `[A.right..., B.left...]`, its external coordinates are
`[A.left..., B.right...]`, and it solves
`(C + D S_ii) X = -D S_ie` before evaluating `S_ee + S_ei X`.  The full
within-group `S_ii` blocks remain present; the implementation does not wrap
sequential one-port connections or convert through Z/Y/transfer parameters.
The public domain retains finite complex and signed negative-real references
with nonzero real parts, exact shared frequency grids, exact surviving
references, exact-pivot singularity, and checked finite arithmetic.  Full S and
directional transmission blocks are not independently required to be
invertible.

Core tests in `crates/rfkit-core/tests/public_cascade.rs` cover two-port
agreement with `connect_direct_power` on the shared well-conditioned domain,
genuinely coupled four-port and inverse composition/removal, exact metadata
and source immutability, complex/signed-reference physical behavior, zero and
larger even-port inputs, the independently stated partial-singular but joint-
nonsingular witness, exact singular and finite near-singular systems,
malformed serde shapes, odd/zero/unequal cardinality, non-finite data/grids,
zero-real references, and checked arithmetic failures.  The witness also
asserts that the former single-junction method continues to reject its
singular intermediate system.

The canonical fixture
`power_wave_cascade_direct_four_port_complex_z0.json` uses seed `20260956`,
three frequencies, two coupled four-port inputs, and unequal complex
frequency-dependent per-port references with strictly positive real parts.
Pinned public scikit-rf `Network.cascade` 2.0.1 at commit
`bd651e923cac6020de49a096e1d7e9b5f949f884` supplies the expected output on
this well-conditioned shared domain.  The generator explicitly calls
`output.renormalize(output.z0, s_def="power")` before extracting the output
S.  As a generation-time guard, it forms the actual Kurokawa `C`, `D`, `S_ii`,
and `S_ie` blocks and requires finite joint determinants with `abs(det) >=
1e-3`, condition number `<= 1e4`, and finite relative solve residual `<=
1e-12` at each frequency; observed diagnostics are not serialized.
`crates/rfkit-core/tests/oracle_cascade_direct.rs` calls only the public
Rust method and checks exact input/group/order/reference metadata plus
output-only `rtol=1e-12`, `atol=1e-12`; `tools/oracle/test_generate_oracle.py`
rejects output, input, group, metadata, and runtime-diagnostic drift.  The
generator/checker registration is verified with:

```text
/home/server/.cache/rfkit-rs-oracle-venv/bin/python tools/oracle/generate_oracle.py check --case power_wave_cascade_direct_four_port_complex_z0
```

The focused external workflow in
`crates/rfkit-touchstone/tests/public_cascade_direct_workflow.rs` and
`crates/rfkit-touchstone/examples/cascade_direct_power_touchstone.rs` parses
independent four-port Touchstone inputs, performs simultaneous cascade and
inverse removal, explicitly renormalizes to the writer's common positive-real
reference, and verifies write/read.  No interpolation, writer repair,
automatic calibration, noise propagation, or broad scikit-rf compatibility is
claimed; complex/negative-real and partial-singularity cases remain local
equation/invariant evidence.

Issue #88 adds the direct physical-junction operation
`Network::connect_direct_power`. Its canonical fixture
`power_wave_connect_direct_three_to_four_port_complex_z0.json` is an
independently generated asymmetric, non-reciprocal three-port A plus four-port
B case at three frequencies, with A[1] connected to B[2]. The single RNG seed
is `20260951`; every reference is complex, frequency-dependent, and has a
strictly positive real part, and the selected A/B references are unequal. The
expected S is produced only by one pinned public
`skrf.network.connect` call from scikit-rf `2.0.1` at commit
`bd651e923cac6020de49a096e1d7e9b5f949f884`; the public Rust method is an
independent physical V/I junction rewrite. The 3+4 shape avoids scikit-rf's
special two-port insertion convention. A survivors `[0,2]` precede B
survivors `[0,1,3]`, and survivor references are derived in Rust by slicing
the two input arrays rather than treating output z0 as a floating oracle.

`crates/rfkit-core/tests/oracle_direct_connection.rs` checks the fixture
schema, pinned versions/seed, exact grids, source S inputs, complex references,
selected ports, shapes, survivor mapping, and output references before calling
only the public `connect_direct_power` method. It compares every connected S
component under exactly `rtol=1e-12`, `atol=1e-12`; no private kernel or matched
connection is used. It also snapshots both inputs to verify borrowing and
immutability. Python oracle tests defend one public connect call, strict
output-only checker tolerance, metadata/input drift rejection, and a focused
real unequal-positive-reference finite check without a second canonical
fixture.

The direct operation requires nonempty matching finite exact frequency grids,
finite square S/z0 data, valid selected ports, nonzero real references, and at
least one survivor. It preserves A-then-B order and exact surviving references;
it does not sort, intersect, interpolate, broadcast, renormalize implicitly,
or promise a broad scikit-rf domain. Exact evaluated singularity of the
two-coordinate physical junction is a structured error; finite nonsingular
near-singular cases remain in-domain without an arbitrary condition cutoff.
Complex and negative-real reference behavior is covered by local physical and
invariant tests, while the pinned differential claim is limited to the
canonical positive-real-part fixture domain.

The focused Touchstone test and executable in
`crates/rfkit-touchstone/tests/public_direct_connection_workflow.rs` and
`crates/rfkit-touchstone/examples/connect_direct_power_touchstone.rs` parse
separate 3-port and 4-port v1.0 inputs at different common references, connect
without pre-renormalizing either source, independently solve the physical
V/I boundary, and assert both input snapshots are unchanged. They then call
`renormalize_direct_power` explicitly to a caller-chosen common positive-real
writer reference before writing and reading the five-port output. The writer
is not asked to repair references, and the workflow makes no Touchstone v2,
mixed-mode, or automatic-renormalization promise.

Issue #90 adds `Network::inner_connect_direct_power` for the same direct
physical V/I junction inside one network. The implementation in
`crates/rfkit-core/src/direct_connection.rs`, exposed through
`crates/rfkit-core/src/lib.rs`, evaluates the complete selected `S_ii` block,
including both off-diagonal couplings, through
`(C + D*S_ii)T = -D*S_ie`; it does not reuse the matched inner-connect
shortcut. The contract is the finite, nonzero-real Kurokawa reference domain,
including unequal/equal complex, frequency-dependent, and negative-real
references. It preserves finite frequency labels (including signed zero),
survivor order, and references exactly, rejects malformed shapes before
indexing, treats only exact evaluated zero pivots as singular, and has no
near-singular condition cutoff or hidden conversion/renormalization.

The canonical fixture generated by `tools/oracle/generate_oracle.py`,
`power_wave_inner_connect_direct_five_port_complex_z0.json`, is an independent
asymmetric non-reciprocal five-port, three-frequency case with selected ports
1 and 3 and seed `20260952`. It records the source data, grid, references,
survivor order, wave definitions, and the pinned scikit-rf `2.0.1` commit
`bd651e923cac6020de49a096e1d7e9b5f949f884` with NumPy `2.5.1`. The oracle
path calls public `skrf.network.innerconnect` once, then explicitly restores
the returned pseudo-wave network with
`result.renormalize(result.z0, s_def="power")` before extracting expected S.
Raw pseudo output is a deliberate trap and is not a power-wave oracle. The
checker and `crates/rfkit-core/tests/oracle_direct_inner_connection.rs` keep all
metadata/input/reference/order fields exact and compare only restored S under
`rtol=1e-12`, `atol=1e-12`; no broad scikit-rf compatibility claim is made.

`crates/rfkit-core/tests/public_direct_inner_connection.rs` independently
covers the public operation's physical reconstruction, full internal block,
3-port→1-port and larger N-port→N-2 reductions, selected-port permutations,
survivor/reference/frequency preservation, source immutability, exact
singularity versus finite near-singularity, signed negative-real references,
malformed serde shapes, non-finite inputs, and checked arithmetic failures.
The focused Touchstone workflow in
`crates/rfkit-touchstone/tests/public_direct_inner_connection_workflow.rs`
and
`crates/rfkit-touchstone/examples/inner_connect_direct_power_touchstone.rs`
parses a multiport input, directly renormalizes into unequal complex selected
references, independently checks voltage continuity and current conservation
for nonzero external excitation, closes the pair, then directly renormalizes
survivors to one common positive-real writer reference before write/read. The
writer remains unchanged and never repairs heterogeneous references.

## Reporting

Eventually CI should publish a machine-generated coverage report such as:

```text
S<->Z conversion        10,000 / 10,000 pass
Renormalization          8,000 /  8,000 pass
Connect/cascade          5,000 /  5,000 pass
Touchstone round-trip    1,000 /  1,000 pass
```
