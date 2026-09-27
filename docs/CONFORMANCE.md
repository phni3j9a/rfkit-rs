# Conformance

scikit-rf is used as a reference oracle, not as the API specification.

Touchstone 2.0 single-ended S ingress is one format-boundary operation exposed
as `parse_touchstone_v2_0_s`. The document's `[Matrix Format]` selects Full,
Lower, or Upper representation; absent format means Full. Full preserves every
row-major complex value, while Lower/Upper expand the declared triangular
storage by plain transpose (never conjugation), including the diagonal. A
two-port document retains its required `[Two-Port Data Order]` directive and
both legal orders produce the same symmetric matrix under the ratified
Touchstone 2.0 semantics. The reader remains pure in-memory and does not infer
representation from scalar count.

The public parser is tested externally with asymmetric one-, two-, and
larger-port Full inputs, deterministic N=3 Lower and Upper inputs, RI/MA/DB
decoding, all frequency units, both explicit two-port orders, arbitrary
complete-record continuation, compact pair counts, continued per-port
`[Reference]` data and option-line override, later option-line ignore
semantics, exact declared frequency counts, `[End]`/trailing-content handling,
and recognized semantic comments. Structural/order/count, unsupported-subset,
numerical, and checked-size failures are tested independently, including
malformed declarations that must not allocate from enormous dimensions.

The canonical pinned fixtures are `touchstone_v2_0_s_full_three_port`,
`touchstone_v2_0_s_lower_three_port`, and
`touchstone_v2_0_s_upper_three_port`. They record scikit-rf `2.0.1` at commit
`bd651e923cac6020de49a096e1d7e9b5f949f884`, NumPy `2.5.1`, literal
independently authored source text, exact frequency/reference/text/shape and
metadata contracts, and strict `rtol=1e-12`, `atol=1e-12` comparison for only
decoded S. Lower and Upper use unequal real references `[25,50,75]`, N=3,
nonzero imaginary entries, and no random generation. The Rust oracle calls
only the new public API. The focused workflow explicitly inspects a loaded
triangular network, permutes ports, writes existing Full v2 text, and checks
readback of S, reference order, and frequencies without renormalization.

The pinned scikit-rf reader has a known two-port triangular `21_12` legacy
defect (the required Lower/`21_12` example is the narrow exception recorded by
Issue #112; the analogous Upper path is also not canonicalized): its source
performs a transpose before mirror fill, so an opposite-triangle value can be
read before it is initialized instead of producing the standards-defined
result. That implementation discrepancy is documented rather than
canonicalized; no uninitialized value is accepted as oracle data. The Rust
regression follows the ratified Touchstone 2.0 two-port semantics for both
`21_12` and `12_21`, and this exception is not a general scikit-rf
compatibility claim.

Touchstone 2.0 Full S egress is covered as Issue #104's bounded complementary
format boundary. Deterministic Rust tests exercise one-, two-, and larger-N
asymmetric networks, unequal/non-50 real references, more than four ports,
multiple samples, exact directive/order text, source immutability, malformed
serde-created shapes, finite binary64 and signed-zero round trips, every
frequency/S/reference domain rejection, later-sample single-port reference
changes, and unchanged v1 heterogeneous-reference rejection. The v2 layout is
one complete row-major record per frequency, with an explicit one-line
`[Reference]` vector and `[Two-Port Data Order] 12_21` only for two ports.
The focused `touchstone_v2_writer_workflow` parses unequal-reference v2 data,
permutes physical ports, writes without renormalization, and independently
checks reordered S/reference values after readback. The writer is still a
subset: MA/DB, Lower/Upper, mixed-mode, noise, vendor metadata, complex or
frequency-varying references, and filesystem I/O remain out of scope.

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
round trips. A CI oracle check invokes the actual public Rust v1 writer for a
legacy two-port and a multiport continuation case, and the actual public Rust
v2 writer for asymmetric unequal-reference two- and five-port cases; it feeds
all emitted texts to scikit-rf and compares frequency, S, and z0. The Python
checker independently verifies v2 header/reference/order layout and has tests
that detect S-pair ordering and reference-vector drift. v1.0 record boundaries,
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

Issue #86's historical direct finite physical-load operation is superseded by
Issue #114's generalized `Network::terminate_port_power` and
`PortLoad`. The canonical finite fixture
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
only `terminate_port_power`'s reduced S against the fixture tolerance, wrapping
each finite fixture value as `PortLoad::ImpedanceOhm`. It derives the expected
survivor references by slicing the source reference array rather than treating
output z0 as a floating oracle. Core edge coverage separately handles
negative/zero load resistance, complex and negative-real source references,
d=0 with nonzero denominator, exact singularity, finite near-singularity,
malformed shapes, non-finite inputs, and arithmetic failure.

Issue #114 adds the pinned mixed-boundary fixture
`power_wave_terminate_port_mixed_open_five_port_complex_z0.json`. It is an
independently generated, asymmetric five-port/four-frequency case with middle
port `2`, seed `20260963`, unequal complex frequency-dependent positive-real
references, and tagged physical loads `[Open, 31+7j, Open, -17+4j]` ohm. The
fixture records `operation=terminate_port_power`, explicit
`PortLoad::Open` tags (never an infinity/NaN sentinel), source/output order,
survivor references, scikit-rf `2.0.1` commit
`bd651e923cac6020de49a096e1d7e9b5f949f884`, NumPy `2.5.1`, the observed raw
and restored wave definitions, and strict output-only `rtol=1e-12`,
`atol=1e-12`. Its expected response is generated via public `z2s`/Network
load construction followed by one public `connect`; complex-reference output
is explicitly restored and checked as power waves before extracting S. All
metadata, source arrays, tagged loads, frequencies, references, shapes, and
ordering are exact contract fields. The new oracle registration and Rust test
must compare only the reduced S within that tolerance.

The focused Touchstone workflow in
`crates/rfkit-touchstone/tests/public_termination_workflow.rs` and executable
`crates/rfkit-touchstone/examples/terminate_port_touchstone.rs` parse an
asymmetric five-port v1.0 S/RI/Hz input, apply a mixed
`[Open, 38+12j, 0]` profile (exact open, finite complex load, and ideal short),
check the reduced response independently from the direct physical boundary,
verify `I_k=0` for the open sample plus `V_k=-Z_L I_k` for finite samples,
check source/load immutability, and write/read the four-port result with
unchanged common 50-ohm survivor references. No writer-side renormalization,
open sentinel, load excitation, or mixed-mode/file-format extension is
involved. The sole public termination method is the generalized operation;
there is no open-only or finite-only companion.

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

Issue #98 adds the sampled `Network::max_singular_value_power` diagnostic. It
returns the dimensionless scalar `sigma_max(S)` from a full dense complex SVD
for every source sample, so its square is the maximum reflected/incident
power-wave ratio. Core coverage includes analytical one-port magnitude,
zero/rank-deficient/repeated-singular-value matrices, ideal and scaled
unitaries, the coherent `[[.6,.6],[.6,.6]]` witness, the nonnormal
`[[0,2],[0,0]]` witness, coupled non-reciprocal N-port data, larger finite
scales, exact sample-order/source immutability, port permutation and unitary
coordinate invariants, unequal complex/frequency-dependent positive-real
references, and malformed serde-created axes/shapes. Finite frequency labels
including negative, duplicate, descending, and signed-zero values are valid;
non-finite labels/data and zero or negative-real references are operation-
specific errors. The public method never classifies around one, clips, forms
`SᴴS`, or falls back to a column/Frobenius norm. Its private nalgebra SVD
adapter uses `5*f64::EPSILON` and a finite 10,000-iteration per-sample budget;
non-convergence and non-finite solver output are structured failures.

The canonical fixture
`max_singular_value_power_four_port_complex_z0.json` is independently seeded
with NumPy `default_rng` seed `20260957`: four coupled, non-reciprocal
four-port samples use fixed binary-exact sample factors `[1.0, 2.0, 4.5, 5.0]`
applied to the seeded raw S stack and use unequal complex, frequency-dependent
positive-real references. Input construction does not call SVD, so exact source
arrays are independent of platform LAPACK details. Expected `sigma_max` values
come only from public NumPy `2.5.1`
`numpy.linalg.svd(..., compute_uv=False)[:,0]`. Pinned scikit-rf `2.0.1` at
commit `bd651e923cac6020de49a096e1d7e9b5f949f884` is called through public
`Network.is_passive(tol=1e-12)` on one-sample Networks solely for a limited
Boolean comparison; all samples are comfortably away from the sigma=1
boundary. `crates/rfkit-core/tests/oracle_max_singular_value.rs` rechecks
versions, seed, shapes, input arrays, references, classes, Boolean evidence,
and source immutability before calling only the public Rust method. Only
`data.sigma_max` uses output-only `rtol=1e-12`, `atol=1e-12`; metadata, source
data, and `data.is_passive` remain exact canonical contract fields. Exact
unitary and one-port semantics are local Rust evidence rather than pinned
Boolean compatibility claims.

The focused Touchstone workflow in
`crates/rfkit-touchstone/tests/public_max_singular_value_workflow.rs` and
`crates/rfkit-touchstone/examples/max_singular_value_touchstone.rs` parses an
independently authored four-port v1.0 RI/Hz sweep with a passive diagonal
sample and a coherent amplifying sample, then prints aligned scalar and
squared-power values without adding a verdict API.

Issue #94 adds the borrowing `Network::inverse_cascade_power` operation for
ordered even-port power-wave cascades. Core coverage includes analytic and
non-reciprocal two-port cases, an ideal through, complex/signed-reference V/I
wave reversal, a genuinely cross-mode-coupled asymmetric four-port, a larger
six-port double inverse, exact metadata/source preservation, and finite
near-singular transmission acceptance. The four-port invariant workflow
connects each group pair through the existing public `connect_power` and
`inner_connect_power` methods, then removes both left-first and
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
with `connect_power`, removes fixtures in both orders, explicitly
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
agreement with `connect_power` on the shared well-conditioned domain,
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
`Network::connect_direct_power`.
Superseded by #109

Its canonical fixture
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
only the public `connect_power` method. It compares every connected S
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
`crates/rfkit-touchstone/tests/public_connect_power_workflow.rs` and
`crates/rfkit-touchstone/examples/connect_power_touchstone.rs` parse
separate 3-port and 4-port v1.0 inputs at different common references, connect
without pre-renormalizing either source, independently solve the physical
V/I boundary, and assert both input snapshots are unchanged. They then call
`renormalize_direct_power` explicitly to a caller-chosen common positive-real
writer reference before writing and reading the five-port output. The writer
is not asked to repair references, and the workflow makes no Touchstone v2,
mixed-mode, or automatic-renormalization promise.

Issue #90 adds `Network::inner_connect_direct_power` for the same direct
physical V/I junction inside one network.
Superseded by #109

The historical implementation in
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
`crates/rfkit-touchstone/tests/public_inner_connect_power_workflow.rs`
and
`crates/rfkit-touchstone/examples/inner_connect_power_touchstone.rs`
parses a multiport input, directly renormalizes into unequal complex selected
references, independently checks voltage continuity and current conservation
for nonzero external excitation, closes the pair, then directly renormalizes
survivors to one common positive-real writer reference before write/read. The
writer remains unchanged and never repairs heterogeneous references.

## Consolidated physical power-wave connection (Issue #109)

The legacy matched and Issue #88 direct inter-network fixture families, plus
the legacy matched and Issue #90 direct inner fixture families, are now
exercised through the consolidated public methods. The matched inter-network fixture
`power_wave_connect_matched_three_to_four_port_real_frequency_dependent_z0.json`
and the direct complex-reference fixture
`power_wave_connect_direct_three_to_four_port_complex_z0.json` both call
`Network::connect_power`; their historical metadata operation strings and
fixture filenames remain unchanged so the pinned generation history stays
auditable. The matched inner fixture calls `Network::inner_connect_power`, as
does the direct inner fixture
`power_wave_inner_connect_direct_five_port_complex_z0.json` for its
complex-reference domain.

The explicit-grid fixture
`power_wave_connect_matched_explicit_grid_three_to_four_port_complex_z0.json`
no longer relies on a private composition adapter. Its conformance case calls
`interpolate_cartesian_linear(&target)` independently on A and B, then calls
`connect_power` on the two interpolated networks. This preserves staged
interpolation-versus-connection diagnostics and makes the interpolation kind
caller-visible.

The selector tests cover the union of the retired physical domains: finite
positive exactly equal selected references use one exact dyadic BigRational
matched Schur evaluator (including zero-real external survivor references and
Issue #44 extreme arithmetic), while unequal, complex, and negative-real
nonzero-real selected references use the direct physical V/I kernel.
Common-domain comparisons use independent physical reconstruction within the
existing fixture tolerances; exact singular, near-singular, survivor-order,
malformed-serde, and Touchstone workflows remain structured and deterministic.
Public regressions A/B/C/D and the huge-internal-block case cover exact
determinants/adjugates, final-Q cancellation, minimum-subnormal and
round-to-even conversion, and a genuinely unrepresentable final result as a
structured non-finite error. The old inner block `S_ii=[[-11,-54],[-2,-15]]`
is documented as a numerical defect: native elimination can falsely report a
nonzero `2^-49` pivot although the exact determinant is zero, so the
consolidated method correctly reports singularity. No case retries through
the other public evaluation. The current executable paths are
`crates/rfkit-touchstone/tests/public_connect_power_workflow.rs`,
`crates/rfkit-touchstone/examples/connect_power_touchstone.rs`,
`crates/rfkit-touchstone/tests/public_inner_connect_power_workflow.rs`, and
`crates/rfkit-touchstone/examples/inner_connect_power_touchstone.rs`.

## Adjacent-interval group-delay conformance (Issue #100)

Issue #100 adds `Network::group_delay_secant_power(port_out, port_in)`, a
selected-coordinate, adjacent-interval seconds estimate.  The Rust kernel
uses principal `atan2` phases, shortest local increments, and each actual
`f[k+1]-f[k]` aperture.  Its output length is `nfreq-1` and value `k` is
attached to `[f[k], f[k+1]]`; this is intentionally different from pinned
scikit-rf 2.0.1 `Network.group_delay`, which applies `gradient` to
`s_rad_unwrap` and returns one sample-aligned value per frequency.  The
canonical differential family therefore calls only public
`Network.s_rad_unwrap` and explicitly differences adjacent intervals; it does
not claim the two APIs are shape- or estimator-equivalent.

`crates/rfkit-core/tests/public_group_delay.rs` covers one-, two-, and larger
N-port traces; uniform and nonuniform grids; constant and quadratic phase;
positive, negative, and zero delay; both branch directions; selected ports;
varying amplitudes and constant phase offsets; complex, unequal,
frequency-dependent, non-50-ohm, and signed references; source immutability;
port permutation covariance; singular S; huge/subnormal selected entries;
signed-zero phase axes; exact +/-pi rejection and nearby acceptance; selected
versus unselected zeros; all malformed serde shapes, ports, grids, references,
non-finite values, arithmetic failures, very large apertures, subnormal
apertures with both phase signs, and tiny intervals; and an explicit
undersampling alias witness.  The shortest-phase rule has no magnitude floor,
smoothing, fitting, or invented inverse.

The pinned fixture
`tools/oracle/fixtures/group_delay_secant_power_three_port_branch_crossing.json`
uses seed `20260958`, frequencies `[0,31,80,143,225,320,429]` MHz, an
asymmetric varying-amplitude S31 branch crossing, and complex
frequency-dependent references.  It records scikit-rf `2.0.1` at commit
`bd651e923cac6020de49a096e1d7e9b5f949f884`, NumPy `2.5.1`, public
`s_rad_unwrap` plus explicit interval differencing, and output-only
`rtol=1e-12`, `atol=1e-21`.  Both
`tools/oracle/test_generate_oracle.py` and
`crates/rfkit-core/tests/oracle_group_delay.rs` enforce exact metadata/input
contracts before comparing only seconds output.

The reproducible checks are:

```text
/home/server/.cache/rfkit-rs-oracle-venv/bin/python tools/oracle/generate_oracle.py check --case group_delay_secant_power_three_port_branch_crossing
cd tools/oracle && /home/server/.cache/rfkit-rs-oracle-venv/bin/python -m unittest test_generate_oracle.py
cargo test -p rfkit-core --test public_group_delay --test oracle_group_delay
cargo test -p rfkit-touchstone --test public_group_delay_workflow
cargo run -p rfkit-touchstone --example group_delay_secant_touchstone
```

The focused Touchstone path loads a v1.0 S/RI/Hz two-port, selects S21, and
checks the displayed interval bounds against an analytical 2 ns line delay.
Exact half-turns and exact selected zeros are actionable structured failures;
other phase undersampling remains an explicit sampling limitation rather than
an automatic detector.

## Reporting

Eventually CI should publish a machine-generated coverage report such as:

```text
S<->Z conversion        10,000 / 10,000 pass
Renormalization          8,000 /  8,000 pass
Connect/cascade          5,000 /  5,000 pass
Touchstone round-trip    1,000 /  1,000 pass
```
