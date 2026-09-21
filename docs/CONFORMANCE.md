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

## Reporting

Eventually CI should publish a machine-generated coverage report such as:

```text
S<->Z conversion        10,000 / 10,000 pass
Renormalization          8,000 /  8,000 pass
Connect/cascade          5,000 /  5,000 pass
Touchstone round-trip    1,000 /  1,000 pass
```
