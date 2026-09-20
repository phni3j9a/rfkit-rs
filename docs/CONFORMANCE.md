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

The pinned direct S→Y differential case is independently specified as a
non-reciprocal three-port with singular `I-S` and nonsingular direct `A`, plus
complex per-port/frequency-dependent references. Its expected Y is generated
only by public `skrf.network.s2y(..., s_def="power")` from pinned
`scikit-rf==2.0.1` (`bd651e923cac6020de49a096e1d7e9b5f949f884`), with the local
seed `20260946` and fixture name
`power_wave_s_to_y_three_port_singular_i_minus_s_complex_z0.json`; the fixture
metadata records operation, units, versions, and strict tolerances. The
input is not derived from an opposite-direction fixture or a round trip.

## Reporting

Eventually CI should publish a machine-generated coverage report such as:

```text
S<->Z conversion        10,000 / 10,000 pass
Renormalization          8,000 /  8,000 pass
Connect/cascade          5,000 /  5,000 pass
Touchstone round-trip    1,000 /  1,000 pass
```
