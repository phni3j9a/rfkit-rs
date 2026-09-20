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

## Reporting

Eventually CI should publish a machine-generated coverage report such as:

```text
S<->Z conversion        10,000 / 10,000 pass
Renormalization          8,000 /  8,000 pass
Connect/cascade          5,000 /  5,000 pass
Touchstone round-trip    1,000 /  1,000 pass
```
