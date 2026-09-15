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

Touchstone v1.0 S parsing is covered by an independently authored text fixture
and direct Rust contract tests. The mutually supported oracle slice records
the pinned `scikit-rf==2.0.1` parser, `numpy==2.5.1`, deterministic input text,
port count, and mixed relative/absolute tolerance. Rust compares the parsed
frequency, S, and expanded real scalar `z0` values; v1.0 record boundaries,
unsupported v2/noise/vendor extensions, and deliberate domain rejections are
specified directly in Rust rather than inferred from permissive oracle
behavior.

## Reporting

Eventually CI should publish a machine-generated coverage report such as:

```text
S<->Z conversion        10,000 / 10,000 pass
Renormalization          8,000 /  8,000 pass
Connect/cascade          5,000 /  5,000 pass
Touchstone round-trip    1,000 /  1,000 pass
```
