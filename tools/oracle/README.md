# scikit-rf oracle

This directory contains the first reproducible Python oracle harness for
`rfkit-rs`. It generates a checked-in JSON fixture through
`scikit-rf==2.0.1`, with `numpy==2.5.1` pinned directly. scikit-rf is an
oracle for numerical behavior here, not the public API specification for the
Rust library.

## Clean-checkout setup

From a clean checkout, create an isolated environment and install the exact
direct dependencies:

```bash
cd tools/oracle
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The harness checks both imported versions before doing any work. A mismatch
fails clearly instead of silently regenerating a fixture with another version.

## Generate and verify

The checked-in fixture is `fixtures/three_port_complex_z0.json` from this
directory (repository path:
`tools/oracle/fixtures/three_port_complex_z0.json`). Check its canonical bytes
against a fresh scikit-rf run with:

```bash
python generate_oracle.py check
```

To intentionally regenerate the fixture after a reviewed case or dependency
change, use write mode and then check mode:

```bash
python generate_oracle.py write
python generate_oracle.py check
```

Both modes return a non-zero status on setup or comparison failure. `write`
is the only mode that changes a file; it writes the canonical bytes from the
generator's in-memory document (the checked-in fixture should be reviewed
together with the generator change).

## Fixture contents and canonicalization

The initial case is a four-frequency, three-port `Network` with a non-
symmetric complex S matrix and frequency-dependent, per-port complex `z0`.
The S and `z0` values are read back from the scikit-rf `Network` object. A
local NumPy `default_rng` uses the recorded seed `20250308`; no process-global
random state is changed.

The JSON representation is deliberately machine-readable and byte-stable:

- UTF-8 encoding, `sort_keys=True`, two-space indentation, and one final LF;
- Python's JSON encoder rejects NaN and infinity (`allow_nan=False`);
- complex numbers are objects with explicit `real` and `imag` fields;
- metadata records schema version, operation, case id, dependency versions,
  seed, array shapes, wave definition, and the tolerance policy;
- check mode compares the complete canonical byte sequence, so regeneration
  has no numeric tolerance. Numerical operations built on this fixture should
  document their own tolerance policy.

## Adding a future case

Add a deterministic case builder beside `_network_fixture` and give it a
unique `case_id`. Keep each case's operation, input dimensions, wave
definition, seed (when random), and tolerance policy in its metadata. Extend
the command-line selection and add a fixture under `fixtures/`; then run
`write` followed by `check` in the pinned environment. Cases should cover
additional N-port, reference-impedance, and edge-condition dimensions without
turning this directory into a plotting/UI or broad feature-porting layer.

Do not copy scikit-rf source code or third-party fixture values. Record any
future behavior reference or adaptation in `docs/PROVENANCE.md`.
