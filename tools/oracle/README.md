# scikit-rf oracle

This directory contains the reproducible Python oracle harness for `rfkit-rs`.
It generates checked-in JSON fixtures through `scikit-rf==2.0.1`, with
`numpy==2.5.1` pinned directly. scikit-rf is an oracle for numerical behavior
here, not the public API specification for the Rust library.

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

The harness has three registered canonical cases:

- `three_port_complex_z0` — the representative four-frequency, three-port
  Network input with frequency-dependent, per-port complex reference
  impedances.
- `power_wave_s_to_z_three_port_complex_z0` — the same input plus the complete
  expected power-wave S-to-Z result obtained from the public `Network.z`
  property.
- `power_wave_z_to_s_three_port_complex_z0` — a directly generated,
  non-symmetric, diagonally dominant complex Z input with frequency-dependent
  per-port `z0`; the complete expected power-wave Z-to-S result is obtained
  from public `Network.from_z(..., s_def="power").s`.

All three registered cases are checked by default against a fresh scikit-rf run;
the default command checks every case:

```bash
python generate_oracle.py check
```

To intentionally regenerate every fixture after a reviewed case or dependency
change, use write mode and then check mode:

```bash
python generate_oracle.py write
python generate_oracle.py check
```

Both modes return a non-zero status on setup or comparison failure. `write` is
the only mode that changes files; it writes every registered case's canonical
bytes from the generator's in-memory documents. The checked-in fixtures should
be reviewed together with the generator change. `check` never rewrites a
fixture, retries a failed comparison, or widens a tolerance.

To work with one case, pass its case id. A temporary path can be supplied for
safe failure testing or review before replacing a checked-in fixture:

```bash
python generate_oracle.py check --case power_wave_s_to_z_three_port_complex_z0
python generate_oracle.py write --case power_wave_s_to_z_three_port_complex_z0 \
  --fixture /tmp/power_wave_s_to_z.json
```

For compatibility with the original one-fixture harness, `--fixture` without
`--case` selects `three_port_complex_z0`; an invocation without either option
always selects all registered cases.

## Fixture contents and canonicalization

The input case is a four-frequency, three-port `Network` with a non-symmetric
complex S matrix and frequency-dependent, per-port complex `z0`. The S and
`z0` values are read back from the scikit-rf `Network` object. A local NumPy
`default_rng` uses the recorded seed `20250308`; no process-global random state
is changed. The S-to-Z operation case obtains its expected values through the
public `Network.z` property. The Z-to-S operation case uses a dedicated local
`default_rng` seed (`20260826`) to construct its input Z directly, then obtains
expected S values only through public `Network.from_z` and `Network.s`.

The JSON representation is deliberately machine-readable and byte-stable:

- UTF-8 encoding, `sort_keys=True`, two-space indentation, and one final LF;
- Python's JSON encoder rejects NaN and infinity (`allow_nan=False`);
- complex numbers are objects with explicit `real` and `imag` fields;
- metadata records schema version, operation, case id, dependency versions,
  seed, input/output array shapes, wave definition, and reference-impedance
  characteristics; operation cases may additionally link a shared input case;
- The `three_port_complex_z0` network case retains an exact canonical UTF-8
  byte comparison.
- The `power_wave_s_to_z_three_port_complex_z0` case requires strict JSON
  parsing (including finite numbers), canonical encoding of the actual
  document, and exact canonical equality for metadata, schema, dependency
  versions, shapes, frequency, S, z0, and every other field except `z_ohm`.
  Its recursively validated `z_ohm` complex array is compared with the
  recorded `abs(actual-expected) <= atol_ohm + rtol*abs(expected)` policy,
  currently `rtol=1e-12` and `atol_ohm=1e-12`. This is a strict binary64
  tolerance for this well-conditioned, modest-magnitude deterministic case:
  it allows normal cross-language linear-algebra rounding while catching
  material disagreement.
- The `power_wave_z_to_s_three_port_complex_z0` case uses the same strict
  contract checks and compares its dimensionless `s` output with the recorded
  `rtol=1e-12` and `atol=1e-12` policy. Its direct Z input is generated with
  the recorded dedicated seed, so it remains an exact contract even when
  linear-algebra implementation details vary. The checker accepts the
  case-specific absolute-tolerance key (`atol` for dimensionless output,
  `atol_ohm` for the existing impedance output) without weakening either
  contract.

## Adding a future case

Add a deterministic case builder beside `_network_fixture`, register it in
`_CASES`, and give it a unique `case_id`. Keep each case's operation, input
dimensions, wave definition, seed (when random), and tolerance policy in its
metadata. Add a fixture under `fixtures/`; then run the default `write`
followed by the default `check` in the pinned environment. Cases should cover
additional N-port, reference-impedance, and edge-condition dimensions without
turning this directory into a plotting/UI or broad feature-porting layer.

Do not copy scikit-rf source code or third-party fixture values. Record any
future behavior reference or adaptation in `docs/PROVENANCE.md`.
