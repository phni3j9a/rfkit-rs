# scikit-rf oracle

This directory contains the reproducible Python oracle harness for `rfkit-rs`.
It generates checked-in JSON fixtures through `scikit-rf==2.0.1`, with
`numpy==2.5.1` pinned directly. scikit-rf is an oracle for numerical behavior
here, not the public API specification for the Rust library. The harness keeps
the original three-port fixture and registers an additional eight-case
S↔Z power-wave conformance matrix, a four-case S renormalization matrix, and
one reciprocal three-port case for each of those existing operations.

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

The harness has eighteen registered canonical cases. The original three cases
remain unchanged:

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

The eight existing S↔Z matrix cases are registered individually as follows.
Each row has multiple frequency samples, and every multiport input matrix is
non-symmetric/non-reciprocal.

| Direction | Ports | Reference-impedance profile | Case-id suffix |
| --- | ---: | --- | --- |
| S→Z | 1 | real scalar, constant-equivalent | `one_port_real_scalar_z0` |
| S→Z | 2 | complex, per-port, constant over frequency | `two_port_complex_per_port_constant_z0` |
| S→Z | 4 | real, frequency-dependent, same across ports | `four_port_real_frequency_dependent_z0` |
| S→Z | 8 | complex, per-port, frequency-dependent | `eight_port_complex_per_port_frequency_dependent_z0` |
| Z→S | 1 | real scalar, constant-equivalent | `one_port_real_scalar_z0` |
| Z→S | 2 | complex, per-port, constant over frequency | `two_port_complex_per_port_constant_z0` |
| Z→S | 4 | real, frequency-dependent, same across ports | `four_port_real_frequency_dependent_z0` |
| Z→S | 8 | complex, per-port, frequency-dependent | `eight_port_complex_per_port_frequency_dependent_z0` |

The full case ids are prefixed with `power_wave_s_to_z_` or
`power_wave_z_to_s_`, and the fixture filenames use the same id with a
`.json` suffix.

The reciprocal S↔Z cases are registered separately. They use one explicit
three-port, three-frequency input per operation, with exact complex transpose
symmetry (the mirrored value is not conjugated) and the same real 61.25 Ω
reference impedance on every port and frequency sample.

| Direction | Ports | Reference-impedance profile | Seed | Case id |
| --- | ---: | --- | ---: | --- |
| S→Z | 3 | real, equal across ports and frequency | `20260921` | `power_wave_s_to_z_three_port_reciprocal_real_equal_z0` |
| Z→S | 3 | real, equal across ports and frequency | `20260922` | `power_wave_z_to_s_three_port_reciprocal_real_equal_z0` |

The S renormalization cases are registered separately:

| Operation | Ports | Reference-impedance profile | Case id |
| --- | ---: | --- | --- |
| S renormalization | 1 | real, scalar/constant-equivalent source and target z0 | `power_wave_renormalize_one_port_real_scalar_z0` |
| S renormalization | 2 | complex, per-port, constant-over-frequency source and target z0 | `power_wave_renormalize_two_port_complex_per_port_constant_z0` |
| S renormalization | 4 | complex, per-port, frequency-dependent source and target z0 | `power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0` |
| S renormalization | 8 | real, frequency-dependent, same across ports source and target z0 | `power_wave_renormalize_eight_port_real_frequency_dependent_z0` |
| S renormalization | 3 | real, equal across ports and frequency; 42.75 Ω → 86.5 Ω | `power_wave_renormalize_three_port_reciprocal_real_equal_z0` |

Each case records explicit source and target reference impedances in the
frequency-major `(frequency, port)` representation after Network read-back.
The source and target values are materially different at every frequency and
port. The expected `s_renormalized` output is obtained through the public
`Network.renormalize(..., s_def="power")` operation followed by the public
`Network.s` property; source and target z0 are read back through `Network.z0`.
The reciprocal renormalization case uses the same exact transpose-symmetric
three-port S input construction, with scalar real source and target values
42.75 Ω and 86.5 Ω respectively; both values are expanded and recorded after
Network read-back.

All registered cases are checked by default against a fresh scikit-rf run; the
default command checks every case:

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

The original input case is a four-frequency, three-port `Network` with a
non-symmetric complex S matrix and frequency-dependent, per-port complex `z0`.
The S and `z0` values are read back from the scikit-rf `Network` object. A
local NumPy `default_rng` uses the recorded seed `20250308`; no process-global
random state is changed. Its S-to-Z output is obtained through the public
`Network.z` property. The original Z-to-S case uses a dedicated local
`default_rng` seed (`20260826`) to construct its input Z directly, then obtains
expected S values only through public `Network.from_z` and `Network.s`.

The eight matrix cases use dedicated local RNG seeds recorded in each
fixture's metadata. Their S inputs are modest and pass a conservative strict
row diagonal-dominance bound for `I-S`; their direct Z inputs are diagonally
dominant and pass the corresponding bound for `Z+G`. These checks are
generation-time guards only: platform-sensitive condition-number values are
not recorded in the schema. Matrix outputs are likewise obtained only from
public scikit-rf `Network.z` or `Network.from_z(..., s_def="power").s`.
Each renormalization case uses its own recorded local RNG seed for a modest
S input (non-symmetric for multiport cases) and applies the same `I-S` guard.
Both source and target z0 values are checked for finite positive real parts,
and their complex/frequency-dependent/per-port metadata flags are checked
against the generated data. Every source/target pair is materially separated.

The three reciprocal cases use dedicated local NumPy `default_rng` seeds
`20260921`, `20260922`, and `20260923`. Each input is generated by filling one
triangle with complex binary64 values and mirroring it to the other triangle
without conjugation. S inputs pass the `I-S` diagonal-dominance guard, direct Z
inputs pass the corresponding `Z+G` guard, and each generated oracle output is
checked for transpose symmetry under the same recorded mixed binary64
tolerance used by the fixture checker. Their reference impedances are real,
equal across ports/frequency, and intentionally non-50 Ω; the renormalization
source/target values are materially separated.

The JSON representation is deliberately machine-readable and byte-stable:

- UTF-8 encoding, `sort_keys=True`, two-space indentation, and one final LF;
- Python's JSON encoder rejects NaN and infinity (`allow_nan=False`);
- complex numbers are objects with explicit `real` and `imag` fields;
- metadata records schema version, operation, case id, dependency versions,
  seed, input/output array shapes, wave definition, and reference-impedance
  characteristics; operation cases may additionally link a shared input case;
  renormalization records separate source and target reference-impedance flags
  and shapes for each input/output array;
- The `three_port_complex_z0` network case retains an exact canonical UTF-8
  byte comparison.
- Every operation case requires strict JSON parsing (including finite
  numbers), canonical encoding of the actual document, and exact canonical
  equality for metadata, schema, dependency versions, shapes, frequency,
  inputs, z0, and every other field except the computed output (`z_ohm` for
  S→Z, `s` for Z→S, or `s_renormalized` for S renormalization). The output's
  recursively validated complex array is
  compared with the recorded
  `abs(actual-expected) <= atol + rtol*abs(expected)` policy. S→Z uses
  `rtol=1e-12` and `atol_ohm=1e-12`; Z→S and S renormalization use
  `rtol=1e-12` and `atol=1e-12`.
  These are strict binary64 tolerances for the well-conditioned,
  modest-magnitude deterministic cases: they allow normal cross-language
  linear-algebra rounding while catching material disagreement.
- The checker removes exactly one computed output field for the contract
  projection. It never tolerates drift in inputs, z0, dimensions, metadata,
  or any unknown/missing complex field, and it never widens a recorded
  tolerance.

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
