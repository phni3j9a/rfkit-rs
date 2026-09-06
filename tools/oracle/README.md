# scikit-rf oracle

This directory contains the reproducible Python oracle harness for `rfkit-rs`.
It generates checked-in JSON fixtures through `scikit-rf==2.0.1`, with
`numpy==2.5.1` and `scipy==1.18.1` pinned directly. scikit-rf is an oracle for
numerical behavior here, not the public API specification for the Rust library.
The harness keeps
the original three-port fixture and registers an additional eight-case
S↔Z power-wave conformance matrix, a four-case S renormalization matrix, and
reciprocal and active three-port cases for each existing operation. The three
existing reciprocal cases also carry optional passive-network evidence; no
new reciprocal fixture or operation case is introduced. Two near-singular
three-port operation cases are also registered; they exercise the existing
exact-pivot kernels at a deterministic nonsingular boundary without changing
production semantics. Four direct impedance/admittance cases are also
registered: Z→Y and Y→Z inputs are constructed independently, each with a
well-conditioned and an explicitly nonsingular near-singular three-port case.
Two direct power-wave S↔Y cases are also registered: each direction has an
independently constructed three-port input, explicit `s_def="power"`, and a
non-50 Ω complex per-port frequency-dependent reference-impedance array.
One matched-junction connection case is also registered: independent
three-port A and four-port B inputs use explicit `s_def="power"`, a
frequency-dependent real-positive junction impedance that is exactly equal on
both sides, and non-trivial frequency/port-dependent external reference
impedances. Its expected output comes from public
`skrf.network.connect(network_a, port_a, network_b, port_b)` behavior.
One same-network inner-connect case is also registered: an independent
five-port input uses explicit `s_def="power"`, two non-adjacent ports, and a
frequency-dependent real-positive junction impedance that is exactly equal at
the selected ports. Its expected output comes from public
`skrf.network.innerconnect(network, k, l)` behavior, and the three surviving
ports are recorded in their original order.
One Cartesian linear interpolation case is also registered: an independent
three-port S/z0 input uses irregular source and target grids, exact endpoints
and a source knot, nonreciprocal complex S data, and non-50 Ω complex,
per-port, frequency-dependent z0. Its expected S and z0 arrays come from the
public `Network.interpolate(..., basis="s", coords="cart", kind="linear")`
behavior. The interpolation fixture records the exact SciPy version used by
that delegated numerical operation.
One explicit-grid matched-connection case is also registered: independent
three-port A and four-port B inputs use distinct irregular source grids and a
caller-supplied target grid. Each network is interpolated once through the
public Cartesian-linear S interpolation API, followed by one public
`skrf.network.connect` call. The selected non-50 Ω real-positive junction is
exactly matched after interpolation, while surviving-port z0 values are
complex and frequency-dependent. The fixture records the target grid,
source inputs, survivor order, and SciPy version exactly; only connected S and
z0 outputs use numeric tolerances.

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

The harness checks all three imported versions before doing any work. A
mismatch fails clearly instead of silently regenerating a fixture with another
version.

## Generate and verify

The harness has thirty-three registered canonical cases. The original three cases
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

The additional interpolation case is:

- `interpolation_cartesian_linear_three_port_complex_z0` — direct irregular
  source/target grids with independent complex S and z0 inputs; expected S and
  z0 are obtained from public Cartesian linear `Network.interpolate` behavior.

The additional explicit-grid composition case is:

- `power_wave_connect_matched_explicit_grid_three_to_four_port_complex_z0` —
  independent three-port A and four-port B inputs with distinct irregular
  source grids, an explicit in-range eight-point target grid, nonreciprocal
  complex S data, an exactly matched non-50 Ω real junction, and complex
  per-port/frequency-dependent external z0. Expected output is obtained by
  interpolating A and B once each with public Cartesian-linear
  `Network.interpolate`, then calling public `skrf.network.connect` once.

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
reference impedance on every port and frequency sample. Their relevant
power-wave S matrices are strictly passive by a deliberately non-marginal
contract: the largest singular value at every frequency is below 0.8 (and
therefore below 1). The generator computes the true largest singular value
with pinned NumPy SVD, records the maximum over all frequencies rounded to 12
decimal places, and stores that evidence in `metadata.passive_network`.

| Direction | Ports | Reference-impedance profile | Seed | Case id |
| --- | ---: | --- | ---: | --- |
| S→Z | 3 | real, equal across ports and frequency | `20260921` | `power_wave_s_to_z_three_port_reciprocal_real_equal_z0` |
| Z→S | 3 | real, equal across ports and frequency | `20260922` | `power_wave_z_to_s_three_port_reciprocal_real_equal_z0` |

The pinned passive maxima are kept with the existing reciprocal fixture data:

| Case | Matrix field | Observed maximum |
| --- | --- | ---: |
| S→Z reciprocal | `s` | `0.151907019275` |
| Z→S reciprocal | `s` | `0.166759615367` |
| Renormalization reciprocal | `s_input` | `0.155165225095` |
| Renormalization reciprocal | `s_renormalized` | `0.456137460254` |

For renormalization, `s_input` and `s_renormalized` are evidenced separately
and in that order. The metadata shape is intentionally:

```json
{
  "criterion": "largest singular value of every relevant power-wave S matrix is strictly less than 1",
  "required_maximum": 0.8,
  "matrices": [
    {"matrix_field": "s", "observed_maximum": 0.151907019275}
  ]
}
```

The S→Z fixture uses the source `s`, Z→S uses the public converted `s`, and
renormalization uses both source and target S stacks. Rust tests independently
certify each relevant matrix with the Frobenius upper bound
`sigma_max(S) <= ||S||_F`, without adding an SVD or production dependency.

The active S↔Z cases are registered separately. Each is a three-frequency,
three-port case with real 57.25 Ω reference impedance on every port and
frequency. The relevant power-wave S matrix has a true largest singular value
strictly above one at every frequency; the generator records the observed
minimum from unrounded NumPy SVD values, rounded to 12 decimal places for
cross-backend portability, and requires the raw minimum to exceed 1.2.

| Direction | Ports | Reference-impedance profile | Seed | Active matrix field | Case id |
| --- | ---: | --- | ---: | --- | --- |
| S→Z | 3 | real, equal across ports and frequency | `20260924` | `s` (direct input) | `power_wave_s_to_z_three_port_active_real_equal_z0` |
| Z→S | 3 | real, equal across ports and frequency | `20260925` | `s` (public output) | `power_wave_z_to_s_three_port_active_real_equal_z0` |

The S renormalization cases are registered separately:

| Operation | Ports | Reference-impedance profile | Case id |
| --- | ---: | --- | --- |
| S renormalization | 1 | real, scalar/constant-equivalent source and target z0 | `power_wave_renormalize_one_port_real_scalar_z0` |
| S renormalization | 2 | complex, per-port, constant-over-frequency source and target z0 | `power_wave_renormalize_two_port_complex_per_port_constant_z0` |
| S renormalization | 4 | complex, per-port, frequency-dependent source and target z0 | `power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0` |
| S renormalization | 8 | real, frequency-dependent, same across ports source and target z0 | `power_wave_renormalize_eight_port_real_frequency_dependent_z0` |
| S renormalization | 3 | real, equal across ports and frequency; 42.75 Ω → 86.5 Ω | `power_wave_renormalize_three_port_reciprocal_real_equal_z0` |
| S renormalization | 3 | real, equal across ports and frequency; 57.25 Ω → 91.75 Ω | `power_wave_renormalize_three_port_active_real_equal_z0` |

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

The active renormalization case uses a directly generated active source S input
with 57.25 Ω source and 91.75 Ω target reference impedances. Its active
metadata explicitly identifies `s_input` as the relevant matrix, so the
recorded 1.2 lower bound is not confused with a classification of the
renormalized output.

The near-singular S↔Z cases are registered separately. Each case has three
frequency samples, three ports, and a real positive 64 Ω reference impedance
expanded equally across every port and frequency. The S-to-Z case constructs S
directly so that the input system `I-S` is upper triangular. The Z-to-S case
constructs Z independently so that the normalized input system
`(Z+z0 I)/z0` is upper triangular; its expected S is obtained only through the
public `Network.from_z(..., s_def="power").s` path. The directions use distinct
local NumPy `default_rng` seeds and no expected output or round-trip is used to
construct either input.

Both systems use the exact binary64 factor `2^-20 =
9.5367431640625e-7` at diagonal port 0, with the other diagonal factors
`0.625` and `0.75`. Their determinant is therefore the explicit non-zero
product `4.470348358154297e-7`; reducing that first factor to zero reaches the
exact-singular boundary. The upper off-diagonal entries come from each case's
recorded local seed and a fixed normal distribution, while the lower triangle
is exactly zero. `metadata.near_singular` records the system name, structure,
binary exponent/value, diagonal factors, and determinant evidence. Generator
and Rust/Python contract tests reconstruct the system from serialized inputs
and verify these facts directly; no platform-sensitive condition number is
recorded or used for acceptance.

| Direction | Ports | Reference impedance | Seed | System | Case id |
| --- | ---: | --- | ---: | --- | --- |
| S→Z | 3 | real, equal 64 Ω | `20260927` | `I-S` | `power_wave_s_to_z_three_port_near_singular_real_equal_z0` |
| Z→S | 3 | real, equal 64 Ω | `20260928` | `(Z+z0 I)/z0` | `power_wave_z_to_s_three_port_near_singular_real_equal_z0` |

The near-singular cases retain the existing strict case-local `rtol=1e-12`
policy with `atol_ohm=1e-12` for S→Z and `atol=1e-12` for Z→S. The policy is
recorded in each fixture with the binary64 and `2^-20` proximity rationale;
the outputs are checked to be finite before comparison.

The direct impedance/admittance cases do not use a reference impedance or a
wave definition. Z→Y inputs are stored under `data.z_ohm` and expected outputs
under `data.y_s`; Y→Z inputs use `data.y_s` and expected outputs use
`data.z_ohm`. The expected values come directly from the public
`skrf.network.z2y` or `skrf.network.y2z` function for the corresponding
independent input. In scikit-rf 2.0.1 these public functions are exposed from
the `network` module rather than as top-level `skrf.z2y`/`skrf.y2z` names.

| Direction | Conditioning | Ports | Seed | Input/output units | Case id |
| --- | --- | ---: | ---: | --- | --- |
| Z→Y | well-conditioned | 3 | `20260933` | Ω → S | `impedance_admittance_z_to_y_three_port_well_conditioned` |
| Y→Z | well-conditioned | 3 | `20260934` | S → Ω | `impedance_admittance_y_to_z_three_port_well_conditioned` |
| Z→Y | near-singular, nonsingular | 3 | `20260935` | Ω → S | `impedance_admittance_z_to_y_three_port_near_singular` |
| Y→Z | near-singular, nonsingular | 3 | `20260936` | S → Ω | `impedance_admittance_y_to_z_three_port_near_singular` |

The well-conditioned inputs use independent local NumPy `default_rng` seeds
and a generation-time strict diagonal-dominance guard. The near inputs are
independent upper-triangular matrices with exact diagonal factors
`[2^-20, 0.625, 0.75]`; the determinant evidence is the product of these
non-zero factors (`4.470348358154297e-7`). Pinned NumPy `matrix_rank` is used
only while generating the fixture to verify full rank. No condition number,
rank cutoff, or runtime tolerance classification is used by the Rust kernel.
The near metadata records the binary exponent, diagonal factors, determinant,
and upper-triangular structure so the construction can be reconstructed from
the serialized direct input.

All four cases use strict binary64 `rtol=1e-12`; Z→Y records `atol_s=1e-12`
and Y→Z records `atol_ohm=1e-12`. The checker tolerates differences only in
the selected computed output and requires exact canonical equality for the
direct input, frequencies, units, metadata, shapes, and all other fields.

The direct power-wave S↔Y cases use separate, non-symmetric three-port inputs
for each direction. Both inputs have three frequency samples and the same
explicitly serialized complex, per-port, frequency-dependent z0 profile; the
S and Y matrices use distinct local NumPy `default_rng` seeds and are never
derived from one another or from an expected output. S→Y stores `data.s` and
`data.y_s`, while Y→S stores `data.y_s` and `data.s`. Expected outputs come
directly from `skrf.network.s2y(..., z0=..., s_def="power")` and
`skrf.network.y2s(..., z0=..., s_def="power")`, respectively. The fixtures
record `rtol=1e-12`, `atol_s=1e-12` for S→Y, and `atol=1e-12` for Y→S.

| Direction | Ports | Reference-impedance profile | Seed | Input/output units | Case id |
| --- | ---: | --- | ---: | --- | --- |
| S→Y | 3 | complex, per-port, frequency-dependent | `20260937` | dimensionless → S | `power_wave_s_to_y_three_port_complex_z0` |
| Y→S | 3 | complex, per-port, frequency-dependent | `20260938` | S → dimensionless | `power_wave_y_to_s_three_port_complex_z0` |

The matched-junction case uses a three-port A network and a four-port B network
at three exactly shared frequencies. A port 1 is connected to B port 2 through
the matched real junction; the resulting five-port order is A ports `[0, 2]`
followed by B ports `[0, 1, 3]`. The junction values are finite, strictly
positive, frequency-dependent, non-50 Ω values and are copied exactly into both
input z0 arrays. External z0 values are finite, real, and independently
frequency/port-dependent. The two S inputs are generated by independent local
NumPy seeds `20260939` and `20260940`. The expected `s_connected` output comes
only from `skrf.network.connect`; `z0_connected` and the explicit ordering
metadata are checked exactly by the Rust and Python contract tests.

| Direction | Input ports | Junction ports | Output order | Case id |
| --- | ---: | --- | --- | --- |
| A + B → connected network | 3 + 4 | A[1] ↔ B[2] | A[0], A[2], B[0], B[1], B[3] | `power_wave_connect_matched_three_to_four_port_real_frequency_dependent_z0` |

The explicit-grid composition case uses independent A and B source grids with
five and six irregular samples, respectively, and an explicit eight-sample
target grid containing both source endpoints, source knots, and interior
points. A port 1 is connected to B port 2 after each side is interpolated once
with `Network.interpolate(..., basis="s", coords="cart", kind="linear")`.
The selected junction is a constant 73.5 Ω real-positive value in both source
arrays, so it remains exactly equal after either interpolation; surviving-port
z0 values are complex and vary by frequency and port. The resulting five-port
order is A ports `[0, 2]` followed by B ports `[0, 1, 3]`. The expected
`s_connected` and `z0_connected_ohm` outputs come from one public
`skrf.network.connect` call on the two interpolated networks. The fixture
uses independent local seeds `20260943` and `20260944`, records the pinned
SciPy version, and treats both output arrays as numeric-tolerance fields while
checking grids, inputs, ordering, and metadata exactly.

The same-network inner-connect case uses one five-port network at three
exactly serialized frequencies. Ports 1 and 3 are connected through a
finite, real, strictly positive, frequency-dependent non-50 Ω junction, and
the resulting three-port order is `[0, 2, 4]`. The input S matrix is generated
from local seed `20260941` and is non-symmetric/non-reciprocal. The expected
`s_inner_connected` output comes only from
`skrf.network.innerconnect(network, 1, 3)` on the explicit power-wave
Network; output z0 and order metadata are checked exactly.

| Direction | Input ports | Junction ports | Output order | Case id |
| --- | ---: | --- | --- | --- |
| Inner connection | 5 | 1 ↔ 3 | 0, 2, 4 | `power_wave_inner_connect_matched_five_port_real_frequency_dependent_z0` |

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
tolerance used by the fixture checker. Their reference impedances are finite,
real, strictly positive, equal across every port and frequency, and
intentionally non-50 Ω; the renormalization source/target values are
materially separated. For each relevant S stack, the generator checks every
raw NumPy SVD sigma-max against `0.8`, checks the rounded maximum against the
same strict bound, and records only the rounded maximum in
`metadata.passive_network`.

The three active cases use dedicated local NumPy `default_rng` seeds
`20260924`, `20260925`, and `20260926`. S-to-Z and renormalization construct
their active S inputs directly with deliberately active diagonal values and
small non-symmetric coupling. The Z-to-S case constructs a separate direct Z
input with negative-resistance diagonal terms far from `-z0`; its active
matrix is the S result read back from public
`Network.from_z(..., s_def="power").s`, never a round trip from that output.
The generator computes the unrounded true largest singular value of each
relevant S sample with NumPy SVD, validates every raw value against 1.2, and
records the minimum rounded to 12 decimal places in
`metadata.active_network` for equivalent-LAPACK portability. Rust independently
certifies that bound using the mathematically valid maximum column 2-norm
lower bound, without adding an SVD or other runtime dependency. All active
inputs pass the existing conservative diagonal-dominance guards; for active
renormalization, the underlying source-referenced Z is explicitly checked
against the target-stage Z+G guard before the public renormalization call.

The two near-singular cases use dedicated local NumPy `default_rng` seeds
`20260927` and `20260928`. For S→Z, each frequency's direct S is defined as
`I-A`, where A is an upper-triangular matrix with diagonal
`[2^-20, 0.625, 0.75]`. For Z→S, each frequency's direct Z is defined from a
separately generated upper-triangular U as `z0 * (U-I)`, so the normalized
system `(Z+z0 I)/z0` is U. The two local RNG streams are independent and the
expected outputs are obtained solely from public scikit-rf properties. The
generator and tests verify the exact lower-triangle zeros, binary diagonal
factors, and product-of-diagonal determinant evidence directly from the input
configuration. These checks intentionally do not calculate or record a
condition number. Every source and expected output is checked for finite
values; the first factor is non-zero and above scikit-rf's `EIG_COND=1e-9`.

The direct impedance/admittance fixtures use a separate input construction for
each direction. The Z→Y builders generate `z_ohm` directly and call only
`skrf.network.z2y` for `y_s`; the Y→Z builders generate `y_s` directly and
call only `skrf.network.y2z` for `z_ohm`. Neither direction uses an expected
output or a round-trip result as its input. Well-conditioned inputs use a
strict generation-time diagonal-dominance guard. Near-singular inputs are
upper triangular with exact diagonal `[2^-20, 0.625, 0.75]`, and the metadata
records the non-zero determinant product and structure. Pinned NumPy
`matrix_rank` verifies full rank during generation only. The four direct cases
use seeds `20260933` through `20260936`; they have no z0 or wave-definition
fields beyond `wave_definition: "not_applicable"`, and record units explicitly
as `ohm`/`S` in `input_unit` and `output_unit`.

The direct power-wave S↔Y fixtures use seeds `20260937` and `20260938` for
independent direct S and Y inputs. Their shared z0 profile is generated by the
local frequency/port construction and serialized in full; every value has a
non-zero real and imaginary component, and both dimensions vary. A strict
generation-time diagonal-dominance/full-rank guard keeps each direct input
well-conditioned. The expected output is produced by the corresponding public
`skrf.network.s2y` or `skrf.network.y2s` call with `s_def="power"`; no output
or round-trip is fed into the opposite direction. S→Y records `input_unit:
dimensionless`, `output_unit: S`, and `atol_s`; Y→S records `input_unit: S`,
`output_unit: dimensionless`, and `atol`.

The inner-connect fixture uses an independent direct five-port S/z0
construction with seed `20260941`. Its expected S output is obtained only
through public `skrf.network.innerconnect(network, 1, 3)` on an explicit
power-wave Network. Every fixture z0 is real, and the selected pair is also
exactly equal, so the pinned helper's internal power/pseudo conversion is
numerically equivalent to the Rust matched power-wave elimination for this
case; no mismatch renormalization or complex junction policy is encoded in the
fixture. The checker removes only
`s_inner_connected` for numeric comparison and checks output z0 and survivor
order exactly.

The interpolation fixture uses an independent local NumPy generator with seed
`20260942`. It serializes `source_frequency_hz`, `target_frequency_hz`,
`s_input`, and `z0_input_ohm` exactly, then obtains both numeric outputs (`s`
and `z0_ohm`) only through public
`Network.interpolate(target, basis="s", coords="cart", kind="linear")`.
The source grid has five irregular samples and the target grid has six samples,
including both endpoints and the exact source knot at 1.11 GHz. Its metadata
records `scikit_rf_version`, `numpy_version`, and the actually used
`scipy_version` (`1.18.1`), plus the explicit interpolation basis, coordinates,
kind, shapes, and `rtol=1e-12`/`atol=1e-12` policy. Both computed outputs are
numeric-tolerance fields; source/target grids, direct inputs, and all metadata
remain exact.

The explicit-grid composition fixture uses independent local NumPy generators
with seeds `20260943` and `20260944`. A's five-sample and B's six-sample
source grids are intentionally different, while the eight-sample target grid
is supplied explicitly and lies within both source spans. Each input has at
least three ports and nonreciprocal complex S data. The expected output is
generated by exactly one public Cartesian-linear `Network.interpolate` call
per input network and exactly one public `skrf.network.connect` call after
interpolation. The selected A[1] and B[2] junction values are exactly 73.5 Ω
at every source sample and remain exactly matched on the target grid; all
surviving-port z0 values are finite complex values varying with frequency and
port. Its metadata records the target/source grids, operation order, survivor
ordering, explicit interpolation settings, and pinned SciPy version. Both
connected S and z0 outputs are numeric-tolerance fields, while all inputs,
grids, ordering, and metadata are exact checker contract fields.

The JSON representation is deliberately machine-readable and byte-stable:

- UTF-8 encoding, `sort_keys=True`, two-space indentation, and one final LF;
- Python's JSON encoder rejects NaN and infinity (`allow_nan=False`);
- complex numbers are objects with explicit `real` and `imag` fields;
- metadata records schema version, operation, case id, dependency versions,
  seed(s), input/output array shapes, wave definition, and reference-impedance
  characteristics; operation cases may additionally link a shared input case;
  renormalization records separate source and target reference-impedance flags
  and shapes for each input/output array. Reciprocal passive cases additionally
  record a `passive_network` contract with `criterion`, `required_maximum`, and
  ordered per-field `matrices` entries containing the pinned NumPy-SVD
  `observed_maximum`; active cases additionally record a
  concise `active_network` contract with `criterion`, `matrix_field`,
  `required_minimum`, and the NumPy-SVD `observed_minimum`; existing fixtures
  omit each optional evidence field when it does not apply. Near-singular
  operation cases additionally record a `near_singular` contract containing
  `system_matrix`, `matrix_structure`, `binary_exponent`,
  `small_diagonal_port`, `small_diagonal_value`, `determinant_factors`,
  `determinant`, and `determinant_factors_nonzero`; the interpolation case
  additionally records its explicit `basis`, `coords`, `kind`, source/target
  frequency shapes, and the actually used `scipy_version`;
- direct impedance/admittance and power-wave S/Y cases record `input_unit` and
  `output_unit`,
  use `y_s` for Z→Y outputs and `z_ohm` for Y→Z outputs, and use exactly one
  of `tolerance_policy.atol_s`, `tolerance_policy.atol_ohm`, or
  `tolerance_policy.atol` as appropriate;
- The `three_port_complex_z0` network case retains an exact canonical UTF-8
  byte comparison.
- Every operation case requires strict JSON parsing (including finite
  numbers), canonical encoding of the actual document, and exact canonical
  equality for metadata, schema, dependency versions, shapes, frequency,
  inputs, z0, and every other field except the computed output (`z_ohm` for
  S→Z, `s` for Z→S or Y→S, `y_s` for Z→Y or S→Y, `z_ohm` for Y→Z, or
  `s_renormalized` for S renormalization, or `s_inner_connected` for inner
  connection; interpolation removes both `s` and `z0_ohm`. The output's
  recursively validated complex array is
  compared with the recorded
  `abs(actual-expected) <= atol + rtol*abs(expected)` policy. S→Z uses
  `rtol=1e-12` and `atol_ohm=1e-12`; Z→S and S renormalization use
  `rtol=1e-12` and `atol=1e-12`; Z→Y uses `rtol=1e-12` and `atol_s=1e-12`,
  Y→Z uses `rtol=1e-12` and `atol_ohm=1e-12`; S→Y uses `rtol=1e-12` and
  `atol_s=1e-12`, Y→S uses `rtol=1e-12` and `atol=1e-12`, and inner-connect
  uses `rtol=1e-12` and `atol=1e-12`; interpolation uses `rtol=1e-12` and
  `atol=1e-12` for both outputs. These are strict binary64 tolerances for
  the well-conditioned, modest-magnitude deterministic cases: they allow
  normal cross-language linear-algebra rounding while catching material
  disagreement.
- The checker removes exactly the registered computed output field(s) for the
  contract projection. Existing cases register one output; interpolation
  registers both `s` and `z0_ohm`. It never tolerates drift in inputs, z0,
  dimensions, metadata, or any unknown/missing complex field, and it never
  widens a recorded tolerance.

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
