# Third-party provenance

The oracle harness and all checked-in fixtures are independent rewrites. They
use the public `scikit-rf==2.0.1` `Network` constructor and relevant public
Network behavior/APIs as numerical references, but do not copy scikit-rf source
code, conversion code, or fixture values. The S-to-Z expected values are
generated through the public `Network.z` property after constructing the
shared input Network. The Z-to-S fixture input is a direct deterministic local
rewrite generated with its recorded dedicated RNG seed; only its expected S
output comes from the public `Network.from_z(..., s_def="power")` constructor
and `Network.s` property. The renormalization fixtures use explicit local
source/target z0 values and obtain the expected output only by calling the
public `Network.renormalize(..., s_def="power")` operation, then reading
`Network.s` and `Network.z0`. The internal power-wave S-to-Z, Z-to-S, and
renormalization kernels are independent rewrites of the published equations
and do not copy implementation code. The active cases add no runtime
classifier: the oracle records NumPy SVD evidence for the relevant real,
equal-per-port power-wave S matrix, while Rust tests independently certify the
required lower bound from column norms. Active classification uses the
unrounded pinned-NumPy SVD values; only the recorded minimum is rounded to 12
decimal places so equivalent LAPACK backends retain an identical contract.
The reciprocal cases likewise add no runtime classifier: their existing
real-positive equal-per-port z0 fixtures now record pinned NumPy SVD maxima for
each relevant S field under a strict 0.8 bound, while Rust independently
certifies the same upper bound with Frobenius norms. This is metadata added to
the existing three fixtures, not a copied fixture or a new operation case. The
near-singular cases add two independent operation fixtures, one for each
direction. They are REWRITE fixtures: S→Z uses a direct S construction whose
`I-S` matrix is upper triangular, while Z→S uses a separate direct Z
construction whose normalized `(Z+z0 I)/z0` matrix is upper triangular. Both
use real-positive equal 64 Ω z0, a binary-exact `2^-20` diagonal factor, and
explicit product-of-diagonal determinant evidence. Expected values are still
generated only through public scikit-rf behavior (`Network.z` for S→Z and
`Network.from_z(..., s_def="power").s` for Z→S); no expected output or
round-trip is used as input, and no condition number is recorded.
The impedance/admittance cases are a separate four-fixture REWRITE: direct Z
and direct Y inputs are independently constructed for the two directions, and
only the expected output is obtained from the public
`skrf.network.z2y`/`skrf.network.y2z` functions. In scikit-rf 2.0.1 these
public functions are exposed from the `network` module. The two near cases use
upper-triangular input matrices with non-zero binary diagonal factors and
record their determinant construction; NumPy `matrix_rank` is used only as a
generation-time full-rank guard. No scikit-rf source, expected output, or
fixture value is copied.
The power-wave S/Y cases are a separate two-fixture REWRITE: S and Y inputs
are constructed independently for each direction with distinct local RNG
seeds, while a shared non-50 Ω complex per-port/frequency-dependent z0 profile
is serialized in full. Expected outputs are obtained only from the public
`skrf.network.s2y(..., s_def="power")` and
`skrf.network.y2s(..., s_def="power")` calls; no source code, expected output,
or opposite-direction round-trip fixture value is copied.
The direct singular-capable Y→S case is an additional independent REWRITE:
its three-port Y input is constructed with an exact row dependence (rank two at
every frequency) and deliberately non-reciprocal entries, while its complex
per-port/frequency-dependent z0 profile is independently authored. Expected S
values come only from pinned public `skrf.network.y2s(..., s_def="power")` in
scikit-rf 2.0.1; the Rust implementation derives and solves the Kurokawa
`A=F(I+GY)`, `B=F(I-conj(G)Y)` equations locally and does not copy code or
fixture values.
Issue #78's direct S→Y method is a separate independent REWRITE from the
published Kurokawa power-wave equations. Its public workflow and Touchstone
boundary tests use the direct left system
`A=(S G + conj(G)) F`, `B=(I-S) F`, `A Y=B`; the pinned public
`skrf.network.s2y(..., s_def="power")` in scikit-rf 2.0.1 is consulted only
for differential behavior. The singular-`I-S` three-port case has
independently specified non-reciprocal S data and complex per-port/frequency-
dependent z0, with seed `20260946` and fixture
`power_wave_s_to_y_three_port_singular_i_minus_s_complex_z0.json`. Its expected
Y comes only from the pinned public `s2y` call; this record does not copy source
code or fixture values and does not infer them from the opposite Y→S fixture or
a round trip.
Issue #80's direct renormalization is an independent REWRITE from the
published Kurokawa wave definitions. The kernel uses the explicit
`K=F_new F^-1 (2 Re(G))^-1`, `D/E/C/J` coefficients and right-system solve;
the existing local multiple-right-hand-side solver is reused as operation-
independent infrastructure. scikit-rf 2.0.1 at the pinned commit is used
only as a behavior reference on the existing shared, non-singular fixture
domain; no source code, regularization policy, or hand-copied expected values
are used. The direct method's analytical, invariant, malformed-serde,
negative-real, near-singular, wave-relation, and Touchstone workflow tests are
locally authored. The selected public behavior is additive coexistence with
the composed method and is reversible during 0.x without storage migration.
Issue #92's sampled two-port stability calculation is an independent REWRITE
of the determinant and Rollett equations, with no scikit-rf source or copied
fixture values. The canonical four-frequency input uses an independent base S
stack plus a deterministic complex perturbation from NumPy `default_rng` with
seed `20260954` and scale `1e-3`; its sample classes are derived from true
largest singular values, with one passive and three active/non-passive S
samples. It uses unequal real/complex frequency-dependent references whose
real parts are strictly positive. The expected finite K values come only from pinned public scikit-rf
`Network.stability`; expected complex deltas come independently from NumPy
`linalg.det`. The fixture records scikit-rf `2.0.1` at commit
`bd651e923cac6020de49a096e1d7e9b5f949f884`, NumPy `2.5.1`, seed `20260954`,
exact input/metadata contracts, and output-only `rtol=1e-12`, `atol=1e-12`.
The explicit `None` behavior for exact zero transmission is intentionally
local API policy rather than scikit-rf's infinity behavior; no undefined case
is serialized in JSON. The public Rust method, result type, arithmetic error
stages, core tests, oracle test, and Touchstone workflow are reversible during
0.x without data migration.
The matched-junction connection case is a separate fixture-generator REWRITE:
the A/B S and z0 inputs use independent local NumPy generators and the
expected result is obtained only from public
`skrf.network.connect(network_a, port_a, network_b, port_b)` with explicit
`s_def="power"` Network constructors. The Rust connection kernel is an
independent REWRITE of the matched-junction wave equations, with no scikit-rf
source code or fixture values copied and no mismatch-renormalization rule.
The same-network inner-connect case is a separate fixture-generator REWRITE:
its five-port S/z0 input uses a dedicated local NumPy seed and the expected
result is obtained only from public
`skrf.network.innerconnect(network, k, l)` after constructing an explicit
`s_def="power"` Network. The Rust inner-connect kernel independently rewrites
the Kurokawa matched-junction elimination equation
`S_EE + S_EI (I - P S_II)^-1 P S_IE`; the selected z0 values are finite, real,
strictly positive, frequency-dependent, and exactly equal, while survivors
retain their original order. The fixture's all-real z0 profile makes the
pinned helper's internal power/pseudo conversion numerically equivalent for
this case; no scikit-rf source code or fixture values were copied.
The Cartesian interpolation case is a separate fixture-generator REWRITE:
its irregular source/target grids and nonreciprocal complex S/z0 input are
constructed directly with a local deterministic NumPy generator, while both
expected outputs are obtained only from public
`Network.interpolate(..., basis="s", coords="cart", kind="linear")` behavior.
The oracle pins and records the actually used `scipy==1.18.1` because
scikit-rf delegates this interpolation to SciPy; no SciPy or scikit-rf source
code or fixture values are copied.
The explicit-grid matched-connection case is a separate fixture-generator
REWRITE: its A and B inputs, distinct irregular source grids, explicit target
grid, and complex external z0 values are constructed directly with local
deterministic NumPy generators. Each public `Network.interpolate` call is made
once before one public `skrf.network.connect` call. The selected 73.5 Ω
junction is finite, real, strictly positive, and exactly matched after
interpolation; no scikit-rf or SciPy source code or fixture values are copied.
The Issue #82 port-permutation oracle is also an independent fixture-generator
REWRITE. Its asymmetric three-port S and unequal complex, frequency-dependent
z0 inputs use local seed `20260947`; the expected output is obtained only from
the pinned public `Network.renumbered([2, 0, 1], [0, 1, 2])` behavior. The
fixture records the new-to-old source mapping, NumPy/scikit-rf versions, the
scikit-rf commit pin, and an exact-copy policy for frequency, S, and z0 because
the operation performs no RF arithmetic. No scikit-rf source code or fixture
values were copied.
Issue #84 mixed-mode fixtures are an independent two-direction REWRITE from
the explicit V/I coordinate equations and the repository's Kurokawa power-wave
definition. The forward five-port, three-frequency input uses local seed
`20260948` and is evaluated only through public
`Network.se2gmm(p=2, s_def="power")`; the inverse uses a separate local seed
`20260949`, independently authored modal S/reference inputs, and public
`Network.gmm2se(p=2, z0_se=..., s_def="power")` with an explicit adjacent
target array. The fixtures record the `(0+,1-)`/`(2+,3-)` pairing, output
`[d...,c...,unpaired]` layout, two distinct complex equal pair references per
frequency, the unpaired reference, pinned NumPy/scikit-rf versions and commit,
and exact contract-vs-floating-S tolerance policy. The Rust methods and
Touchstone workflow are independently authored; no scikit-rf source, expected
output, instrument illustration, or fixture value was copied. Complex and
negative-real references remain local-domain coverage rather than a broad
scikit-rf compatibility claim.
Issue #88's direct physical connection is an independent REWRITE from the
published Kurokawa power-wave equations plus voltage continuity and current
conservation at one two-coordinate junction. It does not copy scikit-rf's
mismatch-network implementation or its special two-port ordering. The public
scikit-rf `network.connect` behavior at commit
`bd651e923cac6020de49a096e1d7e9b5f949f884` (`2.0.1`) is used only as a
differential behavior reference. The canonical 3+4-port fixture is independently
authored with seed `20260951`; only its connected S is numerically tolerant,
while metadata, inputs, grids, references, and survivor order are exact.
The method's complex and negative-real nonzero-real domain is governed by the
repository equations, not claimed as broad scikit-rf compatibility. Rollback
of this additive 0.x slice requires no persisted-data migration.

Issue #90's same-network direct inner connection is a separate REWRITE from
the same Kurokawa power-wave equations plus voltage continuity and current
conservation, now applied to the complete selected 2×2 `S_ii` block. The
implementation retains both off-diagonal internal couplings and does not copy
scikit-rf's implementation or its singular fallback. The canonical fixture
`power_wave_inner_connect_direct_five_port_complex_z0.json` is independently
authored with seed `20260952`, NumPy `2.5.1`, and scikit-rf `2.0.1` at commit
`bd651e923cac6020de49a096e1d7e9b5f949f884`. Its expected path deliberately
calls public `skrf.network.innerconnect` on a power-wave input and then calls
the public `result.renormalize(result.z0, s_def="power")` before extracting
S: the raw return is pseudo-wave for this complex-reference case and is a
deliberate oracle trap, not a power-wave expected output. In the pinned
feasibility check, raw pseudo output differed from the independently derived
physical response by `0.09591317335389245`, while explicit power restoration
reduced the maximum difference to `1.18774731498903e-16`; the V/I residual was
`1.2412670766236366e-16`. Only output S is tolerance-compared; input data,
frequency labels, references, survivor order, wave-definition metadata, and
fixture recipe remain exact. Complex and negative-real references are local
equation-domain coverage rather than a broad scikit-rf compatibility claim.
Rollback of this additive Green 0.x slice requires no persisted-data
migration.

| Local path | Source project | Source commit/tag | Source path | Use | License | Notes |
|---|---|---|---|---|---|---|
| `crates/rfkit-core/src/direct_connection.rs`; `crates/rfkit-core/src/lib.rs`; `crates/rfkit-core/tests/public_direct_inner_connection.rs`; `crates/rfkit-core/tests/oracle_direct_inner_connection.rs` | Kurokawa power-wave equations plus physical V/I junction conditions; scikit-rf public behavior reference | Kurokawa DOI `10.1109/TMTT.1965.1125964`; scikit-rf `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Issue #90 direct same-network inner junction over a full selected 2×2 S block | REWRITE | Published mathematics; scikit-rf BSD-3-Clause | Independent Rust kernel, public adapter, invariant tests, and pinned-fixture checker. The finite nonzero-real reference domain includes unequal/equal complex, frequency-dependent, and negative-real references; signed `Re(z)` is retained, survivor/frequency/reference order is exact, only exact evaluated zero pivots are singular, and no code or fixture values are copied. The oracle comparison restores scikit-rf's raw pseudo result explicitly to power waves before extracting S. |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_inner_connect_direct_five_port_complex_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`); NumPy `2.5.1`; seed `20260952` | Public `skrf.network.innerconnect` followed by public `result.renormalize(result.z0, s_def="power")` | REWRITE | BSD-3-Clause | Independently generated asymmetric non-reciprocal five-port, three-frequency canonical fixture with selected ports 1/3, unequal complex frequency-dependent positive-real references, full non-reciprocal internal coupling, exact contract metadata, and output-only `rtol=1e-12`/`atol=1e-12`. The explicit restoration is mandatory because the raw public return is pseudo-wave; no source code, output values, or upstream fixture values were copied. |
| `crates/rfkit-touchstone/tests/public_direct_inner_connection_workflow.rs`; `crates/rfkit-touchstone/examples/inner_connect_direct_power_touchstone.rs` | Kurokawa physical V/I equations; IBIS Open Forum Touchstone File Format Specification | Kurokawa DOI `10.1109/TMTT.1965.1125964`; Touchstone Format Specification v2.1 (2024-01-26) | Five-port v1.0 ingress → explicit unequal-complex direct renormalization → full-block inner physical check → direct inner connection → explicit common positive-real writer renormalization → write/read | REWRITE | Published mathematics; IBIS Open Forum specification | The input reuses the existing local five-port Touchstone pattern; the direct V/I reconstruction and executable/test workflow are independently authored. The writer is called only after explicit direct restoration to one common finite positive-real reference; no hidden wave conversion, automatic repair, format extension, copied third-party source, or broad scikit-rf compatibility promise is introduced. |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_renormalize_{one,two,eight}_port_*.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network` constructor, `Network.renormalize(..., s_def="power")`, `Network.s`, and `Network.z0` APIs | REWRITE | BSD-3-Clause | Three additional deterministic local S inputs covering real scalar, complex per-port constant, and real frequency-dependent same-across-port source/target z0; expected output is obtained only through public renormalization behavior, with no source code or fixture values copied |
| `tools/oracle/generate_oracle.py` | scikit-rf | `v2.0.1` | `skrf.Network` constructor, public `Network.z` and `Network.renormalize` APIs, and public `Network.from_z`/`Network.s` API | REWRITE | BSD-3-Clause | Independent deterministic generator; direct Z inputs and renormalization source/target z0 use local deterministic construction; scikit-rf is a behavior/API reference only; no source code or fixture copied |
| `tools/oracle/fixtures/three_port_complex_z0.json`; `tools/oracle/fixtures/power_wave_s_to_z_three_port_complex_z0.json`; `tools/oracle/fixtures/power_wave_z_to_s_three_port_complex_z0.json` | scikit-rf | `v2.0.1` | Public `Network` read-back, `Network.z`, and `Network.from_z`/`Network.s` behavior | REWRITE | BSD-3-Clause | Independently generated canonical outputs; Z-to-S input is direct deterministic data and expected S is obtained through public `Network.from_z`/`Network.s`; no fixture values copied |
| `tools/oracle/fixtures/power_wave_s_to_z_{one,two,four,eight}_port_*.json`; `tools/oracle/fixtures/power_wave_z_to_s_{one,two,four,eight}_port_*.json` | scikit-rf | `v2.0.1` | Public `Network.z` and `Network.from_z(..., s_def="power").s` behavior | REWRITE | BSD-3-Clause | Independently generated deterministic matrix cases covering 1/2/4/8 ports and scalar/per-port/frequency-dependent real/complex z0; inputs use local RNG and conservative generation-time diagonal-dominance checks; no source code or fixture values copied |
| `tools/oracle/fixtures/power_wave_s_to_z_three_port_reciprocal_real_equal_z0.json`; `tools/oracle/fixtures/power_wave_z_to_s_three_port_reciprocal_real_equal_z0.json`; `tools/oracle/fixtures/power_wave_renormalize_three_port_reciprocal_real_equal_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network.z`, `Network.from_z(..., s_def="power").s`, and `Network.renormalize(..., s_def="power")` APIs; pinned NumPy SVD is used only to record passive evidence | REWRITE | BSD-3-Clause | Three independent deterministic three-port reciprocal cases reused without new fixture data; local RNG seeds are 20260921/20260922/20260923, inputs mirror one complex triangle without conjugation, all S/Z cases use explicit real-positive 61.25 Ω equal-across-port/frequency z0, and renormalization uses real-positive equal-across-port/frequency 42.75 Ω → 86.5 Ω. Passive maxima are rounded to 12 decimals in metadata; Rust uses a Frobenius upper-bound certificate. No source code or fixture values were copied |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_s_to_z_three_port_active_real_equal_z0.json`; `tools/oracle/fixtures/power_wave_z_to_s_three_port_active_real_equal_z0.json`; `tools/oracle/fixtures/power_wave_renormalize_three_port_active_real_equal_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network.z`, `Network.from_z(..., s_def="power").s`, and `Network.renormalize(..., s_def="power")` APIs; unrounded NumPy SVD is used only for active validation, with the recorded minimum rounded to 12 decimal places | REWRITE | BSD-3-Clause | Three independent deterministic three-port active cases with real equal-per-port 57.25 Ω z0 (renormalization target 91.75 Ω), local RNG seeds 20260924/20260925/20260926, and required sigma-max minimum 1.2. S-to-Z/renormalization use direct active S; Z-to-S uses direct negative-resistance Z and never round-trips expected S; no source code or fixture values copied |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network` constructor, `Network.renormalize(..., s_def="power")`, `Network.s`, and `Network.z0` APIs | REWRITE | BSD-3-Clause | Four-port deterministic local S input and explicit complex per-port frequency-dependent source/target z0; expected output is obtained only through public renormalization behavior, with no source code or fixture values copied |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_s_to_z_three_port_near_singular_real_equal_z0.json`; `tools/oracle/fixtures/power_wave_z_to_s_three_port_near_singular_real_equal_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network.z` and `Network.from_z(..., s_def="power").s` APIs; inputs are independent local direct constructions and metadata records binary upper-triangular determinant evidence | REWRITE | BSD-3-Clause | Two independent three-frequency, three-port cases with real-positive equal 64 Ω z0, seeds 20260927/20260928, and a binary-exact `2^-20` diagonal factor above scikit-rf's `EIG_COND=1e-9`; no source code, fixture values, expected-output round trip, or condition number was copied or recorded |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/impedance_admittance_{z_to_y,y_to_z}_three_port_{well_conditioned,near_singular}.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `skrf.network.z2y` and `skrf.network.y2z` behavior | REWRITE | BSD-3-Clause | Four independent direct Z/Y three-port fixtures (well-conditioned and near-singular per direction), seeds 20260933–20260936; Z is recorded in ohms and Y in siemens, expected output is computed only from the corresponding public function, and no source code or fixture values are copied |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_s_to_y_three_port_complex_z0.json`; `tools/oracle/fixtures/power_wave_y_to_s_three_port_complex_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `skrf.network.s2y(..., s_def="power")` and `skrf.network.y2s(..., s_def="power")` behavior | REWRITE | BSD-3-Clause | Two independent direct three-port S/Y fixtures with distinct seeds 20260937/20260938 and a shared non-50 Ω complex per-port frequency-dependent z0 profile; all input/output values are serialized and expected output comes only from the corresponding public function; no source code, fixture values, or opposite-direction round trip is copied |
| `crates/rfkit-core/src/interpolation.rs` | Cartesian linear interpolation; scikit-rf and SciPy behavior references | N/A for the independent kernel; `v2.0.1` and `1.18.1` for behavior | Private Cartesian linear N-port interpolation kernel; scikit-rf v2.0.1 public `Network.interpolate(..., basis="s", coords="cart", kind="linear")` behavior is used only as a behavior reference | REWRITE | Mathematical reference; scikit-rf BSD-3-Clause; SciPy BSD-3-Clause | Independent REWRITE of the Rust kernel for frequency-major S and z0 arrays; no scikit-rf or SciPy source code was copied |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/interpolation_cartesian_linear_three_port_complex_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network.interpolate(..., basis="s", coords="cart", kind="linear")` behavior; SciPy `1.18.1` is pinned and recorded for delegated interpolation numerics | REWRITE | BSD-3-Clause | One independent three-port irregular source/target fixture with seed 20260942, nonreciprocal complex S, non-50 Ω complex per-port frequency-dependent z0, exact endpoints and source-knot target, and output-only mixed `rtol=1e-12`/`atol=1e-12` checking for both interpolated S and z0; no source code or fixture values were copied |
| `crates/rfkit-core/src/composition.rs` | Matched-junction power-wave equations; scikit-rf and SciPy behavior references | N/A for the independent composition; `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) for behavior | Explicit target-grid composition: public `Network.interpolate(..., basis="s", coords="cart", kind="linear")` semantics followed by the existing matched real-junction connection semantics | REWRITE | Mathematical reference; scikit-rf BSD-3-Clause; SciPy BSD-3-Clause | Private raw-array wrapper with deterministic A-interpolation → B-interpolation → matched-connection failure attribution; target selection/intersection/subsetting/extrapolation is not added, and existing interpolation/connection kernels are reused without mathematical changes |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_connect_matched_explicit_grid_three_to_four_port_complex_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network.interpolate(..., basis="s", coords="cart", kind="linear")` and `skrf.network.connect` behavior | REWRITE | BSD-3-Clause | Independent three-port + four-port direct inputs with source seeds 20260943/20260944, distinct irregular source grids, explicit in-range target grid, exact 73.5 Ω matched real junction, complex frequency/port-dependent external z0, and A-survivors-then-B-survivors ordering; both connected S and z0 outputs use numeric tolerance, all inputs/grids/metadata are exact; no source code or fixture values were copied |
| `crates/rfkit-core/src/connection.rs`; `crates/rfkit-core/src/lib.rs` | Matched-junction power-wave equations; scikit-rf behavior reference | N/A for the equations; `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) for behavior | Independent Kurokawa matched-real-junction elimination of one A port and one B port; public `skrf.network.connect` behavior is used only as the differential oracle | REWRITE | Mathematical reference; scikit-rf BSD-3-Clause | Private frequency-major N-port kernel; junction z0 is finite, real, strictly positive, and exactly equal at each frequency, external z0 may be finite complex, output order is A survivors then B survivors, and only exact-zero denominator is singular; no source code copied |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_connect_matched_three_to_four_port_real_frequency_dependent_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `skrf.Network(..., s_def="power")` constructors and `skrf.network.connect(network_a, port_a, network_b, port_b)` | REWRITE | BSD-3-Clause | One independent three-port + four-port, three-frequency fixture with local seeds 20260939/20260940, non-50 Ω frequency-dependent real-positive exactly matched junction z0, non-trivial external z0, explicit port-order metadata, and output-only numeric tolerance; no source code or fixture values copied |
| `crates/rfkit-core/src/connection.rs` | Matched-junction power-wave equations; scikit-rf behavior reference | N/A for the equations; `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) for behavior | Independent same-network elimination of selected ports `k` and `l` using `S_EE + S_EI (I - P S_II)^-1 P S_IE`; public `skrf.network.innerconnect` is used only as the differential behavior oracle | REWRITE | Mathematical reference; scikit-rf BSD-3-Clause | Private frequency-major N-port kernel; selected junction z0 is finite, real, strictly positive, and exactly equal at each frequency, external z0 may be finite complex, output is the original survivor order, exact-zero solver pivots are the only singularity criterion, and no source code was copied |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_inner_connect_matched_five_port_real_frequency_dependent_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Explicit `skrf.Network(..., s_def="power")` constructor and public `skrf.network.innerconnect(network, k, l)` | REWRITE | BSD-3-Clause | One independent five-port, three-frequency fixture with seed 20260941, non-adjacent ports 1 and 3, non-50 Ω frequency-dependent real-positive exactly matched junction z0, non-trivial surviving-port data, explicit survivor-order metadata, and output-only numeric tolerance; no source code or fixture values copied |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/port_permutation_three_port_complex_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network.renumbered(order, list(range(nport)))` behavior | REWRITE | BSD-3-Clause | One independently authored asymmetric three-port, three-frequency input with seed `20260947`, unequal complex per-port/frequency-dependent z0, and non-involutive `order=[2,0,1]`; frequency/S/z0 output is a pure exact reindexing copy and the metadata records the source mapping and exact canonical comparison policy; no source code or fixture values copied |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/mixed_mode_{forward,inverse}_five_port_complex_z0.json`; `crates/rfkit-core/tests/oracle_mixed_mode.rs` | Published V/I coordinate equations; scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`); NumPy `2.5.1` | Public `Network.se2gmm` / `Network.gmm2se` behavior for the two direction-specific expected S outputs | REWRITE | Mathematical reference; scikit-rf BSD-3-Clause | Independent asymmetric five-port/three-frequency fixtures with seeds `20260948` and `20260949`, `p=2`, adjacent positive/negative pairs, distinct complex equal pair references and one unpaired reference; inverse supplies explicit adjacent target `z0_se`. Metadata, shapes, inputs, frequencies, and references are exact; only S output uses strict `rtol=1e-12`/`atol=1e-12`. No source code, fixture values, or instrument material copied |
| `crates/rfkit-core/src/impedance_admittance.rs`; `crates/rfkit-core/src/linalg.rs` | Published network-parameter definitions (`Y=Z^-1`, `Z=Y^-1`) and Gaussian elimination with partial pivoting | N/A (mathematical definition) | N-port inversion kernel and wave-definition-independent exact-pivot solver | REWRITE | Mathematical reference | Private Rust implementation; the shared solver was extracted from the existing power-wave kernel, uses exact complex-zero pivot detection only, and adds no rank cutoff, regularization, condition-number test, or public API |
| `crates/rfkit-core/src/power_wave_admittance.rs` | K. Kurokawa, “Power Waves and the Scattering Matrix,” IEEE Transactions on Microwave Theory and Techniques 13(2), 194–202 (1965) | DOI `10.1109/TMTT.1965.1125964` | S↔Y behavior as the composition S↔Z power-wave conversion with Z↔Y inversion | REWRITE | IEEE publication (mathematical reference) | Private Rust composition of the existing verified kernels; no independent S/Y equation, source code, or public Network conversion API was copied |
| `crates/rfkit-core/src/power_waves.rs` | K. Kurokawa, “Power Waves and the Scattering Matrix,” IEEE Transactions on Microwave Theory and Techniques 13(2), 194–202 (1965) | DOI `10.1109/TMTT.1965.1125964` | Eq. (1) power-wave definitions, Eq. (18) Z-to-S relation, and Eq. (19) S-to-Z conversion | REWRITE | IEEE publication (mathematical reference) | Independent Rust implementation of the published equations; no source code copied |
| `crates/rfkit-core/src/power_waves.rs` (behavior reference) | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | `skrf/network.py::s2z`/`z2s`/`Network.renormalize`; conversion, renormalization, multiport, and complex-z0 tests in `skrf/tests/test_network.py` | REFERENCE | BSD-3-Clause | Used to check expected conversion and renormalization behavior while intentionally rewriting the algorithms and tests; no source code or fixture copied |
| `crates/rfkit-core/src/power_wave_admittance.rs` (behavior reference) | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `skrf.network.s2y(..., s_def="power")` and `skrf.network.y2s(..., s_def="power")` behavior | REFERENCE | BSD-3-Clause | Used only to generate direct differential expected outputs; Rust uses the existing S↔Z and Z↔Y kernels as a private composition, with no scikit-rf source code or fixture values copied |
| `crates/rfkit-core/src/power_waves.rs`; `crates/rfkit-core/tests/public_parameter_ingress.rs`; `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_y_to_s_three_port_rank_deficient_complex_z0.json` | Kurokawa power-wave equations; scikit-rf | Kurokawa DOI `10.1109/TMTT.1965.1125964`; scikit-rf `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Direct Y→S solve from `A=F(I+GY)` and `B=F(I-conj(G)Y)`; pinned public `skrf.network.y2s` expected output for a rank-deficient non-reciprocal N-port | REWRITE | IEEE publication; scikit-rf BSD-3-Clause | Equations and Rust kernel are independently implemented; the fixture uses seed `20260945`, three rank-two Y slices, complex per-port/frequency-dependent z0, and strict `rtol=1e-12`/`atol=1e-12` output-only comparison. No source code or fixture values were copied. |
| `crates/rfkit-core/src/power_waves.rs`; `crates/rfkit-core/src/lib.rs`; `crates/rfkit-core/tests/public_direct_s_to_y.rs`; `crates/rfkit-touchstone/tests/public_parameter_ingress.rs`; `crates/rfkit-touchstone/examples/modelled_parameter_touchstone.rs`; `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_s_to_y_three_port_singular_i_minus_s_complex_z0.json` | Kurokawa power-wave equations; scikit-rf | Kurokawa DOI `10.1109/TMTT.1965.1125964`; scikit-rf `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Issue #78 direct S→Y solve from `A=(S G + conj(G)) F` and `B=(I-S) F`; pinned public `skrf.network.s2y(..., s_def="power")` behavior for the singular-I-S direct fixture | REWRITE | IEEE publication; scikit-rf BSD-3-Clause | Independently authored Rust kernel/method, public and Touchstone workflow tests, executable usage, and oracle generator. The non-reciprocal three-port fixture uses seed `20260946`, complex per-port/frequency-dependent z0, and strict output-only tolerances; no scikit-rf source, opposite-direction fixture, or fixture values are copied. |
| `crates/rfkit-core/src/power_waves.rs`; `crates/rfkit-core/src/lib.rs`; `crates/rfkit-core/tests/public_direct_power_renormalization.rs`; `crates/rfkit-core/tests/public_power_renormalization.rs`; `crates/rfkit-touchstone/tests/public_parameter_ingress.rs`; `crates/rfkit-touchstone/examples/modelled_parameter_touchstone.rs` | Kurokawa power-wave equations; local multiple-RHS solver; scikit-rf behavior reference | Kurokawa DOI `10.1109/TMTT.1965.1125964`; scikit-rf commit `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Issue #80 direct source→target wave change and shared-domain comparison with existing public renormalization fixtures | REWRITE | IEEE publication; scikit-rf BSD-3-Clause | Direct `D/E/C/J` coefficients and right solve are independently authored from the equations (REWRITE), the existing local multiple-RHS solver is reused as operation-independent infrastructure, and scikit-rf is consulted only as a behavior reference. Existing pinned renormalization fixtures are reused without modification and no new oracle fixture is added. Tests cover singular-Z/Y workflows, complex and negative-real references, larger N-port data, direct invariants, and Touchstone writer/read; no source code or hand-copied expected values. |
| `crates/rfkit-core/src/lib.rs`; `crates/rfkit-core/tests/public_parameter_ingress.rs`; `crates/rfkit-core/examples/modelled_parameter_network.rs` | Published Kurokawa power-wave equations and local verified kernels; scikit-rf | Kurokawa DOI `10.1109/TMTT.1965.1125964`; scikit-rf commit `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public Z→S and explicitly composed Y→Z→S parameter-ingress adapters; pinned Z→S/Y→S fixtures and locally authored analytical/boundary tests | REWRITE | BSD-3-Clause for scikit-rf behavior reference; published mathematics | The constructors are independent Rust adapters that reuse existing local kernels. scikit-rf `Network.from_z`/parameter-setter behavior and `test_constructor_from_parameters*`/`test_zy_singularities` were inspected as references only; no source code or fixture values were copied. The local API intentionally rejects empty/mismatched axes before the kernel, preserves pointwise labels and explicit z0, and retains the composed Y singularity domain. |
| `crates/rfkit-core/src/termination.rs`; `crates/rfkit-core/src/lib.rs`; `crates/rfkit-core/tests/public_termination.rs` | Kurokawa power-wave equations plus physical boundary `V_k=-Z_L I_k`; scikit-rf behavior reference | Kurokawa DOI `10.1109/TMTT.1965.1125964`; scikit-rf commit `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Direct finite complex physical-load elimination for one selected port, with exact singular-denominator and finite-arithmetic diagnostics | REWRITE | BSD-3-Clause for scikit-rf behavior reference; published mathematics | Independent N-port Rust kernel and public adapter. The implementation evaluates `den=(Z_L+conj(z_k))-(Z_L-z_k)S_kk` directly, supports finite `d=Z_L+conj(z_k)==0` when `den` is nonzero, preserves survivor order/references, and does not form a mandatory `c/d`, invert S/Z/Y, renormalize, regularize, or copy third-party code. |
| `crates/rfkit-core/src/stability.rs`; `crates/rfkit-core/src/lib.rs`; `crates/rfkit-core/tests/public_two_port_stability.rs`; `crates/rfkit-core/tests/oracle_two_port_stability.rs`; `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/two_port_stability_power_four_frequency.json` | Rollett two-port stability equation; scikit-rf public behavior oracle | Published two-port stability mathematics; scikit-rf commit `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`); NumPy `2.5.1`; seed `20260954` | Sampled `delta=S11*S22-S12*S21` and optional finite K for exact two-port power-wave data; pinned public `Network.stability` supplies finite K and NumPy `linalg.det` supplies delta in the canonical fixture | REWRITE | Mathematical reference; scikit-rf BSD-3-Clause | The Rust kernel independently evaluates the formula and validates positive-real references, exact undefined transmission semantics, malformed serde shapes, and arithmetic representability. The seeded S input uses a deterministic base stack plus NumPy `default_rng` complex perturbation (scale `1e-3`); true 2×2 largest-singular-value checks derive one passive and three active classes. scikit-rf source is not copied; the unilateral/isolated `None` policy deliberately remains local and no non-standard JSON infinity is serialized. Only output delta/K use strict `rtol=1e-12`, `atol=1e-12`; all input/metadata fields are exact. |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_terminate_port_impedance_five_port_complex_z0.json`; `crates/rfkit-core/tests/oracle_termination.rs` | scikit-rf public `z2s`/`Network`/`connect` behavior | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`); NumPy `2.5.1`; seed `20260950` | Canonical finite physical-load differential fixture: asymmetric five-port source, selected port 2, explicit `[0, 38+12j, 73-9j]` ohm loads, and original-order survivors `[0,1,3,4]` | REWRITE | BSD-3-Clause | Inputs and metadata are independently authored and generated through public scikit-rf calls; only `s_terminated` is numeric-tolerance output (`rtol=1e-12`, `atol=1e-12`). No scikit-rf source, output values, or fixture values were copied; Rust compares only expected S while deriving survivor z0 independently. |
| `crates/rfkit-core/src/direct_connection.rs`; `crates/rfkit-core/src/lib.rs`; `crates/rfkit-core/tests/oracle_direct_connection.rs` | Kurokawa power-wave equations plus physical V/I junction conditions; scikit-rf public behavior reference | Kurokawa DOI `10.1109/TMTT.1965.1125964`; scikit-rf `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Issue #88 direct one-port A/B physical junction solve for unequal finite nonzero-real references | REWRITE | Published mathematics; scikit-rf BSD-3-Clause | Independent N-port implementation and public adapter. The direct solve preserves the signed `Re(z)` factor, exact frequency grid, A-then-B survivor order, and exact survivor references; it does not insert a mismatch network, convert S/Z/Y, renormalize implicitly, or copy scikit-rf source. |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_connect_direct_three_to_four_port_complex_z0.json` | scikit-rf public `Network` constructor and `skrf.network.connect` behavior | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`); NumPy `2.5.1`; seed `20260951` | Canonical direct physical-junction differential fixture: asymmetric non-reciprocal 3-port A + 4-port B, selected A[1]/B[2], unequal complex frequency-dependent positive-real references | REWRITE | BSD-3-Clause | Inputs are independently generated; expected S uses one public `network.connect` call, while z0/order/metadata are exact contract fields. The 3+4 shape avoids scikit-rf two-port insertion ambiguity; no source code, output values, or fixture values were copied. |
| `crates/rfkit-touchstone/tests/public_direct_connection_workflow.rs`; `crates/rfkit-touchstone/examples/connect_direct_power_touchstone.rs` | Kurokawa physical V/I equations; IBIS Open Forum Touchstone File Format Specification | Kurokawa DOI `10.1109/TMTT.1965.1125964`; Touchstone Format Specification v2.1 (2024-01-26) | Two v1.0 inputs at different common references → direct connection → independent physical check → explicit writer-compatible renormalization → write/read | REWRITE | Published mathematics; IBIS Open Forum specification | Workflow inputs and physical reduction check are independently authored. The existing writer is called only after explicit direct renormalization to one common positive-real reference; no automatic writer repair, format extension, source copying, or broad compatibility promise is introduced. |
| `crates/rfkit-touchstone/examples/terminate_port_touchstone.rs`; `crates/rfkit-touchstone/tests/public_termination_workflow.rs` | Kurokawa physical boundary equations; Touchstone v1.0 specification | Kurokawa DOI `10.1109/TMTT.1965.1125964`; IBIS Open Forum Touchstone File Format Specification v2.1 (2024-01-26) | Focused ingress → finite termination → independent reduction check → writer/read workflow with common 50 Ω survivors | REWRITE | Published mathematics; IBIS Open Forum specification | Literal asymmetric five-port text and analytical reduction are independently authored. The existing writer is used with its declared common finite positive-real reference contract; no hidden writer renormalization, mixed-mode extension, source copying, or broader Touchstone promise is introduced. |

Before adding copied or closely adapted third-party code or fixtures, record an entry here and preserve the applicable license notice under `THIRD_PARTY_LICENSES/`.

The Touchstone reader and its synthetic text cases are independent REWRITEs.
The format boundary follows the IBIS Open Forum Touchstone File Format
Specification Version 2.1 (January 2024), specifically the rules that define
the historical v1.0 syntax. The implementation does not copy specification
examples or scikit-rf code. scikit-rf 2.0.1 at commit
`bd651e923cac6020de49a096e1d7e9b5f949f884` is consulted only for behavior and
for one independently authored, mutually supported oracle input; deliberate
v2/noise/vendor-extension rejections remain specification-driven Rust tests.
The Touchstone writer is likewise an independent REWRITE of the v1.0 physical
record rules. Its fixed RI/Hz output, common real-positive reference contract,
and shortest-round-tripping binary64 formatting are local policy decisions
recorded in Issue #72; scikit-rf is used only as a pinned interoperability
oracle through the public parser, not as a source of writer code or fixtures.

## Required record format

| Local path | Source project | Source commit/tag | Source path | Use | License | Notes |
|---|---|---|---|---|---|---|
| _example_ | rust-rf | `<sha>` | `src/network.rs` | ADAPT | BSD-3-Clause | Renormalization math rewritten around local API |
| `crates/rfkit-touchstone/src/lib.rs`; `crates/rfkit-touchstone/tests/public_parser.rs` | IBIS Open Forum Touchstone File Format Specification | Version 2.1 (2024-01-26) | v1.0 option-line, comment, pair-format, frequency, and row/continuation rules (pp. 4–9, 14–19) | REWRITE | IBIS Open Forum specification; no text/code copied | Pure parser and synthetic inputs independently authored; the supported real-scalar-R S subset is explicit and reversible |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/touchstone_v1_0_s_three_port.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `skrf.io.touchstone.Touchstone` parser behavior | REWRITE | BSD-3-Clause | Deterministic text and expected frequency/S/z0 are independently authored and generated through the pinned public parser; no source or upstream fixture copied |
| `crates/rfkit-touchstone/src/writer.rs`; `crates/rfkit-touchstone/tests/public_writer.rs`; `tools/oracle/check_touchstone_writer.py`; `crates/rfkit-touchstone/examples/touchstone_writer_oracle.rs` | IBIS Open Forum Touchstone File Format Specification; scikit-rf | Version 2.1 (2024-01-26); `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | v1.0 S/RI/Hz option line, 2-port order, row-major and four-pair physical-line/continuation rules; public scikit-rf reader interoperability | REWRITE | IBIS specification; scikit-rf BSD-3-Clause | Independently authored deterministic writer, contract tests, and Rust-generated two-/five-port oracle inputs; no code or fixture values copied. The oracle compares frequency/S/z0 after feeding actual Rust output to pinned scikit-rf. |

`Use` should be one of:

- `REFERENCE`: behavior inspected; no copyrightable code copied
- `REUSE`: code/fixture copied substantially as-is
- `ADAPT`: code/fixture closely modified from source
- `REWRITE`: independent implementation based on specification/math; prior art consulted only for behavior

If in doubt, attribute rather than hide provenance.
