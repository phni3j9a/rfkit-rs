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

| Local path | Source project | Source commit/tag | Source path | Use | License | Notes |
|---|---|---|---|---|---|---|
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_renormalize_{one,two,eight}_port_*.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network` constructor, `Network.renormalize(..., s_def="power")`, `Network.s`, and `Network.z0` APIs | REWRITE | BSD-3-Clause | Three additional deterministic local S inputs covering real scalar, complex per-port constant, and real frequency-dependent same-across-port source/target z0; expected output is obtained only through public renormalization behavior, with no source code or fixture values copied |
| `tools/oracle/generate_oracle.py` | scikit-rf | `v2.0.1` | `skrf.Network` constructor, public `Network.z` and `Network.renormalize` APIs, and public `Network.from_z`/`Network.s` API | REWRITE | BSD-3-Clause | Independent deterministic generator; direct Z inputs and renormalization source/target z0 use local deterministic construction; scikit-rf is a behavior/API reference only; no source code or fixture copied |
| `tools/oracle/fixtures/three_port_complex_z0.json`; `tools/oracle/fixtures/power_wave_s_to_z_three_port_complex_z0.json`; `tools/oracle/fixtures/power_wave_z_to_s_three_port_complex_z0.json` | scikit-rf | `v2.0.1` | Public `Network` read-back, `Network.z`, and `Network.from_z`/`Network.s` behavior | REWRITE | BSD-3-Clause | Independently generated canonical outputs; Z-to-S input is direct deterministic data and expected S is obtained through public `Network.from_z`/`Network.s`; no fixture values copied |
| `tools/oracle/fixtures/power_wave_s_to_z_{one,two,four,eight}_port_*.json`; `tools/oracle/fixtures/power_wave_z_to_s_{one,two,four,eight}_port_*.json` | scikit-rf | `v2.0.1` | Public `Network.z` and `Network.from_z(..., s_def="power").s` behavior | REWRITE | BSD-3-Clause | Independently generated deterministic matrix cases covering 1/2/4/8 ports and scalar/per-port/frequency-dependent real/complex z0; inputs use local RNG and conservative generation-time diagonal-dominance checks; no source code or fixture values copied |
| `tools/oracle/fixtures/power_wave_s_to_z_three_port_reciprocal_real_equal_z0.json`; `tools/oracle/fixtures/power_wave_z_to_s_three_port_reciprocal_real_equal_z0.json`; `tools/oracle/fixtures/power_wave_renormalize_three_port_reciprocal_real_equal_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network.z`, `Network.from_z(..., s_def="power").s`, and `Network.renormalize(..., s_def="power")` APIs; pinned NumPy SVD is used only to record passive evidence | REWRITE | BSD-3-Clause | Three independent deterministic three-port reciprocal cases reused without new fixture data; local RNG seeds are 20260921/20260922/20260923, inputs mirror one complex triangle without conjugation, all S/Z cases use explicit real-positive 61.25 Ω equal-across-port/frequency z0, and renormalization uses real-positive equal-across-port/frequency 42.75 Ω → 86.5 Ω. Passive maxima are rounded to 12 decimals in metadata; Rust uses a Frobenius upper-bound certificate. No source code or fixture values were copied |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_s_to_z_three_port_active_real_equal_z0.json`; `tools/oracle/fixtures/power_wave_z_to_s_three_port_active_real_equal_z0.json`; `tools/oracle/fixtures/power_wave_renormalize_three_port_active_real_equal_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network.z`, `Network.from_z(..., s_def="power").s`, and `Network.renormalize(..., s_def="power")` APIs; unrounded NumPy SVD is used only for active validation, with the recorded minimum rounded to 12 decimal places | REWRITE | BSD-3-Clause | Three independent deterministic three-port active cases with real equal-per-port 57.25 Ω z0 (renormalization target 91.75 Ω), local RNG seeds 20260924/20260925/20260926, and required sigma-max minimum 1.2. S-to-Z/renormalization use direct active S; Z-to-S uses direct negative-resistance Z and never round-trips expected S; no source code or fixture values copied |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network` constructor, `Network.renormalize(..., s_def="power")`, `Network.s`, and `Network.z0` APIs | REWRITE | BSD-3-Clause | Four-port deterministic local S input and explicit complex per-port frequency-dependent source/target z0; expected output is obtained only through public renormalization behavior, with no source code or fixture values copied |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_s_to_z_three_port_near_singular_real_equal_z0.json`; `tools/oracle/fixtures/power_wave_z_to_s_three_port_near_singular_real_equal_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network.z` and `Network.from_z(..., s_def="power").s` APIs; inputs are independent local direct constructions and metadata records binary upper-triangular determinant evidence | REWRITE | BSD-3-Clause | Two independent three-frequency, three-port cases with real-positive equal 64 Ω z0, seeds 20260927/20260928, and a binary-exact `2^-20` diagonal factor above scikit-rf's `EIG_COND=1e-9`; no source code, fixture values, expected-output round trip, or condition number was copied or recorded |
| `crates/rfkit-core/src/power_waves.rs` | K. Kurokawa, “Power Waves and the Scattering Matrix,” IEEE Transactions on Microwave Theory and Techniques 13(2), 194–202 (1965) | DOI `10.1109/TMTT.1965.1125964` | Eq. (1) power-wave definitions, Eq. (18) Z-to-S relation, and Eq. (19) S-to-Z conversion | REWRITE | IEEE publication (mathematical reference) | Independent Rust implementation of the published equations; no source code copied |
| `crates/rfkit-core/src/power_waves.rs` (behavior reference) | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | `skrf/network.py::s2z`/`z2s`/`Network.renormalize`; conversion, renormalization, multiport, and complex-z0 tests in `skrf/tests/test_network.py` | REFERENCE | BSD-3-Clause | Used to check expected conversion and renormalization behavior while intentionally rewriting the algorithms and tests; no source code or fixture copied |

Before adding copied or closely adapted third-party code or fixtures, record an entry here and preserve the applicable license notice under `THIRD_PARTY_LICENSES/`.

## Required record format

| Local path | Source project | Source commit/tag | Source path | Use | License | Notes |
|---|---|---|---|---|---|---|
| _example_ | rust-rf | `<sha>` | `src/network.rs` | ADAPT | BSD-3-Clause | Renormalization math rewritten around local API |

`Use` should be one of:

- `REFERENCE`: behavior inspected; no copyrightable code copied
- `REUSE`: code/fixture copied substantially as-is
- `ADAPT`: code/fixture closely modified from source
- `REWRITE`: independent implementation based on specification/math; prior art consulted only for behavior

If in doubt, attribute rather than hide provenance.
