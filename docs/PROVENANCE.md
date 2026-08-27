# Third-party provenance

The oracle harness and all checked-in fixtures are independent rewrites. They
use the public `scikit-rf==2.0.1` `Network` constructor and relevant public
Network behavior/APIs as numerical references, but do not copy scikit-rf source
code, conversion code, or fixture values. The S-to-Z expected values are
generated through the public `Network.z` property after constructing the
shared input Network. The Z-to-S fixture input is a direct deterministic local
rewrite generated with its recorded dedicated RNG seed; only its expected S
output comes from the public `Network.from_z(..., s_def="power")` constructor
and `Network.s` property. The renormalization fixture uses explicit local
source/target z0 arrays and obtains the expected output only by calling the
public `Network.renormalize(..., s_def="power")` operation, then reading
`Network.s` and `Network.z0`. The internal power-wave S-to-Z, Z-to-S, and
renormalization kernels are independent rewrites of the published equations
and do not copy implementation code.

| Local path | Source project | Source commit/tag | Source path | Use | License | Notes |
|---|---|---|---|---|---|---|
| `tools/oracle/generate_oracle.py` | scikit-rf | `v2.0.1` | `skrf.Network` constructor, public `Network.z` and `Network.renormalize` APIs, and public `Network.from_z`/`Network.s` API | REWRITE | BSD-3-Clause | Independent deterministic generator; direct Z inputs and renormalization source/target z0 use local deterministic construction; scikit-rf is a behavior/API reference only; no source code or fixture copied |
| `tools/oracle/fixtures/three_port_complex_z0.json`; `tools/oracle/fixtures/power_wave_s_to_z_three_port_complex_z0.json`; `tools/oracle/fixtures/power_wave_z_to_s_three_port_complex_z0.json` | scikit-rf | `v2.0.1` | Public `Network` read-back, `Network.z`, and `Network.from_z`/`Network.s` behavior | REWRITE | BSD-3-Clause | Independently generated canonical outputs; Z-to-S input is direct deterministic data and expected S is obtained through public `Network.from_z`/`Network.s`; no fixture values copied |
| `tools/oracle/fixtures/power_wave_s_to_z_{one,two,four,eight}_port_*.json`; `tools/oracle/fixtures/power_wave_z_to_s_{one,two,four,eight}_port_*.json` | scikit-rf | `v2.0.1` | Public `Network.z` and `Network.from_z(..., s_def="power").s` behavior | REWRITE | BSD-3-Clause | Independently generated deterministic matrix cases covering 1/2/4/8 ports and scalar/per-port/frequency-dependent real/complex z0; inputs use local RNG and conservative generation-time diagonal-dominance checks; no source code or fixture values copied |
| `tools/oracle/generate_oracle.py`; `tools/oracle/fixtures/power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0.json` | scikit-rf | `bd651e923cac6020de49a096e1d7e9b5f949f884` (`v2.0.1`) | Public `Network` constructor, `Network.renormalize(..., s_def="power")`, `Network.s`, and `Network.z0` APIs | REWRITE | BSD-3-Clause | Four-port deterministic local S input and explicit complex per-port frequency-dependent source/target z0; expected output is obtained only through public renormalization behavior, with no source code or fixture values copied |
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
