# Third-party provenance

The oracle harness and both checked-in fixtures are independent rewrites. They
use the public `scikit-rf==2.0.1` `Network` constructor and `Network.z`
behavior/API as a numerical reference, but do not copy scikit-rf source code,
conversion code, or fixture values. The S-to-Z expected values are generated
through the public `Network.z` property after constructing the shared input
Network.

| Local path | Source project | Source commit/tag | Source path | Use | License | Notes |
|---|---|---|---|---|---|---|
| `tools/oracle/generate_oracle.py` | scikit-rf | `v2.0.1` | `skrf.Network` constructor and public `Network.z` API | REWRITE | BSD-3-Clause | Independent deterministic generator; behavior/API reference only; no source code or fixture copied |
| `tools/oracle/fixtures/three_port_complex_z0.json`; `tools/oracle/fixtures/power_wave_s_to_z_three_port_complex_z0.json` | scikit-rf | `v2.0.1` | Public `Network` read-back and `Network.z` behavior | REWRITE | BSD-3-Clause | Independently generated canonical outputs; no fixture values copied |

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
