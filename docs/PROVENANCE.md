# Third-party provenance

The initial scaffold contains no source code copied from scikit-rf, rust-rf, or rust-skrf.

The oracle harness is an independent rewrite. It uses the public
`scikit-rf==2.0.1` `Network` constructor and read-back behavior as a reference,
but does not copy source code or fixture values.

| Local path | Source project | Source commit/tag | Source path | Use | License | Notes |
|---|---|---|---|---|---|---|
| `tools/oracle/generate_oracle.py` | scikit-rf | `2.0.1` | `skrf.Network` public API | REWRITE | BSD-3-Clause | Behavior/API reference only; no source code or fixture copied |

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
