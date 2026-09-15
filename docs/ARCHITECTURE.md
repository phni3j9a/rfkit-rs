# Architecture

## Core model

The core representation is frequency-major N-port data:

- `Frequency`: 1-D frequency axis in Hz
- `Network.s`: `(frequency, port_out, port_in)` complex scattering matrix
- `Network.z0`: `(frequency, port)` complex reference impedance

The exact public representation may evolve, but these invariants should remain explicit.

## Planned crate boundaries

```text
rfkit-core       numerical network analysis + file-independent RF math
rfkit-touchstone dedicated pure in-memory Touchstone parser (v1.0 S subset)
rfkit-cal        calibration/de-embedding algorithms
rfkit-plot       backend-neutral Smith/plot data and optional renderers
rfkit-python     PyO3 binding
rfkit-wasm       wasm-bindgen binding
```

Do not split crates merely for aesthetics. Split when dependency boundaries or compile-time/platform isolation justify it.

`rfkit-touchstone` currently depends toward `rfkit-core` and owns only the
bounded Touchstone 1.0 single-ended S-parameter text reader. It has no
filesystem, async, instrument, or metadata responsibilities; callers provide
text obtained through their own I/O layer. Touchstone v2.x, writers, and
vendor-specific extensions remain outside this boundary until a separately
verified design justifies them.

## API philosophy

The implemented public `Network` baseline and its Green/Yellow/Red extension envelope are defined in `docs/PUBLIC_API.md`. Treat that document and `docs/LOOP_ENGINEERING.md` as the policy boundary for public RF-operation work.

- typed, fallible operations
- explicit units at boundaries
- owned core model first; add zero-copy/view APIs only when profiling justifies complexity
- numerical semantics documented independently of scikit-rf naming
- `Network`-centric methods for operations with one natural primary network
- explicit frequency-grid policy; no hidden alignment or interpolation defaults
- wave-sensitive public names remain explicit until a typed wave-definition model is justified by actual multi-convention support
- avoid speculative traits, wrappers, and parameter hierarchies before concrete use requires them
- permit bounded reversible Green/Yellow evolution before stabilization while reserving canonical-model replacement and irreversible commitments for Red human decisions
