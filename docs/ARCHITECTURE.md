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
rfkit-touchstone optional dedicated parser/writer if I/O grows large
rfkit-cal        calibration/de-embedding algorithms
rfkit-plot       backend-neutral Smith/plot data and optional renderers
rfkit-python     PyO3 binding
rfkit-wasm       wasm-bindgen binding
```

Do not split crates merely for aesthetics. Split when dependency boundaries or compile-time/platform isolation justify it.

## API philosophy

- typed, fallible operations
- explicit units at boundaries
- owned core model first; add zero-copy/view APIs only when profiling justifies complexity
- numerical semantics documented independently of scikit-rf naming
