#!/usr/bin/env python3
"""Generate and check the checked-in scikit-rf oracle fixture.

The fixture is intentionally small, but exercises a frequency-dependent
three-port Network with complex, per-port reference impedances.  This module
is kept independent of the Rust implementation so it can serve as a stable
reference when the Rust fixture reader and numerical operations are added.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


EXPECTED_NUMPY_VERSION = "2.5.1"
EXPECTED_SCIKIT_RF_VERSION = "2.0.1"
RANDOM_SEED = 20_250_308
SCHEMA_VERSION = 1
DEFAULT_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "three_port_complex_z0.json"
)


def _load_dependencies() -> tuple[Any, Any]:
    """Import the pinned dependencies and fail with an actionable message."""

    # Check distribution metadata before importing NumPy/scikit-rf.  This
    # keeps an incompatible installation (including an ABI-incompatible
    # NumPy) on the explicit wrong-version failure path.
    try:
        installed_numpy = version("numpy")
        installed_skrf = version("scikit-rf")
    except PackageNotFoundError as error:  # pragma: no cover - clean env path
        package = getattr(error, "name", "a required package")
        raise RuntimeError(
            f"{package} is not installed; create an isolated environment and run "
            "`python -m pip install -r requirements.txt`"
        ) from error

    requirements = Path(__file__).resolve().parent / "requirements.txt"
    distribution_version_errors: list[str] = []
    if installed_numpy != EXPECTED_NUMPY_VERSION:
        distribution_version_errors.append(
            f"numpy=={EXPECTED_NUMPY_VERSION} (found {installed_numpy})"
        )
    if installed_skrf != EXPECTED_SCIKIT_RF_VERSION:
        distribution_version_errors.append(
            f"scikit-rf=={EXPECTED_SCIKIT_RF_VERSION} (found {installed_skrf})"
        )
    if distribution_version_errors:
        raise RuntimeError(
            "wrong oracle dependency version(s): "
            + ", ".join(distribution_version_errors)
            + f"; install the exact pins from {requirements}"
        )

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - exercised in a clean env
        raise RuntimeError(
            "numpy is not installed; create an isolated environment and run "
            "`python -m pip install -r requirements.txt`"
        ) from error

    try:
        import skrf
    except ImportError as error:  # pragma: no cover - exercised in a clean env
        raise RuntimeError(
            "scikit-rf is not installed; create an isolated environment and run "
            "`python -m pip install -r requirements.txt`"
        ) from error

    actual_numpy = np.__version__
    actual_skrf = skrf.__version__
    version_errors: list[str] = []
    if actual_numpy != EXPECTED_NUMPY_VERSION:
        version_errors.append(f"numpy=={EXPECTED_NUMPY_VERSION} (found {actual_numpy})")
    if actual_skrf != EXPECTED_SCIKIT_RF_VERSION:
        version_errors.append(
            f"scikit-rf=={EXPECTED_SCIKIT_RF_VERSION} (found {actual_skrf})"
        )
    if version_errors:
        raise RuntimeError(
            "wrong oracle dependency version(s): "
            + ", ".join(version_errors)
            + f"; install the exact pins from {requirements}"
        )

    return np, skrf


def _complex_value(value: complex) -> dict[str, float]:
    """Represent a complex scalar without relying on JSON extensions."""

    return {"imag": float(value.imag), "real": float(value.real)}


def _complex_array(values: Any) -> Any:
    """Convert an arbitrary NumPy complex array to JSON-native nested values."""

    if values.ndim == 0:
        return _complex_value(complex(values))
    if values.ndim == 1:
        return [_complex_value(complex(item)) for item in values]
    return [_complex_array(row) for row in values]


def _network_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build the deterministic Network and return its canonical data model."""

    # The seed is part of the fixture contract.  A local Generator avoids
    # mutating NumPy's process-global RNG state and is stable for this pinned
    # NumPy release.
    rng = np.random.default_rng(RANDOM_SEED)
    frequency_hz = np.array([1.0e9, 1.5e9, 2.0e9, 2.5e9], dtype=np.float64)
    nports = 3

    # Deliberately use a non-symmetric complex matrix so this remains useful
    # for future N-port and non-reciprocal operation checks.
    s = (
        rng.normal(loc=0.0, scale=0.1, size=(frequency_hz.size, nports, nports))
        + 1j * rng.normal(loc=0.0, scale=0.1, size=(frequency_hz.size, nports, nports))
    ).astype(np.complex128)

    # Per-port values vary with frequency and have non-zero imaginary parts.
    # Network accepts (frequency, port) z0, which preserves both dimensions.
    base_real = np.array([50.0, 60.0, 75.0], dtype=np.float64)
    slope_real = np.array([1.5, 2.0, 2.5], dtype=np.float64)
    base_imag = np.array([1.2, -0.8, 2.4], dtype=np.float64)
    z0 = (
        base_real[None, :]
        + np.arange(frequency_hz.size, dtype=np.float64)[:, None] * slope_real[None, :]
        + 1j
        * (
            base_imag[None, :]
            + 0.25 * np.arange(frequency_hz.size, dtype=np.float64)[:, None]
        )
    ).astype(np.complex128)

    network = skrf.Network(
        f=frequency_hz,
        s=s,
        z0=z0,
        s_def="power",
        name="three_port_complex_z0",
    )

    # Read values back through Network rather than serializing the pre-
    # constructor arrays.  This makes the fixture explicitly an oracle output.
    frequency = np.asarray(network.f, dtype=np.float64)
    network_s = np.asarray(network.s, dtype=np.complex128)
    network_z0 = np.asarray(network.z0, dtype=np.complex128)
    shape = {
        "frequency": list(frequency.shape),
        "s": list(network_s.shape),
        "z0": list(network_z0.shape),
    }

    return {
        "metadata": {
            "case_id": "three_port_complex_z0",
            "numpy_version": np.__version__,
            "operation": "network_fixture",
            "random_seed": RANDOM_SEED,
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": shape,
            "tolerance_policy": {
                "comparison": "exact canonical UTF-8 JSON bytes",
                "floating_point": "IEEE-754 binary64 values serialized by Python json",
                "numeric_tolerance": (
                    "not applicable to regeneration; downstream numerical comparisons "
                    "must define operation-specific tolerances"
                ),
            },
            "wave_definition": network.s_def,
        },
        "data": {
            "frequency_hz": [float(value) for value in frequency],
            "s": _complex_array(network_s),
            "z0_ohm": _complex_array(network_z0),
        },
    }


def _canonical_bytes(document: dict[str, Any]) -> bytes:
    """Serialize a fixture using the repository's byte-stable JSON policy."""

    # allow_nan=False rejects NaN and infinities instead of emitting the
    # non-standard JSON tokens accepted by Python's default encoder.
    text = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        separators=(",", ": "),
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def _write_fixture(path: Path, expected: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)


def _check_fixture(path: Path, expected: bytes) -> int:
    try:
        actual = path.read_bytes()
    except FileNotFoundError:
        print(f"fixture missing: {path}; run `generate_oracle.py write`", file=sys.stderr)
        return 1

    if actual != expected:
        print(
            f"fixture differs from regenerated canonical output: {path}\n"
            "run `generate_oracle.py write` only when intentionally updating the fixture",
            file=sys.stderr,
        )
        return 1

    print(f"fixture check passed: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("check", "write"),
        help="check the checked-in canonical bytes or write regenerated bytes",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help=f"fixture path (default: {DEFAULT_FIXTURE})",
    )
    args = parser.parse_args(argv)

    try:
        np, skrf = _load_dependencies()
        expected = _canonical_bytes(_network_fixture(np, skrf))
        if args.mode == "write":
            _write_fixture(args.fixture, expected)
            print(f"fixture written: {args.fixture}")
            return 0
        return _check_fixture(args.fixture, expected)
    except RuntimeError as error:
        print(f"oracle setup error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"oracle fixture I/O error: {error}", file=sys.stderr)
        return 2
    except (TypeError, ValueError) as error:
        print(f"oracle generation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
