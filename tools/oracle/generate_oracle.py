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
from typing import Any, Callable, NamedTuple


EXPECTED_NUMPY_VERSION = "2.5.1"
EXPECTED_SCIKIT_RF_VERSION = "2.0.1"
RANDOM_SEED = 20_250_308
SCHEMA_VERSION = 1
DEFAULT_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "three_port_complex_z0.json"
)
S_TO_Z_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_s_to_z_three_port_complex_z0.json"
)

S_TO_Z_RTOL = 1e-12
S_TO_Z_ATOL_OHM = 1e-12
S_TO_Z_TOLERANCE_JUSTIFICATION = (
    "Strict binary64 tolerance for this well-conditioned, modest-magnitude "
    "deterministic case; it allows normal cross-language linear-algebra rounding "
    "while catching material disagreement."
)


class _OracleCase(NamedTuple):
    """A registered canonical fixture and its deterministic document builder."""

    case_id: str
    path: Path
    builder: Callable[[Any, Any], dict[str, Any]]


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


def _network_inputs(np: Any) -> tuple[Any, Any, Any]:
    """Build the shared deterministic frequency, S, and z0 input arrays."""

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

    return frequency_hz, s, z0


def _build_network(np: Any, skrf: Any) -> Any:
    """Construct the shared deterministic scikit-rf Network input."""

    frequency_hz, s, z0 = _network_inputs(np)

    return skrf.Network(
        f=frequency_hz,
        s=s,
        z0=z0,
        s_def="power",
        name="three_port_complex_z0",
    )


def _network_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build the deterministic Network and return its canonical data model."""

    network = _build_network(np, skrf)

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


def _s_to_z_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build the power-wave S-to-Z operation fixture from the shared Network."""

    network = _build_network(np, skrf)

    # Obtain the expected operation output exclusively through scikit-rf's
    # public Network.z property after constructing the Network.  Do not replace
    # this with a local conversion formula: this document is a behavior oracle.
    frequency = np.asarray(network.f, dtype=np.float64)
    network_s = np.asarray(network.s, dtype=np.complex128)
    network_z0 = np.asarray(network.z0, dtype=np.complex128)
    network_z = np.asarray(network.z, dtype=np.complex128)

    shape = {
        "frequency": list(frequency.shape),
        "input_s": list(network_s.shape),
        "input_z0": list(network_z0.shape),
        "output_z": list(network_z.shape),
    }

    return {
        "metadata": {
            "case_id": "power_wave_s_to_z_three_port_complex_z0",
            "input_case_id": "three_port_complex_z0",
            "numpy_version": np.__version__,
            "operation": "s_to_z",
            "random_seed": RANDOM_SEED,
            "reference_impedance": {
                "complex": True,
                "frequency_dependent": True,
                "per_port": True,
                "unit": "ohm",
            },
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": shape,
            "tolerance_policy": {
                "atol_ohm": S_TO_Z_ATOL_OHM,
                "comparison": (
                    "abs(actual-expected) <= "
                    "atol_ohm + rtol*abs(expected)"
                ),
                "justification": S_TO_Z_TOLERANCE_JUSTIFICATION,
                "regeneration": "exact canonical UTF-8 JSON bytes",
                "rtol": S_TO_Z_RTOL,
            },
            "wave_definition": network.s_def,
        },
        "data": {
            "frequency_hz": [float(value) for value in frequency],
            "s": _complex_array(network_s),
            "z0_ohm": _complex_array(network_z0),
            "z_ohm": _complex_array(network_z),
        },
    }


_CASES = (
    _OracleCase("three_port_complex_z0", DEFAULT_FIXTURE, _network_fixture),
    _OracleCase(
        "power_wave_s_to_z_three_port_complex_z0",
        S_TO_Z_FIXTURE,
        _s_to_z_fixture,
    ),
)
_CASES_BY_ID = {case.case_id: case for case in _CASES}


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


def _selected_cases(case_id: str | None, fixture: Path | None) -> tuple[_OracleCase, ...]:
    """Resolve the requested cases while retaining the old fixture override."""

    if case_id is not None:
        return (_CASES_BY_ID[case_id],)

    # Before operation cases were registered, --fixture selected the sole
    # network fixture.  Keep that invocation useful; --case selects an
    # operation fixture when an alternate path is needed.
    if fixture is not None:
        return (_CASES_BY_ID["three_port_complex_z0"],)

    return _CASES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("check", "write"),
        help="check the checked-in canonical bytes or write regenerated bytes",
    )
    parser.add_argument(
        "--case",
        choices=tuple(_CASES_BY_ID),
        help="select one case (default: check or write every registered case)",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        help=(
            "override the selected fixture path; without --case this retains "
            f"the legacy network-case behavior (default: all registered paths)"
        ),
    )
    args = parser.parse_args(argv)

    try:
        np, skrf = _load_dependencies()
        cases = _selected_cases(args.case, args.fixture)
        status = 0
        for case in cases:
            path = args.fixture if args.fixture is not None and len(cases) == 1 else case.path
            expected = _canonical_bytes(case.builder(np, skrf))
            if args.mode == "write":
                _write_fixture(path, expected)
                print(f"fixture written: {path}")
            else:
                status = max(status, _check_fixture(path, expected))
        return status
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
