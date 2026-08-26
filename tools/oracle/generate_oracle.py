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
import math
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
Z_TO_S_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_z_to_s_three_port_complex_z0.json"
)

S_TO_Z_RTOL = 1e-12
S_TO_Z_ATOL_OHM = 1e-12
S_TO_Z_TOLERANCE_JUSTIFICATION = (
    "Strict binary64 tolerance for this well-conditioned, modest-magnitude "
    "deterministic case; it allows normal cross-language linear-algebra rounding "
    "while catching material disagreement."
)
Z_TO_S_RTOL = 1e-12
Z_TO_S_ATOL = 1e-12
Z_TO_S_RANDOM_SEED = 20_260_826
Z_TO_S_TOLERANCE_JUSTIFICATION = (
    "Strict binary64 tolerance for this well-conditioned, modest-magnitude, "
    "dimensionless deterministic case; it allows normal cross-language "
    "linear-algebra rounding while catching material disagreement."
)


class _OracleCase(NamedTuple):
    """A registered fixture, builder, and case-specific check strategy."""

    case_id: str
    path: Path
    builder: Callable[[Any, Any], dict[str, Any]]
    comparison: str
    numeric_output_key: str | None = None


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
                "regeneration": (
                    "canonical UTF-8 JSON serialization; z_ohm is checked "
                    "with the recorded numeric tolerance"
                ),
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


def _z_to_s_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build the power-wave Z-to-S operation fixture from direct Z inputs."""

    # Keep the operation input independent of Network.z: that property is a
    # floating-point matrix solve and is therefore unsuitable as an exact
    # fixture contract.  Reuse only the stable frequency/z0 construction from
    # the representative input case, and generate a separate deterministic,
    # non-symmetric, diagonally dominant complex Z array locally.
    frequency_hz, _unused_s, source_z0 = _network_inputs(np)
    rng = np.random.default_rng(Z_TO_S_RANDOM_SEED)
    nfreq = frequency_hz.size
    nports = source_z0.shape[1]
    source_z = (
        rng.normal(loc=0.0, scale=0.35, size=(nfreq, nports, nports))
        + 1j * rng.normal(loc=0.0, scale=0.35, size=(nfreq, nports, nports))
    ).astype(np.complex128)
    for frequency_index in range(nfreq):
        for port in range(nports):
            source_z[frequency_index, port, port] += (
                55.0
                + 3.5 * frequency_index
                + 5.0 * port
                + 1j * (2.0 + 0.25 * frequency_index - 0.5 * port)
            )

    # The expected output still comes exclusively from scikit-rf's public
    # Network.from_z constructor and public Network.s property.
    converted = skrf.Network.from_z(
        source_z,
        f=frequency_hz,
        z0=source_z0,
        s_def="power",
        name="power_wave_z_to_s_three_port_complex_z0",
    )
    converted_s = np.asarray(converted.s, dtype=np.complex128)

    shape = {
        "frequency": list(frequency_hz.shape),
        "input_z": list(source_z.shape),
        "input_z0": list(source_z0.shape),
        "output_s": list(converted_s.shape),
    }

    return {
        "metadata": {
            "case_id": "power_wave_z_to_s_three_port_complex_z0",
            "numpy_version": np.__version__,
            "operation": "z_to_s",
            "random_seed": Z_TO_S_RANDOM_SEED,
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
                "atol": Z_TO_S_ATOL,
                "comparison": "abs(actual-expected) <= atol + rtol*abs(expected)",
                "justification": Z_TO_S_TOLERANCE_JUSTIFICATION,
                "regeneration": (
                    "canonical UTF-8 JSON serialization; s is checked with the "
                    "recorded numeric tolerance"
                ),
                "rtol": Z_TO_S_RTOL,
            },
            "wave_definition": converted.s_def,
        },
        "data": {
            "frequency_hz": [float(value) for value in frequency_hz],
            "s": _complex_array(converted_s),
            "z0_ohm": _complex_array(source_z0),
            "z_ohm": _complex_array(source_z),
        },
    }


_CASES = (
    _OracleCase(
        "three_port_complex_z0",
        DEFAULT_FIXTURE,
        _network_fixture,
        "exact",
    ),
    _OracleCase(
        "power_wave_s_to_z_three_port_complex_z0",
        S_TO_Z_FIXTURE,
        _s_to_z_fixture,
        "numeric_output",
        "z_ohm",
    ),
    _OracleCase(
        "power_wave_z_to_s_three_port_complex_z0",
        Z_TO_S_FIXTURE,
        _z_to_s_fixture,
        "numeric_output",
        "s",
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


def _reject_json_constant(value: str) -> None:
    """Reject JSON extensions such as NaN and Infinity during parsing."""

    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Reject duplicate object members instead of silently keeping the last."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _assert_finite_json(value: Any, path: str = "$") -> None:
    """Reject non-finite JSON numbers, including overflowed ``1e999``."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number at {path}")
        return

    if isinstance(value, int) and not isinstance(value, bool):
        # Python accepts arbitrarily large JSON integers.  Reject integers that
        # cannot be represented as a finite binary64 value because all oracle
        # numerical data is consumed as binary64 downstream.
        try:
            finite = math.isfinite(float(value))
        except OverflowError as error:
            raise ValueError(f"non-finite JSON number at {path}") from error
        if not finite:
            raise ValueError(f"non-finite JSON number at {path}")
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")
        return

    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_json(item, f"{path}.{key}")


def _parse_strict_json(raw: bytes) -> dict[str, Any]:
    """Parse a fixture with strict JSON and finite-number semantics."""

    try:
        text = raw.decode("utf-8")
        document = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
            strict=True,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON: {error}") from error

    if not isinstance(document, dict):
        raise ValueError("fixture root must be a JSON object")
    _assert_finite_json(document)
    return document


def _read_canonical_json(path: Path) -> dict[str, Any]:
    """Read, strictly parse, and canonical-encoding-check one fixture."""

    raw = path.read_bytes()
    document = _parse_strict_json(raw)
    try:
        canonical = _canonical_bytes(document)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"fixture cannot be canonically encoded: {error}") from error
    if raw != canonical:
        raise ValueError("fixture is not in canonical UTF-8 JSON encoding")
    return document


def _contract_projection(document: dict[str, Any], output_key: str) -> dict[str, Any]:
    """Remove only the operation output before exact contract comparison."""

    data = document.get("data")
    if not isinstance(data, dict):
        raise ValueError("fixture data must be a JSON object")
    if output_key not in data:
        raise ValueError(f"fixture data is missing numeric output {output_key!r}")

    projection = dict(document)
    projected_data = dict(data)
    del projected_data[output_key]
    projection["data"] = projected_data
    return projection


def _numeric_tolerance(document: dict[str, Any]) -> tuple[float, float]:
    """Read and validate the recorded relative/absolute output tolerances."""

    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("fixture metadata must be a JSON object")
    policy = metadata.get("tolerance_policy")
    if not isinstance(policy, dict):
        raise ValueError("fixture tolerance_policy must be a JSON object")

    absolute_keys = [key for key in ("atol", "atol_ohm") if key in policy]
    if len(absolute_keys) != 1:
        raise ValueError(
            "fixture tolerance_policy must contain exactly one of "
            "'atol' or 'atol_ohm'"
        )

    absolute_key = absolute_keys[0]
    values: list[tuple[str, Any]] = [
        ("rtol", policy.get("rtol")),
        (absolute_key, policy.get(absolute_key)),
    ]
    parsed: dict[str, float] = {}
    for name, value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"tolerance_policy.{name} must be a JSON number")
        try:
            converted = float(value)
        except OverflowError as error:
            raise ValueError(
                f"tolerance_policy.{name} must be finite and non-negative"
            ) from error
        if not math.isfinite(converted) or converted < 0.0:
            raise ValueError(f"tolerance_policy.{name} must be finite and non-negative")
        parsed[name] = converted
    return parsed["rtol"], parsed[absolute_key]


def _numeric_value(value: Any, path: str) -> float:
    """Validate one JSON number used as a real or imaginary component."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a JSON number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{path} must be finite") from error
    if not math.isfinite(converted):
        raise ValueError(f"{path} must be finite")
    return converted


def _compare_numeric_output(
    actual: Any,
    expected: Any,
    *,
    path: str,
    rtol: float,
    atol: float,
) -> str | None:
    """Validate shape and compare complex leaves with one absolute bound."""

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return f"{path} must be an array"
        if len(actual) != len(expected):
            return f"{path} has length {len(actual)}; expected {len(expected)}"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            mismatch = _compare_numeric_output(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
            )
            if mismatch is not None:
                return mismatch
        return None

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{path} must be a complex object"
        expected_keys = {"real", "imag"}
        if set(expected) != expected_keys:
            return (
                f"{path} expected value must contain exactly real/imag fields; "
                f"found {sorted(expected)!r}"
            )
        actual_keys = set(actual)
        if actual_keys != expected_keys:
            return (
                f"{path} must contain exactly real/imag fields; "
                f"found {sorted(actual_keys)!r}"
            )
        try:
            actual_real = _numeric_value(actual["real"], f"{path}.real")
            actual_imag = _numeric_value(actual["imag"], f"{path}.imag")
            expected_real = _numeric_value(expected["real"], f"{path}.real (expected)")
            expected_imag = _numeric_value(expected["imag"], f"{path}.imag (expected)")
        except ValueError as error:
            return str(error)

        actual_complex = complex(actual_real, actual_imag)
        expected_complex = complex(expected_real, expected_imag)
        difference = abs(actual_complex - expected_complex)
        bound = atol + rtol * abs(expected_complex)
        if not math.isfinite(difference) or difference > bound:
            return (
                f"{path} differs by {difference:.17g}; "
                f"allowed {bound:.17g}"
            )
        return None

    return f"{path} has an invalid expected numeric-output shape"


def _check_numeric_fixture(
    path: Path,
    expected: dict[str, Any],
    output_key: str,
) -> int:
    """Check a canonical fixture whose selected output is numerically tolerant."""

    try:
        actual = _read_canonical_json(path)
    except FileNotFoundError:
        print(f"fixture missing: {path}; run `generate_oracle.py write`", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"fixture schema/encoding check failed: {path}: {error}", file=sys.stderr)
        return 1

    try:
        expected_contract = _contract_projection(expected, output_key)
        actual_contract = _contract_projection(actual, output_key)
        if _canonical_bytes(actual_contract) != _canonical_bytes(expected_contract):
            print(
                f"fixture contract differs from regenerated canonical output: {path}\n"
                "metadata, inputs, shape, and non-output fields must match exactly",
                file=sys.stderr,
            )
            return 1

        rtol, atol = _numeric_tolerance(actual)
        regenerated_output = expected["data"][output_key]
        checked_in_output = actual["data"][output_key]
        mismatch = _compare_numeric_output(
            regenerated_output,
            checked_in_output,
            path=f"data.{output_key}",
            rtol=rtol,
            atol=atol,
        )
        if mismatch is not None:
            print(f"fixture numeric output check failed: {path}: {mismatch}", file=sys.stderr)
            return 1
    except (KeyError, TypeError, ValueError) as error:
        print(f"fixture schema check failed: {path}: {error}", file=sys.stderr)
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


def _check_registered_case(
    case: _OracleCase,
    expected: dict[str, Any],
    expected_bytes: bytes,
) -> int:
    """Dispatch checking according to the registered case strategy."""

    if case.comparison == "exact":
        return _check_fixture(case.path, expected_bytes)
    if case.comparison == "numeric_output":
        if case.numeric_output_key is None:
            raise ValueError(f"numeric case {case.case_id!r} has no output key")
        return _check_numeric_fixture(case.path, expected, case.numeric_output_key)
    raise ValueError(f"unknown fixture comparison strategy: {case.comparison!r}")


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
            document = case.builder(np, skrf)
            expected = _canonical_bytes(document)
            if args.mode == "write":
                _write_fixture(path, expected)
                print(f"fixture written: {path}")
            else:
                # Keep the generated document independent from the checked
                # path: --fixture is a path override, not a second source of
                # expected values.
                selected_case = case._replace(path=path)
                status = max(
                    status,
                    _check_registered_case(selected_case, document, expected),
                )
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
