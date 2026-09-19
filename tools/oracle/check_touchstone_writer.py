"""Feed output from the public Rust Touchstone writer into pinned scikit-rf.

This is an interoperability check, not a same-parser round trip.  The Rust
example constructs deterministic two-port and five-port Networks, invokes the
public writer, and writes only the resulting text to stdout.  This checker
then asks scikit-rf 2.0.1 to parse that text and compares frequency, S, and z0
against the independently reproduced construction recipe.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


EXPECTED_NUMPY_VERSION = "2.5.1"
EXPECTED_SCIKIT_RF_VERSION = "2.0.1"
EXPECTED_SCIPY_VERSION = "1.18.1"
ROOT = Path(__file__).resolve().parents[2]


def _load_dependencies():
    try:
        import numpy as np
        import scipy
        import skrf
    except ImportError as error:
        raise RuntimeError(
            "pinned oracle dependencies are missing; install tools/oracle/requirements.txt"
        ) from error
    versions = (np.__version__, skrf.__version__, scipy.__version__)
    expected = (
        EXPECTED_NUMPY_VERSION,
        EXPECTED_SCIKIT_RF_VERSION,
        EXPECTED_SCIPY_VERSION,
    )
    if versions != expected:
        raise RuntimeError(
            "wrong pinned oracle dependency versions: "
            f"numpy={versions[0]}, scikit-rf={versions[1]}, scipy={versions[2]}; "
            f"expected numpy={expected[0]}, scikit-rf={expected[1]}, scipy={expected[2]}"
        )
    return np, skrf


def _rust_output(case: str) -> str:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--package",
            "rfkit-touchstone",
            "--example",
            "touchstone_writer_oracle",
            "--",
            case,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Rust writer oracle failed for {case!r} with exit code "
            f"{completed.returncode}:\n{completed.stderr}"
        )
    text = completed.stdout
    if not text.isascii() or not text.endswith("\n"):
        raise RuntimeError(f"Rust writer emitted invalid ASCII/final-newline text for {case!r}")
    return text


def _expected(np, nfreq: int, nports: int, reference: float):
    frequency = np.asarray([1.0e6 * (index + 1) for index in range(nfreq)])
    s = np.empty((nfreq, nports, nports), dtype=complex)
    for f in range(nfreq):
        for row in range(nports):
            for column in range(nports):
                value = float(f * 100 + row * 10 + column + 1)
                s[f, row, column] = complex(0.005 * value, -0.003 * value)
    z0 = np.full((nfreq, nports), complex(reference, 0.0))
    return frequency, s, z0


def _check_case(np, skrf, case: str, extension: str, nports: int, reference: float) -> None:
    text = _rust_output(case)
    with tempfile.TemporaryDirectory(prefix="rfkit-touchstone-oracle-") as directory:
        path = Path(directory) / f"writer{extension}"
        path.write_text(text, encoding="ascii", newline="")
        parsed = skrf.Network(str(path))

    expected_f, expected_s, expected_z0 = _expected(np, 2, nports, reference)
    np.testing.assert_allclose(parsed.f, expected_f, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(parsed.s, expected_s, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(parsed.z0, expected_z0, rtol=1e-12, atol=1e-12)

    if nports > 4:
        # The second physical line completes the first row and contains no
        # frequency token; this independently checks the continuation case.
        lines = text.splitlines()
        if len(lines) != 1 + 2 * nports * 2:
            raise AssertionError(f"unexpected five-port physical line count: {len(lines)}")
        if len(lines[1].split()) != 1 + 8 or len(lines[2].split()) != 2:
            raise AssertionError("writer did not use four-pair continuation layout")


def main() -> int:
    np, skrf = _load_dependencies()
    _check_case(np, skrf, "two-port", ".s2p", 2, 73.5)
    _check_case(np, skrf, "five-port", ".s5p", 5, 88.25)
    print("Touchstone writer interoperability check passed (Rust → scikit-rf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
