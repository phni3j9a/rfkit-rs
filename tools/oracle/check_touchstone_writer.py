"""Feed output from the public Rust Touchstone writers into pinned scikit-rf.

This is an interoperability check, not a same-parser round trip.  The Rust
example constructs deterministic common-reference v1 and unequal-reference
v2 two-/five-port Networks, invokes the public writers, and writes only the
resulting text to stdout.  This checker
then asks scikit-rf 2.0.1 to parse that text and compares frequency, S, and z0
against the independently reproduced construction recipe.
"""

from __future__ import annotations

import math
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


def _expected(np, nfreq: int, nports: int, references):
    frequency = np.asarray([1.0e6 * (index + 1) for index in range(nfreq)])
    s = np.empty((nfreq, nports, nports), dtype=complex)
    for f in range(nfreq):
        for row in range(nports):
            for column in range(nports):
                value = float(f * 100 + row * 10 + column + 1)
                s[f, row, column] = complex(0.005 * value, -0.003 * value)
    z0 = np.empty((nfreq, nports), dtype=complex)
    for port, reference in enumerate(references):
        z0[:, port] = complex(reference, 0.0)
    return frequency, s, z0


def _check_layout_contract(text: str, case: str, nports: int, references) -> None:
    """Check deterministic directives and physical line boundaries.

    The asymmetric record values are checked independently by scikit-rf
    below. Keeping this small textual check separate means a future parser
    cannot hide a writer ordering/reference/header drift by accepting both
    layouts.
    """
    lines = text.splitlines()
    is_v2 = case.startswith("v2-")
    if is_v2:
        expected_prefix = [
            "[Version] 2.0",
            f"# Hz S RI R {references[0]:g}",
            f"[Number of Ports] {nports}",
        ]
        if lines[:3] != expected_prefix:
            raise AssertionError(f"unexpected v2 header for {case!r}: {lines[:3]!r}")
        cursor = 3
        if nports == 2:
            if lines[cursor] != "[Two-Port Data Order] 12_21":
                raise AssertionError("v2 writer did not emit natural two-port order")
            cursor += 1
        expected_header = [
            "[Number of Frequencies] 2",
            "[Reference] " + " ".join(f"{reference:g}" for reference in references),
            "[Matrix Format] Full",
            "[Network Data]",
        ]
        if lines[cursor : cursor + len(expected_header)] != expected_header:
            raise AssertionError(f"unexpected v2 directives for {case!r}")
        data_start = cursor + len(expected_header)
        if len(lines) != data_start + 3:
            raise AssertionError(f"unexpected v2 physical line count for {case!r}")
        if lines[data_start + 2] != "[End]":
            raise AssertionError("v2 writer did not terminate with [End]")
        expected_values = 1 + 2 * nports * nports
        if any(len(line.split()) != expected_values for line in lines[data_start : data_start + 2]):
            raise AssertionError("v2 writer did not keep one complete record per line")
        for frequency_index, line in enumerate(lines[data_start : data_start + 2]):
            tokens = line.split()
            expected_frequency = 1.0e6 * (frequency_index + 1)
            if float(tokens[0]) != expected_frequency:
                raise AssertionError("v2 writer frequency ordering drifted")
            for matrix_index in range(nports * nports):
                row, column = divmod(matrix_index, nports)
                value = float(frequency_index * 100 + row * 10 + column + 1)
                actual_real = float(tokens[1 + 2 * matrix_index])
                actual_imag = float(tokens[2 + 2 * matrix_index])
                if not math.isclose(actual_real, 0.005 * value, rel_tol=1e-12, abs_tol=1e-12):
                    raise AssertionError("v2 writer S row/column ordering drifted")
                if not math.isclose(actual_imag, -0.003 * value, rel_tol=1e-12, abs_tol=1e-12):
                    raise AssertionError("v2 writer S row/column ordering drifted")
    else:
        if lines[0] != f"# Hz S RI R {references[0]:g}":
            raise AssertionError(f"unexpected v1 option line for {case!r}")


def _check_case(np, skrf, case: str, extension: str, nports: int, references) -> None:
    text = _rust_output(case)
    _check_layout_contract(text, case, nports, references)
    with tempfile.TemporaryDirectory(prefix="rfkit-touchstone-oracle-") as directory:
        path = Path(directory) / f"writer{extension}"
        path.write_text(text, encoding="ascii", newline="")
        parsed = skrf.Network(str(path))

    expected_f, expected_s, expected_z0 = _expected(np, 2, nports, references)
    np.testing.assert_array_equal(parsed.f, expected_f)
    np.testing.assert_allclose(parsed.s, expected_s, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(parsed.z0, expected_z0)

    if not case.startswith("v2-") and nports > 4:
        # The second physical line completes the first row and contains no
        # frequency token; this independently checks the continuation case.
        lines = text.splitlines()
        if len(lines) != 1 + 2 * nports * 2:
            raise AssertionError(f"unexpected five-port physical line count: {len(lines)}")
        if len(lines[1].split()) != 1 + 8 or len(lines[2].split()) != 2:
            raise AssertionError("writer did not use four-pair continuation layout")


def main() -> int:
    np, skrf = _load_dependencies()
    _check_case(np, skrf, "two-port", ".s2p", 2, [73.5, 73.5])
    _check_case(np, skrf, "five-port", ".s5p", 5, [88.25] * 5)
    _check_case(np, skrf, "v2-two-port", ".s2p", 2, [37.0, 48.0])
    _check_case(np, skrf, "v2-five-port", ".s5p", 5, [37.0, 48.0, 59.0, 70.0, 81.0])
    print("Touchstone writer interoperability check passed (Rust → scikit-rf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
