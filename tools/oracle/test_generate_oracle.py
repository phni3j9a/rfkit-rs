"""Deterministic tests for the oracle fixture checker semantics."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import generate_oracle as oracle


class NumericFixtureCheckerTests(unittest.TestCase):
    """Exercise the tolerant output path without importing pinned dependencies."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_path = oracle.S_TO_Z_FIXTURE
        cls.fixture = oracle._read_canonical_json(cls.fixture_path)

    def _check_document(self, document: dict[str, object]) -> int:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / self.fixture_path.name
            path.write_bytes(oracle._canonical_bytes(document))
            return oracle._check_numeric_fixture(path, self.fixture, "z_ohm")

    def test_output_within_recorded_complex_tolerance_passes(self) -> None:
        document = copy.deepcopy(self.fixture)
        document["data"]["z_ohm"][0][0][0]["real"] += 1e-13

        self.assertEqual(self._check_document(document), 0)

    def test_output_outside_recorded_complex_tolerance_fails(self) -> None:
        document = copy.deepcopy(self.fixture)
        document["data"]["z_ohm"][0][0][0]["real"] += 1e-3

        self.assertEqual(self._check_document(document), 1)

    def test_contract_input_drift_fails_even_when_output_is_unchanged(self) -> None:
        document = copy.deepcopy(self.fixture)
        document["data"]["frequency_hz"][0] += 1.0

        self.assertEqual(self._check_document(document), 1)

    def test_unknown_complex_field_fails(self) -> None:
        document = copy.deepcopy(self.fixture)
        document["data"]["z_ohm"][0][0][0]["unexpected"] = 0.0

        self.assertEqual(self._check_document(document), 1)

    def test_missing_complex_field_fails(self) -> None:
        document = copy.deepcopy(self.fixture)
        del document["data"]["z_ohm"][0][0][0]["imag"]

        self.assertEqual(self._check_document(document), 1)

    def test_non_finite_number_fails_strict_parse(self) -> None:
        document = copy.deepcopy(self.fixture)
        document["data"]["z_ohm"][0][0][0]["real"] = float("inf")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / self.fixture_path.name
            raw = json.dumps(
                document,
                allow_nan=True,
                ensure_ascii=False,
                indent=2,
                separators=(",", ": "),
                sort_keys=True,
            )
            path.write_bytes((raw + "\n").encode("utf-8"))
            self.assertEqual(
                oracle._check_numeric_fixture(path, self.fixture, "z_ohm"),
                1,
            )

    def test_overflowed_json_number_fails_finite_check(self) -> None:
        target = self.fixture["data"]["z_ohm"][0][0][0]["real"]
        needle = f'"real": {target}'.encode("utf-8")
        raw = oracle._canonical_bytes(self.fixture).replace(needle, b'"real": 1e999', 1)
        self.assertNotEqual(raw, oracle._canonical_bytes(self.fixture))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / self.fixture_path.name
            path.write_bytes(raw)
            self.assertEqual(
                oracle._check_numeric_fixture(path, self.fixture, "z_ohm"),
                1,
            )

    def test_non_canonical_document_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / self.fixture_path.name
            path.write_bytes(oracle._canonical_bytes(self.fixture) + b"\n")
            self.assertEqual(
                oracle._check_numeric_fixture(path, self.fixture, "z_ohm"),
                1,
            )


class ZToSNumericFixtureCheckerTests(unittest.TestCase):
    """Exercise the dimensionless ``atol`` policy used by Z-to-S."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_path = oracle.Z_TO_S_FIXTURE
        cls.fixture = oracle._read_canonical_json(cls.fixture_path)

    def _check_document(self, document: dict[str, object]) -> int:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / self.fixture_path.name
            path.write_bytes(oracle._canonical_bytes(document))
            return oracle._check_numeric_fixture(path, self.fixture, "s")

    def test_output_within_recorded_atol_tolerance_passes(self) -> None:
        document = copy.deepcopy(self.fixture)
        document["data"]["s"][0][0][0]["real"] += 1e-13

        self.assertEqual(self._check_document(document), 0)

    def test_output_outside_recorded_atol_tolerance_fails(self) -> None:
        document = copy.deepcopy(self.fixture)
        document["data"]["s"][0][0][0]["real"] += 1e-3

        self.assertEqual(self._check_document(document), 1)


if __name__ == "__main__":
    unittest.main()
