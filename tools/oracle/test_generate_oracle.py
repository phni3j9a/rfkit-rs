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


MATRIX_CASE_SPECS = {
    "power_wave_s_to_z_one_port_real_scalar_z0": {
        "operation": "s_to_z",
        "ports": 1,
        "frequencies": 3,
        "output": "z_ohm",
        "complex": False,
        "frequency_dependent": False,
        "per_port": False,
    },
    "power_wave_s_to_z_two_port_complex_per_port_constant_z0": {
        "operation": "s_to_z",
        "ports": 2,
        "frequencies": 4,
        "output": "z_ohm",
        "complex": True,
        "frequency_dependent": False,
        "per_port": True,
    },
    "power_wave_s_to_z_four_port_real_frequency_dependent_z0": {
        "operation": "s_to_z",
        "ports": 4,
        "frequencies": 3,
        "output": "z_ohm",
        "complex": False,
        "frequency_dependent": True,
        "per_port": False,
    },
    "power_wave_s_to_z_eight_port_complex_per_port_frequency_dependent_z0": {
        "operation": "s_to_z",
        "ports": 8,
        "frequencies": 3,
        "output": "z_ohm",
        "complex": True,
        "frequency_dependent": True,
        "per_port": True,
    },
    "power_wave_z_to_s_one_port_real_scalar_z0": {
        "operation": "z_to_s",
        "ports": 1,
        "frequencies": 3,
        "output": "s",
        "complex": False,
        "frequency_dependent": False,
        "per_port": False,
    },
    "power_wave_z_to_s_two_port_complex_per_port_constant_z0": {
        "operation": "z_to_s",
        "ports": 2,
        "frequencies": 4,
        "output": "s",
        "complex": True,
        "frequency_dependent": False,
        "per_port": True,
    },
    "power_wave_z_to_s_four_port_real_frequency_dependent_z0": {
        "operation": "z_to_s",
        "ports": 4,
        "frequencies": 3,
        "output": "s",
        "complex": False,
        "frequency_dependent": True,
        "per_port": False,
    },
    "power_wave_z_to_s_eight_port_complex_per_port_frequency_dependent_z0": {
        "operation": "z_to_s",
        "ports": 8,
        "frequencies": 3,
        "output": "s",
        "complex": True,
        "frequency_dependent": True,
        "per_port": True,
    },
}


class MatrixRegistrationAndCheckerTests(unittest.TestCase):
    """Protect registration, z0-pattern, and output-only checker semantics."""

    @staticmethod
    def _complex(value: dict[str, float]) -> complex:
        return complex(value["real"], value["imag"])

    def test_all_matrix_cases_are_registered_individually(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        self.assertEqual(len(registered), 11)
        self.assertTrue(set(MATRIX_CASE_SPECS).issubset(registered))
        for case_id, spec in MATRIX_CASE_SPECS.items():
            case = registered[case_id]
            self.assertEqual(case.comparison, "numeric_output")
            self.assertEqual(case.numeric_output_key, spec["output"])
            self.assertEqual(case.path.stem, case_id)

    def test_matrix_fixtures_match_declared_dimensions_and_z0_profiles(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        for case_id, spec in MATRIX_CASE_SPECS.items():
            with self.subTest(case_id=case_id):
                document = oracle._read_canonical_json(registered[case_id].path)
                metadata = document["metadata"]
                data = document["data"]
                ports = spec["ports"]
                frequencies = spec["frequencies"]
                self.assertEqual(metadata["case_id"], case_id)
                self.assertEqual(metadata["operation"], spec["operation"])
                self.assertEqual(metadata["numpy_version"], "2.5.1")
                self.assertEqual(metadata["scikit_rf_version"], "2.0.1")
                self.assertEqual(metadata["wave_definition"], "power")
                self.assertEqual(metadata["shape"]["frequency"], [frequencies])
                z0 = data["z0_ohm"]
                self.assertEqual(len(z0), frequencies)
                self.assertTrue(all(len(row) == ports for row in z0))
                self.assertEqual(
                    metadata["shape"]["input_z0"], [frequencies, ports]
                )

                z0_values = [[self._complex(value) for value in row] for row in z0]
                has_imaginary = any(value.imag != 0.0 for row in z0_values for value in row)
                self.assertEqual(has_imaginary, spec["complex"])
                rows_differ = any(row != z0_values[0] for row in z0_values[1:])
                self.assertEqual(rows_differ, spec["frequency_dependent"])
                ports_differ = any(
                    any(value != row[0] for value in row[1:])
                    for row in z0_values
                )
                self.assertEqual(ports_differ, spec["per_port"])

                input_key = "s" if spec["operation"] == "s_to_z" else "z_ohm"
                input_matrices = data[input_key]
                self.assertEqual(len(input_matrices), frequencies)
                self.assertTrue(all(len(matrix) == ports for matrix in input_matrices))
                self.assertTrue(
                    all(
                        all(len(row) == ports for row in matrix)
                        for matrix in input_matrices
                    )
                )
                if ports > 1:
                    for matrix in input_matrices:
                        values = [
                            [self._complex(value) for value in row] for row in matrix
                        ]
                        self.assertTrue(
                            any(
                                values[row][column] != values[column][row]
                                for row in range(ports)
                                for column in range(row + 1, ports)
                            )
                        )

    def test_matrix_checker_tolerates_only_computed_output(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        for case_id, spec in MATRIX_CASE_SPECS.items():
            with self.subTest(case_id=case_id):
                case = registered[case_id]
                fixture = oracle._read_canonical_json(case.path)
                output_key = spec["output"]
                checked_in = copy.deepcopy(fixture)
                output = checked_in["data"][output_key]
                output[0][0][0]["real"] += 1e-13
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / case.path.name
                    path.write_bytes(oracle._canonical_bytes(checked_in))
                    self.assertEqual(
                        oracle._check_numeric_fixture(path, fixture, output_key), 0
                    )

                drifted = copy.deepcopy(fixture)
                drifted["data"]["frequency_hz"][0] += 1.0
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / case.path.name
                    path.write_bytes(oracle._canonical_bytes(drifted))
                    self.assertEqual(
                        oracle._check_numeric_fixture(path, fixture, output_key), 1
                    )


if __name__ == "__main__":
    unittest.main()
