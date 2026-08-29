"""Deterministic tests for the oracle fixture checker semantics."""

from __future__ import annotations

import copy
import json
import math
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

RECIPROCAL_CASE_SPECS = {
    "power_wave_s_to_z_three_port_reciprocal_real_equal_z0": {
        "operation": "s_to_z",
        "ports": 3,
        "frequencies": 3,
        "input": "s",
        "output": "z_ohm",
        "seed": 20_260_921,
        "z0": 61.25,
        "absolute_tolerance": "atol_ohm",
    },
    "power_wave_z_to_s_three_port_reciprocal_real_equal_z0": {
        "operation": "z_to_s",
        "ports": 3,
        "frequencies": 3,
        "input": "z_ohm",
        "output": "s",
        "seed": 20_260_922,
        "z0": 61.25,
        "absolute_tolerance": "atol",
    },
    "power_wave_renormalize_three_port_reciprocal_real_equal_z0": {
        "operation": "renormalize_s",
        "ports": 3,
        "frequencies": 3,
        "input": "s_input",
        "output": "s_renormalized",
        "seed": 20_260_923,
        "z0_source": 42.75,
        "z0_target": 86.5,
        "absolute_tolerance": "atol",
    },
}

PASSIVE_CASE_SPECS = {
    "power_wave_s_to_z_three_port_reciprocal_real_equal_z0": {
        "operation": "s_to_z",
        "matrix_fields": ["s"],
        "observed_maxima": [0.151907019275],
        "z0": [61.25],
        "input": "s",
        "output": "z_ohm",
    },
    "power_wave_z_to_s_three_port_reciprocal_real_equal_z0": {
        "operation": "z_to_s",
        "matrix_fields": ["s"],
        "observed_maxima": [0.166759615367],
        "z0": [61.25],
        "input": "z_ohm",
        "output": "s",
    },
    "power_wave_renormalize_three_port_reciprocal_real_equal_z0": {
        "operation": "renormalize_s",
        "matrix_fields": ["s_input", "s_renormalized"],
        "observed_maxima": [0.155165225095, 0.456137460254],
        "z0": [42.75, 86.5],
        "input": "s_input",
        "output": "s_renormalized",
    },
}

ACTIVE_CASE_SPECS = {
    "power_wave_s_to_z_three_port_active_real_equal_z0": {
        "operation": "s_to_z",
        "ports": 3,
        "frequencies": 3,
        "matrix_field": "s",
        "output": "z_ohm",
        "seed": 20_260_924,
        "z0": 57.25,
        "observed_minimum": 1.799980263141,
    },
    "power_wave_z_to_s_three_port_active_real_equal_z0": {
        "operation": "z_to_s",
        "ports": 3,
        "frequencies": 3,
        "matrix_field": "s",
        "output": "s",
        "seed": 20_260_925,
        "z0": 57.25,
        "observed_minimum": 2.275294506203,
    },
    "power_wave_renormalize_three_port_active_real_equal_z0": {
        "operation": "renormalize_s",
        "ports": 3,
        "frequencies": 3,
        "matrix_field": "s_input",
        "output": "s_renormalized",
        "seed": 20_260_926,
        "z0_source": 57.25,
        "z0_target": 91.75,
        "observed_minimum": 1.815482853976,
    },
}

RENORMALIZATION_CASE_SPECS = {
    "power_wave_renormalize_one_port_real_scalar_z0": {
        "ports": 1,
        "frequencies": 3,
        "complex": False,
        "frequency_dependent": False,
        "per_port": False,
    },
    "power_wave_renormalize_two_port_complex_per_port_constant_z0": {
        "ports": 2,
        "frequencies": 4,
        "complex": True,
        "frequency_dependent": False,
        "per_port": True,
    },
    "power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0": {
        "ports": 4,
        "frequencies": 3,
        "complex": True,
        "frequency_dependent": True,
        "per_port": True,
    },
    "power_wave_renormalize_eight_port_real_frequency_dependent_z0": {
        "ports": 8,
        "frequencies": 3,
        "complex": False,
        "frequency_dependent": True,
        "per_port": False,
    },
}
RENORMALIZATION_CASE_IDS = tuple(RENORMALIZATION_CASE_SPECS)
# Retain the original single-case name for the focused legacy assertions
# below; the table-driven assertions exercise every renormalization case.
RENORMALIZATION_CASE_ID = (
    "power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0"
)
RENORMALIZATION_REFERENCE_FLAGS = {
    "complex": True,
    "frequency_dependent": True,
    "per_port": True,
    "unit": "ohm",
}


class MatrixRegistrationAndCheckerTests(unittest.TestCase):
    """Protect registration, z0-pattern, and output-only checker semantics."""

    @staticmethod
    def _complex(value: dict[str, float]) -> complex:
        return complex(value["real"], value["imag"])

    def test_all_matrix_cases_are_registered_individually(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        self.assertEqual(
            len(registered),
            3
            + len(MATRIX_CASE_SPECS)
            + len(RENORMALIZATION_CASE_SPECS)
            + len(RECIPROCAL_CASE_SPECS)
            + len(ACTIVE_CASE_SPECS),
        )
        self.assertTrue(set(MATRIX_CASE_SPECS).issubset(registered))
        for case_id, spec in MATRIX_CASE_SPECS.items():
            case = registered[case_id]
            self.assertEqual(case.comparison, "numeric_output")
            self.assertEqual(case.numeric_output_key, spec["output"])
            self.assertEqual(case.path.stem, case_id)


class ReciprocalRegistrationAndCheckerTests(unittest.TestCase):
    """Protect the three reciprocal fixture contracts and symmetry checks."""

    @staticmethod
    def _complex(value: dict[str, float]) -> complex:
        return complex(value["real"], value["imag"])

    @staticmethod
    def _assert_symmetric(
        testcase: unittest.TestCase,
        matrices: list[list[list[dict[str, float]]]],
        *,
        exact: bool,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> None:
        for frequency, matrix in enumerate(matrices):
            ports = len(matrix)
            for row in range(ports):
                for column in range(row + 1, ports):
                    lhs = ReciprocalRegistrationAndCheckerTests._complex(
                        matrix[row][column]
                    )
                    rhs = ReciprocalRegistrationAndCheckerTests._complex(
                        matrix[column][row]
                    )
                    if exact:
                        testcase.assertEqual(
                            lhs,
                            rhs,
                            f"input is not exactly symmetric at frequency {frequency}, "
                            f"ports ({row}, {column})",
                        )
                    else:
                        difference = abs(lhs - rhs)
                        bound = atol + rtol * max(abs(lhs), abs(rhs))
                        testcase.assertLessEqual(
                            difference,
                            bound,
                            f"output is not symmetric at frequency {frequency}, "
                            f"ports ({row}, {column}): difference={difference:e}, "
                            f"bound={bound:e}",
                        )

    def test_all_reciprocal_cases_are_registered_with_contracts(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        expected_paths = {
            "power_wave_s_to_z_three_port_reciprocal_real_equal_z0": oracle.RECIPROCAL_S_TO_Z_FIXTURE,
            "power_wave_z_to_s_three_port_reciprocal_real_equal_z0": oracle.RECIPROCAL_Z_TO_S_FIXTURE,
            "power_wave_renormalize_three_port_reciprocal_real_equal_z0": oracle.RECIPROCAL_RENORMALIZE_FIXTURE,
        }
        self.assertEqual(set(RECIPROCAL_CASE_SPECS), set(expected_paths))
        self.assertEqual(
            len({RECIPROCAL_CASE_SPECS[case_id]["seed"] for case_id in RECIPROCAL_CASE_SPECS}),
            3,
        )
        for case_id, spec in RECIPROCAL_CASE_SPECS.items():
            with self.subTest(case_id=case_id):
                case = registered[case_id]
                self.assertEqual(case.path, expected_paths[case_id])
                self.assertEqual(case.path.stem, case_id)
                self.assertEqual(case.comparison, "numeric_output")
                self.assertEqual(case.numeric_output_key, spec["output"])

    def test_reciprocal_fixtures_are_symmetric_real_equal_and_deterministic(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        for case_id, spec in RECIPROCAL_CASE_SPECS.items():
            with self.subTest(case_id=case_id):
                fixture = oracle._read_canonical_json(registered[case_id].path)
                metadata = fixture["metadata"]
                data = fixture["data"]
                self.assertEqual(metadata["case_id"], case_id)
                self.assertEqual(metadata["operation"], spec["operation"])
                self.assertEqual(metadata["numpy_version"], "2.5.1")
                self.assertEqual(metadata["scikit_rf_version"], "2.0.1")
                self.assertEqual(metadata["wave_definition"], "power")
                self.assertEqual(metadata["random_seed"], spec["seed"])
                self.assertEqual(metadata["shape"]["frequency"], [3])

                if spec["operation"] == "renormalize_s":
                    flags = {
                        "complex": False,
                        "frequency_dependent": False,
                        "per_port": False,
                        "unit": "ohm",
                    }
                    self.assertEqual(metadata["reference_impedance"]["source"], flags)
                    self.assertEqual(metadata["reference_impedance"]["target"], flags)
                    self.assertEqual(
                        metadata["shape"],
                        {
                            "frequency": [3],
                            "s_input": [3, 3, 3],
                            "s_renormalized": [3, 3, 3],
                            "z0_source": [3, 3],
                            "z0_target": [3, 3],
                        },
                    )
                    input_matrices = data[spec["input"]]
                    output_matrices = data[spec["output"]]
                    source_z0 = data["z0_source_ohm"]
                    target_z0 = data["z0_target_ohm"]
                    for z0_values, expected in (
                        (source_z0, spec["z0_source"]),
                        (target_z0, spec["z0_target"]),
                    ):
                        values = [
                            [self._complex(value) for value in row]
                            for row in z0_values
                        ]
                        self.assertTrue(
                            all(
                                value == complex(expected, 0.0)
                                for row in values
                                for value in row
                            )
                        )
                    self.assertNotEqual(spec["z0_source"], 50.0)
                    self.assertNotEqual(spec["z0_target"], 50.0)
                    self.assertGreater(
                        abs(spec["z0_source"] - spec["z0_target"]), 1.0
                    )
                    tolerance = metadata["tolerance_policy"]["atol"]
                    rtol = metadata["tolerance_policy"]["rtol"]
                else:
                    flags = {
                        "complex": False,
                        "frequency_dependent": False,
                        "per_port": False,
                        "unit": "ohm",
                    }
                    self.assertEqual(metadata["reference_impedance"], flags)
                    shape_key = "input_s" if spec["operation"] == "s_to_z" else "input_z"
                    output_shape_key = "output_z" if spec["operation"] == "s_to_z" else "output_s"
                    self.assertEqual(metadata["shape"][shape_key], [3, 3, 3])
                    self.assertEqual(metadata["shape"]["input_z0"], [3, 3])
                    self.assertEqual(metadata["shape"][output_shape_key], [3, 3, 3])
                    input_matrices = data[spec["input"]]
                    output_matrices = data[spec["output"]]
                    z0_values = [
                        [self._complex(value) for value in row]
                        for row in data["z0_ohm"]
                    ]
                    self.assertTrue(
                        all(
                            value == complex(spec["z0"], 0.0)
                            for row in z0_values
                            for value in row
                        )
                    )
                    self.assertNotEqual(spec["z0"], 50.0)
                    tolerance = metadata["tolerance_policy"][spec["absolute_tolerance"]]
                    rtol = metadata["tolerance_policy"]["rtol"]
                self.assertEqual(metadata["tolerance_policy"]["rtol"], 1e-12)
                self.assertEqual(
                    metadata["tolerance_policy"][spec["absolute_tolerance"]],
                    1e-12,
                )
                self.assertIn("binary64", metadata["tolerance_policy"]["justification"])
                self.assertEqual(len(input_matrices), 3)
                self.assertEqual(len(output_matrices), 3)
                for matrix in input_matrices:
                    self.assertEqual(len(matrix), 3)
                    self.assertTrue(all(len(row) == 3 for row in matrix))
                for matrix in output_matrices:
                    self.assertEqual(len(matrix), 3)
                    self.assertTrue(all(len(row) == 3 for row in matrix))
                self._assert_symmetric(self, input_matrices, exact=True)
                self.assertTrue(
                    any(
                        self._complex(matrix[row][column]).imag != 0.0
                        for matrix in input_matrices
                        for row in range(3)
                        for column in range(row + 1, 3)
                    ),
                    "reciprocal inputs must exercise complex transpose symmetry, "
                    "not only real symmetry",
                )
                self._assert_symmetric(
                    self,
                    output_matrices,
                    exact=False,
                    rtol=rtol,
                    atol=tolerance,
                )

    def test_reciprocal_checker_tolerates_only_computed_output(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        for case_id, spec in RECIPROCAL_CASE_SPECS.items():
            with self.subTest(case_id=case_id):
                case = registered[case_id]
                fixture = oracle._read_canonical_json(case.path)
                adjusted = copy.deepcopy(fixture)
                adjusted["data"][spec["output"]][0][0][0]["real"] += 1e-13
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / case.path.name
                    path.write_bytes(oracle._canonical_bytes(adjusted))
                    self.assertEqual(
                        oracle._check_numeric_fixture(path, fixture, spec["output"]),
                        0,
                    )

                drifted = copy.deepcopy(fixture)
                input_key = spec["input"]
                drifted["data"][input_key][0][0][0]["real"] += 1e-3
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / case.path.name
                    path.write_bytes(oracle._canonical_bytes(drifted))
                    self.assertEqual(
                        oracle._check_numeric_fixture(path, fixture, spec["output"]),
                        1,
                    )

class PassiveRegistrationAndEvidenceTests(unittest.TestCase):
    """Protect passive metadata and its independent Frobenius certificate."""

    @staticmethod
    def _complex(value: dict[str, float]) -> complex:
        return complex(value["real"], value["imag"])

    @classmethod
    def _matrix_frobenius_norm(
        cls, matrix: list[list[dict[str, float]]]
    ) -> float:
        return math.sqrt(
            sum(
                abs(cls._complex(value)) ** 2
                for row in matrix
                for value in row
            )
        )

    @classmethod
    def _matrix_numpy(cls, np: object, matrix: list[list[dict[str, float]]]) -> object:
        return np.asarray(
            [[cls._complex(value) for value in row] for row in matrix],
            dtype=np.complex128,
        )

    def test_passive_cases_are_registered_without_adding_fixture_cases(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        self.assertEqual(set(PASSIVE_CASE_SPECS), set(RECIPROCAL_CASE_SPECS))
        for case_id, spec in PASSIVE_CASE_SPECS.items():
            with self.subTest(case_id=case_id):
                case = registered[case_id]
                self.assertEqual(case.path.stem, case_id)
                self.assertEqual(case.comparison, "numeric_output")
                self.assertEqual(case.numeric_output_key, spec["output"])
                self.assertEqual(len(spec["matrix_fields"]), len(spec["observed_maxima"]))

    def test_passive_metadata_is_pinned_and_nonpassive_cases_omit_it(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        for case_id, spec in PASSIVE_CASE_SPECS.items():
            with self.subTest(case_id=case_id):
                fixture = oracle._read_canonical_json(registered[case_id].path)
                metadata = fixture["metadata"]
                passive = metadata.get("passive_network")
                self.assertIsInstance(passive, dict)
                self.assertEqual(passive["criterion"], oracle.PASSIVE_NETWORK_CRITERION)
                self.assertEqual(
                    passive["required_maximum"], oracle.PASSIVE_REQUIRED_SIGMA_MAX
                )
                evidence = passive["matrices"]
                self.assertEqual(
                    [item["matrix_field"] for item in evidence], spec["matrix_fields"]
                )
                self.assertEqual(
                    [item["observed_maximum"] for item in evidence],
                    spec["observed_maxima"],
                )
                for item in evidence:
                    observed = item["observed_maximum"]
                    self.assertTrue(math.isfinite(observed))
                    self.assertGreaterEqual(observed, 0.0)
                    self.assertLess(observed, oracle.PASSIVE_REQUIRED_SIGMA_MAX)
                    self.assertEqual(
                        observed,
                        round(
                            observed,
                            oracle.PASSIVE_OBSERVED_MAXIMUM_DECIMAL_PLACES,
                        ),
                    )

        passive_case_ids = set(PASSIVE_CASE_SPECS)
        for case in oracle._CASES:
            if case.case_id in passive_case_ids:
                continue
            with self.subTest(nonpassive_case=case.case_id):
                fixture = oracle._read_canonical_json(case.path)
                self.assertNotIn("passive_network", fixture["metadata"])

    def test_pinned_numpy_svd_records_the_true_maximum_for_every_field(self) -> None:
        np, _skrf = oracle._load_dependencies()
        registered = {case.case_id: case for case in oracle._CASES}
        for case_id, spec in PASSIVE_CASE_SPECS.items():
            with self.subTest(case_id=case_id):
                fixture = oracle._read_canonical_json(registered[case_id].path)
                evidence = fixture["metadata"]["passive_network"]["matrices"]
                data = fixture["data"]
                for item in evidence:
                    field = item["matrix_field"]
                    raw_maximum = max(
                        float(
                            np.linalg.svd(
                                self._matrix_numpy(np, matrix), compute_uv=False
                            )[0]
                        )
                        for matrix in data[field]
                    )
                    self.assertTrue(math.isfinite(raw_maximum))
                    self.assertGreaterEqual(raw_maximum, 0.0)
                    self.assertLess(raw_maximum, oracle.PASSIVE_REQUIRED_SIGMA_MAX)
                    self.assertEqual(
                        item["observed_maximum"],
                        round(
                            raw_maximum,
                            oracle.PASSIVE_OBSERVED_MAXIMUM_DECIMAL_PLACES,
                        ),
                    )

    def test_frobenius_upper_bound_certifies_every_fixture_s_field(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        for case_id in PASSIVE_CASE_SPECS:
            with self.subTest(case_id=case_id):
                fixture = oracle._read_canonical_json(registered[case_id].path)
                passive = fixture["metadata"]["passive_network"]
                bounds = []
                for item in passive["matrices"]:
                    bounds_for_field = [
                        self._matrix_frobenius_norm(matrix)
                        for matrix in fixture["data"][item["matrix_field"]]
                    ]
                    self.assertTrue(all(math.isfinite(bound) for bound in bounds_for_field))
                    self.assertTrue(
                        all(
                            bound < oracle.PASSIVE_REQUIRED_SIGMA_MAX
                            for bound in bounds_for_field
                        )
                    )
                    maximum_bound = max(bounds_for_field)
                    bounds.append(maximum_bound)
                    self.assertLessEqual(
                        item["observed_maximum"], maximum_bound + 1e-12
                    )
                self.assertEqual(len(bounds), len(passive["matrices"]))

    def test_passive_z0_is_real_positive_and_equal_per_port(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        for case_id, spec in PASSIVE_CASE_SPECS.items():
            with self.subTest(case_id=case_id):
                fixture = oracle._read_canonical_json(registered[case_id].path)
                data = fixture["data"]
                if spec["operation"] == "renormalize_s":
                    z0_fields = ("z0_source_ohm", "z0_target_ohm")
                    expected_z0 = spec["z0"]
                else:
                    z0_fields = ("z0_ohm",)
                    expected_z0 = [spec["z0"][0]]
                for field, expected in zip(z0_fields, expected_z0):
                    values = [
                        self._complex(value)
                        for row in data[field]
                        for value in row
                    ]
                    self.assertTrue(values)
                    self.assertTrue(
                        all(
                            value.real > 0.0
                            and math.isfinite(value.real)
                            and value.imag == 0.0
                            for value in values
                        )
                    )
                    self.assertTrue(all(value == values[0] for value in values))
                    self.assertEqual(values[0], complex(expected, 0.0))

    def test_z_to_s_reciprocal_fixture_keeps_independent_direct_z_input(self) -> None:
        np, _skrf = oracle._load_dependencies()
        case_id = "power_wave_z_to_s_three_port_reciprocal_real_equal_z0"
        fixture = oracle._read_canonical_json(oracle.RECIPROCAL_Z_TO_S_FIXTURE)
        _frequency, source_z, _constructor_z0, _expanded_z0 = oracle._reciprocal_z_inputs(
            np,
            nfreq=3,
            nports=3,
            seed=RECIPROCAL_CASE_SPECS[case_id]["seed"],
            z0_ohm=RECIPROCAL_CASE_SPECS[case_id]["z0"],
        )
        for frequency in range(3):
            for row in range(3):
                for column in range(3):
                    self.assertEqual(
                        self._complex(fixture["data"]["z_ohm"][frequency][row][column]),
                        source_z[frequency, row, column],
                    )
        self.assertTrue(
            all(
                self._complex(fixture["data"]["z_ohm"][frequency][port][port]).real
                > 60.0
                for frequency in range(3)
                for port in range(3)
            )
        )

    def test_renormalization_evidence_keeps_source_then_target_order(self) -> None:
        case_id = "power_wave_renormalize_three_port_reciprocal_real_equal_z0"
        fixture = oracle._read_canonical_json(oracle.RECIPROCAL_RENORMALIZE_FIXTURE)
        passive = fixture["metadata"]["passive_network"]
        self.assertEqual(
            [item["matrix_field"] for item in passive["matrices"]],
            ["s_input", "s_renormalized"],
        )
        self.assertEqual(
            fixture["metadata"]["shape"]["s_input"], [3, 3, 3]
        )
        self.assertEqual(
            fixture["metadata"]["shape"]["s_renormalized"], [3, 3, 3]
        )
        self.assertEqual(
            [
                self._complex(value)
                for row in fixture["data"]["z0_source_ohm"]
                for value in row
            ],
            [complex(42.75, 0.0)] * 9,
        )
        self.assertEqual(
            [
                self._complex(value)
                for row in fixture["data"]["z0_target_ohm"]
                for value in row
            ],
            [complex(86.5, 0.0)] * 9,
        )
        self.assertEqual(case_id, fixture["metadata"]["case_id"])


class ActiveRegistrationAndEvidenceTests(unittest.TestCase):
    """Protect the active-network metadata and independent evidence contract."""

    @staticmethod
    def _complex(value: dict[str, float]) -> complex:
        return complex(value["real"], value["imag"])

    @classmethod
    def _column_norm(cls, matrix: list[list[dict[str, float]]], column: int) -> float:
        return math.sqrt(
            sum(abs(cls._complex(matrix[row][column])) ** 2 for row in range(len(matrix)))
        )

    def test_active_cases_are_registered_with_distinct_three_port_contracts(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        expected_paths = {
            "power_wave_s_to_z_three_port_active_real_equal_z0": oracle.ACTIVE_S_TO_Z_FIXTURE,
            "power_wave_z_to_s_three_port_active_real_equal_z0": oracle.ACTIVE_Z_TO_S_FIXTURE,
            "power_wave_renormalize_three_port_active_real_equal_z0": oracle.ACTIVE_RENORMALIZE_FIXTURE,
        }
        self.assertEqual(set(ACTIVE_CASE_SPECS), set(expected_paths))
        self.assertEqual(
            len({ACTIVE_CASE_SPECS[case_id]["seed"] for case_id in ACTIVE_CASE_SPECS}),
            3,
        )
        for case_id, spec in ACTIVE_CASE_SPECS.items():
            with self.subTest(case_id=case_id):
                case = registered[case_id]
                self.assertEqual(case.path, expected_paths[case_id])
                self.assertEqual(case.path.stem, case_id)
                self.assertEqual(case.comparison, "numeric_output")
                self.assertEqual(case.numeric_output_key, spec["output"])

    def test_active_fixtures_record_true_singular_value_evidence_and_real_equal_z0(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        for case_id, spec in ACTIVE_CASE_SPECS.items():
            with self.subTest(case_id=case_id):
                fixture = oracle._read_canonical_json(registered[case_id].path)
                metadata = fixture["metadata"]
                data = fixture["data"]
                active = metadata.get("active_network")
                self.assertIsInstance(active, dict)
                self.assertEqual(active["criterion"], oracle.ACTIVE_NETWORK_CRITERION)
                self.assertEqual(active["matrix_field"], spec["matrix_field"])
                self.assertEqual(
                    active["required_minimum"], oracle.ACTIVE_REQUIRED_SIGMA_MAX
                )
                self.assertEqual(active["observed_minimum"], spec["observed_minimum"])
                self.assertEqual(
                    active["observed_minimum"],
                    round(
                        active["observed_minimum"],
                        oracle.ACTIVE_OBSERVED_MINIMUM_DECIMAL_PLACES,
                    ),
                )
                self.assertTrue(math.isfinite(active["observed_minimum"]))
                self.assertGreater(
                    active["observed_minimum"], active["required_minimum"]
                )

                self.assertEqual(metadata["case_id"], case_id)
                self.assertEqual(metadata["operation"], spec["operation"])
                self.assertEqual(metadata["random_seed"], spec["seed"])
                self.assertEqual(metadata["numpy_version"], "2.5.1")
                self.assertEqual(metadata["scikit_rf_version"], "2.0.1")
                self.assertEqual(metadata["wave_definition"], "power")
                self.assertEqual(metadata["shape"]["frequency"], [3])

                if spec["operation"] == "renormalize_s":
                    self.assertEqual(metadata["shape"]["z0_source"], [3, 3])
                    flags = {
                        "complex": False,
                        "frequency_dependent": False,
                        "per_port": False,
                        "unit": "ohm",
                    }
                    self.assertEqual(metadata["reference_impedance"]["source"], flags)
                    self.assertEqual(metadata["reference_impedance"]["target"], flags)
                    z0_values = data["z0_source_ohm"]
                    target_values = data["z0_target_ohm"]
                    self.assertTrue(
                        all(
                            self._complex(value) == complex(spec["z0_source"], 0.0)
                            for row in z0_values
                            for value in row
                        )
                    )
                    self.assertTrue(
                        all(
                            self._complex(value) == complex(spec["z0_target"], 0.0)
                            for row in target_values
                            for value in row
                        )
                    )
                    relevant = data[spec["matrix_field"]]
                    self.assertEqual(metadata["shape"]["s_input"], [3, 3, 3])
                    self.assertEqual(metadata["shape"]["s_renormalized"], [3, 3, 3])
                else:
                    self.assertEqual(metadata["shape"]["input_z0"], [3, 3])
                    flags = {
                        "complex": False,
                        "frequency_dependent": False,
                        "per_port": False,
                        "unit": "ohm",
                    }
                    self.assertEqual(metadata["reference_impedance"], flags)
                    z0_values = data["z0_ohm"]
                    self.assertTrue(
                        all(
                            self._complex(value) == complex(spec["z0"], 0.0)
                            for row in z0_values
                            for value in row
                        )
                    )
                    relevant = data[spec["matrix_field"]]
                    output_shape = (
                        "input_s" if spec["operation"] == "s_to_z" else "output_s"
                    )
                    self.assertEqual(metadata["shape"][output_shape], [3, 3, 3])

                lower_bound_minimum = math.inf
                for frequency, matrix in enumerate(relevant):
                    self.assertEqual(len(matrix), 3)
                    self.assertTrue(all(len(row) == 3 for row in matrix))
                    lower_bound = max(
                        self._column_norm(matrix, column) for column in range(3)
                    )
                    lower_bound_minimum = min(lower_bound_minimum, lower_bound)
                    self.assertGreater(
                        lower_bound,
                        active["required_minimum"],
                        f"frequency {frequency} has no active column-norm lower bound",
                    )

                # The generator records the true NumPy SVD minimum.  It must
                # dominate the independently checked column-norm lower bound,
                # up to decimal serialization round-off.
                self.assertGreaterEqual(
                    active["observed_minimum"] + 1e-12,
                    lower_bound_minimum,
                )

                if spec["operation"] == "z_to_s":
                    # The direct Z input is a negative-resistance construction,
                    # not a round trip from the expected S output.
                    self.assertTrue(
                        all(
                            self._complex(data["z_ohm"][frequency][port][port]).real < 0.0
                            for frequency in range(3)
                            for port in range(3)
                        )
                    )


class MatrixFixtureCheckerTests(unittest.TestCase):
    """Protect the existing non-reciprocal matrix fixture contracts."""

    @staticmethod
    def _complex(value: dict[str, float]) -> complex:
        return complex(value["real"], value["imag"])

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


class RenormalizationRegistrationAndCheckerTests(unittest.TestCase):
    """Protect the renormalization fixture contract and output-only checks."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.case = next(
            case for case in oracle._CASES if case.case_id == RENORMALIZATION_CASE_ID
        )
        cls.fixture = oracle._read_canonical_json(cls.case.path)

    @staticmethod
    def _complex(value: dict[str, float]) -> complex:
        return complex(value["real"], value["imag"])

    def _check_document(self, document: dict[str, object]) -> int:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / self.case.path.name
            path.write_bytes(oracle._canonical_bytes(document))
            return oracle._check_numeric_fixture(
                path,
                self.fixture,
                "s_renormalized",
            )

    def test_case_is_registered_as_renormalization_numeric_output(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        self.assertIn(RENORMALIZATION_CASE_ID, registered)
        case = registered[RENORMALIZATION_CASE_ID]
        self.assertEqual(case.path, oracle.RENORMALIZE_FIXTURE)
        self.assertEqual(case.path.stem, RENORMALIZATION_CASE_ID)
        self.assertEqual(case.comparison, "numeric_output")
        self.assertEqual(case.numeric_output_key, "s_renormalized")

    def test_all_renormalization_cases_are_registered_with_profiles(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        expected_paths = {
            "power_wave_renormalize_one_port_real_scalar_z0": oracle.RENORMALIZE_ONE_PORT_FIXTURE,
            "power_wave_renormalize_two_port_complex_per_port_constant_z0": oracle.RENORMALIZE_TWO_PORT_FIXTURE,
            "power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0": oracle.RENORMALIZE_FOUR_PORT_FIXTURE,
            "power_wave_renormalize_eight_port_real_frequency_dependent_z0": oracle.RENORMALIZE_EIGHT_PORT_FIXTURE,
        }
        self.assertEqual(set(RENORMALIZATION_CASE_SPECS), set(expected_paths))
        for case_id, spec in RENORMALIZATION_CASE_SPECS.items():
            with self.subTest(case_id=case_id):
                case = registered[case_id]
                self.assertEqual(case.path, expected_paths[case_id])
                self.assertEqual(case.path.stem, case_id)
                self.assertEqual(case.comparison, "numeric_output")
                self.assertEqual(case.numeric_output_key, "s_renormalized")

                fixture = oracle._read_canonical_json(case.path)
                metadata = fixture["metadata"]
                data = fixture["data"]
                self.assertEqual(metadata["case_id"], case_id)
                self.assertEqual(metadata["operation"], "renormalize_s")
                self.assertEqual(metadata["numpy_version"], "2.5.1")
                self.assertEqual(metadata["scikit_rf_version"], "2.0.1")
                self.assertEqual(metadata["wave_definition"], "power")
                self.assertGreater(metadata["random_seed"], 0)
                expected_flags = {
                    "complex": spec["complex"],
                    "frequency_dependent": spec["frequency_dependent"],
                    "per_port": spec["per_port"],
                    "unit": "ohm",
                }
                self.assertEqual(
                    metadata["reference_impedance"]["source"], expected_flags
                )
                self.assertEqual(
                    metadata["reference_impedance"]["target"], expected_flags
                )
                expected_shape = {
                    "frequency": [spec["frequencies"]],
                    "s_input": [spec["frequencies"], spec["ports"], spec["ports"]],
                    "s_renormalized": [
                        spec["frequencies"],
                        spec["ports"],
                        spec["ports"],
                    ],
                    "z0_source": [spec["frequencies"], spec["ports"]],
                    "z0_target": [spec["frequencies"], spec["ports"]],
                }
                self.assertEqual(metadata["shape"], expected_shape)
                self.assertEqual(metadata["tolerance_policy"]["rtol"], 1e-12)
                self.assertEqual(metadata["tolerance_policy"]["atol"], 1e-12)
                self.assertIn("binary64", metadata["tolerance_policy"]["justification"])

                frequencies = spec["frequencies"]
                ports = spec["ports"]
                self.assertEqual(len(data["frequency_hz"]), frequencies)
                self.assertEqual(len(data["s_input"]), frequencies)
                self.assertEqual(len(data["s_renormalized"]), frequencies)
                self.assertEqual(len(data["z0_source_ohm"]), frequencies)
                self.assertEqual(len(data["z0_target_ohm"]), frequencies)
                for key in ("s_input", "s_renormalized"):
                    self.assertTrue(all(len(matrix) == ports for matrix in data[key]))
                    self.assertTrue(
                        all(
                            all(len(row) == ports for row in matrix)
                            for matrix in data[key]
                        )
                    )

                source_z0 = [
                    [self._complex(value) for value in row]
                    for row in data["z0_source_ohm"]
                ]
                target_z0 = [
                    [self._complex(value) for value in row]
                    for row in data["z0_target_ohm"]
                ]
                for z0 in (source_z0, target_z0):
                    self.assertTrue(all(len(row) == ports for row in z0))
                    self.assertTrue(
                        all(
                            value.real > 0.0
                            and math.isfinite(value.real)
                            and math.isfinite(value.imag)
                            for row in z0
                            for value in row
                        )
                    )
                    has_imaginary = any(
                        value.imag != 0.0 for row in z0 for value in row
                    )
                    self.assertEqual(has_imaginary, spec["complex"])
                    if spec["complex"]:
                        self.assertTrue(
                            all(
                                value.imag != 0.0
                                for row in z0
                                for value in row
                            )
                        )
                    rows_differ = any(row != z0[0] for row in z0[1:])
                    self.assertEqual(rows_differ, spec["frequency_dependent"])
                    ports_differ = any(
                        value != row[0] for row in z0 for value in row[1:]
                    )
                    self.assertEqual(ports_differ, spec["per_port"])
                self.assertTrue(
                    all(
                        abs(source_z0[frequency][port] - target_z0[frequency][port])
                        > 1.0
                        for frequency in range(frequencies)
                        for port in range(ports)
                    )
                )

                if ports > 1:
                    for matrix in data["s_input"]:
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

    def test_fixture_schema_and_reference_impedance_profiles(self) -> None:
        metadata = self.fixture["metadata"]
        data = self.fixture["data"]
        self.assertEqual(metadata["case_id"], RENORMALIZATION_CASE_ID)
        self.assertEqual(metadata["operation"], "renormalize_s")
        self.assertEqual(metadata["numpy_version"], "2.5.1")
        self.assertEqual(metadata["scikit_rf_version"], "2.0.1")
        self.assertEqual(metadata["wave_definition"], "power")
        self.assertGreater(metadata["random_seed"], 0)
        self.assertEqual(
            metadata["reference_impedance"]["source"],
            RENORMALIZATION_REFERENCE_FLAGS,
        )
        self.assertEqual(
            metadata["reference_impedance"]["target"],
            RENORMALIZATION_REFERENCE_FLAGS,
        )
        self.assertEqual(
            metadata["shape"],
            {
                "frequency": [3],
                "s_input": [3, 4, 4],
                "s_renormalized": [3, 4, 4],
                "z0_source": [3, 4],
                "z0_target": [3, 4],
            },
        )
        self.assertEqual(metadata["tolerance_policy"]["rtol"], 1e-12)
        self.assertEqual(metadata["tolerance_policy"]["atol"], 1e-12)
        self.assertIn("binary64", metadata["tolerance_policy"]["justification"])

        self.assertEqual(len(data["frequency_hz"]), 3)
        self.assertEqual(len(data["s_input"]), 3)
        self.assertEqual(len(data["s_renormalized"]), 3)
        for key in ("s_input", "s_renormalized"):
            self.assertTrue(all(len(matrix) == 4 for matrix in data[key]))
            self.assertTrue(
                all(
                    all(len(row) == 4 for row in matrix)
                    for matrix in data[key]
                )
            )

        source_z0 = [
            [self._complex(value) for value in row]
            for row in data["z0_source_ohm"]
        ]
        target_z0 = [
            [self._complex(value) for value in row]
            for row in data["z0_target_ohm"]
        ]
        for z0 in (source_z0, target_z0):
            self.assertEqual(len(z0), 3)
            self.assertTrue(all(len(row) == 4 for row in z0))
            self.assertTrue(all(value.real > 0.0 for row in z0 for value in row))
            self.assertTrue(all(value.imag != 0.0 for row in z0 for value in row))
            self.assertNotEqual(z0[0], z0[1])
            self.assertNotEqual(z0[1], z0[2])
            self.assertTrue(any(row[0] != row[1] for row in z0))
        self.assertTrue(
            all(
                abs(source_z0[frequency][port] - target_z0[frequency][port]) > 1.0
                for frequency in range(3)
                for port in range(4)
            )
        )

    def test_checker_tolerates_only_computed_renormalized_output(self) -> None:
        adjusted = copy.deepcopy(self.fixture)
        adjusted["data"]["s_renormalized"][0][0][0]["real"] += 1e-13
        self.assertEqual(self._check_document(adjusted), 0)

        drifted = copy.deepcopy(self.fixture)
        drifted["data"]["z0_target_ohm"][0][0]["real"] += 1.0
        self.assertEqual(self._check_document(drifted), 1)

        drifted_output = copy.deepcopy(self.fixture)
        drifted_output["data"]["s_input"][0][0][0]["real"] += 1e-3
        self.assertEqual(self._check_document(drifted_output), 1)

    def test_checker_tolerates_only_computed_output_for_all_cases(self) -> None:
        registered = {case.case_id: case for case in oracle._CASES}
        for case_id in RENORMALIZATION_CASE_IDS:
            with self.subTest(case_id=case_id):
                case = registered[case_id]
                fixture = oracle._read_canonical_json(case.path)
                adjusted = copy.deepcopy(fixture)
                adjusted["data"]["s_renormalized"][0][0][0]["real"] += 1e-13
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / case.path.name
                    path.write_bytes(oracle._canonical_bytes(adjusted))
                    self.assertEqual(
                        oracle._check_numeric_fixture(
                            path, fixture, "s_renormalized"
                        ),
                        0,
                    )

                drifted = copy.deepcopy(fixture)
                drifted["data"]["z0_target_ohm"][0][0]["real"] += 1.0
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / case.path.name
                    path.write_bytes(oracle._canonical_bytes(drifted))
                    self.assertEqual(
                        oracle._check_numeric_fixture(
                            path, fixture, "s_renormalized"
                        ),
                        1,
                    )


if __name__ == "__main__":
    unittest.main()
