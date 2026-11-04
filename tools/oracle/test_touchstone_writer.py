"""Regression tests for the Touchstone writer interoperability checker."""

from __future__ import annotations

import unittest

import check_touchstone_writer as checker


def _valid_v2_text(nports: int = 2) -> str:
    references = [37.0 + 11.0 * port for port in range(nports)]
    lines = [
        "[Version] 2.0",
        f"# Hz S RI R {references[0]:g}",
        f"[Number of Ports] {nports}",
    ]
    if nports == 2:
        lines.append("[Two-Port Data Order] 12_21")
    lines.extend(
        [
            "[Number of Frequencies] 2",
            "[Reference] " + " ".join(f"{reference:g}" for reference in references),
            "[Matrix Format] Full",
            "[Network Data]",
        ]
    )
    for frequency_index in range(2):
        values = [f"{1.0e6 * (frequency_index + 1):g}"]
        for row in range(nports):
            for column in range(nports):
                value = float(frequency_index * 100 + row * 10 + column + 1)
                values.extend((f"{0.005 * value:g}", f"{-0.003 * value:g}"))
        lines.append(" ".join(values))
    lines.append("[End]")
    return "\n".join(lines) + "\n"


class TouchstoneWriterLayoutCheckerTests(unittest.TestCase):
    def test_independent_v2_layout_contract_accepts_asymmetric_records(self) -> None:
        checker._check_layout_contract(
            _valid_v2_text(), "v2-two-port", 2, [37.0, 48.0]
        )

    def test_v2_data_order_drift_is_detected(self) -> None:
        lines = _valid_v2_text().splitlines()
        first_record = lines[8].split()
        first_record[1:5], first_record[5:9] = first_record[5:9], first_record[1:5]
        lines[8] = " ".join(first_record)
        with self.assertRaises(AssertionError):
            checker._check_layout_contract(
                "\n".join(lines) + "\n", "v2-two-port", 2, [37.0, 48.0]
            )

    def test_v2_reference_drift_is_detected(self) -> None:
        drifted = _valid_v2_text().replace("[Reference] 37 48", "[Reference] 37 49")
        with self.assertRaises(AssertionError):
            checker._check_layout_contract(
                drifted, "v2-two-port", 2, [37.0, 48.0]
            )

    def test_physical_parameter_header_and_raw_records_are_independent(self) -> None:
        physical = _valid_v2_text().replace("# Hz S RI R", "# Hz Z RI R")
        checker._check_layout_contract(
            physical, "v2-z-three-port", 2, [37.0, 48.0], parameter="Z"
        )
        frequency, matrix = checker._parse_v2_full_records(
            physical, checker._load_dependencies()[0], 2
        )
        self.assertEqual(frequency.tolist(), [1.0e6, 2.0e6])
        self.assertEqual(matrix.shape, (2, 2, 2))
        self.assertEqual(matrix[0, 0, 0], complex(0.005, -0.003))

    def test_physical_recipe_is_asymmetric_finite_and_well_conditioned(self) -> None:
        np, _ = checker._load_dependencies()
        frequency, z_physical, source_s, z0 = checker._physical_expected(np, "Z")
        _, y_physical, y_source_s, _ = checker._physical_expected(np, "Y")
        self.assertEqual(frequency.tolist(), [1.0e6, 2.0e6])
        self.assertEqual(source_s.shape, (2, 3, 3))
        self.assertTrue(np.isfinite(source_s).all())
        self.assertNotEqual(source_s[0, 0, 1], source_s[0, 1, 0])
        np.testing.assert_array_equal(y_source_s, source_s)
        self.assertEqual(z0[0].real.tolist(), [31.0, 47.0, 73.0])
        self.assertTrue(np.isfinite(z_physical).all())
        self.assertTrue(np.isfinite(y_physical).all())
        self.assertNotEqual(z_physical[0, 0, 1], z_physical[0, 1, 0])
        self.assertNotEqual(y_physical[0, 0, 1], y_physical[0, 1, 0])
        self.assertFalse(np.allclose(z_physical, source_s))
        self.assertFalse(np.allclose(y_physical, source_s))
        for sample in range(2):
            scale = 2.0 * np.sqrt(np.real(z0[sample]))
            f = np.diag(1.0 / scale)
            g = np.diag(z0[sample])
            identity = np.eye(3, dtype=complex)
            z_a = (identity - source_s[sample]) @ f
            z_b = (source_s[sample] @ g + g.conj()) @ f
            y_a = (source_s[sample] @ g + g.conj()) @ f
            y_b = (identity - source_s[sample]) @ f
            self.assertLess(np.linalg.cond(z_a), 10.0)
            self.assertLess(np.linalg.cond(y_a), 10.0)
            np.testing.assert_allclose(z_a @ z_physical[sample], z_b)
            np.testing.assert_allclose(y_a @ y_physical[sample], y_b)


if __name__ == "__main__":
    unittest.main()
