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


if __name__ == "__main__":
    unittest.main()
