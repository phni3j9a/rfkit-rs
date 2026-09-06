#!/usr/bin/env python3
"""Generate and check the checked-in scikit-rf oracle fixtures.

The original three-port Network fixture is retained as a stable input
contract.  The power-wave operation fixtures additionally form a small
conformance matrix over port count and reference-impedance structure, with
dedicated reciprocal, passive, and active three-port cases for each existing
kernel.  Reciprocal cases additionally carry deterministic passive-network
evidence for every relevant power-wave S matrix.
The near-singular operation fixtures additionally exercise an explicitly
constructed, nonsingular upper-triangular conversion system with one binary
power-of-two diagonal factor close to (but safely above) scikit-rf's
eigenvalue-nudging threshold.  This module is kept independent of the Rust
implementation so it can serve as a stable numerical reference for the
internal conversion kernels.  The impedance/admittance fixtures use the
public ``skrf.network.z2y`` and ``skrf.network.y2z`` functions for expected
outputs, with direct Z and Y inputs constructed independently for each
direction.  The power-wave S/Y fixtures use the public
``skrf.network.s2y`` and ``skrf.network.y2s`` functions with explicit
``s_def="power"`` and independently constructed direct inputs in each
direction.  The matched-junction fixture uses independent three-port and
four-port inputs and the public ``skrf.network.connect`` operation with
explicit power-wave Network constructors; its output-only comparison leaves
inputs, z0, and port-order metadata exact.
The interpolation fixture uses the public
``Network.interpolate(..., basis="s", coords="cart", kind="linear")`` operation
for both S and z0 outputs.  SciPy is pinned explicitly because scikit-rf
delegates the interpolation numerics to it.
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
EXPECTED_SCIPY_VERSION = "1.18.1"
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

# The matrix cases intentionally use descriptive, stable ids and one fixture
# per operation/profile.  Keep these paths explicit so registration remains
# visible in reviews and the default harness checks every case.
S_TO_Z_ONE_PORT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_s_to_z_one_port_real_scalar_z0.json"
)
S_TO_Z_TWO_PORT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_s_to_z_two_port_complex_per_port_constant_z0.json"
)
S_TO_Z_FOUR_PORT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_s_to_z_four_port_real_frequency_dependent_z0.json"
)
S_TO_Z_EIGHT_PORT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_s_to_z_eight_port_complex_per_port_frequency_dependent_z0.json"
)
Z_TO_S_ONE_PORT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_z_to_s_one_port_real_scalar_z0.json"
)
Z_TO_S_TWO_PORT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_z_to_s_two_port_complex_per_port_constant_z0.json"
)
Z_TO_S_FOUR_PORT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_z_to_s_four_port_real_frequency_dependent_z0.json"
)
Z_TO_S_EIGHT_PORT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_z_to_s_eight_port_complex_per_port_frequency_dependent_z0.json"
)
RENORMALIZE_ONE_PORT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_renormalize_one_port_real_scalar_z0.json"
)
RENORMALIZE_TWO_PORT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_renormalize_two_port_complex_per_port_constant_z0.json"
)
RENORMALIZE_FOUR_PORT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0.json"
)
RENORMALIZE_EIGHT_PORT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_renormalize_eight_port_real_frequency_dependent_z0.json"
)
# Keep the original constant as a compatibility alias for callers of the
# pre-matrix harness.  New registrations use the descriptive per-case names.
RENORMALIZE_FIXTURE = RENORMALIZE_FOUR_PORT_FIXTURE

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
MATRIX_S_TO_Z_TOLERANCE_JUSTIFICATION = (
    "Strict binary64 tolerance for a well-conditioned, modest-magnitude "
    "deterministic matrix case; diagonal-dominance checks keep I-S away from "
    "singularity while allowing normal cross-language linear-algebra rounding."
)
MATRIX_Z_TO_S_TOLERANCE_JUSTIFICATION = (
    "Strict binary64 tolerance for a well-conditioned, modest-magnitude "
    "deterministic matrix case; diagonal-dominance checks keep Z+G away from "
    "singularity while allowing normal cross-language linear-algebra rounding."
)
RENORMALIZE_FOUR_PORT_RANDOM_SEED = 20_260_915
RENORMALIZE_ONE_PORT_RANDOM_SEED = 20_260_916
RENORMALIZE_TWO_PORT_RANDOM_SEED = 20_260_917
RENORMALIZE_EIGHT_PORT_RANDOM_SEED = 20_260_918
# Preserve the old name for code that imported the original single-case
# generator constant.  The four-port case remains the existing fixture.
RENORMALIZE_RANDOM_SEED = RENORMALIZE_FOUR_PORT_RANDOM_SEED
RENORMALIZE_RTOL = 1e-12
RENORMALIZE_ATOL = 1e-12
RENORMALIZE_TOLERANCE_JUSTIFICATION = (
    "Strict binary64 tolerance for a well-conditioned, modest-magnitude "
    "deterministic renormalization case; the source S input passes a conservative "
    "I-S diagonal-dominance guard while allowing normal cross-language "
    "linear-algebra rounding."
)

RECIPROCAL_S_TO_Z_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_s_to_z_three_port_reciprocal_real_equal_z0.json"
)
RECIPROCAL_Z_TO_S_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_z_to_s_three_port_reciprocal_real_equal_z0.json"
)
RECIPROCAL_RENORMALIZE_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_renormalize_three_port_reciprocal_real_equal_z0.json"
)
RECIPROCAL_S_TO_Z_RANDOM_SEED = 20_260_921
RECIPROCAL_Z_TO_S_RANDOM_SEED = 20_260_922
RECIPROCAL_RENORMALIZE_RANDOM_SEED = 20_260_923
RECIPROCAL_S_TO_Z_Z0_OHM = 61.25
RECIPROCAL_RENORMALIZE_SOURCE_Z0_OHM = 42.75
RECIPROCAL_RENORMALIZE_TARGET_Z0_OHM = 86.5
RECIPROCAL_TOLERANCE_JUSTIFICATION = (
    "Strict binary64 tolerance for a well-conditioned, modest-magnitude, "
    "deterministic reciprocal three-port case; generation-time diagonal-"
    "dominance checks keep the conversion systems away from exact singularity "
    "while allowing normal cross-language linear-algebra rounding."
)

PASSIVE_REQUIRED_SIGMA_MAX = 0.8
PASSIVE_OBSERVED_MAXIMUM_DECIMAL_PLACES = 12
PASSIVE_NETWORK_CRITERION = (
    "largest singular value of every relevant power-wave S matrix is strictly less than 1"
)

ACTIVE_S_TO_Z_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_s_to_z_three_port_active_real_equal_z0.json"
)
ACTIVE_Z_TO_S_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_z_to_s_three_port_active_real_equal_z0.json"
)
ACTIVE_RENORMALIZE_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_renormalize_three_port_active_real_equal_z0.json"
)
ACTIVE_S_TO_Z_RANDOM_SEED = 20_260_924
ACTIVE_Z_TO_S_RANDOM_SEED = 20_260_925
ACTIVE_RENORMALIZE_RANDOM_SEED = 20_260_926
ACTIVE_S_TO_Z_Z0_OHM = 57.25
ACTIVE_RENORMALIZE_SOURCE_Z0_OHM = 57.25
ACTIVE_RENORMALIZE_TARGET_Z0_OHM = 91.75
ACTIVE_REQUIRED_SIGMA_MAX = 1.2
ACTIVE_OBSERVED_MINIMUM_DECIMAL_PLACES = 12
ACTIVE_NETWORK_CRITERION = (
    "largest singular value of the relevant power-wave S matrix is strictly greater than 1"
)

NEAR_SINGULAR_S_TO_Z_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_s_to_z_three_port_near_singular_real_equal_z0.json"
)
NEAR_SINGULAR_Z_TO_S_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_z_to_s_three_port_near_singular_real_equal_z0.json"
)
NEAR_SINGULAR_S_TO_Z_RANDOM_SEED = 20_260_927
NEAR_SINGULAR_Z_TO_S_RANDOM_SEED = 20_260_928
NEAR_SINGULAR_NFREQ = 3
NEAR_SINGULAR_NPORTS = 3
NEAR_SINGULAR_Z0_OHM = 64.0
NEAR_SINGULAR_BINARY_EXPONENT = -20
NEAR_SINGULAR_DIAGONAL_VALUE = 2.0**NEAR_SINGULAR_BINARY_EXPONENT
NEAR_SINGULAR_DIAGONAL_FACTORS = (
    NEAR_SINGULAR_DIAGONAL_VALUE,
    0.625,
    0.75,
)
NEAR_SINGULAR_DETERMINANT = math.prod(NEAR_SINGULAR_DIAGONAL_FACTORS)
NEAR_SINGULAR_SYSTEM_STRUCTURE = "upper_triangular"
NEAR_SINGULAR_S_TO_Z_SYSTEM = "I-S"
NEAR_SINGULAR_Z_TO_S_SYSTEM = "(Z+z0 I)/z0"
NEAR_SINGULAR_S_TO_Z_RTOL = 1e-12
NEAR_SINGULAR_S_TO_Z_ATOL_OHM = 1e-12
NEAR_SINGULAR_Z_TO_S_RTOL = 1e-12
NEAR_SINGULAR_Z_TO_S_ATOL = 1e-12
NEAR_SINGULAR_TOLERANCE_JUSTIFICATION = (
    "case-local strict binary64 tolerance for a deterministic three-port "
    "upper-triangular system with one diagonal factor 2^-20="
    f"{NEAR_SINGULAR_DIAGONAL_VALUE:.17g}; this factor is non-zero, "
    "comfortably above scikit-rf EIG_COND=1e-9, and intentionally much "
    "smaller than the diagonal factors in the well-conditioned fixtures. "
    "The resulting finite outputs are O(10^8 ohm) for S-to-Z and O(10^6) "
    "for Z-to-S; the existing 1e-12 relative/absolute policy therefore "
    "allows binary64 solve round-off at the amplified scale while still "
    "catching material disagreement. No platform-sensitive condition number "
    "is recorded."
)

# The impedance/admittance cases deliberately use direct parameter arrays and
# no reference impedance or wave definition.  Keep the four paths explicit so
# registration and the default all-case harness remain auditable.
IMPEDANCE_ADMITTANCE_Z_TO_Y_WELL_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "impedance_admittance_z_to_y_three_port_well_conditioned.json"
)
IMPEDANCE_ADMITTANCE_Y_TO_Z_WELL_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "impedance_admittance_y_to_z_three_port_well_conditioned.json"
)
IMPEDANCE_ADMITTANCE_Z_TO_Y_NEAR_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "impedance_admittance_z_to_y_three_port_near_singular.json"
)
IMPEDANCE_ADMITTANCE_Y_TO_Z_NEAR_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "impedance_admittance_y_to_z_three_port_near_singular.json"
)
IMPEDANCE_ADMITTANCE_Z_TO_Y_WELL_RANDOM_SEED = 20_260_933
IMPEDANCE_ADMITTANCE_Y_TO_Z_WELL_RANDOM_SEED = 20_260_934
IMPEDANCE_ADMITTANCE_Z_TO_Y_NEAR_RANDOM_SEED = 20_260_935
IMPEDANCE_ADMITTANCE_Y_TO_Z_NEAR_RANDOM_SEED = 20_260_936
IMPEDANCE_ADMITTANCE_NFREQ = 3
IMPEDANCE_ADMITTANCE_NPORTS = 3
IMPEDANCE_ADMITTANCE_RTOL = 1e-12
IMPEDANCE_ADMITTANCE_ATOL_S = 1e-12
IMPEDANCE_ADMITTANCE_ATOL_OHM = 1e-12
IMPEDANCE_ADMITTANCE_NEAR_BINARY_EXPONENT = -20
IMPEDANCE_ADMITTANCE_NEAR_DIAGONAL_VALUE = (
    2.0**IMPEDANCE_ADMITTANCE_NEAR_BINARY_EXPONENT
)
IMPEDANCE_ADMITTANCE_NEAR_DIAGONAL_FACTORS = (
    IMPEDANCE_ADMITTANCE_NEAR_DIAGONAL_VALUE,
    0.625,
    0.75,
)
IMPEDANCE_ADMITTANCE_NEAR_DETERMINANT = math.prod(
    IMPEDANCE_ADMITTANCE_NEAR_DIAGONAL_FACTORS
)
IMPEDANCE_ADMITTANCE_NEAR_TOLERANCE_JUSTIFICATION = (
    "Case-local strict binary64 tolerance for a deterministic three-port "
    "upper-triangular direct parameter matrix with one exact binary factor "
    f"2^{IMPEDANCE_ADMITTANCE_NEAR_BINARY_EXPONENT}="
    f"{IMPEDANCE_ADMITTANCE_NEAR_DIAGONAL_VALUE:.17g}; the factor is finite "
    "and non-zero, and pinned NumPy matrix_rank reports full rank. The "
    "determinant product is recorded from the input diagonal factors, with "
    "no condition number used for acceptance."
)

# The power-wave S/Y cases use direct, direction-specific inputs.  They share
# only the explicit non-50-ohm complex per-port/frequency-dependent z0 profile;
# neither direction is built from the other direction's expected output.
POWER_WAVE_S_TO_Y_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_s_to_y_three_port_complex_z0.json"
)
POWER_WAVE_Y_TO_S_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_y_to_s_three_port_complex_z0.json"
)
POWER_WAVE_S_TO_Y_RANDOM_SEED = 20_260_937
POWER_WAVE_Y_TO_S_RANDOM_SEED = 20_260_938
POWER_WAVE_ADMITTANCE_NFREQ = 3
POWER_WAVE_ADMITTANCE_NPORTS = 3
POWER_WAVE_S_TO_Y_RTOL = 1e-12
POWER_WAVE_S_TO_Y_ATOL_S = 1e-12
POWER_WAVE_Y_TO_S_RTOL = 1e-12
POWER_WAVE_Y_TO_S_ATOL = 1e-12
POWER_WAVE_ADMITTANCE_TOLERANCE_JUSTIFICATION = (
    "Strict binary64 tolerance for a well-conditioned, modest-magnitude, "
    "deterministic complex N-port case; generation-time diagonal-dominance and "
    "finite-output guards keep direct public scikit-rf conversion away from "
    "singularity while allowing normal cross-language solve round-off."
)

# Matched-junction connection is intentionally kept as one standalone oracle
# case.  A and B use independent local generators so the fixture exercises all
# four N-port blocks rather than certifying a symmetric or identity shortcut.
CONNECT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_connect_matched_three_to_four_port_real_frequency_dependent_z0.json"
)
CONNECT_CASE_ID = "power_wave_connect_matched_three_to_four_port_real_frequency_dependent_z0"
CONNECT_RANDOM_SEED_A = 20_260_939
CONNECT_RANDOM_SEED_B = 20_260_940
CONNECT_NFREQ = 3
CONNECT_NPORTS_A = 3
CONNECT_NPORTS_B = 4
CONNECT_PORT_A = 1
CONNECT_PORT_B = 2
CONNECT_RTOL = 1e-12
CONNECT_ATOL = 1e-12
CONNECT_TOLERANCE_JUSTIFICATION = (
    "Strict binary64 tolerance for a finite, well-conditioned matched-junction "
    "three-to-four-port case; the output is checked with the recorded numeric "
    "bound while frequencies, inputs, z0, ordering metadata, and all other "
    "contract fields remain exact."
)

# Same-network inner-connect is kept as a separate oracle case.  The direct
# input uses one local RNG and connects two non-adjacent ports, leaving three
# survivors in their original order.  Its selected junction impedances are
# real, positive, frequency-dependent, and intentionally non-50 ohm.
INNER_CONNECT_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_inner_connect_matched_five_port_real_frequency_dependent_z0.json"
)
INNER_CONNECT_CASE_ID = (
    "power_wave_inner_connect_matched_five_port_real_frequency_dependent_z0"
)
INNER_CONNECT_RANDOM_SEED = 20_260_941
INNER_CONNECT_NFREQ = 3
INNER_CONNECT_NPORTS = 5
INNER_CONNECT_PORT_K = 1
INNER_CONNECT_PORT_L = 3
INNER_CONNECT_RTOL = 1e-12
INNER_CONNECT_ATOL = 1e-12
INNER_CONNECT_TOLERANCE_JUSTIFICATION = (
    "Strict binary64 tolerance for a finite, well-conditioned matched-junction "
    "five-port case; the output is checked with the recorded numeric bound while "
    "frequencies, inputs, z0, ordering metadata, and all other contract fields "
    "remain exact."
)

# The interpolation case is kept as one direct, independently generated
# fixture.  Its expected S and z0 arrays come from public scikit-rf
# ``Network.interpolate`` behavior; both outputs remain numeric-tolerance
# fields while the source/target grids and all input metadata stay exact.
INTERPOLATION_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "interpolation_cartesian_linear_three_port_complex_z0.json"
)
INTERPOLATION_CASE_ID = "interpolation_cartesian_linear_three_port_complex_z0"
INTERPOLATION_RANDOM_SEED = 20_260_942
INTERPOLATION_RTOL = 1e-12
INTERPOLATION_ATOL = 1e-12
INTERPOLATION_TOLERANCE_JUSTIFICATION = (
    "Strict binary64 mixed relative/absolute tolerance for a deterministic, "
    "irregular-grid three-port Cartesian linear interpolation case; it allows "
    "normal cross-language interpolation rounding while catching material "
    "disagreement."
)

# Explicit-grid matched connection composes two independently sampled inputs:
# each side is interpolated once through the public Network API, then the
# public matched connection operation is called once on the common target
# grid.  Keep the source axes distinct and retain enough ports on each side to
# exercise all four N-port output blocks.
COMPOSITION_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "power_wave_connect_matched_explicit_grid_three_to_four_port_complex_z0.json"
)
COMPOSITION_CASE_ID = (
    "power_wave_connect_matched_explicit_grid_three_to_four_port_complex_z0"
)
COMPOSITION_RANDOM_SEED_A = 20_260_943
COMPOSITION_RANDOM_SEED_B = 20_260_944
COMPOSITION_NPORTS_A = 3
COMPOSITION_NPORTS_B = 4
COMPOSITION_PORT_A = 1
COMPOSITION_PORT_B = 2
COMPOSITION_RTOL = 1e-12
COMPOSITION_ATOL = 1e-12
COMPOSITION_TOLERANCE_JUSTIFICATION = (
    "Strict binary64 mixed relative/absolute tolerance for a deterministic, "
    "well-conditioned explicit-grid matched connection; each source network "
    "is interpolated independently by pinned SciPy-backed scikit-rf before "
    "the public matched connection, allowing normal cross-language rounding "
    "while catching material disagreement."
)


class _OracleCase(NamedTuple):
    """A registered fixture, builder, and case-specific check strategy."""

    case_id: str
    path: Path
    builder: Callable[[Any, Any], dict[str, Any]]
    comparison: str
    numeric_output_key: str | tuple[str, ...] | None = None


def _load_dependencies() -> tuple[Any, Any]:
    """Import the pinned dependencies and fail with an actionable message."""

    # Check distribution metadata before importing NumPy/scikit-rf.  This
    # keeps an incompatible installation (including an ABI-incompatible
    # NumPy) on the explicit wrong-version failure path.
    try:
        installed_numpy = version("numpy")
        installed_skrf = version("scikit-rf")
        installed_scipy = version("scipy")
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
    if installed_scipy != EXPECTED_SCIPY_VERSION:
        distribution_version_errors.append(
            f"scipy=={EXPECTED_SCIPY_VERSION} (found {installed_scipy})"
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

    try:
        import scipy
    except ImportError as error:  # pragma: no cover - exercised in a clean env
        raise RuntimeError(
            "scipy is not installed; create an isolated environment and run "
            "`python -m pip install -r requirements.txt`"
        ) from error

    actual_numpy = np.__version__
    actual_skrf = skrf.__version__
    actual_scipy = scipy.__version__
    version_errors: list[str] = []
    if actual_numpy != EXPECTED_NUMPY_VERSION:
        version_errors.append(f"numpy=={EXPECTED_NUMPY_VERSION} (found {actual_numpy})")
    if actual_skrf != EXPECTED_SCIKIT_RF_VERSION:
        version_errors.append(
            f"scikit-rf=={EXPECTED_SCIKIT_RF_VERSION} (found {actual_skrf})"
        )
    if actual_scipy != EXPECTED_SCIPY_VERSION:
        version_errors.append(
            f"scipy=={EXPECTED_SCIPY_VERSION} (found {actual_scipy})"
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


def _assert_non_symmetric(np: Any, matrix: Any, *, name: str) -> None:
    """Require every multiport input matrix to be non-symmetric.

    The operation fixtures are intended to exercise genuine N-port behavior,
    not a reciprocal/symmetric shortcut.  A one-port matrix is necessarily
    symmetric and is therefore exempt from this assertion.
    """

    if matrix.shape[1] <= 1:
        return
    for frequency in range(matrix.shape[0]):
        if np.array_equal(matrix[frequency], matrix[frequency].T):
            raise ValueError(
                f"{name} must be non-symmetric at frequency index {frequency}"
            )


def _assert_exact_symmetric(np: Any, matrix: Any, *, name: str) -> None:
    """Require every frequency slice to be exactly transpose-symmetric.

    Reciprocal fixtures mirror one generated triangle without conjugation.
    Keeping this assertion separate from the non-reciprocal guard makes that
    construction contract explicit and catches accidental Hermitian mirroring.
    """

    if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
        raise ValueError(f"{name} must be a stack of square matrices")
    if not np.array_equal(matrix, np.swapaxes(matrix, 1, 2)):
        raise ValueError(f"{name} must be exactly transpose-symmetric")


def _assert_output_symmetric(
    np: Any,
    matrix: Any,
    *,
    name: str,
    rtol: float,
    atol: float,
) -> None:
    """Check reciprocal output symmetry with the recorded mixed tolerance."""

    if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
        raise ValueError(f"{name} must be a stack of square matrices")
    for frequency in range(matrix.shape[0]):
        for row in range(matrix.shape[1]):
            for column in range(row + 1, matrix.shape[2]):
                lhs = complex(matrix[frequency, row, column])
                rhs = complex(matrix[frequency, column, row])
                difference = abs(lhs - rhs)
                bound = atol + rtol * max(abs(lhs), abs(rhs))
                if not np.isfinite(difference) or difference > bound:
                    raise ValueError(
                        f"{name} is not reciprocal at frequency {frequency}, "
                        f"ports ({row}, {column}): difference={difference:.17g}, "
                        f"bound={bound:.17g}"
                    )


def _assert_real_positive_equal_z0(np: Any, z0: Any, *, name: str) -> None:
    """Require finite, real-positive, equal-per-port reference impedances."""

    if z0.ndim != 2 or z0.shape[0] < 1 or z0.shape[1] < 1:
        raise ValueError(f"{name} z0 must be a non-empty (frequency, port) array")
    if not np.isfinite(z0).all():
        raise ValueError(f"{name} z0 must be finite")
    if not (z0.real > 0.0).all() or (z0.imag != 0.0).any():
        raise ValueError(f"{name} z0 must be real and strictly positive")
    if not np.all(z0 == z0[0, 0]):
        raise ValueError(f"{name} z0 must be equal across frequency and ports")


def _assert_s_conditioning(s: Any) -> None:
    """Check a conservative diagonal-dominance bound for ``I-S``.

    The bound is deliberately used only as a generation-time guard.  No
    platform-sensitive condition-number estimate is written into a fixture.
    Strict row diagonal dominance with a comfortable margin is enough to keep
    the conversion solve away from the exact-singular edge case.
    """

    for frequency in range(s.shape[0]):
        for row in range(s.shape[1]):
            diagonal = abs(1.0 - s[frequency, row, row])
            off_diagonal = sum(
                abs(s[frequency, row, column])
                for column in range(s.shape[2])
                if column != row
            )
            if diagonal <= off_diagonal + 0.5:
                raise ValueError(
                    "S input failed the conservative diagonal-dominance "
                    f"bound at frequency {frequency}, row {row}"
                )


def _assert_z_conditioning(z: Any, z0: Any) -> None:
    """Check a conservative diagonal-dominance bound for ``Z+G``.

    As with :func:`_assert_s_conditioning`, this validates a generous margin
    without recording condition values that could vary across linear-algebra
    implementations or platforms.
    """

    for frequency in range(z.shape[0]):
        for row in range(z.shape[1]):
            diagonal = abs(z[frequency, row, row] + z0[frequency, row])
            off_diagonal = sum(
                abs(z[frequency, row, column])
                for column in range(z.shape[2])
                if column != row
            )
            if diagonal <= off_diagonal + 10.0:
                raise ValueError(
                    "Z input failed the conservative diagonal-dominance "
                    f"bound at frequency {frequency}, row {row}"
                )


def _assert_upper_triangular_system(
    np: Any,
    system: Any,
    *,
    system_matrix: str,
) -> None:
    """Validate the exact, input-derived structure of a near-singular system.

    This deliberately uses only structural and binary-value checks.  A
    platform-sensitive condition-number estimate would make the fixture
    contract less reproducible and is not needed to establish a non-zero
    determinant for the upper-triangular construction.
    """

    if (
        system.ndim != 3
        or system.shape[1] != NEAR_SINGULAR_NPORTS
        or system.shape[2] != NEAR_SINGULAR_NPORTS
    ):
        raise ValueError(
            f"{system_matrix} near-singular system must have shape "
            f"(nfreq, {NEAR_SINGULAR_NPORTS}, {NEAR_SINGULAR_NPORTS})"
        )
    if not np.isfinite(system).all():
        raise ValueError(f"{system_matrix} near-singular system must be finite")

    lower_triangle = np.tril(system, k=-1)
    if not np.array_equal(lower_triangle, np.zeros_like(lower_triangle)):
        raise ValueError(f"{system_matrix} near-singular system must be upper triangular")

    expected_diagonal = np.asarray(
        NEAR_SINGULAR_DIAGONAL_FACTORS,
        dtype=np.complex128,
    )
    actual_diagonal = np.diagonal(system, axis1=1, axis2=2)
    if not np.array_equal(
        actual_diagonal,
        np.broadcast_to(expected_diagonal, actual_diagonal.shape),
    ):
        raise ValueError(
            f"{system_matrix} near-singular system has unexpected diagonal factors"
        )

    determinant = math.prod(NEAR_SINGULAR_DIAGONAL_FACTORS)
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("near-singular diagonal factors must have a finite non-zero product")


def _near_singular_metadata(*, system_matrix: str) -> dict[str, Any]:
    """Describe the deterministic binary upper-triangular construction."""

    factors = [float(value) for value in NEAR_SINGULAR_DIAGONAL_FACTORS]
    determinant = float(NEAR_SINGULAR_DETERMINANT)
    if not all(math.isfinite(value) and value != 0.0 for value in factors):
        raise ValueError("near-singular determinant factors must be finite and non-zero")
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("near-singular determinant must be finite and non-zero")

    return {
        "binary_exponent": NEAR_SINGULAR_BINARY_EXPONENT,
        "determinant": determinant,
        "determinant_factors": factors,
        "determinant_factors_nonzero": True,
        "matrix_structure": NEAR_SINGULAR_SYSTEM_STRUCTURE,
        "small_diagonal_port": 0,
        "small_diagonal_value": float(NEAR_SINGULAR_DIAGONAL_VALUE),
        "system_matrix": system_matrix,
    }


def _near_singular_frequency_and_z0(np: Any) -> tuple[Any, Any, Any]:
    """Build the shared multi-frequency real/equal reference impedance."""

    frequency_hz = np.array(
        [0.77e9 + 0.31e9 * index for index in range(NEAR_SINGULAR_NFREQ)],
        dtype=np.float64,
    )
    constructor_z0 = float(NEAR_SINGULAR_Z0_OHM)
    expanded_z0 = np.full(
        (NEAR_SINGULAR_NFREQ, NEAR_SINGULAR_NPORTS),
        constructor_z0,
        dtype=np.complex128,
    )
    _assert_real_positive_equal_z0(np, expanded_z0, name="near-singular")
    return frequency_hz, constructor_z0, expanded_z0


def _near_singular_s_inputs(
    np: Any,
    *,
    seed: int,
) -> tuple[Any, Any, Any, Any]:
    """Build direct S data whose ``I-S`` system is upper triangular."""

    frequency_hz, constructor_z0, expanded_z0 = _near_singular_frequency_and_z0(np)
    rng = np.random.default_rng(seed)
    system = np.zeros(
        (NEAR_SINGULAR_NFREQ, NEAR_SINGULAR_NPORTS, NEAR_SINGULAR_NPORTS),
        dtype=np.complex128,
    )
    for frequency in range(NEAR_SINGULAR_NFREQ):
        for row, factor in enumerate(NEAR_SINGULAR_DIAGONAL_FACTORS):
            system[frequency, row, row] = factor
            for column in range(row + 1, NEAR_SINGULAR_NPORTS):
                system[frequency, row, column] = complex(
                    rng.normal(loc=0.0, scale=0.03125),
                    rng.normal(loc=0.0, scale=0.03125),
                )

    _assert_upper_triangular_system(
        np,
        system,
        system_matrix=NEAR_SINGULAR_S_TO_Z_SYSTEM,
    )
    identity = np.eye(NEAR_SINGULAR_NPORTS, dtype=np.complex128)
    source_s = identity[None, :, :] - system
    if not np.isfinite(source_s).all():
        raise ValueError("near-singular S input must be finite")
    return frequency_hz, source_s, constructor_z0, expanded_z0


def _near_singular_z_inputs(
    np: Any,
    *,
    seed: int,
) -> tuple[Any, Any, Any, Any]:
    """Build direct Z data whose normalized ``(Z+z0 I)/z0`` is upper triangular."""

    frequency_hz, constructor_z0, expanded_z0 = _near_singular_frequency_and_z0(np)
    rng = np.random.default_rng(seed)
    normalized_system = np.zeros(
        (NEAR_SINGULAR_NFREQ, NEAR_SINGULAR_NPORTS, NEAR_SINGULAR_NPORTS),
        dtype=np.complex128,
    )
    for frequency in range(NEAR_SINGULAR_NFREQ):
        for row, factor in enumerate(NEAR_SINGULAR_DIAGONAL_FACTORS):
            normalized_system[frequency, row, row] = factor
            for column in range(row + 1, NEAR_SINGULAR_NPORTS):
                normalized_system[frequency, row, column] = complex(
                    rng.normal(loc=0.0, scale=0.03125),
                    rng.normal(loc=0.0, scale=0.03125),
                )

    _assert_upper_triangular_system(
        np,
        normalized_system,
        system_matrix=NEAR_SINGULAR_Z_TO_S_SYSTEM,
    )
    identity = np.eye(NEAR_SINGULAR_NPORTS, dtype=np.complex128)
    source_z = expanded_z0[:, :, None] * (normalized_system - identity[None, :, :])
    if not np.isfinite(source_z).all():
        raise ValueError("near-singular Z input must be finite")
    return frequency_hz, source_z, constructor_z0, expanded_z0


def _active_network_metadata(np: Any, s: Any, *, matrix_field: str) -> dict[str, Any]:
    """Return and validate the active-network evidence for a power-wave S stack.

    ``numpy.linalg.svd`` is intentionally used only by this oracle generator:
    it computes the true largest singular value for every frequency sample in
    the pinned NumPy environment.  The raw minimum is validated strictly
    against the active bound, then rounded to twelve decimal places before it
    is recorded so equivalent LAPACK backends produce the same contract. Rust
    tests independently certify the same strict lower bound with a matrix
    column norm, so the fixture does not turn SVD into a runtime dependency or
    a production classifier.
    """

    if s.ndim != 3 or s.shape[1] != s.shape[2] or s.shape[1] < 2:
        raise ValueError("active-network evidence requires a multiport S stack")
    singular_values: list[float] = []
    for frequency in range(s.shape[0]):
        value = float(np.linalg.svd(s[frequency], compute_uv=False)[0])
        if not np.isfinite(value):
            raise ValueError(
                f"active-network singular-value evidence is non-finite at frequency {frequency}"
            )
        singular_values.append(value)

    raw_observed_minimum = min(singular_values)
    if raw_observed_minimum <= ACTIVE_REQUIRED_SIGMA_MAX:
        raise ValueError(
            "active-network S input/output must exceed the required singular-value "
            f"bound {ACTIVE_REQUIRED_SIGMA_MAX}; observed {raw_observed_minimum}"
        )
    observed_minimum = round(
        raw_observed_minimum,
        ACTIVE_OBSERVED_MINIMUM_DECIMAL_PLACES,
    )
    if observed_minimum <= ACTIVE_REQUIRED_SIGMA_MAX:
        raise ValueError(
            "rounded active-network evidence must remain above the required "
            f"bound {ACTIVE_REQUIRED_SIGMA_MAX}; observed {observed_minimum}"
        )

    return {
        "criterion": ACTIVE_NETWORK_CRITERION,
        "matrix_field": matrix_field,
        "observed_minimum": observed_minimum,
        "required_minimum": ACTIVE_REQUIRED_SIGMA_MAX,
    }


def _passive_network_metadata(
    np: Any,
    matrices: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Return pinned NumPy SVD evidence for each relevant passive S stack.

    The oracle is the only place where the true largest singular value is
    computed.  Every frequency sample is checked before taking the raw maximum
    for a matrix field; the rounded value is then recorded with twelve decimal
    places so equivalent LAPACK backends share one deterministic fixture
    contract.  Rust tests independently certify the same strict bound with a
    Frobenius norm, which is an upper bound on the largest singular value and
    does not require an SVD dependency.
    """

    if not matrices:
        raise ValueError("passive-network evidence requires at least one S stack")

    evidence: list[dict[str, Any]] = []
    seen_fields: set[str] = set()
    allowed_fields = {"s", "s_input", "s_renormalized"}
    for matrix_field, s in matrices:
        if matrix_field not in allowed_fields:
            raise ValueError(f"unsupported passive-network matrix field: {matrix_field!r}")
        if matrix_field in seen_fields:
            raise ValueError(f"duplicate passive-network matrix field: {matrix_field!r}")
        seen_fields.add(matrix_field)
        if s.ndim != 3 or s.shape[1] != s.shape[2] or s.shape[1] < 1:
            raise ValueError(
                "passive-network evidence requires a stack of square S matrices"
            )

        singular_values: list[float] = []
        for frequency in range(s.shape[0]):
            value = float(np.linalg.svd(s[frequency], compute_uv=False)[0])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "passive-network singular-value evidence must be finite and "
                    f"non-negative at frequency {frequency} for {matrix_field!r}"
                )
            singular_values.append(value)

        raw_observed_maximum = max(singular_values)
        if raw_observed_maximum >= PASSIVE_REQUIRED_SIGMA_MAX:
            raise ValueError(
                "passive-network S matrices must remain strictly below the "
                f"required bound {PASSIVE_REQUIRED_SIGMA_MAX}; "
                f"{matrix_field!r} observed {raw_observed_maximum}"
            )
        observed_maximum = round(
            raw_observed_maximum,
            PASSIVE_OBSERVED_MAXIMUM_DECIMAL_PLACES,
        )
        if observed_maximum >= PASSIVE_REQUIRED_SIGMA_MAX:
            raise ValueError(
                "rounded passive-network evidence must remain strictly below "
                f"the required bound {PASSIVE_REQUIRED_SIGMA_MAX}; "
                f"{matrix_field!r} observed {observed_maximum}"
            )
        evidence.append(
            {
                "matrix_field": matrix_field,
                "observed_maximum": observed_maximum,
            }
        )

    return {
        "criterion": PASSIVE_NETWORK_CRITERION,
        "required_maximum": PASSIVE_REQUIRED_SIGMA_MAX,
        "matrices": evidence,
    }


def _active_s_inputs(
    np: Any,
    *,
    seed: int,
    z0_ohm: float = ACTIVE_S_TO_Z_Z0_OHM,
) -> tuple[Any, Any, float, Any]:
    """Build a deterministic, non-symmetric, deliberately active S stack."""

    nfreq = 3
    nports = 3
    frequency_hz = np.array(
        [0.87e9 + 0.43e9 * index for index in range(nfreq)], dtype=np.float64
    )
    expanded_z0 = np.full((nfreq, nports), z0_ohm, dtype=np.complex128)
    rng = np.random.default_rng(seed)
    s = (
        rng.normal(loc=0.0, scale=0.025, size=(nfreq, nports, nports))
        + 1j * rng.normal(loc=0.0, scale=0.025, size=(nfreq, nports, nports))
    ).astype(np.complex128)
    for frequency in range(nfreq):
        for port in range(nports):
            # These diagonal values are deliberately well above one while
            # keeping I-S comfortably diagonally dominant for the existing
            # exact-pivot conversion solver.
            s[frequency, port, port] += complex(
                1.72 + 0.06 * frequency + 0.025 * port,
                0.035 + 0.006 * frequency - 0.004 * port,
            )

    _assert_non_symmetric(np, s, name="active S input")
    _assert_s_conditioning(s)
    return frequency_hz, s, z0_ohm, expanded_z0


def _active_z_inputs(
    np: Any,
    *,
    seed: int,
    z0_ohm: float = ACTIVE_S_TO_Z_Z0_OHM,
) -> tuple[Any, Any, float, Any]:
    """Build a direct deterministic Z stack whose power-wave S is active."""

    nfreq = 3
    nports = 3
    frequency_hz = np.array(
        [0.87e9 + 0.43e9 * index for index in range(nfreq)], dtype=np.float64
    )
    expanded_z0 = np.full((nfreq, nports), z0_ohm, dtype=np.complex128)
    rng = np.random.default_rng(seed)
    z = (
        rng.normal(loc=0.0, scale=0.35, size=(nfreq, nports, nports))
        + 1j * rng.normal(loc=0.0, scale=0.35, size=(nfreq, nports, nports))
    ).astype(np.complex128)
    for frequency in range(nfreq):
        for port in range(nports):
            # Negative-resistance diagonal terms are far from -z0.  The
            # modest non-symmetric coupling keeps this a genuine N-port input.
            z[frequency, port, port] += complex(
                -(2.4 + 0.08 * frequency + 0.04 * port) * z0_ohm,
                1.5 + 0.2 * frequency - 0.15 * port,
            )

    _assert_non_symmetric(np, z, name="active Z input")
    _assert_z_conditioning(z, expanded_z0)
    return frequency_hz, z, z0_ohm, expanded_z0


def _matrix_frequency_and_z0(
    np: Any,
    *,
    nfreq: int,
    nports: int,
    z0_profile: str,
) -> tuple[Any, Any, Any]:
    """Return frequency samples, constructor z0, and expanded z0 values.

    ``constructor_z0`` intentionally preserves scalar and per-port forms where
    those forms are part of the profile.  ``expanded_z0`` is the normalized
    frequency-major array used for validation and direct-Z generation.
    """

    frequency_hz = np.array(
        [0.85e9 + 0.37e9 * index for index in range(nfreq)], dtype=np.float64
    )
    frequency_index = np.arange(nfreq, dtype=np.float64)[:, None]
    port_index = np.arange(nports, dtype=np.float64)[None, :]

    if z0_profile == "real_scalar":
        scalar = 43.75
        expanded_z0 = np.full((nfreq, nports), scalar, dtype=np.complex128)
        return frequency_hz, scalar, expanded_z0

    if z0_profile == "complex_per_port_constant":
        values = (
            41.0
            + 4.5 * np.arange(nports, dtype=np.float64)
            + 1j * (1.75 - 0.3 * np.arange(nports, dtype=np.float64))
        ).astype(np.complex128)
        expanded_z0 = np.broadcast_to(values[None, :], (nfreq, nports)).copy()
        return frequency_hz, values, expanded_z0

    if z0_profile == "real_frequency_dependent":
        values = 46.5 + 2.75 * frequency_index
        constructor_z0 = np.broadcast_to(values, (nfreq, nports)).copy()
        expanded_z0 = constructor_z0.astype(np.complex128)
        return frequency_hz, constructor_z0, expanded_z0

    if z0_profile == "complex_per_port_frequency_dependent":
        expanded_z0 = (
            39.0
            + 3.75 * port_index
            + 1.9 * frequency_index
            + 1j
            * (
                0.8
                + 0.22 * port_index
                + 0.17 * frequency_index
            )
        ).astype(np.complex128)
        return frequency_hz, expanded_z0, expanded_z0

    raise ValueError(f"unknown z0 profile: {z0_profile!r}")


def _matrix_s_inputs(
    np: Any,
    *,
    nfreq: int,
    nports: int,
    seed: int,
    z0_profile: str,
) -> tuple[Any, Any, Any, Any]:
    """Build deterministic S, z0, and frequency arrays for a matrix case."""

    frequency_hz, constructor_z0, expanded_z0 = _matrix_frequency_and_z0(
        np,
        nfreq=nfreq,
        nports=nports,
        z0_profile=z0_profile,
    )
    rng = np.random.default_rng(seed)
    scale = 0.018 if nports >= 8 else 0.028
    s = (
        rng.normal(loc=0.0, scale=scale, size=(nfreq, nports, nports))
        + 1j * rng.normal(loc=0.0, scale=scale, size=(nfreq, nports, nports))
    ).astype(np.complex128)
    _assert_non_symmetric(np, s, name="S input")
    _assert_s_conditioning(s)
    return frequency_hz, s, constructor_z0, expanded_z0


def _assert_renormalization_z0(
    np: Any,
    source_z0: Any,
    target_z0: Any,
    *,
    expected_shape: tuple[int, int],
    z0_profile: str,
) -> None:
    """Validate the explicit source/target z0 contract for renormalization."""

    if source_z0.shape != expected_shape or target_z0.shape != expected_shape:
        raise ValueError(
            "renormalization z0 arrays must have shape "
            f"{expected_shape}; got {source_z0.shape} and {target_z0.shape}"
        )

    profile = _matrix_reference_impedance_flags(z0_profile)
    for name, z0 in (("source", source_z0), ("target", target_z0)):
        if not np.isfinite(z0).all():
            raise ValueError(f"renormalization {name} z0 must be finite")
        if not (z0.real > 0.0).all():
            raise ValueError(
                f"renormalization {name} z0 must have positive real parts"
            )

        has_imaginary = (z0.imag != 0.0).any()
        if has_imaginary != profile["complex"]:
            raise ValueError(
                f"renormalization {name} z0 complex flag does not match data"
            )
        if profile["complex"] and (z0.imag == 0.0).any():
            raise ValueError(
                f"renormalization {name} z0 must have non-zero imaginary parts"
            )

        rows_differ = not np.all(z0[1:] == z0[0])
        if rows_differ != profile["frequency_dependent"]:
            raise ValueError(
                f"renormalization {name} z0 frequency-dependence flag does not "
                "match data"
            )

        ports_differ = not np.all(z0[:, 1:] == z0[:, :1])
        if ports_differ != profile["per_port"]:
            raise ValueError(
                f"renormalization {name} z0 per-port flag does not match data"
            )

    # Keep the two reference-impedance sets materially separated at every
    # frequency/port so this is a genuine renormalization rather than an
    # identity or near-identity exercise.
    if (np.abs(source_z0 - target_z0) <= 1.0).any():
        raise ValueError(
            "renormalization source and target z0 must differ materially "
            "at every frequency/port"
        )


def _renormalize_inputs(
    np: Any,
    *,
    nfreq: int,
    nports: int,
    seed: int,
    z0_profile: str,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Build one deterministic S/source-z0/target-z0 renormalization case.

    The first four returned values are the frequency, S matrix, and the
    constructor values for source and target ``z0``.  The last two values are
    their normalized frequency-major arrays used by the generation-time
    contract checks.  Keeping the constructor values separate lets the scalar
    and per-port profiles exercise the corresponding public Network input
    forms while retaining one stable fixture shape after read-back.
    """

    if nfreq == 3:
        # Keep the original four-port fixture's frequency samples byte-for-byte
        # stable while using the same deterministic samples for the 8-port
        # profile.  Other cases use the same progression with their own count.
        frequency_hz = np.array([0.91e9, 1.37e9, 2.11e9], dtype=np.float64)
    else:
        frequency_hz = np.array(
            [0.91e9 + 0.46e9 * index for index in range(nfreq)],
            dtype=np.float64,
        )

    # Use a local Generator so each case's recorded seed is independent of
    # every other fixture and does not mutate NumPy's process-global RNG.
    rng = np.random.default_rng(seed)
    scale = 0.018 if nports >= 8 else 0.032
    s = (
        rng.normal(loc=0.0, scale=scale, size=(nfreq, nports, nports))
        + 1j * rng.normal(loc=0.0, scale=scale, size=(nfreq, nports, nports))
    ).astype(np.complex128)
    _assert_non_symmetric(np, s, name="renormalization S input")
    _assert_s_conditioning(s)

    frequency_index = np.arange(nfreq, dtype=np.float64)[:, None]
    port_index = np.arange(nports, dtype=np.float64)[None, :]
    if z0_profile == "real_scalar":
        source_constructor_z0 = 47.25
        target_constructor_z0 = 86.5
        source_z0 = np.full((nfreq, nports), source_constructor_z0, dtype=np.complex128)
        target_z0 = np.full((nfreq, nports), target_constructor_z0, dtype=np.complex128)
    elif z0_profile == "complex_per_port_constant":
        source_constructor_z0 = (
            42.0
            + 4.5 * np.arange(nports, dtype=np.float64)
            + 1j * (1.2 + 0.2 * np.arange(nports, dtype=np.float64))
        ).astype(np.complex128)
        target_constructor_z0 = (
            65.0
            + 3.0 * np.arange(nports, dtype=np.float64)
            + 1j * (-2.5 + 0.4 * np.arange(nports, dtype=np.float64))
        ).astype(np.complex128)
        source_z0 = np.broadcast_to(
            source_constructor_z0[None, :], (nfreq, nports)
        ).copy()
        target_z0 = np.broadcast_to(
            target_constructor_z0[None, :], (nfreq, nports)
        ).copy()
    elif z0_profile == "real_frequency_dependent":
        source_frequency_z0 = 44.5 + 2.2 * frequency_index[:, 0]
        target_frequency_z0 = 72.0 + 2.7 * frequency_index[:, 0]
        source_constructor_z0 = np.broadcast_to(
            source_frequency_z0[:, None], (nfreq, nports)
        ).copy()
        target_constructor_z0 = np.broadcast_to(
            target_frequency_z0[:, None], (nfreq, nports)
        ).copy()
        source_z0 = source_constructor_z0.astype(np.complex128)
        target_z0 = target_constructor_z0.astype(np.complex128)
    elif z0_profile == "complex_per_port_frequency_dependent":
        if (nfreq, nports) == (3, 4):
            # Preserve the original Issue #20 fixture's source and target
            # arrays exactly; this remains the canonical 4-port case.
            source_z0 = np.array(
                [
                    [42.0 + 1.25j, 49.5 - 2.0j, 63.0 + 3.25j, 78.0 - 1.5j],
                    [44.0 + 1.6j, 52.0 - 1.7j, 66.5 + 3.6j, 80.5 - 1.1j],
                    [46.0 + 1.95j, 54.5 - 1.4j, 70.0 + 3.95j, 83.0 - 0.7j],
                ],
                dtype=np.complex128,
            )
            target_z0 = np.array(
                [
                    [58.5 - 2.75j, 43.0 + 1.45j, 72.5 - 3.8j, 91.0 + 2.25j],
                    [61.0 - 2.35j, 46.0 + 1.85j, 76.0 - 3.35j, 95.0 + 2.7j],
                    [63.5 - 1.95j, 49.0 + 2.25j, 79.5 - 2.9j, 99.0 + 3.15j],
                ],
                dtype=np.complex128,
            )
        else:
            source_z0 = (
                40.0
                + 3.25 * port_index
                + 1.5 * frequency_index
                + 1j * (1.0 + 0.2 * port_index + 0.1 * frequency_index)
            ).astype(np.complex128)
            target_z0 = (
                68.0
                + 2.25 * port_index
                + 1.7 * frequency_index
                + 1j * (-2.0 + 0.15 * port_index + 0.2 * frequency_index)
            ).astype(np.complex128)
        source_constructor_z0 = source_z0
        target_constructor_z0 = target_z0
    else:
        raise ValueError(f"unknown z0 profile: {z0_profile!r}")

    _assert_renormalization_z0(
        np,
        source_z0,
        target_z0,
        expected_shape=(nfreq, nports),
        z0_profile=z0_profile,
    )
    return (
        frequency_hz,
        s,
        source_constructor_z0,
        target_constructor_z0,
        source_z0,
        target_z0,
    )


def _matrix_z_inputs(
    np: Any,
    *,
    nfreq: int,
    nports: int,
    seed: int,
    z0_profile: str,
) -> tuple[Any, Any, Any, Any]:
    """Build deterministic diagonally dominant direct-Z operation inputs."""

    frequency_hz, constructor_z0, expanded_z0 = _matrix_frequency_and_z0(
        np,
        nfreq=nfreq,
        nports=nports,
        z0_profile=z0_profile,
    )
    rng = np.random.default_rng(seed)
    scale = 0.18 if nports >= 8 else 0.25
    z = (
        rng.normal(loc=0.0, scale=scale, size=(nfreq, nports, nports))
        + 1j * rng.normal(loc=0.0, scale=scale, size=(nfreq, nports, nports))
    ).astype(np.complex128)
    for frequency in range(nfreq):
        for port in range(nports):
            z[frequency, port, port] += (
                68.0
                + 2.5 * frequency
                + 1.8 * port
                + 1j * (2.0 + 0.13 * frequency - 0.08 * port)
            )
    _assert_non_symmetric(np, z, name="Z input")
    _assert_z_conditioning(z, expanded_z0)
    return frequency_hz, z, constructor_z0, expanded_z0


def _assert_parameter_diagonal_dominance(
    np: Any,
    matrix: Any,
    *,
    name: str,
    margin: float,
) -> None:
    """Keep a well-conditioned direct parameter input away from singularity.

    This is a generation-time construction guard only.  It is intentionally a
    simple strict row diagonal-dominance check rather than a runtime policy or
    a recorded condition-number acceptance value.
    """

    if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
        raise ValueError(f"{name} must be a stack of square matrices")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    for frequency in range(matrix.shape[0]):
        for row in range(matrix.shape[1]):
            diagonal = abs(matrix[frequency, row, row])
            off_diagonal = sum(
                abs(matrix[frequency, row, column])
                for column in range(matrix.shape[2])
                if column != row
            )
            if diagonal <= off_diagonal + margin:
                raise ValueError(
                    f"{name} failed the strict diagonal-dominance guard at "
                    f"frequency {frequency}, row {row}"
                )


def _impedance_admittance_frequency(np: Any) -> Any:
    """Build deterministic frequency samples for direct Z/Y fixtures."""

    return np.array(
        [0.73e9 + 0.37e9 * index for index in range(IMPEDANCE_ADMITTANCE_NFREQ)],
        dtype=np.float64,
    )


def _impedance_admittance_well_inputs(
    np: Any,
    *,
    quantity: str,
    seed: int,
) -> tuple[Any, Any]:
    """Construct an independent, non-symmetric well-conditioned Z or Y stack."""

    if quantity not in {"z", "y"}:
        raise ValueError(f"unknown impedance/admittance quantity: {quantity!r}")

    frequency_hz = _impedance_admittance_frequency(np)
    nfreq = frequency_hz.size
    nports = IMPEDANCE_ADMITTANCE_NPORTS
    rng = np.random.default_rng(seed)
    if quantity == "z":
        scale = 0.35
        diagonal_base = 18.0
        diagonal_step = 2.5
        name = "well-conditioned Z input"
    else:
        scale = 0.0035
        diagonal_base = 0.095
        diagonal_step = 0.011
        name = "well-conditioned Y input"

    matrix = (
        rng.normal(loc=0.0, scale=scale, size=(nfreq, nports, nports))
        + 1j * rng.normal(loc=0.0, scale=scale, size=(nfreq, nports, nports))
    ).astype(np.complex128)
    for frequency in range(nfreq):
        for port in range(nports):
            matrix[frequency, port, port] += complex(
                diagonal_base + diagonal_step * frequency + 0.8 * port,
                0.75 + 0.12 * frequency - 0.09 * port,
            )

    _assert_non_symmetric(np, matrix, name=name)
    _assert_parameter_diagonal_dominance(
        np,
        matrix,
        name=name,
        margin=0.25 * diagonal_base,
    )
    for frequency in range(nfreq):
        if np.linalg.matrix_rank(matrix[frequency]) != nports:
            raise ValueError(f"{name} must have full pinned-NumPy matrix rank")
    return frequency_hz, matrix


def _impedance_admittance_near_inputs(
    np: Any,
    *,
    quantity: str,
    seed: int,
) -> tuple[Any, Any]:
    """Construct an upper-triangular near-singular direct Z or Y stack.

    The binary diagonal factor and determinant product are generated from the
    input itself.  Full rank is checked with the pinned NumPy matrix-rank
    implementation at generation time; no condition number is recorded or
    used as a fixture acceptance criterion.
    """

    if quantity not in {"z", "y"}:
        raise ValueError(f"unknown impedance/admittance quantity: {quantity!r}")

    frequency_hz = _impedance_admittance_frequency(np)
    nfreq = frequency_hz.size
    nports = IMPEDANCE_ADMITTANCE_NPORTS
    rng = np.random.default_rng(seed)
    matrix = np.zeros((nfreq, nports, nports), dtype=np.complex128)
    for frequency in range(nfreq):
        for row, factor in enumerate(IMPEDANCE_ADMITTANCE_NEAR_DIAGONAL_FACTORS):
            matrix[frequency, row, row] = complex(factor, 0.0)
            for column in range(row + 1, nports):
                matrix[frequency, row, column] = complex(
                    rng.normal(loc=0.0, scale=0.03125),
                    rng.normal(loc=0.0, scale=0.03125),
                )

    name = f"near-singular {quantity.upper()} input"
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    lower_triangle = np.tril(matrix, k=-1)
    if not np.array_equal(lower_triangle, np.zeros_like(lower_triangle)):
        raise ValueError(f"{name} must be upper triangular")
    expected_diagonal = np.asarray(
        IMPEDANCE_ADMITTANCE_NEAR_DIAGONAL_FACTORS,
        dtype=np.complex128,
    )
    if not np.array_equal(
        np.diagonal(matrix, axis1=1, axis2=2),
        np.broadcast_to(expected_diagonal, (nfreq, nports)),
    ):
        raise ValueError(f"{name} has unexpected binary diagonal factors")
    for frequency in range(nfreq):
        if np.linalg.matrix_rank(matrix[frequency]) != nports:
            raise ValueError(f"{name} must have full pinned-NumPy matrix rank")
    return frequency_hz, matrix


def _impedance_admittance_near_metadata(*, system_matrix: str) -> dict[str, Any]:
    """Describe the input-derived nonsingular near-singular construction."""

    factors = [float(value) for value in IMPEDANCE_ADMITTANCE_NEAR_DIAGONAL_FACTORS]
    determinant = float(IMPEDANCE_ADMITTANCE_NEAR_DETERMINANT)
    if not all(math.isfinite(value) and value != 0.0 for value in factors):
        raise ValueError("near-singular diagonal factors must be finite and non-zero")
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("near-singular determinant must be finite and non-zero")
    return {
        "binary_exponent": IMPEDANCE_ADMITTANCE_NEAR_BINARY_EXPONENT,
        "determinant": determinant,
        "determinant_factors": factors,
        "determinant_factors_nonzero": True,
        "matrix_structure": "upper_triangular",
        "small_diagonal_port": 0,
        "small_diagonal_value": float(IMPEDANCE_ADMITTANCE_NEAR_DIAGONAL_VALUE),
        "system_matrix": system_matrix,
    }


def _impedance_admittance_fixture(
    np: Any,
    skrf: Any,
    *,
    case_id: str,
    operation: str,
    seed: int,
    near_singular: bool,
) -> dict[str, Any]:
    """Build one direct Z↔Y fixture from public scikit-rf conversion behavior."""

    if operation == "z_to_y":
        input_quantity = "z"
        input_key = "z_ohm"
        output_key = "y_s"
        input_unit = "ohm"
        output_unit = "S"
        absolute_tolerance_key = "atol_s"
        absolute_tolerance = IMPEDANCE_ADMITTANCE_ATOL_S
        if near_singular:
            frequency_hz, direct_input = _impedance_admittance_near_inputs(
                np,
                quantity=input_quantity,
                seed=seed,
            )
        else:
            frequency_hz, direct_input = _impedance_admittance_well_inputs(
                np,
                quantity=input_quantity,
                seed=seed,
            )
        expected_output = skrf.network.z2y(direct_input)
    elif operation == "y_to_z":
        input_quantity = "y"
        input_key = "y_s"
        output_key = "z_ohm"
        input_unit = "S"
        output_unit = "ohm"
        absolute_tolerance_key = "atol_ohm"
        absolute_tolerance = IMPEDANCE_ADMITTANCE_ATOL_OHM
        if near_singular:
            frequency_hz, direct_input = _impedance_admittance_near_inputs(
                np,
                quantity=input_quantity,
                seed=seed,
            )
        else:
            frequency_hz, direct_input = _impedance_admittance_well_inputs(
                np,
                quantity=input_quantity,
                seed=seed,
            )
        expected_output = skrf.network.y2z(direct_input)
    else:
        raise ValueError(f"unknown impedance/admittance operation: {operation!r}")

    expected_output = np.asarray(expected_output, dtype=np.complex128)
    if not np.isfinite(expected_output).all():
        raise ValueError(f"{case_id} output must be finite")
    if near_singular:
        near_metadata = _impedance_admittance_near_metadata(
            system_matrix=input_quantity.upper()
        )
        justification = IMPEDANCE_ADMITTANCE_NEAR_TOLERANCE_JUSTIFICATION
    else:
        near_metadata = None
        justification = (
            "Strict binary64 tolerance for a well-conditioned, modest-magnitude "
            "deterministic direct complex N-port parameter matrix; a generation-"
            "time diagonal-dominance guard keeps the input away from exact "
            "singularity while allowing normal cross-language solve round-off."
        )

    input_shape_key = f"input_{input_quantity}"
    output_shape_key = f"output_{'y' if operation == 'z_to_y' else 'z'}"
    data = {
        "frequency_hz": [float(value) for value in frequency_hz],
        input_key: _complex_array(direct_input),
        output_key: _complex_array(expected_output),
    }
    metadata = {
        "case_id": case_id,
        "input_unit": input_unit,
        "numpy_version": np.__version__,
        "operation": operation,
        "output_unit": output_unit,
        "random_seed": seed,
        "schema": "rfkit-rs.oracle.fixture",
        "schema_version": SCHEMA_VERSION,
        "scikit_rf_version": skrf.__version__,
        "shape": {
            "frequency": list(frequency_hz.shape),
            input_shape_key: list(direct_input.shape),
            output_shape_key: list(expected_output.shape),
        },
        "tolerance_policy": {
            absolute_tolerance_key: absolute_tolerance,
            "comparison": (
                f"abs(actual-expected) <= {absolute_tolerance_key} + "
                "rtol*abs(expected)"
            ),
            "justification": justification,
            "regeneration": (
                f"canonical UTF-8 JSON serialization; {output_key} is checked "
                "with the recorded numeric tolerance"
            ),
            "rtol": IMPEDANCE_ADMITTANCE_RTOL,
        },
        "wave_definition": "not_applicable",
    }
    if near_metadata is not None:
        metadata["near_singular"] = near_metadata

    return {"metadata": metadata, "data": data}


def _reciprocal_frequency_and_z0(
    np: Any,
    *,
    nfreq: int,
    nports: int,
    z0_ohm: float,
) -> tuple[Any, float, Any]:
    """Build the shared frequency and scalar real reference impedance data."""

    frequency_hz = np.array(
        [0.95e9 + 0.41e9 * index for index in range(nfreq)], dtype=np.float64
    )
    expanded_z0 = np.full((nfreq, nports), z0_ohm, dtype=np.complex128)
    _assert_real_positive_equal_z0(np, expanded_z0, name="reciprocal")
    return frequency_hz, z0_ohm, expanded_z0


def _reciprocal_s_inputs(
    np: Any,
    *,
    nfreq: int,
    nports: int,
    seed: int,
    z0_ohm: float,
) -> tuple[Any, Any, float, Any]:
    """Build a deterministic complex S stack by mirroring one triangle.

    The assignment to both matrix locations is deliberately a plain copy, not
    a conjugate.  This records transpose reciprocity rather than Hermitian
    symmetry and keeps the fixture independent of any conversion formula.
    """

    frequency_hz, constructor_z0, expanded_z0 = _reciprocal_frequency_and_z0(
        np,
        nfreq=nfreq,
        nports=nports,
        z0_ohm=z0_ohm,
    )
    rng = np.random.default_rng(seed)
    s = np.empty((nfreq, nports, nports), dtype=np.complex128)
    for frequency in range(nfreq):
        for row in range(nports):
            for column in range(row, nports):
                value = complex(
                    rng.normal(loc=0.0, scale=0.04),
                    rng.normal(loc=0.0, scale=0.04),
                )
                s[frequency, row, column] = value
                s[frequency, column, row] = value

    _assert_exact_symmetric(np, s, name="reciprocal S input")
    _assert_s_conditioning(s)
    return frequency_hz, s, constructor_z0, expanded_z0


def _reciprocal_z_inputs(
    np: Any,
    *,
    nfreq: int,
    nports: int,
    seed: int,
    z0_ohm: float,
) -> tuple[Any, Any, float, Any]:
    """Build a deterministic complex Z stack by mirroring one triangle."""

    frequency_hz, constructor_z0, expanded_z0 = _reciprocal_frequency_and_z0(
        np,
        nfreq=nfreq,
        nports=nports,
        z0_ohm=z0_ohm,
    )
    rng = np.random.default_rng(seed)
    z = np.empty((nfreq, nports, nports), dtype=np.complex128)
    for frequency in range(nfreq):
        for row in range(nports):
            for column in range(row, nports):
                value = complex(
                    rng.normal(loc=0.0, scale=0.35),
                    rng.normal(loc=0.0, scale=0.35),
                )
                if row == column:
                    value += complex(
                        73.0 + 2.75 * frequency + 3.5 * row,
                        2.5 + 0.2 * frequency - 0.15 * row,
                    )
                z[frequency, row, column] = value
                z[frequency, column, row] = value

    _assert_exact_symmetric(np, z, name="reciprocal Z input")
    _assert_z_conditioning(z, expanded_z0)
    return frequency_hz, z, constructor_z0, expanded_z0


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


def _interpolation_inputs(np: Any) -> tuple[Any, Any, Any, Any]:
    """Build direct irregular-grid S/z0 inputs for interpolation conformance."""

    source_frequency_hz = np.array(
        [0.72e9, 1.11e9, 1.93e9, 2.68e9, 3.47e9],
        dtype=np.float64,
    )
    target_frequency_hz = np.array(
        [0.72e9, 0.91e9, 1.11e9, 1.53e9, 2.305e9, 3.47e9],
        dtype=np.float64,
    )
    nports = 3
    rng = np.random.default_rng(INTERPOLATION_RANDOM_SEED)
    source_s = (
        rng.normal(loc=0.0, scale=0.04, size=(source_frequency_hz.size, nports, nports))
        + 1j
        * rng.normal(
            loc=0.0,
            scale=0.04,
            size=(source_frequency_hz.size, nports, nports),
        )
    ).astype(np.complex128)
    source_z0 = (
        37.5
        + 3.2 * np.arange(nports, dtype=np.float64)[None, :]
        + 1.7 * np.arange(source_frequency_hz.size, dtype=np.float64)[:, None]
        + 1j
        * (
            1.2
            + 0.4 * np.arange(nports, dtype=np.float64)[None, :]
            - 0.17 * np.arange(source_frequency_hz.size, dtype=np.float64)[:, None]
        )
    ).astype(np.complex128)

    _assert_non_symmetric(np, source_s, name="interpolation S input")
    if not np.isfinite(source_frequency_hz).all():
        raise ValueError("interpolation source frequencies must be finite")
    if not np.isfinite(target_frequency_hz).all():
        raise ValueError("interpolation target frequencies must be finite")
    if not np.isfinite(source_z0).all():
        raise ValueError("interpolation z0 input must be finite")
    if np.any(source_z0.imag == 0.0):
        raise ValueError("interpolation z0 input must be complex at every port/frequency")
    return source_frequency_hz, target_frequency_hz, source_s, source_z0


def _interpolation_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build the direct Cartesian interpolation fixture through public scikit-rf."""

    (
        source_frequency_hz,
        target_frequency_hz,
        source_s,
        source_z0,
    ) = _interpolation_inputs(np)
    network = skrf.Network(
        f=source_frequency_hz,
        s=source_s,
        z0=source_z0,
        s_def="power",
        name=INTERPOLATION_CASE_ID,
    )
    input_frequency = np.asarray(network.f, dtype=np.float64)
    input_s = np.asarray(network.s, dtype=np.complex128)
    input_z0 = np.asarray(network.z0, dtype=np.complex128)
    interpolated = network.interpolate(
        target_frequency_hz,
        basis="s",
        coords="cart",
        kind="linear",
    )
    output_frequency = np.asarray(interpolated.f, dtype=np.float64)
    output_s = np.asarray(interpolated.s, dtype=np.complex128)
    output_z0 = np.asarray(interpolated.z0, dtype=np.complex128)
    if not np.array_equal(input_frequency, source_frequency_hz):
        raise ValueError("scikit-rf changed interpolation source frequencies on read-back")
    if not np.array_equal(input_s, source_s):
        raise ValueError("scikit-rf changed interpolation S inputs on read-back")
    if not np.array_equal(input_z0, source_z0):
        raise ValueError("scikit-rf changed interpolation z0 inputs on read-back")
    if not np.array_equal(output_frequency, target_frequency_hz):
        raise ValueError("scikit-rf changed interpolation target frequencies")
    if not np.isfinite(output_s).all() or not np.isfinite(output_z0).all():
        raise ValueError("interpolation output must be finite")

    return {
        "metadata": {
            "basis": "s",
            "case_id": INTERPOLATION_CASE_ID,
            "coords": "cart",
            "input_reference_impedance": {
                "complex": True,
                "frequency_dependent": True,
                "per_port": True,
                "unit": "ohm",
            },
            "kind": "linear",
            "numpy_version": np.__version__,
            "operation": "interpolate_s",
            "random_seed": INTERPOLATION_RANDOM_SEED,
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "scipy_version": version("scipy"),
            "shape": {
                "input_s": list(input_s.shape),
                "input_z0": list(input_z0.shape),
                "source_frequency": list(input_frequency.shape),
                "output_s": list(output_s.shape),
                "output_z0": list(output_z0.shape),
                "target_frequency": list(output_frequency.shape),
            },
            "tolerance_policy": {
                "atol": INTERPOLATION_ATOL,
                "comparison": "abs(actual-expected) <= atol + rtol*abs(expected)",
                "justification": INTERPOLATION_TOLERANCE_JUSTIFICATION,
                "regeneration": (
                    "canonical UTF-8 JSON serialization; s and z0_ohm are checked "
                    "with the recorded mixed numeric tolerance"
                ),
                "rtol": INTERPOLATION_RTOL,
            },
            "wave_definition": network.s_def,
        },
        "data": {
            "s": _complex_array(output_s),
            "s_input": _complex_array(input_s),
            "source_frequency_hz": [float(value) for value in input_frequency],
            "target_frequency_hz": [float(value) for value in output_frequency],
            "z0_input_ohm": _complex_array(input_z0),
            "z0_ohm": _complex_array(output_z0),
        },
    }


def _composition_inputs(
    np: Any,
    *,
    seed_a: int,
    seed_b: int,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Build independently sampled explicit-grid connection inputs."""

    source_frequency_a_hz = np.array(
        [0.73e9, 1.21e9, 1.89e9, 2.67e9, 3.41e9],
        dtype=np.float64,
    )
    source_frequency_b_hz = np.array(
        [0.73e9, 0.96e9, 1.48e9, 2.22e9, 2.93e9, 3.41e9],
        dtype=np.float64,
    )
    target_frequency_hz = np.array(
        [0.73e9, 0.96e9, 1.21e9, 1.73e9, 2.22e9, 2.67e9, 2.93e9, 3.41e9],
        dtype=np.float64,
    )

    def random_s(rng: Any, frequencies: Any, nports: int, diagonal_base: float) -> Any:
        matrix = (
            rng.normal(
                loc=0.0,
                scale=0.018,
                size=(frequencies.size, nports, nports),
            )
            + 1j
            * rng.normal(
                loc=0.0,
                scale=0.018,
                size=(frequencies.size, nports, nports),
            )
        ).astype(np.complex128)
        for frequency in range(frequencies.size):
            for port in range(nports):
                matrix[frequency, port, port] += complex(
                    diagonal_base + 0.009 * frequency + 0.005 * port,
                    0.008 + 0.0015 * frequency - 0.0007 * port,
                )
        return matrix

    rng_a = np.random.default_rng(seed_a)
    rng_b = np.random.default_rng(seed_b)
    source_s_a = random_s(rng_a, source_frequency_a_hz, COMPOSITION_NPORTS_A, 0.11)
    source_s_b = random_s(rng_b, source_frequency_b_hz, COMPOSITION_NPORTS_B, 0.14)
    _assert_non_symmetric(np, source_s_a, name="explicit-grid connection A S input")
    _assert_non_symmetric(np, source_s_b, name="explicit-grid connection B S input")
    _assert_s_conditioning(source_s_a)
    _assert_s_conditioning(source_s_b)

    frequency_index_a = np.arange(source_frequency_a_hz.size, dtype=np.float64)[:, None]
    frequency_index_b = np.arange(source_frequency_b_hz.size, dtype=np.float64)[:, None]
    port_index_a = np.arange(COMPOSITION_NPORTS_A, dtype=np.float64)[None, :]
    port_index_b = np.arange(COMPOSITION_NPORTS_B, dtype=np.float64)[None, :]
    source_z0_a = (
        41.0
        + 2.4 * port_index_a
        + 1.35 * frequency_index_a
        + 1j * (1.5 + 0.37 * port_index_a - 0.11 * frequency_index_a)
    ).astype(np.complex128)
    source_z0_b = (
        83.0
        + 3.1 * port_index_b
        + 1.9 * frequency_index_b
        + 1j * (2.1 + 0.29 * port_index_b + 0.13 * frequency_index_b)
    ).astype(np.complex128)

    # Keep the selected junction profile finite, real, positive, and exactly
    # equal in both inputs.  A constant non-50-ohm value is intentional: it
    # remains bit-for-bit equal after either source grid is interpolated.
    junction_z0 = np.float64(73.5)
    source_z0_a[:, COMPOSITION_PORT_A] = junction_z0
    source_z0_b[:, COMPOSITION_PORT_B] = junction_z0

    if not np.isfinite(source_frequency_a_hz).all() or not np.isfinite(
        source_frequency_b_hz
    ).all():
        raise ValueError("explicit-grid connection source frequencies must be finite")
    if not np.isfinite(target_frequency_hz).all():
        raise ValueError("explicit-grid connection target frequencies must be finite")
    if not np.isfinite(source_s_a).all() or not np.isfinite(source_s_b).all():
        raise ValueError("explicit-grid connection S inputs must be finite")
    if not np.isfinite(source_z0_a).all() or not np.isfinite(source_z0_b).all():
        raise ValueError("explicit-grid connection z0 inputs must be finite")
    if not (source_z0_a[:, COMPOSITION_PORT_A].imag == 0.0).all() or not (
        source_z0_b[:, COMPOSITION_PORT_B].imag == 0.0
    ).all():
        raise ValueError("explicit-grid connection junction z0 must be real")
    if not np.all(source_z0_a[:, COMPOSITION_PORT_A] == junction_z0) or not np.all(
        source_z0_b[:, COMPOSITION_PORT_B] == junction_z0
    ):
        raise ValueError("explicit-grid connection junction z0 must match exactly")
    if not (source_z0_a[:, COMPOSITION_PORT_A].real > 0.0).all():
        raise ValueError("explicit-grid connection junction z0 must be positive")
    if np.array_equal(source_frequency_a_hz, source_frequency_b_hz):
        raise ValueError("explicit-grid connection source grids must differ")
    if np.any(source_z0_a[:, np.arange(COMPOSITION_NPORTS_A) != COMPOSITION_PORT_A].imag == 0.0):
        raise ValueError("explicit-grid connection A external z0 must be complex")
    if np.any(source_z0_b[:, np.arange(COMPOSITION_NPORTS_B) != COMPOSITION_PORT_B].imag == 0.0):
        raise ValueError("explicit-grid connection B external z0 must be complex")

    return (
        source_frequency_a_hz,
        source_frequency_b_hz,
        target_frequency_hz,
        source_s_a,
        source_z0_a,
        source_s_b,
        source_z0_b,
    )


def _composition_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build explicit-grid connection output through public scikit-rf APIs."""

    (
        source_frequency_a_hz,
        source_frequency_b_hz,
        target_frequency_hz,
        source_s_a,
        source_z0_a,
        source_s_b,
        source_z0_b,
    ) = _composition_inputs(
        np,
        seed_a=COMPOSITION_RANDOM_SEED_A,
        seed_b=COMPOSITION_RANDOM_SEED_B,
    )
    case_id = COMPOSITION_CASE_ID
    network_a = skrf.Network(
        f=source_frequency_a_hz,
        s=source_s_a,
        z0=source_z0_a,
        s_def="power",
        name=f"{case_id}_a",
    )
    network_b = skrf.Network(
        f=source_frequency_b_hz,
        s=source_s_b,
        z0=source_z0_b,
        s_def="power",
        name=f"{case_id}_b",
    )

    # Each side is interpolated exactly once through the public API.  The
    # expected output is then produced by one public network.connect call.
    interpolated_a = network_a.interpolate(
        target_frequency_hz,
        basis="s",
        coords="cart",
        kind="linear",
    )
    interpolated_b = network_b.interpolate(
        target_frequency_hz,
        basis="s",
        coords="cart",
        kind="linear",
    )

    interpolated_frequency_a = np.asarray(interpolated_a.f, dtype=np.float64)
    interpolated_frequency_b = np.asarray(interpolated_b.f, dtype=np.float64)
    if not np.array_equal(interpolated_frequency_a, target_frequency_hz):
        raise ValueError("scikit-rf changed explicit-grid A interpolation frequencies")
    if not np.array_equal(interpolated_frequency_b, target_frequency_hz):
        raise ValueError("scikit-rf changed explicit-grid B interpolation frequencies")

    junction_a = np.asarray(interpolated_a.z0, dtype=np.complex128)[:, COMPOSITION_PORT_A]
    junction_b = np.asarray(interpolated_b.z0, dtype=np.complex128)[:, COMPOSITION_PORT_B]
    if not np.isfinite(junction_a).all() or not np.isfinite(junction_b).all():
        raise ValueError("interpolated explicit-grid junction z0 must be finite")
    if not (junction_a.imag == 0.0).all() or not (junction_b.imag == 0.0).all():
        raise ValueError("interpolated explicit-grid junction z0 must be real")
    if not (junction_a.real > 0.0).all() or not (junction_b.real > 0.0).all():
        raise ValueError("interpolated explicit-grid junction z0 must be positive")
    if not np.array_equal(junction_a, junction_b):
        raise ValueError("interpolated explicit-grid junction z0 must match exactly")

    connected = skrf.network.connect(
        interpolated_a,
        COMPOSITION_PORT_A,
        interpolated_b,
        COMPOSITION_PORT_B,
    )

    source_frequency_a = np.asarray(network_a.f, dtype=np.float64)
    source_frequency_b = np.asarray(network_b.f, dtype=np.float64)
    source_s_a_readback = np.asarray(network_a.s, dtype=np.complex128)
    source_z0_a_readback = np.asarray(network_a.z0, dtype=np.complex128)
    source_s_b_readback = np.asarray(network_b.s, dtype=np.complex128)
    source_z0_b_readback = np.asarray(network_b.z0, dtype=np.complex128)
    output_frequency = np.asarray(connected.f, dtype=np.float64)
    connected_s = np.asarray(connected.s, dtype=np.complex128)
    connected_z0 = np.asarray(connected.z0, dtype=np.complex128)
    if not np.array_equal(source_frequency_a, source_frequency_a_hz):
        raise ValueError("scikit-rf changed explicit-grid A source frequencies")
    if not np.array_equal(source_frequency_b, source_frequency_b_hz):
        raise ValueError("scikit-rf changed explicit-grid B source frequencies")
    if not np.array_equal(source_s_a_readback, source_s_a):
        raise ValueError("scikit-rf changed explicit-grid A S inputs")
    if not np.array_equal(source_z0_a_readback, source_z0_a):
        raise ValueError("scikit-rf changed explicit-grid A z0 inputs")
    if not np.array_equal(source_s_b_readback, source_s_b):
        raise ValueError("scikit-rf changed explicit-grid B S inputs")
    if not np.array_equal(source_z0_b_readback, source_z0_b):
        raise ValueError("scikit-rf changed explicit-grid B z0 inputs")
    if not np.array_equal(output_frequency, target_frequency_hz):
        raise ValueError("scikit-rf changed explicit-grid target frequencies")
    if not np.isfinite(connected_s).all() or not np.isfinite(connected_z0).all():
        raise ValueError("explicit-grid connection output must be finite")

    a_survivors = [port for port in range(COMPOSITION_NPORTS_A) if port != COMPOSITION_PORT_A]
    b_survivors = [port for port in range(COMPOSITION_NPORTS_B) if port != COMPOSITION_PORT_B]
    output_order = [
        *({"network": "A", "port": port} for port in a_survivors),
        *({"network": "B", "port": port} for port in b_survivors),
    ]
    shape = {
        "frequency": list(output_frequency.shape),
        "source_frequency_a": list(source_frequency_a.shape),
        "source_frequency_b": list(source_frequency_b.shape),
        "target_frequency": list(output_frequency.shape),
        "input_s_a": list(source_s_a_readback.shape),
        "input_s_b": list(source_s_b_readback.shape),
        "input_z0_a": list(source_z0_a_readback.shape),
        "input_z0_b": list(source_z0_b_readback.shape),
        "output_s": list(connected_s.shape),
        "output_z0": list(connected_z0.shape),
    }
    return {
        "metadata": {
            "case_id": case_id,
            "interpolation": {
                "basis": "s",
                "coords": "cart",
                "kind": "linear",
                "scipy_version": version("scipy"),
            },
            "junction_ports": {"a": COMPOSITION_PORT_A, "b": COMPOSITION_PORT_B},
            "numpy_version": np.__version__,
            "operation": "connect_matched_on_grid",
            "port_order": {
                "a_survivors": a_survivors,
                "b_survivors": b_survivors,
                "description": (
                    "A unconnected ports in original order, followed by B "
                    "unconnected ports in original order"
                ),
                "output": output_order,
            },
            "random_seeds": {
                "a": COMPOSITION_RANDOM_SEED_A,
                "b": COMPOSITION_RANDOM_SEED_B,
            },
            "reference_impedance": {
                "external_complex": True,
                "external_frequency_dependent": True,
                "junction_exactly_equal_after_interpolation": True,
                "junction_frequency_dependent": False,
                "junction_real_strictly_positive": True,
                "unit": "ohm",
            },
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": shape,
            "tolerance_policy": {
                "atol": COMPOSITION_ATOL,
                "comparison": "abs(actual-expected) <= atol + rtol*abs(expected)",
                "justification": COMPOSITION_TOLERANCE_JUSTIFICATION,
                "regeneration": (
                    "canonical UTF-8 JSON serialization; s_connected and "
                    "z0_connected_ohm are checked with the recorded mixed "
                    "numeric tolerance; inputs, grids, ordering, and metadata "
                    "are checked exactly"
                ),
                "rtol": COMPOSITION_RTOL,
            },
            "wave_definition": connected.s_def,
        },
        "data": {
            "frequency_hz": [float(value) for value in output_frequency],
            "source_frequency_a_hz": [float(value) for value in source_frequency_a],
            "source_frequency_b_hz": [float(value) for value in source_frequency_b],
            "target_frequency_hz": [float(value) for value in output_frequency],
            "s_a": _complex_array(source_s_a_readback),
            "s_b": _complex_array(source_s_b_readback),
            "s_connected": _complex_array(connected_s),
            "z0_a_ohm": _complex_array(source_z0_a_readback),
            "z0_b_ohm": _complex_array(source_z0_b_readback),
            "z0_connected_ohm": _complex_array(connected_z0),
        },
    }


def _connect_matched_inputs(
    np: Any,
    *,
    seed_a: int,
    seed_b: int,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Build independent finite S/z0 inputs for the matched connection case."""

    frequency_hz = np.array(
        [0.79e9, 1.23e9, 1.91e9],
        dtype=np.float64,
    )
    rng_a = np.random.default_rng(seed_a)
    rng_b = np.random.default_rng(seed_b)

    def random_s(rng: Any, nports: int, *, diagonal_base: float) -> Any:
        scale = 0.021
        matrix = (
            rng.normal(
                loc=0.0,
                scale=scale,
                size=(CONNECT_NFREQ, nports, nports),
            )
            + 1j
            * rng.normal(
                loc=0.0,
                scale=scale,
                size=(CONNECT_NFREQ, nports, nports),
            )
        ).astype(np.complex128)
        for frequency in range(CONNECT_NFREQ):
            for port in range(nports):
                matrix[frequency, port, port] += complex(
                    diagonal_base
                    + 0.013 * frequency
                    + 0.006 * port,
                    0.011 + 0.002 * frequency - 0.001 * port,
                )
        return matrix

    s_a = random_s(rng_a, CONNECT_NPORTS_A, diagonal_base=0.115)
    s_b = random_s(rng_b, CONNECT_NPORTS_B, diagonal_base=0.145)
    _assert_non_symmetric(np, s_a, name="matched-connection A S input")
    _assert_non_symmetric(np, s_b, name="matched-connection B S input")
    _assert_s_conditioning(s_a)
    _assert_s_conditioning(s_b)

    # The junction value is frequency-dependent, finite, real, positive, and
    # copied exactly into both input arrays.  Every surviving port gets a
    # distinct real value that also varies by frequency, making ordering and
    # z0 propagation visible in the fixture rather than metadata alone.
    junction_z0 = np.array([61.25, 73.5, 89.75], dtype=np.float64)
    frequency_index_a = np.arange(CONNECT_NFREQ, dtype=np.float64)[:, None]
    frequency_index_b = np.arange(CONNECT_NFREQ, dtype=np.float64)[:, None]
    port_index_a = np.arange(CONNECT_NPORTS_A, dtype=np.float64)[None, :]
    port_index_b = np.arange(CONNECT_NPORTS_B, dtype=np.float64)[None, :]
    z0_a = (
        42.0
        + 2.35 * port_index_a
        + 1.65 * frequency_index_a
    ).astype(np.complex128)
    z0_b = (
        86.0
        + 3.15 * port_index_b
        + 2.25 * frequency_index_b
    ).astype(np.complex128)
    z0_a[:, CONNECT_PORT_A] = junction_z0
    z0_b[:, CONNECT_PORT_B] = junction_z0

    if not np.isfinite(frequency_hz).all():
        raise ValueError("matched-connection frequencies must be finite")
    if not np.isfinite(s_a).all() or not np.isfinite(s_b).all():
        raise ValueError("matched-connection S inputs must be finite")
    if not np.isfinite(z0_a).all() or not np.isfinite(z0_b).all():
        raise ValueError("matched-connection z0 inputs must be finite")
    if not (z0_a[:, CONNECT_PORT_A].imag == 0.0).all() or not (
        z0_b[:, CONNECT_PORT_B].imag == 0.0
    ).all():
        raise ValueError("matched-connection junction z0 must be real")
    if not (z0_a[:, CONNECT_PORT_A] == z0_b[:, CONNECT_PORT_B]).all():
        raise ValueError("matched-connection junction z0 must match exactly")
    if not (z0_a[:, CONNECT_PORT_A].real > 0.0).all():
        raise ValueError("matched-connection junction z0 must be positive")
    if np.array_equal(z0_a[0], z0_a[1]) or np.array_equal(z0_b[0], z0_b[1]):
        raise ValueError("matched-connection z0 must vary by frequency")
    if np.array_equal(z0_a[:, 0], z0_a[:, 2]) or np.array_equal(
        z0_b[:, 0], z0_b[:, 1]
    ):
        raise ValueError("matched-connection external z0 must vary by port")

    return frequency_hz, s_a, z0_a, s_b, z0_b, junction_z0


def _connect_matched_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build the matched-junction fixture through public scikit-rf connect."""

    (
        frequency_hz,
        source_s_a,
        source_z0_a,
        source_s_b,
        source_z0_b,
        _junction_z0,
    ) = _connect_matched_inputs(
        np,
        seed_a=CONNECT_RANDOM_SEED_A,
        seed_b=CONNECT_RANDOM_SEED_B,
    )
    case_id = CONNECT_CASE_ID
    network_a = skrf.Network(
        f=frequency_hz,
        s=source_s_a,
        z0=source_z0_a,
        s_def="power",
        name=f"{case_id}_a",
    )
    network_b = skrf.Network(
        f=frequency_hz,
        s=source_s_b,
        z0=source_z0_b,
        s_def="power",
        name=f"{case_id}_b",
    )

    # scikit-rf 2.0.1 exposes this operation as skrf.network.connect.  Keep
    # the expected result exclusively on that public behavior path; the Rust
    # implementation is an independent matched-junction rewrite.
    connected = skrf.network.connect(
        network_a,
        CONNECT_PORT_A,
        network_b,
        CONNECT_PORT_B,
    )
    frequency = np.asarray(connected.f, dtype=np.float64)
    connected_s = np.asarray(connected.s, dtype=np.complex128)
    connected_z0 = np.asarray(connected.z0, dtype=np.complex128)
    if not np.array_equal(frequency, frequency_hz):
        raise ValueError("matched-connection output frequency changed")
    if not np.isfinite(connected_s).all() or not np.isfinite(connected_z0).all():
        raise ValueError("matched-connection output must be finite")

    a_survivors = [port for port in range(CONNECT_NPORTS_A) if port != CONNECT_PORT_A]
    b_survivors = [port for port in range(CONNECT_NPORTS_B) if port != CONNECT_PORT_B]
    output_order = [
        *({"network": "A", "port": port} for port in a_survivors),
        *({"network": "B", "port": port} for port in b_survivors),
    ]
    shape = {
        "frequency": list(frequency.shape),
        "input_s_a": list(source_s_a.shape),
        "input_s_b": list(source_s_b.shape),
        "input_z0_a": list(source_z0_a.shape),
        "input_z0_b": list(source_z0_b.shape),
        "output_s": list(connected_s.shape),
        "output_z0": list(connected_z0.shape),
    }
    return {
        "metadata": {
            "case_id": case_id,
            "junction_ports": {"a": CONNECT_PORT_A, "b": CONNECT_PORT_B},
            "numpy_version": np.__version__,
            "operation": "connect_matched",
            "port_order": {
                "a_survivors": a_survivors,
                "b_survivors": b_survivors,
                "description": (
                    "A unconnected ports in original order, followed by B "
                    "unconnected ports in original order"
                ),
                "output": output_order,
            },
            "random_seeds": {
                "a": CONNECT_RANDOM_SEED_A,
                "b": CONNECT_RANDOM_SEED_B,
            },
            "reference_impedance": {
                "external_complex_allowed": True,
                "junction_exactly_equal": True,
                "junction_frequency_dependent": True,
                "junction_real_strictly_positive": True,
                "unit": "ohm",
            },
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": shape,
            "tolerance_policy": {
                "atol": CONNECT_ATOL,
                "comparison": "abs(actual-expected) <= atol + rtol*abs(expected)",
                "justification": CONNECT_TOLERANCE_JUSTIFICATION,
                "regeneration": (
                    "canonical UTF-8 JSON serialization; s_connected is checked "
                    "with the recorded numeric tolerance; z0/order/inputs are "
                    "checked exactly"
                ),
                "rtol": CONNECT_RTOL,
            },
            "wave_definition": connected.s_def,
        },
        "data": {
            "frequency_hz": [float(value) for value in frequency],
            "s_a": _complex_array(source_s_a),
            "s_b": _complex_array(source_s_b),
            "s_connected": _complex_array(connected_s),
            "z0_a_ohm": _complex_array(source_z0_a),
            "z0_b_ohm": _complex_array(source_z0_b),
            "z0_connected_ohm": _complex_array(connected_z0),
        },
    }


def _inner_connect_inputs(np: Any, *, seed: int) -> tuple[Any, Any, Any, Any]:
    """Build independent finite S/z0 input for the inner-connect case."""

    frequency_hz = np.array(
        [0.71e9, 1.17e9, 1.89e9],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    s = (
        rng.normal(
            loc=0.0,
            scale=0.021,
            size=(INNER_CONNECT_NFREQ, INNER_CONNECT_NPORTS, INNER_CONNECT_NPORTS),
        )
        + 1j
        * rng.normal(
            loc=0.0,
            scale=0.021,
            size=(INNER_CONNECT_NFREQ, INNER_CONNECT_NPORTS, INNER_CONNECT_NPORTS),
        )
    ).astype(np.complex128)
    for frequency in range(INNER_CONNECT_NFREQ):
        for port in range(INNER_CONNECT_NPORTS):
            s[frequency, port, port] += complex(
                0.12 + 0.014 * frequency + 0.006 * port,
                0.009 + 0.002 * frequency - 0.001 * port,
            )

    # External z0 values are deliberately varied by both frequency and port;
    # the selected pair is overwritten with the exact matched junction profile.
    frequency_index = np.arange(INNER_CONNECT_NFREQ, dtype=np.float64)[:, None]
    port_index = np.arange(INNER_CONNECT_NPORTS, dtype=np.float64)[None, :]
    z0 = (
        34.0
        + 2.6 * port_index
        + 1.8 * frequency_index
    ).astype(np.complex128)
    junction_z0 = np.array([64.25, 77.5, 93.75], dtype=np.float64)
    z0[:, INNER_CONNECT_PORT_K] = junction_z0
    z0[:, INNER_CONNECT_PORT_L] = junction_z0

    if not np.isfinite(frequency_hz).all():
        raise ValueError("inner-connect frequencies must be finite")
    if not np.isfinite(s).all():
        raise ValueError("inner-connect S input must be finite")
    if not np.isfinite(z0).all():
        raise ValueError("inner-connect z0 input must be finite")
    if not (z0[:, INNER_CONNECT_PORT_K].imag == 0.0).all() or not (
        z0[:, INNER_CONNECT_PORT_L].imag == 0.0
    ).all():
        raise ValueError("inner-connect junction z0 must be real")
    if not np.array_equal(
        z0[:, INNER_CONNECT_PORT_K], z0[:, INNER_CONNECT_PORT_L]
    ):
        raise ValueError("inner-connect junction z0 must match exactly")
    if not (z0[:, INNER_CONNECT_PORT_K].real > 0.0).all():
        raise ValueError("inner-connect junction z0 must be positive")
    if np.array_equal(z0[0], z0[1]):
        raise ValueError("inner-connect z0 must vary by frequency")
    if np.array_equal(z0[:, 0], z0[:, 2]):
        raise ValueError("inner-connect external z0 must vary by port")
    _assert_non_symmetric(np, s, name="inner-connect S input")
    _assert_s_conditioning(s)

    return frequency_hz, s, z0, junction_z0


def _inner_connect_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build the inner-connect fixture through public scikit-rf behavior."""

    frequency_hz, source_s, source_z0, _junction_z0 = _inner_connect_inputs(
        np,
        seed=INNER_CONNECT_RANDOM_SEED,
    )
    case_id = INNER_CONNECT_CASE_ID
    network = skrf.Network(
        f=frequency_hz,
        s=source_s,
        z0=source_z0,
        s_def="power",
        name=case_id,
    )

    # Use only the public same-network innerconnect behavior for expected S.
    # The Rust kernel is an independent rewrite of the matched power-wave
    # elimination equation.  Every fixture z0 is real (with the selected pair
    # also exactly equal), so scikit-rf's internal power/pseudo conversion is
    # numerically equivalent here.
    connected = skrf.network.innerconnect(
        network,
        INNER_CONNECT_PORT_K,
        INNER_CONNECT_PORT_L,
    )
    frequency = np.asarray(connected.f, dtype=np.float64)
    connected_s = np.asarray(connected.s, dtype=np.complex128)
    connected_z0 = np.asarray(connected.z0, dtype=np.complex128)
    if not np.array_equal(frequency, frequency_hz):
        raise ValueError("inner-connect output frequency changed")
    if not np.isfinite(connected_s).all() or not np.isfinite(connected_z0).all():
        raise ValueError("inner-connect output must be finite")

    survivors = [
        port
        for port in range(INNER_CONNECT_NPORTS)
        if port not in (INNER_CONNECT_PORT_K, INNER_CONNECT_PORT_L)
    ]
    shape = {
        "frequency": list(frequency.shape),
        "input_s": list(source_s.shape),
        "input_z0": list(source_z0.shape),
        "output_s": list(connected_s.shape),
        "output_z0": list(connected_z0.shape),
    }
    return {
        "metadata": {
            "case_id": case_id,
            "junction_ports": {"k": INNER_CONNECT_PORT_K, "l": INNER_CONNECT_PORT_L},
            "numpy_version": np.__version__,
            "operation": "inner_connect_matched",
            "port_order": {
                "survivors": survivors,
                "output": survivors,
                "description": "Input ports excluding k and l, in original order",
            },
            "random_seed": INNER_CONNECT_RANDOM_SEED,
            "reference_impedance": {
                "external_complex_allowed": True,
                "junction_exactly_equal": True,
                "junction_frequency_dependent": True,
                "junction_real_strictly_positive": True,
                "unit": "ohm",
            },
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": shape,
            "tolerance_policy": {
                "atol": INNER_CONNECT_ATOL,
                "comparison": "abs(actual-expected) <= atol + rtol*abs(expected)",
                "justification": INNER_CONNECT_TOLERANCE_JUSTIFICATION,
                "regeneration": (
                    "canonical UTF-8 JSON serialization; s_inner_connected is "
                    "checked with the recorded numeric tolerance; z0/order/inputs "
                    "are checked exactly"
                ),
                "rtol": INNER_CONNECT_RTOL,
            },
            # The input is explicitly constructed with s_def="power".  The
            # pinned public helper currently returns a pseudo-labelled object
            # after its internal conversion, but the fixture's all-real z0
            # profile leaves the expected numeric S values identical to the
            # power-wave result.
            "wave_definition": network.s_def,
        },
        "data": {
            "frequency_hz": [float(value) for value in frequency],
            "s": _complex_array(source_s),
            "s_inner_connected": _complex_array(connected_s),
            "z0_ohm": _complex_array(source_z0),
            "z0_inner_connected_ohm": _complex_array(connected_z0),
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


def _power_wave_admittance_inputs(
    np: Any,
    *,
    quantity: str,
    seed: int,
) -> tuple[Any, Any, Any]:
    """Build one direct S or Y input and the shared explicit complex z0.

    The two operation builders call this helper with independent seeds.  The
    helper never derives one parameter matrix from the other, so a fixture
    cannot accidentally certify a round-trip instead of the requested direct
    public scikit-rf operation.
    """

    if quantity not in {"s", "y"}:
        raise ValueError(f"unknown power-wave admittance quantity: {quantity!r}")

    frequency_hz, _constructor_z0, expanded_z0 = _matrix_frequency_and_z0(
        np,
        nfreq=POWER_WAVE_ADMITTANCE_NFREQ,
        nports=POWER_WAVE_ADMITTANCE_NPORTS,
        z0_profile="complex_per_port_frequency_dependent",
    )
    if expanded_z0.shape != (
        POWER_WAVE_ADMITTANCE_NFREQ,
        POWER_WAVE_ADMITTANCE_NPORTS,
    ):
        raise ValueError("power-wave S/Y z0 has an unexpected shape")
    if not np.isfinite(expanded_z0).all() or not (expanded_z0.real != 0.0).all():
        raise ValueError("power-wave S/Y z0 must be finite with non-zero real parts")
    if not (expanded_z0.imag != 0.0).all():
        raise ValueError("power-wave S/Y z0 must be genuinely complex")
    if np.array_equal(expanded_z0[0], expanded_z0[1]):
        raise ValueError("power-wave S/Y z0 must vary by frequency")
    if np.array_equal(expanded_z0[:, 0], expanded_z0[:, 1]):
        raise ValueError("power-wave S/Y z0 must vary by port")

    rng = np.random.default_rng(seed)
    if quantity == "s":
        scale = 0.012
        diagonal_base = 0.14
        diagonal_step = 0.012
        name = "direct power-wave S input"
    else:
        scale = 0.0002
        diagonal_base = 0.022
        diagonal_step = 0.0018
        name = "direct power-wave Y input"
    matrix = (
        rng.normal(
            loc=0.0,
            scale=scale,
            size=(
                POWER_WAVE_ADMITTANCE_NFREQ,
                POWER_WAVE_ADMITTANCE_NPORTS,
                POWER_WAVE_ADMITTANCE_NPORTS,
            ),
        )
        + 1j
        * rng.normal(
            loc=0.0,
            scale=scale,
            size=(
                POWER_WAVE_ADMITTANCE_NFREQ,
                POWER_WAVE_ADMITTANCE_NPORTS,
                POWER_WAVE_ADMITTANCE_NPORTS,
            ),
        )
    ).astype(np.complex128)
    for frequency in range(POWER_WAVE_ADMITTANCE_NFREQ):
        for port in range(POWER_WAVE_ADMITTANCE_NPORTS):
            matrix[frequency, port, port] += complex(
                diagonal_base + diagonal_step * frequency + 0.0015 * port,
                0.004 + 0.0007 * frequency - 0.0003 * port,
            )

    _assert_non_symmetric(np, matrix, name=name)
    _assert_parameter_diagonal_dominance(
        np,
        matrix,
        name=name,
        margin=0.25 * diagonal_base,
    )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    for frequency in range(POWER_WAVE_ADMITTANCE_NFREQ):
        if np.linalg.matrix_rank(matrix[frequency]) != POWER_WAVE_ADMITTANCE_NPORTS:
            raise ValueError(f"{name} must have full pinned-NumPy matrix rank")

    return frequency_hz, matrix, expanded_z0


def _power_wave_admittance_fixture(
    np: Any,
    skrf: Any,
    *,
    case_id: str,
    operation: str,
    seed: int,
) -> dict[str, Any]:
    """Build one direct S↔Y fixture through explicit public scikit-rf APIs."""

    if operation == "s_to_y":
        input_quantity = "s"
        input_key = "s"
        output_key = "y_s"
        input_unit = "dimensionless"
        output_unit = "S"
        absolute_tolerance_key = "atol_s"
        absolute_tolerance = POWER_WAVE_S_TO_Y_ATOL_S
        rtol = POWER_WAVE_S_TO_Y_RTOL
    elif operation == "y_to_s":
        input_quantity = "y"
        input_key = "y_s"
        output_key = "s"
        input_unit = "S"
        output_unit = "dimensionless"
        absolute_tolerance_key = "atol"
        absolute_tolerance = POWER_WAVE_Y_TO_S_ATOL
        rtol = POWER_WAVE_Y_TO_S_RTOL
    else:
        raise ValueError(f"unknown power-wave S/Y operation: {operation!r}")

    frequency_hz, direct_input, expanded_z0 = _power_wave_admittance_inputs(
        np,
        quantity=input_quantity,
        seed=seed,
    )
    if operation == "s_to_y":
        # Keep the wave definition explicit even though power is the current
        # scikit-rf default; this is the behavior contract for the Rust kernel.
        expected_output = skrf.network.s2y(
            direct_input,
            z0=expanded_z0,
            s_def="power",
        )
    else:
        expected_output = skrf.network.y2s(
            direct_input,
            z0=expanded_z0,
            s_def="power",
        )
    expected_output = np.asarray(expected_output, dtype=np.complex128)
    if not np.isfinite(expected_output).all():
        raise ValueError(f"{case_id} output must be finite")

    reference_impedance = _matrix_reference_impedance_flags(
        "complex_per_port_frequency_dependent"
    )
    input_shape_key = f"input_{input_quantity}"
    output_shape_key = f"output_{output_key.removesuffix('_s')}"
    data = {
        "frequency_hz": [float(value) for value in frequency_hz],
        input_key: _complex_array(direct_input),
        "z0_ohm": _complex_array(expanded_z0),
        output_key: _complex_array(expected_output),
    }
    metadata = {
        "case_id": case_id,
        "input_unit": input_unit,
        "numpy_version": np.__version__,
        "operation": operation,
        "output_unit": output_unit,
        "random_seed": seed,
        "reference_impedance": reference_impedance,
        "schema": "rfkit-rs.oracle.fixture",
        "schema_version": SCHEMA_VERSION,
        "scikit_rf_version": skrf.__version__,
        "shape": {
            "frequency": list(frequency_hz.shape),
            input_shape_key: list(direct_input.shape),
            "input_z0": list(expanded_z0.shape),
            output_shape_key: list(expected_output.shape),
        },
        "tolerance_policy": {
            absolute_tolerance_key: absolute_tolerance,
            "comparison": (
                f"abs(actual-expected) <= {absolute_tolerance_key} + "
                "rtol*abs(expected)"
            ),
            "justification": POWER_WAVE_ADMITTANCE_TOLERANCE_JUSTIFICATION,
            "regeneration": (
                f"canonical UTF-8 JSON serialization; {output_key} is checked "
                "with the recorded numeric tolerance"
            ),
            "rtol": rtol,
        },
        "wave_definition": "power",
    }
    return {"metadata": metadata, "data": data}


def _power_wave_s_to_y_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    return _power_wave_admittance_fixture(
        np,
        skrf,
        case_id="power_wave_s_to_y_three_port_complex_z0",
        operation="s_to_y",
        seed=POWER_WAVE_S_TO_Y_RANDOM_SEED,
    )


def _power_wave_y_to_s_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    return _power_wave_admittance_fixture(
        np,
        skrf,
        case_id="power_wave_y_to_s_three_port_complex_z0",
        operation="y_to_s",
        seed=POWER_WAVE_Y_TO_S_RANDOM_SEED,
    )


def _matrix_reference_impedance_flags(z0_profile: str) -> dict[str, Any]:
    """Return the contract flags for one matrix-case z0 profile."""

    profiles = {
        "real_scalar": {
            "complex": False,
            "frequency_dependent": False,
            "per_port": False,
        },
        "complex_per_port_constant": {
            "complex": True,
            "frequency_dependent": False,
            "per_port": True,
        },
        "real_frequency_dependent": {
            "complex": False,
            "frequency_dependent": True,
            "per_port": False,
        },
        "complex_per_port_frequency_dependent": {
            "complex": True,
            "frequency_dependent": True,
            "per_port": True,
        },
    }
    try:
        flags = dict(profiles[z0_profile])
    except KeyError as error:
        raise ValueError(f"unknown z0 profile: {z0_profile!r}") from error
    flags["unit"] = "ohm"
    return flags


def _matrix_s_to_z_fixture(
    np: Any,
    skrf: Any,
    *,
    case_id: str,
    nfreq: int,
    nports: int,
    seed: int,
    z0_profile: str,
) -> dict[str, Any]:
    """Build one matrix-case S-to-Z fixture through public scikit-rf APIs."""

    frequency_hz, source_s, constructor_z0, _expanded_z0 = _matrix_s_inputs(
        np,
        nfreq=nfreq,
        nports=nports,
        seed=seed,
        z0_profile=z0_profile,
    )
    network = skrf.Network(
        f=frequency_hz,
        s=source_s,
        z0=constructor_z0,
        s_def="power",
        name=case_id,
    )

    # Read both inputs and the expected output back through the public Network
    # object.  The expected operation result is intentionally not computed by
    # this module's local equations.
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
            "case_id": case_id,
            "numpy_version": np.__version__,
            "operation": "s_to_z",
            "random_seed": seed,
            "reference_impedance": _matrix_reference_impedance_flags(z0_profile),
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
                "justification": MATRIX_S_TO_Z_TOLERANCE_JUSTIFICATION,
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


def _matrix_z_to_s_fixture(
    np: Any,
    skrf: Any,
    *,
    case_id: str,
    nfreq: int,
    nports: int,
    seed: int,
    z0_profile: str,
) -> dict[str, Any]:
    """Build one matrix-case Z-to-S fixture through public scikit-rf APIs."""

    frequency_hz, source_z, constructor_z0, _expanded_z0 = _matrix_z_inputs(
        np,
        nfreq=nfreq,
        nports=nports,
        seed=seed,
        z0_profile=z0_profile,
    )
    converted = skrf.Network.from_z(
        source_z,
        f=frequency_hz,
        z0=constructor_z0,
        s_def="power",
        name=case_id,
    )
    converted_s = np.asarray(converted.s, dtype=np.complex128)
    network_z0 = np.asarray(converted.z0, dtype=np.complex128)
    shape = {
        "frequency": list(frequency_hz.shape),
        "input_z": list(source_z.shape),
        "input_z0": list(network_z0.shape),
        "output_s": list(converted_s.shape),
    }

    return {
        "metadata": {
            "case_id": case_id,
            "numpy_version": np.__version__,
            "operation": "z_to_s",
            "random_seed": seed,
            "reference_impedance": _matrix_reference_impedance_flags(z0_profile),
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": shape,
            "tolerance_policy": {
                "atol": Z_TO_S_ATOL,
                "comparison": "abs(actual-expected) <= atol + rtol*abs(expected)",
                "justification": MATRIX_Z_TO_S_TOLERANCE_JUSTIFICATION,
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
            "z0_ohm": _complex_array(network_z0),
            "z_ohm": _complex_array(source_z),
        },
    }


def _reciprocal_s_to_z_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build the deterministic reciprocal S-to-Z three-port fixture."""

    case_id = "power_wave_s_to_z_three_port_reciprocal_real_equal_z0"
    nfreq = 3
    nports = 3
    frequency_hz, source_s, constructor_z0, _expanded_z0 = _reciprocal_s_inputs(
        np,
        nfreq=nfreq,
        nports=nports,
        seed=RECIPROCAL_S_TO_Z_RANDOM_SEED,
        z0_ohm=RECIPROCAL_S_TO_Z_Z0_OHM,
    )
    network = skrf.Network(
        f=frequency_hz,
        s=source_s,
        z0=constructor_z0,
        s_def="power",
        name=case_id,
    )

    # Read inputs and the expected result through public Network properties;
    # no local conversion equation is used to produce the oracle output.
    frequency = np.asarray(network.f, dtype=np.float64)
    network_s = np.asarray(network.s, dtype=np.complex128)
    network_z0 = np.asarray(network.z0, dtype=np.complex128)
    network_z = np.asarray(network.z, dtype=np.complex128)
    _assert_real_positive_equal_z0(np, network_z0, name="reciprocal S-to-Z")
    _assert_exact_symmetric(np, network_s, name="reciprocal S input")
    _assert_output_symmetric(
        np,
        network_z,
        name="reciprocal S-to-Z output",
        rtol=S_TO_Z_RTOL,
        atol=S_TO_Z_ATOL_OHM,
    )
    passive_network = _passive_network_metadata(np, [("s", network_s)])

    return {
        "metadata": {
            "passive_network": passive_network,
            "case_id": case_id,
            "numpy_version": np.__version__,
            "operation": "s_to_z",
            "random_seed": RECIPROCAL_S_TO_Z_RANDOM_SEED,
            "reference_impedance": {
                "complex": False,
                "frequency_dependent": False,
                "per_port": False,
                "unit": "ohm",
            },
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": {
                "frequency": list(frequency.shape),
                "input_s": list(network_s.shape),
                "input_z0": list(network_z0.shape),
                "output_z": list(network_z.shape),
            },
            "tolerance_policy": {
                "atol_ohm": S_TO_Z_ATOL_OHM,
                "comparison": (
                    "abs(actual-expected) <= "
                    "atol_ohm + rtol*abs(expected)"
                ),
                "justification": RECIPROCAL_TOLERANCE_JUSTIFICATION,
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


def _reciprocal_z_to_s_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build the deterministic reciprocal Z-to-S three-port fixture."""

    case_id = "power_wave_z_to_s_three_port_reciprocal_real_equal_z0"
    nfreq = 3
    nports = 3
    frequency_hz, source_z, constructor_z0, _expanded_z0 = _reciprocal_z_inputs(
        np,
        nfreq=nfreq,
        nports=nports,
        seed=RECIPROCAL_Z_TO_S_RANDOM_SEED,
        z0_ohm=RECIPROCAL_S_TO_Z_Z0_OHM,
    )

    # The expected output is obtained exclusively through public
    # Network.from_z(..., s_def="power") and the public Network.s property.
    converted = skrf.Network.from_z(
        source_z,
        f=frequency_hz,
        z0=constructor_z0,
        s_def="power",
        name=case_id,
    )
    frequency = np.asarray(converted.f, dtype=np.float64)
    converted_s = np.asarray(converted.s, dtype=np.complex128)
    network_z0 = np.asarray(converted.z0, dtype=np.complex128)
    _assert_real_positive_equal_z0(np, network_z0, name="reciprocal Z-to-S")
    _assert_exact_symmetric(np, source_z, name="reciprocal Z input")
    _assert_output_symmetric(
        np,
        converted_s,
        name="reciprocal Z-to-S output",
        rtol=Z_TO_S_RTOL,
        atol=Z_TO_S_ATOL,
    )
    passive_network = _passive_network_metadata(np, [("s", converted_s)])

    return {
        "metadata": {
            "passive_network": passive_network,
            "case_id": case_id,
            "numpy_version": np.__version__,
            "operation": "z_to_s",
            "random_seed": RECIPROCAL_Z_TO_S_RANDOM_SEED,
            "reference_impedance": {
                "complex": False,
                "frequency_dependent": False,
                "per_port": False,
                "unit": "ohm",
            },
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": {
                "frequency": list(frequency.shape),
                "input_z": list(source_z.shape),
                "input_z0": list(network_z0.shape),
                "output_s": list(converted_s.shape),
            },
            "tolerance_policy": {
                "atol": Z_TO_S_ATOL,
                "comparison": "abs(actual-expected) <= atol + rtol*abs(expected)",
                "justification": RECIPROCAL_TOLERANCE_JUSTIFICATION,
                "regeneration": (
                    "canonical UTF-8 JSON serialization; s is checked with the "
                    "recorded numeric tolerance"
                ),
                "rtol": Z_TO_S_RTOL,
            },
            "wave_definition": converted.s_def,
        },
        "data": {
            "frequency_hz": [float(value) for value in frequency],
            "s": _complex_array(converted_s),
            "z0_ohm": _complex_array(network_z0),
            "z_ohm": _complex_array(source_z),
        },
    }


def _reciprocal_renormalize_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build the deterministic reciprocal S-renormalization fixture."""

    case_id = "power_wave_renormalize_three_port_reciprocal_real_equal_z0"
    nfreq = 3
    nports = 3
    frequency_hz, source_s, source_constructor_z0, _expanded_z0 = _reciprocal_s_inputs(
        np,
        nfreq=nfreq,
        nports=nports,
        seed=RECIPROCAL_RENORMALIZE_RANDOM_SEED,
        z0_ohm=RECIPROCAL_RENORMALIZE_SOURCE_Z0_OHM,
    )
    network = skrf.Network(
        f=frequency_hz,
        s=source_s,
        z0=source_constructor_z0,
        s_def="power",
        name=case_id,
    )

    # Read source values back, then use only the public in-place renormalize
    # operation and read-back properties for the expected output and target
    # reference impedance.
    frequency = np.asarray(network.f, dtype=np.float64)
    network_s_input = np.asarray(network.s, dtype=np.complex128)
    network_z0_source = np.asarray(network.z0, dtype=np.complex128)
    _assert_real_positive_equal_z0(
        np, network_z0_source, name="reciprocal renormalization source"
    )
    network.renormalize(RECIPROCAL_RENORMALIZE_TARGET_Z0_OHM, s_def="power")
    network_s_renormalized = np.asarray(network.s, dtype=np.complex128)
    network_z0_target = np.asarray(network.z0, dtype=np.complex128)
    _assert_real_positive_equal_z0(
        np, network_z0_target, name="reciprocal renormalization target"
    )
    _assert_exact_symmetric(np, network_s_input, name="reciprocal S input")
    _assert_output_symmetric(
        np,
        network_s_renormalized,
        name="reciprocal renormalized-S output",
        rtol=RENORMALIZE_RTOL,
        atol=RENORMALIZE_ATOL,
    )
    passive_network = _passive_network_metadata(
        np,
        [("s_input", network_s_input), ("s_renormalized", network_s_renormalized)],
    )

    flags = {
        "complex": False,
        "frequency_dependent": False,
        "per_port": False,
        "unit": "ohm",
    }
    return {
        "metadata": {
            "passive_network": passive_network,
            "case_id": case_id,
            "numpy_version": np.__version__,
            "operation": "renormalize_s",
            "random_seed": RECIPROCAL_RENORMALIZE_RANDOM_SEED,
            "reference_impedance": {
                "source": dict(flags),
                "target": dict(flags),
            },
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": {
                "frequency": list(frequency.shape),
                "s_input": list(network_s_input.shape),
                "s_renormalized": list(network_s_renormalized.shape),
                "z0_source": list(network_z0_source.shape),
                "z0_target": list(network_z0_target.shape),
            },
            "tolerance_policy": {
                "atol": RENORMALIZE_ATOL,
                "comparison": "abs(actual-expected) <= atol + rtol*abs(expected)",
                "justification": RECIPROCAL_TOLERANCE_JUSTIFICATION,
                "regeneration": (
                    "canonical UTF-8 JSON serialization; s_renormalized is checked "
                    "with the recorded numeric tolerance"
                ),
                "rtol": RENORMALIZE_RTOL,
            },
            "wave_definition": network.s_def,
        },
        "data": {
            "frequency_hz": [float(value) for value in frequency],
            "s_input": _complex_array(network_s_input),
            "s_renormalized": _complex_array(network_s_renormalized),
            "z0_source_ohm": _complex_array(network_z0_source),
            "z0_target_ohm": _complex_array(network_z0_target),
        },
    }


def _active_s_to_z_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build the active three-port S-to-Z fixture through public scikit-rf."""

    case_id = "power_wave_s_to_z_three_port_active_real_equal_z0"
    frequency_hz, source_s, constructor_z0, _expanded_z0 = _active_s_inputs(
        np,
        seed=ACTIVE_S_TO_Z_RANDOM_SEED,
    )
    network = skrf.Network(
        f=frequency_hz,
        s=source_s,
        z0=constructor_z0,
        s_def="power",
        name=case_id,
    )
    frequency = np.asarray(network.f, dtype=np.float64)
    network_s = np.asarray(network.s, dtype=np.complex128)
    network_z0 = np.asarray(network.z0, dtype=np.complex128)
    network_z = np.asarray(network.z, dtype=np.complex128)
    active_network = _active_network_metadata(np, network_s, matrix_field="s")

    return {
        "metadata": {
            "active_network": active_network,
            "case_id": case_id,
            "numpy_version": np.__version__,
            "operation": "s_to_z",
            "random_seed": ACTIVE_S_TO_Z_RANDOM_SEED,
            "reference_impedance": {
                "complex": False,
                "frequency_dependent": False,
                "per_port": False,
                "unit": "ohm",
            },
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": {
                "frequency": list(frequency.shape),
                "input_s": list(network_s.shape),
                "input_z0": list(network_z0.shape),
                "output_z": list(network_z.shape),
            },
            "tolerance_policy": {
                "atol_ohm": S_TO_Z_ATOL_OHM,
                "comparison": (
                    "abs(actual-expected) <= "
                    "atol_ohm + rtol*abs(expected)"
                ),
                "justification": (
                    "Strict binary64 tolerance for a well-conditioned, modest-"
                    "magnitude deterministic active three-port case; the active "
                    "input passes a conservative I-S diagonal-dominance guard "
                    "while allowing normal cross-language linear-algebra rounding."
                ),
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


def _active_z_to_s_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build the active three-port Z-to-S fixture from direct Z data."""

    case_id = "power_wave_z_to_s_three_port_active_real_equal_z0"
    frequency_hz, source_z, constructor_z0, _expanded_z0 = _active_z_inputs(
        np,
        seed=ACTIVE_Z_TO_S_RANDOM_SEED,
    )
    converted = skrf.Network.from_z(
        source_z,
        f=frequency_hz,
        z0=constructor_z0,
        s_def="power",
        name=case_id,
    )
    frequency = np.asarray(converted.f, dtype=np.float64)
    converted_s = np.asarray(converted.s, dtype=np.complex128)
    network_z0 = np.asarray(converted.z0, dtype=np.complex128)
    active_network = _active_network_metadata(np, converted_s, matrix_field="s")

    return {
        "metadata": {
            "active_network": active_network,
            "case_id": case_id,
            "numpy_version": np.__version__,
            "operation": "z_to_s",
            "random_seed": ACTIVE_Z_TO_S_RANDOM_SEED,
            "reference_impedance": {
                "complex": False,
                "frequency_dependent": False,
                "per_port": False,
                "unit": "ohm",
            },
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": {
                "frequency": list(frequency.shape),
                "input_z": list(source_z.shape),
                "input_z0": list(network_z0.shape),
                "output_s": list(converted_s.shape),
            },
            "tolerance_policy": {
                "atol": Z_TO_S_ATOL,
                "comparison": "abs(actual-expected) <= atol + rtol*abs(expected)",
                "justification": (
                    "Strict binary64 tolerance for a well-conditioned, modest-"
                    "magnitude deterministic active three-port case; the direct "
                    "negative-resistance Z input passes a conservative Z+G "
                    "diagonal-dominance guard while allowing normal cross-language "
                    "linear-algebra rounding."
                ),
                "regeneration": (
                    "canonical UTF-8 JSON serialization; s is checked with the "
                    "recorded numeric tolerance"
                ),
                "rtol": Z_TO_S_RTOL,
            },
            "wave_definition": converted.s_def,
        },
        "data": {
            "frequency_hz": [float(value) for value in frequency],
            "s": _complex_array(converted_s),
            "z0_ohm": _complex_array(network_z0),
            "z_ohm": _complex_array(source_z),
        },
    }


def _active_renormalize_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build the active three-port S-renormalization fixture."""

    case_id = "power_wave_renormalize_three_port_active_real_equal_z0"
    frequency_hz, source_s, source_z0, _expanded_z0 = _active_s_inputs(
        np,
        seed=ACTIVE_RENORMALIZE_RANDOM_SEED,
        z0_ohm=ACTIVE_RENORMALIZE_SOURCE_Z0_OHM,
    )
    network = skrf.Network(
        f=frequency_hz,
        s=source_s,
        z0=source_z0,
        s_def="power",
        name=case_id,
    )
    frequency = np.asarray(network.f, dtype=np.float64)
    network_s_input = np.asarray(network.s, dtype=np.complex128)
    network_z0_source = np.asarray(network.z0, dtype=np.complex128)
    # Validate the second conversion stage explicitly before mutating the
    # Network.  Network.z is the public scikit-rf read-back of the underlying
    # Z matrix, and the expanded target array is the exact real/equal-per-port
    # reference used by the subsequent renormalization.
    network_z = np.asarray(network.z, dtype=np.complex128)
    target_z0_for_guard = np.full(
        network_z0_source.shape,
        ACTIVE_RENORMALIZE_TARGET_Z0_OHM,
        dtype=np.complex128,
    )
    _assert_z_conditioning(network_z, target_z0_for_guard)
    network.renormalize(ACTIVE_RENORMALIZE_TARGET_Z0_OHM, s_def="power")
    network_s_renormalized = np.asarray(network.s, dtype=np.complex128)
    network_z0_target = np.asarray(network.z0, dtype=np.complex128)
    # The contract deliberately records source S as the relevant active
    # matrix.  Renormalization is still checked through its complete output;
    # no passive/active classifier is introduced for a target network.
    active_network = _active_network_metadata(
        np,
        network_s_input,
        matrix_field="s_input",
    )
    flags = {
        "complex": False,
        "frequency_dependent": False,
        "per_port": False,
        "unit": "ohm",
    }

    return {
        "metadata": {
            "active_network": active_network,
            "case_id": case_id,
            "numpy_version": np.__version__,
            "operation": "renormalize_s",
            "random_seed": ACTIVE_RENORMALIZE_RANDOM_SEED,
            "reference_impedance": {
                "source": dict(flags),
                "target": dict(flags),
            },
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": {
                "frequency": list(frequency.shape),
                "s_input": list(network_s_input.shape),
                "s_renormalized": list(network_s_renormalized.shape),
                "z0_source": list(network_z0_source.shape),
                "z0_target": list(network_z0_target.shape),
            },
            "tolerance_policy": {
                "atol": RENORMALIZE_ATOL,
                "comparison": "abs(actual-expected) <= atol + rtol*abs(expected)",
                "justification": (
                    "Strict binary64 tolerance for a well-conditioned, modest-"
                    "magnitude deterministic active three-port case; the source "
                    "S input passes a conservative I-S diagonal-dominance guard "
                    "and the underlying target-stage Z passes the corresponding "
                    "Z+G guard while allowing normal cross-language linear-algebra "
                    "rounding."
                ),
                "regeneration": (
                    "canonical UTF-8 JSON serialization; s_renormalized is checked "
                    "with the recorded numeric tolerance"
                ),
                "rtol": RENORMALIZE_RTOL,
            },
            "wave_definition": network.s_def,
        },
        "data": {
            "frequency_hz": [float(value) for value in frequency],
            "s_input": _complex_array(network_s_input),
            "s_renormalized": _complex_array(network_s_renormalized),
            "z0_source_ohm": _complex_array(network_z0_source),
            "z0_target_ohm": _complex_array(network_z0_target),
        },
    }


def _near_singular_s_to_z_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build a direct-S near-singular three-port S-to-Z fixture."""

    case_id = "power_wave_s_to_z_three_port_near_singular_real_equal_z0"
    frequency_hz, source_s, constructor_z0, _expanded_z0 = _near_singular_s_inputs(
        np,
        seed=NEAR_SINGULAR_S_TO_Z_RANDOM_SEED,
    )
    network = skrf.Network(
        f=frequency_hz,
        s=source_s,
        z0=constructor_z0,
        s_def="power",
        name=case_id,
    )

    # The operation input and expected output are read from the public Network
    # object.  The system evidence is reconstructed from the input S itself,
    # never from the expected Z output.
    frequency = np.asarray(network.f, dtype=np.float64)
    network_s = np.asarray(network.s, dtype=np.complex128)
    network_z0 = np.asarray(network.z0, dtype=np.complex128)
    network_z = np.asarray(network.z, dtype=np.complex128)
    identity = np.eye(NEAR_SINGULAR_NPORTS, dtype=np.complex128)
    input_system = identity[None, :, :] - network_s
    _assert_real_positive_equal_z0(np, network_z0, name="near-singular S-to-Z")
    _assert_upper_triangular_system(
        np,
        input_system,
        system_matrix=NEAR_SINGULAR_S_TO_Z_SYSTEM,
    )
    if not np.isfinite(network_z).all():
        raise ValueError("near-singular S-to-Z output must be finite")

    return {
        "metadata": {
            "case_id": case_id,
            "near_singular": _near_singular_metadata(
                system_matrix=NEAR_SINGULAR_S_TO_Z_SYSTEM
            ),
            "numpy_version": np.__version__,
            "operation": "s_to_z",
            "random_seed": NEAR_SINGULAR_S_TO_Z_RANDOM_SEED,
            "reference_impedance": {
                "complex": False,
                "frequency_dependent": False,
                "per_port": False,
                "unit": "ohm",
            },
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": {
                "frequency": list(frequency.shape),
                "input_s": list(network_s.shape),
                "input_z0": list(network_z0.shape),
                "output_z": list(network_z.shape),
            },
            "tolerance_policy": {
                "atol_ohm": NEAR_SINGULAR_S_TO_Z_ATOL_OHM,
                "comparison": (
                    "abs(actual-expected) <= "
                    "atol_ohm + rtol*abs(expected)"
                ),
                "justification": NEAR_SINGULAR_TOLERANCE_JUSTIFICATION,
                "regeneration": (
                    "canonical UTF-8 JSON serialization; z_ohm is checked "
                    "with the case-local recorded numeric tolerance"
                ),
                "rtol": NEAR_SINGULAR_S_TO_Z_RTOL,
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


def _near_singular_z_to_s_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    """Build a direct-Z near-singular three-port Z-to-S fixture."""

    case_id = "power_wave_z_to_s_three_port_near_singular_real_equal_z0"
    frequency_hz, source_z, constructor_z0, expanded_z0 = _near_singular_z_inputs(
        np,
        seed=NEAR_SINGULAR_Z_TO_S_RANDOM_SEED,
    )

    # Keep this expected output independent of the S-to-Z case: source_z is a
    # separate direct construction and scikit-rf's public from_z/s path is the
    # only source of the expected S values.
    converted = skrf.Network.from_z(
        source_z,
        f=frequency_hz,
        z0=constructor_z0,
        s_def="power",
        name=case_id,
    )
    frequency = np.asarray(converted.f, dtype=np.float64)
    converted_s = np.asarray(converted.s, dtype=np.complex128)
    network_z0 = np.asarray(converted.z0, dtype=np.complex128)
    normalized_system = source_z / expanded_z0[:, :, None]
    for port in range(NEAR_SINGULAR_NPORTS):
        normalized_system[:, port, port] += 1.0
    _assert_real_positive_equal_z0(np, network_z0, name="near-singular Z-to-S")
    _assert_upper_triangular_system(
        np,
        normalized_system,
        system_matrix=NEAR_SINGULAR_Z_TO_S_SYSTEM,
    )
    if not np.isfinite(converted_s).all():
        raise ValueError("near-singular Z-to-S output must be finite")

    return {
        "metadata": {
            "case_id": case_id,
            "near_singular": _near_singular_metadata(
                system_matrix=NEAR_SINGULAR_Z_TO_S_SYSTEM
            ),
            "numpy_version": np.__version__,
            "operation": "z_to_s",
            "random_seed": NEAR_SINGULAR_Z_TO_S_RANDOM_SEED,
            "reference_impedance": {
                "complex": False,
                "frequency_dependent": False,
                "per_port": False,
                "unit": "ohm",
            },
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": {
                "frequency": list(frequency.shape),
                "input_z": list(source_z.shape),
                "input_z0": list(network_z0.shape),
                "output_s": list(converted_s.shape),
            },
            "tolerance_policy": {
                "atol": NEAR_SINGULAR_Z_TO_S_ATOL,
                "comparison": "abs(actual-expected) <= atol + rtol*abs(expected)",
                "justification": NEAR_SINGULAR_TOLERANCE_JUSTIFICATION,
                "regeneration": (
                    "canonical UTF-8 JSON serialization; s is checked "
                    "with the case-local recorded numeric tolerance"
                ),
                "rtol": NEAR_SINGULAR_Z_TO_S_RTOL,
            },
            "wave_definition": converted.s_def,
        },
        "data": {
            "frequency_hz": [float(value) for value in frequency],
            "s": _complex_array(converted_s),
            "z0_ohm": _complex_array(network_z0),
            "z_ohm": _complex_array(source_z),
        },
    }


def _renormalize_fixture(
    np: Any,
    skrf: Any,
    *,
    case_id: str = (
        "power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0"
    ),
    nfreq: int = 3,
    nports: int = 4,
    seed: int = RENORMALIZE_FOUR_PORT_RANDOM_SEED,
    z0_profile: str = "complex_per_port_frequency_dependent",
) -> dict[str, Any]:
    """Build one S-renormalization fixture through public scikit-rf APIs."""

    (
        frequency_hz,
        source_s,
        source_constructor_z0,
        target_constructor_z0,
        _source_z0,
        _target_z0,
    ) = _renormalize_inputs(
        np,
        nfreq=nfreq,
        nports=nports,
        seed=seed,
        z0_profile=z0_profile,
    )
    network = skrf.Network(
        f=frequency_hz,
        s=source_s,
        z0=source_constructor_z0,
        s_def="power",
        name=case_id,
    )

    # Read the constructor inputs back through the public Network object so
    # the fixture records scikit-rf's canonical representations.  The output
    # is obtained only through the public in-place renormalize operation and
    # public Network.s/Network.z0 properties; no conversion formula is used
    # by this oracle generator.
    frequency = np.asarray(network.f, dtype=np.float64)
    network_s_input = np.asarray(network.s, dtype=np.complex128)
    network_z0_source = np.asarray(network.z0, dtype=np.complex128)
    network.renormalize(target_constructor_z0, s_def="power")
    network_s_renormalized = np.asarray(network.s, dtype=np.complex128)
    network_z0_target = np.asarray(network.z0, dtype=np.complex128)

    shape = {
        "frequency": list(frequency.shape),
        "s_input": list(network_s_input.shape),
        "s_renormalized": list(network_s_renormalized.shape),
        "z0_source": list(network_z0_source.shape),
        "z0_target": list(network_z0_target.shape),
    }

    reference_impedance_flags = _matrix_reference_impedance_flags(z0_profile)
    return {
        "metadata": {
            "case_id": case_id,
            "numpy_version": np.__version__,
            "operation": "renormalize_s",
            "random_seed": seed,
            "reference_impedance": {
                "source": dict(reference_impedance_flags),
                "target": dict(reference_impedance_flags),
            },
            "schema": "rfkit-rs.oracle.fixture",
            "schema_version": SCHEMA_VERSION,
            "scikit_rf_version": skrf.__version__,
            "shape": shape,
            "tolerance_policy": {
                "atol": RENORMALIZE_ATOL,
                "comparison": "abs(actual-expected) <= atol + rtol*abs(expected)",
                "justification": RENORMALIZE_TOLERANCE_JUSTIFICATION,
                "regeneration": (
                    "canonical UTF-8 JSON serialization; s_renormalized is checked "
                    "with the recorded numeric tolerance"
                ),
                "rtol": RENORMALIZE_RTOL,
            },
            "wave_definition": network.s_def,
        },
        "data": {
            "frequency_hz": [float(value) for value in frequency],
            "s_input": _complex_array(network_s_input),
            "s_renormalized": _complex_array(network_s_renormalized),
            "z0_source_ohm": _complex_array(network_z0_source),
            "z0_target_ohm": _complex_array(network_z0_target),
        },
    }


def _renormalize_one_port_real_scalar_z0_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _renormalize_fixture(
        np,
        skrf,
        case_id="power_wave_renormalize_one_port_real_scalar_z0",
        nfreq=3,
        nports=1,
        seed=RENORMALIZE_ONE_PORT_RANDOM_SEED,
        z0_profile="real_scalar",
    )


def _renormalize_two_port_complex_per_port_constant_z0_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _renormalize_fixture(
        np,
        skrf,
        case_id="power_wave_renormalize_two_port_complex_per_port_constant_z0",
        nfreq=4,
        nports=2,
        seed=RENORMALIZE_TWO_PORT_RANDOM_SEED,
        z0_profile="complex_per_port_constant",
    )


def _renormalize_four_port_complex_per_port_frequency_dependent_z0_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _renormalize_fixture(
        np,
        skrf,
        case_id=(
            "power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0"
        ),
        nfreq=3,
        nports=4,
        seed=RENORMALIZE_FOUR_PORT_RANDOM_SEED,
        z0_profile="complex_per_port_frequency_dependent",
    )


def _renormalize_eight_port_real_frequency_dependent_z0_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _renormalize_fixture(
        np,
        skrf,
        case_id="power_wave_renormalize_eight_port_real_frequency_dependent_z0",
        nfreq=3,
        nports=8,
        seed=RENORMALIZE_EIGHT_PORT_RANDOM_SEED,
        z0_profile="real_frequency_dependent",
    )


def _s_to_z_one_port_real_scalar_z0_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    return _matrix_s_to_z_fixture(
        np,
        skrf,
        case_id="power_wave_s_to_z_one_port_real_scalar_z0",
        nfreq=3,
        nports=1,
        seed=20_260_901,
        z0_profile="real_scalar",
    )


def _s_to_z_two_port_complex_per_port_constant_z0_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _matrix_s_to_z_fixture(
        np,
        skrf,
        case_id="power_wave_s_to_z_two_port_complex_per_port_constant_z0",
        nfreq=4,
        nports=2,
        seed=20_260_902,
        z0_profile="complex_per_port_constant",
    )


def _s_to_z_four_port_real_frequency_dependent_z0_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _matrix_s_to_z_fixture(
        np,
        skrf,
        case_id="power_wave_s_to_z_four_port_real_frequency_dependent_z0",
        nfreq=3,
        nports=4,
        seed=20_260_903,
        z0_profile="real_frequency_dependent",
    )


def _s_to_z_eight_port_complex_per_port_frequency_dependent_z0_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _matrix_s_to_z_fixture(
        np,
        skrf,
        case_id="power_wave_s_to_z_eight_port_complex_per_port_frequency_dependent_z0",
        nfreq=3,
        nports=8,
        seed=20_260_904,
        z0_profile="complex_per_port_frequency_dependent",
    )


def _z_to_s_one_port_real_scalar_z0_fixture(np: Any, skrf: Any) -> dict[str, Any]:
    return _matrix_z_to_s_fixture(
        np,
        skrf,
        case_id="power_wave_z_to_s_one_port_real_scalar_z0",
        nfreq=3,
        nports=1,
        seed=20_260_911,
        z0_profile="real_scalar",
    )


def _z_to_s_two_port_complex_per_port_constant_z0_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _matrix_z_to_s_fixture(
        np,
        skrf,
        case_id="power_wave_z_to_s_two_port_complex_per_port_constant_z0",
        nfreq=4,
        nports=2,
        seed=20_260_912,
        z0_profile="complex_per_port_constant",
    )


def _z_to_s_four_port_real_frequency_dependent_z0_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _matrix_z_to_s_fixture(
        np,
        skrf,
        case_id="power_wave_z_to_s_four_port_real_frequency_dependent_z0",
        nfreq=3,
        nports=4,
        seed=20_260_913,
        z0_profile="real_frequency_dependent",
    )


def _z_to_s_eight_port_complex_per_port_frequency_dependent_z0_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _matrix_z_to_s_fixture(
        np,
        skrf,
        case_id="power_wave_z_to_s_eight_port_complex_per_port_frequency_dependent_z0",
        nfreq=3,
        nports=8,
        seed=20_260_914,
        z0_profile="complex_per_port_frequency_dependent",
    )


def _impedance_admittance_z_to_y_well_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _impedance_admittance_fixture(
        np,
        skrf,
        case_id="impedance_admittance_z_to_y_three_port_well_conditioned",
        operation="z_to_y",
        seed=IMPEDANCE_ADMITTANCE_Z_TO_Y_WELL_RANDOM_SEED,
        near_singular=False,
    )


def _impedance_admittance_y_to_z_well_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _impedance_admittance_fixture(
        np,
        skrf,
        case_id="impedance_admittance_y_to_z_three_port_well_conditioned",
        operation="y_to_z",
        seed=IMPEDANCE_ADMITTANCE_Y_TO_Z_WELL_RANDOM_SEED,
        near_singular=False,
    )


def _impedance_admittance_z_to_y_near_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _impedance_admittance_fixture(
        np,
        skrf,
        case_id="impedance_admittance_z_to_y_three_port_near_singular",
        operation="z_to_y",
        seed=IMPEDANCE_ADMITTANCE_Z_TO_Y_NEAR_RANDOM_SEED,
        near_singular=True,
    )


def _impedance_admittance_y_to_z_near_fixture(
    np: Any, skrf: Any
) -> dict[str, Any]:
    return _impedance_admittance_fixture(
        np,
        skrf,
        case_id="impedance_admittance_y_to_z_three_port_near_singular",
        operation="y_to_z",
        seed=IMPEDANCE_ADMITTANCE_Y_TO_Z_NEAR_RANDOM_SEED,
        near_singular=True,
    )


_CASES = (
    _OracleCase(
        "three_port_complex_z0",
        DEFAULT_FIXTURE,
        _network_fixture,
        "exact",
    ),
    _OracleCase(
        INTERPOLATION_CASE_ID,
        INTERPOLATION_FIXTURE,
        _interpolation_fixture,
        "numeric_output",
        ("s", "z0_ohm"),
    ),
    _OracleCase(
        COMPOSITION_CASE_ID,
        COMPOSITION_FIXTURE,
        _composition_fixture,
        "numeric_output",
        ("s_connected", "z0_connected_ohm"),
    ),
    _OracleCase(
        CONNECT_CASE_ID,
        CONNECT_FIXTURE,
        _connect_matched_fixture,
        "numeric_output",
        "s_connected",
    ),
    _OracleCase(
        INNER_CONNECT_CASE_ID,
        INNER_CONNECT_FIXTURE,
        _inner_connect_fixture,
        "numeric_output",
        "s_inner_connected",
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
    _OracleCase(
        "power_wave_s_to_y_three_port_complex_z0",
        POWER_WAVE_S_TO_Y_FIXTURE,
        _power_wave_s_to_y_fixture,
        "numeric_output",
        "y_s",
    ),
    _OracleCase(
        "power_wave_y_to_s_three_port_complex_z0",
        POWER_WAVE_Y_TO_S_FIXTURE,
        _power_wave_y_to_s_fixture,
        "numeric_output",
        "s",
    ),
    _OracleCase(
        "power_wave_s_to_z_one_port_real_scalar_z0",
        S_TO_Z_ONE_PORT_FIXTURE,
        _s_to_z_one_port_real_scalar_z0_fixture,
        "numeric_output",
        "z_ohm",
    ),
    _OracleCase(
        "power_wave_s_to_z_two_port_complex_per_port_constant_z0",
        S_TO_Z_TWO_PORT_FIXTURE,
        _s_to_z_two_port_complex_per_port_constant_z0_fixture,
        "numeric_output",
        "z_ohm",
    ),
    _OracleCase(
        "power_wave_s_to_z_four_port_real_frequency_dependent_z0",
        S_TO_Z_FOUR_PORT_FIXTURE,
        _s_to_z_four_port_real_frequency_dependent_z0_fixture,
        "numeric_output",
        "z_ohm",
    ),
    _OracleCase(
        "power_wave_s_to_z_eight_port_complex_per_port_frequency_dependent_z0",
        S_TO_Z_EIGHT_PORT_FIXTURE,
        _s_to_z_eight_port_complex_per_port_frequency_dependent_z0_fixture,
        "numeric_output",
        "z_ohm",
    ),
    _OracleCase(
        "power_wave_z_to_s_one_port_real_scalar_z0",
        Z_TO_S_ONE_PORT_FIXTURE,
        _z_to_s_one_port_real_scalar_z0_fixture,
        "numeric_output",
        "s",
    ),
    _OracleCase(
        "power_wave_z_to_s_two_port_complex_per_port_constant_z0",
        Z_TO_S_TWO_PORT_FIXTURE,
        _z_to_s_two_port_complex_per_port_constant_z0_fixture,
        "numeric_output",
        "s",
    ),
    _OracleCase(
        "power_wave_z_to_s_four_port_real_frequency_dependent_z0",
        Z_TO_S_FOUR_PORT_FIXTURE,
        _z_to_s_four_port_real_frequency_dependent_z0_fixture,
        "numeric_output",
        "s",
    ),
    _OracleCase(
        "power_wave_z_to_s_eight_port_complex_per_port_frequency_dependent_z0",
        Z_TO_S_EIGHT_PORT_FIXTURE,
        _z_to_s_eight_port_complex_per_port_frequency_dependent_z0_fixture,
        "numeric_output",
        "s",
    ),
    _OracleCase(
        "power_wave_s_to_z_three_port_reciprocal_real_equal_z0",
        RECIPROCAL_S_TO_Z_FIXTURE,
        _reciprocal_s_to_z_fixture,
        "numeric_output",
        "z_ohm",
    ),
    _OracleCase(
        "power_wave_z_to_s_three_port_reciprocal_real_equal_z0",
        RECIPROCAL_Z_TO_S_FIXTURE,
        _reciprocal_z_to_s_fixture,
        "numeric_output",
        "s",
    ),
    _OracleCase(
        "power_wave_renormalize_one_port_real_scalar_z0",
        RENORMALIZE_ONE_PORT_FIXTURE,
        _renormalize_one_port_real_scalar_z0_fixture,
        "numeric_output",
        "s_renormalized",
    ),
    _OracleCase(
        "power_wave_renormalize_two_port_complex_per_port_constant_z0",
        RENORMALIZE_TWO_PORT_FIXTURE,
        _renormalize_two_port_complex_per_port_constant_z0_fixture,
        "numeric_output",
        "s_renormalized",
    ),
    _OracleCase(
        "power_wave_renormalize_four_port_complex_per_port_frequency_dependent_z0",
        RENORMALIZE_FOUR_PORT_FIXTURE,
        _renormalize_four_port_complex_per_port_frequency_dependent_z0_fixture,
        "numeric_output",
        "s_renormalized",
    ),
    _OracleCase(
        "power_wave_renormalize_eight_port_real_frequency_dependent_z0",
        RENORMALIZE_EIGHT_PORT_FIXTURE,
        _renormalize_eight_port_real_frequency_dependent_z0_fixture,
        "numeric_output",
        "s_renormalized",
    ),
    _OracleCase(
        "power_wave_renormalize_three_port_reciprocal_real_equal_z0",
        RECIPROCAL_RENORMALIZE_FIXTURE,
        _reciprocal_renormalize_fixture,
        "numeric_output",
        "s_renormalized",
    ),
    _OracleCase(
        "power_wave_s_to_z_three_port_active_real_equal_z0",
        ACTIVE_S_TO_Z_FIXTURE,
        _active_s_to_z_fixture,
        "numeric_output",
        "z_ohm",
    ),
    _OracleCase(
        "power_wave_z_to_s_three_port_active_real_equal_z0",
        ACTIVE_Z_TO_S_FIXTURE,
        _active_z_to_s_fixture,
        "numeric_output",
        "s",
    ),
    _OracleCase(
        "power_wave_renormalize_three_port_active_real_equal_z0",
        ACTIVE_RENORMALIZE_FIXTURE,
        _active_renormalize_fixture,
        "numeric_output",
        "s_renormalized",
    ),
    _OracleCase(
        "power_wave_s_to_z_three_port_near_singular_real_equal_z0",
        NEAR_SINGULAR_S_TO_Z_FIXTURE,
        _near_singular_s_to_z_fixture,
        "numeric_output",
        "z_ohm",
    ),
    _OracleCase(
        "power_wave_z_to_s_three_port_near_singular_real_equal_z0",
        NEAR_SINGULAR_Z_TO_S_FIXTURE,
        _near_singular_z_to_s_fixture,
        "numeric_output",
        "s",
    ),
    _OracleCase(
        "impedance_admittance_z_to_y_three_port_well_conditioned",
        IMPEDANCE_ADMITTANCE_Z_TO_Y_WELL_FIXTURE,
        _impedance_admittance_z_to_y_well_fixture,
        "numeric_output",
        "y_s",
    ),
    _OracleCase(
        "impedance_admittance_y_to_z_three_port_well_conditioned",
        IMPEDANCE_ADMITTANCE_Y_TO_Z_WELL_FIXTURE,
        _impedance_admittance_y_to_z_well_fixture,
        "numeric_output",
        "z_ohm",
    ),
    _OracleCase(
        "impedance_admittance_z_to_y_three_port_near_singular",
        IMPEDANCE_ADMITTANCE_Z_TO_Y_NEAR_FIXTURE,
        _impedance_admittance_z_to_y_near_fixture,
        "numeric_output",
        "y_s",
    ),
    _OracleCase(
        "impedance_admittance_y_to_z_three_port_near_singular",
        IMPEDANCE_ADMITTANCE_Y_TO_Z_NEAR_FIXTURE,
        _impedance_admittance_y_to_z_near_fixture,
        "numeric_output",
        "z_ohm",
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


def _numeric_output_keys(
    output_key: str | tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Normalize one or more numerically tolerant output field names."""

    if isinstance(output_key, str):
        output_keys = (output_key,)
    elif isinstance(output_key, (tuple, list)):
        output_keys = tuple(output_key)
    else:
        raise TypeError("numeric output key must be a string or a sequence of strings")
    if not output_keys or any(not isinstance(key, str) or not key for key in output_keys):
        raise ValueError("numeric output keys must be non-empty strings")
    if len(set(output_keys)) != len(output_keys):
        raise ValueError("numeric output keys must be unique")
    return output_keys


def _contract_projection(
    document: dict[str, Any],
    output_key: str | tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Remove only the operation outputs before exact contract comparison."""

    output_keys = _numeric_output_keys(output_key)

    data = document.get("data")
    if not isinstance(data, dict):
        raise ValueError("fixture data must be a JSON object")
    for key in output_keys:
        if key not in data:
            raise ValueError(f"fixture data is missing numeric output {key!r}")

    projection = dict(document)
    projected_data = dict(data)
    for key in output_keys:
        del projected_data[key]
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

    absolute_keys = [
        key for key in ("atol", "atol_ohm", "atol_s") if key in policy
    ]
    if len(absolute_keys) != 1:
        raise ValueError(
            "fixture tolerance_policy must contain exactly one of "
            "'atol', 'atol_ohm', or 'atol_s'"
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
    output_key: str | tuple[str, ...] | list[str],
) -> int:
    """Check canonical fixture outputs whose values are numerically tolerant."""

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
        for key in _numeric_output_keys(output_key):
            regenerated_output = expected["data"][key]
            checked_in_output = actual["data"][key]
            mismatch = _compare_numeric_output(
                regenerated_output,
                checked_in_output,
                path=f"data.{key}",
                rtol=rtol,
                atol=atol,
            )
            if mismatch is not None:
                print(
                    f"fixture numeric output check failed: {path}: {mismatch}",
                    file=sys.stderr,
                )
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
