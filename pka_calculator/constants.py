"""Physical constants used by the thermodynamic layer."""

R_J_MOL_K = 8.31446261815324
HARTREE_TO_KJ_MOL = 2625.4996394799
HARTREE_TO_KCAL_MOL = HARTREE_TO_KJ_MOL / 4.184
DEFAULT_TEMPERATURE_K = 298.15
DEFAULT_PKW_298 = 14.0


def rtln10_kj_mol(temperature_k: float) -> float:
    """Return R*T*ln(10) in kJ/mol."""
    import math

    return R_J_MOL_K * float(temperature_k) * math.log(10.0) / 1000.0
