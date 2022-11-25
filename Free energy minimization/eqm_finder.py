import numpy as np
from scipy.integrate import simpson
from scipy.signal import argrelextrema

def calc_sin_squared_half_arccot(x: np.ndarray) -> np.ndarray:
    return 0.5 - 0.5 * x / np.sqrt(x**2 + 1)

def calc_cos_squared_half_arccot(x: np.ndarray) -> np.ndarray:
    return 0.5 + 0.5 * x / np.sqrt(x**2 + 1)

def calc_sin_arccot(x: np.ndarray) -> np.ndarray:
    return 1 / np.sqrt(x**2 + 1)

def calc_cot_double_psi(psi_0: np.ndarray, lambda_r: float, v_swell: float, zeta: float) -> np.ndarray:
    """Return `cot(2*psi)` where psi is the post-transformation twist angle."""
    lambda_z = v_swell / lambda_r**2

    numerator_1 = (lambda_z**2 + lambda_r**2) * (zeta - 1) * np.cos(2 * psi_0)
    numerator_2 = (lambda_z**2 - lambda_r**2) * (zeta + 1)
    denominator = 2 * lambda_r * lambda_z * (zeta - 1) * np.sin(2 * psi_0)

    return (numerator_1 + numerator_2) / denominator

def calc_psi(psi_0: np.ndarray, lambda_r: float, v_swell: float, zeta: float) -> np.ndarray:
    """Calculate final twist-angle."""
    return np.pi/4 - np.arctan(calc_cot_double_psi(psi_0, lambda_r, v_swell, zeta))/2

def calc_reduced_free_energy_density(psi_0: np.ndarray, lambda_r: float, v_swell: float, zeta: float) -> np.ndarray:
    """Calculate the free energy density relative to energy-scale."""
    cos_squared_psi_0 = np.cos(psi_0)**2
    sin_squared_psi_0 = np.sin(psi_0)**2
    sin_double_psi_0 = np.sin(2 * psi_0)

    cot_double_psi = calc_cot_double_psi(psi_0, lambda_r, v_swell, zeta)
    cos_squared_psi = calc_cos_squared_half_arccot(cot_double_psi)
    sin_squared_psi = calc_sin_squared_half_arccot(cot_double_psi)
    sin_double_psi = calc_sin_arccot(cot_double_psi)

    lambda_z = v_swell / lambda_r**2

    term_1 = lambda_r**2 * ((zeta - 1) * sin_squared_psi_0 * cos_squared_psi
                            + (1/zeta - 1) * cos_squared_psi_0 * sin_squared_psi
                            + 2)
    
    term_2 = lambda_z**2 * ((zeta - 1) * cos_squared_psi_0 * sin_squared_psi
                            + (1/zeta - 1) * sin_squared_psi_0 * cos_squared_psi
                            + 1)

    term_3 = lambda_r * lambda_z * ((2 - zeta - 1/zeta) * sin_double_psi_0 * sin_double_psi) / 2

    return (term_1 + term_2 + term_3) / 2

def calc_reduced_free_energy_linear_twist_angle(psi_0_surf: float, lambda_r: float, v_swell: float, zeta: float,
                                                array_length: int = 10000) -> float:
    reduced_radii = np.linspace(0, 1, array_length+1)[1:]

    psi_0 = lambda_r * psi_0_surf * reduced_radii

    reduced_free_energy_density = calc_reduced_free_energy_density(psi_0, lambda_r, v_swell, zeta)

    return 2 * simpson(reduced_free_energy_density * reduced_radii, reduced_radii)

def find_local_optima(array):
    local_maxima = argrelextrema(array, np.greater_equal)
    local_minima = argrelextrema(array, np.less_equal)

    return local_minima, local_maxima