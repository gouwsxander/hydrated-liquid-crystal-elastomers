from typing import Any

import numpy as np
import matplotlib.pyplot as plt

import eqm_finder

def plot_reduced_free_energy_vs_lambda_r_constant_twist(lambda_r: np.ndarray, psi_0: float, v_swell: float, zeta: float, plot_kwargs: dict[str, Any]):
    """Plots reduced free energy against lambda_r for some given range of lambda_r values.
    In the case of constant twist, the reduced free energy is equal to the free energy density."""

    free_energy_array = eqm_finder.calc_reduced_free_energy_density(psi_0, lambda_r, v_swell, zeta)
    plt.plot(lambda_r, free_energy_array, **plot_kwargs)

def plot_reduced_free_energy_vs_swell_anisotropy_constant_twist(lambda_r: np.ndarray, psi_0: float, v_swell: float, zeta: float, plot_kwargs: dict[str, Any]):
    """Plots reduced free energy against swell anisotropy = lambda_r/lambda_z for some given range of lambda_r values.
    In the case of constant twist, the reduced free energy is equal to the free energy density."""

    free_energy_array = eqm_finder.calc_reduced_free_energy_density(psi_0, lambda_r, v_swell, zeta)

    lambda_z = v_swell / lambda_r**2

    plt.plot(lambda_r / lambda_z, free_energy_array, **plot_kwargs)

plot_reduced_free_energy_vs_lambda_r_constant_twist(np.linspace(0.5, 1.5), 0.1, 2, 1.3, {'label': "Wahoo"})