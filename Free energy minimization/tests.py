import numpy as np
import matplotlib.pyplot as plt

import eqm_finder

def plot_free_energy_vs_lambda_r(v_swell, zeta, psi_0):
    lambda_r_min = 0.9
    lambda_r_max = 2
    num_points = 100
    lambda_r_array = np.linspace(lambda_r_min, lambda_r_max, num_points)

    free_energy_density_per_mu_array = np.zeros(num_points)

    for index, lambda_r in enumerate(lambda_r_array):
        free_energy_density_per_mu_array[index] = eqm_finder.calc_reduced_free_energy_density(psi_0, lambda_r, v_swell, zeta)

    plt.plot(lambda_r_array, free_energy_density_per_mu_array)

    plt.xlabel("Radial deformation, $\\lambda_r$")
    plt.ylabel("Dimensionless free energy density, $f/\\mu$")

    #plt.yscale('log')

    plt.title(f"{v_swell=}, {zeta=}, {psi_0=}")

    plt.show()

if __name__ == "__main__":
    plot_free_energy_vs_lambda_r(1.3, 1.3, 0.78) # hair
    plot_free_energy_vs_lambda_r(2, 1.3, 0.02) # collagen