import numpy as np
import matplotlib.pyplot as plt

import eqm_finder

def plot_free_energy_vs_lambda_r(v_swell, zeta, psi_0, label):
    lambda_r_min = 1.0
    lambda_r_max = 1.6
    num_points = 1000
    lambda_r_array = np.linspace(lambda_r_min, lambda_r_max, num_points)

    free_energy_density_per_mu_array = np.zeros(num_points)

    for index, lambda_r in enumerate(lambda_r_array):
        free_energy_density_per_mu_array[index] = eqm_finder.calc_reduced_free_energy_density(psi_0, lambda_r, v_swell, zeta)

    plt.plot(lambda_r_array, free_energy_density_per_mu_array, label = label)

def plot_deriv_free_energy_vs_lambda_r(v_swell, zeta, psi_0, label):
    lambda_r_min = 0.9
    lambda_r_max = 1.4
    num_points = 1000
    lambda_r_array = np.linspace(lambda_r_min, lambda_r_max, num_points)

    free_energy_density_per_mu_array = np.zeros(num_points)

    for index, lambda_r in enumerate(lambda_r_array):
        free_energy_density_per_mu_array[index] = eqm_finder.calc_reduced_free_energy_density(psi_0, lambda_r, v_swell, zeta)

    plt.plot(lambda_r_array, np.gradient(free_energy_density_per_mu_array, lambda_r_array), label = label)

    plt.plot(lambda_r_array, np.zeros(num_points), 'k--')

def compare_hair_collagen():
    plot_free_energy_vs_lambda_r(1.3, 1.3, 0.78, 'hair')
    plot_free_energy_vs_lambda_r(2, 1.3, 0.02, 'collagen')

    plt.xlabel("Radial deformation, $\\lambda_r$")
    plt.ylabel("Dimensionless free energy density, $f/\\mu$")

    #plt.yscale('log')

    plt.show()

def vary_twist_angle(psi_0_array, v_swell, zeta):
    for psi_0 in psi_0_array:
        plot_free_energy_vs_lambda_r(v_swell, zeta, psi_0, f"{psi_0=}")

    plt.xlabel("Radial deformation, $\\lambda_r$")
    plt.ylabel("Reduced free energy density, $f/\\mu$")

    plt.title(f"{zeta=}, {v_swell=}")

    #plt.yscale('log')

    plt.legend(loc='best')

    plt.show()

def vary_hydration(v_swell_array, zeta, psi_0):
    for v_swell in v_swell_array:
        plot_free_energy_vs_lambda_r(v_swell, zeta, psi_0, f"{v_swell=}")

    plt.xlabel("Radial deformation, $\\lambda_r$")
    plt.ylabel("Reduced free energy density, $f/\\mu$")

    plt.title(f"{zeta=}, {psi_0=}")

    #plt.yscale('log')

    plt.legend(loc='best')

    plt.show()

def vary_hydration_deriv(v_swell_array, zeta, psi_0):
    for v_swell in v_swell_array:
        plot_deriv_free_energy_vs_lambda_r(v_swell, zeta, psi_0, f"{v_swell=}")

    plt.xlabel("Radial deformation, $\\lambda_r$")
    plt.ylabel("Derivative free energy density, $d(f/\\mu)/d\\lambda_r$")

    plt.title(f"{zeta=}, {psi_0=}")

    #plt.yscale('log')

    plt.legend(loc='best')

    plt.show()

def plot_linear_free_energy_vs_lambda_r(v_swell, zeta, psi_surf, label):
    lambda_r_min = 0.9
    lambda_r_max = 1.6
    num_points = 1000
    lambda_r_array = np.linspace(lambda_r_min, lambda_r_max, num_points)

    free_energy_density_per_mu_array = np.zeros(num_points)

    for index, lambda_r in enumerate(lambda_r_array):
        free_energy_density_per_mu_array[index] = eqm_finder.calc_reduced_free_energy_linear_twist_angle(psi_surf, lambda_r, v_swell, zeta)

    plt.plot(lambda_r_array, free_energy_density_per_mu_array, label = label)

if __name__ == "__main__":
    # compare_hair_collagen()
    #vary_hydration([1,1.25,1.5,1.75,2], 1.3, np.pi/180)

    #vary_twist_angle(np.array([1, 15, 30, 45]) * np.pi/180, 1.5, 1.3)

    plot_linear_free_energy_vs_lambda_r(2, 1.3, 1 * np.pi/180, "Collagen")
    plot_linear_free_energy_vs_lambda_r(1.3, 1.3, 45 * np.pi/180, "Hair")
    plt.show()
    