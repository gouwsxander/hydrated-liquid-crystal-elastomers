import numpy as np
import matplotlib.pyplot as plt

import eqm_finder

def plot_free_energy_vs_lambda_r(v_swell, zeta, psi_0, label):
    lambda_r_min = 0.5
    lambda_r_max = 1.5
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
    plot_free_energy_vs_lambda_r(1.3, 1.3, 0.78, 'Hair')
    plot_free_energy_vs_lambda_r(2, 1.3, 0.02, 'Collagen')

    plt.xlabel("Radial deformation, $\\lambda_r$")
    plt.ylabel("Dimensionless free energy density, $f/\\mu$")

    plt.legend(loc='best')

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
    lambda_r_min = 0.5
    lambda_r_max = 1.5
    num_points = 1000
    lambda_r_array = np.linspace(lambda_r_min, lambda_r_max, num_points)

    free_energy_per_mu_array = np.zeros(num_points)

    for index, lambda_r in enumerate(lambda_r_array):
        free_energy_per_mu_array[index] = eqm_finder.calc_reduced_free_energy_linear_twist_angle(psi_surf, lambda_r, v_swell, zeta)

    plt.plot(lambda_r_array, free_energy_per_mu_array, label = label)

def plot_eqm_deformation_anisotropy_3d_constant_twist(lambda_r_array, psi_0_array, v_swell, zeta_array, plot_kwargs = {}):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    minima_psi_0s = []
    minima_zetas = []
    local_minima = []
    maxima_psi_0s = []
    maxima_zetas = []
    local_maxima = []

    for psi_0 in psi_0_array:
        for zeta in zeta_array:
            free_energy_array = eqm_finder.calc_reduced_free_energy_density(psi_0, lambda_r_array, v_swell, zeta)

            lambda_z = v_swell / lambda_r_array**2
            deformation_anisotropy = lambda_r_array / lambda_z

            new_local_minima, new_local_maxima = eqm_finder.find_local_optima(free_energy_array)

            for new_local_min in new_local_minima[0]:
                #print(v_swell)
                #print(deformation_anisotropy[new_local_min])
                if new_local_min not in [0, len(deformation_anisotropy) - 1]:
                    minima_psi_0s.append(psi_0)
                    minima_zetas.append(zeta)
                    local_minima.append(deformation_anisotropy[new_local_min])

            for new_local_max in new_local_maxima[0]:
                if new_local_max not in [0, len(deformation_anisotropy) - 1]:
                    maxima_psi_0s.append(psi_0)
                    maxima_zetas.append(zeta)
                    local_maxima.append(deformation_anisotropy[new_local_max])


    ax.scatter(minima_psi_0s, minima_zetas, local_minima, s=1, **plot_kwargs)
    ax.scatter(maxima_psi_0s, maxima_zetas, local_maxima, s=1, **plot_kwargs)

if __name__ == "__main__":
    #compare_hair_collagen()
    #vary_hydration([1,1.25,1.5,1.75,2], 1.3, 1 * np.pi/180)

    #vary_twist_angle(np.array([1, 15, 30, 45]) * np.pi/180, 1.5, 1.3)

    #plot_linear_free_energy_vs_lambda_r(0.5, 1.3, np.pi/180, "Collagen (Linear)")
    #plot_free_energy_vs_lambda_r(0.5, 1.3, np.pi/180, "Collagen (Constant) 1.3")
    #plot_linear_free_energy_vs_lambda_r(0.5, 1.3, np.pi/180, "Collagen 1.3")
    #plot_linear_free_energy_vs_lambda_r(0.5, 1.2, np.pi/180, "Collagen 1.2")
    #plot_linear_free_energy_vs_lambda_r(0.5, 1.1, np.pi/180, "Collagen 1.1")
    #plot_linear_free_energy_vs_lambda_r(0.5, 0.8, np.pi/180, "Collagen 0.8")

    #plot_linear_free_energy_vs_lambda_r(1.3, 1.3, 45 * np.pi/180, "Hair (Linear)")
    #plot_free_energy_vs_lambda_r(1.3, 1.3, 45 * np.pi/180, "Hair (Constant)")

    #plt.legend(loc='best')
    #plt.show()

    ZERO = 1e-9

    lambda_r_array_super_close = np.linspace(0.2, 1.7, 1000)
    good_zeta_range = np.linspace(0.01, 5, 1000)
    good_psi_0_range = np.linspace(ZERO, 90 - ZERO, 1000)
    v_swell = 2

    plot_eqm_deformation_anisotropy_3d_constant_twist(lambda_r_array_super_close, good_psi_0_range, v_swell, good_zeta_range)
    plt.show()
    