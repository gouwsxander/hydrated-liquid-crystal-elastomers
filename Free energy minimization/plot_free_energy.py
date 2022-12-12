from typing import Any

import numpy as np
import matplotlib.pyplot as plt

import eqm_finder

DEFAULT_DPI = 400
DEFAULT_WIDTH = 6
DEFAULT_HEIGHT = 7.5

ZERO = 1e-7

def create_fig_ax(parameter_name, parameter_symbol, parameter_units="", dpi=DEFAULT_DPI, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    fig, (free_energy_ax, eqm_ax) = plt.subplots(2, 1)
    fig.set_size_inches(width, height)
    fig.set_dpi(dpi)

    free_energy_ax.set_xlabel("Deformation anisotropy, $\\lambda_r / \\lambda_z$")
    free_energy_ax.set_ylabel("Reduced free energy, $\\tilde{F}$")
    free_energy_ax.set_title("(A)")

    eqm_ax_xlabel = f"{parameter_name}, ${parameter_symbol}$"
    if parameter_units != "":
        eqm_ax_xlabel += f" ({parameter_units})"
    eqm_ax.set_xlabel(eqm_ax_xlabel)
    eqm_ax.set_ylabel("Deformation anisotropy, $\\lambda_r / \\lambda_z$")
    eqm_ax.set_title("(B)")

    fig.tight_layout()

    return fig, (free_energy_ax, eqm_ax)

def plot_constant_twist_swell_dependence(deformation_anisotropy_array: np.ndarray,
                                    sparse_swell_ratio_array: np.ndarray, dense_swell_ratio_array: np.ndarray,
                                    initial_twist_angle: float, anisotropy: float,
                                    free_energy_min: float, free_energy_max: float):
    fig, (free_energy_ax, eqm_ax) = create_fig_ax("Swell ratio", "v")
    free_energy_ax.set_ylabel("Scaled free energy, $\\tilde{F} / v^{2/3}$")

    # Plot reduced free energy vs deformation anisotropy
    for swell_ratio in sparse_swell_ratio_array:
        radial_deformation_array = (swell_ratio * deformation_anisotropy_array)**(1/3)
        reduced_free_energy_array = eqm_finder.calc_reduced_free_energy_density(initial_twist_angle, radial_deformation_array, swell_ratio, anisotropy)
        free_energy_ax.plot(deformation_anisotropy_array, reduced_free_energy_array / swell_ratio**(2/3), label = f"$v = {swell_ratio}$")

    free_energy_ax.legend(loc='best')
    free_energy_ax.set_xlim(np.min(deformation_anisotropy_array) - ZERO, np.max(deformation_anisotropy_array))
    free_energy_ax.set_ylim(free_energy_min, free_energy_max)

    # Plot equilibria
    minima_curves = [[[], []], [[], []]]
    maxima_curves = [[[], []]]

    for swell_ratio in dense_swell_ratio_array:
        radial_deformation_array = (swell_ratio * deformation_anisotropy_array)**(1/3)
        reduced_free_energy_array = eqm_finder.calc_reduced_free_energy_density(initial_twist_angle, radial_deformation_array, swell_ratio, anisotropy)

        arg_local_minima, arg_local_maxima = eqm_finder.find_local_optima(reduced_free_energy_array)

        for minimum_index, arg_local_minimum in enumerate(arg_local_minima):
            minima_curves[minimum_index][0].append(swell_ratio)
            minima_curves[minimum_index][1].append(deformation_anisotropy_array[arg_local_minimum])

        for maximum_index, arg_local_maximum in enumerate(arg_local_maxima):
            maxima_curves[maximum_index][0].append(swell_ratio)
            maxima_curves[maximum_index][1].append(deformation_anisotropy_array[arg_local_maximum])

    for curve in minima_curves:
        eqm_ax.plot(curve[0], curve[1], 'C0-')

    for curve in maxima_curves:
        eqm_ax.plot(curve[0], curve[1], 'C0--')

    eqm_ax.set_xlim(np.min(dense_swell_ratio_array) - ZERO, np.max(dense_swell_ratio_array))

    # Finalize figure
    fig.savefig("Figures/constant_twist_free_energy_vary_swell_ratio.pdf", bbox_inches='tight')

def create_swell_ratio_plots():
    deformation_anisotropy_array = np.linspace(ZERO, 3, 5000)
    sparse_swell_ratio_array = np.array([0.5, 1.0, 1.5, 2.0])
    dense_swell_ratio_array = np.linspace(ZERO, 3, 500)
    initial_twist_angle = 5 * np.pi / 180
    anisotropy = 1.5

    plot_constant_twist_swell_dependence(deformation_anisotropy_array,
                                    sparse_swell_ratio_array, dense_swell_ratio_array,
                                    initial_twist_angle, anisotropy,
                                    1.25, 2.5)

def plot_constant_twist_anisotropy_dependence(deformation_anisotropy_array: np.ndarray,
                                    sparse_anisotropy_array: np.ndarray, dense_anisotropy_array: np.ndarray,
                                    initial_twist_angle: float, swell_ratio: float,
                                    free_energy_min: float, free_energy_max: float):
    fig, (free_energy_ax, eqm_ax) = create_fig_ax("Anisotropy parameter", "\\zeta")

    radial_deformation_array = (swell_ratio * deformation_anisotropy_array)**(1/3)

    # Plot reduced free energy vs deformation anisotropy
    for anisotropy in sparse_anisotropy_array:
        reduced_free_energy_array = eqm_finder.calc_reduced_free_energy_density(initial_twist_angle, radial_deformation_array, swell_ratio, anisotropy)
        free_energy_ax.plot(deformation_anisotropy_array, reduced_free_energy_array, label = f"$\\zeta = {anisotropy:.01f}$")

    free_energy_ax.legend(loc='best')
    free_energy_ax.set_xlim(np.min(deformation_anisotropy_array) - ZERO, np.max(deformation_anisotropy_array))
    free_energy_ax.set_ylim(free_energy_min, free_energy_max)

    # Plot equilibria
    minima_curves = [[[], []], [[], []]]
    maxima_curves = [[[], []]]

    for anisotropy in dense_anisotropy_array:
        reduced_free_energy_array = eqm_finder.calc_reduced_free_energy_density(initial_twist_angle, radial_deformation_array, swell_ratio, anisotropy)

        arg_local_minima, arg_local_maxima = eqm_finder.find_local_optima(reduced_free_energy_array)

        for maximum_index, arg_local_maximum in enumerate(arg_local_maxima):
            maxima_curves[maximum_index][0].append(anisotropy)
            maxima_curves[maximum_index][1].append(deformation_anisotropy_array[arg_local_maximum])
        
        if anisotropy < 1 and len(arg_local_minima) > 1:
            for minimum_index, arg_local_minimum in enumerate(arg_local_minima):
                minima_curves[1 - minimum_index][0].append(anisotropy)
                minima_curves[1 - minimum_index][1].append(deformation_anisotropy_array[arg_local_minimum])
        else:
            for minimum_index, arg_local_minimum in enumerate(arg_local_minima):
                minima_curves[minimum_index][0].append(anisotropy)
                minima_curves[minimum_index][1].append(deformation_anisotropy_array[arg_local_minimum])

    for curve in minima_curves:
        eqm_ax.plot(curve[0], curve[1], 'C0-')

    for curve in maxima_curves:
        eqm_ax.plot(curve[0], curve[1], 'C0--')

    eqm_ax.set_xlim(np.min(dense_anisotropy_array) - ZERO, np.max(dense_anisotropy_array))
    _, default_top = eqm_ax.get_ylim()
    eqm_ax.set_ylim(np.min(deformation_anisotropy_array) - ZERO, default_top)

    # Finalize figure
    fig.savefig("Figures/constant_twist_free_energy_vary_anisotropy.pdf", bbox_inches='tight')

def create_anisotropy_plots(): 
    deformation_anisotropy_array = np.linspace(ZERO, 4, 5000)
    sparse_anisotropy_array = np.array([1.0 + ZERO, 2, 3, 4, 5]) #np.array([0.5, 1.0 + ZERO, 2, 3, 4, 5, 6])
    dense_anisotropy_array = np.linspace(ZERO, 5, 1000)
    initial_twist_angle = 5 * np.pi / 180
    swell_ratio = 2

    plot_constant_twist_anisotropy_dependence(deformation_anisotropy_array,
                                    sparse_anisotropy_array, dense_anisotropy_array,
                                    initial_twist_angle, swell_ratio,
                                    2.25, 3.2)

def plot_constant_twist_initial_twist_angle_dependence(deformation_anisotropy_array: np.ndarray,
                                    sparse_initial_twist_angle_array: np.ndarray, dense_initial_twist_angle_array: np.ndarray,
                                    anisotropy: float, swell_ratio: float,
                                    free_energy_min: float, free_energy_max: float):
    fig, (free_energy_ax, eqm_ax) = create_fig_ax("Initial twist-angle", "\\psi_0", "$\\degree$")

    radial_deformation_array = (swell_ratio * deformation_anisotropy_array)**(1/3)

    # Plot reduced free energy vs deformation anisotropy
    for initial_twist_angle in sparse_initial_twist_angle_array:
        reduced_free_energy_array = eqm_finder.calc_reduced_free_energy_density(initial_twist_angle, radial_deformation_array, swell_ratio, anisotropy)
        free_energy_ax.plot(deformation_anisotropy_array, reduced_free_energy_array, label = f"$\\psi_0 = {initial_twist_angle * 180/np.pi:.00f} \\degree$")

    free_energy_ax.legend(loc='best')
    free_energy_ax.set_xlim(np.min(deformation_anisotropy_array) - ZERO, np.max(deformation_anisotropy_array))
    free_energy_ax.set_ylim(free_energy_min, free_energy_max)

    # Plot equilibria
    minima_curves = [[[], []], [[], []], [[], []]]
    maxima_curves = [[[], []], [[], []]]

    for initial_twist_angle in dense_initial_twist_angle_array:
        reduced_free_energy_array = eqm_finder.calc_reduced_free_energy_density(initial_twist_angle, radial_deformation_array, swell_ratio, anisotropy)

        arg_local_minima, arg_local_maxima = eqm_finder.find_local_optima(reduced_free_energy_array)

        for arg_local_maximum in arg_local_maxima:
            maxima_curves[int(initial_twist_angle > np.pi/4)][0].append(initial_twist_angle)
            maxima_curves[int(initial_twist_angle > np.pi/4)][1].append(deformation_anisotropy_array[arg_local_maximum])

        #print(arg_local_maxima)

        if len(arg_local_minima) == 2 and initial_twist_angle > np.pi/4:
            minima_curves[2][0].append(initial_twist_angle)
            minima_curves[2][1].append(deformation_anisotropy_array[arg_local_minima[0]])

            minima_curves[0][0].append(initial_twist_angle)
            minima_curves[0][1].append(deformation_anisotropy_array[arg_local_minima[1]])
        else:
            for minimum_index, arg_local_minimum in enumerate(arg_local_minima):
                minima_curves[minimum_index][0].append(initial_twist_angle)
                minima_curves[minimum_index][1].append(deformation_anisotropy_array[arg_local_minimum])

    for curve in minima_curves:
        eqm_ax.plot(np.array(curve[0]) * 180/np.pi, curve[1], 'C0-')

    #print(maxima_curves)
    for curve in maxima_curves:
        eqm_ax.plot(np.array(curve[0]) * 180/np.pi, curve[1], 'C0--')

    eqm_ax.set_xlim(np.min(dense_initial_twist_angle_array) * 180 / np.pi - ZERO, np.max(dense_initial_twist_angle_array) * 180 / np.pi + ZERO)
    #_, default_top = eqm_ax.get_ylim()
    #eqm_ax.set_ylim(np.min(deformation_anisotropy_array) - ZERO, default_top)

    # Finalize figure
    fig.savefig("Figures/constant_twist_free_energy_vary_initial_twist_angle.pdf", bbox_inches='tight')

def create_initial_twist_angle_plots(): 
    deformation_anisotropy_array = np.linspace(0.5, 1.7, 5000)
    sparse_initial_twist_angle_array = np.array([ZERO, 5, 10]) * np.pi/180 #np.array([ZERO, 5, 10, 90 - ZERO]) * np.pi/180
    dense_initial_twist_angle_array = np.linspace(ZERO, 90 - ZERO, 1000) * np.pi / 180
    anisotropy = 1.5
    swell_ratio = 2

    plot_constant_twist_initial_twist_angle_dependence(deformation_anisotropy_array,
                                    sparse_initial_twist_angle_array, dense_initial_twist_angle_array,
                                    anisotropy, swell_ratio,
                                    2.37, 2.47)

def create_analytical_free_energy_plots():
    plt.rcParams.update({'font.size': 8})
    plt.figure(figsize=(6, 4), dpi = 400)

    zeta = np.linspace(ZERO, 10, 1000)

    triv_min = 3/2 * (zeta / zeta)
    plt.plot(zeta, triv_min, label = "Trivial minimum")

    cusp = zeta**(1/3) + 0.5 * zeta**(-2/3)
    plt.plot(zeta, cusp, label = "Local maximum")

    sec_min = 1/2 * ((1 + 1/zeta) * (2*zeta / (1 + 1/zeta))**(1/3) + zeta * ((1 + 1/zeta)/(2*zeta))**(2/3))
    plt.plot(zeta, sec_min, label = "Secondary minimum")

    plt.xlabel("Anisotropy parameter, $\\zeta$")
    plt.ylabel("Scaled and reduced free energy, $\\tilde{F}/v^{2/3}$")

    plt.xlim(0, 5)
    plt.ylim(1, 3)

    plt.legend(loc='best')

    plt.savefig("Figures/free_energy_vs_zeta.pdf", bbox_inches='tight')

if __name__ == "__main__":
    #create_swell_ratio_plots()

    create_anisotropy_plots()

    #create_initial_twist_angle_plots()

    #create_analytical_free_energy_plots()