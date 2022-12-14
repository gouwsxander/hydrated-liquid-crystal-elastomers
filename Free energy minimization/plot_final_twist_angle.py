import numpy as np
import matplotlib.pyplot as plt

import eqm_finder

def against_deformation_anisotropy(lambda_r_array: np.ndarray, psi_0_array: np.ndarray, v_swell, zeta):
    lambda_z_array = v_swell / lambda_r_array**2

    for psi_0 in psi_0_array:
        psi_array = eqm_finder.calc_psi(psi_0, lambda_r_array, v_swell, zeta)

        plt.plot(lambda_r_array / lambda_z_array, psi_array * 180/np.pi, label = f"$\\psi_0 = {psi_0 * 180 / np.pi:.0f}\\degree$")

    plt.legend(loc = 'best')

    plt.xlabel('Deformation anisotropy, $\\lambda_r / \\lambda_z$')
    plt.ylabel('Final twist-angle, $\\psi$ ($\\degree$)')


def against_psi_0(lambda_r_array: np.ndarray, psi_0_array: np.ndarray, v_swell, zeta, include_small_angle = False):
    for i, lambda_r in enumerate(lambda_r_array):
        psi_array = eqm_finder.calc_psi(psi_0_array, lambda_r, v_swell, zeta)

        plt.plot(psi_0_array * 180/np.pi, psi_array * 180/np.pi, f'C{i}', label = f"$\\lambda_r = {lambda_r}$")

        if include_small_angle:
            lambda_z = v_swell / lambda_r**2
            psi_small_array = psi_0_array * (zeta - 1) * lambda_r * lambda_z / (zeta * lambda_z**2 - lambda_r**2)

            if zeta * lambda_z**2 > lambda_r**2:
                plt.plot(psi_0_array * 180/np.pi, psi_small_array * 180/np.pi, f'C{i}--')
            else:
                plt.plot(psi_0_array * 180/np.pi, 90 + psi_small_array * 180/np.pi, f'C{i}--')

    plt.legend(loc = 'best')

    plt.xlabel('Initial twist-angle, $\\psi_0$ ($\\degree$)')
    plt.ylabel('Final twist-angle, $\\psi$ ($\\degree$)')

def against_psi_0(deformation_anisotropy_array: np.ndarray, psi_0_array: np.ndarray, v_swell, zeta, include_small_angle = False):
    for i, deformation_anisotropy in enumerate(deformation_anisotropy_array):
        lambda_r = (deformation_anisotropy * v_swell)**(1/3)
        psi_array = eqm_finder.calc_psi(psi_0_array, lambda_r, v_swell, zeta)

        plt.plot(psi_0_array * 180/np.pi, psi_array * 180/np.pi, f'C{i}', label = f"$\\lambda_r/\\lambda_z = {deformation_anisotropy}$")

        if include_small_angle:
            lambda_z = v_swell / lambda_r**2
            psi_small_array = psi_0_array * (zeta - 1) * lambda_r * lambda_z / (zeta * lambda_z**2 - lambda_r**2)

            if zeta * lambda_z**2 > lambda_r**2:
                plt.plot(psi_0_array * 180/np.pi, psi_small_array * 180/np.pi, f'C{i}--')
            else:
                plt.plot(psi_0_array * 180/np.pi, 90 + psi_small_array * 180/np.pi, f'C{i}--')

    plt.legend(loc = 'best')

    plt.xlabel('Initial twist-angle, $\\psi_0$ ($\\degree$)')
    plt.ylabel('Final twist-angle, $\\psi$ ($\\degree$)')

def main():
    plt.rcParams.update({'font.size': 8})

    v_swell = 2
    zeta = 1.3

    plt.figure(figsize=(6, 4), dpi = 400)
    lambda_r_array = np.linspace(0.5, 2, 1000)
    psi_0_array = np.array([1e-9, 15, 30, 45]) * np.pi / 180
    against_deformation_anisotropy(lambda_r_array, psi_0_array, v_swell, zeta)
    plt.xlim(0.1, 2.6)
    plt.savefig("Figures/psi_vs_lambda_r.pdf", bbox_inches='tight')

    plt.figure(figsize=(6, 4), dpi = 400)
    psi_0_array = np.linspace(1e-9, 90, 1000) * np.pi / 180
    lambda_r_array = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    against_psi_0(lambda_r_array, psi_0_array, v_swell, zeta, include_small_angle=True)
    plt.xlim(0,90)
    plt.ylim(0,90)
    plt.savefig("Figures/psi_vs_psi_0.pdf", bbox_inches='tight')

    # psi_0_array = np.linspace(0.01, 10, 1000) * np.pi / 180
    # lambda_r_array = np.linspace(1.3, 1.4, 9) #np.array([1, 1.2, 1.4, 1.6])
    # against_psi_0(lambda_r_array, psi_0_array, v_swell, zeta, include_small_angle = True)
    # plt.show()

if __name__ == "__main__":
    main()