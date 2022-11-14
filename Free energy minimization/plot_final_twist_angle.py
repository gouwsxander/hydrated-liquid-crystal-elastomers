import numpy as np
import matplotlib.pyplot as plt

import eqm_finder

def against_lambda_r(lambda_r_array: np.ndarray, psi_0_array: np.ndarray, v_swell, zeta):
    for psi_0 in psi_0_array:
        psi_array = eqm_finder.calc_psi(psi_0, lambda_r_array, v_swell, zeta)

        plt.plot(lambda_r_array, psi_array * 180/np.pi, label = f"$\\psi_0 = {psi_0 * 180 / np.pi:.02f}^\\circ$")

    plt.legend(loc = 'best')

    plt.xlabel('Radial deformation, $\\lambda_r$')
    plt.ylabel('Final twist-angle, $\\psi$ ($^\\circ$)')


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

    plt.xlabel('Initial twist-angle, $\\psi_0$ ($^\\circ$)')
    plt.ylabel('Final twist-angle, $\\psi$ ($^\\circ$)')

def main():
    v_swell = 2
    zeta = 1.5

    # lambda_r_array = np.linspace(0.5, 2, 1000)
    # psi_0_array = np.array([0.1, 1, 15, 30, 45]) * np.pi / 180
    # against_lambda_r(lambda_r_array, psi_0_array, v_swell, zeta)
    # plt.show()

    # psi_0_array = np.linspace(0.01, 90, 1000) * np.pi / 180
    # lambda_r_array = np.array([1, 1.1, 1.2, 1.3, 1.4, 1.5])
    # against_psi_0(lambda_r_array, psi_0_array, v_swell, zeta)
    # plt.show()

    psi_0_array = np.linspace(0.01, 10, 1000) * np.pi / 180
    lambda_r_array = np.linspace(1.3, 1.4, 9) #np.array([1, 1.2, 1.4, 1.6])
    against_psi_0(lambda_r_array, psi_0_array, v_swell, zeta, include_small_angle = True)
    plt.show()

if __name__ == "__main__":
    main()