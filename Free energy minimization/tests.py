import numpy as np
import matplotlib.pyplot as plt

import eqm_finder

v_swell = 1.3
zeta = 1.3
psi_0 = 0.78

lambda_r_min = 1
lambda_r_max = 1.5
num_points = 100
lambda_r_array = np.linspace(lambda_r_min, lambda_r_max, num_points)

free_energy_density_per_mu_array = np.zeros(num_points)

for index, lambda_r in enumerate(lambda_r_array):
    free_energy_density_per_mu_array[index] = eqm_finder.calc_free_energy_density_per_mu(psi_0, lambda_r, v_swell, zeta)

plt.plot(lambda_r_array, free_energy_density_per_mu_array)

plt.xlabel("Radial deformation, $\\lambda_r$")
plt.ylabel("Dimensionless free energy density, $f/\\mu$")

plt.title(f"{v_swell=}, {zeta=}, {psi_0=}")

plt.show()