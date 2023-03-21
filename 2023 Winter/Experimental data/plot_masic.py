import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import lagrange

# Masic data
df_r_strain = pd.read_csv('2023 Winter/Experimental data/masic_r_strain.csv')
df_z_strain = pd.read_csv('2023 Winter/Experimental data/masic_z_strain.csv')

relative_humidities = np.array((df_r_strain["RH(%)"] + df_z_strain["RH(%)"])/2 / 100)
r_strains = np.array(df_r_strain["Strain(%)"] / 100)
z_strains = np.array(df_z_strain["Strain(%)"] / 100)

r_deformations = 1 + r_strains
z_deformations = 1 + z_strains

r_deformations_hydration = r_deformations / r_deformations[0]
z_deformations_hydration = z_deformations / z_deformations[0]

plt.scatter(r_deformations**2 * z_deformations, r_deformations/z_deformations)

plt.xlabel("Dehydration swelling ratio $\lambda_r^2 \lambda_z$")
plt.ylabel("Dehydration deformation anisotropy $\lambda_r/\lambda_z$")

plt.show()

plt.scatter(r_deformations_hydration**2 * z_deformations_hydration, r_deformations_hydration/z_deformations_hydration)

plt.xlabel("Hydration swelling ratio $\lambda_r^2 \lambda_z$")
plt.ylabel("Hydration deformation anisotropy $\lambda_r/\lambda_z$")

plt.show()

plt.scatter(r_deformations[:-1]**2 * z_deformations[:-1], r_strains[:-1] / z_strains[:-1])
plt.xlabel("Dehydration swelling ratio $\lambda_r^2 \lambda_z$")
plt.ylabel("Dehydration strain anisotropy $\\varepsilon_r/\\varepsilon_z$")
plt.show()

plt.scatter(r_deformations_hydration**2 * z_deformations_hydration, (r_deformations_hydration-1)/(z_deformations_hydration-1))
plt.xlabel("Hydration swelling ratio $\lambda_r^2 \lambda_z$")
plt.ylabel("Hydration strain anisotropy $\\varepsilon_r/\\varepsilon_z$")
plt.show()


# plt.scatter(r_deformations**2 * z_deformations, (r_deformations-1)/(z_deformations-1))

# plt.xlabel("Swelling ratio $\lambda_r^2 \lambda_z$")
# plt.ylabel("Strain anisotropy $\\varepsilon_r/\\varepsilon_z$")

# plt.show()

# plt.scatter(r_deformations[:-1]**2 * z_deformations[:-1], (r_deformations[1:] - r_deformations[:-1])/(z_deformations[1:] - z_deformations[:-1]))

# plt.xlabel("Swelling ratio $\lambda_r^2 \lambda_z$")
# plt.ylabel("Deformation anisotropy $d\lambda_r/d\lambda_z$")

# plt.show()

# Converting RH to saturation
df = pd.read_csv('2023 Winter/Experimental data/RH_saturation_data.csv')

saturation_data_relative_humidities = df["RH (%)"]/100
changes_in_mass = df["Total per dry"]

saturations = changes_in_mass / 67 # chosen qualitatively

sat_from_RH = lagrange(list(saturation_data_relative_humidities) + [0,1], list(saturations) + [0,1])

plt.scatter(sat_from_RH(relative_humidities) * 100, r_deformations**2 * z_deformations)
plt.xlabel("Saturation (%)")
plt.ylabel("Dehydration swelling ratio $\\beta = \\lambda_r^2 \\lambda_z$")
plt.show()

plt.scatter(sat_from_RH(relative_humidities) * 100, r_deformations_hydration**2 * z_deformations_hydration)
plt.xlabel("Saturation (%)")
plt.ylabel("Hydration swelling ratio $\\beta = \\lambda_r^2 \\lambda_z$")
plt.xlim(0,100)
plt.show()

plt.scatter(sat_from_RH(relative_humidities) * 100, r_deformations / z_deformations)
plt.xlabel("Saturation (%)")
plt.ylabel("Dehydration deformation anisotropy $\lambda_r/\lambda_z$")
plt.show()

plt.scatter(sat_from_RH(relative_humidities) * 100, r_deformations_hydration/z_deformations_hydration)
plt.xlabel("Saturation (%)")
plt.ylabel("Hydration deformation anisotropy $\lambda_r/\lambda_z$")
plt.xlim(0,100)
plt.show()

plt.scatter(sat_from_RH(relative_humidities[:-1]) * 100, r_strains[:-1] / z_strains[:-1])
plt.xlabel("Saturation (%)")
plt.ylabel("Dehydration strain anisotropy $\\varepsilon_r/\\varepsilon_z$")
plt.show()

plt.scatter(sat_from_RH(relative_humidities) * 100, (r_deformations_hydration-1)/(z_deformations_hydration-1))
plt.xlabel("Saturation (%)")
plt.ylabel("Hydration strain anisotropy $\\varepsilon_r/\\varepsilon_z$")
plt.xlim(0,100)
plt.show()