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

plt.scatter(r_deformations**2 * z_deformations, r_deformations/z_deformations)

plt.xlabel("Swelling ratio $\lambda_r^2 \lambda_z$")
plt.ylabel("Deformation anisotropy $\lambda_r/\lambda_z$")

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

sat_from_RH = lagrange(saturation_data_relative_humidities, saturations)

plt.scatter(sat_from_RH(relative_humidities) * 100, r_deformations**2 * z_deformations)
plt.xlabel("Saturation (%)")
plt.ylabel("Swelling ratio $\\beta = \\lambda_r^2 \\lambda_z$")
plt.show()

plt.scatter(sat_from_RH(relative_humidities) * 100, r_deformations / z_deformations)
plt.xlabel("Saturation (%)")
plt.ylabel("Deformation anisotropy $\lambda_r/\lambda_z$")
plt.show()

plt.scatter(sat_from_RH(relative_humidities[:-1]) * 100, r_strains[:-1] / z_strains[:-1])
plt.xlabel("Saturation (%)")
plt.ylabel("Strain anisotropy $\\varepsilon_r/\\varepsilon_z$")
plt.show()