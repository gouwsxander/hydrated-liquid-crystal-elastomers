import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

plt.scatter(r_deformations[:-1]**2 * z_deformations[:-1], (r_deformations[1:] - r_deformations[:-1])/(z_deformations[1:] - z_deformations[:-1]))

plt.xlabel("Swelling ratio $\lambda_r^2 \lambda_z$")
plt.ylabel("Deformation anisotropy $d\lambda_r/d\lambda_z$")

plt.show()