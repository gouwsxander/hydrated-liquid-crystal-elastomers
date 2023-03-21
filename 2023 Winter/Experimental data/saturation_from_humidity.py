import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import lagrange

df = pd.read_csv('2023 Winter/Experimental data/RH_saturation_data.csv')

relative_humidities = df["RH (%)"]
changes_in_mass = df["Total per dry"]

saturation = changes_in_mass / 67 # chosen qualitatively

polynomial_fit = lagrange(relative_humidities, saturation)

plt.scatter(relative_humidities, saturation)
plt.plot(np.linspace(0,100), polynomial_fit(np.linspace(0,100)))

plt.show()