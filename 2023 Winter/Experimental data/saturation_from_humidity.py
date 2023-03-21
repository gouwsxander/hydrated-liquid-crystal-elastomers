import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import lagrange

df = pd.read_csv('2023 Winter/Experimental data/RH_saturation_data.csv')

relative_humidities = df["RH (%)"]
changes_in_mass = df["Total per dry"]

saturation = changes_in_mass / 67 # chosen qualitatively

polynomial_fit = lagrange(list(relative_humidities) + [0,100], list(saturation) + [0,1])

plt.scatter(relative_humidities, 100*saturation)
plt.scatter([0,100],[0,100])
plt.plot(np.linspace(0, 100, 1000), 100 * polynomial_fit(np.linspace(0, 100, 1000)))

plt.xlabel("Relative humidity (%)")
plt.ylabel("Saturation (%)")

plt.show()