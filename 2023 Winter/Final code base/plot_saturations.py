import numpy as np
import matplotlib.pyplot as plt

import plot_utils
import data_utils

SUBSTANCES = ["collagen", "wool"]

MAX_SATURATIONS = {"collagen": [0.5, 0.6, 1.0, 1.1], "wool": [0.3, 0.4, 0.8, 0.9]}

MARKERS = ["o", "s", "^", "d"]

B_perps = {"collagen": 0.61, "wool": 0.96}
B_pars = {"collagen": 0.109, "wool": 0.17}

theta_0 = {"collagen": 0.501, "wool": 0}

for substance in SUBSTANCES:
    print(substance)
    plot_utils.create_figure()

    df = data_utils.get_data_frame("saturation_data")
    
    humidities = data_utils.get_humidities(df)
    saturations = data_utils.get_saturations(df, substance)

    for i, max_saturation in enumerate(MAX_SATURATIONS[substance]):
        plot_humidities = [humidity for humidity in humidities] + [100]
        plot_saturations = [100 * saturation / max_saturation for saturation in saturations] + [100]

        plt.plot(plot_humidities, plot_saturations, marker=MARKERS[i], label=f"$\\theta_\\mathrm{{max}} = {max_saturation}$ g/g")
        
        perp_ratio = 1 + B_perps[substance] * max_saturation / (1 - B_perps[substance] * theta_0[substance])
        par_ratio = 1 + B_pars[substance] * max_saturation / (1 - B_pars[substance] * theta_0[substance])
        print(f"{max_saturation=} --> {perp_ratio=}, {par_ratio=}")
    
    plt.xlabel("Relative humidity (%)")
    plt.ylabel("Percent saturation, $\\theta/\\theta_\mathrm{max}$ (%)")
    plt.legend(loc='best')

    plt.xlim(0,100)
    plt.ylim(0,100)

    plot_utils.save_figure(f"{substance} saturation curve")

    print("")