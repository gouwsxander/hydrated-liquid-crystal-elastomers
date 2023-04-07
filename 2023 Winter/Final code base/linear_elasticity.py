import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import Model, RealData, ODR

import plot_utils
import data_utils

DATA_FILENAMES = ["masic_data", "haverkamp_data", "stam_data"]

SUBSTANCES = {"masic_data": "collagen", "haverkamp_data": "collagen", "stam_data": "hair"}

POISSONS_RATIOS = {"collagen": 0, "hair": 0.4} # collagen ~ 2 hair might be ~0.4 or 1.0
POISSONS_RATIOS_UNCERTAINTY = {"collagen": 0, "hair": 0.01}

def evaluate_linear_model(A, x):
    return A[0] * (x - 1) + 1

def material_anisotropy_from_slope(slope, stderr, poisson_ratio, poisson_ratio_uncertainty):
    ratio = ((1 - 4 * poisson_ratio) * slope + 1 - poisson_ratio) / (1 - 2 * slope)
    dratio = np.sqrt((3*(poisson_ratio - 1) / (2*slope - 1)**2)**2 * stderr**2 + ((slope+1)/(2*slope - 1))**2 * poisson_ratio_uncertainty**2)

    return ratio, dratio

def slope_from_strain_anisotropy(strain_anisotropy, dstrain_anisotropy):
    A = (strain_anisotropy - 1) / (2 * strain_anisotropy + 1)
    dA = 3/(2*strain_anisotropy + 1)**2 * dstrain_anisotropy

    return A, dA

def strain_anisotropy_from_slope(slope, stderr):
    ratio = (1+slope) / (1 - 2*slope)
    dratio = 3/(2*slope - 1)**2 * stderr

    return ratio, dratio

def create_anisotropy_vs_swell():
    plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        # Get data
        df = data_utils.get_data_frame(filename)

        swell_ratios, swell_ratio_uncertainties = data_utils.get_swell_ratios(df)
        deformation_anisotropies, deformation_anisotropy_uncertainties = data_utils.get_deformation_anisotropies(df)

        # Get plot properties
        color = plot_utils.COLORS[filename]
        marker = plot_utils.MARKERS[filename]
        label = plot_utils.LABELS[filename]

        # Plot data
        plt.errorbar(swell_ratios, deformation_anisotropies, deformation_anisotropy_uncertainties, xerr=swell_ratio_uncertainties, c=color, fmt=marker, label=label)

        # Regression
        linear_model = Model(evaluate_linear_model)
        model_data = RealData(swell_ratios[swell_ratios != 1], deformation_anisotropies[swell_ratios != 1], sx=swell_ratio_uncertainties[swell_ratios != 1], sy=deformation_anisotropy_uncertainties[swell_ratios != 1])
        model_odr = ODR(model_data, linear_model, beta0 = [0.2])
        odr_output = model_odr.run()

        slope = odr_output.beta[0]
        slope_std_err = odr_output.sd_beta[0]

        material_anisotropy, material_anisotropy_unc = material_anisotropy_from_slope(slope, slope_std_err, POISSONS_RATIOS[SUBSTANCES[filename]], POISSONS_RATIOS_UNCERTAINTY[SUBSTANCES[filename]])
        strain_anisotropy, strain_anisotropy_unc = strain_anisotropy_from_slope(slope, slope_std_err)

        print(f"{filename}, slope: {slope} +- {slope_std_err}, material anisotropy: {material_anisotropy} +- {material_anisotropy_unc}, strain ansio: {strain_anisotropy} +- {strain_anisotropy_unc}")

        swell_ratio_range = max(swell_ratios) - min(swell_ratios)
        dense_swell_ratios = np.linspace(min(swell_ratios) - 0.05*swell_ratio_range, max(swell_ratios) + 0.05*swell_ratio_range)

        linestyle=plot_utils.LINESTYLES[filename]
        plt.plot(dense_swell_ratios, evaluate_linear_model([slope], dense_swell_ratios), c=color, linestyle=linestyle)

    plt.legend(loc='best')
    plt.xlabel("Swelling ratio, $\\beta = \\lambda_r^2 \\lambda_z$")
    plt.ylabel("Deformation anisotropy, $\\alpha = \\lambda_r/\\lambda_z$")

    plot_utils.save_figure(f"Deformation anisotropy vs swell ratio")

def create_strain_anisotropy_vs_swell():
    plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        swell_ratios, swell_ratio_uncertainties = data_utils.get_swell_ratios(df)
        strain_anisotropies, strain_anisotropy_uncertainties = data_utils.get_strain_anisotropies(df)

        color = plot_utils.COLORS[filename]
        marker = plot_utils.MARKERS[filename]
        label = plot_utils.LABELS[filename]
        # TODO: Not sure I trust this
        plt.errorbar(swell_ratios[swell_ratios != 1], strain_anisotropies[swell_ratios != 1],
                     strain_anisotropy_uncertainties[swell_ratios != 1], xerr=swell_ratio_uncertainties[swell_ratios != 1],
                     c=color, fmt=marker, label=label,
                     )
    
        weights = 1/strain_anisotropy_uncertainties[swell_ratios != 1]**2
        avg_strain_anisotropy = np.average(strain_anisotropies[swell_ratios != 1], weights=weights) # TODO: weighted
        std_strain_anisotropy = 1/np.sqrt(np.sum(weights))

        slope, stderr = slope_from_strain_anisotropy(avg_strain_anisotropy, std_strain_anisotropy)
        material_anisotropy, material_anisotropy_unc = material_anisotropy_from_slope(slope, stderr, POISSONS_RATIOS[SUBSTANCES[filename]], POISSONS_RATIOS_UNCERTAINTY[SUBSTANCES[filename]])

        print(f"{filename}, strain anisotropy: {avg_strain_anisotropy} +- {std_strain_anisotropy}, slope: {slope} +- {stderr}, material: {material_anisotropy} +- {material_anisotropy_unc}")

    plt.legend(loc='best')
    plt.xlabel("Swelling ratio, $\\beta = \\lambda_r^2 \\lambda_z$")
    plt.ylabel("Strain anisotropy, $\\varepsilon_r/\\varepsilon_z$")

    plot_utils.save_figure(f"Strain anisotropy vs swell ratio")

create_anisotropy_vs_swell()
create_strain_anisotropy_vs_swell()