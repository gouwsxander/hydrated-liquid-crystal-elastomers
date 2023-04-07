import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import Model, RealData, ODR

import plot_utils
import data_utils

DATA_FILENAMES = ["masic_data", "stam_data"]

SUBSTANCES = {"masic_data": "collagen", "stam_data": "wool"}

DEFORMATION_SUBSCRIPTS = {"axial": "z", "radial": "r"}

def evaluate_linear_model(A, x):
    return A[0] * x + 1

def get_regression_parameters(filename, direction):
    df = data_utils.get_data_frame(filename)

    saturations = data_utils.get_saturations(df, SUBSTANCES[filename])

    deformations, deformation_uncertainties = data_utils.get_deformations(df, direction)

    square_deformations = deformations**2
    square_deformation_uncertainties = 2 * square_deformations * deformation_uncertainties

    initial_saturation = float(saturations[deformations == 1.])

    # Regression
    linear_model = Model(evaluate_linear_model)
    model_data = RealData(saturations[deformations != 1] - initial_saturation, square_deformations[deformations != 1],
                          sy=square_deformation_uncertainties[deformations != 1])
    model_odr = ODR(model_data, linear_model, beta0 = [1])
    odr_output = model_odr.run()

    slope = odr_output.beta[0]
    slope_std_err = odr_output.sd_beta[0]

    return slope, slope_std_err, initial_saturation

def create_square_deformation_figure(direction):
    """
    direction: axial or radial
    """
    plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        saturations = data_utils.get_saturations(df, SUBSTANCES[filename])

        deformations, deformation_uncertainties = data_utils.get_deformations(df, direction)
        square_deformations = deformations**2
        square_deformation_uncertainties = 2 * deformations * deformation_uncertainties

        color = plot_utils.COLORS[filename]
        marker = plot_utils.MARKERS[filename]
        label = plot_utils.LABELS[filename]
        plt.errorbar(saturations, square_deformations, square_deformation_uncertainties, c=color, fmt=marker, label=label)

        slope, std_err, initial_saturation = get_regression_parameters(filename, direction)
        print(f"{filename} {direction}, slope: {slope} +- {std_err}")

        linestyle=plot_utils.LINESTYLES[filename]

        saturation_range = max(saturations) - min(saturations)
        dense_saturations = np.linspace(min(saturations) - 0.05*saturation_range, max(saturations) + 0.05*saturation_range, 1000)
        plt.plot(dense_saturations, evaluate_linear_model([slope], dense_saturations - initial_saturation), c=color, linestyle=linestyle)

    plt.legend(loc='best')
    plt.xlabel("Saturation (grams of water per gram of dry mass), $\\theta$")
    plt.ylabel(f"Squared {direction} deformation, $\\lambda_{DEFORMATION_SUBSCRIPTS[direction]}^2$")

    plot_utils.save_figure(f"Squared {direction} deformation vs saturation")

def create_deformation_anisotropy_figure():
    plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        saturations = data_utils.get_saturations(df, SUBSTANCES[filename])

        deformation_anisotropies, deformation_anisotropy_uncertainties = data_utils.get_deformation_anisotropies(df)

        color = plot_utils.COLORS[filename]
        marker = plot_utils.MARKERS[filename]
        label = plot_utils.LABELS[filename]
        plt.errorbar(saturations, deformation_anisotropies, deformation_anisotropy_uncertainties, c=color, fmt=marker, label=label)

        slope_axial, _, _ = get_regression_parameters(filename, "axial")
        slope_radial, _, initial_saturation = get_regression_parameters(filename, "radial")

        linestyle=plot_utils.LINESTYLES[filename]

        saturation_range = max(saturations) - min(saturations)
        dense_saturations = np.linspace(min(saturations) - 0.05*saturation_range, max(saturations) + 0.05*saturation_range, 1000)
        estimated_radial_deformation = np.sqrt(evaluate_linear_model([slope_radial], dense_saturations - initial_saturation))
        estimated_axial_deformation = np.sqrt(evaluate_linear_model([slope_axial], dense_saturations - initial_saturation))
        estimated_anisotropy = estimated_radial_deformation / estimated_axial_deformation

        plt.plot(dense_saturations, estimated_anisotropy, c=color, linestyle=linestyle)

    plt.legend(loc='best')
    plt.xlabel("Saturation (grams of water per gram of dry mass), $\\theta$")
    plt.ylabel("Deformation anisotropy, $\\alpha = \\lambda_r/\\lambda_z$")

    plot_utils.save_figure(f"Deformation anisotropy vs saturation")

def create_swell_ratio_figure():
    plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        saturations = data_utils.get_saturations(df, SUBSTANCES[filename])

        swell_ratios, swell_ratio_uncertainties = data_utils.get_swell_ratios(df)

        color = plot_utils.COLORS[filename]
        marker = plot_utils.MARKERS[filename]
        label = plot_utils.LABELS[filename]
        plt.errorbar(saturations, swell_ratios, swell_ratio_uncertainties, c=color, fmt=marker, label=label)

        slope_axial, _, _ = get_regression_parameters(filename, "axial")
        slope_radial, _, initial_saturation = get_regression_parameters(filename, "radial")

        saturation_range = max(saturations) - min(saturations)
        dense_saturations = np.linspace(min(saturations) - 0.05*saturation_range, max(saturations) + 0.05*saturation_range, 1000)
        estimated_radial_deformation = np.sqrt(evaluate_linear_model([slope_radial], dense_saturations - initial_saturation))
        estimated_axial_deformation = np.sqrt(evaluate_linear_model([slope_axial], dense_saturations - initial_saturation))
        estimated_swell_ratio = estimated_radial_deformation**2 * estimated_axial_deformation

        linestyle = plot_utils.LINESTYLES[filename]
        plt.plot(dense_saturations, estimated_swell_ratio, c=color, linestyle=linestyle)

    plt.legend(loc='best')
    plt.xlabel("Saturation (grams of water per gram of dry mass), $\\theta$")
    plt.ylabel("Swell ratio, $\\beta = \\lambda_r^2 \\lambda_z$")

    plot_utils.save_figure(f"Swell ratio vs saturation")

if __name__ == "__main__":
    create_square_deformation_figure(direction="axial")
    create_square_deformation_figure(direction="radial")
    create_deformation_anisotropy_figure()
    create_swell_ratio_figure()