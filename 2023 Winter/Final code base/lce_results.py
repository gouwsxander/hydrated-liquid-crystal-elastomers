import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import Model, RealData, ODR

import plot_utils
import data_utils

DATA_FILENAMES = ["masic_data", "stam_data"]

SUBSTANCES = {"masic_data": "collagen", "stam_data": "hair"}

DEFORMATION_SUBSCRIPTS = {"axial": "z", "radial": "r"}

def evaluate_linear_model(A, x):
    return A[0] * x + 1

def get_regression_parameters(filename, direction):
    df = data_utils.get_data_frame(filename)

    saturations = data_utils.get_saturations(df, SUBSTANCES[filename])

    deformations, deformation_uncertainties = data_utils.get_deformations(df, direction)

    square_deformations = deformations**2
    square_deformation_uncertainties = 2 * square_deformations * deformation_uncertainties

    initial_saturation = float(saturations[deformations == 1.][0])

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
        
        filled_plot = deformations != 1
        hollow_plot = deformations == 1
        plt.errorbar(saturations[filled_plot], square_deformations[filled_plot], square_deformation_uncertainties[filled_plot],
                     c=color, fmt=marker, label=label, mfc=color)
        plt.errorbar(saturations[hollow_plot], square_deformations[hollow_plot], square_deformation_uncertainties[hollow_plot],
                     c=color, fmt=marker, mfc='white')

        slope, std_err, initial_saturation = get_regression_parameters(filename, direction)
        print(f"{filename} {direction}, slope: {slope} +- {std_err}")

        linestyle=plot_utils.LINESTYLES[filename]

        saturation_range = max(saturations) - min(saturations)
        dense_saturations = np.linspace(min(saturations) - 0.05*saturation_range, max(saturations) + 0.05*saturation_range, 1000)
        plt.plot(dense_saturations, evaluate_linear_model([slope], dense_saturations - initial_saturation), c=color, linestyle=linestyle)

    plt.legend(loc='best')
    plt.xlabel("Water saturation (g/g), $\\theta$")
    plt.ylabel(f"{direction.capitalize()} deformation, $\\lambda_{DEFORMATION_SUBSCRIPTS[direction]}^2$")

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

        filled_plot = deformation_anisotropies != 1
        hollow_plot = deformation_anisotropies == 1
        plt.errorbar(saturations[filled_plot], deformation_anisotropies[filled_plot], deformation_anisotropy_uncertainties[filled_plot],
                     c=color, fmt=marker, label=label, mfc=color)
        plt.errorbar(saturations[hollow_plot], deformation_anisotropies[hollow_plot], deformation_anisotropy_uncertainties[hollow_plot],
                     c=color, fmt=marker, mfc='white')

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
    xlabel = plt.xlabel("Water saturation (g/g), $\\theta$")
    ylabel = plt.ylabel("Deformation\nanisotropy, $\\alpha = \\lambda_r/\\lambda_z$")

    #xlabel.set_wrap(True)
    #ylabel.set_wrap(True)

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

        filled_plot = swell_ratios != 1
        hollow_plot = swell_ratios == 1
        plt.errorbar(saturations[filled_plot], swell_ratios[filled_plot], swell_ratio_uncertainties[filled_plot],
                     c=color, fmt=marker, label=label, mfc=color)
        plt.errorbar(saturations[hollow_plot], swell_ratios[hollow_plot], swell_ratio_uncertainties[hollow_plot],
                     c=color, fmt=marker, mfc='white')

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
    plt.xlabel("Water saturation (g/g), $\\theta$")
    plt.ylabel("Swelling ratio, $\\beta = \\lambda_r^2 \\lambda_z$")

    plot_utils.save_figure(f"Swell ratio vs saturation")

def create_deformation_anisotropy_against_swell_figure():
    plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        saturations = data_utils.get_saturations(df, SUBSTANCES[filename])

        deformation_anisotropies, deformation_anisotropy_uncertainties = data_utils.get_deformation_anisotropies(df)
        swell_ratios, swell_ratio_uncertainties = data_utils.get_swell_ratios(df)

        color = plot_utils.COLORS[filename]
        marker = plot_utils.MARKERS[filename]
        label = plot_utils.LABELS[filename]
        
        filled_plot = swell_ratios != 1
        hollow_plot = swell_ratios == 1
        plt.errorbar(swell_ratios[filled_plot], deformation_anisotropies[filled_plot],
                     deformation_anisotropy_uncertainties[filled_plot], xerr=swell_ratio_uncertainties[filled_plot],
                     c=color, fmt=marker, label=label, mfc=color)
        #plt.errorbar(swell_ratios[hollow_plot], deformation_anisotropies[hollow_plot],
        #             deformation_anisotropy_uncertainties[hollow_plot], xerr=swell_ratio_uncertainties[hollow_plot],
        #             c=color, fmt=marker, mfc='white')

        slope_axial, _, _ = get_regression_parameters(filename, "axial")
        slope_radial, _, initial_saturation = get_regression_parameters(filename, "radial")

        saturation_range = max(saturations) - min(saturations)
        dense_saturations = np.linspace(min(saturations) - 0.05*saturation_range, max(saturations) + 0.05*saturation_range, 1000)
        estimated_radial_deformation = np.sqrt(evaluate_linear_model([slope_radial], dense_saturations - initial_saturation))
        estimated_axial_deformation = np.sqrt(evaluate_linear_model([slope_axial], dense_saturations - initial_saturation))
        estimated_swell_ratio = estimated_radial_deformation**2 * estimated_axial_deformation
        estimated_anisotropy = estimated_radial_deformation / estimated_axial_deformation

        linestyle = plot_utils.LINESTYLES[filename]
        plt.plot(estimated_swell_ratio, estimated_anisotropy, c=color, linestyle=linestyle)

    plt.legend(loc='best')
    plt.xlabel("Swelling ratio, $\\beta = \\lambda_r^2 \\lambda_z$")
    plt.ylabel("Deformation anisotropy, $\\alpha = \\lambda_r/\\lambda_z$")

    plot_utils.save_figure(f"deformation anisotropy vs swell ratio")

def create_strain_anisotropy_against_swell_figure():
    plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        saturations = data_utils.get_saturations(df, SUBSTANCES[filename])

        strain_anisotropies, strain_anisotropy_uncertainties = data_utils.get_strain_anisotropies(df)
        swell_ratios, swell_ratio_uncertainties = data_utils.get_swell_ratios(df)

        color = plot_utils.COLORS[filename]
        marker = plot_utils.MARKERS[filename]
        label = plot_utils.LABELS[filename]
        
        filled_plot = swell_ratios != 1
        hollow_plot = swell_ratios == 1
        plt.errorbar(swell_ratios[filled_plot], strain_anisotropies[filled_plot],
                     strain_anisotropy_uncertainties[filled_plot], xerr=swell_ratio_uncertainties[filled_plot],
                     c=color, fmt=marker, label=label, mfc=color)
        # note: No hollow plot because undefined for strain_anisotropy

        slope_axial, _, _ = get_regression_parameters(filename, "axial")
        slope_radial, _, initial_saturation = get_regression_parameters(filename, "radial")

        saturation_range = max(saturations) - min(saturations)
        dense_saturations = np.linspace(min(saturations) - 0.05*saturation_range, max(saturations) + 0.05*saturation_range, 1000)
        estimated_radial_deformation = np.sqrt(evaluate_linear_model([slope_radial], dense_saturations - initial_saturation))
        estimated_axial_deformation = np.sqrt(evaluate_linear_model([slope_axial], dense_saturations - initial_saturation))
        estimated_swell_ratio = estimated_radial_deformation**2 * estimated_axial_deformation
        estimated_strain_anisotropy = (estimated_radial_deformation - 1) / (estimated_axial_deformation - 1)

        linestyle = plot_utils.LINESTYLES[filename]
        plt.plot(estimated_swell_ratio, estimated_strain_anisotropy, c=color, linestyle=linestyle)

    plt.legend(loc='best')
    plt.xlabel("Swelling ratio, $\\beta = \\lambda_r^2 \\lambda_z$")
    plt.ylabel("Strain anisotropy, $\\varepsilon_r/\\varepsilon_z$")

    plot_utils.save_figure(f"strain anisotropy vs swell ratio")

if __name__ == "__main__":
    create_square_deformation_figure(direction="axial")
    create_square_deformation_figure(direction="radial")
    create_deformation_anisotropy_figure()
    create_swell_ratio_figure()
    create_deformation_anisotropy_against_swell_figure()
    create_strain_anisotropy_against_swell_figure()
    print("Finished!")