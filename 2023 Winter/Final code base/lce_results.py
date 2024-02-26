import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import Model, RealData, ODR

import plot_utils
import data_utils

DATA_FILENAMES = ["stam_data", "masic_data"]

SUBSTANCES = {"masic_data": "collagen", "stam_data": "hair"}

DEFORMATION_SUBSCRIPTS = {"axial": "z", "radial": "r"}

# Fitting of raw RH vs saturation was done on Desmos (desmos.com/calculator/eolbvs6eny)
HAIR_SATURATION_FIT_PARAMS = {"Xm": 0.0665914, "C": 7.91942, "k": 0.788352}
COLLAGEN_SATURATION_FIT_PARAMS = {"Xm": 0.109003, "C": 20.1506, "k": 0.823467}
SATURATION_FIT_PARAMS = {"hair": HAIR_SATURATION_FIT_PARAMS, "collagen": COLLAGEN_SATURATION_FIT_PARAMS}

def saturation_humidity_model(aw, saturation_fit_params):
    Xm = saturation_fit_params["Xm"]
    C = saturation_fit_params["C"]
    k = saturation_fit_params["k"]

    return Xm*C*k*aw / ((1-k*aw) * (1 + (C-1)*k*aw))

def evaluate_linear_model(A, x):
    return A[0] * x + 1

def get_regression_parameters(filename, direction):
    df = data_utils.get_data_frame(filename)

    humidities = data_utils.get_humidities(df)
    saturations = saturation_humidity_model(humidities/100, SATURATION_FIT_PARAMS[SUBSTANCES[filename]])

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

def create_linear_deformation_figure(direction):
    """
    direction: axial or radial
    """
    # plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        humidities = data_utils.get_humidities(df)
        saturations = saturation_humidity_model(humidities/100, SATURATION_FIT_PARAMS[SUBSTANCES[filename]])
        #saturations = data_utils.get_saturations(df, SUBSTANCES[filename])

        deformations, deformation_uncertainties = data_utils.get_deformations(df, direction)

        color = plot_utils.COLORS[filename]
        marker = plot_utils.MARKERS[filename]
        label = plot_utils.LABELS[filename]
        
        filled_plot = deformations != 1
        hollow_plot = deformations == 1
        plt.errorbar(saturations[filled_plot], deformations[filled_plot], deformation_uncertainties[filled_plot],
                     c=color, fmt=marker, label=label, mfc=color)
        plt.errorbar(saturations[hollow_plot], deformations[hollow_plot], deformation_uncertainties[hollow_plot],
                     c=color, fmt=marker, mfc='white')

        slope, std_err, initial_saturation = get_regression_parameters(filename, direction)
        #print(f"{filename} {direction}, slope: {slope} +- {std_err}")

        linestyle = plot_utils.LINESTYLES[filename]

        saturation_range = max(saturations) - min(saturations)
        dense_saturations = np.linspace(min(saturations) - 0.05*saturation_range, max(saturations) + 0.05*saturation_range, 1000)
        plt.plot(dense_saturations, np.sqrt(evaluate_linear_model([slope], dense_saturations - initial_saturation)), c=color, linestyle=linestyle)

    if direction == "radial":
        plt.legend(loc='best')
    if direction == "axial":
        plt.xlabel("Water saturation (g/g), $\\theta$")
    plt.ylabel(f"{direction.capitalize()}\ndeformation, $\\lambda_{DEFORMATION_SUBSCRIPTS[direction]}$")

    # plot_utils.save_figure(f"{direction.capitalize()} deformation vs saturation")

def create_square_deformation_figure(direction):
    """
    direction: axial or radial
    """
    #plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        humidities = data_utils.get_humidities(df)
        saturations = saturation_humidity_model(humidities/100, SATURATION_FIT_PARAMS[SUBSTANCES[filename]])
        #saturations = data_utils.get_saturations(df, SUBSTANCES[filename])

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
        #print(f"{filename} {direction}, slope: {slope} +- {std_err}")

        linestyle = plot_utils.LINESTYLES[filename]

        saturation_range = max(saturations) - min(saturations)
        dense_saturations = np.linspace(min(saturations) - 0.05*saturation_range, max(saturations) + 0.05*saturation_range, 1000)
        plt.plot(dense_saturations, evaluate_linear_model([slope], dense_saturations - initial_saturation), c=color, linestyle=linestyle)

    if direction == "radial":
        plt.legend(loc='best')
    if direction == "axial":
        plt.xlabel("Water saturation (g/g), $\\theta$")
    plt.ylabel(f"Squared {direction}\ndeformation, $\\lambda_{DEFORMATION_SUBSCRIPTS[direction]}^2$")

    #plot_utils.save_figure(f"Squared {direction} deformation vs saturation")

def create_deformation_anisotropy_figure():
    #plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        humidities = data_utils.get_humidities(df)
        saturations = saturation_humidity_model(humidities/100, SATURATION_FIT_PARAMS[SUBSTANCES[filename]])

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
    #xlabel = plt.xlabel("Water saturation (g/g), $\\theta$")
    ylabel = plt.ylabel("Deformation\nanisotropy, $\\alpha = \\lambda_r/\\lambda_z$")

    #xlabel.set_wrap(True)
    #ylabel.set_wrap(True)

    #plot_utils.save_figure(f"Deformation anisotropy vs saturation")

def create_swell_ratio_figure():
    #plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        humidities = data_utils.get_humidities(df)
        saturations = saturation_humidity_model(humidities/100, SATURATION_FIT_PARAMS[SUBSTANCES[filename]])

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

    #plt.legend(loc='best')
    plt.xlabel("Water saturation (g/g), $\\theta$")
    plt.ylabel("Swelling ratio, $\\beta = \\lambda_r^2 \\lambda_z$")

    #plot_utils.save_figure(f"Swell ratio vs saturation")

def create_deformation_anisotropy_against_swell_figure():
    plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        humidities = data_utils.get_humidities(df)
        saturations = saturation_humidity_model(humidities/100, SATURATION_FIT_PARAMS[SUBSTANCES[filename]])

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
        plt.errorbar(swell_ratios[hollow_plot], deformation_anisotropies[hollow_plot],
                     deformation_anisotropy_uncertainties[hollow_plot], xerr=swell_ratio_uncertainties[hollow_plot],
                     c=color, fmt=marker, mfc='white')

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
    plt.ylabel("Deformation\nanisotropy, $\\alpha = \\lambda_r/\\lambda_z$")

    plot_utils.save_figure(f"deformation anisotropy vs swell ratio")

def create_strain_anisotropy_against_swell_figure():
    plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        humidities = data_utils.get_humidities(df)
        saturations = saturation_humidity_model(humidities/100, SATURATION_FIT_PARAMS[SUBSTANCES[filename]])

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

def create_deformation_raw_data_figure(direction):
    #plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        humidities = data_utils.get_humidities(df)
        deformations, deformation_errors = data_utils.get_deformations(df, direction)

        color = plot_utils.COLORS[filename]
        marker = plot_utils.MARKERS[filename]
        label = plot_utils.LABELS[filename]

        filled_plot = deformations != 1
        hollow_plot = deformations == 1
        plt.errorbar(humidities[filled_plot], deformations[filled_plot],
                     deformation_errors[filled_plot],
                     c=color, fmt=marker, label=label, mfc=color)
        plt.errorbar(humidities[hollow_plot], deformations[hollow_plot],
                     deformation_errors[hollow_plot],
                     c=color, fmt=marker, mfc='white')

    if direction == "radial":
        plt.legend(loc='lower right')
    #plt.xlabel("Relative humidity (%)")

    subscript = "z"
    if direction == "radial":
        subscript = "r" 
    plt.ylabel(f"{direction.capitalize()} deformation, $\\lambda_{subscript}$")

    plt.xlim(-5, 105)

    #plot_utils.save_figure(f"{direction} deformation vs relative humidity")

def create_saturations_raw_data_figure():
    #plot_utils.create_figure()

    markers = {"collagen": "o", "hair": "^"}
    colors = {"collagen": "C0", "hair": "C1"}

    for substance in ["hair", "collagen"]:
        # Plot raw data
        df = data_utils.get_data_frame(f"saturation_data_{substance}")
    
        humidities = data_utils.get_humidities(df)
        saturations = data_utils.get_saturations(df, substance)

        plt.scatter(humidities, saturations, color=colors[substance], marker=markers[substance], label=f"{substance.capitalize()}")

        # Plot fit
        aw = np.linspace(0, 1, 1000)
        predicted_saturation = saturation_humidity_model(aw, SATURATION_FIT_PARAMS[substance])
        plt.plot(100 * aw, predicted_saturation, colors[substance])
    
    plt.xlabel("Relative humidity (%)")
    plt.ylabel("Water saturation (g/g), $\\theta$")

    plt.xlim(-5,105)

    #plot_utils.save_figure(f"saturation vs relative humidity raw data")

def create_raw_data_superfigure_vertical():
    plot_utils.create_figure(3, 6)

    nrows = 3
    ncols = 1

    ax = plt.subplot(nrows, ncols, 1)
    #plt.title("(a)")
    create_deformation_raw_data_figure(direction="radial")

    plt.subplot(nrows, ncols, 2, sharex = ax)
    #plt.title("(b)")
    create_deformation_raw_data_figure(direction="axial")

    plt.subplot(nrows, ncols, 3, sharex = ax)
    #plt.title("(c)")
    create_saturations_raw_data_figure()

    plot_utils.save_figure("Raw data superfigure")

def create_square_deformation_superfigure():
    plot_utils.create_figure(3, 4)

    nrows = 2
    ncols = 1

    ax = plt.subplot(nrows, ncols, 1)
    create_square_deformation_figure(direction="radial")

    plt.subplot(nrows, ncols, 2, sharex = ax)
    create_square_deformation_figure(direction="axial")

    plot_utils.save_figure("Square deformation superfigure")

def create_linear_deformation_superfigure():
    plot_utils.create_figure(3, 4)

    nrows = 2
    ncols = 1

    ax = plt.subplot(nrows, ncols, 1)
    create_linear_deformation_figure(direction="radial")

    plt.subplot(nrows, ncols, 2, sharex = ax)
    create_linear_deformation_figure(direction="axial")

    plot_utils.save_figure("Deformation superfigure")

def create_deformation_quantities_superfigure():
    plot_utils.create_figure(3, 4)

    nrows = 2
    ncols = 1

    ax = plt.subplot(nrows, ncols, 1)
    create_deformation_anisotropy_figure()

    plt.subplot(nrows, ncols, 2, sharex = ax)
    create_swell_ratio_figure()

    plot_utils.save_figure("Deformation quantities superfigure")

def calc_slope_ratio_error(B_perp, delta_B_perp, B_parallel, delta_B_parallel):
    error_term_1 = delta_B_perp**2 / B_parallel**2
    error_term_2 = (B_perp * delta_B_parallel)**2 / (B_parallel)**4

    return np.sqrt(error_term_1 + error_term_2)

def calculate_step_length_ratio(B, delta_B, max_saturation, initial_saturation):
    return 1 + B * max_saturation / (1 - B * initial_saturation), max_saturation * delta_B / (1 - B * initial_saturation)**2

def output_numerical_quantities():
    B_perp_collagen, delta_B_perp_collagen, initial_saturation_collagen = get_regression_parameters("masic_data", "radial")
    B_parallel_collagen, delta_B_parallel_collagen, _ = get_regression_parameters("masic_data", "axial")

    B_perp_hair, delta_B_perp_hair, initial_saturation_hair = get_regression_parameters("stam_data", "radial")
    B_parallel_hair, delta_B_parallel_hair, _ = get_regression_parameters("stam_data", "axial")

    print("Slopes:")
    print(f"B_perp_collagen = {B_perp_collagen} +- {delta_B_perp_collagen}")
    print(f"B_parallel_collagen = {B_parallel_collagen} +- {delta_B_parallel_collagen}")
    print(f"B_perp_hair = {B_perp_hair} +- {delta_B_perp_hair}")
    print(f"B_parallel_hair = {B_parallel_hair} +- {delta_B_parallel_hair}")
    print("")

    print("Slope Ratios:")
    print(f"(B_perp / B_parallel)_collagen = {B_perp_collagen / B_parallel_collagen} +- {calc_slope_ratio_error(B_perp_collagen, delta_B_perp_collagen, B_parallel_collagen, delta_B_parallel_collagen)}")
    print(f"(B_perp / B_parallel)_hair = {B_perp_hair / B_parallel_hair} +- {calc_slope_ratio_error(B_perp_hair, delta_B_perp_hair, B_parallel_hair, delta_B_parallel_hair)}")
    print("")

    print("Max saturations:")
    max_saturation_collagen = saturation_humidity_model(1, SATURATION_FIT_PARAMS["collagen"])
    max_saturation_hair = saturation_humidity_model(1, SATURATION_FIT_PARAMS["hair"])
    print(f"max_saturation_collagen = {max_saturation_collagen}")
    print(f"max_saturation_hair = {max_saturation_hair}")

    print("Step Lenghts:")
    print(f"(l_wet/l_dry)_perp_collagen = {calculate_step_length_ratio(B_perp_collagen, delta_B_perp_collagen, max_saturation_collagen, initial_saturation_collagen)}")
    print(f"(l_wet/l_dry)_parallel_collagen = {calculate_step_length_ratio(B_parallel_collagen, delta_B_parallel_collagen, max_saturation_collagen, initial_saturation_collagen)}")
    print(f"(l_wet/l_dry)_perp_hair = {calculate_step_length_ratio(B_perp_hair, delta_B_perp_hair, max_saturation_hair, initial_saturation_hair)}")
    print(f"(l_wet/l_dry)_parallel_hair = {calculate_step_length_ratio(B_parallel_hair, delta_B_parallel_hair, max_saturation_hair, initial_saturation_hair)}")


if __name__ == "__main__":
    # create_square_deformation_figure(direction="axial")
    # create_square_deformation_figure(direction="radial")

    # create_deformation_anisotropy_figure()
    # create_swell_ratio_figure()

    # create_deformation_anisotropy_against_swell_figure()
    # create_strain_anisotropy_against_swell_figure()

    #create_deformation_raw_data_figure(direction="axial")
    #create_deformation_raw_data_figure(direction="radial")
    #create_saturations_raw_data_figure()

    # Figure 1
    create_raw_data_superfigure_vertical()

    # Figure 2
    create_square_deformation_superfigure()

    # Figure 3
    create_deformation_quantities_superfigure()

    # Figure 4
    create_deformation_anisotropy_against_swell_figure()

    # Figure 5
    create_strain_anisotropy_against_swell_figure()

    # Numerical quantities
    output_numerical_quantities()

    # Referree-requested Figure
    create_linear_deformation_superfigure()

    print("Finished!")