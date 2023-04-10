import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import Model, RealData, ODR

import plot_utils
import data_utils

DATA_FILENAMES = ["masic_data", "haverkamp_data", "stam_data"] #["masic_macro_data", "stam_data"] #

SUBSTANCES = {"masic_data": "collagen", "haverkamp_data": "collagen", "stam_data": "hair"}

POISSONS_RATIOS = {"collagen": 0, "hair": 0.38} # collagen ~ 2.1 hair might be ~0.38 or 1.0
POISSONS_RATIOS_UNCERTAINTY = {"collagen": 0, "hair": 0.01} #0.7, 0.01

def evaluate_linear_model(A, x):
    return A[0] * (x - 1) + 1

def evaluate_linear_origin(A, x):
    return A[0] * x

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

def strain_difference_vs_total():
    plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        axial_deformations, axial_deformation_uncertainties = data_utils.get_deformations(df, "axial")
        radial_deformations, radial_deformation_uncertainties = data_utils.get_deformations(df, "radial")

        axial_strains = axial_deformations - 1
        radial_strains = radial_deformations - 1

        total_strains = 2 * radial_strains + axial_strains
        delta_strains = radial_strains - axial_strains

        total_strain_uncertainty = np.sqrt(4 * radial_deformation_uncertainties**2 + axial_deformation_uncertainties**2)
        delta_strain_uncertainty = np.sqrt(radial_deformation_uncertainties**2 + axial_deformation_uncertainties**2)

        color = plot_utils.COLORS[filename]
        marker = plot_utils.MARKERS[filename]
        label = plot_utils.LABELS[filename]
        
        plt.errorbar(total_strains, delta_strains,
                     delta_strain_uncertainty, xerr=total_strain_uncertainty,
                     c=color, fmt=marker, label=label,
                     )
        
        # Regression
        linear_model = Model(evaluate_linear_origin)
        if filename == "masic_macro_data":
            model_data = RealData(total_strains[:3], delta_strains[:3], sx=total_strain_uncertainty[:3], sy=delta_strain_uncertainty[:3])
        else:
            model_data = RealData(total_strains[1:], delta_strains[1:], sx=total_strain_uncertainty[1:], sy=delta_strain_uncertainty[1:])
        model_odr = ODR(model_data, linear_model, beta0 = [1.0])
        odr_output = model_odr.run()

        slope = odr_output.beta[0]
        slope_std_err = odr_output.sd_beta[0]

        mat_aniso, mat_aniso_err = material_anisotropy_from_slope(slope, slope_std_err, POISSONS_RATIOS[SUBSTANCES[filename]], POISSONS_RATIOS_UNCERTAINTY[SUBSTANCES[filename]])

        print(f"{filename}, slope: {slope} +- {slope_std_err}, material_anisotropy: {mat_aniso} +- {mat_aniso_err}")

        total_strain_range = max(total_strains) - min(total_strains)
        dense_swell_ratios = np.linspace(min(total_strains) - 0.05*total_strain_range, max(total_strains) + 0.05*total_strain_range)

        linestyle=plot_utils.LINESTYLES[filename]
        plt.plot(dense_swell_ratios, evaluate_linear_origin([slope], dense_swell_ratios), c=color, linestyle=linestyle)
    
    plt.legend(loc='best')
    plt.xlabel("Total strain, $2\\varepsilon_r + \\varepsilon_z$")
    plt.ylabel("Strain difference, $\\varepsilon_r - \\varepsilon_z$")

    plot_utils.save_figure(f"Total strain figure")

def strain_radial_vs_axial():
    plot_utils.create_figure()

    for filename in DATA_FILENAMES:
        df = data_utils.get_data_frame(filename)

        axial_deformations, axial_deformation_uncertainties = data_utils.get_deformations(df, "axial")
        radial_deformations, radial_deformation_uncertainties = data_utils.get_deformations(df, "radial")

        axial_strains = axial_deformations - 1
        radial_strains = radial_deformations - 1

        color = plot_utils.COLORS[filename]
        marker = plot_utils.MARKERS[filename]
        label = plot_utils.LABELS[filename]
        
        plt.errorbar(axial_strains, radial_strains,
                     radial_deformation_uncertainties, xerr=axial_deformation_uncertainties,
                     c=color, fmt=marker, label=label,
                     )
        
        # Regression
        linear_model = Model(evaluate_linear_origin)
        if filename == "stam_data":
            model_data = RealData(axial_strains[1:-1], radial_strains[1:-1], sx=axial_deformation_uncertainties[1:-1], sy=radial_deformation_uncertainties[1:-1])
        elif filename == "masic_macro_data":
            model_data = RealData(axial_strains[:1], radial_strains[:1], sx=axial_deformation_uncertainties[:1], sy=radial_deformation_uncertainties[:1])
        else:
            model_data = RealData(axial_strains[1:], radial_strains[1:], sx=axial_deformation_uncertainties[1:], sy=radial_deformation_uncertainties[1:])
        model_odr = ODR(model_data, linear_model, beta0 = [1.0])
        odr_output = model_odr.run()

        slope = odr_output.beta[0]
        slope_std_err = odr_output.sd_beta[0]

        model_1_slope, model_1_slope_err = slope_from_strain_anisotropy(slope, slope_std_err)
        material_anisotropy, mat_aniso_err = material_anisotropy_from_slope(model_1_slope, model_1_slope_err, POISSONS_RATIOS[SUBSTANCES[filename]], POISSONS_RATIOS_UNCERTAINTY[SUBSTANCES[filename]])

        print(f"{filename}, slope: {slope} +- {slope_std_err}, material anisotropy: {material_anisotropy} +- {mat_aniso_err}")

        axial_strain_range = max(axial_strains) - min(axial_strains)
        dense_axial_strains = np.linspace(min(axial_strains) - 0.05*axial_strain_range, max(axial_strains) + 0.05*axial_strain_range)

        linestyle=plot_utils.LINESTYLES[filename]
        plt.plot(dense_axial_strains, evaluate_linear_origin([slope], dense_axial_strains), c=color, linestyle=linestyle)
    
    plt.legend(loc='best')
    plt.ylabel("Radial strain, $\\varepsilon_r$")
    plt.xlabel("Axial strain, $\\varepsilon_z$")

    plot_utils.save_figure(f"Radial vs axial strain figure")

print("Deformation anisotropy:")
create_anisotropy_vs_swell()
print("\nStrain anisotropy:")
create_strain_anisotropy_vs_swell()

print("\nstrain radial vs axial:")
strain_radial_vs_axial()
print("\nvolumetric vs strain difference:")
strain_difference_vs_total()


# strain_anisotropy = 0.96/0.17
# dstrain_anisotropy = strain_anisotropy * np.sqrt((0.03/0.96)**2 + (0.02/0.17)**2)

# slope, stderr = slope_from_strain_anisotropy(strain_anisotropy, dstrain_anisotropy)
# mat_aniso, dmat_aniso = material_anisotropy_from_slope(slope, stderr, POISSONS_RATIOS["hair"], POISSONS_RATIOS_UNCERTAINTY["hair"])

# print(mat_aniso, dmat_aniso)

