import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd


def get_data(data_path):
    df = pd.read_csv(data_path)

    deformation_anisotropy_array = df['radial deformation']/df['axial deformation']
    swell_array = df['swelling ratio']

    return deformation_anisotropy_array, swell_array


def perform_regression(deformation_anisotropy_array, swell_array, verbose = True):
    linregress_result = linregress(swell_array, deformation_anisotropy_array)

    if verbose:
        print(f"Intercept: {linregress_result.intercept}")
        print(f"Slope: {linregress_result.slope}")
        print(f"r: {linregress_result.rvalue}")

    return linregress_result
    

def create_and_save_figure(deformation_anisotropy_array, swell_array, linregress_result, show_model = True):
    # Set up figure
    plt.rcParams.update({'font.size': 8})
    plt.figure(figsize=(6, 4), dpi = 400)

    # Plot data
    plt.scatter(swell_array, deformation_anisotropy_array, label = "Data")

    # Plot trend line
    trend_line_x_range = np.linspace(1, 1.35, 1000)
    slope = linregress_result.slope
    intercept = linregress_result.intercept
    plt.plot(trend_line_x_range, intercept + slope * trend_line_x_range, label = "Line of best fit")
    if show_model:
        plt.plot(trend_line_x_range, trend_line_x_range * 0 + 1, "--", label = "Current model")

    # Labels
    plt.xlabel("Swell ratio, $v$")
    plt.ylabel("Observed deformation anisotropy, $\lambda_r / \lambda_z$")
    plt.legend(loc='best')

    # Save figure
    plt.savefig("Figures/experimental_plot.pdf", bbox_inches='tight')


def main(data_path):
    deformation_anisotropy_array, swell_array = get_data(data_path)

    linregress_result = perform_regression(deformation_anisotropy_array, swell_array)

    create_and_save_figure(deformation_anisotropy_array, swell_array, linregress_result)


if __name__ == "__main__":
    main("Experimental data/stam_hair_data.csv") # From "The Swelling of Human Hair in Water and Water Vapor" by Stam et al.