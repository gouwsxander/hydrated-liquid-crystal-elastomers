import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd


def get_data(data_path):
    df = pd.read_csv(data_path)

    deformation_anisotropy_array = df['radial deformation']/df['axial deformation']
    swell_array = df['radial deformation']**2 * df['axial deformation'] #df['swelling ratio']

    return deformation_anisotropy_array, swell_array

def perform_regression(deformation_anisotropy_array, swell_array, verbose = True):
    linregress_result = linregress(swell_array, deformation_anisotropy_array)

    if verbose:
        print(f"Intercept: {linregress_result.intercept}")
        print(f"Slope: {linregress_result.slope}")
        print(f"r: {linregress_result.rvalue}")

    return linregress_result

def create_figure():
    # Set up figure
    plt.rcParams.update({'font.size': 8})
    plt.figure(figsize=(6, 4), dpi = 400)

    # Labels
    plt.xlabel("Swelling ratio, $\\beta$")
    plt.ylabel("Deformation anisotropy, $\\lambda_r / \\lambda_z$")
    
def plot_data(deformation_anisotropy_array, swell_array, marker='o', label="Data"):
    plt.scatter(swell_array, deformation_anisotropy_array, marker=marker, label=label)

def calc_trend_end(swell_array):
    return 1.2 * (max(swell_array) - 1) + 1

def plot_trend_line(linregress_result, max_swell, marker='--', label="Line of best fit"):
    trend_line_x_range = np.linspace(1, max_swell, 1000)
    slope = linregress_result.slope
    intercept = linregress_result.intercept
    plt.plot(trend_line_x_range, intercept + slope * trend_line_x_range, marker, label=label)

def save_figure():
    plt.legend(loc='best')
    plt.savefig("2023 Winter/Figures/experimental_plot.pdf", bbox_inches='tight')

def process_data(data_path):
    deformation_anisotropy_array, swell_array = get_data(data_path)
    linregress_result = perform_regression(deformation_anisotropy_array, swell_array)

    return deformation_anisotropy_array, swell_array, linregress_result


if __name__ == "__main__":
    create_figure()

    deformation_anisotropy_array_hair, swell_array_hair, linregress_result_hair = process_data("2023 Winter/Experimental data/stam_hair_data.csv") # From "The Swelling of Human Hair in Water and Water Vapor" by Stam et al.
    plot_data(deformation_anisotropy_array_hair, swell_array_hair, 'o', "Hair data")
    max_swell = 1 + 1.2 * (max(swell_array_hair) - 1)
    plot_trend_line(linregress_result_hair, max_swell, '-', "Hair trendline")

    deformation_anisotropy_array_collagen, swell_array_collagen, linregress_result_collagen = process_data("2023 Winter/Experimental data/haverkamp_collagen_data.csv") # From "Collagen Dehydration" by Haverkamp
    plot_data(deformation_anisotropy_array_collagen, swell_array_collagen, '^', "Collagen data")
    max_swell = 1 + 1.05 * (max(swell_array_collagen) - 1)
    plot_trend_line(linregress_result_collagen, max_swell, '--', "Collagen trendline")

    save_figure()