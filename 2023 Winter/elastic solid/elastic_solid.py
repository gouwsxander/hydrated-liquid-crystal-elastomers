import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def polynomial(swell_anisotropy, swell_ratio, modulus_ratio):
    """modulus_ratio: Axial Young's modulus / Radial Young's modulus"""
    x = swell_anisotropy**(-1/3)

    return modulus_ratio * swell_ratio**(1/3) * x**6 - modulus_ratio * x**4 + x - swell_ratio**(1/3)

def derivative(function, x0, step_size, params):
    return (function(x0 + step_size, *params) - function(x0, *params)) / step_size

def root_finder(function, initial_guess, params, step_size=1e-9, max_error = 1e-9):
    guess = initial_guess
    error = max_error + 1
    while error > max_error:
        guess = guess - function(guess, *params) / derivative(function, guess, step_size, params)
        error = function(guess, *params)

    return guess

def create_figure():
    plt.rcParams.update({'font.size': 8})
    plt.figure(figsize=(6, 4), dpi = 400)

def calc_anisotropies(swell_ratios, modulus_ratio):
    anisotropic_ratios = np.zeros(len(swell_ratios))

    for index, swell_ratio in enumerate(swell_ratios):
        if index == 0:
            guess = 1
        else:
            guess = anisotropic_ratios[index - 1]

        anisotropic_ratios[index] = root_finder(polynomial, guess, (swell_ratio, modulus_ratio))

    return anisotropic_ratios

def plot_curve(swell_ratios, modulus_ratio, marker):
    plt.plot(swell_ratios, calc_anisotropies(swell_ratios, modulus_ratio), marker, label=f"$E_z / E_r = {modulus_ratio:.02f}$")

def plot_asymptotics(swell_ratios, modulus_ratio, marker, label=""):
    slope = (modulus_ratio - 1) / (2 * modulus_ratio + 1)

    plt.plot(swell_ratios, 1 + slope * (swell_ratios - 1), marker, label=label)


def process_data(data_path):
    df = pd.read_csv(data_path)

    deformation_anisotropy_array = df['radial deformation']/df['axial deformation']
    swell_array = df['radial deformation']**2 * df['axial deformation'] #df['swelling ratio']

    return deformation_anisotropy_array, swell_array

def plot_data(deformation_anisotropy_array, swell_array, marker='o', label="Data"):
    plt.scatter(swell_array, deformation_anisotropy_array, marker=marker, label=label)

def anisotropy_slope_to_modulus_ratio(slope):
    return (slope + 1) / (1 - 2 * slope)

if __name__ == "__main__":
    swell_ratios = np.linspace(1, 2.12, 1000)

    modulus_ratio_hair = anisotropy_slope_to_modulus_ratio(0.3888143622512576)
    plot_curve(swell_ratios, modulus_ratio_hair, 'C0-')
    plot_asymptotics(swell_ratios, modulus_ratio_hair, 'C0--')
    deformation_anisotropy_array_hair, swell_array_hair = process_data("2023 Winter/Experimental data/stam_hair_data.csv") # From "The Swelling of Human Hair in Water and Water Vapor" by Stam et al.
    plot_data(deformation_anisotropy_array_hair, swell_array_hair, 'o', "Hair data")

    modulus_ratio_collagen = anisotropy_slope_to_modulus_ratio(0.3007976912418099)
    plot_curve(swell_ratios, modulus_ratio_collagen, 'C1-')
    plot_asymptotics(swell_ratios, modulus_ratio_collagen, 'C1--')
    deformation_anisotropy_array_collagen, swell_array_collagen = process_data("2023 Winter/Experimental data/haverkamp_collagen_data.csv") # From "The Swelling of Human Hair in Water and Water Vapor" by Stam et al.
    plot_data(deformation_anisotropy_array_collagen, swell_array_collagen, '^', "Collagen data")

    plt.xlabel("Swelling ratio, $\\beta$")
    plt.ylabel("Deformation anisotropy, $\\lambda_r / \\lambda_z$")

    plt.ylim(0.975, 1.375)

    plt.legend(loc='best')
    plt.savefig("2023 Winter/Figures/elastic_solid_energy_plot.pdf", bbox_inches='tight')

        
