import numpy as np
import matplotlib.pyplot as plt

def create_figure():
    # Set up figure
    plt.rcParams.update({'font.size': 8})
    plt.figure(figsize=(6, 4), dpi = 400)


def calc_free_energy(deformation_anisotropy, saturation, perp_wet_dry_ratio, par_wet_dry_ratio):
    return 0.5 * (2 / (1 + (perp_wet_dry_ratio - 1) * saturation) * deformation_anisotropy**(2/3) + 1/(1 + (par_wet_dry_ratio - 1) * saturation) * deformation_anisotropy**(-4/3))

def plot_free_energy(deformation_anisotropy, saturation, perp_wet_dry_ratio, par_wet_dry_ratio):
    plt.plot(deformation_anisotropy, calc_free_energy(deformation_anisotropy, saturation, perp_wet_dry_ratio, par_wet_dry_ratio), label=f"$\\phi = {saturation*100}\\%$")

def make_free_energy_plot():
    create_figure()

    plt.xlabel("Deformation anisotropy, $\\lambda_r / \\lambda_z$")
    plt.ylabel("Scaled and reduced free energy density, $f \\mu^{-1} \\left(v_{\\mathrm{water}}\\phi/v_{\\mathrm{fiber}} + 1 \\right)^{-2/3}$")
    
    deformation_anisotropies = np.linspace(0, 8, 1000)
    perp_wet_dry_ratio = 4
    par_wet_dry_ratio = 2
    plot_free_energy(deformation_anisotropies, 0, perp_wet_dry_ratio, par_wet_dry_ratio)
    plot_free_energy(deformation_anisotropies, 0.25, perp_wet_dry_ratio, par_wet_dry_ratio)
    plot_free_energy(deformation_anisotropies, 0.5, perp_wet_dry_ratio, par_wet_dry_ratio)
    plot_free_energy(deformation_anisotropies, 0.75, perp_wet_dry_ratio, par_wet_dry_ratio)
    plot_free_energy(deformation_anisotropies, 1.0, perp_wet_dry_ratio, par_wet_dry_ratio)

    plt.xlim(0, 4)
    plt.ylim(0.4, 2.8)

    plt.legend(loc='best')
    plt.savefig("2023 Winter/Figures/free_energy_density_plot.pdf", bbox_inches='tight')


def calc_eqm_anisotropy(saturation, perp_wet_dry_ratio, par_wet_dry_ratio):
    return np.sqrt((1 + saturation * (perp_wet_dry_ratio - 1)) / (1 + saturation * (par_wet_dry_ratio - 1)))

def plot_eqm_anisotropy(saturation, perp_wet_dry_ratio, par_wet_dry_ratio):
    plt.plot(saturation, calc_eqm_anisotropy(saturation, perp_wet_dry_ratio, par_wet_dry_ratio),
        label=f"$\\ell^\\perp_{{\\mathrm{{wet}}}} / \\ell^\\perp_{{\\mathrm{{dry}}}} = {perp_wet_dry_ratio}, \ell^\\parallel_{{\\mathrm{{wet}}}} / \\ell^\\parallel_{{\\mathrm{{dry}}}} = {par_wet_dry_ratio} ",
    )

make_free_energy_plot()
