import plot_utils
import matplotlib.pyplot as plt

# Data
collagen_perp = 1.55
collagen_perp_unc = 0.02

collagen_par = 1.071
collagen_par_unc = 0.004

hair_perp = 1.320
hair_perp_unc = 0.008

hair_par = 1.06
hair_par_unc = 0.01

# Plotting
plot_utils.create_figure()

plt.errorbar([1, 2], [collagen_perp, collagen_par], yerr=[collagen_perp_unc, collagen_par_unc], fmt='o', color='C0', label='Collagen')
plt.errorbar([3, 4], [hair_perp, hair_par], yerr=[hair_perp_unc, hair_par_unc], fmt='o', color='C1', label='Hair')

# Customize plot
plt.xticks([1, 2, 3, 4], ['Perpendicular', 'Parallel', 'Perpendicular', 'Parallel'])
plt.ylabel('Wet-to-dry step-length ratio')
plt.legend()

plot_utils.save_figure("Wet-to-dry step-length ratios")