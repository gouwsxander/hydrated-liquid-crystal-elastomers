import matplotlib.pyplot as plt

#FIGURE_ROOT = "2023 Winter/Figures" #Figure Root Windows
FIGURE_ROOT = "./2023 Winter/Figures" #Figure Root Mac

#MARKERS = ["o", "s", "^", "d"]
#LINESTYLES = ["-", "--", "-."]

COLORS = {"masic_data": "C0", "masic_macro_data": "C0", "haverkamp_data": "C0", "stam_data": "C1"}
MARKERS = {"masic_data": "o", "masic_macro_data": "o", "haverkamp_data": "s", "stam_data": "^"}
LABELS = {"masic_data": "Collagen", "masic_macro_data": "Collagen",
          "haverkamp_data": "Collagen", "stam_data": "Hair"}
LINESTYLES = {"masic_data": "-", "masic_macro_data": "-", "haverkamp_data": "-.", "stam_data": "--"}

def create_figure(width=3, height=2):
    plt.rcParams.update({'font.size': 7})
    plt.figure(figsize=(width, height), dpi = 600)

def save_figure(figure_name):
    plt.savefig(f"{FIGURE_ROOT}/{figure_name}.pdf", bbox_inches='tight')