import matplotlib.pyplot as plt

FIGURE_ROOT = "2023 Winter/Figures"

#MARKERS = ["o", "s", "^", "d"]
#LINESTYLES = ["-", "--", "-."]

COLORS = {"masic_data": "C0", "haverkamp_data": "C0", "stam_data": "C1"}
MARKERS = {"masic_data": "o", "haverkamp_data": "s", "stam_data": "^"}
LABELS = {"masic_data": "Collagen (Masic $\\it{et}$ $\\it{al.}$)", "haverkamp_data": "Collagen (Haverkamp $\\it{et}$ $\\it{al.}$)", "stam_data": "Hair (Stam $\\it{et}$ $\\it{al.}$)"}
LINESTYLES = {"masic_data": "-", "haverkamp_data": "-", "stam_data": "--"}

def create_figure():
    plt.rcParams.update({'font.size': 8})
    plt.figure(figsize=(6, 4), dpi = 400)

def save_figure(figure_name):
    plt.savefig(f"{FIGURE_ROOT}/{figure_name}.pdf", bbox_inches='tight')