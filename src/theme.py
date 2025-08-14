import seaborn as sns
import matplotlib.pyplot as plt

FIGSIZE_SMALL = (10, 6)
FIGSIZE_LARGE = (18, 14)
PLOT_PALETTE = 'viridis'
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 8
DPI = 100

def set_project_theme():
    sns.set_theme(style="whitegrid", palette=PLOT_PALETTE)
    plt.rc('axes', titlesize=TITLE_FONTSIZE)
    plt.rc('axes', labelsize=LABEL_FONTSIZE)
    plt.rc('xtick', labelsize=TICK_FONTSIZE)
    plt.rc('ytick', labelsize=TICK_FONTSIZE)
    plt.rc('figure', dpi=DPI)