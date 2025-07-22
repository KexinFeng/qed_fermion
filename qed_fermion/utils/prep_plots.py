import matplotlib as mpl
import numpy as np

def set_default_plotting():
    """Set default plotting settings for physics scientific publication (Matlab style)."""
    width = 6      # inches
    height = 4.5   # inches
    fsz = 16       # Font size
    fna = 'Helvetica'  # Font name
    line_width = 2.5       # Line width
    msz = 10        # Marker size
    interp = 'tex' # Text interpreter

    mpl.rcParams.update({
        "font.family": fna,
        "font.size": fsz,
        "axes.labelsize": fsz + 1,
        "axes.titlesize": fsz,
        "xtick.labelsize": fsz,
        "ytick.labelsize": fsz,
        "legend.fontsize": fsz - 2,
        "figure.titlesize": fsz + 2,
        "lines.linewidth": line_width,
        "lines.markersize": msz,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "figure.figsize": (width, height),
        "legend.frameon": False,
        "text.usetex": interp == 'latex',
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "",
        "figure.autolayout": True,
    }) 

def selective_log_label_func(ax, numticks=6):
    def selective_log_label(y, pos):
        # Only label at most numticks ticks, spaced logarithmically
        ticks = ax.get_yticks()
        # Only label the first, last, and up to 4 evenly spaced in between
        if len(ticks) <= numticks:
            show = ticks
        else:
            idx = np.linspace(0, len(ticks)-1, numticks, dtype=int)
            show = set(np.array(ticks)[idx])
        if y in show and y != 0:
            exponent = int(np.log10(y))
            return f"$10^{{{exponent}}}$"
        else:
            return ""
    return selective_log_label