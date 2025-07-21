import matplotlib as mpl

def set_default_plotting():
    """Set default plotting settings for physics scientific publication (Matlab style)."""
    width = 8      # inches
    height = 6   # inches
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
        "legend.fontsize": fsz,
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
        "grid.linestyle": "-",
        "figure.autolayout": True,
    }) 