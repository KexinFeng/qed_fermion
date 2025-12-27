import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import importlib
import contextlib

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../../')

try:
    from qed_fermion.utils.prep_plots import set_default_plotting
    set_default_plotting()
except Exception:
    pass


def add_panel_label(ax, label, x=0.02, y=0.98):
    ax.annotate(
        label,
        xy=(0, 1), xycoords='axes fraction',
        xytext=(-70, -12), textcoords='offset points',
        fontsize=22, fontweight='bold',
        va='bottom', ha='left',
        annotation_clip=False
    )


def _inject_ax_context_for_second_figure(ax):
    """Inject axis context but only redirect the second figure created."""
    import matplotlib.pyplot as _plt
    saved = {
        'figure': _plt.figure,
        'subplots': _plt.subplots,
        'gca': _plt.gca,
        'savefig': _plt.savefig,
        'show': _plt.show,
        'sca': _plt.sca,
        'gcf': _plt.gcf,
    }
    
    figure_count = [0]  # Use list to allow modification in nested function
    redirect_to_ax = [False]
    
    def fake_figure(*args, **kwargs):
        figure_count[0] += 1
        if figure_count[0] == 1:
            # First figure - create normally
            return saved['figure'](*args, **kwargs)
        elif figure_count[0] == 2:
            # Second figure - redirect to provided ax
            redirect_to_ax[0] = True
            _plt.sca(ax)
            return ax.figure
        else:
            return saved['figure'](*args, **kwargs)
    
    def fake_subplots(*args, **kwargs):
        figure_count[0] += 1
        if figure_count[0] == 1:
            return saved['subplots'](*args, **kwargs)
        elif figure_count[0] == 2:
            redirect_to_ax[0] = True
            _plt.sca(ax)
            return ax.figure, ax
        else:
            return saved['subplots'](*args, **kwargs)
    
    def fake_gca():
        if redirect_to_ax[0]:
            return ax
        return saved['gca']()
    
    def fake_gcf():
        if redirect_to_ax[0]:
            return ax.figure
        return saved['gcf']()
    
    def fake_savefig(*args, **kwargs):
        # Suppress savefig for both figures in merged context
        return None
    
    def fake_show(*args, **kwargs):
        # Suppress show for both figures
        return None
    
    _plt.figure = fake_figure
    _plt.subplots = fake_subplots
    _plt.gca = fake_gca
    _plt.gcf = fake_gcf
    _plt.savefig = fake_savefig
    _plt.show = fake_show
    return saved


def _restore_context(saved):
    import matplotlib.pyplot as _plt
    for k, v in saved.items():
        setattr(_plt, k, v)


def plot_slope_vs_invL_spsm(ax):
    """Plot slope vs 1/L for spsm from plot_slope_vs_invL.py ax2."""
    import importlib
    # Suppress print output during execution
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            saved = _inject_ax_context_for_second_figure(ax)
            try:
                mod = importlib.import_module('qed_fermion.observables.spin_r_cmp_large7_spsm-TAG.plot_slope_vs_invL')
                if hasattr(mod, 'plot_slope_vs_invL'):
                    mod.plot_slope_vs_invL()
            finally:
                _restore_context(saved)
    return ax


def plot_slope_vs_invL_bond(ax):
    """Plot slope vs 1/L for bond from plot_slope_vs_invL.py ax2."""
    import importlib
    # Suppress print output during execution
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            saved = _inject_ax_context_for_second_figure(ax)
            try:
                mod = importlib.import_module('qed_fermion.observables.spin_r_cmp_large9_bond_corr-TAG.plot_slope_vs_invL')
                if hasattr(mod, 'plot_slope_vs_invL'):
                    mod.plot_slope_vs_invL()
            finally:
                _restore_context(saved)
    return ax


def plot_flux_slope_vs_invL(ax):
    """Plot flux slope vs 1/L from plot_flux_slope_vs_invL.py ax2."""
    import importlib
    # Suppress print output during execution
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            saved = _inject_ax_context_for_second_figure(ax)
            try:
                mod = importlib.import_module('qed_fermion.observables.spin_r_cmp_large7_spsm-TAG.plot_flux_slope_vs_invL')
                if hasattr(mod, 'plot_flux_slope_vs_invL'):
                    mod.plot_flux_slope_vs_invL()
            finally:
                _restore_context(saved)
    return ax


def main():
    from importlib import import_module

    plot_spsm = import_module(
        "qed_fermion.observables.spin_r_cmp_large7_spsm-TAG.plot_fit_spsm_r_plot"
    ).plot
    plot_bondcorr = import_module(
        "qed_fermion.observables.spin_r_cmp_large9_bond_corr-TAG.plot_fit_bond_corr_correction_plot"
    ).plot
    plot_flux = import_module(
        "qed_fermion.observables.spin_r_cmp_large7_spsm-TAG.plot_flux_plot"
    ).plot

    base_w, base_h = mpl.rcParams.get('figure.figsize', (6, 4.5))
    # First row subfigures: original size
    first_row_w, first_row_h = 7.3, 6.7

    # Second row (three subfigures) should each use aspect ratio 8:6,
    # and total width must match the first row => each subfig width = first_row_w
    # So: width = 7.3, height = width * 6/8 = 7.3 * 0.75 = 5.475
    second_row_w = first_row_w
    second_row_h = second_row_w * 6.0 / 8.0

    # Build heights spec for two rows
    height_ratios = [first_row_h, second_row_h]
    total_height = sum(height_ratios)
    # The overall figure width is 3 panels wide = 3 * first_row_w
    fig_width = 3 * first_row_w
    fig_height = total_height

    fig, axes = plt.subplots(
        2, 3,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        gridspec_kw=dict(height_ratios=height_ratios)
    )

    # First row
    plot_spsm(axes[0, 0])
    add_panel_label(axes[0, 0], '(a)')
    axes[0, 0].yaxis.set_minor_locator(plt.NullLocator())

    plot_bondcorr(axes[0, 1])
    add_panel_label(axes[0, 1], '(b)')
    axes[0, 1].yaxis.set_minor_locator(plt.NullLocator())

    plot_flux(axes[0, 2])
    add_panel_label(axes[0, 2], '(c)')
    axes[0, 2].yaxis.set_minor_locator(plt.NullLocator())

    # Second row - slope vs 1/L plots
    plot_slope_vs_invL_spsm(axes[1, 0])
    add_panel_label(axes[1, 0], '(d)')
    axes[1, 0].yaxis.set_minor_locator(plt.NullLocator())

    plot_slope_vs_invL_bond(axes[1, 1])
    add_panel_label(axes[1, 1], '(e)')
    axes[1, 1].yaxis.set_minor_locator(plt.NullLocator())

    plot_flux_slope_vs_invL(axes[1, 2])
    add_panel_label(axes[1, 2], '(f)')
    axes[1, 2].yaxis.set_minor_locator(plt.NullLocator())

    save_dir = os.path.join(script_path, 'figures')
    os.makedirs(save_dir, exist_ok=True)
    out_path_pdf = os.path.join(save_dir, 'fermion_corr_cmp.pdf')
    plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
    print(f'Saved merged figure to: {out_path_pdf}')


if __name__ == '__main__':
    main()


