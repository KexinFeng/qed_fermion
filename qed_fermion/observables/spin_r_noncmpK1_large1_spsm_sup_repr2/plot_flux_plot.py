import importlib
import matplotlib.pyplot as plt


def _inject_ax_context(ax):
    import matplotlib.pyplot as _plt
    saved = {
        'figure': _plt.figure,
        'subplots': _plt.subplots,
        'gca': _plt.gca,
        'savefig': _plt.savefig,
        'show': _plt.show,
        'sca': _plt.sca,
    }

    def fake_figure(*args, **kwargs):
        return ax.figure

    def fake_subplots(*args, **kwargs):
        return ax.figure, ax

    def fake_gca():
        return ax

    def fake_savefig(*args, **kwargs):
        return None

    def fake_show(*args, **kwargs):
        return None

    _plt.figure = fake_figure
    _plt.subplots = fake_subplots
    _plt.gca = fake_gca
    _plt.savefig = fake_savefig
    _plt.show = fake_show
    _plt.sca(ax)
    return saved


def _restore_context(saved):
    import matplotlib.pyplot as _plt
    for k, v in saved.items():
        setattr(_plt, k, v)


def plot(ax: plt.Axes):
    """Render plot_flux.py main plot into provided axis without saving/showing."""
    saved = _inject_ax_context(ax)
    try:
        mod = importlib.import_module('qed_fermion.observables.spin_r_noncmpK0_large4_BBr.plot_flux')
        # This module executes plotting at import time; ensure context is applied
        if hasattr(mod, 'main_ax') or hasattr(mod, 'plot_data'):
            pass
    finally:
        _restore_context(saved)
    return ax


__all__ = ["plot"]


