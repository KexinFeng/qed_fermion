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
        # no-op in aggregated plot
        return None

    def fake_show(*args, **kwargs):
        # no-op in aggregated plot
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
    """Render plot_fit_spsm_r.plot_spin_r() into provided axis without saving/showing."""
    saved = _inject_ax_context(ax)
    try:
        mod = importlib.import_module('qed_fermion.observables.spin_r_cmp_large7_spsm-TAG.plot_fit_spsm_r')
        # Execute plotting logic
        if hasattr(mod, 'plot_spin_r'):
            mod.plot_spin_r()
        else:
            raise AttributeError('plot_spin_r not found in plot_fit_spsm_r')
    finally:
        _restore_context(saved)
    return ax


__all__ = ["plot"]


