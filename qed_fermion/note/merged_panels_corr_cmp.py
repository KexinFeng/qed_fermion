import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

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


def main():
    from qed_fermion.observables.spin_r_cmp_large7_spsm.plot_fit_spsm_r_plot import plot as plot_spsm
    from qed_fermion.observables.spin_r_noncmpK0_large6_bond_corr_sup.plot_fit_bond_corr_correction_plot import plot as plot_bondcorr
    # from qed_fermion.observables.spin_r_noncmpK0_large4_BBr.plot_flux_plot import plot as plot_flux

    base_w, base_h = mpl.rcParams.get('figure.figsize', (6, 4.5))
    base_w, base_h = 7.3, 6.7
    fig, axes = plt.subplots(1, 3, figsize=(base_w * 3, base_h), constrained_layout=True)

    plot_spsm(axes[0])
    add_panel_label(axes[0], '(a)')
    axes[0].yaxis.set_minor_locator(plt.NullLocator())

    plot_bondcorr(axes[1])
    add_panel_label(axes[1], '(b)')
    axes[1].yaxis.set_minor_locator(plt.NullLocator())

    plot_flux(axes[2])
    add_panel_label(axes[2], '(c)')
    axes[2].yaxis.set_minor_locator(plt.NullLocator())

    save_dir = os.path.join(script_path, 'figures')
    os.makedirs(save_dir, exist_ok=True)
    out_path_pdf = os.path.join(save_dir, 'fermion_corr.pdf')
    plt.savefig(out_path_pdf, bbox_inches='tight')
    print(f'Saved merged figure to: {out_path_pdf}')


if __name__ == '__main__':
    main()


