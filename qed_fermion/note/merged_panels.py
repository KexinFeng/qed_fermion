import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

# Ensure project root is on path for plotting defaults and local imports
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../../')

try:
    from qed_fermion.utils.prep_plots import set_default_plotting
    set_default_plotting()
except Exception:
    pass


def add_panel_label(ax, label, x=0.02, y=0.98):
    # Place label above the y-axis, outside the plot area
    ax.annotate(
        label,
        xy=(0, 1), xycoords='axes fraction',
        xytext=(-49, -12), textcoords='offset points',
        fontsize=18, fontweight='bold',
        va='bottom', ha='left',
        annotation_clip=False
    )

def main():
    # Import plotting helpers
    from qed_fermion.note.perf_cuda_graph.speedup_plot import plot as plot_cudagraph
    from qed_fermion.note.perf_note_cuda_ker.speedup_plot import plot as plot_cudakernel
    from qed_fermion.note.Nrv_perf.latency_Nrv40_plt3_plot import plot as plot_latency

    # Preserve original per-panel aspect ratio from prep_plots defaults
    base_w, base_h = mpl.rcParams.get('figure.figsize', (10, 4.5))
    base_w, base_h = 5, 5
    fig, axes = plt.subplots(1, 3, figsize=(base_w * 3, base_h), constrained_layout=True)

    # Panel (a)
    plot_cudakernel(axes[0])
    add_panel_label(axes[0], '(a)')

    # Panel (b)
    plot_cudagraph(axes[1])
    add_panel_label(axes[1], '(b)')

    # Panel (c)
    plot_latency(axes[2])
    add_panel_label(axes[2], '(c)')

    # Save aggregated figure
    save_dir = os.path.join(script_path, 'figures')
    os.makedirs(save_dir, exist_ok=True)
    out_path_pdf = os.path.join(save_dir, 'speedup_complexity.pdf')
    plt.savefig(out_path_pdf, bbox_inches='tight')
    print(f'Saved merged figure to: {out_path_pdf}')


if __name__ == '__main__':
    main()


