import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Define the values of L to iterate over
L_values = [10, 15, 20]
L_values = [10, 14, 20]
L_values = [20]

# # Set up an interactive plot
# plt.ion()
# fig, ax = plt.subplots()

# Get distinct colors for each L
colors = plt.get_cmap("tab10").colors

# Iterate over different L values
for i, L in enumerate(L_values):
    Ltau, Lx, Ly, num_tau = (L,) * 4

    # Load numerical data
    data_folder_num = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/tau_dependence/data_hmc"
    filename = f"corr_N_{Ltau}_Nx_{Lx}_Ny_{Ly}_tau-max_{num_tau}.pkl"
    filepath = os.path.join(data_folder_num, filename)

    with open(filepath, "rb") as f:
        loaded_data = pickle.load(f)

    G_mean, G_std = loaded_data['mean'].numpy(), loaded_data['std'].numpy()
    G_mean = np.append(G_mean, G_mean[0])  # Append the first element at the end
    G_std = np.append(G_std, G_std[0])  # Append the first element at the end


    # Load analytical data
    data_folder_anal = "/Users/kx/Desktop/hmc/correlation_kspace/code/tau_dependence/data_tau_dependence"
    name = f"N_{Ltau}_Nx_{Lx}_Ny_{Ly}_tau-max_{num_tau}_num_{num_tau}"
    pkl_filepath = os.path.join(data_folder_anal, f"{name}.pkl")

    with open(pkl_filepath, 'r') as f:
        loaded_data_anal = json.load(f)

    G_mean_anal = np.array(loaded_data_anal['corr'])[:, -1]
    taus = np.array(loaded_data_anal['taus'])  # The last element is the same as the first one due to the tau periodic boundary, and is thus removed.

    # Scale G_mean_anal to match G_mean based on their first elements
    idx_ref = int(Ltau/2)
    scale_factor = G_mean[idx_ref] / G_mean_anal[idx_ref]
    G_mean_anal *= scale_factor

    # -------- Plotting --------
    fig, ax = plt.subplots()

    # # Plot analytical data
    # ax.plot(taus, G_mean_anal, linestyle='-', color=colors[i], label=f'L={L}')

    # Plot numerical data with error bars
    ax.errorbar(taus, G_mean, yerr=G_std, fmt='o', color=colors[i], label=f'L={L}')

    # Customize plot for each L
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$G(\tau)$')
    ax.set_title(f'Correlation Function Comparison for L={L}')
    ax.legend()

    # -------- Save plot --------
    # Save the figure as a .png file with the name based on L value
    folder = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/tau_dependence/figure_tau_dependence/"
    os.makedirs(folder, exist_ok=True)
    plot_filename = f'cmp_correlation_L_{L}.png'
    fig.savefig(folder + plot_filename)

    # Show plot for current L
    plt.show()

# # Customize plot
# ax.set_xlabel(r'$\tau$')
# ax.set_ylabel(r'$G(\tau)$')
# ax.set_title('Correlation Function Comparison')
# ax.legend()
plt.show()


# ---------- loglog plot -----------
# Set up an interactive plot
plt.ion()
fig, ax = plt.subplots()

# Iterate over different L values
for i, L in enumerate(L_values):
    print(f'-------{i}--------')
    Ltau, Lx, Ly, num_tau = (L,) * 4

    # Load numerical data
    data_folder_num = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/tau_dependence/data_hmc"
    filename = f"corr_N_{Ltau}_Nx_{Lx}_Ny_{Ly}_tau-max_{num_tau}.pkl"
    filepath = os.path.join(data_folder_num, filename)

    with open(filepath, "rb") as f:
        loaded_data = pickle.load(f)

    G_mean, G_std = loaded_data['mean'].numpy(), loaded_data['std'].numpy()
    G_mean = np.append(G_mean, G_mean[0])  # Append the first element at the end
    G_std = np.append(G_std, G_std[0])  # Append the first element at the end


    # Load analytical data
    data_folder_anal = "/Users/kx/Desktop/hmc/correlation_kspace/code/tau_dependence/data_tau_dependence"
    name = f"N_{Ltau}_Nx_{Lx}_Ny_{Ly}_tau-max_{num_tau}_num_{num_tau}"
    pkl_filepath = os.path.join(data_folder_anal, f"{name}.pkl")

    with open(pkl_filepath, 'r') as f:
        loaded_data_anal = json.load(f)

    G_mean_anal = np.array(loaded_data_anal['corr'])[:, -1]
    taus = np.array(loaded_data_anal['taus'])  # The last element is the same as the first one due to the tau periodic boundary, and is thus removed.

    # Scale G_mean_anal to match G_mean based on their first elements
    idx_ref = int(Ltau/2)
    scale_factor = G_mean[idx_ref] / G_mean_anal[idx_ref]
    G_mean_anal *= scale_factor

    # -------- Plotting --------
    fig, ax = plt.subplots()

    mask = (taus > 0) & (G_mean > 0) & (G_mean_anal > 0)  # Filter out zeros or negatives

    # # Plot analytical data (log-log)
    # ax.plot(np.log10(taus[mask]), np.log10(G_mean_anal[mask]), linestyle='-', color='b', label=f'Analytical L={L}')

    # Plot numerical data with error bars (log-log)
    ax.errorbar(np.log10(taus[mask]), np.log10(G_mean[mask]), yerr=G_std[mask]/G_mean[mask]/np.log(10), fmt='o', color='r', label=f'Numerical L={L}', capsize=3)

    # Customize plot for each L
    ax.set_xlabel(r'$\tau$ (log scale)')
    ax.set_ylabel(r'$G(\tau)$ (log scale)')
    ax.set_title(f'Log-Log Correlation Function Comparison for L={L}')
    ax.legend()
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    # -------- Save plot --------
    # Save the figure as a .png file with the name based on L value
    folder = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/tau_dependence/figure_tau_dependence/"
    os.makedirs(folder, exist_ok=True)
    plot_filename = f'cmp_log_plot_correlation_L_{L}.png'
    fig.savefig(folder + plot_filename)

    # Show plot for current L
    plt.show()

    dbstop = 1


dbstop = 1
