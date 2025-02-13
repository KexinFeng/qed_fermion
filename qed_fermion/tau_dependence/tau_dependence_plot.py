import pickle
import os
import json
import numpy as np

L = [10, 15, 20]
L = 20

Ltau, Lx, Ly, num_tau = (L,)*4  

# Load numerical data
data_folder = "/Users/kx/Desktop/hmc/qed_fermion/qed_fermion/tau_dependence/data_hmc"
filename = f"corr_N_{Ltau}_Nx_{Lx}_Ny_{Ly}_tau-max_{num_tau}.pkl"
filepath = os.path.join(data_folder, filename)

# Load the data
with open(filepath, "rb") as f:
    loaded_data = pickle.load(f)

# The data is a tuple (mean, std), assuming it was saved as (mean, std)
G_mean, G_std = loaded_data['mean'], loaded_data['std']


# Load analytical data
data_folder = "/Users/kx/Desktop/hmc/correlation_kspace/code/tau_dependence/data_tau_dependence"
name = "N_{}_Nx_{}_Ny_{}_tau-max_{}_num_{}".format(Ltau, Lx, Ly, num_tau, num_tau)
pkl_filepath = os.path.join(data_folder, f"{name}.pkl")

# Read JSON data from file
with open(pkl_filepath, 'r') as f:
    loaded_data_anal = json.load(f)
G_mean_anal = np.array(loaded_data_anal['corr'])[:, -1][:-1]  # The last element is the same as the first one due to the tau periodic boundary, and is thus removed.
taus = np.array(loaded_data_anal['taus'])[:-1]


dbstop = 1

