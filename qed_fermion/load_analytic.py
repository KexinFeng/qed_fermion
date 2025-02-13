import json
import pickle
import os
import numpy as np

# Define file path
data_folder = "/Users/kx/Desktop/hmc/correlation_kspace/code/tau_dependence/data_tau_dependence"
N, Nx, Ny, tau_max, num = (30,)*5
name = "N_{}_Nx_{}_Ny_{}_tau-max_{}_num_{}".format(N, Nx, Ny, tau_max, num)
pkl_filepath = os.path.join(data_folder, f"{name}.pkl")

# Read JSON data from file
with open(pkl_filepath, 'r') as f:
    data = json.load(f)

# # Convert to NumPy arrays (optional, depending on how MATLAB stores them)
# data["corr"] = np.array(data["corr"])
# data["taus"] = np.array(data["taus"])
# data["lambdas"] = np.array(data["lambdas"])
#
# # Save as a proper Python pickle file
# with open(pkl_filepath, 'wb') as f:
#     pickle.dump(data, f)
#
# print("Data loaded and saved as Python pickle.")
