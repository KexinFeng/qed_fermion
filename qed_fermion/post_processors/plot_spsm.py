import re
import time
import matplotlib.pyplot as plt

plt.ion()

import numpy as np

from matplotlib import rcParams
rcParams['figure.raise_window'] = False

import os
script_path = os.path.dirname(os.path.abspath(__file__))

import torch
import sys
sys.path.insert(0, script_path + '/../../')

from qed_fermion.utils.stat import error_mean, t_based_error, std_root_n, init_convex_seq_estimator

def time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper


@time_execution
def plot_spsm(Lsize=(6, 6, 10)):
    Js = [1.0, 1.5, 2.0, 2.5, 3.0]
    input_folder = "/Users/kx/Desktop/hmc/fignote/ftdqmc/hmc_check_point_L6/"
    Lx, Ly, Ltau = Lsize
    vs = Lx**2
    for J in Js:
        input_folder = f"/Users/kx/Desktop/forked/dqmc_u1sl_mag/run2/run_meas_J_{J:.2g}_L_{Lx}/"
        name = f"spsm.bin"
        dqmc_filename = os.path.join(input_folder, name)
        data = np.genfromtxt(dqmc_filename)
        data = data.reshape(-1, vs, 4)
        # data has shape [num_sample, vs, 4], where the last dim has entries: kx, ky, val, error. 
        # [num_sample]
        r_afm = 1 - data[:, 1, 2] / data[:, 0, 2]
        rtol = data[:, :, 3] / data[:, :, 2]
        r_afm_err = abs(rtol[:, 0] - rtol[:, 1]) * (1 - r_afm)




if __name__ == '__main__':
    Lx = 6
    Ltau = Lx*40
    plot_spsm(Lsize=(Lx, Lx, Ltau))



