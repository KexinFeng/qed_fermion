cd /Users/kx/Desktop/hmc/qed_fermion/qed_fermion/post_processors/6x60_bs2
# export Lx=6
# export Ltau=6
# python3 load_write2file_convert.py

# export Lx=8
# export Ltau=8
# python3 load_write2file_convert.py

# export Lx=10
# export Ltau=10
# python3 load_write2file_convert.py

## Plot
export Lx=6
export Ltau=6
python3 plot_local_hmc_energy_J_self_dqmc.py
python3 plot_local_hmc_energy_J_dqmc_vs_hmc.py
python3 plot_spsm_dqmc_vs_hmc.py

export Lx=8
export Ltau=8
python3 plot_local_hmc_energy_J_self_dqmc.py
python3 plot_local_hmc_energy_J_dqmc_vs_hmc.py
python3 plot_spsm_dqmc_vs_hmc.py

export Lx=10
export Ltau=10
python3 plot_local_hmc_energy_J_self_dqmc.py
python3 plot_local_hmc_energy_J_dqmc_vs_hmc.py
python3 plot_spsm_dqmc_vs_hmc.py
